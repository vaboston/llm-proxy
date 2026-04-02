"""Core proxy logic: backend selection, request translation, streaming.

Handles routing requests to the appropriate backend (Ollama, LM Studio,
llama.cpp), translating between API formats, filtering thinking tokens,
and deduplicating concurrent identical requests.
"""

import asyncio
import hashlib
import json
import time

import httpx

from app.config import BackendConfig, get_config, should_filter_thinking
from app.thinking import ThinkingFilter, strip_thinking
from app.stats import stats
from app.logs import log_manager

# ==========================================================================
# HTTP client
# ==========================================================================

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(get_config().timeout, connect=5.0))
    return _client


async def close_client():
    global _client
    if _client:
        await _client.aclose()
        _client = None


# ==========================================================================
# Backend health & selection
# ==========================================================================

async def check_backend_health(backend: BackendConfig) -> bool:
    client = get_client()
    try:
        url = (f"{backend.base_url}/api/tags" if backend.type == "ollama"
               else f"{backend.base_url}/v1/models")
        resp = await client.get(url, timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


async def get_available_backend() -> BackendConfig | None:
    """Return first enabled + reachable backend by priority order."""
    for b in get_config().backends:
        if not b.enabled:
            continue
        if await check_backend_health(b):
            return b
        log_manager.log("error", f"Backend '{b.name}' unreachable, trying next")
    return None


async def fetch_backend_models(backend: BackendConfig) -> list[str]:
    client = get_client()
    try:
        if backend.type == "ollama":
            resp = await client.get(f"{backend.base_url}/api/tags", timeout=5.0)
            return [m["name"] for m in resp.json().get("models", [])]
        else:
            resp = await client.get(f"{backend.base_url}/v1/models", timeout=5.0)
            return [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        log_manager.log("error", f"Failed to fetch models from '{backend.name}': {e}")
        return []


# ==========================================================================
# Thinking suppression
# ==========================================================================

_THINKING_MODELS = ("qwen3", "qwq", "deepseek-r1")


def _is_thinking_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(t in lower for t in _THINKING_MODELS)


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Append /no_think to the last user message for Qwen3-family models."""
    messages = [m.copy() for m in messages]
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if "/no_think" not in content:
                m["content"] = content + " /no_think"
            break
    return messages


# ==========================================================================
# Message format sanitization
# ==========================================================================

def _sanitize_messages_for_ollama(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Ollama-compatible format.

    - tool_calls[].function.arguments: string JSON -> parsed object
    - tool role content: ensure it's a string
    """
    out = []
    for m in messages:
        m = m.copy()
        if "tool_calls" in m and m["tool_calls"]:
            fixed = []
            for tc in m["tool_calls"]:
                tc = tc.copy()
                fn = tc.get("function", {})
                if isinstance(fn, dict):
                    fn = fn.copy()
                    args = fn.get("arguments", "")
                    if isinstance(args, str) and args:
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        if m.get("role") == "tool" and not isinstance(m.get("content", ""), str):
            m["content"] = json.dumps(m["content"])
        out.append(m)
    return out


# ==========================================================================
# Request format translation
# ==========================================================================

def _forward_tools(body: dict, out: dict):
    """Copy tools/tool_choice from request body if present."""
    if "tools" in body:
        out["tools"] = body["tools"]
    if "tool_choice" in body:
        out["tool_choice"] = body["tool_choice"]


def ollama_to_openai_request(body: dict, model: str, no_think: bool = False) -> dict:
    """Ollama /api/chat body -> OpenAI /v1/chat/completions body."""
    messages = body.get("messages", [])
    if no_think and _is_thinking_model(model):
        messages = _inject_no_think(messages)
    out = {"model": model, "messages": messages, "stream": body.get("stream", True)}
    opts = body.get("options", {})
    if "temperature" in opts:
        out["temperature"] = opts["temperature"]
    if "top_p" in opts:
        out["top_p"] = opts["top_p"]
    if "num_predict" in opts:
        out["max_tokens"] = opts["num_predict"]
    if "stop" in opts:
        out["stop"] = opts["stop"]
    _forward_tools(body, out)
    return out


def openai_to_ollama_request(body: dict, model: str, no_think: bool = False) -> dict:
    """OpenAI /v1/chat/completions body -> Ollama /api/chat body."""
    messages = _sanitize_messages_for_ollama(body.get("messages", []))
    if no_think and _is_thinking_model(model):
        messages = _inject_no_think(messages)
    out = {"model": model, "messages": messages, "stream": body.get("stream", True)}
    options = {}
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "stop" in body:
        options["stop"] = body["stop"]
    if options:
        out["options"] = options
    _forward_tools(body, out)
    return out


def ollama_generate_to_openai_request(body: dict, model: str, no_think: bool = False) -> dict:
    """Ollama /api/generate body -> OpenAI chat request."""
    messages = [{"role": "user", "content": body.get("prompt", "")}]
    if body.get("system"):
        messages.insert(0, {"role": "system", "content": body["system"]})
    if no_think and _is_thinking_model(model):
        messages = _inject_no_think(messages)
    out = {"model": model, "messages": messages, "stream": body.get("stream", True)}
    opts = body.get("options", {})
    if "temperature" in opts:
        out["temperature"] = opts["temperature"]
    if "num_predict" in opts:
        out["max_tokens"] = opts["num_predict"]
    return out


# ==========================================================================
# Backend stream iterators (with aggregated logging)
# ==========================================================================

async def _iter_ollama_stream(response: httpx.Response):
    """Yield parsed JSON chunks from an Ollama NDJSON stream."""
    chunk_count = 0
    content_buf = ""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as e:
            log_manager.log("error", f"[ollama] JSON decode error: {e}", {"line": line[:500]})
            continue

        chunk_count += 1
        msg = parsed.get("message", {})
        content = msg.get("content", "")
        done = parsed.get("done", False)

        if content:
            content_buf += content

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            for name in tc_names:
                log_manager.log("info", f"tool_call: {name}()", {"raw": parsed})
            stats.record_tool_calls(tc_names)

        if done:
            if content_buf:
                preview = content_buf[:200].replace("\n", "\\n")
                suffix = f"... ({len(content_buf)} chars)" if len(content_buf) > 200 else ""
                log_manager.log("response", f"{preview}{suffix}", {"full_content": content_buf})
            dur = parsed.get("total_duration")
            tok = parsed.get("eval_count")
            info = f"stream done - {chunk_count} chunks"
            if tok:
                info += f", {tok} tokens"
            if dur:
                info += f", {dur / 1e9:.1f}s"
            log_manager.log("info", info, {"raw": parsed})

        yield parsed


async def _iter_openai_stream(response: httpx.Response):
    """Yield parsed JSON chunks from an OpenAI SSE stream.

    Handles LM Studio/llama.cpp error events. Content tokens are aggregated
    into a single summary log at stream end. Yields None as sentinel for [DONE].
    """
    chunk_count = 0
    content_buf = ""
    pending_event = None

    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            pending_event = None
            continue

        if line.startswith("event:"):
            pending_event = line.split(":", 1)[1].strip()
            continue

        if not line.startswith("data: "):
            continue

        data = line[6:]
        if data == "[DONE]":
            if content_buf:
                preview = content_buf[:200].replace("\n", "\\n")
                suffix = f"... ({len(content_buf)} chars)" if len(content_buf) > 200 else ""
                log_manager.log("response", f"{preview}{suffix}", {"full_content": content_buf})
            log_manager.log("info", f"stream done - {chunk_count} chunks")
            yield None
            return

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            log_manager.log("error", f"[sse] JSON decode error: {e}", {"line": data[:500]})
            pending_event = None
            continue

        # Backend error event (LM Studio / llama.cpp)
        if pending_event == "error" or "error" in parsed:
            err_msg = parsed.get("error", {})
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", str(err_msg))
            log_manager.log("error", f"backend error: {err_msg}", {"raw": parsed})
            yield {"choices": [{"index": 0, "delta": {"content": f"[Backend error: {err_msg}]"}, "finish_reason": "stop"}]}
            yield None
            return

        chunk_count += 1
        pending_event = None
        choices = parsed.get("choices", [])

        if not choices:
            usage = parsed.get("usage", {})
            if usage:
                pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                stats.record_tokens(pt, ct)
                log_manager.log("info", f"tokens: {pt} prompt + {ct} completion", {"raw": parsed})
            yield parsed
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "") or ""
        if content:
            content_buf += content

        for tc in delta.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name:
                log_manager.log("info", f"tool_call: {name}()", {"raw": parsed})
                stats.record_tool_calls([name])

        if choices[0].get("finish_reason"):
            log_manager.log("info", f"finish: {choices[0]['finish_reason']}", {"raw": parsed})

        yield parsed

    # Stream ended without [DONE]
    if content_buf:
        preview = content_buf[:200].replace("\n", "\\n")
        suffix = f"... ({len(content_buf)} chars)" if len(content_buf) > 200 else ""
        log_manager.log("response", f"{preview}{suffix}", {"full_content": content_buf})
    log_manager.log("info", f"stream ended without [DONE] - {chunk_count} chunks")


# ==========================================================================
# Request deduplication
# ==========================================================================

_inflight: dict[str, asyncio.Task] = {}
_RESULT_CACHE_TTL = 120  # seconds to keep completed results


def abort_all_inflight() -> int:
    """Cancel all in-flight backend tasks. Returns count cancelled."""
    count = 0
    for key, task in list(_inflight.items()):
        if not task.done():
            task.cancel()
            count += 1
        _inflight.pop(key, None)
    stats.record_aborted(count)
    return count


def _request_key(body: dict) -> str:
    canonical = json.dumps(
        {"messages": body.get("messages", []), "tools": body.get("tools")},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.md5(canonical.encode()).hexdigest()


async def _coalesced_stream_and_buffer(backend, body) -> tuple[str, list]:
    """Deduplicated _stream_and_buffer.

    If an identical request is already in-flight, wait for it instead of
    starting a new one. Completed results are cached for 120s.
    """
    key = _request_key(body)

    if key in _inflight:
        task = _inflight[key]
        if task.done():
            log_manager.log("info", "returning cached result for duplicate request")
            stats.record_cached()
            return task.result()
        log_manager.log("info", "request coalesced - waiting for in-flight duplicate")
        stats.record_coalesced()
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            log_manager.log("info", "client disconnected, backend task continues in background")
            raise

    task = asyncio.create_task(_stream_and_buffer(backend, body))
    _inflight[key] = task
    task.add_done_callback(
        lambda t: asyncio.get_event_loop().call_later(_RESULT_CACHE_TTL, lambda: _inflight.pop(key, None))
    )

    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        log_manager.log("info", "client disconnected, backend task continues in background")
        raise


# ==========================================================================
# Streaming proxy: Ollama output format
# ==========================================================================

def _prepare_request(backend: BackendConfig, body: dict, inject_no_think: bool, is_reasoning: bool):
    """Build target URL and request body for a backend."""
    if backend.type == "ollama":
        url = f"{backend.base_url}/api/chat"
        req = {**body, "model": backend.model,
               "messages": _sanitize_messages_for_ollama(body.get("messages", []))}
        if inject_no_think and is_reasoning:
            req["messages"] = _inject_no_think(req["messages"])
    else:
        url = f"{backend.base_url}/v1/chat/completions"
        req = ollama_to_openai_request(body, backend.model, no_think=inject_no_think)
    return url, req


async def stream_chat_ollama_out(backend: BackendConfig, body: dict):
    """Proxy a chat request -> Ollama NDJSON output."""
    cfg = get_config()
    inject_no_think = should_filter_thinking(backend)
    is_reasoning = _is_thinking_model(backend.model)
    thinking_filter = ThinkingFilter() if is_reasoning else None
    client = get_client()

    target_url, request_body = _prepare_request(backend, body, inject_no_think, is_reasoning)
    log_manager.log("request", f"-> {backend.name} ({backend.type}) {target_url}",
                    {"backend": backend.name, "model": backend.model})

    async with client.stream("POST", target_url, json=request_body,
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        ct = response.headers.get("content-type", "?")
        log_manager.log("info", f"[backend] HTTP {response.status_code} content-type={ct}")

        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            yield json.dumps({"model": cfg.proxy_model_name,
                              "message": {"role": "assistant", "content": f"[Proxy error: backend returned {response.status_code}]"},
                              "done": True}) + "\n"
            return

        total_content = ""
        if backend.type == "ollama":
            async for chunk in _iter_ollama_stream(response):
                content = chunk.get("message", {}).get("content", "")
                done = chunk.get("done", False)
                if thinking_filter:
                    content = thinking_filter.feed(content)
                    if done:
                        content += thinking_filter.flush()
                total_content += content
                chunk["model"] = cfg.proxy_model_name
                chunk["message"]["content"] = content
                if content or done:
                    yield json.dumps(chunk) + "\n"
        else:
            async for chunk in _iter_openai_stream(response):
                if chunk is None:
                    remaining = thinking_filter.flush() if thinking_filter else ""
                    total_content += remaining
                    yield json.dumps({
                        "model": cfg.proxy_model_name,
                        "message": {"role": "assistant", "content": remaining},
                        "done": True,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                    }) + "\n"
                    break

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "") or ""

                if is_reasoning and "reasoning_content" in delta:
                    reasoning = delta["reasoning_content"]
                    if reasoning:
                        log_manager.log("thinking", f"Filtered reasoning_content ({len(reasoning)} chars)")

                if thinking_filter and content:
                    content = thinking_filter.feed(content)

                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    yield json.dumps({
                        "model": cfg.proxy_model_name,
                        "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                        "done": False,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                    }) + "\n"

                total_content += content
                if content:
                    yield json.dumps({
                        "model": cfg.proxy_model_name,
                        "message": {"role": "assistant", "content": content},
                        "done": False,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                    }) + "\n"

    log_manager.log("response", f"<- {backend.name} stream complete ({len(total_content)} chars output)")


# ==========================================================================
# Streaming proxy: OpenAI SSE output format
# ==========================================================================

async def stream_chat_openai_out(backend: BackendConfig, body: dict):
    """Proxy a chat request -> OpenAI SSE output."""
    cfg = get_config()
    inject_no_think = should_filter_thinking(backend)
    is_reasoning = _is_thinking_model(backend.model)
    thinking_filter = ThinkingFilter() if is_reasoning else None
    client = get_client()
    req_id = f"chatcmpl-{int(time.time())}"

    if backend.type == "ollama":
        target_url = f"{backend.base_url}/api/chat"
        request_body = openai_to_ollama_request(body, backend.model, no_think=inject_no_think)
    else:
        target_url = f"{backend.base_url}/v1/chat/completions"
        request_body = {**body, "model": backend.model}
        if inject_no_think and is_reasoning:
            request_body["messages"] = _inject_no_think(request_body.get("messages", []))

    log_manager.log("request", f"-> {backend.name} ({backend.type}) {target_url}",
                    {"backend": backend.name, "model": backend.model})

    async with client.stream("POST", target_url, json=request_body,
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': f'[Proxy error: backend returned {response.status_code}]'}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if backend.type == "ollama":
            async for chunk in _iter_ollama_stream(response):
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                done = chunk.get("done", False)
                if thinking_filter:
                    content = thinking_filter.feed(content)
                    if done:
                        content += thinking_filter.flush()

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    delta = {"tool_calls": tool_calls}
                    if content:
                        delta["content"] = content
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'model': cfg.proxy_model_name, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
                elif content:
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'model': cfg.proxy_model_name, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

                if done:
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'model': cfg.proxy_model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
        else:
            # OpenAI -> OpenAI passthrough with filtering
            async for chunk in _iter_openai_stream(response):
                if chunk is None:
                    remaining = thinking_filter.flush() if thinking_filter else ""
                    if remaining:
                        yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'model': cfg.proxy_model_name, 'choices': [{'index': 0, 'delta': {'content': remaining}, 'finish_reason': None}]})}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                chunk["model"] = cfg.proxy_model_name
                choices = chunk.get("choices", [{}])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                if is_reasoning and "reasoning_content" in delta:
                    reasoning = delta.pop("reasoning_content", None)
                    if reasoning:
                        log_manager.log("thinking", f"Filtered reasoning_content ({len(reasoning)} chars)")

                content = delta.get("content", "") or ""
                if thinking_filter and content:
                    content = thinking_filter.feed(content)
                    delta["content"] = content

                has_tool_calls = "tool_calls" in delta
                if content or has_tool_calls or choices[0].get("finish_reason"):
                    yield f"data: {json.dumps(chunk)}\n\n"

    log_manager.log("response", f"<- {backend.name} stream complete")


# ==========================================================================
# Streaming proxy: Ollama /api/generate
# ==========================================================================

async def stream_generate_ollama_out(backend: BackendConfig, body: dict):
    """Proxy an /api/generate request -> Ollama generate NDJSON output."""
    cfg = get_config()
    inject_no_think = should_filter_thinking(backend)
    is_reasoning = _is_thinking_model(backend.model)
    thinking_filter = ThinkingFilter() if is_reasoning else None
    client = get_client()

    if backend.type == "ollama":
        target_url = f"{backend.base_url}/api/generate"
        request_body = {**body, "model": backend.model}
        if inject_no_think and is_reasoning:
            prompt = request_body.get("prompt", "")
            if "/no_think" not in prompt:
                request_body["prompt"] = prompt + " /no_think"
    else:
        target_url = f"{backend.base_url}/v1/chat/completions"
        request_body = ollama_generate_to_openai_request(body, backend.model, no_think=inject_no_think)

    log_manager.log("request", f"-> {backend.name} /api/generate",
                    {"backend": backend.name, "model": backend.model})

    async with client.stream("POST", target_url, json=request_body,
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            yield json.dumps({"model": cfg.proxy_model_name, "response": f"[Error: {response.status_code}]", "done": True}) + "\n"
            return

        if backend.type == "ollama":
            async for chunk in _iter_ollama_stream(response):
                content = chunk.get("response", "")
                done = chunk.get("done", False)
                if thinking_filter:
                    content = thinking_filter.feed(content)
                    if done:
                        content += thinking_filter.flush()
                chunk["model"] = cfg.proxy_model_name
                chunk["response"] = content
                if content or done:
                    yield json.dumps(chunk) + "\n"
        else:
            async for chunk in _iter_openai_stream(response):
                if chunk is None:
                    remaining = thinking_filter.flush() if thinking_filter else ""
                    yield json.dumps({"model": cfg.proxy_model_name, "response": remaining, "done": True}) + "\n"
                    break
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "") or ""
                if is_reasoning and "reasoning_content" in delta:
                    delta.pop("reasoning_content", None)
                if thinking_filter and content:
                    content = thinking_filter.feed(content)
                if content:
                    yield json.dumps({"model": cfg.proxy_model_name, "response": content, "done": False}) + "\n"

    log_manager.log("response", f"<- {backend.name} generate complete")


# ==========================================================================
# Buffered streaming (for non-streaming client requests)
# ==========================================================================

async def _stream_and_buffer(backend: BackendConfig, body: dict) -> tuple[str, list]:
    """Stream from backend internally, buffer content + tool_calls.

    Always uses streaming to the backend to keep the connection alive
    and avoid timeouts on long generations.
    """
    cfg = get_config()
    is_reasoning = _is_thinking_model(backend.model)
    inject_no_think = should_filter_thinking(backend)
    thinking_filter = ThinkingFilter() if is_reasoning else None
    client = get_client()

    if backend.type == "ollama":
        target_url = f"{backend.base_url}/api/chat"
        request_body = {**body, "model": backend.model, "stream": True,
                        "messages": _sanitize_messages_for_ollama(body.get("messages", []))}
        if inject_no_think and is_reasoning:
            request_body["messages"] = _inject_no_think(request_body["messages"])
    else:
        target_url = f"{backend.base_url}/v1/chat/completions"
        request_body = ollama_to_openai_request(body, backend.model, no_think=inject_no_think)
        request_body["stream"] = True

    log_manager.log("request", f"-> {backend.name} (buffered-stream)", {"url": target_url})

    content_buf = ""
    tool_calls_buf: list = []
    tool_calls_by_index: dict[int, dict] = {}

    async with client.stream("POST", target_url, json=request_body,
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            return f"[Backend error: {response.status_code}]", []

        if backend.type == "ollama":
            async for chunk in _iter_ollama_stream(response):
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                if thinking_filter:
                    content = thinking_filter.feed(content)
                    if chunk.get("done"):
                        content += thinking_filter.flush()
                content_buf += content
                if msg.get("tool_calls"):
                    tool_calls_buf.extend(msg["tool_calls"])
        else:
            async for chunk in _iter_openai_stream(response):
                if chunk is None:
                    if thinking_filter:
                        content_buf += thinking_filter.flush()
                    break
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "") or ""
                if is_reasoning and "reasoning_content" in delta:
                    delta.pop("reasoning_content", None)
                if thinking_filter and content:
                    content = thinking_filter.feed(content)
                content_buf += content

                # Assemble tool_calls from incremental SSE deltas
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc.get("id", ""), "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_by_index[idx]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        entry["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]
                    if tc.get("id"):
                        entry["id"] = tc["id"]

    if tool_calls_by_index:
        tool_calls_buf = [tool_calls_by_index[k] for k in sorted(tool_calls_by_index)]

    log_manager.log("response", f"<- {backend.name} buffered ({len(content_buf)} chars, {len(tool_calls_buf)} tool_calls)")
    return content_buf, tool_calls_buf


# ==========================================================================
# Non-streaming response builders
# ==========================================================================

async def non_stream_chat_ollama(backend: BackendConfig, body: dict) -> dict:
    cfg = get_config()
    content, tool_calls = await _coalesced_stream_and_buffer(backend, body)
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "model": cfg.proxy_model_name, "message": msg, "done": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }


async def non_stream_chat_openai(backend: BackendConfig, body: dict) -> dict:
    cfg = get_config()
    content, tool_calls = await _coalesced_stream_and_buffer(backend, body)
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion",
        "model": cfg.proxy_model_name,
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
