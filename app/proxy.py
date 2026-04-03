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
from app.balancer import balancer

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

def _auth_headers(backend: BackendConfig) -> dict:
    """Return auth headers for a backend (empty dict for local backends)."""
    if backend.api_key:
        return {"Authorization": f"Bearer {backend.api_key}"}
    return {}


def _is_cloud(backend: BackendConfig) -> bool:
    return backend.type == "cloud" or bool(backend.api_key)


def _chat_url(backend: BackendConfig) -> str:
    """Return the chat completions URL, accounting for cloud base URLs that already include /v1."""
    if _is_cloud(backend):
        return f"{backend.base_url}/chat/completions"
    return f"{backend.base_url}/v1/chat/completions"


async def check_backend_health(backend: BackendConfig) -> bool:
    client = get_client()
    try:
        if backend.type == "ollama":
            url = f"{backend.base_url}/api/tags"
        elif _is_cloud(backend):
            url = f"{backend.base_url}/models"
        else:
            url = f"{backend.base_url}/v1/models"
        resp = await client.get(url, timeout=5.0, headers=_auth_headers(backend))
        healthy = resp.status_code == 200
        balancer.mark_healthy(backend.id, healthy)
        return healthy
    except Exception:
        balancer.mark_healthy(backend.id, False)
        return False


async def get_available_backend() -> BackendConfig | None:
    """Select a backend using the configured load balancing strategy."""
    available = []
    for b in get_config().backends:
        if not b.enabled:
            continue
        # Check max_concurrent limit
        if b.max_concurrent > 0:
            state = balancer.get_state(b.id)
            if state.active_requests >= b.max_concurrent:
                log_manager.log("info", f"Backend '{b.name}' at capacity ({state.active_requests}/{b.max_concurrent})")
                continue
        if await check_backend_health(b):
            available.append(b)
        else:
            log_manager.log("error", f"Backend '{b.name}' unreachable, skipping")

    selected = await balancer.select_backend(available)
    if selected:
        log_manager.log("info", f"Selected backend '{selected.name}' (strategy={get_config().load_balancing_strategy}, "
                        f"active={balancer.get_state(selected.id).active_requests})")
    return selected


async def fetch_backend_models(backend: BackendConfig) -> list[str]:
    client = get_client()
    headers = _auth_headers(backend)
    try:
        if backend.type == "ollama":
            resp = await client.get(f"{backend.base_url}/api/tags", timeout=5.0, headers=headers)
            return [m["name"] for m in resp.json().get("models", [])]
        elif _is_cloud(backend):
            resp = await client.get(f"{backend.base_url}/models", timeout=5.0, headers=headers)
            return [m["id"] for m in resp.json().get("data", [])]
        else:
            resp = await client.get(f"{backend.base_url}/v1/models", timeout=5.0, headers=headers)
            return [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        log_manager.log("error", f"Failed to fetch models from '{backend.name}': {e}")
        return []


# ==========================================================================
# Thinking suppression
# ==========================================================================

_THINKING_MODELS = ("qwen3", "qwq", "deepseek-r1")
_GEMMA_MODELS = ("gemma",)


def _is_thinking_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(t in lower for t in _THINKING_MODELS)


def _is_gemma_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(t in lower for t in _GEMMA_MODELS)


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

def _sanitize_messages_for_gemma(messages: list[dict]) -> list[dict]:
    """Sanitize messages for Gemma 4 compatibility.

    Gemma 4 only supports a single system turn at the beginning.
    - Merges all system messages into one at position 0
    - Removes tool_calls from assistant messages in history (Gemma 4's
      native tool format is handled by LM Studio's template; passing
      OpenAI-format tool_calls in history can break the template)
    - Strips reasoning_content that may have leaked into history
    - Converts tool_calls[].function.arguments from dict to JSON string
      (Ollama stores arguments as a parsed object; OpenAI/LM Studio requires
      a JSON string — sending a dict causes 400 "Invalid 'content'" from LM Studio)
    """
    # Collect all system message contents
    system_parts = []
    non_system = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "") or ""
            if content:
                system_parts.append(content)
        else:
            non_system.append(m.copy())

    result = []
    if system_parts:
        result.append({"role": "system", "content": "\n\n".join(system_parts)})

    for m in non_system:
        # Strip fields that can break Gemma's template
        m.pop("reasoning_content", None)
        # Convert tool_calls arguments from dict (Ollama) to JSON string (OpenAI)
        if m.get("tool_calls"):
            fixed = []
            for tc in m["tool_calls"]:
                tc = tc.copy()
                fn = tc.get("function", {})
                if isinstance(fn, dict):
                    fn = fn.copy()
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        fn["arguments"] = json.dumps(args)
                    tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        result.append(m)

    if len(system_parts) > 1:
        log_manager.log("info",
            f"[gemma] merged {len(system_parts)} system messages into one")

    return result


def _sanitize_messages_for_cloud(messages: list[dict]) -> list[dict]:
    """Sanitize messages for cloud APIs (Moonshot Kimi, etc.).

    Kimi requires reasoning_content on ALL assistant messages when thinking
    is enabled on the model. Empty string is rejected for tool_call messages,
    so we inject a placeholder.
    """
    out = []
    for m in messages:
        m = m.copy()
        if m.get("role") == "assistant":
            if "reasoning_content" not in m or not m["reasoning_content"]:
                m["reasoning_content"] = "ok"
        out.append(m)
    return out


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
    if _is_gemma_model(model):
        messages = _sanitize_messages_for_gemma(messages)
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

class BackendRetryableError(Exception):
    """Raised when a backend returns a retryable status (429, 502, 503)."""
    def __init__(self, backend_name: str, status_code: int):
        self.backend_name = backend_name
        self.status_code = status_code
        super().__init__(f"{backend_name} returned {status_code}")


_inflight: dict[str, asyncio.Task] = {}
_RESULT_CACHE_TTL = 300  # seconds to keep completed results

_RETRYABLE_STATUS = {429, 503, 502}


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
    """Deduplicated _stream_and_buffer with retry on 429/502/503.

    If an identical request is already in-flight, wait for it instead of
    starting a new one. Completed results are cached for 5 minutes.
    On retryable errors, automatically tries the next available backend.
    """
    key = _request_key(body)

    if key in _inflight:
        task = _inflight[key]
        if task.done():
            # Check if the cached result was an error
            try:
                result = task.result()
                log_manager.log("info", "returning cached result for duplicate request")
                stats.record_cached()
                return result
            except BackendRetryableError:
                _inflight.pop(key, None)  # Clear failed cache, retry below
            except Exception:
                _inflight.pop(key, None)
        else:
            log_manager.log("info", "request coalesced - waiting for in-flight duplicate")
            stats.record_coalesced()
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                log_manager.log("info", "client disconnected, backend task continues in background")
                raise
            except BackendRetryableError:
                pass  # Fall through to retry below

    # Try with the given backend, then fallback to others on retryable errors
    tried = {backend.id}
    current_backend = backend

    while True:
        task = asyncio.create_task(_stream_and_buffer(current_backend, body))
        _inflight[key] = task
        task.add_done_callback(
            lambda t, k=key: asyncio.get_event_loop().call_later(
                _RESULT_CACHE_TTL, lambda: _inflight.pop(k, None))
        )

        try:
            result = await asyncio.shield(task)
            return result
        except asyncio.CancelledError:
            log_manager.log("info", "client disconnected, backend task continues in background")
            raise
        except BackendRetryableError as e:
            _inflight.pop(key, None)
            balancer.mark_error(current_backend.id)
            log_manager.log("info",
                f"Backend '{e.backend_name}' returned {e.status_code}, trying next backend")

            # Find next available backend
            next_backend = None
            for b in get_config().backends:
                if not b.enabled or b.id in tried:
                    continue
                if await check_backend_health(b):
                    next_backend = b
                    break

            if next_backend is None:
                log_manager.log("error", "All backends failed or exhausted")
                return f"[All backends unavailable (last: {e.status_code})]", []

            tried.add(next_backend.id)
            current_backend = next_backend
            log_manager.log("info", f"Retrying with backend '{current_backend.name}'")


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
        url = _chat_url(backend)
        req = ollama_to_openai_request(body, backend.model, no_think=inject_no_think)
        if _is_cloud(backend):
            req["messages"] = _sanitize_messages_for_cloud(req["messages"])
    return url, req


async def _find_fallback_backend(exclude: set[str] = set()) -> BackendConfig | None:
    """Find next available backend, excluding already-tried ones."""
    for b in get_config().backends:
        if not b.enabled or b.id in exclude:
            continue
        if await check_backend_health(b):
            return b
    return None


async def _tracked_stream(backend_id: str, gen):
    """Wrap an async generator with request tracking."""
    tracker = balancer.track(backend_id)
    tracker.__enter__()
    exc_info = (None, None, None)
    try:
        async for chunk in gen:
            yield chunk
    except BaseException as e:
        exc_info = (type(e), e, e.__traceback__)
        raise
    finally:
        tracker.__exit__(*exc_info)


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
                             headers=_auth_headers(backend),
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        ct = response.headers.get("content-type", "?")
        log_manager.log("info", f"[backend] HTTP {response.status_code} content-type={ct}")

        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend '{backend.name}' returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            # On retryable errors, try next backend
            if response.status_code in _RETRYABLE_STATUS:
                balancer.mark_error(backend.id)
                next_b = await _find_fallback_backend(exclude={backend.id})
                if next_b:
                    log_manager.log("info", f"Retrying with backend '{next_b.name}'")
                    async for chunk in stream_chat_ollama_out(next_b, body):
                        yield chunk
                    return
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
        target_url = _chat_url(backend)
        request_body = {**body, "model": backend.model}
        msgs = request_body.get("messages", [])
        if _is_gemma_model(backend.model):
            msgs = _sanitize_messages_for_gemma(msgs)
        if _is_cloud(backend):
            msgs = _sanitize_messages_for_cloud(msgs)
        if inject_no_think and is_reasoning:
            msgs = _inject_no_think(msgs)
        request_body["messages"] = msgs

    log_manager.log("request", f"-> {backend.name} ({backend.type}) {target_url}",
                    {"backend": backend.name, "model": backend.model})

    async with client.stream("POST", target_url, json=request_body,
                             headers=_auth_headers(backend),
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend '{backend.name}' returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            if response.status_code in _RETRYABLE_STATUS:
                balancer.mark_error(backend.id)
                next_b = await _find_fallback_backend(exclude={backend.id})
                if next_b:
                    log_manager.log("info", f"Retrying with backend '{next_b.name}'")
                    async for chunk in stream_chat_openai_out(next_b, body):
                        yield chunk
                    return
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
        target_url = _chat_url(backend)
        request_body = ollama_generate_to_openai_request(body, backend.model, no_think=inject_no_think)

    log_manager.log("request", f"-> {backend.name} /api/generate",
                    {"backend": backend.name, "model": backend.model})

    async with client.stream("POST", target_url, json=request_body,
                             headers=_auth_headers(backend),
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
        msgs = _sanitize_messages_for_ollama(body.get("messages", []))
        if _is_gemma_model(backend.model):
            msgs = _sanitize_messages_for_gemma(msgs)
        if inject_no_think and is_reasoning:
            msgs = _inject_no_think(msgs)
        request_body = {**body, "model": backend.model, "stream": True, "messages": msgs}
    else:
        target_url = _chat_url(backend)
        request_body = ollama_to_openai_request(body, backend.model, no_think=inject_no_think)
        request_body["stream"] = True
        if _is_cloud(backend):
            request_body["messages"] = _sanitize_messages_for_cloud(request_body["messages"])

    log_manager.log("request", f"-> {backend.name} (buffered-stream)", {"url": target_url})

    content_buf = ""
    tool_calls_buf: list = []
    tool_calls_by_index: dict[int, dict] = {}

    async with client.stream("POST", target_url, json=request_body,
                             headers=_auth_headers(backend),
                             timeout=httpx.Timeout(cfg.timeout, connect=5.0)) as response:
        if response.status_code != 200:
            error_body = await response.aread()
            log_manager.log("error", f"Backend '{backend.name}' returned {response.status_code}",
                            {"body": error_body.decode(errors="replace")})
            if response.status_code in _RETRYABLE_STATUS:
                raise BackendRetryableError(backend.name, response.status_code)
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
