"""LLM Proxy - FastAPI app exposing Ollama & OpenAI compatible endpoints."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from app.config import BackendConfig, get_config, save_config, load_config
from app.proxy import (
    get_available_backend, fetch_backend_models, close_client,
    stream_chat_ollama_out, stream_chat_openai_out, stream_generate_ollama_out,
    non_stream_chat_ollama, non_stream_chat_openai, check_backend_health,
    abort_all_inflight, _tracked_stream,
)
from app.logs import log_manager
from app.stats import stats
from app.balancer import balancer


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config()
    log_manager.log("info", "Proxy started", {"port": 11435})
    yield
    stats.save()
    await close_client()


app = FastAPI(title="LLM Proxy", lifespan=lifespan)


# ==========================================================================
# Helpers
# ==========================================================================

def _summarize_messages(messages: list[dict]) -> list[str]:
    """Build a human-readable summary of the conversation tail for logs."""
    lines = []
    tail = [m for m in messages if m.get("role") != "system"][-6:]
    for m in tail:
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        preview = content[:120].replace("\n", "\\n").strip()

        if role == "user":
            lines.append(f"user: {preview}" + ("..." if len(content) > 120 else ""))
        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?")
                         for tc in tool_calls if isinstance(tc, dict)]
                lines.append(f"assistant: tool_calls: {', '.join(names)}")
            elif preview:
                lines.append(f"assistant: {preview}" + ("..." if len(content) > 120 else ""))
            else:
                lines.append("assistant: (empty)")
        elif role == "tool":
            name = m.get("name", "?")
            lines.append(f"tool({name}): {preview[:80]}" + ("..." if len(content) > 80 else ""))
    return lines


# ==========================================================================
# Ollama-compatible endpoints
# ==========================================================================

@app.post("/api/chat")
async def ollama_chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", True)
    stats.record_request("/api/chat", stream)
    log_manager.log("request", f"/api/chat - {len(messages)} msgs",
                    {"conversation": _summarize_messages(messages),
                     "model": body.get("model"), "stream": stream})

    backend = await get_available_backend()
    if not backend:
        raise HTTPException(503, "No backends available")
    stats.record_backend(backend.name)

    if stream:
        return StreamingResponse(
            _tracked_stream(backend.id, stream_chat_ollama_out(backend, body)),
            media_type="application/x-ndjson",
        )
    else:
        async def _gen():
            with balancer.track(backend.id):
                result = await non_stream_chat_ollama(backend, body)
            yield json.dumps(result)
        return StreamingResponse(_gen(), media_type="application/json")


@app.post("/api/generate")
async def ollama_generate(request: Request):
    body = await request.json()
    stream = body.get("stream", True)
    stats.record_request("/api/generate", stream)

    backend = await get_available_backend()
    if not backend:
        raise HTTPException(503, "No backends available")
    stats.record_backend(backend.name)

    if stream:
        return StreamingResponse(
            _tracked_stream(backend.id, stream_generate_ollama_out(backend, body)),
            media_type="application/x-ndjson",
        )
    else:
        chat_body = {
            "messages": [{"role": "user", "content": body.get("prompt", "")}],
            "stream": False,
        }
        if body.get("system"):
            chat_body["messages"].insert(0, {"role": "system", "content": body["system"]})
        if "options" in body:
            chat_body["options"] = body["options"]

        async def _gen():
            with balancer.track(backend.id):
                result = await non_stream_chat_ollama(backend, chat_body)
                content = result.get("message", {}).get("content", "")
            yield json.dumps({
                "model": get_config().proxy_model_name,
                "response": content, "done": True,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            })
        return StreamingResponse(_gen(), media_type="application/json")


@app.get("/api/tags")
async def ollama_tags():
    cfg = get_config()
    return {"models": [{
        "name": cfg.proxy_model_name,
        "model": cfg.proxy_model_name,
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "size": 0, "digest": "proxy",
        "details": {"parent_model": "", "format": "proxy", "family": "proxy",
                     "parameter_size": "proxy", "quantization_level": "proxy"},
    }]}


@app.get("/api/version")
async def ollama_version():
    return {"version": "proxy-0.1.0"}


# ==========================================================================
# OpenAI-compatible endpoints
# ==========================================================================

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    stream = body.get("stream", False)
    stats.record_request("/v1/chat/completions", stream)
    log_manager.log("request",
        f"/v1/chat/completions - {len(messages)} msgs" + (f", {len(tools)} tools" if tools else ""),
        {"conversation": _summarize_messages(messages),
         "tools": [t.get("function", {}).get("name", "?") for t in tools] if tools else [],
         "model": body.get("model"), "stream": stream})

    backend = await get_available_backend()
    if not backend:
        raise HTTPException(503, "No backends available")
    stats.record_backend(backend.name)

    if stream:
        return StreamingResponse(
            _tracked_stream(backend.id, stream_chat_openai_out(backend, body)),
            media_type="text/event-stream",
        )
    else:
        async def _gen():
            with balancer.track(backend.id):
                result = await non_stream_chat_openai(backend, body)
            yield json.dumps(result)
        return StreamingResponse(_gen(), media_type="application/json")


@app.get("/v1/models")
async def openai_models():
    cfg = get_config()
    return {"object": "list", "data": [{
        "id": cfg.proxy_model_name, "object": "model",
        "created": int(time.time()), "owned_by": "proxy",
    }]}


# ==========================================================================
# Config API
# ==========================================================================

@app.get("/config")
async def get_full_config():
    return get_config().model_dump()


@app.put("/config/settings")
async def update_settings(request: Request):
    body = await request.json()
    cfg = get_config()
    for key in ("proxy_model_name", "timeout", "disable_thinking_global", "load_balancing_strategy"):
        if key in body:
            setattr(cfg, key, body[key])
    save_config()
    log_manager.log("info", "Settings updated", body)
    return cfg.model_dump()


@app.get("/config/backends")
async def list_backends():
    return [b.model_dump() for b in get_config().backends]


@app.post("/config/backends")
async def add_backend(request: Request):
    body = await request.json()
    backend = BackendConfig(**body)
    cfg = get_config()
    cfg.backends.append(backend)
    save_config()
    log_manager.log("info", f"Backend added: {backend.name}")
    return backend.model_dump()


@app.put("/config/backends/reorder")
async def reorder_backends(request: Request):
    body = await request.json()
    ids = body.get("ids", [])
    cfg = get_config()
    by_id = {b.id: b for b in cfg.backends}
    reordered = [by_id[bid] for bid in ids if bid in by_id]
    seen = set(ids)
    for b in cfg.backends:
        if b.id not in seen:
            reordered.append(b)
    cfg.backends = reordered
    save_config()
    log_manager.log("info", "Backends reordered")
    return [b.model_dump() for b in cfg.backends]


@app.put("/config/backends/{backend_id}")
async def update_backend(backend_id: str, request: Request):
    body = await request.json()
    cfg = get_config()
    for i, b in enumerate(cfg.backends):
        if b.id == backend_id:
            updated = b.model_dump()
            updated.update(body)
            updated["id"] = backend_id
            cfg.backends[i] = BackendConfig(**updated)
            save_config()
            log_manager.log("info", f"Backend updated: {cfg.backends[i].name}")
            return cfg.backends[i].model_dump()
    raise HTTPException(404, "Backend not found")


@app.delete("/config/backends/{backend_id}")
async def delete_backend(backend_id: str):
    cfg = get_config()
    cfg.backends = [b for b in cfg.backends if b.id != backend_id]
    save_config()
    log_manager.log("info", f"Backend deleted: {backend_id}")
    return {"ok": True}


@app.post("/config/backends/probe-models")
async def probe_models(request: Request):
    body = await request.json()
    tmp = BackendConfig(name="_probe", **body)
    models = await fetch_backend_models(tmp)
    return {"models": models}


@app.get("/config/backends/{backend_id}/models")
async def get_backend_models(backend_id: str):
    cfg = get_config()
    for b in cfg.backends:
        if b.id == backend_id:
            return {"models": await fetch_backend_models(b)}
    raise HTTPException(404, "Backend not found")


@app.get("/config/backends/{backend_id}/health")
async def backend_health(backend_id: str):
    cfg = get_config()
    for b in cfg.backends:
        if b.id == backend_id:
            return {"healthy": await check_backend_health(b)}
    raise HTTPException(404, "Backend not found")


# ==========================================================================
# Abort / Stats / Logs
# ==========================================================================

@app.get("/balancer")
async def get_balancer_status():
    return {
        "strategy": get_config().load_balancing_strategy,
        "backends": balancer.get_all_states(),
    }


@app.post("/abort")
async def abort_requests():
    count = abort_all_inflight()
    log_manager.log("info", f"Aborted {count} in-flight request(s)")
    return {"aborted": count}


@app.get("/stats")
async def get_stats():
    return stats.to_dict()


@app.post("/stats/reset")
async def reset_stats():
    stats.reset()
    log_manager.log("info", "Stats reset")
    return {"ok": True}


@app.get("/logs/stream")
async def logs_stream():
    queue = log_manager.subscribe()

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            log_manager.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/logs/recent")
async def logs_recent():
    return log_manager.get_recent(100)


# ==========================================================================
# Web UI
# ==========================================================================

@app.get("/")
async def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())
