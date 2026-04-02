# LLM Proxy

A local LLM proxy that exposes both **Ollama** and **OpenAI**-compatible APIs, routing requests to multiple backends with automatic fallback.

Tested with **Ollama** and **LM Studio** running **Qwen3.5-9B**.

## Features

- **Multi-backend**: supports Ollama, LM Studio, and llama.cpp. Each backend is configured with host, port, and model. Automatic fallback by priority order if a backend is unreachable.
- **Dual API**: exposes Ollama endpoints (`/api/chat`, `/api/generate`, `/api/tags`) and OpenAI endpoints (`/v1/chat/completions`, `/v1/models`). Translates between formats transparently.
- **Tool calling**: full support for `tools`, `tool_choice`, and `tool_calls` in both directions. Handles the OpenAI string-arguments vs Ollama object-arguments difference automatically.
- **Thinking suppression**: for reasoning models (Qwen3, QwQ, DeepSeek-R1), strips `<think>...</think>` blocks and `reasoning_content` from responses. Optionally injects `/no_think` to prevent thinking entirely.
- **Real-time streaming**: responses are streamed directly to the client without buffering.
- **Request deduplication**: identical concurrent requests are coalesced into a single backend call. Results are cached for 2 minutes so retries get instant responses.
- **Disconnect resilience**: when a client disconnects mid-generation, the backend task continues in the background. The next retry picks up the same result.
- **Web UI**: dark-themed interface to manage backends, view live logs, and monitor stats.
- **Live logs**: real-time SSE log stream with expandable entries. Content tokens are aggregated (not one line per token), tool calls and errors are shown individually.
- **Stats dashboard**: request counts, token usage, tool call frequency, backend distribution. Persisted to disk.
- **Persistent config**: `config.json` with REST API for CRUD operations.

## Quick Start

```bash
git clone <repo-url> && cd proxy-llmv2
./run.sh
```

The script creates a Python venv, installs dependencies, and starts the server on port **11435**.

Open the Web UI at `http://localhost:11435`, add a backend, and you're ready.

## Usage

Point any Ollama or OpenAI client at the proxy:

```bash
# As an Ollama replacement
export OLLAMA_HOST=http://localhost:11435

# As an OpenAI endpoint
export OPENAI_API_BASE=http://localhost:11435/v1
export OPENAI_API_KEY=not-needed
```

### Hermes Agent

The proxy acts as a drop-in Ollama replacement. In your Hermes config, point the Ollama provider to the proxy:

```yaml
# hermes config
providers:
  ollama:
    base_url: http://<proxy-ip>:11435  # instead of http://localhost:11434
    model: proxy-model
```

Or if Hermes uses the OpenAI provider:

```yaml
providers:
  openai:
    base_url: http://<proxy-ip>:11435/v1
    api_key: not-needed
    model: proxy-model
```

The model name must match the `proxy_model_name` in the proxy settings (default: `proxy-model`). The proxy translates it to the actual model configured on each backend.

### Open WebUI

In Open WebUI settings, add the proxy as an Ollama connection:

```
Settings > Connections > Ollama
URL: http://<proxy-ip>:11435
```

The proxy will appear as an Ollama server with the model `proxy-model`. All tool calls, streaming, and multi-turn conversations work transparently.

### OpenClaw / any OpenAI-compatible client

Any client that supports the OpenAI API can use the proxy:

```bash
export OPENAI_API_BASE=http://<proxy-ip>:11435/v1
export OPENAI_API_KEY=not-needed
export OPENAI_MODEL=proxy-model
```

### curl

```bash
# Ollama format
curl http://localhost:11435/api/chat -d '{
  "model": "proxy-model",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# OpenAI format
curl http://localhost:11435/v1/chat/completions -d '{
  "model": "proxy-model",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

### How it works

The client talks to the proxy using any model name (it's ignored — the proxy uses `proxy_model_name` for responses). The proxy picks a backend based on the load balancing strategy, translates the request format if needed (OpenAI <-> Ollama), and forwards to the actual backend model. The response is translated back and streamed to the client.

```
Client (Hermes, Open WebUI, curl...)
  │
  ▼
LLM Proxy (:11435)
  ├── /api/chat, /api/generate     (Ollama API)
  └── /v1/chat/completions         (OpenAI API)
        │
        ▼  (format translation + thinking filter)
  ┌─────┴─────────┐
  │  Backend #1   │  Ollama (localhost:11434) — qwen3:8b
  │  Backend #2   │  LM Studio (192.168.1.19:1234) — qwen/qwen3.5-9b
  │  Backend #3   │  llama.cpp (server2:8080) — model.gguf
  └───────────────┘
```

## Configuration

Copy the example config:

```bash
cp config.json.example config.json
```

Or configure via the Web UI at `http://localhost:11435`.

### Backend types

| Type | Default port | API used |
|------|-------------|----------|
| `ollama` | 11434 | Ollama native API |
| `lmstudio` | 1234 | OpenAI-compatible API |
| `llamacpp` | 8080 | OpenAI-compatible API |

### Thinking suppression

For reasoning models (Qwen3, QwQ, DeepSeek-R1):

- **`<think>` filtering** is always active: `<think>...</think>` blocks and `reasoning_content` fields are stripped from responses automatically.
- **`/no_think` injection** is controlled by the "Disable thinking" toggle (per backend or globally). When enabled, `/no_think` is appended to the last user message, telling the model to skip reasoning entirely (saves compute).

## API Reference

### Proxy endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/chat` | Ollama chat (streaming/non-streaming) |
| `POST /api/generate` | Ollama generate |
| `GET /api/tags` | Ollama model list |
| `POST /v1/chat/completions` | OpenAI chat completions |
| `GET /v1/models` | OpenAI model list |

### Management endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /config` | Full config |
| `PUT /config/settings` | Update global settings |
| `GET /config/backends` | List backends |
| `POST /config/backends` | Add backend |
| `PUT /config/backends/reorder` | Reorder backends |
| `PUT /config/backends/{id}` | Update backend |
| `DELETE /config/backends/{id}` | Delete backend |
| `POST /config/backends/probe-models` | Fetch models from arbitrary host |
| `GET /config/backends/{id}/models` | Fetch models from saved backend |
| `GET /config/backends/{id}/health` | Health check |
| `POST /abort` | Cancel all in-flight requests |
| `GET /stats` | Usage statistics |
| `POST /stats/reset` | Reset statistics |
| `GET /logs/stream` | SSE log stream |
| `GET /logs/recent` | Recent log entries |

## Deployment

### systemd

```bash
sudo cp llm-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-proxy
```

Edit `llm-proxy.service` to match your install path and user.

## Architecture

```
app/
  main.py       FastAPI app, all HTTP endpoints
  proxy.py      Backend routing, format translation, streaming, deduplication
  config.py     Pydantic models, JSON persistence
  thinking.py   <think> tag filter for streaming text
  logs.py       Real-time log manager with SSE broadcast
  stats.py      Usage statistics with disk persistence
  static/
    index.html  Single-page Web UI
```

## Stack

Python 3.11+, FastAPI, uvicorn, httpx, Pydantic.

---

100% vibecoded with [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6).
