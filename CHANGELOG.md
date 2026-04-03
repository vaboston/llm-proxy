# Changelog

## v0.3.0 — Load Balancing

### Added

- **Load balancing**: 4 strategies to distribute requests across backends
  - `priority` — first available by list order (default, existing behavior)
  - `round-robin` — rotate evenly between backends
  - `least-connections` — pick backend with fewest active requests
  - `weighted-round-robin` — rotate proportionally using per-backend weights
- **Per-backend settings**: `weight` (traffic share) and `max_concurrent` (request limit per backend)
- **Live backend status**: active request count, total requests, average latency, error rate — visible in Web UI and via `GET /balancer`
- **Request tracking**: every request is tracked with timing, counts automatically incremented/decremented
- **Backend edit**: inline edit form on each backend card to modify all fields (name, type, host, port, model, weight, max_concurrent, thinking, API key/base)
- **Strategy selector**: new dropdown in Settings to switch load balancing strategy

### Changed

- Backend selection now goes through the load balancer instead of simple first-available loop
- Backend cards show weight, max concurrent, and live active request count
- Backends at max capacity are automatically skipped

## v0.2.0 — Cloud Backends

### Added

- **Cloud backend support**: new `cloud` backend type for remote APIs with API key authentication (Bearer token)
- **Moonshot AI (Kimi)**: tested with Kimi K2.5, including tool calling and thinking disabled mode
- **Provider presets**: Web UI includes pre-configured presets for Moonshot AI, OpenRouter, Together AI, and Groq with auto-filled base URLs and model lists
- **API key management**: keys stored in `config.json`, masked in Web UI and API responses (`****abcd`)
- **Message sanitization for cloud APIs**: automatic injection of `reasoning_content` field required by Kimi's thinking models
- **`api_base` field**: custom base URL per backend, supporting providers that include `/v1` in their base path
- **Stats dashboard**: new Stats tab with request counts, token usage, tool call frequency, and backend distribution (persisted to `stats.json`)
- **Abort button**: cancel all in-flight backend requests from the Web UI header
- **Backend reorder buttons**: up/down arrows replace drag & drop for reliable priority reordering

### Fixed

- **Double `/v1` in cloud URLs**: `api_base` ending with `/v1` no longer produces `/v1/v1/chat/completions`
- **Ollama tool_calls format**: `function.arguments` properly converted from JSON string (OpenAI) to parsed object (Ollama)
- **SSE error handling**: LM Studio/llama.cpp `event: error` SSE events are now detected and forwarded to the client instead of being silently dropped
- **Thinking filter**: `<think>` blocks are now always stripped for reasoning models regardless of the disable_thinking toggle (toggle controls `/no_think` injection only)
- **Request deduplication**: identical concurrent requests are coalesced into a single backend call; results cached for 2 minutes
- **Client disconnect resilience**: backend generation continues in background when client disconnects; retries reuse the in-flight result
- **Route ordering**: `/config/backends/reorder` no longer matches as `backend_id` parameter
- **Log verbosity**: content tokens aggregated into single summary lines instead of one log entry per token

### Changed

- Non-streaming responses now use internal streaming to the backend (prevents timeout on long generations)
- Conversation summary in logs shows last 6 messages with role icons
- Cloud backend health checks and model fetches include proper authentication headers

## v0.1.0 — Proxy One

First public release.

- Dual API: Ollama + OpenAI compatible endpoints
- Multi-backend: Ollama, LM Studio, llama.cpp with priority-based fallback
- Full tool calling support with format translation
- Thinking suppression for Qwen3, QwQ, DeepSeek-R1
- Real-time streaming
- Web UI with backend management and live logs
- Persistent config via `config.json`

---

100% vibecoded with [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6).
