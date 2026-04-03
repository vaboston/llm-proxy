"""Tests for request format translation functions.

Covers the Ollama ↔ OpenAI translation pipeline for two real backends:

- Ollama + qwen3.5      : openai_to_ollama_request, _sanitize_messages_for_ollama
- LM Studio + qwen3.5  : ollama_to_openai_request (no gemma path taken)

Goal: ensure the Gemma 4 fix (_sanitize_messages_for_gemma) did NOT alter
the qwen3.5 path, and that round-trip field mapping is correct.

All functions are inlined from app/proxy.py to avoid importing the full stack
(httpx/fastapi not available on the system Python). Keep them in sync.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Inlined helpers (mirrors app/proxy.py)
# ---------------------------------------------------------------------------

_THINKING_MODELS = ("qwen3", "qwq", "deepseek-r1")
_GEMMA_MODELS = ("gemma",)


def _is_thinking_model(model_name: str) -> bool:
    return any(t in model_name.lower() for t in _THINKING_MODELS)


def _is_gemma_model(model_name: str) -> bool:
    return any(t in model_name.lower() for t in _GEMMA_MODELS)


def _inject_no_think(messages: list) -> list:
    messages = [m.copy() for m in messages]
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if "/no_think" not in content:
                m["content"] = content + " /no_think"
            break
    return messages


def _sanitize_messages_for_ollama(messages: list) -> list:
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


def _sanitize_messages_for_gemma(messages: list) -> list:
    system_parts, non_system = [], []
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
        m.pop("reasoning_content", None)
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
    return result


def _forward_tools(body: dict, out: dict):
    if "tools" in body:
        out["tools"] = body["tools"]
    if "tool_choice" in body:
        out["tool_choice"] = body["tool_choice"]


def ollama_to_openai_request(body: dict, model: str, no_think: bool = False) -> dict:
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


# ===========================================================================
# Fixtures
# ===========================================================================

MODEL_OLLAMA_QWEN = "qwen3.5:latest"
MODEL_LMS_QWEN   = "qwen/qwen3.5-9b"
MODEL_LMS_GEMMA  = "google/gemma-4-26b-a4b"


# ===========================================================================
# LM Studio + Qwen3.5  →  ollama_to_openai_request
# ===========================================================================

class TestOllamaToOpenaiQwen35:
    """Hermes sends Ollama-format body → proxy converts for LM Studio (qwen3.5)."""

    def test_model_name_substituted(self):
        body = {"messages": [{"role": "user", "content": "Hi"}], "model": "proxy-model"}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        assert result["model"] == MODEL_LMS_QWEN

    def test_stream_defaults_true(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        assert ollama_to_openai_request(body, MODEL_LMS_QWEN)["stream"] is True

    def test_stream_false_forwarded(self):
        body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
        assert ollama_to_openai_request(body, MODEL_LMS_QWEN)["stream"] is False

    def test_options_mapped_to_openai_params(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512, "stop": ["\n"]},
        }
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 512
        assert result["stop"] == ["\n"]

    def test_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "my_fn", "parameters": {}}}]
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": tools,
            "tool_choice": "auto",
        }
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        assert result["tools"] == tools
        assert result["tool_choice"] == "auto"

    def test_no_think_injected_for_qwen3(self):
        body = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN, no_think=True)
        last_user = next(m for m in reversed(result["messages"]) if m["role"] == "user")
        assert "/no_think" in last_user["content"]

    def test_no_think_not_injected_when_disabled(self):
        body = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN, no_think=False)
        last_user = next(m for m in reversed(result["messages"]) if m["role"] == "user")
        assert "/no_think" not in last_user["content"]

    def test_no_think_not_duplicated(self):
        body = {"messages": [{"role": "user", "content": "Hello /no_think"}]}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN, no_think=True)
        assert result["messages"][-1]["content"].count("/no_think") == 1

    def test_gemma_sanitization_NOT_applied(self):
        """qwen3.5 must NOT go through _sanitize_messages_for_gemma."""
        # Arguments as dict: for qwen3.5 they should pass through unchanged
        # (LM Studio + Qwen3.5 accepts both formats; the important thing is
        # we don't accidentally apply Gemma logic to qwen3.5 messages)
        assert not _is_gemma_model(MODEL_LMS_QWEN)

    def test_messages_passed_through_unchanged_for_qwen(self):
        """For non-gemma models, messages are NOT sanitized by _sanitize_messages_for_gemma."""
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "I thought..."},
        ]
        body = {"messages": msgs}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        # reasoning_content must NOT be stripped for qwen (not gemma path)
        assert "reasoning_content" in result["messages"][2]

    def test_multi_turn_conversation_preserved(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        body = {"messages": msgs}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        assert len(result["messages"]) == 4
        assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant", "user"]

    def test_empty_options_not_added(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        assert "temperature" not in result
        assert "max_tokens" not in result


# ===========================================================================
# Ollama backend + Qwen3.5  →  openai_to_ollama_request
# ===========================================================================

class TestOpenaiToOllamaQwen35:
    """Client sends OpenAI-format body → proxy converts for Ollama (qwen3.5)."""

    def test_model_name_substituted(self):
        body = {"messages": [{"role": "user", "content": "Hi"}], "model": "proxy-model"}
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        assert result["model"] == MODEL_OLLAMA_QWEN

    def test_stream_defaults_true(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        assert openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)["stream"] is True

    def test_openai_params_mapped_to_ollama_options(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 256,
            "stop": ["</s>"],
        }
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        assert result["options"]["temperature"] == 0.5
        assert result["options"]["top_p"] == 0.8
        assert result["options"]["num_predict"] == 256
        assert result["options"]["stop"] == ["</s>"]

    def test_empty_options_not_added(self):
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        assert "options" not in result

    def test_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
        body = {"messages": [{"role": "user", "content": "Hi"}], "tools": tools}
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        assert result["tools"] == tools

    def test_no_think_injected_for_qwen3(self):
        body = {"messages": [{"role": "user", "content": "Explain gravity"}]}
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN, no_think=True)
        last_user = next(m for m in reversed(result["messages"]) if m["role"] == "user")
        assert "/no_think" in last_user["content"]

    def test_tool_call_arguments_string_to_dict(self):
        """Core Ollama requirement: arguments must be a parsed dict, not a JSON string."""
        body = {
            "messages": [
                {"role": "user", "content": "List /tmp"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "t1", "type": "function",
                     "function": {"name": "list_dir", "arguments": "{\"path\": \"/tmp\"}"}}
                ]},
            ]
        }
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        args = result["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"path": "/tmp"}

    def test_tool_call_arguments_invalid_json_left_as_string(self):
        """Non-parseable JSON string must survive without crashing."""
        body = {
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "t1", "type": "function",
                     "function": {"name": "fn", "arguments": "not-json"}}
                ]},
            ]
        }
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        args = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        assert args == "not-json"

    def test_tool_role_content_non_string_serialized(self):
        """tool role with dict content must be JSON-serialized to string."""
        body = {
            "messages": [
                {"role": "tool", "tool_call_id": "t1", "content": {"result": 42}},
            ]
        }
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        content = result["messages"][0]["content"]
        assert isinstance(content, str)
        assert json.loads(content) == {"result": 42}

    def test_tool_role_content_string_unchanged(self):
        body = {
            "messages": [
                {"role": "tool", "tool_call_id": "t1", "content": "42"},
            ]
        }
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        assert result["messages"][0]["content"] == "42"

    def test_does_not_mutate_input(self):
        args_str = "{\"x\": 1}"
        body = {
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "t1", "type": "function",
                     "function": {"name": "fn", "arguments": args_str}}
                ]},
            ]
        }
        openai_to_ollama_request(body, MODEL_OLLAMA_QWEN)
        # original string must not have been mutated
        assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == args_str

    def test_full_hermes_tool_round_trip_ollama(self):
        """Simulate a full Hermes tool-call conversation converted for Ollama."""
        body = {
            "messages": [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "List /tmp"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "list_dir", "arguments": "{\"path\": \"/tmp\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
                {"role": "user", "content": "Which is bigger?"},
            ],
            "temperature": 0.3,
        }
        result = openai_to_ollama_request(body, MODEL_OLLAMA_QWEN, no_think=True)

        # Model and stream
        assert result["model"] == MODEL_OLLAMA_QWEN
        assert result["stream"] is True

        # Options
        assert result["options"]["temperature"] == 0.3

        # Arguments converted to dict for Ollama
        args = result["messages"][2]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"path": "/tmp"}

        # tool role content kept as string
        assert result["messages"][3]["content"] == "file1.txt"

        # /no_think on last user message
        assert "/no_think" in result["messages"][4]["content"]


# ===========================================================================
# Isolation: Gemma fix does NOT affect Qwen3.5 path
# ===========================================================================

class TestGemmaFixIsolation:
    """Regression tests: gemma-specific logic must stay isolated."""

    def test_gemma_model_detection_correct(self):
        assert _is_gemma_model(MODEL_LMS_GEMMA) is True
        assert _is_gemma_model(MODEL_LMS_QWEN) is False
        assert _is_gemma_model(MODEL_OLLAMA_QWEN) is False

    def test_reasoning_content_preserved_for_qwen_lmstudio(self):
        """qwen assistant messages with reasoning_content must pass through."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "<think>...</think>"},
        ]
        body = {"messages": msgs}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        assert "reasoning_content" in result["messages"][1]

    def test_reasoning_content_stripped_for_gemma_lmstudio(self):
        """Gemma assistant messages with reasoning_content must be stripped."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "reasoning_content": "<think>...</think>"},
        ]
        body = {"messages": msgs}
        result = ollama_to_openai_request(body, MODEL_LMS_GEMMA)
        assert "reasoning_content" not in result["messages"][1]

    def test_tool_args_dict_NOT_converted_for_qwen_lmstudio(self):
        """For qwen3.5 on LM Studio, dict arguments pass through (no gemma path)."""
        msgs = [
            {"role": "user", "content": "Use tool"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "fn", "arguments": {"x": 1}}}
            ]},
        ]
        body = {"messages": msgs}
        result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        args = result["messages"][1]["tool_calls"][0]["function"]["arguments"]
        # NOT converted — qwen path doesn't call _sanitize_messages_for_gemma
        assert isinstance(args, dict)

    def test_tool_args_dict_converted_for_gemma_lmstudio(self):
        """For gemma4 on LM Studio, dict arguments must be converted to JSON string."""
        msgs = [
            {"role": "user", "content": "Use tool"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "fn", "arguments": {"x": 1}}}
            ]},
        ]
        body = {"messages": msgs}
        result = ollama_to_openai_request(body, MODEL_LMS_GEMMA)
        args = result["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"x": 1}

    def test_system_messages_merged_for_gemma_only(self):
        """Multiple system messages must be merged only for Gemma, not Qwen."""
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        body = {"messages": msgs}

        # Gemma: merged into one
        gemma_result = ollama_to_openai_request(body, MODEL_LMS_GEMMA)
        system_msgs = [m for m in gemma_result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "Rule 1" in system_msgs[0]["content"]
        assert "Rule 2" in system_msgs[0]["content"]

        # Qwen: both kept as-is
        qwen_result = ollama_to_openai_request(body, MODEL_LMS_QWEN)
        qwen_system_msgs = [m for m in qwen_result["messages"] if m["role"] == "system"]
        assert len(qwen_system_msgs) == 2
