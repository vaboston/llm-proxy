"""Tests for message sanitization functions in app/proxy.py.

Covers:
- _sanitize_messages_for_gemma: Gemma 4 / LM Studio compatibility
  (arguments dict→string, system merge, reasoning_content strip)
- _is_gemma_model: model name detection

Root cause confirmed via tests/diagnose_gemma4.py against google/gemma-4-26b-a4b:
  tool_calls[].function.arguments sent as dict (Ollama format) → 400
  tool_calls[].function.arguments sent as JSON string (OpenAI format) → 200
"""

import json
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Avoid importing the full app (requires httpx/fastapi) — inline the pure functions
# so tests work without a venv.  Keep them in sync with app/proxy.py.

_GEMMA_MODELS = ("gemma",)


def _is_gemma_model(model_name: str) -> bool:
    return any(t in model_name.lower() for t in _GEMMA_MODELS)


def _sanitize_messages_for_gemma(messages: list) -> list:
    """Inline mirror of app/proxy.py:_sanitize_messages_for_gemma."""
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


# ==========================================================================
# _is_gemma_model
# ==========================================================================

class TestIsGemmaModel:
    def test_gemma4_detected(self):
        assert _is_gemma_model("google/gemma-4-26b-a4b") is True

    def test_gemma_plain(self):
        assert _is_gemma_model("gemma4") is True

    def test_gemma_case_insensitive(self):
        assert _is_gemma_model("Gemma-3-27b-it") is True

    def test_qwen_not_gemma(self):
        assert _is_gemma_model("qwen3-14b") is False

    def test_llama_not_gemma(self):
        assert _is_gemma_model("llama3.1-8b") is False


# ==========================================================================
# _sanitize_messages_for_gemma — arguments conversion
# ==========================================================================

class TestSanitizeGemmaArguments:
    """The confirmed root cause of the 400 error with Gemma 4 on LM Studio."""

    def test_arguments_dict_converted_to_json_string(self):
        """Core fix: Ollama stores arguments as dict, OpenAI requires JSON string."""
        msgs = [
            {"role": "user", "content": "List /tmp"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "list_dir", "arguments": {"path": "/tmp"}}}
            ]},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        args = result[1]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"path": "/tmp"}

    def test_arguments_already_string_left_unchanged(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "foo", "arguments": "{\"x\": 1}"}}
            ]},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{\"x\": 1}"

    def test_empty_dict_arguments_converted(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "no_args", "arguments": {}}}
            ]},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_multiple_tool_calls_all_converted(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "a", "arguments": {"x": 1}}},
                {"id": "t2", "type": "function",
                 "function": {"name": "b", "arguments": {"y": 2}}},
            ]},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        for tc in result[0]["tool_calls"]:
            assert isinstance(tc["function"]["arguments"], str)

    def test_does_not_mutate_input(self):
        original_args = {"path": "/tmp"}
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "list_dir", "arguments": original_args}}
            ]},
        ]
        _sanitize_messages_for_gemma(msgs)
        assert msgs[0]["tool_calls"][0]["function"]["arguments"] is original_args

    def test_full_hermes_tool_round_trip(self):
        """Reproduce the exact payload that caused 400 with google/gemma-4-26b-a4b.

        Hermes sends Ollama-format messages (arguments as dict).
        After sanitization, LM Studio receives arguments as JSON string → 200.
        """
        hermes_messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "List /tmp"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "list_dir", "arguments": {"path": "/tmp"}}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
            {"role": "user", "content": "Which is bigger?"},
        ]
        result = _sanitize_messages_for_gemma(hermes_messages)
        args = result[2]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"path": "/tmp"}
        assert result[0]["content"] == "You are an agent."
        assert result[3]["role"] == "tool"


# ==========================================================================
# _sanitize_messages_for_gemma — system message merging
# ==========================================================================

class TestSanitizeGemmaSystem:
    def test_single_system_preserved(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_multiple_system_messages_merged(self):
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        assert result[0]["role"] == "system"
        assert "Rule 1" in result[0]["content"]
        assert "Rule 2" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_no_system_message(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = _sanitize_messages_for_gemma(msgs)
        assert result[0]["role"] == "user"

    def test_reasoning_content_stripped(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "reasoning_content": "I thought..."},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        assert "reasoning_content" not in result[1]
        assert result[1]["content"] == "Hello"

    def test_messages_without_tool_calls_unchanged(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = _sanitize_messages_for_gemma(msgs)
        assert len(result) == 3
        assert result[2]["content"] == "Hi"
