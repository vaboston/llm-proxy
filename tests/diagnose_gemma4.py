"""Diagnostic: find the exact message format causing Gemma 4 to return 400.

Usage:
    python3 tests/diagnose_gemma4.py --host 127.0.0.1 --port 1234 --model <name>

Sends different message structures directly to LM Studio /v1/chat/completions
and shows which ones fail, then also shows what the proxy currently sends for
a typical Hermes tool-call conversation.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def post(url: str, body: dict) -> tuple[int, str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode(errors="replace")[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:600]
    except Exception as e:
        return 0, str(e)


# ---------------------------------------------------------------------------
# Simulate what the proxy currently sends (mirrors app/proxy.py logic)
# ---------------------------------------------------------------------------

def _proxy_ollama_to_openai(ollama_messages: list, model: str) -> list:
    """Mirror of ollama_to_openai_request (no sanitization, as in current code)."""
    return ollama_messages  # current code passes messages as-is


def _merge_consecutive(messages: list) -> list:
    """Mirror of _merge_consecutive_messages added in previous fix."""
    if not messages:
        return messages
    out = []
    for m in messages:
        m = m.copy()
        if (out and out[-1].get("role") == m.get("role")
                and m.get("role") != "system"
                and isinstance(m.get("content"), str)
                and isinstance(out[-1].get("content"), str)):
            out[-1] = out[-1].copy()
            out[-1]["content"] = out[-1]["content"] + "\n\n" + m["content"]
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Typical Hermes messages (Ollama format → converted to OpenAI for LM Studio)
# ---------------------------------------------------------------------------

# A typical multi-turn Hermes conversation with a tool call
HERMES_TOOL_CALL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
    {"role": "user", "content": "What files are in /tmp?"},
    # Hermes/Ollama format: tool_calls with arguments as dict (object), not string
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_abc", "type": "function",
         "function": {"name": "list_dir", "arguments": {"path": "/tmp"}}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc", "content": "file1.txt\nfile2.txt"},
    {"role": "user", "content": "Thanks, which is bigger?"},
]

# Same but arguments already as JSON string (OpenAI format)
HERMES_TOOL_CALL_MESSAGES_ARGS_STRING = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
    {"role": "user", "content": "What files are in /tmp?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_abc", "type": "function",
         "function": {"name": "list_dir", "arguments": "{\"path\": \"/tmp\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc", "content": "file1.txt\nfile2.txt"},
    {"role": "user", "content": "Thanks, which is bigger?"},
]

# Same but content="" instead of None on assistant
HERMES_TOOL_CALL_MESSAGES_EMPTY_CONTENT = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
    {"role": "user", "content": "What files are in /tmp?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_abc", "type": "function",
         "function": {"name": "list_dir", "arguments": "{\"path\": \"/tmp\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc", "content": "file1.txt\nfile2.txt"},
    {"role": "user", "content": "Thanks, which is bigger?"},
]

# Tool result converted to user message (no tool role)
HERMES_TOOL_RESULT_AS_USER = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What files are in /tmp?"},
    {"role": "assistant", "content": "Let me check.", "tool_calls": [
        {"id": "call_abc", "type": "function",
         "function": {"name": "list_dir", "arguments": "{\"path\": \"/tmp\"}"}}
    ]},
    {"role": "user", "content": "Tool result (list_dir): file1.txt\nfile2.txt\n\nThanks, which is bigger?"},
]


def run(url: str, model: str):
    print(f"\nTarget: {url}")
    print(f"Model:  {model}")
    print("=" * 70)

    cases = [
        # ------------------------------------------------------------------ #
        # Basic sanity
        # ------------------------------------------------------------------ #
        ("1. Minimal user message", [
            {"role": "user", "content": "Say hi"},
        ]),
        ("2. System + user", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi"},
        ]),
        ("3. Multi-turn no tools (valid alternation)", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]),

        # ------------------------------------------------------------------ #
        # Consecutive same-role (strict-alternation violation)
        # ------------------------------------------------------------------ #
        ("4. Consecutive user messages", [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ]),
        ("5. Consecutive assistant messages", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Part one"},
            {"role": "assistant", "content": "Part two"},
        ]),

        # ------------------------------------------------------------------ #
        # content=null variants
        # ------------------------------------------------------------------ #
        ("6. assistant content=null (no tool_calls)", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Still there?"},
        ]),
        ("7. assistant content=null WITH tool_calls", [
            {"role": "user", "content": "Call a tool"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "my_tool", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
            {"role": "user", "content": "Thanks"},
        ]),
        ("8. assistant content='' WITH tool_calls", [
            {"role": "user", "content": "Call a tool"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "my_tool", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
            {"role": "user", "content": "Thanks"},
        ]),

        # ------------------------------------------------------------------ #
        # tool role
        # ------------------------------------------------------------------ #
        ("9. tool role message only", [
            {"role": "user", "content": "Use tool"},
            {"role": "tool", "tool_call_id": "t1", "content": "42"},
            {"role": "user", "content": "Got it?"},
        ]),
        ("10. Full tool round-trip (args as string)", [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "List /tmp"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "list_dir", "arguments": "{\"path\": \"/tmp\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "file1.txt"},
            {"role": "user", "content": "Which is bigger?"},
        ]),
        ("11. Full tool round-trip (args as OBJECT - Ollama format)", [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "List /tmp"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "list_dir", "arguments": {"path": "/tmp"}}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "file1.txt"},
            {"role": "user", "content": "Which is bigger?"},
        ]),

        # ------------------------------------------------------------------ #
        # What the proxy currently sends (as-is, no sanitization for LM Studio)
        # ------------------------------------------------------------------ #
        ("12. PROXY CURRENT: Hermes tool-call (args object, content null)", HERMES_TOOL_CALL_MESSAGES),

        # ------------------------------------------------------------------ #
        # After _merge_consecutive_messages only (current fix)
        # ------------------------------------------------------------------ #
        ("13. PROXY+merge: same but merged", _merge_consecutive(HERMES_TOOL_CALL_MESSAGES)),

        # ------------------------------------------------------------------ #
        # Potential fixes to test
        # ------------------------------------------------------------------ #
        ("14. FIX candidate: args string, content null, tool role", HERMES_TOOL_CALL_MESSAGES_ARGS_STRING),
        ("15. FIX candidate: args string, content '', tool role", HERMES_TOOL_CALL_MESSAGES_EMPTY_CONTENT),
        ("16. FIX candidate: tool result as user message", HERMES_TOOL_RESULT_AS_USER),
    ]

    results = []
    for name, messages in cases:
        body = {"model": model, "stream": False, "max_tokens": 8, "messages": messages}
        status, resp = post(url, body)
        ok = status == 200
        results.append((ok, name, status, resp))

        icon = "✓" if ok else "✗"
        print(f"\n[{icon}] {name}")
        print(f"    status={status}")
        if not ok:
            try:
                parsed = json.loads(resp)
                print(f"    error: {json.dumps(parsed)[:300]}")
            except Exception:
                print(f"    raw:   {resp[:300]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ok, name, status, _ in results:
        icon = "✓ OK  " if ok else "✗ FAIL"
        print(f"  [{icon}] {name}")

    failures = [name for ok, name, _, _ in results if not ok]
    if failures:
        print(f"\n{len(failures)} failing case(s). Look for patterns in what they share.")
    else:
        print("\nAll passed. The issue may be in the request structure the proxy builds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--model", required=True, help="Exact model name in LM Studio")
    args = parser.parse_args()
    run(f"http://{args.host}:{args.port}/v1/chat/completions", args.model)
