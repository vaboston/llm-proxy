"""Stateful filter to strip <think>...</think> blocks from streaming text.

Used to remove reasoning tokens from models like Qwen3, QwQ, DeepSeek-R1
before forwarding the response to the client.
"""

import re


class ThinkingFilter:
    """Streaming filter that removes <think>...</think> blocks mid-stream."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self.in_thinking = False
        self.buffer = ""

    def feed(self, text: str) -> str:
        """Process a chunk of text. Returns the filtered output."""
        self.buffer += text
        output = ""

        while self.buffer:
            if self.in_thinking:
                idx = self.buffer.find(self.CLOSE_TAG)
                if idx != -1:
                    self.buffer = self.buffer[idx + len(self.CLOSE_TAG):]
                    self.in_thinking = False
                    continue
                else:
                    keep = len(self.CLOSE_TAG) - 1
                    self.buffer = self.buffer[-keep:] if len(self.buffer) > keep else self.buffer
                    break
            else:
                idx = self.buffer.find(self.OPEN_TAG)
                if idx != -1:
                    output += self.buffer[:idx]
                    self.buffer = self.buffer[idx + len(self.OPEN_TAG):]
                    self.in_thinking = True
                    continue
                else:
                    safe = len(self.buffer)
                    for i in range(min(len(self.OPEN_TAG) - 1, len(self.buffer)), 0, -1):
                        if self.OPEN_TAG.startswith(self.buffer[-i:]):
                            safe = len(self.buffer) - i
                            break
                    output += self.buffer[:safe]
                    self.buffer = self.buffer[safe:]
                    break

        return output

    def flush(self) -> str:
        """Flush remaining buffer. Call at end of stream."""
        if self.in_thinking:
            self.buffer = ""
            return ""
        remaining = self.buffer
        self.buffer = ""
        return remaining


def strip_thinking(text: str) -> str:
    """Remove all <think>...</think> blocks from a complete text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
