"""Request/response statistics collector with persistence."""

import json
import time
from pathlib import Path
from collections import defaultdict

STATS_PATH = Path(__file__).parent.parent / "stats.json"


class StatsCollector:
    def __init__(self):
        self.total_requests = 0
        self.total_streaming = 0
        self.total_non_streaming = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tool_calls = 0
        self.tool_usage: dict[str, int] = defaultdict(int)  # tool_name → count
        self.backend_requests: dict[str, int] = defaultdict(int)  # backend_name → count
        self.backend_errors: dict[str, int] = defaultdict(int)
        self.requests_by_endpoint: dict[str, int] = defaultdict(int)  # endpoint → count
        self.requests_coalesced = 0
        self.requests_cached = 0
        self.requests_aborted = 0
        self.started_at = time.time()
        self._load()

    def _load(self):
        if STATS_PATH.exists():
            try:
                raw = json.loads(STATS_PATH.read_text())
                self.total_requests = raw.get("total_requests", 0)
                self.total_streaming = raw.get("total_streaming", 0)
                self.total_non_streaming = raw.get("total_non_streaming", 0)
                self.total_prompt_tokens = raw.get("total_prompt_tokens", 0)
                self.total_completion_tokens = raw.get("total_completion_tokens", 0)
                self.total_tool_calls = raw.get("total_tool_calls", 0)
                self.tool_usage = defaultdict(int, raw.get("tool_usage", {}))
                self.backend_requests = defaultdict(int, raw.get("backend_requests", {}))
                self.backend_errors = defaultdict(int, raw.get("backend_errors", {}))
                self.requests_by_endpoint = defaultdict(int, raw.get("requests_by_endpoint", {}))
                self.requests_coalesced = raw.get("requests_coalesced", 0)
                self.requests_cached = raw.get("requests_cached", 0)
                self.requests_aborted = raw.get("requests_aborted", 0)
                self.started_at = raw.get("started_at", self.started_at)
            except (json.JSONDecodeError, KeyError):
                pass

    def save(self):
        STATS_PATH.write_text(json.dumps(self.to_dict(), indent=2))

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_streaming": self.total_streaming,
            "total_non_streaming": self.total_non_streaming,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_tool_calls": self.total_tool_calls,
            "tool_usage": dict(sorted(self.tool_usage.items(), key=lambda x: -x[1])),
            "backend_requests": dict(self.backend_requests),
            "backend_errors": dict(self.backend_errors),
            "requests_by_endpoint": dict(self.requests_by_endpoint),
            "requests_coalesced": self.requests_coalesced,
            "requests_cached": self.requests_cached,
            "requests_aborted": self.requests_aborted,
            "started_at": self.started_at,
            "uptime_seconds": time.time() - self.started_at,
        }

    def record_request(self, endpoint: str, streaming: bool):
        self.total_requests += 1
        self.requests_by_endpoint[endpoint] += 1
        if streaming:
            self.total_streaming += 1
        else:
            self.total_non_streaming += 1
        self._autosave()

    def record_backend(self, backend_name: str):
        self.backend_requests[backend_name] += 1

    def record_backend_error(self, backend_name: str):
        self.backend_errors[backend_name] += 1

    def record_tokens(self, prompt: int, completion: int):
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self._autosave()

    def record_tool_calls(self, tool_names: list[str]):
        for name in tool_names:
            self.tool_usage[name] += 1
            self.total_tool_calls += 1
        self._autosave()

    def record_coalesced(self):
        self.requests_coalesced += 1

    def record_cached(self):
        self.requests_cached += 1

    def record_aborted(self, count: int):
        self.requests_aborted += count

    def reset(self):
        self.__init__()
        self.started_at = time.time()
        self.save()

    def _autosave(self):
        # Save every 10 requests
        if self.total_requests % 10 == 0:
            self.save()


stats = StatsCollector()
