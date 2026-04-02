"""Real-time log manager with SSE broadcast."""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, asdict, field


@dataclass
class LogEntry:
    timestamp: float
    event_type: str  # request | response | error | info | thinking
    summary: str
    raw_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class LogManager:
    def __init__(self, max_entries: int = 500):
        self.entries: deque[LogEntry] = deque(maxlen=max_entries)
        self._subscribers: list[asyncio.Queue] = []

    def log(self, event_type: str, summary: str, raw_data: dict | None = None):
        entry = LogEntry(
            timestamp=time.time(),
            event_type=event_type,
            summary=summary,
            raw_data=raw_data or {},
        )
        self.entries.append(entry)
        data = json.dumps(entry.to_dict())
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def get_recent(self, limit: int = 50) -> list[dict]:
        entries = list(self.entries)[-limit:]
        return [e.to_dict() for e in entries]


log_manager = LogManager()
