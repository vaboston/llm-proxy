"""Load balancer for distributing requests across backends.

Strategies:
- priority: first available backend by list order (original behavior)
- round-robin: rotate between available backends
- least-connections: pick the backend with fewest active requests
- weighted-round-robin: rotate with weights
"""

import asyncio
import time
from dataclasses import dataclass, field

from app.config import BackendConfig, get_config
from app.logs import log_manager


@dataclass
class BackendState:
    """Live state for a single backend."""
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    total_latency: float = 0.0
    last_used: float = 0.0
    last_error: float = 0.0
    healthy: bool = True

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_latency / self.total_requests) * 1000

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    def to_dict(self) -> dict:
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_rate": round(self.error_rate, 4),
            "healthy": self.healthy,
        }


class RequestTracker:
    """Context manager to track an active request on a backend."""

    def __init__(self, balancer: "LoadBalancer", backend_id: str):
        self.balancer = balancer
        self.backend_id = backend_id
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.monotonic()
        state = self.balancer.get_state(self.backend_id)
        state.active_requests += 1
        state.total_requests += 1
        state.last_used = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        state = self.balancer.get_state(self.backend_id)
        state.active_requests = max(0, state.active_requests - 1)
        elapsed = time.monotonic() - self.start_time
        state.total_latency += elapsed
        if exc_type is not None:
            state.total_errors += 1
            state.last_error = time.time()
        return False


class LoadBalancer:
    """Selects backends according to the configured strategy."""

    def __init__(self):
        self._states: dict[str, BackendState] = {}
        self._rr_index: int = 0
        self._lock = asyncio.Lock()

    def get_state(self, backend_id: str) -> BackendState:
        if backend_id not in self._states:
            self._states[backend_id] = BackendState()
        return self._states[backend_id]

    def get_all_states(self) -> dict[str, dict]:
        """Return all backend states as a serializable dict."""
        return {bid: state.to_dict() for bid, state in self._states.items()}

    def track(self, backend_id: str) -> RequestTracker:
        """Return a context manager to track an active request."""
        return RequestTracker(self, backend_id)

    def mark_healthy(self, backend_id: str, healthy: bool):
        self.get_state(backend_id).healthy = healthy

    def mark_error(self, backend_id: str):
        state = self.get_state(backend_id)
        state.total_errors += 1
        state.last_error = time.time()

    async def select_backend(self, available: list[BackendConfig]) -> BackendConfig | None:
        """Select a backend from the available list using the configured strategy."""
        if not available:
            return None

        cfg = get_config()
        strategy = cfg.load_balancing_strategy

        if strategy == "round-robin":
            return await self._round_robin(available)
        elif strategy == "least-connections":
            return self._least_connections(available)
        elif strategy == "weighted-round-robin":
            return await self._weighted_round_robin(available)
        else:
            # priority (default): first available
            return available[0]

    async def _round_robin(self, available: list[BackendConfig]) -> BackendConfig:
        async with self._lock:
            self._rr_index = self._rr_index % len(available)
            backend = available[self._rr_index]
            self._rr_index += 1
            return backend

    def _least_connections(self, available: list[BackendConfig]) -> BackendConfig:
        def sort_key(b: BackendConfig):
            state = self.get_state(b.id)
            max_conc = b.max_concurrent or 999
            # Prefer backends under their limit, then fewest active
            at_limit = 1 if state.active_requests >= max_conc else 0
            return (at_limit, state.active_requests)

        return min(available, key=sort_key)

    async def _weighted_round_robin(self, available: list[BackendConfig]) -> BackendConfig:
        # Build weighted list
        weighted = []
        for b in available:
            weight = b.weight or 1
            weighted.extend([b] * weight)
        if not weighted:
            return available[0]
        async with self._lock:
            self._rr_index = self._rr_index % len(weighted)
            backend = weighted[self._rr_index]
            self._rr_index += 1
            return backend

    def reset(self):
        self._states.clear()
        self._rr_index = 0


balancer = LoadBalancer()
