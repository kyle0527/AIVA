from __future__ import annotations


class UrlQueueManager:
    """FIFO queue for discovered URLs (placeholder)."""

    def __init__(self, seeds: list[str]) -> None:
        self._q = list(seeds)

    def has_next(self) -> bool:
        return bool(self._q)

    def next(self) -> str:
        return self._q.pop(0)
