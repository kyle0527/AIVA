from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
import heapq
import itertools
import time

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class QueuedTask:
    """Represents a task stored within the local worker queue."""

    task: FunctionTaskPayload
    trace_id: str
    enqueued_at: float


@dataclass(order=True)
class _QueueEntry:
    ready_at: float
    priority: int
    sequence: int
    payload: QueuedTask = field(compare=False)
    valid: bool = field(default=True, compare=False)


class XssTaskQueue:
    """Async priority queue that honours per-task scheduling hints.

    The worker can receive bursts of tasks from the MQ.  This queue keeps track
    of their desired execution priority as well as optional "retry after"
    delays.  It exposes an async ``get`` API so the worker can process tasks in
    order without blocking the broker subscription loop.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self._condition = asyncio.Condition()
        self._counter = itertools.count()
        self._heap: list[_QueueEntry] = []
        self._entries: dict[str, _QueueEntry] = {}
        self._closed = False

    async def put(
        self,
        task: FunctionTaskPayload,
        *,
        trace_id: str,
        delay: float = 0.0,
    ) -> None:
        """Insert a task into the queue, replacing any previous entry."""

        ready_at = self._clock() + max(0.0, delay)
        payload = QueuedTask(
            task=task,
            trace_id=trace_id,
            enqueued_at=self._clock(),
        )

        async with self._condition:
            if self._closed:
                raise RuntimeError("Queue has been closed")

            entry = _QueueEntry(
                ready_at=ready_at,
                priority=task.priority,
                sequence=next(self._counter),
                payload=payload,
            )

            existing = self._entries.get(task.task_id)
            if existing is not None:
                existing.valid = False

            self._entries[task.task_id] = entry
            heapq.heappush(self._heap, entry)
            self._condition.notify_all()

    async def get(self) -> QueuedTask | None:
        """Return the next ready task, waiting for availability if needed."""

        async with self._condition:
            while True:
                self._discard_invalid_locked()

                if not self._heap:
                    if self._closed:
                        return None
                    await self._condition.wait()
                    continue

                now = self._clock()

                first_entry = self._heap[0]
                if first_entry.ready_at > now:
                    timeout = first_entry.ready_at - now
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                    continue

                ready_entries: list[_QueueEntry] = []
                while self._heap and self._heap[0].ready_at <= now:
                    entry = heapq.heappop(self._heap)
                    if entry.valid:
                        ready_entries.append(entry)

                if not ready_entries:
                    continue

                chosen = min(
                    ready_entries, key=lambda item: (item.priority, item.sequence)
                )
                for entry in ready_entries:
                    if entry is chosen:
                        continue
                    heapq.heappush(self._heap, entry)

                self._entries.pop(chosen.payload.task.task_id, None)
                chosen.valid = False
                return chosen.payload

    async def close(self) -> None:
        """Signal that no additional tasks will be queued."""

        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return sum(1 for entry in self._entries.values() if entry.valid)

    def _discard_invalid_locked(self) -> None:
        while self._heap and not self._heap[0].valid:
            heapq.heappop(self._heap)
