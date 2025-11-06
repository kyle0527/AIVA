

import asyncio
from dataclasses import dataclass

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class QueuedTask:
    task: FunctionTaskPayload
    trace_id: str


class SqliTaskQueue:
    """Simple wrapper around :class:`asyncio.Queue` for SQLi tasks."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueuedTask | None] = asyncio.Queue()
        self._closed = False

    async def put(self, task: FunctionTaskPayload, *, trace_id: str) -> None:
        if self._closed:
            raise RuntimeError("Queue is closed")
        await self._queue.put(QueuedTask(task=task, trace_id=trace_id))

    async def get(self) -> QueuedTask | None:
        item = await self._queue.get()
        self._queue.task_done()
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)
