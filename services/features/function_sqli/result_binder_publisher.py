

import json
from typing import Any
import uuid

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import AbstractBroker
from services.aiva_common.schemas import (
    AivaMessage,
    FindingPayload,
    MessageHeader,
    TaskUpdatePayload,
)
from services.aiva_common.utils import new_id


class SqliResultBinderPublisher:
    """Publish SQLi findings and task status updates to the platform bus."""

    def __init__(self, broker: AbstractBroker, *, worker_id: str | None = None) -> None:
        self._broker = broker
        self._worker_id = worker_id or f"sqli-worker-{uuid.uuid4().hex[:8]}"

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def publish_status(
        self,
        task,
        status: str,
        *,
        trace_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = TaskUpdatePayload(
            task_id=task.task_id,
            scan_id=task.scan_id,
            status=status,
            worker_id=self._worker_id,
            details=details,
        )
        await self._publish(Topic.STATUS_TASK_UPDATE, payload, trace_id=trace_id)

    async def publish_error(self, task, error: Exception, *, trace_id: str) -> None:
        details = {"error": type(error).__name__, "message": str(error)}
        await self.publish_status(
            task,
            "FAILED",
            trace_id=trace_id,
            details=details,
        )

    async def publish_finding(self, finding: FindingPayload, *, trace_id: str) -> None:
        await self._publish(Topic.LOG_RESULTS_ALL, finding, trace_id=trace_id)

    async def _publish(self, topic: Topic, payload, *, trace_id: str) -> None:
        message = AivaMessage(
            header=MessageHeader(
                message_id=new_id("msg"),
                trace_id=trace_id,
                correlation_id=getattr(payload, "scan_id", None),
                source_module=ModuleName.FUNC_SQLI,
            ),
            topic=topic,
            payload=payload.model_dump(),
        )
        await self._broker.publish(
            topic,
            json.dumps(message.model_dump()).encode("utf-8"),
        )
