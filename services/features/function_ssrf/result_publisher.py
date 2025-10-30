

import json
import os
from typing import Any

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import AbstractBroker
from services.aiva_common.schemas import (
    AivaMessage,
    FindingPayload,
    FunctionTaskPayload,
    MessageHeader,
    TaskUpdatePayload,
)
from services.aiva_common.utils import new_id


class SsrfResultPublisher:
    """Publish status updates and findings for the SSRF worker."""

    def __init__(
        self,
        broker: AbstractBroker,
        *,
        worker_id: str | None = None,
    ) -> None:
        self._broker = broker
        self._worker_id = (
            worker_id or os.getenv("SSRF_WORKER_ID") or new_id("ssrf-worker")
        )

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def publish_status(
        self,
        task: FunctionTaskPayload,
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
        message = AivaMessage(
            header=MessageHeader(
                message_id=new_id("msg"),
                trace_id=trace_id,
                correlation_id=task.scan_id,
                source_module=ModuleName.FUNC_SSRF,
            ),
            topic=Topic.STATUS_TASK_UPDATE,
            payload=payload.model_dump(),
        )
        await self._publish(Topic.STATUS_TASK_UPDATE, message)

    async def publish_finding(
        self,
        finding: FindingPayload,
        *,
        trace_id: str,
    ) -> None:
        message = AivaMessage(
            header=MessageHeader(
                message_id=new_id("msg"),
                trace_id=trace_id,
                correlation_id=finding.scan_id,
                source_module=ModuleName.FUNC_SSRF,
            ),
            topic=Topic.LOG_RESULTS_ALL,
            payload=finding.model_dump(),
        )
        await self._publish(Topic.LOG_RESULTS_ALL, message)

    async def publish_error(
        self,
        task: FunctionTaskPayload,
        error: Exception,
        *,
        trace_id: str,
    ) -> None:
        await self.publish_status(
            task,
            "FAILED",
            trace_id=trace_id,
            details={"error": repr(error)},
        )

    async def _publish(self, topic: Topic, message: AivaMessage) -> None:
        payload = message.model_dump(mode="json")
        await self._broker.publish(topic, json.dumps(payload).encode("utf-8"))
