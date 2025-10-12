"""
重構的 SQLi Worker - 實現依賴注入和責任分離
這是原始 worker.py 的重構版本，解決複雜度過高的問題
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    FindingPayload,
    FunctionTaskPayload,
)
from services.aiva_common.utils import get_logger, new_id

from .detection_models import DetectionResult
from .engines import (
    BooleanDetectionEngine,
    ErrorDetectionEngine,
    OOBDetectionEngine,
    TimeDetectionEngine,
    UnionDetectionEngine,
)
from .result_binder_publisher import SqliResultBinderPublisher
from .task_queue import QueuedTask, SqliTaskQueue
from .telemetry import SqliExecutionTelemetry

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 20.0


# 定義檢測引擎接口，實現策略模式
class DetectionEngineProtocol(Protocol):
    """檢測引擎協議，所有具體檢測引擎都應實現此接口"""

    async def detect(
        self, task: FunctionTaskPayload, client: httpx.AsyncClient
    ) -> list[DetectionResult]:
        """執行漏洞檢測"""
        ...


@dataclass
class SqliEngineConfig:
    """SQLi 引擎配置"""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 3
    enable_error_detection: bool = True
    enable_boolean_detection: bool = True
    enable_time_detection: bool = True
    enable_union_detection: bool = True
    enable_oob_detection: bool = True


@dataclass
class SqliContext:
    """檢測上下文，包含共享的檢測狀態"""

    task: FunctionTaskPayload
    config: SqliEngineConfig
    telemetry: SqliExecutionTelemetry = field(
        default_factory=lambda: SqliExecutionTelemetry()
    )
    findings: list[FindingPayload] = field(default_factory=list)


class SqliOrchestrator:
    """SQLi 檢測協調器 - 負責協調多個檢測引擎"""

    def __init__(self, config: SqliEngineConfig | None = None):
        self.config = config or SqliEngineConfig()
        self._engines: dict[str, DetectionEngineProtocol] = {}
        self._setup_default_engines()

    def register_engine(self, name: str, engine: DetectionEngineProtocol) -> None:
        """註冊檢測引擎"""
        self._engines[name] = engine
        logger.debug(f"Registered detection engine: {name}")

    def unregister_engine(self, name: str) -> None:
        """取消註冊檢測引擎"""
        if name in self._engines:
            del self._engines[name]
            logger.debug(f"Unregistered detection engine: {name}")

    def _setup_default_engines(self) -> None:
        """設置預設檢測引擎"""
        # 錯誤檢測引擎
        if self.config.enable_error_detection:
            self.register_engine("error", ErrorDetectionEngine())

        # 布林檢測引擎
        if self.config.enable_boolean_detection:
            self.register_engine("boolean", BooleanDetectionEngine())

        # 時間檢測引擎
        if self.config.enable_time_detection:
            self.register_engine("time", TimeDetectionEngine())

        # 聯合檢測引擎
        if self.config.enable_union_detection:
            self.register_engine("union", UnionDetectionEngine())

        # OOB檢測引擎
        if self.config.enable_oob_detection:
            self.register_engine("oob", OOBDetectionEngine())

    async def execute_detection(
        self, context: SqliContext, client: httpx.AsyncClient
    ) -> SqliContext:
        """執行所有已註冊的檢測引擎"""

        for engine_name, engine in self._engines.items():
            try:
                logger.debug(f"Executing engine: {engine_name}")
                results = await engine.detect(context.task, client)

                # 處理檢測結果
                for result in results:
                    if result.is_vulnerable:
                        finding = await self._build_finding(result, context.task)
                        context.findings.append(finding)

                context.telemetry.add_engine(engine_name)

            except Exception as e:
                error_msg = f"Engine {engine_name} failed: {str(e)}"
                logger.exception(error_msg)
                context.telemetry.add_error(error_msg)
                # 繼續執行其他引擎，不因單個引擎失敗而停止

        return context

    async def _build_finding(
        self, result: DetectionResult, task: FunctionTaskPayload
    ) -> FindingPayload:
        """構建漏洞發現"""
        # 這裡可以進一步重構為獨立的 FindingBuilder 類
        return FindingPayload(
            finding_id=new_id("finding"),
            task_id=task.task_id,
            scan_id=task.scan_id,
            status="VULNERABILITY_FOUND",
            vulnerability=result.vulnerability,
            evidence=result.evidence,
            impact=result.impact,
            recommendation=result.recommendation,
            target=result.target,
            strategy=task.strategy,
        )


class SqliWorkerService:
    """重構後的 SQLi Worker 服務"""

    def __init__(
        self,
        orchestrator: SqliOrchestrator | None = None,
        publisher: SqliResultBinderPublisher | None = None,
        config: SqliEngineConfig | None = None,
    ):
        self.orchestrator = orchestrator or SqliOrchestrator(config)
        self.publisher = publisher
        self.config = config or SqliEngineConfig()

    async def process_task(
        self, task: FunctionTaskPayload, http_client: httpx.AsyncClient | None = None
    ) -> SqliContext:
        """處理單個檢測任務"""

        timeout = task.test_config.timeout or self.config.timeout_seconds
        context = SqliContext(task=task, config=self.config)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                context = await self.orchestrator.execute_detection(context, client)
        else:
            context = await self.orchestrator.execute_detection(context, http_client)

        return context


# 為了向後兼容，保留原始接口
async def run() -> None:
    """主要運行函數 - 保持向後兼容"""
    broker = await get_broker()
    publisher = SqliResultBinderPublisher(broker)
    queue = SqliTaskQueue()

    # 使用重構後的服務
    service = SqliWorkerService(publisher=publisher)

    consumer = asyncio.create_task(_consume_queue(queue, service, publisher))

    try:
        async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_SQLI):
            msg = AivaMessage.model_validate_json(mqmsg.body)
            task = FunctionTaskPayload(**msg.payload)
            trace_id = msg.header.trace_id
            await queue.put(task, trace_id=trace_id)
    finally:
        await queue.close()
        await consumer


async def _consume_queue(
    queue: SqliTaskQueue,
    service: SqliWorkerService,
    publisher: SqliResultBinderPublisher,
) -> None:
    """消費任務佇列"""
    while True:
        queued: QueuedTask | None = await queue.get()
        if queued is None:
            return
        await _execute_task(queued, service, publisher)


async def _execute_task(
    queued: QueuedTask, service: SqliWorkerService, publisher: SqliResultBinderPublisher
) -> None:
    """執行單個任務"""
    task = queued.task
    trace_id = queued.trace_id

    await publisher.publish_status(task, "IN_PROGRESS", trace_id=trace_id)

    try:
        context = await service.process_task(task)

        # 發布結果
        for finding in context.findings:
            await publisher.publish_finding(finding, trace_id=trace_id)

        await publisher.publish_status(
            task,
            "COMPLETED",
            trace_id=trace_id,
            details=context.telemetry.to_details(len(context.findings)),
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error while processing SQLi task",
            extra={"task_id": task.task_id},
        )
        await publisher.publish_error(task, exc, trace_id=trace_id)


# 向後兼容的 process_task 函數
async def process_task(
    task: FunctionTaskPayload,
    *,
    http_client: httpx.AsyncClient | None = None,
    fingerprinter=None,  # 保持向後兼容，但不再使用
) -> dict:
    """向後兼容的 process_task 函數"""
    service = SqliWorkerService()
    context = await service.process_task(task, http_client)

    # 返回與原始格式兼容的結果
    return {"findings": context.findings, "telemetry": context.telemetry}


# 向後兼容的別名
SqliDetectionContext = SqliContext
SqliDetectionOrchestrator = SqliOrchestrator
