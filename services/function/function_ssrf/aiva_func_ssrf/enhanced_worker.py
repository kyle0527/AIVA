"""
Enhanced SSRF Worker - 增強版 SSRF 工作器
整合智能檢測器，提供自適應超時和性能優化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    FindingPayload,
    FunctionTaskPayload,
)
from services.aiva_common.utils import get_logger
from services.function.common.detection_config import SSRFConfig

from .internal_address_detector import InternalAddressDetector
from .oast_dispatcher import OastDispatcher
from .param_semantics_analyzer import ParamSemanticsAnalyzer
from .result_publisher import SsrfResultPublisher
from .smart_ssrf_detector import SmartSSRFDetector

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0


@dataclass
class EnhancedSsrfTelemetry:
    """增強版 SSRF 遙測數據"""

    attempts: int = 0
    findings: int = 0
    oast_callbacks: int = 0
    errors: list[str] = field(default_factory=list)

    # 智能檢測指標
    adaptive_timeout_used: bool = False
    early_stopping_triggered: bool = False
    rate_limiting_applied: bool = False
    session_duration: float = 0.0
    protection_detected: bool = False

    def to_details(self) -> dict[str, Any]:
        """轉換為詳細信息字典"""
        details: dict[str, Any] = {
            "attempts": self.attempts,
            "findings": self.findings,
            "oast_callbacks": self.oast_callbacks,
            "smart_detection": {
                "adaptive_timeout_used": self.adaptive_timeout_used,
                "early_stopping_triggered": self.early_stopping_triggered,
                "rate_limiting_applied": self.rate_limiting_applied,
                "session_duration": self.session_duration,
                "protection_detected": self.protection_detected,
            },
        }
        if self.errors:
            details["errors"] = self.errors
        return details


@dataclass
class EnhancedTaskExecutionResult:
    """增強版任務執行結果"""

    findings: list[FindingPayload]
    telemetry: EnhancedSsrfTelemetry


class EnhancedSSRFWorker:
    """增強版 SSRF 工作器"""

    def __init__(self, config: SSRFConfig | None = None) -> None:
        """
        初始化增強版 SSRF 工作器

        Args:
            config: SSRF 檢測配置
        """
        self.config = config or SSRFConfig()
        self.smart_detector = SmartSSRFDetector(self.config)

        logger.info(
            "Enhanced SSRF Worker initialized",
            extra={
                "smart_detection_enabled": True,
                "config": {
                    "max_vulnerabilities": self.config.max_vulnerabilities,
                    "timeout_base": self.config.timeout_base,
                    "timeout_max": self.config.timeout_max,
                    "requests_per_second": self.config.requests_per_second,
                },
            },
        )

    async def run(self) -> None:
        """運行增強版 SSRF 工作器"""
        broker = await get_broker()
        publisher = SsrfResultPublisher(broker)
        analyzer = ParamSemanticsAnalyzer()
        detector = InternalAddressDetector()
        dispatcher = OastDispatcher()

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=DEFAULT_TIMEOUT_SECONDS
        ) as client:
            try:
                async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_SSRF):
                    msg = AivaMessage.model_validate_json(mqmsg.body)
                    task = FunctionTaskPayload(**msg.payload)
                    trace_id = msg.header.trace_id

                    await self._execute_task(
                        task,
                        trace_id=trace_id,
                        client=client,
                        publisher=publisher,
                        analyzer=analyzer,
                        detector=detector,
                        dispatcher=dispatcher,
                    )
            finally:  # pragma: no cover - defensive guard for shutdown
                await dispatcher.close()

    async def _execute_task(
        self,
        task: FunctionTaskPayload,
        *,
        trace_id: str,
        client: httpx.AsyncClient,
        publisher: SsrfResultPublisher,
        analyzer: ParamSemanticsAnalyzer,
        detector: InternalAddressDetector,
        dispatcher: OastDispatcher,
    ) -> None:
        """執行任務"""
        await publisher.publish_status(task, "IN_PROGRESS", trace_id=trace_id)

        try:
            result = await self.process_task(
                task,
                client=client,
                analyzer=analyzer,
                detector=detector,
                dispatcher=dispatcher,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception(
                "Unhandled error while processing enhanced SSRF task",
                extra={"task_id": task.task_id},
            )
            await publisher.publish_error(task, exc, trace_id=trace_id)
            return

        for finding in result.findings:
            await publisher.publish_finding(finding, trace_id=trace_id)

        await publisher.publish_status(
            task,
            "COMPLETED",
            trace_id=trace_id,
            details=result.telemetry.to_details(),
        )

    async def process_task(
        self,
        task: FunctionTaskPayload,
        *,
        client: httpx.AsyncClient,
        analyzer: ParamSemanticsAnalyzer,
        detector: InternalAddressDetector,
        dispatcher: OastDispatcher,
    ) -> EnhancedTaskExecutionResult:
        """
        處理任務（使用智能檢測器）

        Args:
            task: 功能任務載荷
            client: HTTP 客戶端
            analyzer: 參數語義分析器
            detector: 內部地址檢測器
            dispatcher: OAST 調度器

        Returns:
            增強版任務執行結果
        """
        logger.info(
            "Processing SSRF task with smart detection",
            extra={"task_id": task.task_id},
        )

        # 使用智能檢測器執行檢測
        (
            findings_data,
            detection_metrics,
        ) = await self.smart_detector.detect_vulnerabilities(
            task,
            client=client,
            analyzer=analyzer,
            detector=detector,
            dispatcher=dispatcher,
        )

        # 轉換為 FindingPayload 對象
        findings = self._convert_to_finding_payloads(findings_data, task)

        # 創建增強版遙測數據
        telemetry = EnhancedSsrfTelemetry(
            attempts=detection_metrics.total_requests,
            findings=len(findings),
            oast_callbacks=0,  # TODO: 從 findings_data 中提取 OAST 回調數據
            errors=[],  # TODO: 從檢測過程中收集錯誤
            adaptive_timeout_used=detection_metrics.timeout_count > 0,
            early_stopping_triggered=False,  # TODO: 從智能管理器獲取
            rate_limiting_applied=detection_metrics.rate_limited_count > 0,
            session_duration=detection_metrics.total_time,
            protection_detected=detection_metrics.rate_limited_count > 0,
        )

        logger.info(
            "SSRF task completed with smart detection",
            extra={
                "task_id": task.task_id,
                "findings": len(findings),
                "attempts": telemetry.attempts,
                "session_duration": telemetry.session_duration,
                "early_stopping": telemetry.early_stopping_triggered,
            },
        )

        return EnhancedTaskExecutionResult(findings=findings, telemetry=telemetry)

    def _convert_to_finding_payloads(
        self, findings_data: list[dict[str, Any]], task: FunctionTaskPayload
    ) -> list[FindingPayload]:
        """
        轉換檢測結果為 FindingPayload 對象

        Args:
            findings_data: 檢測結果數據列表
            task: 任務載荷，用於提取 task_id 和 scan_id

        Returns:
            FindingPayload 對象列表
        """
        findings = []

        for finding_data in findings_data:
            # 提取必要的欄位來創建 FindingPayload
            finding_payload = FindingPayload(
                finding_id=finding_data["finding_id"],
                task_id=task.task_id,
                scan_id=task.scan_id,
                status="COMPLETED",
                vulnerability=finding_data["vulnerability"],
                target=finding_data["target"],
                strategy="smart_ssrf_detection",
                evidence=finding_data.get("evidence"),
                impact=finding_data.get("impact"),
                recommendation=finding_data.get("recommendation"),
            )
            findings.append(finding_payload)

        return findings


# 兼容性函數，保持向後兼容
async def run() -> None:
    """運行增強版 SSRF 工作器（兼容性入口點）"""
    worker = EnhancedSSRFWorker()
    await worker.run()


async def process_task(
    task: FunctionTaskPayload,
    *,
    client: httpx.AsyncClient,
    analyzer: ParamSemanticsAnalyzer | None = None,
    detector: InternalAddressDetector | None = None,
    dispatcher: OastDispatcher | None = None,
) -> dict[str, Any]:
    """
    處理任務（兼容性函數）

    Args:
        task: 功能任務載荷
        client: HTTP 客戶端
        analyzer: 參數語義分析器
        detector: 內部地址檢測器
        dispatcher: OAST 調度器

    Returns:
        任務執行結果
    """
    # 使用默認實例如果未提供
    analyzer = analyzer or ParamSemanticsAnalyzer()
    detector = detector or InternalAddressDetector()
    dispatcher = dispatcher or OastDispatcher()

    # 創建增強版工作器並處理任務
    worker = EnhancedSSRFWorker()
    result = await worker.process_task(
        task,
        client=client,
        analyzer=analyzer,
        detector=detector,
        dispatcher=dispatcher,
    )

    # 轉換為兼容格式
    return {
        "findings": result.findings,
        "telemetry": result.telemetry,
    }
