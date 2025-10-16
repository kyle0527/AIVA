"""
Enhanced IDOR Worker - 增強版 IDOR 工作器
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
from services.function.common.detection_config import IDORConfig
from services.function.common.worker_statistics import (
    StatisticsCollector,
    StoppingReason,
)

from .cross_user_tester import CrossUserTester
from .resource_id_extractor import ResourceIdExtractor
from .smart_idor_detector import SmartIDORDetector
from .vertical_escalation_tester import VerticalEscalationTester

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0


@dataclass
class EnhancedIdorTelemetry:
    """增強版 IDOR 遙測數據"""

    attempts: int = 0
    findings: int = 0
    horizontal_tests: int = 0
    vertical_tests: int = 0
    id_extraction_attempts: int = 0
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
            "horizontal_tests": self.horizontal_tests,
            "vertical_tests": self.vertical_tests,
            "id_extraction_attempts": self.id_extraction_attempts,
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
class EnhancedIdorTaskExecutionResult:
    """增強版 IDOR 任務執行結果"""

    findings: list[FindingPayload]
    telemetry: EnhancedIdorTelemetry


class EnhancedIDORWorker:
    """增強版 IDOR 工作器 - 維持四大模組架構"""

    def __init__(self, config: IDORConfig | None = None) -> None:
        """
        初始化增強版 IDOR 工作器

        Args:
            config: IDOR 檢測配置
        """
        self.config = config or IDORConfig()
        self.smart_detector = SmartIDORDetector(self.config)

        logger.info(
            "Enhanced IDOR Worker initialized",
            extra={
                "smart_detection_enabled": True,
                "worker_module": "IDOR",
                "config": {
                    "max_vulnerabilities": self.config.max_vulnerabilities,
                    "timeout_base": self.config.timeout_base,
                    "timeout_max": self.config.timeout_max,
                    "requests_per_second": self.config.requests_per_second,
                    "horizontal_enabled": self.config.horizontal_escalation_enabled,
                    "vertical_enabled": self.config.vertical_escalation_enabled,
                },
            },
        )

    async def run(self) -> None:
        """運行增強版 IDOR 工作器"""
        broker = await get_broker()

        # 初始化 IDOR 專用組件
        id_extractor = ResourceIdExtractor()

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=DEFAULT_TIMEOUT_SECONDS
        ) as client:
            # 初始化需要 HTTP 客戶端的測試器
            cross_user_tester = CrossUserTester(client)
            vertical_tester = VerticalEscalationTester(client)

            try:
                async for mqmsg in broker.subscribe(Topic.FUNCTION_IDOR_TASK):
                    msg = AivaMessage.model_validate_json(mqmsg.body)
                    task = FunctionTaskPayload(**msg.payload)
                    trace_id = msg.header.trace_id

                    await self._execute_task(
                        task,
                        trace_id=trace_id,
                        client=client,
                        id_extractor=id_extractor,
                        cross_user_tester=cross_user_tester,
                        vertical_tester=vertical_tester,
                    )
            except Exception as exc:  # pragma: no cover - defensive guard for shutdown
                logger.exception("Enhanced IDOR Worker shutdown error", exc_info=exc)

    async def _execute_task(
        self,
        task: FunctionTaskPayload,
        *,
        trace_id: str,
        client: httpx.AsyncClient,
        id_extractor: ResourceIdExtractor,
        cross_user_tester: CrossUserTester,
        vertical_tester: VerticalEscalationTester,
    ) -> None:
        """執行任務"""
        import json

        from services.aiva_common.enums import ModuleName
        from services.aiva_common.schemas import AivaMessage, MessageHeader
        from services.aiva_common.utils import new_id

        broker = await get_broker()

        # 發布任務開始狀態
        logger.info(f"Starting IDOR task: {task.task_id}")

        try:
            result = await self.process_task(
                task,
                client=client,
                id_extractor=id_extractor,
                cross_user_tester=cross_user_tester,
                vertical_tester=vertical_tester,
            )

            # 發布檢測結果
            for finding in result.findings:
                out = AivaMessage(
                    header=MessageHeader(
                        message_id=new_id("msg"),
                        trace_id=trace_id,
                        correlation_id=task.scan_id,
                        source_module=ModuleName.FUNC_IDOR,
                    ),
                    topic=Topic.FINDING_DETECTED,
                    payload=finding.model_dump(),
                )
                await broker.publish(
                    Topic.FINDING_DETECTED,
                    json.dumps(out.model_dump()).encode("utf-8"),
                )

            logger.info(
                f"IDOR task {task.task_id} completed with "
                f"{len(result.findings)} findings",
                extra={
                    "task_id": task.task_id,
                    "findings": len(result.findings),
                    "attempts": result.telemetry.attempts,
                    "session_duration": result.telemetry.session_duration,
                },
            )

        except Exception:  # pragma: no cover - defensive guard
            logger.exception(
                "Unhandled error while processing enhanced IDOR task",
                extra={"task_id": task.task_id},
            )

    async def process_task(
        self,
        task: FunctionTaskPayload,
        *,
        client: httpx.AsyncClient,
        id_extractor: ResourceIdExtractor,
        cross_user_tester: CrossUserTester,
        vertical_tester: VerticalEscalationTester,
    ) -> EnhancedIdorTaskExecutionResult:
        """
        處理任務（使用智能檢測器）

        Args:
            task: 功能任務載荷
            client: HTTP 客戶端
            id_extractor: 資源 ID 提取器
            cross_user_tester: 跨用戶測試器
            vertical_tester: 垂直權限提升測試器

        Returns:
            增強版 IDOR 任務執行結果
        """
        logger.info(
            "Processing IDOR task with smart detection",
            extra={"task_id": task.task_id, "worker_module": "IDOR"},
        )

        # 創建統計數據收集器
        stats_collector = StatisticsCollector(
            task_id=task.task_id, worker_type="idor"
        )

        # 使用智能檢測器執行檢測
        (
            findings_data,
            detection_metrics,
        ) = await self.smart_detector.detect_vulnerabilities(
            task,
            client=client,
            id_extractor=id_extractor,
            cross_user_tester=cross_user_tester,
            vertical_tester=vertical_tester,
        )

        # 從檢測指標更新統計數據
        stats_collector.stats.total_requests = detection_metrics.total_requests
        stats_collector.stats.successful_requests = (
            detection_metrics.total_requests - detection_metrics.failed_requests
        )
        stats_collector.stats.failed_requests = detection_metrics.failed_requests
        stats_collector.stats.timeout_requests = detection_metrics.timeout_count
        stats_collector.stats.rate_limited_requests = (
            detection_metrics.rate_limited_count
        )

        # 從檢測上下文中提取水平和垂直測試計數
        horizontal_tests = 0
        vertical_tests = 0
        for finding in findings_data:
            vuln = finding.get("vulnerability", {})
            escalation_type = vuln.get("escalation_type", "")
            if "horizontal" in escalation_type.lower():
                horizontal_tests += 1
            elif "vertical" in escalation_type.lower():
                vertical_tests += 1

        # 設置 IDOR 特定統計數據
        stats_collector.set_module_specific("horizontal_tests", horizontal_tests)
        stats_collector.set_module_specific("vertical_tests", vertical_tests)
        stats_collector.set_module_specific(
            "id_extraction_attempts", 1
        )  # 每個任務執行一次

        # 轉換為 FindingPayload 對象
        findings = self._convert_to_finding_payloads(findings_data, task)

        # 設置自適應行為標記
        stats_collector.set_adaptive_behavior(
            adaptive_timeout=detection_metrics.timeout_count > 0,
            rate_limiting=detection_metrics.rate_limited_count > 0,
            protection_detected=detection_metrics.rate_limited_count > 0,
        )

        # 檢查是否觸發 early stopping
        if len(findings) >= self.config.max_vulnerabilities:
            stats_collector.record_early_stopping(
                reason=StoppingReason.MAX_VULNERABILITIES,
                details={
                    "max_allowed": self.config.max_vulnerabilities,
                    "found": len(findings),
                },
            )

        # 完成統計數據收集
        final_stats = stats_collector.finalize()

        # 創建增強版遙測數據（保持向後兼容）
        telemetry = EnhancedIdorTelemetry(
            attempts=final_stats.total_requests,
            findings=len(findings),
            horizontal_tests=horizontal_tests,
            vertical_tests=vertical_tests,
            id_extraction_attempts=1,
            errors=[
                error["message"] for error in final_stats.error_details
            ],  # 簡化錯誤列表
            adaptive_timeout_used=final_stats.adaptive_timeout_used,
            early_stopping_triggered=final_stats.early_stopping_triggered,
            rate_limiting_applied=final_stats.rate_limiting_applied,
            session_duration=final_stats.duration_seconds,
            protection_detected=final_stats.protection_detected,
        )

        logger.info(
            "IDOR task completed with smart detection",
            extra={
                "task_id": task.task_id,
                "worker_module": "IDOR",
                "findings": len(findings),
                "attempts": telemetry.attempts,
                "horizontal_tests": telemetry.horizontal_tests,
                "vertical_tests": telemetry.vertical_tests,
                "session_duration": telemetry.session_duration,
                "early_stopping": telemetry.early_stopping_triggered,
                "statistics_summary": stats_collector.get_summary(),
            },
        )

        return EnhancedIdorTaskExecutionResult(findings=findings, telemetry=telemetry)

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
                strategy="smart_idor_detection",
                evidence=finding_data.get("evidence"),
                impact=finding_data.get("impact"),
                recommendation=finding_data.get("recommendation"),
            )
            findings.append(finding_payload)

        return findings


# 兼容性函數，保持向後兼容
async def run() -> None:
    """運行增強版 IDOR 工作器（兼容性入口點）"""
    worker = EnhancedIDORWorker()
    await worker.run()


async def process_task(
    task: FunctionTaskPayload,
    *,
    client: httpx.AsyncClient,
    id_extractor: ResourceIdExtractor | None = None,
    cross_user_tester: CrossUserTester | None = None,
    vertical_tester: VerticalEscalationTester | None = None,
) -> dict[str, Any]:
    """
    處理任務（兼容性函數）

    Args:
        task: 功能任務載荷
        client: HTTP 客戶端
        id_extractor: 資源 ID 提取器
        cross_user_tester: 跨用戶測試器
        vertical_tester: 垂直權限提升測試器

    Returns:
        任務執行結果
    """
    # 使用默認實例如果未提供
    id_extractor = id_extractor or ResourceIdExtractor()
    cross_user_tester = cross_user_tester or CrossUserTester(client)
    vertical_tester = vertical_tester or VerticalEscalationTester(client)

    # 創建增強版工作器並處理任務
    worker = EnhancedIDORWorker()
    result = await worker.process_task(
        task,
        client=client,
        id_extractor=id_extractor,
        cross_user_tester=cross_user_tester,
        vertical_tester=vertical_tester,
    )

    # 轉換為兼容格式
    return {
        "findings": result.findings,
        "telemetry": result.telemetry,
    }
