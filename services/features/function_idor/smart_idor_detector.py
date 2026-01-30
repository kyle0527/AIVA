"""
Smart IDOR Detector - 智能 IDOR 檢測器
整合統一檢測管理器，提供自適應超時、速率限制和早期停止功能
"""



import asyncio
from dataclasses import dataclass, field
import time
from typing import Any

import httpx

from aiva_common.schemas import FunctionTaskPayload
from aiva_common.utils import get_logger
from aiva_common.detection import (
    UnifiedSmartDetectionManager,
    DetectionPhase,
)
from .config.idor_config import IdorConfig
from .testers.cross_user_tester import CrossUserTester, CrossUserTestResult
from .resource_id_extractor import ResourceId, ResourceIdExtractor
from .testers.vertical_escalation_tester import (
    PrivilegeLevel,
    VerticalEscalationTester,
    VerticalTestResult,
)

logger = get_logger(__name__)


@dataclass
class IDORDetectionContext:
    """IDOR 檢測上下文"""

    task: FunctionTaskPayload
    client: httpx.AsyncClient
    id_extractor: ResourceIdExtractor
    cross_user_tester: CrossUserTester
    vertical_tester: VerticalEscalationTester

    # 智能檢測狀態
    start_time: float = field(default_factory=time.time)
    attempts: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # IDOR 特定統計
    horizontal_tests: int = 0
    vertical_tests: int = 0
    id_extraction_attempts: int = 0

    def add_finding(self, finding: dict[str, Any]) -> None:
        """添加發現的漏洞"""
        self.findings.append(finding)

    def add_error(self, error: str) -> None:
        """添加錯誤"""
        self.errors.append(error)

    def increment_attempts(self) -> None:
        """增加嘗試次數"""
        self.attempts += 1


class SmartIDORDetector:
    """智能 IDOR 檢測器 - 基於統一檢測管理器"""

    def __init__(self, config: IdorConfig | None = None) -> None:
        """
        初始化智能 IDOR 檢測器

        Args:
            config: IDOR 檢測配置，如果未提供則使用默認配置
        """
        self.config = config or IdorConfig()

        logger.info(
            "Smart IDOR Detector initialized",
            extra={
                "max_id_variations": self.config.max_id_variations,
                "request_timeout": self.config.request_timeout,
                "horizontal_enabled": self.config.horizontal_enabled,
                "vertical_enabled": self.config.vertical_enabled,
                "safe_mode": self.config.safe_mode,
            },
        )

    async def detect_vulnerabilities(
        self,
        task: FunctionTaskPayload,
        *,
        client: httpx.AsyncClient,
        id_extractor: ResourceIdExtractor,
        cross_user_tester: CrossUserTester,
        vertical_tester: VerticalEscalationTester,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        執行智能 IDOR 檢測

        Args:
            task: 功能任務載荷
            client: HTTP 客戶端
            id_extractor: 資源 ID 提取器
            cross_user_tester: 跨用戶測試器
            vertical_tester: 垂直權限提升測試器

        Returns:
            檢測到的漏洞列表和檢測指標
        """
        # 創建統一智能檢測管理器
        smart_manager = UnifiedSmartDetectionManager(
            detection_id=task.task_id,
            detection_type="idor",
            target_url=str(task.target.url)
        )
        
        context = IDORDetectionContext(
            task=task,
            client=client,
            id_extractor=id_extractor,
            cross_user_tester=cross_user_tester,
            vertical_tester=vertical_tester,
        )

        async with smart_manager.start_detection():
            try:
                # 1. 提取資源 ID
                smart_manager.metrics.start_phase(DetectionPhase.RECONNAISSANCE)
                resource_ids = await self._extract_resource_ids(context)

                if not resource_ids:
                    logger.debug(
                        "No resource IDs found for IDOR testing",
                        extra={"task_id": task.task_id},
                    )
                    return [], smart_manager.get_metrics().to_dict()

                logger.info(
                    f"Starting IDOR detection with {len(resource_ids)} resource IDs",
                    extra={
                        "task_id": task.task_id,
                        "resource_ids": [rid.value for rid in resource_ids],
                    },
                )

                # 2. 執行水平權限測試（如果啟用）
                if self.config.horizontal_enabled:
                    smart_manager.metrics.start_phase(DetectionPhase.PAYLOAD_INJECTION)
                    await self._execute_horizontal_testing(context, resource_ids, smart_manager)

                # 3. 執行垂直權限測試（如果啟用）
                if self.config.vertical_enabled:
                    smart_manager.metrics.start_phase(DetectionPhase.VALIDATION)
                    await self._execute_vertical_testing(context, resource_ids, smart_manager)

                logger.info(
                    "IDOR detection completed",
                    extra={
                        "task_id": task.task_id,
                        "findings": len(context.findings),
                        "attempts": context.attempts,
                        "horizontal_tests": context.horizontal_tests,
                        "vertical_tests": context.vertical_tests,
                    },
                )

                metrics = smart_manager.get_metrics().to_dict()
                return context.findings, metrics

            except Exception as exc:
                logger.exception(
                    "Error during IDOR detection",
                    extra={"task_id": task.task_id},
                )
                context.add_error(str(exc))
                metrics = smart_manager.get_metrics().to_dict()
                return context.findings, metrics

    def _calculate_total_steps(self) -> int:
        """計算總檢測步驟數"""
        steps = 1  # ID 提取

        if self.config.horizontal_enabled:
            steps += self.config.max_id_variations * 2  # 每個變異有前後兩個測試

        if self.config.vertical_enabled:
            steps += 4  # 基本的權限提升測試

        return steps

    async def _extract_resource_ids(
        self, context: IDORDetectionContext
    ) -> list[ResourceId]:
        """
        提取資源 ID

        Args:
            context: 檢測上下文

        Returns:
            提取到的資源 ID 列表
        """
        try:
            context.id_extraction_attempts += 1

            # 從 URL 提取 ID (異步操作以符合方法簽名)
            target_url = str(context.task.target.url)
            await asyncio.sleep(0)  # 確保方法是真正的異步方法
            resource_ids = context.id_extractor.extract_from_url(target_url)

            # 從查詢參數提取 ID - 使用 URL 提取方法
            if context.task.target.parameter:
                # 將參數附加到 URL 中進行提取
                param_url = f"{target_url}?{context.task.target.parameter}=dummy_value"
                query_ids = context.id_extractor.extract_from_url(param_url)
                resource_ids.extend(query_ids)

            # 從表單數據提取 ID - 使用 URL 提取方法
            if context.task.target.form_data:
                # 將表單數據轉換為查詢字符串進行提取
                form_params = "&".join(
                    [f"{k}={v}" for k, v in context.task.target.form_data.items()]
                )
                form_url = f"{target_url}?{form_params}"
                form_ids = context.id_extractor.extract_from_url(form_url)
                resource_ids.extend(form_ids)

            logger.debug(
                f"Extracted {len(resource_ids)} resource IDs",
                extra={
                    "task_id": context.task.task_id,
                    "ids": [f"{rid.value}({rid.pattern})" for rid in resource_ids],
                },
            )

            return resource_ids

        except Exception as exc:
            context.add_error(f"ID extraction failed: {exc}")
            logger.warning(
                "Failed to extract resource IDs",
                extra={
                    "task_id": context.task.task_id,
                    "error": str(exc),
                },
            )
            return []

    async def _execute_horizontal_testing(
        self,
        context: IDORDetectionContext,
        resource_ids: list[ResourceId],
        smart_manager: UnifiedSmartDetectionManager,
    ) -> None:
        """
        執行水平權限測試

        Args:
            context: 檢測上下文
            resource_ids: 資源 ID 列表
            smart_manager: 智能檢測管理器
        """
        total = len(resource_ids) * self.config.max_id_variations
        current = 0
        
        for resource_id in resource_ids:
            # 基於配置的最大變異數執行測試
            for _ in range(self.config.max_id_variations):
                if smart_manager.should_continue_testing():
                    await self._test_horizontal_access(context, resource_id, smart_manager)
                    context.horizontal_tests += 1
                    current += 1
                    smart_manager.update_progress(current, total)

    async def _execute_vertical_testing(
        self,
        context: IDORDetectionContext,
        resource_ids: list[ResourceId],
        smart_manager: UnifiedSmartDetectionManager,
    ) -> None:
        """
        執行垂直權限測試

        Args:
            context: 檢測上下文
            resource_ids: 資源 ID 列表
            smart_manager: 智能檢測管理器
        """
        # 基本的權限級別測試
        privilege_levels = [PrivilegeLevel.ADMIN, PrivilegeLevel.USER, PrivilegeLevel.GUEST]
        total = len(resource_ids) * len(privilege_levels)
        current = 0
        
        for resource_id in resource_ids:
            # 為每個權限級別執行測試
            for privilege_level in privilege_levels:
                if smart_manager.should_continue_testing():
                    await self._test_vertical_access(context, resource_id, privilege_level, smart_manager)
                    context.vertical_tests += 1
                    current += 1
                    smart_manager.update_progress(current, total)

    async def _test_horizontal_access(
        self,
        context: IDORDetectionContext,
        resource_id: ResourceId,
        smart_manager: UnifiedSmartDetectionManager,
    ) -> None:
        """
        測試水平權限訪問

        Args:
            context: 檢測上下文
            resource_id: 資源 ID
            smart_manager: 智能檢測管理器
        """
        try:
            # 執行跨用戶測試
            test_result = await context.cross_user_tester.test_horizontal_idor(
                url=str(context.task.target.url),
                resource_id=resource_id.value,
                method=context.task.target.method or "GET",
            )

            if isinstance(test_result, CrossUserTestResult) and test_result.vulnerable:
                finding = self._build_horizontal_finding(
                    context.task,
                    resource_id,
                    test_result,
                )
                context.add_finding(finding)
                smart_manager.report_vulnerability_found()

                logger.info(
                    "Horizontal IDOR vulnerability detected",
                    extra={
                        "task_id": context.task.task_id,
                        "resource_id": resource_id.value,
                    },
                )

            context.increment_attempts()

        except Exception as exc:
            error_msg = f"Horizontal test failed for {resource_id.value}: {exc}"
            context.add_error(error_msg)

            logger.warning(
                "Horizontal IDOR test failed",
                extra={
                    "task_id": context.task.task_id,
                    "resource_id": resource_id.value,
                    "error": str(exc),
                },
            )

    async def _test_vertical_access(
        self,
        context: IDORDetectionContext,
        resource_id: ResourceId,
        privilege_level: PrivilegeLevel,
        smart_manager: UnifiedSmartDetectionManager,
    ) -> None:
        """
        測試垂直權限訪問

        Args:
            context: 檢測上下文
            resource_id: 資源 ID
            privilege_level: 權限級別
            smart_manager: 智能檢測管理器
        """
        try:
            # 執行垂直權限提升測試
            test_result = await context.vertical_tester.test_vertical_escalation(
                url=str(context.task.target.url),
                test_privilege=privilege_level,
                method=context.task.target.method or "GET",
            )

            if isinstance(test_result, VerticalTestResult) and test_result.vulnerable:
                finding = self._build_vertical_finding(
                    context.task,
                    resource_id,
                    test_result,
                    privilege_level,
                )
                context.add_finding(finding)
                smart_manager.report_vulnerability_found()

                logger.info(
                    "Vertical IDOR vulnerability detected",
                    extra={
                        "task_id": context.task.task_id,
                        "resource_id": resource_id.value,
                        "privilege_level": privilege_level.value,
                    },
                )

            context.increment_attempts()

        except Exception as exc:
            error_msg = f"Vertical test failed for {resource_id.value}: {exc}"
            context.add_error(error_msg)

            logger.warning(
                "Vertical IDOR test failed",
                extra={
                    "task_id": context.task.task_id,
                    "resource_id": resource_id.value,
                    "privilege_level": privilege_level.value,
                    "error": str(exc),
                },
            )

    def _build_horizontal_finding(
        self,
        task: FunctionTaskPayload,
        resource_id: ResourceId,
        test_result: CrossUserTestResult,
    ) -> dict[str, Any]:
        """構建水平 IDOR 檢測結果"""
        from aiva_common.enums import Confidence, Severity, VulnerabilityType
        from aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
        )
        from aiva_common.utils import new_id

        return {
            "finding_id": new_id("finding"),
            "vulnerability": {
                "name": VulnerabilityType.IDOR,
                "severity": Severity.HIGH,
                "confidence": Confidence.FIRM,
            },
            "target": FindingTarget(
                url=str(task.target.url),
                method=task.target.method or "GET",
                parameter=task.target.parameter,
            ),
            "evidence": FindingEvidence(
                request=f"{task.target.method or 'GET'} {task.target.url}",
                response=(
                    f"Cross-user test result: "
                    f"Status {test_result.test_status}, "
                    f"Similarity: {test_result.similarity_score:.2f}"
                ),
                payload=(
                    f"Modified resource ID for horizontal testing: {resource_id.value}"
                ),
                proof=test_result.evidence,
            ),
            "impact": FindingImpact(
                description=(
                    "An attacker could access resources belonging to "
                    "other users, leading to unauthorized data access "
                    "and privacy violations."
                ),
            ),
            "recommendation": FindingRecommendation(
                fix=(
                    "Implement proper authorization checks to ensure "
                    "users can only access their own resources. "
                    "Validate resource ownership before granting access."
                ),
            ),
        }

    def _build_vertical_finding(
        self,
        task: FunctionTaskPayload,
        resource_id: ResourceId,
        test_result: VerticalTestResult,
        privilege_level: PrivilegeLevel,
    ) -> dict[str, Any]:
        """構建垂直 IDOR 檢測結果"""
        from aiva_common.enums import Confidence, Severity, VulnerabilityType
        from aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
        )
        from aiva_common.utils import new_id

        return {
            "finding_id": new_id("finding"),
            "vulnerability": {
                "name": VulnerabilityType.IDOR,
                "severity": Severity.HIGH,
                "confidence": Confidence.FIRM,
            },
            "target": FindingTarget(
                url=str(task.target.url),
                method=task.target.method or "GET",
                parameter=task.target.parameter,
            ),
            "evidence": FindingEvidence(
                request=f"{task.target.method or 'GET'} {task.target.url}",
                response=(
                    f"Vertical test result: "
                    f"Status {getattr(test_result, 'status_code', 'Unknown')}, "
                    f"Should deny: {getattr(test_result, 'should_deny', True)}"
                ),
                payload=(
                    f"Privilege escalation attempt to {privilege_level.value}: "
                    f"{resource_id.value}"
                ),
                proof=test_result.evidence,
            ),
            "impact": FindingImpact(
                description=(
                    f"An attacker could escalate privileges to "
                    f"{privilege_level.value} level, potentially gaining "
                    f"administrative access and compromising the "
                    f"entire system."
                ),
            ),
            "recommendation": FindingRecommendation(
                fix=(
                    "Implement strict role-based access control (RBAC) "
                    "and validate user permissions for each resource access. "
                    "Ensure proper privilege separation."
                ),
            ),
        }

