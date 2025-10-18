"""
Smart SSRF Detector - 智能 SSRF 檢測器
整合統一檢測管理器，提供自適應超時、速率限制和早期停止功能
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger
from services.function.common.detection_config import SSRFConfig
from services.function.common.unified_smart_detection_manager import (
    DetectionMetrics,
    UnifiedSmartDetectionManager,
)

from .internal_address_detector import InternalAddressDetector
from .oast_dispatcher import OastDispatcher, OastEvent
from .param_semantics_analyzer import (
    OAST_PLACEHOLDER,
    AnalysisPlan,
    ParamSemanticsAnalyzer,
    SsrfTestVector,
)

logger = get_logger(__name__)


@dataclass
class SSRFDetectionContext:
    """SSRF 檢測上下文"""

    task: FunctionTaskPayload
    client: httpx.AsyncClient
    analyzer: ParamSemanticsAnalyzer
    detector: InternalAddressDetector
    dispatcher: OastDispatcher

    # 智能檢測狀態
    start_time: float = field(default_factory=time.time)
    attempts: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    oast_callbacks: int = 0

    def add_finding(self, finding: dict[str, Any]) -> None:
        """添加發現的漏洞"""
        self.findings.append(finding)

    def add_error(self, error: str) -> None:
        """添加錯誤"""
        self.errors.append(error)

    def increment_attempts(self) -> None:
        """增加嘗試次數"""
        self.attempts += 1

    def add_oast_callbacks(self, count: int) -> None:
        """添加 OAST 回調數量"""
        self.oast_callbacks += count


class SmartSSRFDetector:
    """智能 SSRF 檢測器"""

    def __init__(self, config: SSRFConfig | None = None) -> None:
        """
        初始化智能 SSRF 檢測器

        Args:
            config: SSRF 檢測配置，如果未提供則使用默認配置
        """
        self.config = config or SSRFConfig()
        self.smart_manager = UnifiedSmartDetectionManager("SSRF", self.config)
        logger.info(
            "Smart SSRF Detector initialized",
            extra={
                "max_vulnerabilities": self.config.max_vulnerabilities,
                "timeout_base": self.config.timeout_base,
                "timeout_max": self.config.timeout_max,
                "requests_per_second": self.config.requests_per_second,
            },
        )

    async def detect_vulnerabilities(
        self,
        task: FunctionTaskPayload,
        *,
        client: httpx.AsyncClient,
        analyzer: ParamSemanticsAnalyzer,
        detector: InternalAddressDetector,
        dispatcher: OastDispatcher,
    ) -> tuple[list[dict[str, Any]], DetectionMetrics]:
        """
        執行智能 SSRF 檢測

        Args:
            task: 功能任務載荷
            client: HTTP 客戶端
            analyzer: 參數語義分析器
            detector: 內部地址檢測器
            dispatcher: OAST 調度器

        Returns:
            檢測到的漏洞列表和檢測指標
        """
        context = SSRFDetectionContext(
            task=task,
            client=client,
            analyzer=analyzer,
            detector=detector,
            dispatcher=dispatcher,
        )

        # 分析任務並生成測試向量
        plan: AnalysisPlan = analyzer.analyze(task)

        if not plan.vectors:
            logger.debug(
                "No SSRF payloads generated for task",
                extra={"task_id": task.task_id},
            )
            return [], self.smart_manager.metrics

        # 開始智能檢測
        total_steps = len(plan.vectors)
        self.smart_manager.start_detection(total_steps)

        try:
            # 優先級排序 - 雲元數據端點優先
            prioritized_vectors = self._prioritize_vectors(plan.vectors)

            logger.info(
                f"Starting SSRF detection with {len(prioritized_vectors)} vectors",
                extra={
                    "task_id": task.task_id,
                    "cloud_metadata_first": self.config.cloud_metadata_first,
                },
            )

            # 執行檢測
            await self._execute_detection(context, prioritized_vectors)

            return context.findings, self.smart_manager.metrics

        finally:
            # 無需特殊的結束會話邏輯
            pass

    def _prioritize_vectors(
        self, vectors: list[SsrfTestVector]
    ) -> list[SsrfTestVector]:
        """
        優先級排序測試向量

        Args:
            vectors: 原始測試向量列表

        Returns:
            按優先級排序的測試向量列表
        """
        if not self.config.cloud_metadata_first:
            return vectors

        cloud_vectors = []
        other_vectors = []

        for vector in vectors:
            payload = str(vector.payload).lower()
            is_cloud = any(
                endpoint in payload for endpoint in self.config.cloud_metadata_endpoints
            )

            if is_cloud:
                cloud_vectors.append(vector)
            else:
                other_vectors.append(vector)

        logger.debug(
            f"Vector prioritization: {len(cloud_vectors)} cloud, "
            f"{len(other_vectors)} others",
        )

        return cloud_vectors + other_vectors

    async def _execute_detection(
        self,
        context: SSRFDetectionContext,
        vectors: list[SsrfTestVector],
    ) -> None:
        """
        執行檢測邏輯

        Args:
            context: 檢測上下文
            vectors: 測試向量列表
        """
        for i, vector in enumerate(vectors, 1):
            # 檢查是否應該早期停止
            if not self.smart_manager.should_continue_testing():
                logger.info(
                    f"Early stopping triggered after {i - 1} vectors",
                    extra={"findings_count": len(context.findings)},
                )
                break

            # 更新進度
            self.smart_manager.update_progress(f"測試向量 {i}/{len(vectors)}")

            # 執行單個向量檢測
            await self._test_vector(context, vector)

            context.increment_attempts()

    async def _test_vector(
        self,
        context: SSRFDetectionContext,
        vector: SsrfTestVector,
    ) -> None:
        """
        測試單個 SSRF 向量

        Args:
            context: 檢測上下文
            vector: 測試向量
        """
        try:
            # 解析載荷
            payload = await self._resolve_payload(
                vector, context.dispatcher, context.task
            )

            # 發送請求
            response = await self._issue_request(
                context.client,
                context.task,
                vector,
                payload,
            )

            # 檢測內部地址訪問
            detection = context.detector.analyze(response)
            if detection.matched:
                finding = self._build_internal_finding(
                    task=context.task,
                    vector=vector,
                    payload=payload,
                    response=response,
                    detection_summary=detection.summary(),
                )
                context.add_finding(finding)

                logger.info(
                    "SSRF vulnerability detected (internal address)",
                    extra={
                        "task_id": context.task.task_id,
                        "payload": payload,
                        "detection": detection.summary(),
                    },
                )
                return

            # 檢查 OAST 回調
            if vector.requires_oast:
                await asyncio.sleep(self.config.oast_wait_time)

                events = await context.dispatcher.fetch_events(payload)
                if not events:
                    events = await context.dispatcher.fetch_events(
                        self._extract_token(payload)
                    )

                if events:
                    context.add_oast_callbacks(len(events))
                    finding = self._build_oast_finding(
                        task=context.task,
                        vector=vector,
                        payload=payload,
                        events=events,
                    )
                    context.add_finding(finding)

                    logger.info(
                        "SSRF vulnerability detected (OAST)",
                        extra={
                            "task_id": context.task.task_id,
                            "payload": payload,
                            "callbacks": len(events),
                        },
                    )

        except Exception as exc:
            error_msg = f"Failed to execute SSRF payload: {exc}"
            context.add_error(error_msg)

            logger.warning(
                "SSRF payload execution failed",
                extra={
                    "task_id": context.task.task_id,
                    "payload": str(vector.payload),
                    "error": str(exc),
                },
            )

    async def _resolve_payload(
        self,
        vector: SsrfTestVector,
        dispatcher: OastDispatcher,
        task: FunctionTaskPayload,
    ) -> str:
        """解析載荷，處理 OAST 占位符"""
        payload = vector.payload
        if vector.requires_oast:
            probe = await dispatcher.register(task)
            payload = payload.replace(OAST_PLACEHOLDER, probe.callback_url)
        return str(payload)

    async def _issue_request(
        self,
        client: httpx.AsyncClient,
        task: FunctionTaskPayload,
        vector: SsrfTestVector,
        payload: str,
    ) -> httpx.Response:
        """發送 HTTP 請求"""
        # 使用統一管理器的自適應超時
        current_timeout = self.smart_manager.timeout_manager.get_timeout()

        # 複製 worker.py 的請求邏輯，但使用自適應超時
        target = task.target
        method = (target.method or "GET").upper()
        parameter = vector.parameter or target.parameter
        location = (vector.location or target.parameter_location or "query").lower()

        from urllib.parse import parse_qsl, urlparse

        parsed = urlparse(str(target.url))
        base_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        headers = dict(target.headers)
        cookies = dict(target.cookies)
        data = dict(target.form_data)
        json_data = dict(target.json_data or {}) if target.json_data else None
        content = target.body

        if location in {"query", "url"} and parameter:
            base_params[parameter] = payload
        elif location in {"body", "form"} and parameter:
            data[parameter] = payload
        elif location == "json" and parameter:
            json_data = json_data or {}
            json_data[parameter] = payload
        elif location == "header" and parameter:
            headers[parameter] = payload
        elif location == "cookie" and parameter:
            cookies[parameter] = payload
        elif location == "body_raw":
            if parameter and content:
                content = content.replace(f"{{{{{parameter}}}}}", payload)
            else:
                content = payload

        from urllib.parse import urlunparse

        new_url = urlunparse(
            parsed._replace(query="&".join(f"{k}={v}" for k, v in base_params.items()))
        )

        # httpx 的正確參數組織方式
        if json_data:
            return await client.request(
                method=method,
                url=new_url,
                headers=headers,
                cookies=cookies,
                json=json_data,
                timeout=current_timeout,
            )
        elif data:
            return await client.request(
                method=method,
                url=new_url,
                headers=headers,
                cookies=cookies,
                data=data,
                timeout=current_timeout,
            )
        elif content:
            return await client.request(
                method=method,
                url=new_url,
                headers=headers,
                cookies=cookies,
                content=content,
                timeout=current_timeout,
            )
        else:
            return await client.request(
                method=method,
                url=new_url,
                headers=headers,
                cookies=cookies,
                timeout=current_timeout,
            )

    def _build_internal_finding(
        self,
        *,
        task: FunctionTaskPayload,
        vector: SsrfTestVector,
        payload: str,
        response: httpx.Response,
        detection_summary: str,
    ) -> dict[str, Any]:
        """構建內部地址檢測結果"""
        from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
        from services.aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
        )
        from services.aiva_common.utils import new_id

        return {
            "finding_id": new_id("finding"),
            "vulnerability": {
                "name": VulnerabilityType.SSRF,
                "severity": Severity.HIGH,
                "confidence": Confidence.FIRM,
            },
            "target": FindingTarget(
                url=str(task.target.url),
                method=task.target.method or "GET",
                parameter=vector.parameter or task.target.parameter,
            ),
            "evidence": FindingEvidence(
                request=f"{task.target.method or 'GET'} {task.target.url}",
                response=(
                    f"Status: {response.status_code}, Detection: {detection_summary}"
                ),
                payload=payload,
                proof=(
                    f"SSRF payload triggered internal address detection: "
                    f"{detection_summary}"
                ),
            ),
            "impact": FindingImpact(
                description=(
                    "An attacker could access internal network resources, "
                    "potentially leading to data exfiltration or lateral "
                    "movement."
                ),
            ),
            "recommendation": FindingRecommendation(
                fix=(
                    "Implement proper input validation and whitelist "
                    "allowed destinations for outbound requests."
                ),
            ),
        }

    def _build_oast_finding(
        self,
        *,
        task: FunctionTaskPayload,
        vector: SsrfTestVector,
        payload: str,
        events: list[OastEvent],
    ) -> dict[str, Any]:
        """構建 OAST 檢測結果"""
        from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
        from services.aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
        )
        from services.aiva_common.utils import new_id

        callbacks_info = f"{len(events)} callback(s) received"

        return {
            "finding_id": new_id("finding"),
            "vulnerability": {
                "name": VulnerabilityType.SSRF,
                "severity": Severity.HIGH,
                "confidence": Confidence.FIRM,
                "title": (
                    "Server-Side Request Forgery (SSRF) - External Network Access"
                ),
                "description": (
                    f"SSRF vulnerability confirmed through out-of-band "
                    f"callbacks. {callbacks_info}"
                ),
            },
            "target": FindingTarget(
                url=str(task.target.url),
                method=task.target.method or "GET",
                parameter=vector.parameter or task.target.parameter,
            ),
            "evidence": FindingEvidence(
                request=f"{task.target.method or 'GET'} {task.target.url}",
                response=f"OAST callbacks: {callbacks_info}",
                payload=payload,
                proof=f"SSRF confirmed via OAST callbacks: {callbacks_info}",
            ),
            "impact": FindingImpact(
                description=(
                    "An attacker could make the server send requests to "
                    "arbitrary external servers, potentially leading to "
                    "data exfiltration."
                ),
            ),
            "recommendation": FindingRecommendation(
                fix=(
                    "Implement proper input validation and restrict "
                    "outbound network access from the application server."
                ),
            ),
        }

    def _extract_token(self, payload: str) -> str:
        """從載荷中提取 token"""
        # 這裡需要根據 OAST 系統的具體實現來提取 token
        # 暫時返回載荷本身
        return payload
