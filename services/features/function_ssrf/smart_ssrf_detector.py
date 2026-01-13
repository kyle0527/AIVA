"""
Smart SSRF Detector - 智能 SSRF 檢測器
整合統一檢測管理器，提供自適應超時、速率限制和早期停止功能
"""



import asyncio
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger
from services.aiva_common.detection import UnifiedSmartDetectionManager, DetectionPhase
from .config.ssrf_config import SsrfConfig

from .internal_address_detector import InternalAddressDetector, InternalAddressDetection
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

    def __init__(self, config: SsrfConfig | None = None) -> None:
        """
        初始化智能 SSRF 檢測器

        Args:
            config: SSRF 檢測配置，如果未提供則使用默認配置
        """
        self.config = config or SsrfConfig()
        logger.info(
            "Smart SSRF Detector initialized",
            extra={
                "enable_internal_scan": self.config.enable_internal_scan,
                "enable_cloud_metadata": self.config.enable_cloud_metadata,
                "request_timeout": self.config.request_timeout,
                "safe_mode": self.config.safe_mode,
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
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        # 創建統一智能檢測管理器
        smart_manager = UnifiedSmartDetectionManager(
            detection_id=task.task_id,
            detection_type="ssrf",
            target_url=str(task.target.url)
        )
        
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
            error_msg = f"No SSRF test vectors generated for task {task.task_id}. Target: {task.target.url}, Parameter: {task.target.parameter}"
            logger.error(
                error_msg,
                extra={
                    "task_id": task.task_id,
                    "target_url": str(task.target.url),
                    "parameter": task.target.parameter,
                    "parameter_location": task.target.parameter_location
                },
            )
            raise ValueError(error_msg)

        async with smart_manager.start_detection():
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
                smart_manager.metrics.start_phase(DetectionPhase.PAYLOAD_INJECTION)
                await self._execute_detection(context, prioritized_vectors, smart_manager)

                metrics = smart_manager.get_metrics().to_dict()
                return context.findings, metrics

            except Exception as exc:
                logger.exception(
                    "Error during SSRF detection",
                    extra={"task_id": task.task_id},
                )
                context.add_error(str(exc))
                metrics = smart_manager.get_metrics().to_dict()
                return context.findings, metrics

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
        smart_manager: UnifiedSmartDetectionManager,
    ) -> None:
        """
        執行檢測邏輯

        Args:
            context: 檢測上下文
            vectors: 測試向量列表
            smart_manager: 智能檢測管理器
        """
        total = len(vectors)
        
        for i, vector in enumerate(vectors, 1):
            # 檢查是否應該早期停止
            if not smart_manager.should_continue_testing():
                logger.info(
                    f"Early stopping triggered after {i - 1} vectors",
                    extra={"findings_count": len(context.findings)},
                )
                break

            # 更新進度
            smart_manager.update_progress(i, total)

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
                # 驗證是否為真實的內部服務訪問
                if self._verify_internal_service_access(response, detection):
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
                else:
                    logger.debug(
                        "Internal address detection dismissed as false positive",
                        extra={
                            "task_id": context.task.task_id,
                            "payload": payload,
                            "reason": "Failed internal service verification",
                        },
                    )

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
        """發送 HTTP 請求 - 重構後複雜度 ≤15
        
        Note: timeout 已在 client 初始化時設定，無需在此處理
        """
        # 解析目標和參數
        target_config = self._parse_target_config(task, vector)
        
        # 處理參數置換
        request_data = self._process_parameter_injection(target_config, payload)
        
        # 發送請求（timeout 已在 client 初始化時設定）
        return await self._execute_http_request(client, request_data)
    
    def _parse_target_config(self, task: FunctionTaskPayload, vector: SsrfTestVector) -> dict:
        """解析目標配置 - 複雜度 8"""
        target = task.target
        method = (target.method or "GET").upper()
        parameter = vector.parameter or target.parameter
        location = (vector.location or target.parameter_location or "query").lower()
        
        from urllib.parse import parse_qsl, urlparse
        parsed = urlparse(str(target.url))
        
        return {
            'method': method,
            'parameter': parameter,
            'location': location,
            'parsed_url': parsed,
            'base_params': dict(parse_qsl(parsed.query, keep_blank_values=True)),
            'headers': dict(target.headers),
            'cookies': dict(target.cookies),
            'form_data': dict(target.form_data),
            'json_data': dict(target.json_data or {}) if target.json_data else None,
            'body': target.body
        }
    
    def _process_parameter_injection(self, config: dict, payload: str) -> dict:
        """處理參數注入邏輯 - 使用策略模式重構，複雜度 ≤15"""
        location = config['location']
        parameter = config['parameter']
        
        # 使用策略模式 - 每個處理器複雜度 ≤5
        injection_handlers = {
            'query': self._inject_query_parameter,
            'url': self._inject_query_parameter,
            'body': self._inject_form_parameter,
            'form': self._inject_form_parameter,
            'json': self._inject_json_parameter,
            'header': self._inject_header_parameter,
            'cookie': self._inject_cookie_parameter,
            'body_raw': self._inject_body_raw_parameter
        }
        
        handler = injection_handlers.get(location)
        if handler:
            handler(config, parameter, payload)
        
        return config
    
    def _inject_query_parameter(self, config: dict, parameter: str, payload: str) -> None:
        """注入查詢參數 - 複雜度 2"""
        if parameter:
            config['base_params'][parameter] = payload
    
    def _inject_form_parameter(self, config: dict, parameter: str, payload: str) -> None:
        """注入表單參數 - 複雜度 2"""
        if parameter:
            config['form_data'][parameter] = payload
    
    def _inject_json_parameter(self, config: dict, parameter: str, payload: str) -> None:
        """注入JSON參數 - 複雜度 3"""
        if parameter:
            config['json_data'] = config['json_data'] or {}
            config['json_data'][parameter] = payload
    
    def _inject_header_parameter(self, config: dict, parameter: str, payload: str) -> None:
        """注入頭部參數 - 複雜度 2"""
        if parameter:
            config['headers'][parameter] = payload
    
    def _inject_cookie_parameter(self, config: dict, parameter: str, payload: str) -> None:
        """注入Cookie參數 - 複雜度 2"""
        if parameter:
            config['cookies'][parameter] = payload
    
    def _inject_body_raw_parameter(self, config: dict, parameter: str, payload: str) -> None:
        """注入原始Body參數 - 複雜度 4"""
        if parameter and config['body']:
            config['body'] = config['body'].replace(f"{{{{{parameter}}}}}", payload)
        else:
            config['body'] = payload
    
    async def _execute_http_request(self, client: httpx.AsyncClient, config: dict) -> httpx.Response:
        """執行 HTTP 請求 - 複雜度 8
        
        Note: timeout 已經在 client 初始化時設定，不需要作為參數傳遞
        """
        from urllib.parse import urlunparse
        
        new_url = urlunparse(
            config['parsed_url']._replace(
                query="&".join(f"{k}={v}" for k, v in config['base_params'].items())
            )
        )
        
        base_kwargs = {
            'method': config['method'],
            'url': new_url,
            'headers': config['headers'],
            'cookies': config['cookies']
        }
        
        # 使用 Early Return Pattern 避免多層 if-elif
        if config['json_data']:
            return await client.request(**base_kwargs, json=config['json_data'])
        if config['form_data']:
            return await client.request(**base_kwargs, data=config['form_data'])
        if config['body']:
            return await client.request(**base_kwargs, content=config['body'])
        
        return await client.request(**base_kwargs)

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

    def _verify_internal_service_access(
        self,
        response: httpx.Response,
        detection: InternalAddressDetection
    ) -> bool:
        """驗證是否為真實的內部服務訪問，過濾虛假內部地址檢測"""
        # 檢查回應狀態碼 (200-299 表示成功訪問)
        if not (200 <= response.status_code < 300):
            # 非 2xx 狀態碼可能是錯誤頁面，不是真實內部服務
            return False
        
        # 檢查回應內容長度 (太短可能是空白回應)
        if len(response.text) < 50:
            return False
        
        # 驗證是否包含真實的服務內容
        service_content_verified = self._verify_service_content(response)
        if not service_content_verified:
            return False
        
        # 驗證雲端 metadata 服務的真實性
        if any(ind.source == "cloud_metadata" for ind in detection.indicators):
            return self._verify_cloud_metadata(response)
        
        return True
    
    def _verify_service_content(
        self,
        response: httpx.Response
    ) -> bool:
        """驗證服務內容的真實性"""
        text = response.text.lower()
        
        # 檢查是否為常見的內部服務回應
        internal_service_indicators = [
            # Web 服務器
            ("apache", "server:", "httpd"),
            ("nginx", "server:", "welcome"),
            # 資料庫
            ("mysql", "database", "sql"),
            ("postgres", "postgresql", "database"),
            ("redis", "redis server", "version"),
            # 訊息佇列
            ("rabbitmq", "amqp", "queue"),
            ("kafka", "broker", "topic"),
            # 容器/編排
            ("docker", "container", "image"),
            ("kubernetes", "pod", "service"),
            # 監控/日誌
            ("elasticsearch", "cluster", "index"),
            ("prometheus", "metrics", "query"),
            ("grafana", "dashboard", "panel"),
        ]
        
        # 檢查是否匹配任何內部服務特徵 (需 3 個指標中至少 2 個)
        for indicators in internal_service_indicators:
            matches = sum(1 for indicator in indicators if indicator in text)
            if matches >= 2:
                return True
        
        # 檢查 JSON API 回應 (常見於內部服務)
        if self._is_valid_json_api(response):
            return True
        
        # 檢查 HTML 管理介面
        if self._is_admin_interface(text):
            return True
        
        return False
    
    def _verify_cloud_metadata(self, response: httpx.Response) -> bool:
        """驗證雲端 metadata 服務回應的真實性"""
        text = response.text
        
        # AWS metadata 特徵
        aws_indicators = [
            "ami-id", "instance-id", "instance-type",
            "security-credentials", "iam/security-credentials"
        ]
        if any(indicator in text for indicator in aws_indicators):
            # 嘗試解析 JSON
            try:
                data = json.loads(text)
                if isinstance(data, dict) and any(k in data for k in ["Code", "AccessKeyId", "Token"]):
                    return True
            except json.JSONDecodeError:
                pass
            return len([ind for ind in aws_indicators if ind in text]) >= 2
        
        # GCP metadata 特徵
        gcp_indicators = [
            "project-id", "instance/id", "service-accounts",
            "attributes/", "metadata.google.internal"
        ]
        if any(indicator in text for indicator in gcp_indicators):
            return len([ind for ind in gcp_indicators if ind in text]) >= 2
        
        # Azure metadata 特徵
        azure_indicators = [
            "compute", "vmId", "subscriptionId",
            "resourceGroupName", "azEnvironment"
        ]
        if any(indicator in text for indicator in azure_indicators):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "compute" in data:
                    return True
            except json.JSONDecodeError:
                pass
        
        return False
    
    def _is_valid_json_api(self, response: httpx.Response) -> bool:
        """檢查是否為有效的 JSON API 回應"""
        # 檢查 Content-Type
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            return False
        
        # 嘗試解析 JSON
        try:
            data = json.loads(response.text)
            # 確認是有結構的 JSON (不是空物件)
            if isinstance(data, (dict, list)):
                return len(data) >= 1
        except json.JSONDecodeError:
            pass
        
        return False
    
    def _is_admin_interface(self, text: str) -> bool:
        """檢查是否為管理介面"""
        admin_patterns = [
            "admin panel", "administration", "dashboard",
            "control panel", "management console",
            "<title>admin", "<title>dashboard"
        ]
        
        return any(pattern in text for pattern in admin_patterns)
    
    def _extract_token(self, payload: str) -> str:
        """從載荷中提取 token"""
        # 嘗試從各種OAST載荷格式中提取token
        
        # 格式1: http://token.burpcollaborator.net
        match = re.search(r'https?://([a-zA-Z0-9]+)\.(?:burpcollaborator|oast|canarytokens)', payload)
        if match:
            return match.group(1)
        
        # 格式2: http://example.com/callback?token=xxxxx
        match = re.search(r'[?&]token=([a-zA-Z0-9]+)', payload)
        if match:
            return match.group(1)
        
        # 格式3: http://xxxxx.interact.sh
        match = re.search(r'https?://([a-zA-Z0-9]+)\.interact\.sh', payload)
        if match:
            return match.group(1)
        
        # 格式4: 直接使用域名部分作為token
        match = re.search(r'https?://([^/:]+)', payload)
        if match:
            domain = match.group(1)
            # 提取子域名第一部分作為token
            parts = domain.split('.')
            if len(parts) > 2:
                return parts[0]
        
        # 如果無法提取，返回payload的hash作為唯一標識
        import hashlib
        return hashlib.md5(payload.encode()).hexdigest()[:8]
