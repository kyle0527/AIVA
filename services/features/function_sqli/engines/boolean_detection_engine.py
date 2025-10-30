"""
布林檢測引擎 - 基於布林邏輯的SQL注入檢測
"""



import asyncio
from typing import cast

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder

logger = get_logger(__name__)


class BooleanDetectionEngine:
    """布林檢測引擎 - 檢測基於布林邏輯的SQL注入"""

    def __init__(self):
        self.boolean_payloads = [
            # True conditions
            ("' OR '1'='1", True),
            ("' OR 1=1 --", True),
            ('" OR "1"="1', True),
            ("') OR ('1'='1", True),
            ("' OR 'a'='a", True),
            ("1' OR '1'='1' #", True),
            # False conditions
            ("' OR '1'='2", False),
            ("' OR 1=2 --", False),
            ('" OR "1"="2', False),
            ("') OR ('1'='2", False),
            ("' OR 'a'='b", False),
            ("1' OR '1'='2' #", False),
        ]

    async def detect(
        self, task: FunctionTaskPayload, client: httpx.AsyncClient
    ) -> list[DetectionResult]:
        """執行布林檢測"""
        results: list[DetectionResult] = []
        encoder = PayloadWrapperEncoder(task)

        logger.debug(f"Starting boolean detection for task {task.task_id}")

        # 獲取基準回應
        baseline_response = await self._get_baseline_response(task, client, encoder)
        if not baseline_response:
            logger.warning("Failed to get baseline response")
            return results

        # 測試布林載荷對
        for i in range(0, len(self.boolean_payloads), 2):
            true_payload = self.boolean_payloads[i][0]
            false_payload = self.boolean_payloads[i + 1][0]

            try:
                # 並行發送True和False條件請求
                results_tuple = await asyncio.gather(
                    self._send_payload_request(true_payload, encoder, client),
                    self._send_payload_request(false_payload, encoder, client),
                    return_exceptions=True,
                )
                true_response, false_response = results_tuple[0], results_tuple[1]

                if isinstance(true_response, Exception) or isinstance(
                    false_response, Exception
                ):
                    logger.warning(
                        f"Failed to test boolean pair: {true_payload}, {false_payload}"
                    )
                    continue

                # 確保回應不是異常並進行類型轉換
                assert not isinstance(true_response, Exception)
                assert not isinstance(false_response, Exception)

                # 類型轉換以滿足靜態類型檢查
                true_resp = cast(httpx.Response, true_response)
                false_resp = cast(httpx.Response, false_response)

                # 分析回應差異
                if self._analyze_boolean_responses(
                    baseline_response, true_resp, false_resp
                ):
                    result = self._build_detection_result(
                        true_payload=true_payload,
                        false_payload=false_payload,
                        true_response=true_resp,
                        false_response=false_resp,
                        task=task,
                    )
                    results.append(result)
                    logger.info(
                        f"Boolean SQL injection detected with payloads: {true_payload} / {false_payload}"
                    )

            except Exception as e:
                logger.warning(
                    f"Boolean detection failed for payloads '{true_payload}' / '{false_payload}': {e}"
                )
                continue

        logger.debug(
            f"Boolean detection completed. Found {len(results)} vulnerabilities"
        )
        return results

    async def _get_baseline_response(
        self,
        task: FunctionTaskPayload,
        client: httpx.AsyncClient,
        encoder: PayloadWrapperEncoder,
    ) -> httpx.Response | None:
        """獲取基準回應"""
        try:
            # 使用原始參數值作為基準
            baseline_encoded = encoder.encode("")  # 空載荷
            return await client.request(
                method=baseline_encoded.method,
                url=baseline_encoded.url,
                **baseline_encoded.request_kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to get baseline response: {e}")
            return None

    async def _send_payload_request(
        self, payload: str, encoder: PayloadWrapperEncoder, client: httpx.AsyncClient
    ) -> httpx.Response:
        """發送載荷請求"""
        encoded = encoder.encode(payload)
        return await client.request(
            method=encoded.method, url=encoded.url, **encoded.request_kwargs
        )

    def _analyze_boolean_responses(
        self,
        baseline: httpx.Response,
        true_response: httpx.Response,
        false_response: httpx.Response,
    ) -> bool:
        """分析布林回應以檢測SQL注入"""
        # 檢查狀態碼差異
        if true_response.status_code != false_response.status_code:
            return True

        # 檢查內容長度差異（閾值：50個字符或10%差異）
        true_len = len(true_response.text or "")
        false_len = len(false_response.text or "")
        baseline_len = len(baseline.text or "")

        if abs(true_len - false_len) > 50:
            return True

        if baseline_len > 0:
            true_diff = abs(true_len - baseline_len) / baseline_len
            false_diff = abs(false_len - baseline_len) / baseline_len
            if abs(true_diff - false_diff) > 0.1:  # 10%差異
                return True

        # 檢查回應時間差異（如果可用）
        true_time = getattr(true_response, "elapsed", None)
        false_time = getattr(false_response, "elapsed", None)
        if true_time and false_time:
            time_diff = abs(true_time.total_seconds() - false_time.total_seconds())
            if time_diff > 2.0:  # 2秒差異
                return True

        return False

    def _build_detection_result(
        self,
        true_payload: str,
        false_payload: str,
        true_response: httpx.Response,
        false_response: httpx.Response,
        task: FunctionTaskPayload,
    ) -> DetectionResult:
        """構建檢測結果"""
        from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
        from services.aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
            Vulnerability,
        )

        vulnerability = Vulnerability(
            name=VulnerabilityType.SQLI,
            severity=Severity.MEDIUM,
            confidence=Confidence.FIRM,
        )

        evidence = FindingEvidence(
            payload=f"True: {true_payload}, False: {false_payload}",
            request=(
                f"True condition: {true_payload}, False condition: {false_payload}"
            ),
            response=(
                f"True response: {true_response.status_code} "
                f"({len(true_response.text or '')} chars), "
                f"False response: {false_response.status_code} "
                f"({len(false_response.text or '')} chars)"
            ),
            proof=(
                f"Different responses for TRUE ({true_payload}) and "
                f"FALSE ({false_payload}) conditions indicate SQL "
                f"injection vulnerability."
            ),
        )

        impact = FindingImpact(
            description=(
                "Boolean-based SQL injection allows attackers to "
                "extract data through conditional queries."
            ),
            business_impact=(
                "High - potential for complete data extraction and system compromise"
            ),
        )

        recommendation = FindingRecommendation(
            fix=(
                "Implement parameterized queries and input validation "
                "immediately. Conduct security code review and implement "
                "comprehensive input sanitization."
            ),
            priority="High",
        )

        target = FindingTarget(
            url=str(true_response.url),
            method=task.target.method,
            parameter=task.target.parameter,
        )

        return DetectionResult(
            is_vulnerable=True,
            vulnerability=vulnerability,
            evidence=evidence,
            impact=impact,
            recommendation=recommendation,
            target=target,
            detection_method="boolean_based",
            payload_used=f"TRUE: {true_payload} | FALSE: {false_payload}",
            confidence_score=0.85,
        )
