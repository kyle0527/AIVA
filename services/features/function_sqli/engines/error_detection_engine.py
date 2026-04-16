"""
錯誤檢測引擎 - 重構後的模組化版本
"""



import re

import aiohttp
from aiva_common.utils import get_logger

from ..config import SqliConfig
from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder
from .base_detector import BaseDetector

logger = get_logger(__name__)


class ErrorDetectionEngine(BaseDetector):
    """錯誤檢測引擎 - 檢測SQL錯誤訊息"""

    def __init__(self, session: aiohttp.ClientSession, config: SqliConfig, payload_encoder: PayloadWrapperEncoder):
        super().__init__(session, config, payload_encoder)
        self.error_patterns = {
            "mysql": [
                r"You have an error in your SQL syntax",
                r"mysql_fetch_array\(\)",
                r"MySQL server version for the right syntax",
            ],
            "postgresql": [
                r"PostgreSQL query failed",
                r"pg_query\(\) \[",
                r"invalid input syntax for",
            ],
            "mssql": [
                r"Microsoft OLE DB Provider for SQL Server",
                r"Unclosed quotation mark after",
                r"Incorrect syntax near",
            ],
            "oracle": [
                r"ORA-\d+:",
                r"Oracle error",
                r"Oracle driver",
            ],
        }

        self.error_payloads = [
            "'",
            "' OR '1'='1' --",
            '" OR "1"="1" --',
            "') OR ('1'='1",
            "admin'--",
            "admin'/*",
            "' or 1=1#",
            "' or 1=1--",
            "' or 1=1/*",
        ]

    async def detect(
        self, target_url: str, params: dict[str, str], method: str = "GET"
    ) -> list[DetectionResult]:
        """執行錯誤檢測"""
        results = []

        # params is passed but mostly managed by task in encoder.
        # BaseDetector signature has params, but for now we rely on the pre-configured encoder.

        logger.debug(f"Starting error detection for {target_url}")

        for payload in self.error_payloads:
            # 應用 Tamper
            tampered_payloads = self.payload_encoder.apply_tamper(payload, self.config.waf_evasion_level)
            final_payload = tampered_payloads[-1] if tampered_payloads else payload

            try:
                # 編碼載荷
                encoded = self.payload_encoder.encode(final_payload)

                # 發送請求
                # 使用 BaseDetector._send_request 邏輯?
                # _send_request 返回 (text, status).
                # 但這裡是調用 client.request.
                # 我們應該直接使用 self.session.

                # 相容舊邏輯，使用 encoder 的 url 和 kwargs
                if encoded.method == "GET":
                     async with self.session.get(encoded.url, **encoded.request_kwargs) as resp:
                         text = await resp.text()
                         conn_status = resp.status
                         response = resp # Keep ref for url
                else:
                     async with self.session.post(encoded.url, **encoded.request_kwargs) as resp:
                         text = await resp.text()
                         conn_status = resp.status
                         response = resp

                # 分析回應中的錯誤
                db_type, error_found = self._analyze_error_response(text)

                if error_found:
                    result = self._build_detection_result(
                        payload=final_payload,
                        response_url=str(response.url),
                        response_status=conn_status,
                        db_type=db_type,
                        method=method,
                        param="unknown"
                    )
                    results.append(result)
                    logger.info(
                        f"SQL error detected: {db_type} with payload '{final_payload}'"
                    )

            except Exception as e:
                logger.warning(f"Error detection failed for payload '{final_payload}': {e}")
                continue

        logger.debug(f"Error detection completed. Found {len(results)} vulnerabilities")
        return results

    def _analyze_error_response(self, response_text: str) -> tuple[str, bool]:
        """分析回應中的SQL錯誤"""
        for db_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_text, re.IGNORECASE):
                    return db_type, True

        return "unknown", False

    def _build_detection_result(
        self,
        payload: str,
        response_url: str,
        response_status: int,
        db_type: str,
        method: str,
        param: str
    ) -> DetectionResult:
        """構建檢測結果"""
        from aiva_common.enums import Confidence, Severity, VulnerabilityType
        from aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
            Vulnerability,
        )

        vulnerability = Vulnerability(
            name=VulnerabilityType.SQLI,
            severity=Severity.HIGH,
            confidence=Confidence.CERTAIN,
        )

        evidence = FindingEvidence(
            payload=payload,
            request=f"Method: {method}, URL: {response_url}",
            response=f"Status: {response_status}, Error type: {db_type}",
            proof=(
                f"The payload '{payload}' triggered a {db_type} database "
                f"error, indicating SQL injection vulnerability."
            ),
            db_version=db_type,
        )

        impact = FindingImpact(
            description=(
                "SQL injection can lead to unauthorized data access, "
                "modification, or deletion."
            ),
            business_impact=("High - potential data breach and system compromise"),
        )

        recommendation = FindingRecommendation(
            fix=(
                "Implement input validation and parameterized queries. "
                "Conduct comprehensive security audit and implement "
                "defense-in-depth strategy."
            ),
            priority="High",
        )

        target = FindingTarget(
            url=response_url,
            method=method,
            parameter=param,
        )

        return DetectionResult(
            is_vulnerable=True,
            vulnerability=vulnerability,
            evidence=evidence,
            impact=impact,
            recommendation=recommendation,
            target=target,
            detection_method="error_based",
            payload_used=payload,
            confidence_score=0.9,
        )

