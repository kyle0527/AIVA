"""
錯誤檢測引擎 - 重構後的模組化版本
"""



import re

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder

logger = get_logger(__name__)


class ErrorDetectionEngine:
    """錯誤檢測引擎 - 檢測SQL錯誤訊息"""

    def __init__(self):
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
        self, task: FunctionTaskPayload, client: httpx.AsyncClient
    ) -> list[DetectionResult]:
        """執行錯誤檢測"""
        results = []
        encoder = PayloadWrapperEncoder(task)

        logger.debug(f"Starting error detection for task {task.task_id}")

        for payload in self.error_payloads:
            try:
                # 編碼載荷
                encoded = encoder.encode(payload)

                # 發送請求
                response = await client.request(
                    method=encoded.method, url=encoded.url, **encoded.request_kwargs
                )

                # 分析回應中的錯誤
                db_type, error_found = self._analyze_error_response(response.text or "")

                if error_found:
                    result = self._build_detection_result(
                        payload=payload, response=response, db_type=db_type, task=task
                    )
                    results.append(result)
                    logger.info(
                        f"SQL error detected: {db_type} with payload '{payload}'"
                    )

            except Exception as e:
                logger.warning(f"Error detection failed for payload '{payload}': {e}")
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
        response: httpx.Response,
        db_type: str,
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
            severity=Severity.HIGH,
            confidence=Confidence.CERTAIN,
        )

        evidence = FindingEvidence(
            payload=payload,
            request=f"Method: {task.target.method}, URL: {task.target.url}",
            response=f"Status: {response.status_code}, Error type: {db_type}",
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
            detection_method="error_based",
            payload_used=payload,
            confidence_score=0.9,
        )
