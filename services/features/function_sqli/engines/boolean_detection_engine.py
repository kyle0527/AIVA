"""
布林檢測引擎 - 基於布林邏輯的SQL注入檢測
"""



import asyncio
from typing import cast, Dict, List, Optional
import aiohttp
import httpx

from aiva_common.schemas import FunctionTaskPayload
from aiva_common.utils import get_logger

from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder
from .base_detector import BaseDetector
from ..config import SqliConfig

logger = get_logger(__name__)


class BooleanDetectionEngine(BaseDetector):
    """布林檢測引擎 - 檢測基於布林邏輯的SQL注入"""

    def __init__(self, session: aiohttp.ClientSession, config: SqliConfig, payload_encoder: PayloadWrapperEncoder):
        super().__init__(session, config, payload_encoder)
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
        self, target_url: str, params: Dict[str, str], method: str = "GET"
    ) -> list[DetectionResult]:
        """執行布林檢測"""
        results: list[DetectionResult] = []

        logger.debug(f"Starting boolean detection for {target_url}")

        # 獲取基準回應
        baseline_text, baseline_status = await self._send_request(target_url, method, params=params)
        if baseline_status == 0:
             # Basic error handling
             return results

        # 測試布林載荷對
        for i in range(0, len(self.boolean_payloads), 2):
            true_payload_base = self.boolean_payloads[i][0]
            false_payload_base = self.boolean_payloads[i + 1][0]

            # 應用 Tamper (根據 config)
            # 雖然我們可以在這裡調用 apply_tamper，但這會產生很多請求。
            # 為了性能，我們可能只對基本 payload 做檢測，或者只選一種 tamper。
            # 這裡簡單起見，我們只使用原始 payload，或隨機選一種。
            # 但 PayloadEncoder.encode 只是編碼，不負責 Tamper?
            # 之前我們把 TamperMixin 混入 PayloadWrapperEncoder 了。
            # 所以我們可以調用 apply_tamper。

            true_payloads = self.payload_encoder.apply_tamper(true_payload_base, self.config.waf_evasion_level)
            false_payloads = self.payload_encoder.apply_tamper(false_payload_base, self.config.waf_evasion_level)

            # 我們只取第一個 tamper 結果來測試，避免請求爆炸
            true_payload = true_payloads[-1] # 取最後一個通常是混淆最強的
            false_payload = false_payloads[-1]

            try:
                # 使用 PayloadEncoder 構建請求參數
                # 注意：BaseDetector.detect 接收 params，但 PayloadEncoder 需要 task。
                # 我們在 SmartDetectionManager 中已經重新實例化了 PayloadEncoder with task。
                # 所以 self.payload_encoder 已經綁定了 task。
                # 我們只需要傳遞 payload 給 encode()。

                true_encoded = self.payload_encoder.encode(true_payload)
                false_encoded = self.payload_encoder.encode(false_payload)

                # 並行發送
                results_tuple = await asyncio.gather(
                    self._send_request(true_encoded.url, true_encoded.method, data=true_encoded.request_kwargs.get("data"), params=true_encoded.request_kwargs.get("params")),
                    self._send_request(false_encoded.url, false_encoded.method, data=false_encoded.request_kwargs.get("data"), params=false_encoded.request_kwargs.get("params")),
                    return_exceptions=True
                )

                (true_text, true_status), (false_text, false_status) = results_tuple

                if true_status == 0 or false_status == 0:
                    continue

                # 分析
                if self._analyze_boolean_responses(baseline_text, true_text, false_text, true_status, false_status):
                    result = self._build_detection_result(
                         true_payload, false_payload,
                         true_status, len(true_text),
                         false_status, len(false_text),
                         target_url, method, "unknown_param" # Param is inside task...
                    )
                    results.append(result)
                    logger.info(f"Boolean SQLi detected: {true_payload}")

            except Exception as e:
                logger.warning(f"Boolean check failed: {e}")
                continue

        return results

    def _analyze_boolean_responses(
        self,
        baseline_text: str,
        true_text: str,
        false_text: str,
        true_status: int,
        false_status: int
    ) -> bool:
        """分析布林回應 (Fuzzy Logic)"""
        # 1. 狀態碼
        if true_status != false_status:
            return True

        # 2. 長度差異
        true_len = len(true_text)
        false_len = len(false_text)
        if abs(true_len - false_len) > 50: # 簡單閾值
            return True

        # 3. Fuzzy Similarity (新功能)
        # 比較 True 與 False 的相似度
        import difflib
        similarity = difflib.SequenceMatcher(None, true_text, false_text).ratio()

        # 比較 True 與 Baseline 的相似度
        true_base_sim = difflib.SequenceMatcher(None, true_text, baseline_text).ratio()
        false_base_sim = difflib.SequenceMatcher(None, false_text, baseline_text).ratio()

        # 邏輯：True 頁面應該像 Baseline，False 頁面應該不像 Baseline，或者 True 與 False 顯著不同
        if similarity < self.config.fuzzy_similarity_threshold:
             # 如果 True 和 False 差異大 (相似度低於閾值)
             return True

        # 或者 True 很像 Baseline, 但 False 不像
        if true_base_sim > 0.95 and false_base_sim < 0.90:
             return True

        return False

    def _build_detection_result(self, true_payload, false_payload, true_status, true_len, false_status, false_len, url, method, param) -> DetectionResult:
        from aiva_common.enums import Confidence, Severity, VulnerabilityType
        from aiva_common.schemas import (
            FindingEvidence,
            FindingImpact,
            FindingRecommendation,
            FindingTarget,
            Vulnerability,
        )

        return DetectionResult(
            is_vulnerable=True,
            vulnerability=Vulnerability(
                name=VulnerabilityType.SQLI,
                severity=Severity.HIGH,
                confidence=Confidence.FIRM,
            ),
            evidence=FindingEvidence(
                payload=f"True: {true_payload}, False: {false_payload}",
                request=f"Method: {method}",
                response=f"True len: {true_len}, False len: {false_len}",
                proof="Boolean response difference detected"
            ),
            impact=FindingImpact(description="SQL Injection"),
            recommendation=FindingRecommendation(fix="Use prepared statements"),
            target=FindingTarget(url=url, method=method, parameter=param),
            detection_method="boolean_based",
            payload_used=true_payload,
            confidence_score=0.9
        )

