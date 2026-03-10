"""
Smart Detection Manager

智能檢測管理器，用於協調各種檢測功能。
負責執行智慧掃描流程：偵察 (Stability, Fingerprint) -> 策略調整 -> 執行檢測
"""

import asyncio
import logging
import difflib
from typing import Dict, Any, List, Optional, Tuple, Type
import aiohttp

from aiva_common.utils import get_logger
from aiva_common.schemas import FunctionTaskPayload

from .config import SqliConfig
from .payload_wrapper_encoder import PayloadWrapperEncoder
from .backend_db_fingerprinter import BackendDbFingerprinter
from .engines.base_detector import BaseDetector, DetectionResult
from .engines.boolean_detection_engine import BooleanDetectionEngine
from .engines.error_detection_engine import ErrorDetectionEngine
from .engines.time_detection_engine import TimeDetectionEngine
from .engines.union_detection_engine import UnionDetectionEngine
from .engines.oob_detection_engine import OOBDetectionEngine

logger = get_logger(__name__)


class SmartDetectionManager:
    """智能檢測管理器

    協調和管理各種安全檢測功能的核心組件。
    """

    def __init__(self, config: Optional[SqliConfig] = None):
        """初始化智能檢測管理器"""
        self.config = config or SqliConfig()
        self.fingerprinter = BackendDbFingerprinter()
        # Store detector classes, not instances, to instantiate with session per scan
        self.detector_classes: List[Type[BaseDetector]] = [
            BooleanDetectionEngine,
            ErrorDetectionEngine,
            TimeDetectionEngine,
            UnionDetectionEngine,
            OOBDetectionEngine
        ]
        self.active_scans = {}
        logger.info("SmartDetectionManager initialized with default engines")

    def register_detector(self, detector_cls: Type[BaseDetector]) -> None:
        """註冊檢測器類別"""
        if detector_cls not in self.detector_classes:
            self.detector_classes.append(detector_cls)
            logger.info(f"Registered detector class: {detector_cls.__name__}")

    async def scan_target(self, task: FunctionTaskPayload) -> List[DetectionResult]:
        """
        執行完整的智慧掃描流程
        """
        results = []
        target_url = str(task.target.url)
        params = {} # Extract from task if needed, but Encoder handles it

        logger.info(f"Starting Smart SQLi Scan for {target_url}")

        async with aiohttp.ClientSession() as session:

            # 1. 穩定性分析
            is_stable, stability_score = await self._check_stability(session, target_url, task)
            if not is_stable:
                logger.warning(f"Target {target_url} is unstable (score: {stability_score:.2f}). Adjusting thresholds.")
                # 若不穩定，可以動態調整 config (需複製 config 以免影響全局)
                # For now using the global config but logging the instability

            # 2. 指紋識別
            db_type, version = await self._fingerprint(session, target_url, task)
            if db_type:
                logger.info(f"Target DB identified: {db_type} {version}")
                # Future: Filter payloads based on DB type

            # 3. 執行檢測器
            # Instantiate detectors with the current session
            for DetectorCls in self.detector_classes:
                try:
                    # Initialize detector with current session, config, and task-specific encoder
                    detector = DetectorCls(
                        session=session,
                        config=self.config,
                        payload_encoder=PayloadWrapperEncoder(task)
                    )

                    # Pass context if supported
                    if hasattr(detector, 'stability_score'):
                        detector.stability_score = stability_score

                    # Execute detection
                    logger.debug(f"Running {DetectorCls.__name__}...")
                    det_results = await detector.detect(target_url, params, task.target.method or "GET")
                    results.extend(det_results)

                except Exception as e:
                    logger.error(f"Error running detector {DetectorCls.__name__}: {e}")
                    continue

        return results

    async def _check_stability(self, session: aiohttp.ClientSession, url: str, task: FunctionTaskPayload) -> Tuple[bool, float]:
        """檢查頁面穩定性"""
        encoder = PayloadWrapperEncoder(task)
        encoded = encoder.encode("") # 基準請求

        samples = []
        for _ in range(3):
            try:
                if encoded.method == "GET":
                    async with session.get(encoded.url, **encoded.request_kwargs) as resp:
                        text = await resp.text()
                        samples.append(text)
                else:
                    async with session.post(encoded.url, **encoded.request_kwargs) as resp:
                        text = await resp.text()
                        samples.append(text)
            except Exception:
                pass
            await asyncio.sleep(self.config.rate_limit_delay)

        if len(samples) < 2:
            return False, 0.0

        # 計算相似度
        similarities = []
        base = samples[0]
        for i in range(1, len(samples)):
            sim = difflib.SequenceMatcher(None, base, samples[i]).ratio()
            similarities.append(sim)

        avg_stability = sum(similarities) / len(similarities)
        is_stable = avg_stability >= self.config.stability_threshold

        return is_stable, avg_stability

    async def _fingerprint(self, session: aiohttp.ClientSession, url: str, task: FunctionTaskPayload) -> Tuple[Optional[str], Optional[str]]:
        """識別資料庫"""
        # 構造報錯請求
        encoder = PayloadWrapperEncoder(task)
        # 簡單在參數後加單引號
        exploit_payload = "'"
        encoded = encoder.encode(exploit_payload)

        try:
            # 我們需要兼容 BackendDbFingerprinter 預期的 httpx.Response
            # 但這裡是 aiohttp。
            # 解決方案：簡單封裝一個類似物件
            if encoded.method == "GET":
                async with session.get(encoded.url, **encoded.request_kwargs) as resp:
                    text = await resp.text()
                    return self._process_fingerprint(resp, text)
            else:
                async with session.post(encoded.url, **encoded.request_kwargs) as resp:
                    text = await resp.text()
                    return self._process_fingerprint(resp, text)
        except Exception as e:
            logger.debug(f"Fingerprinting failed: {e}")
            return None, None

    def _process_fingerprint(self, aiohttp_resp, text):
        # 建立 Mock 物件適配 httpx.Response 介面供 Fingerprinter 使用
        class MockResponse:
            def __init__(self, text, code, headers):
                self.text = text
                self.status_code = code
                self.headers = headers
                self.cookies = {} # Add cookies if needed

        mock_resp = MockResponse(text, aiohttp_resp.status, aiohttp_resp.headers)
        return self.fingerprinter.fingerprint(mock_resp)

    # 保留舊方法以兼容現有代碼 (如果有)
    def start_detection(self, target: str, config: Dict[str, Any]) -> str:
        session_id = f"scan_{len(self.active_scans)}"
        self.active_scans[session_id] = {"target": target, "status": "running"}
        return session_id

    def get_detection_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.active_scans.get(session_id)

    def stop_detection(self, session_id: str) -> bool:
        if session_id in self.active_scans:
            self.active_scans[session_id]["status"] = "stopped"
            return True
        return False