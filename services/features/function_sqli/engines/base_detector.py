"""
Base SQL Injection Detector Interface
所有具體檢測引擎 (Boolean, Error, Time, Union) 的基類
"""

from abc import ABC, abstractmethod
import logging
from typing import Any

import aiohttp

from ..config import SqliConfig
from ..detection_models import DetectionResult
from ..payload_wrapper_encoder import PayloadWrapperEncoder

logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    """SQL 注入檢測器基類"""

    def __init__(self, session: aiohttp.ClientSession, config: SqliConfig, payload_encoder: PayloadWrapperEncoder):
        self.session = session
        self.config = config
        self.payload_encoder = payload_encoder

    @abstractmethod
    async def detect(self, target_url: str, params: dict[str, str], method: str = "GET") -> list[DetectionResult]:
        """
        執行檢測
        :param target_url: 目標 URL
        :param params: 參數字典
        :param method: 請求方法
        :return: 檢測結果列表
        """

    async def _send_request(self, url: str, method: str, data: dict | None = None, params: dict | None = None) -> Any:
        """
        發送 HTTP 請求 (統一處理錯誤與重試)
        """
        try:
            if method.upper() == "GET":
                async with self.session.get(url, params=params, timeout=self.config.timeout_seconds) as resp:
                    return await resp.text(), resp.status
            elif method.upper() == "POST":
                async with self.session.post(url, data=data, timeout=self.config.timeout_seconds) as resp:
                    return await resp.text(), resp.status
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            return None, 0
