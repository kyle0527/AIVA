"""
Smart Rate Limiter

智能速率限制器，基於 Token Bucket 算法實現
支持異步操作和動態速率調整

參考最佳實踐:
- aiolimiter: https://github.com/mjpieters/aiolimiter
- Token Bucket 算法
"""

import asyncio
import time
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """速率限制配置"""

    requests_per_second: float = 10.0  # 每秒請求數
    burst_size: int = 20  # 突發容量
    enable_adaptive: bool = True  # 啟用自適應調整
    min_rate: float = 1.0  # 最小速率
    max_rate: float = 100.0  # 最大速率


class SmartRateLimiter:
    """
    智能速率限制器
    
    基於 Token Bucket (令牌桶) 算法實現:
    - 固定速率產生 token
    - 請求消耗 token
    - token 不足時等待
    - 支持突發流量 (burst)
    
    自適應特性:
    - 根據錯誤率動態調整速率
    - 429/503 響應時降低速率
    - 正常響應時逐步恢復速率
    
    參考: https://github.com/mjpieters/aiolimiter
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """
        初始化速率限制器
        
        Args:
            config: 速率限制配置
        """
        self.config = config or RateLimitConfig()

        # Token Bucket 狀態
        self._tokens = float(self.config.burst_size)
        self._max_tokens = float(self.config.burst_size)
        self._rate = self.config.requests_per_second
        self._last_refill = time.monotonic()

        # 統計信息
        self._total_requests = 0
        self._rate_limited_count = 0
        self._error_429_count = 0

        # 異步鎖
        self._lock = asyncio.Lock()

        logger.info(
            f"SmartRateLimiter initialized: "
            f"rate={self._rate}/s, burst={self.config.burst_size}"
        )

    async def acquire(self, tokens: int = 1) -> float:
        """
        獲取 token（異步等待直到可用）
        
        Args:
            tokens: 需要的 token 數量
        
        Returns:
            等待時間(秒)
        """
        async with self._lock:
            self._refill_tokens()

            # 如果 token 不足，計算需要等待的時間
            if self._tokens < tokens:
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._rate

                self._rate_limited_count += 1
                logger.debug(
                    f"Rate limit hit, waiting {wait_time:.2f}s "
                    f"(tokens={self._tokens:.1f}/{self._max_tokens}, "
                    f"rate={self._rate}/s)"
                )

                await asyncio.sleep(wait_time)
                self._refill_tokens()

            # 消耗 token
            self._tokens -= tokens
            self._total_requests += 1

            return 0.0 if self._tokens >= 0 else abs(self._tokens) / self._rate

    def _refill_tokens(self) -> None:
        """根據時間流逝補充 token"""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # 按速率補充 token
        new_tokens = elapsed * self._rate
        self._tokens = min(self._tokens + new_tokens, self._max_tokens)
        self._last_refill = now

    def record_429_error(self) -> None:
        """
        記錄 429 (Too Many Requests) 錯誤
        
        自適應降低速率
        """
        self._error_429_count += 1

        if self.config.enable_adaptive:
            old_rate = self._rate
            # 降低速率到 70%
            self._rate = max(
                old_rate * 0.7,
                self.config.min_rate
            )

            logger.warning(
                f"429 error detected, reducing rate: "
                f"{old_rate:.1f}/s -> {self._rate:.1f}/s"
            )

    def record_success(self) -> None:
        """
        記錄成功請求
        
        自適應恢復速率
        """
        if self.config.enable_adaptive and self._rate < self.config.requests_per_second:
            # 每100次成功請求恢復5%速率
            if self._total_requests % 100 == 0:
                old_rate = self._rate
                self._rate = min(
                    old_rate * 1.05,
                    self.config.requests_per_second,
                    self.config.max_rate
                )

                if abs(self._rate - old_rate) > 0.1:
                    logger.info(
                        f"Rate recovering: {old_rate:.1f}/s -> {self._rate:.1f}/s"
                    )

    async def execute_with_rate_limit(self, coro, tokens: int = 1):
        """
        在速率限制下執行協程
        
        Args:
            coro: 要執行的協程
            tokens: 消耗的 token 數量
        
        Returns:
            協程執行結果
        """
        await self.acquire(tokens)
        return await coro

    def get_current_rate(self) -> float:
        """獲取當前速率"""
        return self._rate

    def get_available_tokens(self) -> float:
        """獲取當前可用 token 數"""
        self._refill_tokens()
        return self._tokens

    def get_stats(self) -> dict:
        """
        獲取統計信息
        
        Returns:
            統計信息字典
        """
        return {
            "current_rate": self._rate,
            "configured_rate": self.config.requests_per_second,
            "available_tokens": self.get_available_tokens(),
            "max_tokens": self._max_tokens,
            "total_requests": self._total_requests,
            "rate_limited_count": self._rate_limited_count,
            "error_429_count": self._error_429_count,
            "rate_limit_percentage": (
                self._rate_limited_count / self._total_requests * 100
                if self._total_requests > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """重置速率限制器"""
        self._tokens = float(self.config.burst_size)
        self._rate = self.config.requests_per_second
        self._last_refill = time.monotonic()
        self._total_requests = 0
        self._rate_limited_count = 0
        self._error_429_count = 0
        logger.info("SmartRateLimiter reset")
