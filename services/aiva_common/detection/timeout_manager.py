"""
Adaptive Timeout Manager

自適應超時管理器，根據歷史響應時間動態調整超時閾值
基於 asyncio.timeout 和統計分析實現
"""

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimeoutConfig:
    """超時配置"""

    initial_timeout: float = 30.0  # 初始超時(秒)
    min_timeout: float = 5.0  # 最小超時
    max_timeout: float = 300.0  # 最大超時
    percentile: float = 95.0  # 使用P95作為超時基準
    history_size: int = 100  # 保留最近N次請求的歷史
    adaptation_factor: float = 1.2  # 超時調整係數


class AdaptiveTimeoutManager:
    """
    自適應超時管理器
    
    根據歷史響應時間動態調整超時閾值，避免：
    - 超時設置過短導致頻繁失敗
    - 超時設置過長浪費等待時間
    
    基於 Python asyncio.timeout 最佳實踐實現
    參考: https://docs.python.org/3/library/asyncio-task.html#timeouts
    """

    def __init__(self, config: TimeoutConfig | None = None):
        """
        初始化自適應超時管理器
        
        Args:
            config: 超時配置，如不提供則使用默認配置
        """
        self.config = config or TimeoutConfig()
        self._response_times: deque[float] = deque(maxlen=self.config.history_size)
        self._current_timeout = self.config.initial_timeout
        self._request_count = 0
        self._timeout_count = 0

        logger.info(
            f"AdaptiveTimeoutManager initialized: "
            f"initial={self.config.initial_timeout}s, "
            f"min={self.config.min_timeout}s, "
            f"max={self.config.max_timeout}s"
        )

    def get_timeout(self) -> float:
        """
        獲取當前超時值
        
        Returns:
            當前建議的超時時間(秒)
        """
        return self._current_timeout

    def record_response_time(self, response_time: float) -> None:
        """
        記錄響應時間並更新超時閾值
        
        Args:
            response_time: 響應時間(秒)
        """
        self._response_times.append(response_time)
        self._request_count += 1

        # 每10次請求重新計算超時
        if self._request_count % 10 == 0:
            self._update_timeout()

    def record_timeout(self) -> None:
        """記錄超時事件"""
        self._timeout_count += 1

        # 超時率過高時增加超時時間
        if self._request_count > 0:
            timeout_rate = self._timeout_count / self._request_count
            if timeout_rate > 0.1:  # 超過10%超時率
                self._current_timeout = min(
                    self._current_timeout * 1.5,
                    self.config.max_timeout
                )
                logger.warning(
                    f"High timeout rate ({timeout_rate:.1%}), "
                    f"increasing timeout to {self._current_timeout:.1f}s"
                )

    def _update_timeout(self) -> None:
        """根據歷史數據更新超時閾值"""
        if len(self._response_times) < 10:
            return  # 數據不足，保持當前值

        try:
            # 計算P95響應時間
            p95_time = statistics.quantiles(
                self._response_times,
                n=20,  # 5%分位數
                method='inclusive'
            )[-1]  # 取95分位

            # 應用調整係數
            new_timeout = p95_time * self.config.adaptation_factor

            # 限制在合理範圍內
            new_timeout = max(
                self.config.min_timeout,
                min(new_timeout, self.config.max_timeout)
            )

            # 避免頻繁小幅調整
            if abs(new_timeout - self._current_timeout) > 1.0:
                old_timeout = self._current_timeout
                self._current_timeout = new_timeout
                logger.info(
                    f"Timeout adapted: {old_timeout:.1f}s -> {new_timeout:.1f}s "
                    f"(P95={p95_time:.1f}s, samples={len(self._response_times)})"
                )

        except Exception as e:
            logger.error(f"Failed to update timeout: {e}", exc_info=True)

    async def execute_with_timeout(self, coro, task_name: str = "task"):
        """
        使用自適應超時執行協程
        
        Args:
            coro: 要執行的協程
            task_name: 任務名稱(用於日誌)
        
        Returns:
            協程執行結果
        
        Raises:
            TimeoutError: 執行超時
        """
        start_time = time.perf_counter()
        timeout_value = self.get_timeout()

        try:
            async with asyncio.timeout(timeout_value):
                result = await coro

            # 記錄成功的響應時間
            elapsed = time.perf_counter() - start_time
            self.record_response_time(elapsed)

            return result

        except TimeoutError:
            self.record_timeout()
            logger.warning(
                f"{task_name} timed out after {timeout_value:.1f}s "
                f"(timeout_rate={self.get_timeout_rate():.1%})"
            )
            raise

    def get_timeout_rate(self) -> float:
        """
        獲取超時率
        
        Returns:
            超時率(0.0-1.0)
        """
        if self._request_count == 0:
            return 0.0
        return self._timeout_count / self._request_count

    def get_stats(self) -> dict:
        """
        獲取統計信息
        
        Returns:
            統計信息字典
        """
        stats = {
            "current_timeout": self._current_timeout,
            "request_count": self._request_count,
            "timeout_count": self._timeout_count,
            "timeout_rate": self.get_timeout_rate(),
            "history_size": len(self._response_times),
        }

        if self._response_times:
            stats.update({
                "avg_response_time": statistics.mean(self._response_times),
                "min_response_time": min(self._response_times),
                "max_response_time": max(self._response_times),
            })

        return stats

    def reset(self) -> None:
        """重置統計數據"""
        self._response_times.clear()
        self._current_timeout = self.config.initial_timeout
        self._request_count = 0
        self._timeout_count = 0
        logger.info("AdaptiveTimeoutManager reset")
