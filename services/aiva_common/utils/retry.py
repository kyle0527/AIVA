"""
AIVA Common - 重試工具模組

提供異步和同步的重試機制，符合 AIVA 開發規範
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass
class RetryConfig:
    """重試配置"""

    max_attempts: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    exceptions: tuple = (Exception,)

    def calculate_delay(self, attempt: int) -> float:
        """計算重試延遲時間"""
        delay = self.delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)


def retry_sync(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """
    同步重試裝飾器

    Args:
        max_attempts: 最大重試次數
        delay: 初始延遲時間（秒）
        backoff_factor: 退避係數
        max_delay: 最大延遲時間（秒）
        exceptions: 需要重試的異常類型
    """
    config = RetryConfig(max_attempts, delay, backoff_factor, max_delay, exceptions)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts"
                        )
                        raise

                    retry_delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt}/{config.max_attempts}, "
                        f"retrying in {retry_delay:.1f}s: {e}"
                    )
                    time.sleep(retry_delay)

            return None  # 不應該到達這裡

        return wrapper

    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """
    異步重試裝飾器

    Args:
        max_attempts: 最大重試次數
        delay: 初始延遲時間（秒）
        backoff_factor: 退避係數
        max_delay: 最大延遲時間（秒）
        exceptions: 需要重試的異常類型
    """
    config = RetryConfig(max_attempts, delay, backoff_factor, max_delay, exceptions)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {attempt} attempts"
                        )
                        raise

                    retry_delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt}/{config.max_attempts}, "
                        f"retrying in {retry_delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)

            return None  # 不應該到達這裡

        return wrapper

    return decorator


class RetryManager:
    """重試管理器 - 用於手動控制重試邏輯"""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """執行異步函數並重試"""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except self.config.exceptions as e:
                if attempt == self.config.max_attempts:
                    self.logger.error(f"Function failed after {attempt} attempts")
                    raise

                retry_delay = self.config.calculate_delay(attempt)
                self.logger.warning(
                    f"Function failed on attempt {attempt}/{self.config.max_attempts}, "
                    f"retrying in {retry_delay:.1f}s: {e}"
                )
                await asyncio.sleep(retry_delay)

    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """執行同步函數並重試"""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.config.exceptions as e:
                if attempt == self.config.max_attempts:
                    self.logger.error(f"Function failed after {attempt} attempts")
                    raise

                retry_delay = self.config.calculate_delay(attempt)
                self.logger.warning(
                    f"Function failed on attempt {attempt}/{self.config.max_attempts}, "
                    f"retrying in {retry_delay:.1f}s: {e}"
                )
                time.sleep(retry_delay)


# 預設配置的便捷函數
def create_default_retry_config(
    max_attempts: int = 3, delay: float = 1.0, exceptions: tuple | None = None
) -> RetryConfig:
    """創建默認重試配置"""
    if exceptions is None:
        exceptions = (Exception,)

    return RetryConfig(max_attempts=max_attempts, delay=delay, exceptions=exceptions)


def create_network_retry_config() -> RetryConfig:
    """創建網絡操作專用重試配置"""
    import socket

    network_exceptions = (
        ConnectionError,
        TimeoutError,
        socket.timeout,
        socket.error,
    )

    return RetryConfig(
        max_attempts=5,
        delay=2.0,
        backoff_factor=1.5,
        max_delay=30.0,
        exceptions=network_exceptions,
    )


def create_database_retry_config() -> RetryConfig:
    """創建數據庫操作專用重試配置"""
    return RetryConfig(
        max_attempts=3,
        delay=0.5,
        backoff_factor=2.0,
        max_delay=10.0,
        exceptions=(Exception,),  # 可以根據具體數據庫驅動調整
    )


# 模組導出
__all__ = [
    "RetryConfig",
    "retry_sync",
    "retry_async",
    "RetryManager",
    "create_default_retry_config",
    "create_network_retry_config",
    "create_database_retry_config",
]


# 使用範例
if __name__ == "__main__":
    import asyncio

    # 同步重試範例
    @retry_sync(max_attempts=3, delay=1.0)
    def unreliable_function():
        import random

        if random.random() < 0.7:
            raise ConnectionError("Network error")
        return "Success!"

    # 異步重試範例
    @retry_async(max_attempts=3, delay=1.0)
    async def unreliable_async_function():
        import random

        if random.random() < 0.7:
            raise ConnectionError("Network error")
        return "Async Success!"

    # 測試
    async def test():
        try:
            result = unreliable_function()
            print(f"Sync result: {result}")
        except Exception as e:
            print(f"Sync failed: {e}")

        try:
            result = await unreliable_async_function()
            print(f"Async result: {result}")
        except Exception as e:
            print(f"Async failed: {e}")

    # asyncio.run(test())
