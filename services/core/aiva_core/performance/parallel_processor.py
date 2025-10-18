"""
並行訊息處理器模組
拆分自 optimized_core.py 的並行訊息處理部分
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class ParallelMessageProcessor:
    """並行訊息處理器 - 替代原本的單線程處理"""

    def __init__(self, max_concurrent: int = 20, batch_size: int = 50):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.message_buffer = []
        self.processing_stats = {
            "processed": 0,
            "errors": 0,
            "avg_duration": 0.0
        }

    async def process_messages(self, broker, topic: str, handler: Callable):
        """並行處理訊息流"""
        async for mqmsg in broker.subscribe(topic):
            self.message_buffer.append(mqmsg)

            # 當累積到批次大小或超時時處理
            if len(self.message_buffer) >= self.batch_size:
                batch = self.message_buffer[:self.batch_size]
                self.message_buffer = self.message_buffer[self.batch_size:]

                # 並行處理批次
                asyncio.create_task(self._process_batch(batch, handler))

    async def _process_batch(self, messages: list, handler: Callable):
        """並行處理一個批次的訊息"""
        tasks = [
            self._process_single_message(msg, handler)
            for msg in messages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 統計處理結果
        for result in results:
            if isinstance(result, Exception):
                self.processing_stats["errors"] += 1
            else:
                self.processing_stats["processed"] += 1

    async def _process_single_message(self, message, handler: Callable):
        """處理單個訊息（帶信號量控制）"""
        async with self.semaphore:
            start_time = time.time()
            try:
                result = await handler(message)
                duration = time.time() - start_time

                # 更新平均處理時間
                self._update_avg_duration(duration)
                return result

            except Exception as e:
                print(f"Message processing error: {e}")
                raise

    def _update_avg_duration(self, duration: float):
        """更新平均處理時間"""
        count = self.processing_stats["processed"]
        current_avg = self.processing_stats["avg_duration"]

        # 計算新的平均值
        new_avg = (current_avg * (count - 1) + duration) / count
        self.processing_stats["avg_duration"] = new_avg