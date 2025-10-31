"""Advanced Memory Manager for AIVA AI Core
高級記憶體管理器 - 用於 AI 核心的性能優化

提供智能快取、記憶體池管理、批次處理優化等功能
"""

import gc
import hashlib
import logging
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class AdvancedMemoryManager:
    """高級記憶體管理器，提供智能快取和記憶體優化功能."""

    def __init__(self, max_cache_size: int = 1000, enable_gc_optimization: bool = True):
        """初始化記憶體管理器.

        Args:
            max_cache_size: 最大快取項目數
            enable_gc_optimization: 是否啟用垃圾回收優化
        """
        self.max_cache_size = max_cache_size
        self.enable_gc_optimization = enable_gc_optimization

        # 預測結果快取
        self.prediction_cache: dict[str, tuple[np.ndarray, float]] = {}

        # 快取統計
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        # 記憶體使用追蹤
        self.memory_usage_history: list[dict[str, Any]] = []

        # LRU 快取實現
        self.access_order: list[str] = []

        logger.info(
            f"AdvancedMemoryManager 初始化完成 (max_cache_size={max_cache_size})"
        )

    def _generate_cache_key(self, input_data: Any) -> str:
        """生成快取鍵值.

        Args:
            input_data: 輸入資料

        Returns:
            快取鍵值字串
        """
        if isinstance(input_data, np.ndarray):
            # 對 numpy 陣列使用內容雜湊
            content = input_data.tobytes()
        else:
            # 對其他類型轉換為字串
            content = str(input_data).encode("utf-8")

        return hashlib.md5(content).hexdigest()

    def get_cached_prediction(self, input_data: Any) -> np.ndarray | None:
        """獲取快取的預測結果.

        Args:
            input_data: 輸入資料

        Returns:
            快取的預測結果，如果不存在則返回 None
        """
        cache_key = self._generate_cache_key(input_data)

        if cache_key in self.prediction_cache:
            # 快取命中
            result, timestamp = self.prediction_cache[cache_key]
            self.cache_hits += 1

            # 更新 LRU 順序
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)

            logger.debug(f"快取命中: {cache_key[:8]}...")
            return result.copy()  # 返回副本避免意外修改
        else:
            # 快取未命中
            self.cache_misses += 1
            logger.debug(f"快取未命中: {cache_key[:8]}...")
            return None

    def cache_prediction(self, input_data: Any, prediction: np.ndarray) -> None:
        """快取預測結果.

        Args:
            input_data: 輸入資料
            prediction: 預測結果
        """
        cache_key = self._generate_cache_key(input_data)
        current_time = time.time()

        # 檢查是否需要清理快取
        if len(self.prediction_cache) >= self.max_cache_size:
            self._evict_oldest_cache_entry()

        # 存儲快取項目
        self.prediction_cache[cache_key] = (prediction.copy(), current_time)

        # 更新 LRU 順序
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

        logger.debug(f"快取預測結果: {cache_key[:8]}...")

    def _evict_oldest_cache_entry(self) -> None:
        """清除最舊的快取項目 (LRU 策略)."""
        if not self.access_order:
            return

        oldest_key = self.access_order.pop(0)
        if oldest_key in self.prediction_cache:
            del self.prediction_cache[oldest_key]
            self.cache_evictions += 1
            logger.debug(f"清除快取項目: {oldest_key[:8]}...")

    def clear_cache(self) -> None:
        """清空所有快取."""
        self.prediction_cache.clear()
        self.access_order.clear()
        logger.info("快取已清空")

    def get_cache_stats(self) -> dict[str, Any]:
        """獲取快取統計信息.

        Returns:
            快取統計數據
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self.prediction_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def optimize_memory(self, force_gc: bool = False) -> dict[str, Any]:
        """優化記憶體使用.

        Args:
            force_gc: 是否強制執行垃圾回收

        Returns:
            優化統計信息
        """
        stats_before = self._get_memory_usage()

        # 清理過期的快取項目 (超過 1 小時)
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp) in self.prediction_cache.items():
            if (current_time - timestamp) > 3600:  # 1 小時
                expired_keys.append(key)

        for key in expired_keys:
            del self.prediction_cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

        # 執行垃圾回收
        if self.enable_gc_optimization or force_gc:
            gc.collect()

        stats_after = self._get_memory_usage()

        optimization_stats = {
            "expired_cache_removed": len(expired_keys),
            "memory_before_mb": stats_before.get("memory_mb", 0),
            "memory_after_mb": stats_after.get("memory_mb", 0),
            "memory_saved_mb": stats_before.get("memory_mb", 0)
            - stats_after.get("memory_mb", 0),
        }

        logger.info(f"記憶體優化完成: 移除 {len(expired_keys)} 個過期快取項目")

        return optimization_stats

    def _get_memory_usage(self) -> dict[str, Any]:
        """獲取當前記憶體使用情況.

        Returns:
            記憶體使用統計
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "memory_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": process.memory_percent(),
            }
        except ImportError:
            # 如果沒有 psutil，使用簡化版本

            cache_size_mb = len(self.prediction_cache) * 0.1  # 估算快取記憶體使用

            return {
                "memory_mb": cache_size_mb,
                "memory_percent": 0,
                "note": "estimated_without_psutil",
            }
        except Exception as e:
            logger.warning(f"記憶體使用檢測失敗: {e}")
            return {"memory_mb": 0, "memory_percent": 0, "error": str(e)}

    def record_memory_usage(self) -> None:
        """記錄當前記憶體使用情況."""
        stats = self._get_memory_usage()
        stats["timestamp"] = time.time()
        stats["cache_size"] = len(self.prediction_cache)

        self.memory_usage_history.append(stats)

        # 保持歷史記錄在合理範圍內
        if len(self.memory_usage_history) > 1000:
            self.memory_usage_history = self.memory_usage_history[-500:]

    def get_memory_trends(self) -> dict[str, Any]:
        """獲取記憶體使用趨勢分析.

        Returns:
            記憶體趨勢統計
        """
        if len(self.memory_usage_history) < 2:
            return {"trend": "insufficient_data"}

        recent_records = self.memory_usage_history[-10:]  # 最近 10 次記錄

        memory_values = [r.get("memory_mb", 0) for r in recent_records]
        cache_sizes = [r.get("cache_size", 0) for r in recent_records]

        return {
            "average_memory_mb": np.mean(memory_values),
            "memory_trend": (
                "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
            ),
            "average_cache_size": np.mean(cache_sizes),
            "peak_memory_mb": max(memory_values),
            "min_memory_mb": min(memory_values),
            "memory_variance": np.var(memory_values),
        }


class BatchProcessor:
    """批次處理器，用於優化大量資料的處理效率."""

    def __init__(
        self,
        batch_size: int = 32,
        memory_manager: AdvancedMemoryManager | None = None,
    ):
        """初始化批次處理器.

        Args:
            batch_size: 批次大小
            memory_manager: 記憶體管理器實例
        """
        self.batch_size = batch_size
        self.memory_manager = memory_manager or AdvancedMemoryManager()
        self.processing_stats: dict[str, float] = {
            "total_items_processed": 0.0,
            "total_batches": 0.0,
            "average_batch_time": 0.0,
            "cache_utilization": 0.0,
        }

        logger.info(f"BatchProcessor 初始化完成 (batch_size={batch_size})")

    def process_batch(self, items: list[Any], process_func: Callable[[Any], Any]) -> list[Any]:
        """處理一個批次的項目.

        Args:
            items: 要處理的項目列表
            process_func: 處理函數

        Returns:
            處理結果列表
        """
        results = []
        cache_hits_in_batch = 0

        batch_start_time = time.time()

        for item in items:
            # 嘗試從快取獲取結果
            cached_result = self.memory_manager.get_cached_prediction(item)

            if cached_result is not None:
                results.append(cached_result)
                cache_hits_in_batch += 1
            else:
                # 執行實際處理
                result = process_func(item)
                results.append(result)

                # 快取結果
                if isinstance(result, np.ndarray):
                    self.memory_manager.cache_prediction(item, result)

        batch_time = time.time() - batch_start_time

        # 更新統計
        self.processing_stats["total_items_processed"] += len(items)
        self.processing_stats["total_batches"] += 1
        self.processing_stats["average_batch_time"] = (
            self.processing_stats["average_batch_time"]
            * (self.processing_stats["total_batches"] - 1)
            + batch_time
        ) / self.processing_stats["total_batches"]
        self.processing_stats["cache_utilization"] = cache_hits_in_batch / len(items)

        logger.debug(
            f"批次處理完成: {len(items)} 項目, {cache_hits_in_batch} 快取命中, {batch_time:.3f}s"
        )

        return results

    def process_large_dataset(
        self, dataset: list[Any], process_func: Callable[[Any], Any]
    ) -> list[Any]:
        """處理大型資料集.

        Args:
            dataset: 資料集
            process_func: 處理函數

        Returns:
            處理結果列表
        """
        all_results = []
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        logger.info(f"開始處理大型資料集: {len(dataset)} 項目, {total_batches} 批次")

        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i : i + self.batch_size]
            batch_results = self.process_batch(batch, process_func)
            all_results.extend(batch_results)

            # 定期優化記憶體
            if (i // self.batch_size + 1) % 10 == 0:
                self.memory_manager.optimize_memory()

        logger.info(f"大型資料集處理完成: {len(all_results)} 結果")

        return all_results

    def get_processing_stats(self) -> dict[str, Any]:
        """獲取處理統計信息.

        Returns:
            處理統計數據
        """
        stats = self.processing_stats.copy()
        stats.update(self.memory_manager.get_cache_stats())
        return stats
