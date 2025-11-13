"""統一記憶體管理器 - 整合AI專用與通用記憶體管理功能

將原本的 ai_engine/memory_manager.py 和 performance/memory_manager.py 
整合為單一、功能完整的記憶體管理系統
"""

import asyncio
from contextlib import asynccontextmanager
import gc
import hashlib
import logging
import time
from typing import Any, Callable
import weakref

# 統一的依賴管理
try:
    import numpy as np
    HAS_NUMPY = True
    NDArray = np.ndarray
except ImportError:
    HAS_NUMPY = False
    np = None
    NDArray = None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class UnifiedMemoryManager:
    """統一記憶體管理器 - 整合AI快取與系統記憶體管理"""
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 gc_threshold_mb: int = 512,
                 enable_monitoring: bool = True,
                 batch_size: int = 32):
        """初始化統一記憶體管理器
        
        Args:
            max_cache_size: AI預測快取最大項目數
            gc_threshold_mb: 垃圾回收觸發閾值(MB)
            enable_monitoring: 是否啟用自動監控
            batch_size: 批次處理預設大小
        """
        # AI快取系統
        self.max_cache_size = max_cache_size
        self.prediction_cache: dict[str, tuple[Any, float]] = {}
        self.access_order: list[str] = []
        
        # 系統記憶體管理
        self.gc_threshold_mb = gc_threshold_mb
        self.weak_refs = weakref.WeakSet()
        self.component_pools: dict[type, 'ComponentPool'] = {}
        
        # 統計資料
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'gc_collections': 0,
            'objects_collected': 0,
            'last_collection': time.time(),
        }
        
        # 記憶體使用歷史
        self.memory_history: list[dict[str, Any]] = []
        
        # 批次處理配置
        self.default_batch_size = batch_size
        
        # 監控任務
        self.monitoring_task = None
        if enable_monitoring:
            self.start_monitoring()
            
        logger.info(f"UnifiedMemoryManager初始化: cache_size={max_cache_size}, "
                   f"gc_threshold={gc_threshold_mb}MB")

    # ==================== AI快取系統 ====================
    
    def _generate_cache_key(self, input_data: Any) -> str:
        """生成快取鍵值"""
        if HAS_NUMPY and np and isinstance(input_data, np.ndarray):
            content = input_data.tobytes()
        else:
            content = str(input_data).encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def get_cached_prediction(self, input_data: Any) -> Any:
        """獲取快取的預測結果"""
        cache_key = self._generate_cache_key(input_data)
        
        if cache_key in self.prediction_cache:
            result, _ = self.prediction_cache[cache_key]
            self.stats['cache_hits'] += 1
            
            # 更新LRU順序
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            
            # 返回副本避免意外修改
            if HAS_NUMPY and np and isinstance(result, np.ndarray):
                return result.copy()
            return result
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def cache_prediction(self, input_data: Any, prediction: Any) -> None:
        """快取預測結果"""
        cache_key = self._generate_cache_key(input_data)
        current_time = time.time()
        
        # 檢查是否需要清理快取
        if len(self.prediction_cache) >= self.max_cache_size:
            self._evict_oldest_cache_entry()
        
        # 存儲快取(複製以避免外部修改)
        if HAS_NUMPY and np and isinstance(prediction, np.ndarray):
            cached_prediction = prediction.copy()
        else:
            cached_prediction = prediction
            
        self.prediction_cache[cache_key] = (cached_prediction, current_time)
        
        # 更新LRU順序
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
    
    def _evict_oldest_cache_entry(self) -> None:
        """清除最舊的快取項目(LRU策略)"""
        if not self.access_order:
            return
        
        oldest_key = self.access_order.pop(0)
        if oldest_key in self.prediction_cache:
            del self.prediction_cache[oldest_key]
            self.stats['cache_evictions'] += 1
    
    def clear_cache(self) -> None:
        """清空AI預測快取"""
        self.prediction_cache.clear()
        self.access_order.clear()
        logger.info("AI預測快取已清空")

    # ==================== 系統記憶體管理 ====================
    
    def create_component_pool(self, component_class: type, pool_size: int = 10) -> 'ComponentPool':
        """創建組件對象池"""
        if component_class not in self.component_pools:
            self.component_pools[component_class] = ComponentPool(component_class, pool_size)
        return self.component_pools[component_class]
    
    def get_component_pool(self, component_class: type) -> 'ComponentPool | None':
        """獲取組件對象池"""
        return self.component_pools.get(component_class)
    
    def register_weak_ref(self, obj: Any) -> None:
        """註冊弱引用用於記憶體追蹤"""
        self.weak_refs.add(obj)
    
    def start_monitoring(self) -> None:
        """啟動記憶體監控任務"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_memory())
    
    async def stop_monitoring(self) -> None:
        """停止記憶體監控"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                logger.info("記憶體監控任務已取消")
                raise
            self.monitoring_task = None
    
    async def _monitor_memory(self) -> None:
        """記憶體監控主循環"""
        while True:
            try:
                current_memory = self._get_memory_usage_mb()
                
                # 記錄記憶體使用歷史
                self._record_memory_usage()
                
                # 檢查是否需要清理
                if current_memory > self.gc_threshold_mb:
                    self._force_cleanup()
                
                # 清理過期快取
                self._cleanup_expired_cache()
                
                # 每30秒檢查一次
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"記憶體監控錯誤: {e}")
                await asyncio.sleep(60)  # 錯誤時延長檢查間隔
    
    def _force_cleanup(self) -> None:
        """強制記憶體清理"""
        logger.info("記憶體使用超過閾值，執行強制清理...")
        
        before_objects = len(gc.get_objects())
        
        # 清理過期快取
        self._cleanup_expired_cache()
        
        # 執行垃圾回收
        collected = gc.collect()
        
        after_objects = len(gc.get_objects())
        
        # 更新統計
        self.stats['gc_collections'] += 1
        self.stats['objects_collected'] += before_objects - after_objects
        self.stats['last_collection'] = time.time()
        
        logger.info(f"記憶體清理完成: 回收{collected}個循環, 清理{before_objects - after_objects}個對象")
    
    def _cleanup_expired_cache(self) -> None:
        """清理過期的快取項目"""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self.prediction_cache.items():
            if (current_time - timestamp) > 3600:  # 1小時過期
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.prediction_cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
        
        if expired_keys:
            logger.debug(f"清理了{len(expired_keys)}個過期快取項目")

    # ==================== 批次處理系統 ====================
    
    def process_batch(self, items: list[Any], process_func: Callable[[Any], Any],
                     batch_size: int | None = None) -> list[Any]:
        """批次處理項目列表"""
        batch_size = batch_size or self.default_batch_size
        results = []
        cache_hits_in_batch = 0
        
        batch_start_time = time.time()
        
        for item in items:
            # 嘗試從快取獲取
            cached_result = self.get_cached_prediction(item)
            
            if cached_result is not None:
                results.append(cached_result)
                cache_hits_in_batch += 1
            else:
                # 執行實際處理
                result = process_func(item)
                results.append(result)
                
                # 快取結果
                self.cache_prediction(item, result)
        
        batch_time = time.time() - batch_start_time
        
        logger.debug(f"批次處理完成: {len(items)}項目, {cache_hits_in_batch}快取命中, {batch_time:.3f}s")
        return results
    
    def process_large_dataset(self, dataset: list[Any], process_func: Callable[[Any], Any],
                            batch_size: int | None = None) -> list[Any]:
        """處理大型資料集"""
        batch_size = batch_size or self.default_batch_size
        all_results = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        
        logger.info(f"開始處理大型資料集: {len(dataset)}項目, {total_batches}批次")
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = self.process_batch(batch, process_func, batch_size)
            all_results.extend(batch_results)
            
            # 定期執行記憶體優化
            if (i // batch_size + 1) % 10 == 0:
                self.optimize_memory()
        
        logger.info(f"大型資料集處理完成: {len(all_results)}結果")
        return all_results

    # ==================== 記憶體統計與優化 ====================
    
    def _get_memory_usage_mb(self) -> float:
        """獲取當前記憶體使用量(MB)"""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except Exception as e:
                logger.warning(f"psutil記憶體檢測失敗: {e}")
        
        # 備用方案: 估算快取記憶體使用
        cache_size_mb = len(self.prediction_cache) * 0.1
        return cache_size_mb
    
    def _record_memory_usage(self) -> None:
        """記錄記憶體使用歷史"""
        stats = {
            'timestamp': time.time(),
            'memory_mb': self._get_memory_usage_mb(),
            'cache_size': len(self.prediction_cache),
            'weak_refs_count': len(self.weak_refs),
        }
        
        self.memory_history.append(stats)
        
        # 保持歷史記錄在合理範圍
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-500:]
    
    def optimize_memory(self, force_gc: bool = False) -> dict[str, Any]:
        """優化記憶體使用"""
        stats_before = {
            'memory_mb': self._get_memory_usage_mb(),
            'cache_size': len(self.prediction_cache)
        }
        
        # 清理過期快取
        self._cleanup_expired_cache()
        
        # 執行垃圾回收
        if force_gc:
            gc.collect()
            self.stats['gc_collections'] += 1
        
        stats_after = {
            'memory_mb': self._get_memory_usage_mb(),
            'cache_size': len(self.prediction_cache)
        }
        
        return {
            'memory_saved_mb': stats_before['memory_mb'] - stats_after['memory_mb'],
            'cache_items_removed': stats_before['cache_size'] - stats_after['cache_size'],
            'gc_collections': self.stats['gc_collections'],
        }
    
    def get_comprehensive_stats(self) -> dict[str, Any]:
        """獲取全面的記憶體統計資訊"""
        cache_stats = self._get_cache_stats()
        memory_stats = self._get_memory_stats()
        pool_stats = self._get_pool_stats()
        
        return {
            'cache': cache_stats,
            'memory': memory_stats,
            'pools': pool_stats,
            'overall': {
                'uptime_seconds': time.time() - self.stats['last_collection'],
                'monitoring_active': self.monitoring_task is not None,
            }
        }
    
    def _get_cache_stats(self) -> dict[str, Any]:
        """獲取快取統計"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.prediction_cache),
            'max_size': self.max_cache_size,
            'hit_rate': hit_rate,
            'hits': self.stats['cache_hits'],
            'misses': self.stats['cache_misses'],
            'evictions': self.stats['cache_evictions'],
        }
    
    def _get_memory_stats(self) -> dict[str, Any]:
        """獲取記憶體統計"""
        current_memory = self._get_memory_usage_mb()
        
        return {
            'current_mb': current_memory,
            'threshold_mb': self.gc_threshold_mb,
            'weak_refs_count': len(self.weak_refs),
            'gc_collections': self.stats['gc_collections'],
            'objects_collected': self.stats['objects_collected'],
            'history_length': len(self.memory_history),
        }
    
    def _get_pool_stats(self) -> dict[str, Any]:
        """獲取組件池統計"""
        pool_stats = {}
        for component_class, pool in self.component_pools.items():
            pool_stats[component_class.__name__] = pool.get_pool_stats()
        return pool_stats


class ComponentPool:
    """組件對象池 - 避免頻繁建立/銷毀對象"""
    
    def __init__(self, component_class: type, pool_size: int = 10):
        self.component_class = component_class
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self.active_components = set()
        
        # 預先建立池中的對象
        for _ in range(pool_size):
            try:
                component = component_class()
                self.pool.put_nowait(component)
            except Exception as e:
                logger.warning(f"創建組件池對象失敗: {e}")
    
    @asynccontextmanager
    async def get_component(self):
        """獲取組件(上下文管理器)"""
        component = await self.pool.get()
        self.active_components.add(id(component))
        
        try:
            yield component
        finally:
            # 重置組件狀態
            if hasattr(component, 'reset'):
                try:
                    component.reset()
                except Exception as e:
                    logger.warning(f"組件重置失敗: {e}")
            
            self.active_components.discard(id(component))
            await self.pool.put(component)
    
    def get_pool_stats(self) -> dict[str, Any]:
        """獲取池統計資訊"""
        return {
            'pool_size': self.pool_size,
            'available': self.pool.qsize(),
            'active': len(self.active_components),
            'utilization': len(self.active_components) / self.pool_size if self.pool_size > 0 else 0,
        }


# 向後兼容的別名
AdvancedMemoryManager = UnifiedMemoryManager
MemoryManager = UnifiedMemoryManager
BatchProcessor = UnifiedMemoryManager  # 批次處理功能已整合