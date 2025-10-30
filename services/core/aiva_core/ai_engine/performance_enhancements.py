"""
Performance Enhancements for BioNeuron Core

從 optimized_core.py 整合性能優化功能到核心 AI 模組
"""



import asyncio
import time
import weakref
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass




@dataclass
class PerformanceConfig:
    """性能配置參數"""
    
    # 並行處理配置
    max_concurrent_tasks: int = 20
    batch_size: int = 32
    
    # 記憶體管理配置
    prediction_cache_size: int = 1000
    gc_threshold_mb: int = 512
    
    # 權重量化配置
    use_quantized_weights: bool = True
    weight_dtype: type = np.float16


class OptimizedBioSpikingLayer:
    """優化的生物尖峰神經層 - 整合性能優化功能"""
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        config: PerformanceConfig = None
    ) -> None:
        """初始化優化的尖峰神經層
        
        Args:
            input_size: 輸入維度
            output_size: 輸出維度
            config: 性能配置
        """
        self.config = config or PerformanceConfig()
        
        # 使用量化權重降低記憶體使用
        weight_dtype = self.config.weight_dtype if self.config.use_quantized_weights else np.float32
        self.weights = np.random.randn(input_size, output_size).astype(weight_dtype) * np.sqrt(2.0 / input_size)
        
        self.threshold = 1.0
        self.refractory_period = 0.1
        self.last_spike_time = np.zeros(output_size) - self.refractory_period
        self.params = input_size * output_size
        
        # 預測快取
        self._prediction_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_requests = 0
    
    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """前向傳播，支援快取優化
        
        Args:
            x: 輸入訊號
            use_cache: 是否使用快取
            
        Returns:
            尖峰輸出
        """
        self._cache_requests += 1
        
        # 檢查快取
        if use_cache:
            cache_key = self._get_cache_key(x)
            if cache_key in self._prediction_cache:
                self._cache_hits += 1
                return self._prediction_cache[cache_key]
        
        # 執行計算
        result = self._compute_forward(x)
        
        # 儲存到快取
        if use_cache and len(self._prediction_cache) < self.config.prediction_cache_size:
            self._prediction_cache[cache_key] = result.copy()
        
        return result
    
    def _compute_forward(self, x: np.ndarray) -> np.ndarray:
        """執行實際的前向傳播計算"""
        current_time = time.time()
        potential = np.dot(x, self.weights.astype(np.float32))
        can_spike = (current_time - self.last_spike_time) > self.refractory_period
        spikes = (potential > self.threshold) & can_spike
        
        self.last_spike_time[spikes] = current_time
        return spikes.astype(np.float32)
    
    def _get_cache_key(self, x: np.ndarray) -> str:
        """生成快取鍵值"""
        return str(hash(x.tobytes()))
    
    def clear_cache(self) -> None:
        """清空快取"""
        self._prediction_cache.clear()
        self._cache_hits = 0
        self._cache_requests = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取快取統計"""
        return {
            "cache_size": len(self._prediction_cache),
            "cache_limit": self.config.prediction_cache_size,
            "hit_rate": self._cache_hits / max(self._cache_requests, 1),
            "hits": self._cache_hits,
            "requests": self._cache_requests,
        }


class OptimizedScalableBioNet:
    """優化的可擴展生物網路 - 整合性能增強功能"""
    
    def __init__(
        self, 
        input_size: int, 
        num_tools: int,
        config: PerformanceConfig = None
    ) -> None:
        """初始化優化的神經網路
        
        Args:
            input_size: 輸入向量大小
            num_tools: 工具數量
            config: 性能配置
        """
        self.config = config or PerformanceConfig()
        
        # 網路結構參數
        self.hidden_size_1 = 2048
        self.hidden_size_2 = 1024
        
        # 使用量化權重
        weight_dtype = self.config.weight_dtype if self.config.use_quantized_weights else np.float32
        
        # 層定義
        self.fc1 = np.random.randn(input_size, self.hidden_size_1).astype(weight_dtype)
        self.spiking1 = OptimizedBioSpikingLayer(
            self.hidden_size_1, 
            self.hidden_size_2,
            self.config
        )
        self.fc2 = np.random.randn(self.hidden_size_2, num_tools).astype(weight_dtype)
        
        # 參數計算
        self.params_fc1 = input_size * self.hidden_size_1
        self.params_spiking1 = self.spiking1.params
        self.params_fc2 = self.hidden_size_2 * num_tools
        self.total_params = self.params_fc1 + self.params_spiking1 + self.params_fc2
        
        # 批次處理緩衝區
        self._batch_buffer: List[np.ndarray] = []
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        # 性能統計
        self.performance_stats = {
            "predictions": 0,
            "batch_predictions": 0,
            "avg_prediction_time": 0.0,
            "total_prediction_time": 0.0,
        }
    
    def forward(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """前向傳播，支援性能優化
        
        Args:
            x: 輸入向量
            use_cache: 是否使用快取
            
        Returns:
            輸出向量
        """
        start_time = time.time()
        
        try:
            # 第一層 (使用量化權重)
            h1 = np.tanh(np.dot(x, self.fc1.astype(np.float32)))
            
            # 尖峰層 (支援快取)
            h2 = self.spiking1.forward(h1, use_cache=use_cache)
            
            # 輸出層
            # 使用自定義 softmax 函數
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)
            
            output = softmax(np.dot(h2, self.fc2.astype(np.float32)))
            
            # 更新性能統計
            prediction_time = time.time() - start_time
            self._update_performance_stats(prediction_time)
            
            return output
            
        except Exception as e:
            import logging
            logging.error(f"Forward pass failed: {e}")
            # 返回安全的預設值
            return np.zeros(self.fc2.shape[1])
    
    async def forward_async(self, x: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """異步前向傳播，支援並行控制
        
        Args:
            x: 輸入向量
            use_cache: 是否使用快取
            
        Returns:
            輸出向量
        """
        async with self._semaphore:
            # 在執行緒池中執行計算密集的操作
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.forward, x, use_cache)
    
    async def predict_batch(self, batch_x: List[np.ndarray]) -> List[np.ndarray]:
        """批次預測，提升吞吐量
        
        Args:
            batch_x: 輸入批次
            
        Returns:
            輸出批次
        """
        start_time = time.time()
        
        # 並行處理批次中的每個輸入
        tasks = [self.forward_async(x) for x in batch_x]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 處理異常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                import logging
                logging.error(f"Batch prediction failed for item {i}: {result}")
                # 返回安全的預設值
                processed_results.append(np.zeros(self.fc2.shape[1]))
            else:
                processed_results.append(result)
        
        # 更新批次統計
        batch_time = time.time() - start_time
        self.performance_stats["batch_predictions"] += 1
        self.performance_stats["total_prediction_time"] += batch_time
        
        return processed_results
    
    def _update_performance_stats(self, prediction_time: float) -> None:
        """更新性能統計"""
        self.performance_stats["predictions"] += 1
        self.performance_stats["total_prediction_time"] += prediction_time
        
        # 計算平均預測時間
        total_predictions = self.performance_stats["predictions"]
        total_time = self.performance_stats["total_prediction_time"]
        self.performance_stats["avg_prediction_time"] = total_time / total_predictions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        stats = self.performance_stats.copy()
        
        # 添加快取統計
        stats["spiking_layer_cache"] = self.spiking1.get_cache_stats()
        
        # 添加記憶體使用情況
        stats["memory_usage"] = self._get_memory_usage()
        
        return stats
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """獲取記憶體使用情況"""
        # 計算權重占用的記憶體
        fc1_memory = self.fc1.nbytes
        fc2_memory = self.fc2.nbytes
        spiking_memory = self.spiking1.weights.nbytes
        
        return {
            "fc1_bytes": fc1_memory,
            "fc2_bytes": fc2_memory,
            "spiking_bytes": spiking_memory,
            "total_bytes": fc1_memory + fc2_memory + spiking_memory,
            "total_mb": (fc1_memory + fc2_memory + spiking_memory) / 1024 / 1024,
        }
    
    def clear_all_caches(self) -> None:
        """清空所有快取"""
        self.spiking1.clear_cache()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """執行記憶體優化"""
        # 清空快取
        self.clear_all_caches()
        
        # 執行垃圾回收
        import gc
        collected = gc.collect()
        
        return {
            "caches_cleared": True,
            "gc_collected": collected,
            "memory_usage": self._get_memory_usage(),
        }


class MemoryManager:
    """記憶體管理器 - 從 optimized_core.py 整合"""
    
    def __init__(self, gc_threshold_mb: int = 512):
        self.gc_threshold_mb = gc_threshold_mb
        self.weak_refs = weakref.WeakSet()
        self.gc_stats = {
            "collections": 0,
            "objects_collected": 0,
            "last_collection": time.time()
        }
    
    async def start_monitoring(self):
        """啟動記憶體監控"""
        while True:
            current_memory = self._get_memory_usage_mb()
            
            if current_memory > self.gc_threshold_mb:
                await self._force_cleanup()
            
            # 每30秒檢查一次
            await asyncio.sleep(30)
    
    async def _force_cleanup(self):
        """強制清理記憶體"""
        import gc
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("Memory threshold exceeded, forcing cleanup...")
        
        before_count = len(gc.get_objects())
        collected = gc.collect()
        after_count = len(gc.get_objects())
        
        # 更新統計
        self.gc_stats["collections"] += 1
        self.gc_stats["objects_collected"] += (before_count - after_count)
        self.gc_stats["last_collection"] = time.time()
        
        logger.info(f"GC completed: {collected} cycles, {before_count - after_count} objects freed")
    
    def _get_memory_usage_mb(self) -> float:
        """獲取當前記憶體使用量（MB）"""
        try:
            # 嘗試使用 psutil
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except (ImportError, Exception):
            # 如果沒有 psutil 或出錯，使用 tracemalloc
            try:
                import tracemalloc
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    return current / 1024 / 1024
            except Exception:
                pass
        # 如果都不可用，返回 0
        return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶體統計"""
        return {
            "current_memory_mb": self._get_memory_usage_mb(),
            "threshold_mb": self.gc_threshold_mb,
            "gc_collections": self.gc_stats["collections"],
            "objects_collected": self.gc_stats["objects_collected"],
            "last_collection": self.gc_stats["last_collection"],
        }


class ComponentPool:
    """組件對象池 - 避免頻繁建立/銷毀物件"""
    
    def __init__(self, component_class: type, pool_size: int = 10):
        self.component_class = component_class
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self.active_components = set()
        
        # 預先建立池子中的物件
        for _ in range(pool_size):
            component = component_class()
            self.pool.put_nowait(component)
    
    @asynccontextmanager
    async def get_component(self):
        """取得組件（上下文管理器）"""
        component = await self.pool.get()
        self.active_components.add(id(component))
        
        try:
            yield component
        finally:
            # 重置組件狀態
            if hasattr(component, 'reset'):
                component.reset()
            
            self.active_components.discard(id(component))
            await self.pool.put(component)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """獲取池子統計資訊"""
        return {
            "pool_size": self.pool_size,
            "available": self.pool.qsize(),
            "active": len(self.active_components),
            "utilization": len(self.active_components) / self.pool_size if self.pool_size > 0 else 0,
        }