"""
AIVA AI 組件性能優化配置
基於 TODO 8 性能分析結果的優化策略實施
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import asyncio
import time
from functools import lru_cache
import logging

class CacheStrategy(Enum):
    """緩存策略枚舉"""
    MEMORY_ONLY = "memory_only"
    REDIS_ONLY = "redis_only"
    HYBRID = "hybrid"
    DISABLED = "disabled"

class ProcessingMode(Enum):
    """處理模式枚舉"""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAM = "stream"

@dataclass
class PerformanceConfig:
    """性能配置基類"""
    
    # 緩存配置
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 1000
    
    # 並發配置
    max_concurrent_operations: int = 10
    operation_timeout_seconds: float = 30.0
    batch_size: int = 100
    
    # 資源池配置
    connection_pool_size: int = 20
    connection_pool_timeout: float = 5.0
    
    # 監控配置
    enable_performance_monitoring: bool = True
    metrics_sampling_rate: float = 0.1  # 10% 採樣
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "cache_strategy": self.cache_strategy.value,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_size": self.max_cache_size,
            "max_concurrent_operations": self.max_concurrent_operations,
            "operation_timeout_seconds": self.operation_timeout_seconds,
            "batch_size": self.batch_size,
            "connection_pool_size": self.connection_pool_size,
            "connection_pool_timeout": self.connection_pool_timeout,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "metrics_sampling_rate": self.metrics_sampling_rate
        }

@dataclass
class CapabilityEvaluatorConfig(PerformanceConfig):
    """能力評估器性能配置"""
    
    # 評估特定配置
    evaluation_cache_enabled: bool = True
    continuous_monitoring: bool = True
    evidence_batch_processing: bool = True
    parallel_evaluation_enabled: bool = True
    max_evaluation_workers: int = 4
    
    # 連續監控優化
    monitoring_interval_seconds: float = 60.0
    lightweight_monitoring: bool = True
    monitoring_batch_size: int = 50
    
    # 基準測試優化
    benchmark_cache_enabled: bool = True
    benchmark_timeout_seconds: float = 10.0
    skip_redundant_benchmarks: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "evaluation_cache_enabled": self.evaluation_cache_enabled,
            "continuous_monitoring": self.continuous_monitoring,
            "evidence_batch_processing": self.evidence_batch_processing,
            "parallel_evaluation_enabled": self.parallel_evaluation_enabled,
            "max_evaluation_workers": self.max_evaluation_workers,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "lightweight_monitoring": self.lightweight_monitoring,
            "monitoring_batch_size": self.monitoring_batch_size,
            "benchmark_cache_enabled": self.benchmark_cache_enabled,
            "benchmark_timeout_seconds": self.benchmark_timeout_seconds,
            "skip_redundant_benchmarks": self.skip_redundant_benchmarks
        })
        return base_dict
    
    def keys(self):
        """提供字典接口的 keys 方法"""
        return self.to_dict().keys()
    
    def __len__(self):
        """提供 len() 支持"""
        return len(self.to_dict())

@dataclass 
class ExperienceManagerConfig(PerformanceConfig):
    """經驗管理器性能配置"""
    
    # 存儲優化
    storage_backend: str = "hybrid"  # "memory", "sqlite", "postgresql", "hybrid"
    batch_insert_enabled: bool = True
    async_storage_enabled: bool = True
    storage_buffer_size: int = 1000
    
    # 查詢優化
    query_result_cache_enabled: bool = True
    index_optimization_enabled: bool = True
    query_planning_enabled: bool = True
    max_query_results: int = 1000
    
    # 數據管理優化
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    retention_policy_days: int = 30
    compression_enabled: bool = True
    
    # 學習會話優化
    session_pooling_enabled: bool = True
    session_cache_size: int = 100
    session_timeout_minutes: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "storage_backend": self.storage_backend,
            "batch_insert_enabled": self.batch_insert_enabled,
            "async_storage_enabled": self.async_storage_enabled,
            "storage_buffer_size": self.storage_buffer_size,
            "query_result_cache_enabled": self.query_result_cache_enabled,
            "index_optimization_enabled": self.index_optimization_enabled,
            "query_planning_enabled": self.query_planning_enabled,
            "max_query_results": self.max_query_results,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "cleanup_interval_hours": self.cleanup_interval_hours,
            "retention_policy_days": self.retention_policy_days,
            "compression_enabled": self.compression_enabled,
            "session_pooling_enabled": self.session_pooling_enabled,
            "session_cache_size": self.session_cache_size,
            "session_timeout_minutes": self.session_timeout_minutes
        })
        return base_dict
    
    def keys(self):
        """提供字典接口的 keys 方法"""
        return self.to_dict().keys()
    
    def __len__(self):
        """提供 len() 支持"""
        return len(self.to_dict())

class PerformanceOptimizer:
    """性能優化器 - 實施優化策略的核心類"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._performance_metrics = {}
        
    @lru_cache(maxsize=128)
    def get_cached_result(self, key: str, operation_type: str) -> Optional[Any]:
        """獲取緩存結果"""
        cache_key = f"{operation_type}:{key}"
        return self._cache.get(cache_key)
    
    def set_cached_result(self, key: str, operation_type: str, result: Any, ttl: int = 3600):
        """設置緩存結果"""
        cache_key = f"{operation_type}:{key}"
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "ttl": ttl
        }
    
    def is_cache_valid(self, key: str, operation_type: str) -> bool:
        """檢查緩存是否有效"""
        cache_key = f"{operation_type}:{key}"
        if cache_key not in self._cache:
            return False
        
        cached_item = self._cache[cache_key]
        age = time.time() - cached_item["timestamp"]
        return age < cached_item["ttl"]
    
    def cached(self, ttl_seconds: int = 3600):
        """緩存裝飾器方法"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 創建緩存鍵
                import hashlib
                key_data = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
                
                # 檢查緩存
                if cache_key in self._cache:
                    cached_entry = self._cache[cache_key]
                    if (time.time() - cached_entry["timestamp"]) < ttl_seconds:
                        return cached_entry["result"]
                
                # 執行函數並緩存結果
                result = func(*args, **kwargs)
                self._cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time(),
                    "ttl": ttl_seconds
                }
                return result
            return wrapper
        return decorator
    
    def batch_processor(self, batch_size: int = 100):
        """批處理裝飾器方法"""
        def decorator(func):
            async def wrapper(items: List[Any], *args, **kwargs):
                if len(items) <= batch_size:
                    return await func(items, *args, **kwargs)

                results = []
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    batch_result = await func(batch, *args, **kwargs)
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                return results
            return wrapper
        return decorator
    
    async def batch_process(self, items: List[Any], processor_func, batch_size: int = 100) -> List[Any]:
        """批量處理優化"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[processor_func(item) for item in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
        
        return results
    
    def record_performance_metric(self, operation: str, execution_time: float, success: bool = True):
        """記錄性能指標"""
        if operation not in self._performance_metrics:
            self._performance_metrics[operation] = {
                "total_calls": 0,
                "total_time": 0.0,
                "success_count": 0,
                "error_count": 0,
                "avg_time": 0.0
            }
        
        metrics = self._performance_metrics[operation]
        metrics["total_calls"] += 1
        metrics["total_time"] += execution_time
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1
            
        metrics["avg_time"] = metrics["total_time"] / metrics["total_calls"]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """獲取性能總結"""
        return dict(self._performance_metrics)

# 預定義的優化配置
OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG = CapabilityEvaluatorConfig(
    # 高性能緩存配置
    cache_strategy=CacheStrategy.HYBRID,
    cache_ttl_seconds=1800,  # 30分鐘
    max_cache_size=2000,
    
    # 並發優化
    max_concurrent_operations=8,
    operation_timeout_seconds=15.0,
    batch_size=50,
    
    # 評估特定優化
    evaluation_cache_enabled=True,
    evidence_batch_processing=True,
    parallel_evaluation_enabled=True,
    max_evaluation_workers=6,
    
    # 輕量級監控
    monitoring_interval_seconds=120.0,
    lightweight_monitoring=True,
    monitoring_batch_size=100,
    
    # 基準測試優化
    benchmark_cache_enabled=True,
    benchmark_timeout_seconds=8.0,
    skip_redundant_benchmarks=True,
    
    # 性能監控
    enable_performance_monitoring=True,
    metrics_sampling_rate=0.05  # 5% 採樣降低開銷
)

OPTIMIZED_EXPERIENCE_MANAGER_CONFIG = ExperienceManagerConfig(
    # 高吞吐量配置
    cache_strategy=CacheStrategy.HYBRID,
    cache_ttl_seconds=7200,  # 2小時
    max_cache_size=5000,
    
    # 並發和批處理
    max_concurrent_operations=12,
    operation_timeout_seconds=20.0,
    batch_size=200,
    
    # 存儲優化
    storage_backend="hybrid",
    batch_insert_enabled=True,
    async_storage_enabled=True,
    storage_buffer_size=2000,
    
    # 查詢優化
    query_result_cache_enabled=True,
    index_optimization_enabled=True,
    query_planning_enabled=True,
    max_query_results=500,
    
    # 自動維護
    auto_cleanup_enabled=True,
    cleanup_interval_hours=12,
    retention_policy_days=60,
    compression_enabled=True,
    
    # 會話管理
    session_pooling_enabled=True,
    session_cache_size=200,
    session_timeout_minutes=60,
    
    # 連接池
    connection_pool_size=30,
    connection_pool_timeout=3.0,
    
    # 輕量級監控
    enable_performance_monitoring=True,
    metrics_sampling_rate=0.02  # 2% 採樣
)

# 性能基準值
PERFORMANCE_BENCHMARKS = {
    "capability_evaluator": {
        "initialization_time_ms": 1.0,
        "evaluation_time_ms": 500.0,
        "monitoring_overhead_percentage": 5.0,
        "cache_hit_rate_percentage": 80.0
    },
    "experience_manager": {
        "initialization_time_ms": 2.0,
        "sample_storage_time_ms": 10.0,
        "query_time_ms": 100.0,
        "batch_throughput_samples_per_second": 1000,
        "cache_hit_rate_percentage": 85.0
    }
}

def create_optimized_configs() -> Dict[str, Any]:
    """創建優化配置"""
    return {
        "capability_evaluator": OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG.to_dict(),
        "experience_manager": OPTIMIZED_EXPERIENCE_MANAGER_CONFIG.to_dict(),
        "performance_benchmarks": PERFORMANCE_BENCHMARKS,
        "global_settings": {
            "async_mode_enabled": True,
            "performance_monitoring_enabled": True,
            "cache_warming_enabled": True,
            "resource_pooling_enabled": True,
            "optimization_level": "high"
        }
    }

# 工具函數
def performance_monitor(operation_name: str):
    """性能監控裝飾器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logging.info(f"Operation {operation_name} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logging.error(f"Operation {operation_name} failed after {execution_time:.4f}s: {e}")
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logging.info(f"Operation {operation_name} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logging.error(f"Operation {operation_name} failed after {execution_time:.4f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def batch_processor(batch_size: int = 100):
    """批處理優化裝飾器"""
    def decorator(func):
        async def wrapper(items: List[Any], *args, **kwargs):
            if len(items) <= batch_size:
                return await func(items, *args, **kwargs)
            
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_result = await func(batch, *args, **kwargs)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            return results
        return wrapper
    return decorator

def create_development_config() -> Dict[str, Any]:
    """創建開發環境配置"""
    return {
        "environment": "development",  
        "cache_ttl_seconds": 300,
        "max_concurrent_operations": 2,
        "enable_monitoring": False,
        "batch_size": 10,
        "timeout_seconds": 5.0,
        "memory_limit_mb": 100,
        "logging_level": "DEBUG"
    }

def create_production_config() -> Dict[str, Any]:
    """創建生產環境配置"""
    return {
        "environment": "production",
        "cache_ttl_seconds": 7200,
        "max_concurrent_operations": 16,
        "enable_monitoring": True,
        "batch_size": 100,
        "timeout_seconds": 30.0,
        "memory_limit_mb": 2048,
        "logging_level": "INFO"
    }