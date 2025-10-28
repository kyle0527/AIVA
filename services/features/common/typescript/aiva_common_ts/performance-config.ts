/**
 * AIVA AI 組件性能優化配置 - TypeScript 版本
 * 與 Python performance_config.py 保持一致的優化策略
 */

export enum CacheStrategy {
  MEMORY_ONLY = "memory_only",
  REDIS_ONLY = "redis_only", 
  HYBRID = "hybrid",
  DISABLED = "disabled"
}

export enum ProcessingMode {
  SYNC = "sync",
  ASYNC = "async",
  BATCH = "batch",
  STREAM = "stream"
}

export interface PerformanceConfig {
  // 緩存配置
  cache_strategy: CacheStrategy;
  cache_ttl_seconds: number;
  max_cache_size: number;
  
  // 並發配置
  max_concurrent_operations: number;
  operation_timeout_seconds: number;
  batch_size: number;
  
  // 資源池配置
  connection_pool_size: number;
  connection_pool_timeout: number;
  
  // 監控配置
  enable_performance_monitoring: boolean;
  metrics_sampling_rate: number;
}

export interface CapabilityEvaluatorConfig extends PerformanceConfig {
  // 評估特定配置
  evaluation_cache_enabled: boolean;
  evidence_batch_processing: boolean;
  parallel_evaluation_enabled: boolean;
  max_evaluation_workers: number;
  
  // 連續監控優化
  monitoring_interval_seconds: number;
  lightweight_monitoring: boolean;
  monitoring_batch_size: number;
  
  // 基準測試優化
  benchmark_cache_enabled: boolean;
  benchmark_timeout_seconds: number;
  skip_redundant_benchmarks: boolean;
}

export interface ExperienceManagerConfig extends PerformanceConfig {
  // 存儲優化
  storage_backend: string;
  batch_insert_enabled: boolean;
  async_storage_enabled: boolean;
  storage_buffer_size: number;
  
  // 查詢優化
  query_result_cache_enabled: boolean;
  index_optimization_enabled: boolean;
  query_planning_enabled: boolean;
  max_query_results: number;
  
  // 數據管理優化
  auto_cleanup_enabled: boolean;
  cleanup_interval_hours: number;
  retention_policy_days: number;
  compression_enabled: boolean;
  
  // 學習會話優化
  session_pooling_enabled: boolean;
  session_cache_size: number;
  session_timeout_minutes: number;
}

export interface PerformanceMetric {
  operation: string;
  total_calls: number;
  total_time: number;
  success_count: number;
  error_count: number;
  avg_time: number;
}

export class PerformanceOptimizer {
  private cache = new Map<string, any>();
  private performanceMetrics = new Map<string, PerformanceMetric>();
  
  getCachedResult(key: string, operationType: string): any | null {
    const cacheKey = `${operationType}:${key}`;
    const cached = this.cache.get(cacheKey);
    
    if (!cached) return null;
    
    const age = Date.now() - cached.timestamp;
    if (age > cached.ttl * 1000) {
      this.cache.delete(cacheKey);
      return null;
    }
    
    return cached.result;
  }
  
  setCachedResult(key: string, operationType: string, result: any, ttl: number = 3600): void {
    const cacheKey = `${operationType}:${key}`;
    this.cache.set(cacheKey, {
      result,
      timestamp: Date.now(),
      ttl
    });
  }
  
  async batchProcess<T, R>(
    items: T[], 
    processorFunc: (item: T) => Promise<R>, 
    batchSize: number = 100
  ): Promise<R[]> {
    const results: R[] = [];
    
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      const batchResults = await Promise.allSettled(
        batch.map(item => processorFunc(item))
      );
      
      results.push(...batchResults
        .filter(result => result.status === 'fulfilled')
        .map(result => (result as PromiseFulfilledResult<R>).value)
      );
    }
    
    return results;
  }
  
  recordPerformanceMetric(operation: string, executionTime: number, success: boolean = true): void {
    if (!this.performanceMetrics.has(operation)) {
      this.performanceMetrics.set(operation, {
        operation,
        total_calls: 0,
        total_time: 0,
        success_count: 0,
        error_count: 0,
        avg_time: 0
      });
    }
    
    const metric = this.performanceMetrics.get(operation)!;
    metric.total_calls++;
    metric.total_time += executionTime;
    
    if (success) {
      metric.success_count++;
    } else {
      metric.error_count++;
    }
    
    metric.avg_time = metric.total_time / metric.total_calls;
  }
  
  getPerformanceSummary(): Record<string, PerformanceMetric> {
    const summary: Record<string, PerformanceMetric> = {};
    this.performanceMetrics.forEach((metric, operation) => {
      summary[operation] = { ...metric };
    });
    return summary;
  }
}

// 預定義的優化配置
export const OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG: CapabilityEvaluatorConfig = {
  // 高性能緩存配置
  cache_strategy: CacheStrategy.HYBRID,
  cache_ttl_seconds: 1800, // 30分鐘
  max_cache_size: 2000,
  
  // 並發優化
  max_concurrent_operations: 8,
  operation_timeout_seconds: 15.0,
  batch_size: 50,
  
  // 評估特定優化
  evaluation_cache_enabled: true,
  evidence_batch_processing: true,
  parallel_evaluation_enabled: true,
  max_evaluation_workers: 6,
  
  // 輕量級監控
  monitoring_interval_seconds: 120.0,
  lightweight_monitoring: true,
  monitoring_batch_size: 100,
  
  // 基準測試優化
  benchmark_cache_enabled: true,
  benchmark_timeout_seconds: 8.0,
  skip_redundant_benchmarks: true,
  
  // 連接池
  connection_pool_size: 20,
  connection_pool_timeout: 5.0,
  
  // 性能監控
  enable_performance_monitoring: true,
  metrics_sampling_rate: 0.05 // 5% 採樣降低開銷
};

export const OPTIMIZED_EXPERIENCE_MANAGER_CONFIG: ExperienceManagerConfig = {
  // 高吞吐量配置
  cache_strategy: CacheStrategy.HYBRID,
  cache_ttl_seconds: 7200, // 2小時
  max_cache_size: 5000,
  
  // 並發和批處理
  max_concurrent_operations: 12,
  operation_timeout_seconds: 20.0,
  batch_size: 200,
  
  // 存儲優化
  storage_backend: "hybrid",
  batch_insert_enabled: true,
  async_storage_enabled: true,
  storage_buffer_size: 2000,
  
  // 查詢優化
  query_result_cache_enabled: true,
  index_optimization_enabled: true,
  query_planning_enabled: true,
  max_query_results: 500,
  
  // 自動維護
  auto_cleanup_enabled: true,
  cleanup_interval_hours: 12,
  retention_policy_days: 60,
  compression_enabled: true,
  
  // 會話管理
  session_pooling_enabled: true,
  session_cache_size: 200,
  session_timeout_minutes: 60,
  
  // 連接池
  connection_pool_size: 30,
  connection_pool_timeout: 3.0,
  
  // 輕量級監控
  enable_performance_monitoring: true,
  metrics_sampling_rate: 0.02 // 2% 採樣
};

// 性能基準值
export const PERFORMANCE_BENCHMARKS = {
  capability_evaluator: {
    initialization_time_ms: 1.0,
    evaluation_time_ms: 500.0,
    monitoring_overhead_percentage: 5.0,
    cache_hit_rate_percentage: 80.0
  },
  experience_manager: {
    initialization_time_ms: 2.0,
    sample_storage_time_ms: 10.0,
    query_time_ms: 100.0,
    batch_throughput_samples_per_second: 1000,
    cache_hit_rate_percentage: 85.0
  }
};

export function createOptimizedConfigs() {
  return {
    capability_evaluator: OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG,
    experience_manager: OPTIMIZED_EXPERIENCE_MANAGER_CONFIG,
    performance_benchmarks: PERFORMANCE_BENCHMARKS,
    global_settings: {
      async_mode_enabled: true,
      performance_monitoring_enabled: true,
      cache_warming_enabled: true,
      resource_pooling_enabled: true,
      optimization_level: "high"
    }
  };
}

// 性能監控裝飾器
export function performanceMonitor(operationName: string) {
  return function(_target: any, _propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args: any[]) {
      const startTime = Date.now();
      try {
        const result = await originalMethod.apply(this, args);
        const executionTime = Date.now() - startTime;
        console.log(`Operation ${operationName} completed in ${executionTime}ms`);
        return result;
      } catch (error) {
        const executionTime = Date.now() - startTime;
        console.error(`Operation ${operationName} failed after ${executionTime}ms:`, error);
        throw error;
      }
    };
    
    return descriptor;
  };
}

// 批處理優化裝飾器
export function batchProcessor(batchSize: number = 100) {
  return function(_target: any, _propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(items: any[], ...args: any[]) {
      if (items.length <= batchSize) {
        return await originalMethod.apply(this, [items, ...args]);
      }
      
      const results = [];
      for (let i = 0; i < items.length; i += batchSize) {
        const batch = items.slice(i, i + batchSize);
        const batchResult = await originalMethod.apply(this, [batch, ...args]);
        results.push(...(Array.isArray(batchResult) ? batchResult : [batchResult]));
      }
      
      return results;
    };
    
    return descriptor;
  };
}

// 緩存裝飾器
export function cached(ttl: number = 3600) {
  const optimizer = new PerformanceOptimizer();
  
  return function(_target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args: any[]) {
      const cacheKey = JSON.stringify(args);
      const cached = optimizer.getCachedResult(cacheKey, propertyKey);
      
      if (cached !== null) {
        return cached;
      }
      
      const result = await originalMethod.apply(this, args);
      optimizer.setCachedResult(cacheKey, propertyKey, result, ttl);
      
      return result;
    };
    
    return descriptor;
  };
}