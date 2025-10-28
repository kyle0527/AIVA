#!/usr/bin/env python3
"""
TODO 8 - AI 組件性能分析器
分析 capability 評估和 experience 管理的性能瓶頸，制定優化策略
"""

import sys
import time
import asyncio
import json
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import cProfile
import pstats
from io import StringIO

# 添加 AIVA 模組路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "services"))

@dataclass
class PerformanceMetrics:
    """性能指標數據結構"""
    component_name: str
    operation: str
    execution_time: float
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    throughput: Optional[float] = None
    error_count: int = 0
    optimization_potential: str = "low"

class AIComponentPerformanceAnalyzer:
    """AI 組件性能分析器"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.profiler = cProfile.Profile()
    
    def analyze_capability_evaluator_performance(self) -> List[PerformanceMetrics]:
        """分析 CapabilityEvaluator 性能"""
        print("🔍 分析 CapabilityEvaluator 性能...")
        
        try:
            from aiva_common.ai.capability_evaluator import AIVACapabilityEvaluator
            
            # 基本實例化性能
            start_time = time.time()
            evaluator = AIVACapabilityEvaluator()
            init_time = time.time() - start_time
            
            init_metric = PerformanceMetrics(
                component_name="AIVACapabilityEvaluator",
                operation="initialization",
                execution_time=init_time,
                optimization_potential="medium" if init_time > 0.1 else "low"
            )
            self.metrics.append(init_metric)
            print(f"  ✅ 初始化時間: {init_time:.4f}s")
            
            # 測試 evaluate_capability 性能
            test_capability_info = {
                "capability_id": "test_cap_001",
                "name": "Test Capability",
                "description": "Performance test capability",
                "category": "testing",
                "version": "1.0.0"
            }
            
            start_time = time.time()
            try:
                # 使用 profiler 詳細分析
                self.profiler.enable()
                result = evaluator.evaluate_capability(test_capability_info)
                self.profiler.disable()
                eval_time = time.time() - start_time
                
                eval_metric = PerformanceMetrics(
                    component_name="AIVACapabilityEvaluator", 
                    operation="evaluate_capability",
                    execution_time=eval_time,
                    optimization_potential="high" if eval_time > 1.0 else "medium" if eval_time > 0.5 else "low"
                )
                self.metrics.append(eval_metric)
                print(f"  ✅ 能力評估時間: {eval_time:.4f}s")
                
            except Exception as e:
                print(f"  ⚠️ 能力評估測試失敗: {e}")
                error_metric = PerformanceMetrics(
                    component_name="AIVACapabilityEvaluator",
                    operation="evaluate_capability",
                    execution_time=0,
                    error_count=1,
                    optimization_potential="high"
                )
                self.metrics.append(error_metric)
            
            # 測試連續監控性能
            start_time = time.time()
            try:
                evaluator.start_continuous_monitoring()
                time.sleep(0.1)  # 短暫監控
                evaluator.stop_continuous_monitoring()
                monitor_time = time.time() - start_time
                
                monitor_metric = PerformanceMetrics(
                    component_name="AIVACapabilityEvaluator",
                    operation="continuous_monitoring",
                    execution_time=monitor_time,
                    optimization_potential="medium" if monitor_time > 0.2 else "low"
                )
                self.metrics.append(monitor_metric)
                print(f"  ✅ 連續監控啟停時間: {monitor_time:.4f}s")
                
            except Exception as e:
                print(f"  ⚠️ 連續監控測試失敗: {e}")
        
        except ImportError as e:
            print(f"  ❌ 無法導入 CapabilityEvaluator: {e}")
        
        return [m for m in self.metrics if m.component_name == "AIVACapabilityEvaluator"]
    
    def analyze_experience_manager_performance(self) -> List[PerformanceMetrics]:
        """分析 ExperienceManager 性能"""
        print("\n🔍 分析 ExperienceManager 性能...")
        
        try:
            from aiva_common.ai.experience_manager import AIVAExperienceManager
            
            # 基本實例化性能
            start_time = time.time()
            manager = AIVAExperienceManager()
            init_time = time.time() - start_time
            
            init_metric = PerformanceMetrics(
                component_name="AIVAExperienceManager",
                operation="initialization", 
                execution_time=init_time,
                optimization_potential="medium" if init_time > 0.1 else "low"
            )
            self.metrics.append(init_metric)
            print(f"  ✅ 初始化時間: {init_time:.4f}s")
            
            # 測試經驗樣本添加性能
            test_sample = {
                "sample_id": "test_sample_001",
                "session_id": "test_session_001",
                "timestamp": time.time(),
                "context_vectors": [0.1, 0.2, 0.3],
                "action_taken": "test_action",
                "reward": 0.8,
                "quality_score": 0.9
            }
            
            # 單次添加性能
            start_time = time.time()
            try:
                manager.add_experience_sample(test_sample)
                add_time = time.time() - start_time
                
                add_metric = PerformanceMetrics(
                    component_name="AIVAExperienceManager",
                    operation="add_experience_sample",
                    execution_time=add_time,
                    optimization_potential="medium" if add_time > 0.1 else "low"
                )
                self.metrics.append(add_metric)
                print(f"  ✅ 單次樣本添加時間: {add_time:.4f}s")
                
            except Exception as e:
                print(f"  ⚠️ 樣本添加測試失敗: {e}")
            
            # 批量添加性能測試
            batch_samples = []
            for i in range(100):
                sample = test_sample.copy()
                sample["sample_id"] = f"batch_sample_{i:03d}"
                batch_samples.append(sample)
            
            start_time = time.time()
            try:
                for sample in batch_samples:
                    manager.add_experience_sample(sample)
                batch_time = time.time() - start_time
                throughput = len(batch_samples) / batch_time
                
                batch_metric = PerformanceMetrics(
                    component_name="AIVAExperienceManager",
                    operation="batch_add_samples",
                    execution_time=batch_time,
                    throughput=throughput,
                    optimization_potential="high" if throughput < 100 else "medium" if throughput < 500 else "low"
                )
                self.metrics.append(batch_metric)
                print(f"  ✅ 批量添加100個樣本: {batch_time:.4f}s (吞吐量: {throughput:.1f} samples/s)")
                
            except Exception as e:
                print(f"  ⚠️ 批量添加測試失敗: {e}")
            
            # 測試經驗檢索性能
            start_time = time.time()
            try:
                experiences = manager.retrieve_experiences(limit=50)
                retrieve_time = time.time() - start_time
                
                retrieve_metric = PerformanceMetrics(
                    component_name="AIVAExperienceManager",
                    operation="retrieve_experiences",
                    execution_time=retrieve_time,
                    optimization_potential="high" if retrieve_time > 0.5 else "medium" if retrieve_time > 0.1 else "low"
                )
                self.metrics.append(retrieve_metric)
                print(f"  ✅ 檢索50個經驗樣本: {retrieve_time:.4f}s")
                
            except Exception as e:
                print(f"  ⚠️ 經驗檢索測試失敗: {e}")
        
        except ImportError as e:
            print(f"  ❌ 無法導入 ExperienceManager: {e}")
        
        return [m for m in self.metrics if m.component_name == "AIVAExperienceManager"]
    
    def analyze_async_performance(self) -> List[PerformanceMetrics]:
        """分析異步操作性能"""
        print("\n🔍 分析異步操作性能...")
        
        async def async_capability_evaluation():
            """異步能力評估測試"""
            try:
                from aiva_common.ai.capability_evaluator import AIVACapabilityEvaluator
                evaluator = AIVACapabilityEvaluator()
                
                start_time = time.time()
                # 模擬異步評估操作
                await asyncio.sleep(0.01)  # 模擬 I/O 操作
                result = evaluator.evaluate_capability({
                    "capability_id": "async_test",
                    "name": "Async Test",
                    "category": "testing"
                })
                execution_time = time.time() - start_time
                
                return PerformanceMetrics(
                    component_name="AIVACapabilityEvaluator",
                    operation="async_evaluate_capability",
                    execution_time=execution_time,
                    optimization_potential="medium" if execution_time > 0.1 else "low"
                )
            except Exception as e:
                print(f"    ⚠️ 異步能力評估失敗: {e}")
                return None
        
        async def async_experience_batch_processing():
            """異步經驗批處理測試"""
            try:
                from aiva_common.ai.experience_manager import AIVAExperienceManager
                manager = AIVAExperienceManager()
                
                # 創建測試數據
                samples = []
                for i in range(50):
                    samples.append({
                        "sample_id": f"async_sample_{i:03d}",
                        "session_id": "async_session",
                        "timestamp": time.time(),
                        "context_vectors": [0.1 * i, 0.2 * i],
                        "action_taken": f"async_action_{i}",
                        "reward": 0.5 + (i % 10) * 0.05
                    })
                
                start_time = time.time()
                # 模擬異步批處理
                for sample in samples:
                    manager.add_experience_sample(sample)
                    await asyncio.sleep(0.001)  # 模擬異步間隔
                execution_time = time.time() - start_time
                
                return PerformanceMetrics(
                    component_name="AIVAExperienceManager",
                    operation="async_batch_processing",
                    execution_time=execution_time,
                    throughput=len(samples) / execution_time,
                    optimization_potential="high" if execution_time > 1.0 else "low"
                )
            except Exception as e:
                print(f"    ⚠️ 異步批處理失敗: {e}")
                return None
        
        # 運行異步測試
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 並發測試
            start_time = time.time()
            tasks = [
                async_capability_evaluation(),
                async_experience_batch_processing()
            ]
            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            concurrent_time = time.time() - start_time
            
            for result in results:
                if isinstance(result, PerformanceMetrics):
                    self.metrics.append(result)
                    print(f"  ✅ {result.operation}: {result.execution_time:.4f}s")
            
            concurrent_metric = PerformanceMetrics(
                component_name="AsyncOperations",
                operation="concurrent_execution",
                execution_time=concurrent_time,
                optimization_potential="medium" if concurrent_time > 2.0 else "low"
            )
            self.metrics.append(concurrent_metric)
            print(f"  ✅ 並發執行總時間: {concurrent_time:.4f}s")
            
        finally:
            loop.close()
        
        return [m for m in self.metrics if "async" in m.operation.lower()]
    
    def generate_optimization_recommendations(self) -> Dict[str, Any]:
        """基於性能分析生成優化建議"""
        print("\n📊 生成優化建議...")
        
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "performance_summary": {},
            "optimization_strategies": {}
        }
        
        # 按組件分組分析
        component_metrics = {}
        for metric in self.metrics:
            if metric.component_name not in component_metrics:
                component_metrics[metric.component_name] = []
            component_metrics[metric.component_name].append(metric)
        
        for component, metrics in component_metrics.items():
            avg_time = sum(m.execution_time for m in metrics) / len(metrics)
            error_count = sum(m.error_count for m in metrics)
            high_potential_ops = [m for m in metrics if m.optimization_potential == "high"]
            
            recommendations["performance_summary"][component] = {
                "average_execution_time": avg_time,
                "total_operations": len(metrics),
                "error_count": error_count,
                "high_optimization_potential": len(high_potential_ops)
            }
            
            # 生成具體建議
            if component == "AIVACapabilityEvaluator":
                if any(m.execution_time > 1.0 for m in metrics):
                    recommendations["high_priority"].append({
                        "component": component,
                        "issue": "能力評估操作耗時過長",
                        "recommendation": "實施評估結果緩存和並行處理",
                        "expected_improvement": "50-70% 性能提升"
                    })
                
                if any(m.operation == "continuous_monitoring" and m.execution_time > 0.2 for m in metrics):
                    recommendations["medium_priority"].append({
                        "component": component,
                        "issue": "連續監控啟停開銷較大",
                        "recommendation": "使用輕量級監控模式和狀態緩存",
                        "expected_improvement": "30-40% 性能提升"
                    })
            
            elif component == "AIVAExperienceManager":
                throughput_metrics = [m for m in metrics if m.throughput is not None]
                if throughput_metrics and min(m.throughput for m in throughput_metrics) < 100:
                    recommendations["high_priority"].append({
                        "component": component,
                        "issue": "經驗樣本處理吞吐量低",
                        "recommendation": "實施批量處理、異步 I/O 和內存緩存",
                        "expected_improvement": "200-300% 吞吐量提升"
                    })
                
                if any(m.operation == "retrieve_experiences" and m.execution_time > 0.1 for m in metrics):
                    recommendations["medium_priority"].append({
                        "component": component,
                        "issue": "經驗檢索延遲較高",
                        "recommendation": "添加索引、實施查詢優化和結果預緩存",
                        "expected_improvement": "60-80% 查詢性能提升"
                    })
        
        # 系統級優化建議
        recommendations["optimization_strategies"] = {
            "caching": {
                "description": "實施多層緩存策略",
                "targets": ["評估結果", "經驗查詢", "配置數據"],
                "implementation": "Redis + 內存緩存",
                "priority": "high"
            },
            "async_processing": {
                "description": "增強異步處理能力",
                "targets": ["批量操作", "I/O 密集任務", "並發評估"],
                "implementation": "asyncio + 工作隊列",
                "priority": "high"
            },
            "resource_pooling": {
                "description": "實施資源池化管理",
                "targets": ["數據庫連接", "計算資源", "臨時對象"],
                "implementation": "連接池 + 對象池",
                "priority": "medium"
            },
            "monitoring_optimization": {
                "description": "優化性能監控開銷",
                "targets": ["連續監控", "指標收集", "日誌記錄"],
                "implementation": "採樣監控 + 輕量級指標",
                "priority": "medium"
            }
        }
        
        return recommendations
    
    def get_profiler_stats(self) -> str:
        """獲取詳細的 profiler 統計信息"""
        if not hasattr(self.profiler, 'getstats') or not self.profiler.getstats():
            return "No profiling data available"
        
        s = StringIO()
        stats = pstats.Stats(self.profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # 顯示前20個最耗時的函數
        return s.getvalue()

def main():
    """主執行函數"""
    print("🚀 開始 TODO 8 - AI 組件性能分析")
    print("=" * 60)
    
    analyzer = AIComponentPerformanceAnalyzer()
    
    # 執行性能分析
    capability_metrics = analyzer.analyze_capability_evaluator_performance()
    experience_metrics = analyzer.analyze_experience_manager_performance() 
    async_metrics = analyzer.analyze_async_performance()
    
    # 生成優化建議
    recommendations = analyzer.generate_optimization_recommendations()
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("📈 性能分析結果總結")
    print("=" * 60)
    
    print(f"\n📊 總計分析了 {len(analyzer.metrics)} 個性能指標")
    
    # 顯示高優先級建議
    if recommendations["high_priority"]:
        print("\n🔥 高優先級優化建議:")
        for rec in recommendations["high_priority"]:
            print(f"  • {rec['component']}: {rec['issue']}")
            print(f"    解決方案: {rec['recommendation']}")
            print(f"    預期改善: {rec['expected_improvement']}")
    
    # 保存詳細報告
    report = {
        "analysis_timestamp": time.time(),
        "metrics": [
            {
                "component": m.component_name,
                "operation": m.operation,
                "execution_time": m.execution_time,
                "throughput": m.throughput,
                "error_count": m.error_count,
                "optimization_potential": m.optimization_potential
            }
            for m in analyzer.metrics
        ],
        "recommendations": recommendations,
        "profiler_stats": analyzer.get_profiler_stats()
    }
    
    with open("TODO8_AI_PERFORMANCE_ANALYSIS_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 詳細分析報告已保存到: TODO8_AI_PERFORMANCE_ANALYSIS_REPORT.json")
    
    # 根據分析結果返回適當的退出碼
    high_priority_count = len(recommendations["high_priority"])
    if high_priority_count > 0:
        print(f"\n⚠️  發現 {high_priority_count} 個高優先級性能問題需要優化")
        return 1
    else:
        print("\n✅ 性能分析完成，整體性能良好")
        return 0

if __name__ == "__main__":
    sys.exit(main())