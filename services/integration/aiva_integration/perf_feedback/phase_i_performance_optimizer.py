"""
效能回饋循環系統 - Phase I 增強版

監控和優化 Phase I 高價值模組的效能表現
提供自適應策略調整建議
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import statistics

from services.aiva_common.utils import get_logger
from services.aiva_common.schemas.generated.findings import FindingPayload

logger = get_logger(__name__)

@dataclass
class ModulePerformanceMetric:
    """模組效能指標"""
    module_name: str
    execution_time: float
    memory_usage: float
    findings_count: int
    success_rate: float
    timestamp: datetime
    target_complexity: str  # SIMPLE, MEDIUM, COMPLEX

@dataclass
class PhaseIOptimizationSuggestion:
    """Phase I 優化建議"""
    module_name: str
    current_strategy: str
    suggested_strategy: str
    reason: str
    expected_improvement: float
    confidence: float

class PhaseIPerformanceFeedbackLoop:
    """Phase I 效能回饋循環"""
    
    def __init__(self):
        self.performance_history: List[ModulePerformanceMetric] = []
        self.optimization_history: List[PhaseIOptimizationSuggestion] = []
        self.threshold_config = {
            "max_execution_time": {
                "FUNC_CLIENT_AUTH_BYPASS": 45.0,  # 客戶端分析較複雜
                "FUNC_SSRF": 30.0,
                "FUNC_ADVANCED_SSRF": 60.0  # 進階 SSRF 需要更多時間
            },
            "min_findings_rate": {
                "FUNC_CLIENT_AUTH_BYPASS": 0.15,  # 15% 發現率算正常
                "FUNC_SSRF": 0.08,
                "FUNC_ADVANCED_SSRF": 0.05  # 進階漏洞相對罕見
            },
            "target_success_rate": 0.95
        }
    
    def record_performance(self, metric: ModulePerformanceMetric):
        """記錄效能指標"""
        self.performance_history.append(metric)
        logger.info(f"Recorded performance for {metric.module_name}: "
                   f"{metric.execution_time:.2f}s, {metric.findings_count} findings")
        
        # 保持最近 1000 條記錄
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def analyze_phase_i_performance(self) -> List[PhaseIOptimizationSuggestion]:
        """分析 Phase I 模組效能並生成優化建議"""
        suggestions = []
        
        # 分析各 Phase I 模組
        phase_i_modules = [
            "FUNC_CLIENT_AUTH_BYPASS", 
            "FUNC_ADVANCED_SSRF",
            "FUNC_SSRF"
        ]
        
        for module in phase_i_modules:
            module_suggestions = self._analyze_module_performance(module)
            suggestions.extend(module_suggestions)
        
        # 分析跨模組協作效能
        cross_module_suggestions = self._analyze_cross_module_efficiency()
        suggestions.extend(cross_module_suggestions)
        
        return suggestions
    
    def _analyze_module_performance(self, module_name: str) -> List[PhaseIOptimizationSuggestion]:
        """分析特定模組效能"""
        suggestions = []
        
        # 獲取最近 30 天的數據
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_metrics = [
            m for m in self.performance_history 
            if m.module_name == module_name and m.timestamp > cutoff_date
        ]
        
        if len(recent_metrics) < 10:  # 數據不足
            return suggestions
        
        # 效能分析
        avg_execution_time = statistics.mean([m.execution_time for m in recent_metrics])
        avg_findings_rate = statistics.mean([
            m.findings_count / max(1, m.execution_time) for m in recent_metrics
        ])
        avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
        
        # 1. 執行時間優化
        max_time_threshold = self.threshold_config["max_execution_time"].get(module_name, 30.0)
        if avg_execution_time > max_time_threshold:
            if module_name == "FUNC_CLIENT_AUTH_BYPASS":
                suggestions.append(PhaseIOptimizationSuggestion(
                    module_name=module_name,
                    current_strategy="COMPREHENSIVE",
                    suggested_strategy="TARGETED",
                    reason=f"平均執行時間 {avg_execution_time:.2f}s 超過閾值 {max_time_threshold}s。"
                           "建議調整 JavaScript 分析深度，專注於高風險模式。",
                    expected_improvement=0.35,
                    confidence=0.8
                ))
            elif "SSRF" in module_name:
                suggestions.append(PhaseIOptimizationSuggestion(
                    module_name=module_name,
                    current_strategy="DEEP_PROBE",
                    suggested_strategy="SMART_PROBE",
                    reason=f"平均執行時間 {avg_execution_time:.2f}s 超過閾值。"
                           "建議使用智能探測，優先檢測高價值內部服務。",
                    expected_improvement=0.4,
                    confidence=0.85
                ))
        
        # 2. 發現率優化
        min_findings_rate = self.threshold_config["min_findings_rate"].get(module_name, 0.1)
        if avg_findings_rate < min_findings_rate:
            if module_name == "FUNC_CLIENT_AUTH_BYPASS":
                suggestions.append(PhaseIOptimizationSuggestion(
                    module_name=module_name,
                    current_strategy="STATIC_ANALYSIS_ONLY",
                    suggested_strategy="DYNAMIC_INTERACTION",
                    reason=f"發現率 {avg_findings_rate:.3f} 低於預期。"
                           "建議增加動態交互測試，模擬用戶操作來觸發授權檢查。",
                    expected_improvement=0.6,
                    confidence=0.75
                ))
            elif module_name == "FUNC_ADVANCED_SSRF":
                suggestions.append(PhaseIOptimizationSuggestion(
                    module_name=module_name,
                    current_strategy="BASIC_ENDPOINTS",
                    suggested_strategy="CLOUD_METADATA_FOCUS",
                    reason="發現率偏低。建議增加雲端元數據端點檢測，"
                           "包括 AWS IMDSv2 和 GCP metadata API。",
                    expected_improvement=0.5,
                    confidence=0.8
                ))
        
        # 3. 成功率優化
        if avg_success_rate < self.threshold_config["target_success_rate"]:
            suggestions.append(PhaseIOptimizationSuggestion(
                module_name=module_name,
                current_strategy="CURRENT",
                suggested_strategy="ENHANCED_ERROR_HANDLING",
                reason=f"成功率 {avg_success_rate:.2%} 低於目標 95%。"
                       "建議增強錯誤處理和超時管理。",
                expected_improvement=0.15,
                confidence=0.9
            ))
        
        return suggestions
    
    def _analyze_cross_module_efficiency(self) -> List[PhaseIOptimizationSuggestion]:
        """分析跨模組協作效能"""
        suggestions = []
        
        # 分析模組執行順序優化
        # 客戶端授權繞過通常應該在 SSRF 之前執行，因為它能更快發現高價值目標
        client_auth_metrics = [m for m in self.performance_history if m.module_name == "FUNC_CLIENT_AUTH_BYPASS"]
        ssrf_metrics = [m for m in self.performance_history if "SSRF" in m.module_name]
        
        if client_auth_metrics and ssrf_metrics:
            avg_client_time = statistics.mean([m.execution_time for m in client_auth_metrics[-20:]])
            avg_ssrf_time = statistics.mean([m.execution_time for m in ssrf_metrics[-20:]])
            
            if avg_client_time < avg_ssrf_time * 0.6:  # 客戶端檢測明顯更快
                suggestions.append(PhaseIOptimizationSuggestion(
                    module_name="EXECUTION_ORDER",
                    current_strategy="PARALLEL_EXECUTION",
                    suggested_strategy="CLIENT_AUTH_FIRST",
                    reason="客戶端授權繞過檢測速度較快且能識別高價值目標，"
                           "建議優先執行以指導後續 SSRF 檢測策略。",
                    expected_improvement=0.25,
                    confidence=0.7
                ))
        
        return suggestions
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成效能報告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": "30_days",
            "phase_i_modules_performance": {},
            "optimization_suggestions": [],
            "trend_analysis": {}
        }
        
        # 按模組統計效能
        phase_i_modules = ["FUNC_CLIENT_AUTH_BYPASS", "FUNC_ADVANCED_SSRF", "FUNC_SSRF"]
        
        for module in phase_i_modules:
            recent_metrics = [
                m for m in self.performance_history 
                if m.module_name == module and 
                m.timestamp > datetime.now() - timedelta(days=30)
            ]
            
            if recent_metrics:
                report["phase_i_modules_performance"][module] = {
                    "total_executions": len(recent_metrics),
                    "avg_execution_time": statistics.mean([m.execution_time for m in recent_metrics]),
                    "avg_findings_count": statistics.mean([m.findings_count for m in recent_metrics]),
                    "success_rate": statistics.mean([m.success_rate for m in recent_metrics]),
                    "efficiency_score": statistics.mean([
                        m.findings_count / max(1, m.execution_time) for m in recent_metrics
                    ])
                }
        
        # 添加最新的優化建議
        report["optimization_suggestions"] = [
            {
                "module": s.module_name,
                "suggestion": s.suggested_strategy,
                "reason": s.reason,
                "expected_improvement": s.expected_improvement
            }
            for s in self.optimization_history[-10:]  # 最近 10 個建議
        ]
        
        return report
    
    def apply_optimization_suggestions(self, suggestions: List[PhaseIOptimizationSuggestion]):
        """應用優化建議"""
        for suggestion in suggestions:
            logger.info(f"Applying optimization for {suggestion.module_name}: "
                       f"{suggestion.current_strategy} -> {suggestion.suggested_strategy}")
            
            # 記錄優化歷史
            self.optimization_history.append(suggestion)
            
            # 實際應用邏輯（需要與各模組的配置系統整合）
            self._apply_module_optimization(suggestion)
    
    def _apply_module_optimization(self, suggestion: PhaseIOptimizationSuggestion):
        """應用特定模組的優化"""
        # 這裡需要與各模組的配置系統整合
        # 例如，更新 client_side_auth_bypass_worker 的掃描策略
        
        if suggestion.module_name == "FUNC_CLIENT_AUTH_BYPASS":
            # 更新客戶端授權繞過的掃描配置
            if suggestion.suggested_strategy in ["TARGETED_SCAN", "DYNAMIC_INTERACTION"]:
                # 調整為目標化掃描或啟用動態交互測試
                logger.info(f"應用策略: {suggestion.suggested_strategy}")
                
        elif "SSRF" in suggestion.module_name:
            if suggestion.suggested_strategy in ["SMART_PROBE", "CLOUD_METADATA_FOCUS"]:
                # 智能探測模式或專注於雲端元數據檢測
                logger.info(f"應用 SSRF 策略: {suggestion.suggested_strategy}")
        
        logger.info(f"Applied optimization for {suggestion.module_name}")


# 使用示例
async def main():
    """測試效能回饋循環"""
    feedback_loop = PhaseIPerformanceFeedbackLoop()
    
    # 模擬一些效能數據
    test_metrics = [
        ModulePerformanceMetric(
            module_name="FUNC_CLIENT_AUTH_BYPASS",
            execution_time=52.5,  # 超過閾值
            memory_usage=128.0,
            findings_count=3,
            success_rate=0.92,
            timestamp=datetime.now(),
            target_complexity="MEDIUM"
        ),
        ModulePerformanceMetric(
            module_name="FUNC_ADVANCED_SSRF",
            execution_time=35.2,
            memory_usage=64.0,
            findings_count=1,
            success_rate=0.96,
            timestamp=datetime.now(),
            target_complexity="COMPLEX"
        )
    ]
    
    # 記錄效能數據
    for metric in test_metrics:
        feedback_loop.record_performance(metric)
    
    # 分析並生成建議
    suggestions = await feedback_loop.analyze_phase_i_performance()
    
    print("=== Phase I 效能分析結果 ===")
    for suggestion in suggestions:
        print(f"\n模組: {suggestion.module_name}")
        print(f"建議: {suggestion.current_strategy} -> {suggestion.suggested_strategy}")
        print(f"原因: {suggestion.reason}")
        print(f"預期改善: {suggestion.expected_improvement:.1%}")
    
    # 生成報告
    report = feedback_loop.generate_performance_report()
    print("\n=== 效能報告 ===")
    print(f"分析期間: {report['analysis_period']}")
    print(f"Phase I 模組數量: {len(report['phase_i_modules_performance'])}")

if __name__ == "__main__":
    asyncio.run(main())