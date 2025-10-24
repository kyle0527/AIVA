#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.1 改善版本 - 修正信心度趨勢和計數邏輯

主要改善:
1. 修正信心度趨勢計算邏輯
2. 統一組件計數方式
3. 增加更詳細的問題分類
4. 優化變異性分析算法

這個文件包含了對V3.0智能分析框架的改善建議和實現代碼。
"""

from typing import Dict, List, Any
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ImprovedVariabilityManager:
    """改善版變異性管理器 - V3.1"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.history_file = workspace_root / "_out" / "analysis_history.json"
    
    def analyze_variability_improved(self) -> Dict[str, Any]:
        """分析結果變異性 - 改善版"""
        history = self._load_history()
        
        if len(history) < 2:
            return {'message': '歷史記錄不足，無法分析變異性'}
        
        # 計算變異性指標 - 修正版
        method_counts = [entry['method_count'] for entry in history]
        component_counts = [entry['component_count'] for entry in history]
        confidence_scores = [entry['quality_metrics']['confidence'] for entry in history]
        
        # 修正信心度趨勢判斷邏輯
        confidence_trend = self._calculate_trend_improved(confidence_scores)
        
        variability_report = {
            'stability_metrics': {
                'method_count': {
                    'mean': sum(method_counts) / len(method_counts),
                    'variance': self._calculate_variance(method_counts),
                    'stability_score': self._calculate_stability_score(method_counts),
                    'trend': self._calculate_trend_improved(method_counts)
                },
                'component_count': {
                    'mean': sum(component_counts) / len(component_counts),
                    'variance': self._calculate_variance(component_counts),
                    'stability_score': self._calculate_stability_score(component_counts),
                    'trend': self._calculate_trend_improved(component_counts)
                },
                'confidence': {
                    'mean': sum(confidence_scores) / len(confidence_scores),
                    'variance': self._calculate_variance(confidence_scores),
                    'trend': confidence_trend,
                    'is_reliable': len(history) >= 5  # 標記數據可靠性
                }
            },
            'data_quality': {
                'history_count': len(history),
                'reliability': 'high' if len(history) >= 10 else 'medium' if len(history) >= 5 else 'low',
                'recommendation': self._get_data_quality_recommendation(len(history))
            },
            'recommendations': self._generate_stability_recommendations_improved(history),
            'last_comparison': self._compare_recent_analyses(history[-2:]) if len(history) >= 2 else None
        }
        
        return variability_report

    def _calculate_trend_improved(self, values: List[float]) -> str:
        """改良的趨勢計算"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # 使用線性回歸計算趨勢
        n = len(values)
        x = list(range(n))
        
        # 計算斜率
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        # 趨勢判斷 (考慮變化幅度)
        threshold = max(y_mean * 0.05, 0.01)  # 5%變化或最小0.01
        
        if slope > threshold:
            return 'improving'
        elif slope < -threshold:
            return 'declining'
        else:
            return 'stable'

    def _calculate_variance(self, values: List[float]) -> float:
        """計算變異數"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance

    def _calculate_stability_score(self, values: List[float]) -> float:
        """計算穩定性分數 (0-1，越高越穩定)"""
        if len(values) < 2:
            return 1.0
        
        variance = self._calculate_variance(values)
        mean = sum(values) / len(values)
        
        # 避免除零錯誤
        if mean == 0:
            return 1.0 if variance == 0 else 0.0
        
        # 使用變異係數計算穩定性
        cv = (variance ** 0.5) / abs(mean)  # 變異係數
        stability = max(0.0, 1.0 - cv)  # 轉換為穩定性分數
        
        return min(1.0, stability)

    def _get_data_quality_recommendation(self, count: int) -> str:
        """數據質量建議"""
        if count < 5:
            return "建議執行更多次分析以獲得可靠的趨勢分析"
        elif count < 10:
            return "數據樣本適中，趨勢分析基本可靠"
        else:
            return "數據樣本充足，趨勢分析高度可靠"

    def _generate_stability_recommendations_improved(self, history: List[Dict]) -> List[str]:
        """產生改善建議"""
        recommendations = []
        
        # 分析最近的趨勢
        if len(history) >= 3:
            recent_methods = [h['method_count'] for h in history[-3:]]
            recent_confidence = [h['quality_metrics']['confidence'] for h in history[-3:]]
            
            method_trend = self._calculate_trend_improved(recent_methods)
            confidence_trend = self._calculate_trend_improved(recent_confidence)
            
            if method_trend == 'declining':
                recommendations.append("方法數量呈下降趨勢，建議檢查分析器配置")
            
            if confidence_trend == 'declining':
                recommendations.append("信心度下降，建議審查分析規則的有效性")
            
            # 檢查變異性
            method_variance = self._calculate_variance([float(x) for x in recent_methods])
            if method_variance > (sum(recent_methods) / len(recent_methods)) * 0.1:  # 10%變異
                recommendations.append("方法數量變異較大，建議標準化分析流程")
        
        # 數據質量建議
        if len(history) < 5:
            recommendations.append("累積更多分析歷史以提高穩定性評估準確性")
        
        return recommendations if recommendations else ["當前分析結果穩定，無需特別調整"]

    def _compare_recent_analyses(self, recent: List[Dict]) -> Dict[str, Any]:
        """比較最近的兩次分析"""
        if len(recent) < 2:
            return {}
        
        current = recent[-1]
        previous = recent[-2]
        
        # 安全的時間戳比較
        timestamp_diff = 0
        try:
            if 'timestamp' in current and 'timestamp' in previous:
                current_ts = current['timestamp']
                previous_ts = previous['timestamp']
                # 如果是字符串，嘗試解析為浮點數
                if isinstance(current_ts, str):
                    current_ts = float(current_ts)
                if isinstance(previous_ts, str):
                    previous_ts = float(previous_ts)
                timestamp_diff = current_ts - previous_ts
        except (ValueError, TypeError):
            timestamp_diff = 0
        
        return {
            'method_count_change': current['method_count'] - previous['method_count'],
            'component_count_change': current['component_count'] - previous['component_count'],
            'confidence_change': current['quality_metrics']['confidence'] - previous['quality_metrics']['confidence'],
            'timestamp_diff': timestamp_diff
        }

    def _load_history(self) -> List[Dict]:
        """載入分析歷史"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"無法載入歷史記錄: {e}")
        
        return []


class ImprovedQualityAnalyzer:
    """改善版品質分析器 - V3.1"""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'acceptable': 0.6,
            'needs_improvement': 0.4
        }

    def analyze_quality_improved(self, methods: List[Dict]) -> Dict[str, Any]:
        """改善版品質分析"""
        if not methods:
            return {'confidence': 0.0, 'issues': ['無方法可分析']}
        
        # 多維度品質評估
        quality_scores = {
            'completeness': self._assess_completeness(methods),
            'consistency': self._assess_consistency(methods),
            'clarity': self._assess_clarity(methods),
            'reliability': self._assess_reliability(methods)
        }
        
        # 計算綜合信心度 (加權平均)
        weights = {'completeness': 0.3, 'consistency': 0.3, 'clarity': 0.2, 'reliability': 0.2}
        confidence = sum(score * weights[dimension] for dimension, score in quality_scores.items())
        
        # 識別具體問題
        issues = self._identify_specific_issues(methods, quality_scores)
        
        return {
            'confidence': round(confidence, 3),
            'quality_breakdown': quality_scores,
            'quality_level': self._determine_quality_level(confidence),
            'issues': issues,
            'improvement_suggestions': self._generate_improvement_suggestions(quality_scores)
        }

    def _assess_completeness(self, methods: List[Dict]) -> float:
        """評估完整性"""
        required_fields = ['name', 'functionality', 'category']
        
        complete_methods = 0
        for method in methods:
            if all(field in method and method[field] for field in required_fields):
                complete_methods += 1
        
        return complete_methods / len(methods) if methods else 0.0

    def _assess_consistency(self, methods: List[Dict]) -> float:
        """評估一致性"""
        # 檢查命名一致性
        naming_patterns = {}
        for method in methods:
            if 'name' in method:
                pattern = self._extract_naming_pattern(method['name'])
                naming_patterns[pattern] = naming_patterns.get(pattern, 0) + 1
        
        # 最常見的命名模式佔比
        if naming_patterns:
            max_pattern_count = max(naming_patterns.values())
            consistency_score = max_pattern_count / len(methods)
        else:
            consistency_score = 0.0
        
        return consistency_score

    def _assess_clarity(self, methods: List[Dict]) -> float:
        """評估清晰度"""
        clear_methods = 0
        for method in methods:
            clarity_score = 0
            
            # 檢查描述長度
            if 'functionality' in method:
                desc_length = len(method['functionality'])
                if 10 <= desc_length <= 200:  # 適當長度
                    clarity_score += 0.5
            
            # 檢查類別明確性
            if 'category' in method and method['category'] != 'unknown':
                clarity_score += 0.5
            
            if clarity_score >= 0.5:  # 至少滿足一個條件
                clear_methods += 1
        
        return clear_methods / len(methods) if methods else 0.0

    def _assess_reliability(self, methods: List[Dict]) -> float:
        """評估可靠性"""
        # 基於方法數量和分類分佈的穩定性
        if len(methods) < 10:
            return 0.3  # 樣本太小
        elif len(methods) < 50:
            return 0.6  # 中等樣本
        else:
            return 0.9  # 大樣本

    def _extract_naming_pattern(self, name: str) -> str:
        """提取命名模式"""
        # 簡化的模式識別
        if '_' in name:
            return 'snake_case'
        elif any(c.isupper() for c in name[1:]):
            return 'camelCase'
        else:
            return 'lowercase'

    def _identify_specific_issues(self, methods: List[Dict], quality_scores: Dict[str, float]) -> List[str]:
        """識別具體問題"""
        issues = []
        
        if quality_scores['completeness'] < 0.8:
            missing_count = sum(1 for m in methods if not all(field in m for field in ['name', 'functionality', 'category']))
            issues.append(f"有 {missing_count} 個方法缺少必要資訊")
        
        if quality_scores['consistency'] < 0.7:
            issues.append("命名規範不一致")
        
        if quality_scores['clarity'] < 0.6:
            unclear_count = sum(1 for m in methods if len(m.get('functionality', '')) < 10)
            issues.append(f"有 {unclear_count} 個方法描述不夠清晰")
        
        if quality_scores['reliability'] < 0.5:
            issues.append(f"樣本數量 ({len(methods)}) 可能不足以保證分析可靠性")
        
        return issues

    def _determine_quality_level(self, confidence: float) -> str:
        """判斷品質等級"""
        for level, threshold in self.quality_thresholds.items():
            if confidence >= threshold:
                return level
        return 'poor'

    def _generate_improvement_suggestions(self, quality_scores: Dict[str, float]) -> List[str]:
        """產生改善建議"""
        suggestions = []
        
        if quality_scores['completeness'] < 0.8:
            suggestions.append("補充缺少的方法資訊，特別是功能描述和分類")
        
        if quality_scores['consistency'] < 0.7:
            suggestions.append("統一命名規範，建議使用一致的命名風格")
        
        if quality_scores['clarity'] < 0.6:
            suggestions.append("改善方法描述的清晰度和詳細程度")
        
        if quality_scores['reliability'] < 0.5:
            suggestions.append("增加分析範圍或執行次數以提高可靠性")
        
        return suggestions if suggestions else ["品質良好，繼續保持當前標準"]


# 使用示例
if __name__ == "__main__":
    # 初始化改善版管理器
    workspace = Path(r"C:\D\fold7\AIVA-git")
    
    variability_manager = ImprovedVariabilityManager(workspace)
    quality_analyzer = ImprovedQualityAnalyzer()
    
    # 分析變異性
    variability_report = variability_manager.analyze_variability_improved()
    print("=== V3.1 變異性分析報告 ===")
    print(json.dumps(variability_report, ensure_ascii=False, indent=2))
    
    # 如果有方法數據，進行品質分析
    # methods_data = []  # 這裡應該載入實際的方法數據
    # quality_report = quality_analyzer.analyze_quality_improved(methods_data)
    # print("\n=== V3.1 品質分析報告 ===")
    # print(json.dumps(quality_report, ensure_ascii=False, indent=2))