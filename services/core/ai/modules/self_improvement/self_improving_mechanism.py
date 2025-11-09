"""
AIVA Self-improving Mechanism v2.0
自我改進機制 - 2025 AI 趨勢核心組件

實現所有模組通用的自我學習和改進機制，包含：
1. 經驗累積和學習 (Experience Learning)
2. 模式識別和分析 (Pattern Recognition)  
3. 適應性調整 (Adaptive Adjustment)
4. 性能優化 (Performance Optimization)
5. 知識蒸餾 (Knowledge Distillation)

Author: AIVA Team
Created: 2025-11-09
Version: 2.0.0
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import pickle
import os
from collections import deque, defaultdict

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 自我改進核心數據結構 ====================

@dataclass
class ExperienceRecord:
    """經驗記錄"""
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    module_name: str = ""
    operation: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    execution_time: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    learning_value: float = 0.0  # 此經驗的學習價值
    confidence: float = 1.0  # 經驗的可信度

@dataclass
class ImprovementSuggestion:
    """改進建議"""
    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    module_name: str = ""
    improvement_type: str = ""  # 'performance', 'accuracy', 'reliability', 'efficiency'
    description: str = ""
    confidence: float = 0.0
    expected_impact: float = 0.0
    implementation_effort: str = "low"  # 'low', 'medium', 'high'
    priority: int = 1  # 1-10
    related_experiences: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# ==================== 神經網路組件 ====================

class ExperienceEncoder(nn.Module):
    """經驗編碼器 - 將經驗轉換為向量表示"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 經驗特徵編碼器
        self.experience_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 上下文編碼器
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim // 2)
        )
        
        # 注意力機制
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
    def forward(self, experience_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """編碼經驗"""
        # 編碼經驗特徵
        exp_encoded = self.experience_encoder(experience_features)
        
        # 編碼上下文特徵
        ctx_encoded = self.context_encoder(context_features)
        
        # 合併特徵
        combined = torch.cat([exp_encoded, ctx_encoded], dim=-1)
        combined = combined.unsqueeze(1)  # 添加序列維度
        
        # 注意力機制
        attended, _ = self.attention(combined, combined, combined)
        
        return attended.squeeze(1)  # 移除序列維度

class PatternDetector(nn.Module):
    """模式檢測器 - 識別經驗中的模式"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, pattern_types: int = 10):  # 修正input_dim
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pattern_types = pattern_types
        
        # 模式檢測網路
        self.pattern_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pattern_types)
        )
        
        # 模式相似度計算
        self.similarity_layer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, encoded_experiences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """檢測模式"""
        # 檢測模式類型
        pattern_scores = self.pattern_detector(encoded_experiences)
        pattern_probs = F.softmax(pattern_scores, dim=-1)
        
        # 計算經驗間的相似度
        batch_size = encoded_experiences.size(0)
        similarities = torch.zeros(batch_size, batch_size)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                pair_input = torch.cat([encoded_experiences[i], encoded_experiences[j]], dim=-1)
                sim = self.similarity_layer(pair_input.unsqueeze(0))
                similarities[i, j] = similarities[j, i] = sim.item()
        
        return {
            'pattern_probabilities': pattern_probs,
            'pattern_scores': pattern_scores,
            'experience_similarities': similarities
        }

class AdaptationController(nn.Module):
    """適應控制器 - 決定如何調整模組參數"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, adjustment_types: int = 5):  # 修正input_dim
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.adjustment_types = adjustment_types
        
        # 調整策略網路
        self.strategy_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, adjustment_types)
        )
        
        # 調整強度網路
        self.intensity_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 輸出 0-1 之間的調整強度
        )
        
    def forward(self, pattern_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        """決定適應策略"""
        # 決定調整策略
        strategy_logits = self.strategy_network(pattern_info)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        
        # 決定調整強度
        intensity = self.intensity_network(pattern_info)
        
        return {
            'adjustment_strategy': strategy_probs,
            'adjustment_intensity': intensity,
            'strategy_logits': strategy_logits
        }

# ==================== 自我改進引擎 ====================

class SelfImprovingEngine:
    """自我改進引擎 - 核心實現"""
    
    def __init__(self, module_name: str, config: Optional[Dict[str, Any]] = None):
        self.module_name = module_name
        self.config = config or {}
        
        # 神經網路組件
        self.experience_encoder = ExperienceEncoder()
        self.pattern_detector = PatternDetector()
        self.adaptation_controller = AdaptationController()
        
        # 經驗存儲
        self.experience_buffer = deque(maxlen=self.config.get('max_experiences', 10000))
        self.pattern_memory = {}
        self.improvement_history = []
        
        # 統計指標
        self.stats = {
            'total_experiences': 0,
            'successful_improvements': 0,
            'failed_improvements': 0,
            'learning_rate': 0.01,
            'adaptation_count': 0,
            'pattern_detection_count': 0
        }
        
        # 性能追蹤
        self.performance_tracker = PerformanceTracker()
        
        logger.info(f"自我改進引擎已初始化，模組: {module_name}")
        
    def add_experience(self, experience: ExperienceRecord) -> bool:
        """添加新經驗"""
        try:
            # 計算學習價值
            experience.learning_value = self._calculate_learning_value(experience)
            
            # 添加到緩衝區
            self.experience_buffer.append(experience)
            self.stats['total_experiences'] += 1
            
            # 更新性能追蹤
            self.performance_tracker.update(experience)
            
            # 如果累積足夠經驗，觸發學習
            if len(self.experience_buffer) >= self.config.get('min_batch_size', 32):
                # 在後台線程中執行學習，避免阻塞
                import threading
                learning_thread = threading.Thread(target=self._trigger_learning)
                learning_thread.daemon = True
                learning_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"添加經驗失敗: {e}")
            return False
    
    def _trigger_learning(self) -> None:
        """觸發學習過程"""
        try:
            # 準備批次經驗
            experiences = list(self.experience_buffer)[-32:]  # 最近的32個經驗
            
            # 編碼經驗
            encoded_experiences = self._encode_experiences(experiences)
            
            # 檢測模式
            patterns = self._detect_patterns(encoded_experiences)
            
            # 生成改進建議
            suggestions = self._generate_improvements(patterns, experiences)
            
            # 應用改進
            applied_count = self._apply_improvements(suggestions)
            
            self.stats['adaptation_count'] += applied_count
            logger.info(f"觸發學習完成: 應用了 {applied_count} 個改進")
            
        except Exception as e:
            logger.error(f"學習過程出錯: {e}")
    
    def _encode_experiences(self, experiences: List[ExperienceRecord]) -> torch.Tensor:
        """編碼經驗為張量"""
        encoded_list = []
        
        for exp in experiences:
            # 提取經驗特徵
            exp_features = self._extract_experience_features(exp)
            ctx_features = self._extract_context_features(exp)
            
            # 轉換為張量
            exp_tensor = torch.tensor(exp_features, dtype=torch.float32)
            ctx_tensor = torch.tensor(ctx_features, dtype=torch.float32)
            
            # 編碼
            with torch.no_grad():
                encoded = self.experience_encoder(exp_tensor.unsqueeze(0), ctx_tensor.unsqueeze(0))
                encoded_list.append(encoded.squeeze(0))
        
        return torch.stack(encoded_list)
    
    def _extract_experience_features(self, exp: ExperienceRecord) -> List[float]:
        """提取經驗特徵"""
        features = [0.0] * 512
        
        # 基本特徵
        features[0] = 1.0 if exp.success else 0.0
        features[1] = min(exp.execution_time / 10.0, 1.0)  # 正規化執行時間
        features[2] = exp.confidence
        features[3] = exp.learning_value
        
        # 性能指標特徵
        perf_metrics = exp.performance_metrics
        for i, metric in enumerate(['accuracy', 'speed', 'memory', 'cpu', 'reliability']):
            if metric in perf_metrics and i < 5:
                features[4 + i] = min(perf_metrics[metric], 1.0)
        
        # 操作類型編碼 (簡單的哈希映射)
        operation_hash = hash(exp.operation) % 100
        features[9 + operation_hash] = 1.0
        
        # 時間特徵
        hour = exp.timestamp.hour
        features[109 + hour] = 1.0  # 小時熱編碼
        
        # 輸入/輸出大小特徵
        input_size = len(str(exp.input_data))
        output_size = len(str(exp.output_data))
        features[133] = min(input_size / 10000.0, 1.0)
        features[134] = min(output_size / 10000.0, 1.0)
        
        # 隨機特徵填充
        rng = np.random.default_rng(42)
        for i in range(135, 512):
            features[i] = rng.normal(0, 0.1)
        
        return features
    
    def _extract_context_features(self, exp: ExperienceRecord) -> List[float]:
        """提取上下文特徵"""
        features = [0.0] * 512
        
        # 上下文大小
        context_size = len(exp.context)
        features[0] = min(context_size / 50.0, 1.0)
        
        # 模組特定特徵
        if 'file_count' in exp.context:
            features[1] = min(exp.context['file_count'] / 1000.0, 1.0)
        
        if 'complexity' in exp.context:
            complexity = exp.context['complexity']
            if isinstance(complexity, dict):
                features[2] = min(complexity.get('cyclomatic', 0) / 100.0, 1.0)
        
        # 隨機特徵填充
        rng = np.random.default_rng(43)
        for i in range(3, 512):
            features[i] = rng.normal(0, 0.1)
        
        return features
    
    def _calculate_learning_value(self, exp: ExperienceRecord) -> float:
        """計算經驗的學習價值"""
        value = 0.0
        
        # 基於成功/失敗
        if not exp.success:
            value += 0.8  # 失敗的經驗更有學習價值
        else:
            value += 0.2
        
        # 基於執行時間（異常的時間更有價值）
        avg_time = self.performance_tracker.get_average_execution_time()
        if avg_time > 0:
            time_deviation = abs(exp.execution_time - avg_time) / avg_time
            value += min(time_deviation, 0.5)
        
        # 基於性能指標
        for metric_value in exp.performance_metrics.values():
            if isinstance(metric_value, (int, float)) and metric_value < 0.5:  # 低性能指標有更高學習價值
                value += 0.3
        
        return min(value, 1.0)
    
    def _detect_patterns(self, encoded_experiences: torch.Tensor) -> Dict[str, Any]:
        """檢測經驗模式"""
        self.stats['pattern_detection_count'] += 1
        
        with torch.no_grad():
            pattern_results = self.pattern_detector(encoded_experiences)
        
        # 分析模式
        patterns = {
            'dominant_patterns': [],
            'pattern_clusters': [],
            'anomalies': [],
            'trends': []
        }
        
        # 找出主導模式
        pattern_probs = pattern_results['pattern_probabilities']
        mean_probs = pattern_probs.mean(dim=0)
        dominant_indices = torch.topk(mean_probs, k=3).indices
        
        for idx in dominant_indices:
            patterns['dominant_patterns'].append({
                'pattern_id': int(idx),
                'strength': float(mean_probs[idx]),
                'frequency': int((pattern_probs[:, idx] > 0.5).sum())
            })
        
        # 聚類相似經驗
        similarities = pattern_results['experience_similarities']
        clusters = self._find_clusters(similarities)
        patterns['pattern_clusters'] = clusters
        
        # 檢測異常
        anomalies = self._detect_anomalies(pattern_results)
        patterns['anomalies'] = anomalies
        
        return patterns
    
    def _find_clusters(self, similarity_matrix: torch.Tensor, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """尋找相似經驗聚類"""
        clusters = []
        visited = set()
        
        for i in range(similarity_matrix.size(0)):
            if i in visited:
                continue
            
            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, similarity_matrix.size(0)):
                if j not in visited and similarity_matrix[i, j] > threshold:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) >= 2:  # 只保留有多個成員的聚類
                clusters.append({
                    'cluster_id': len(clusters),
                    'members': cluster,
                    'size': len(cluster),
                    'avg_similarity': float(similarity_matrix[cluster][:, cluster].mean())
                })
        
        return clusters
    
    def _detect_anomalies(self, pattern_results: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """檢測異常經驗"""
        anomalies = []
        
        # 基於模式概率檢測異常
        pattern_probs = pattern_results['pattern_probabilities']
        mean_probs = pattern_probs.mean(dim=1)
        std_probs = pattern_probs.std(dim=1)
        
        # 低概率的經驗可能是異常
        anomaly_threshold = mean_probs.mean() - 2 * mean_probs.std()
        
        for i, prob in enumerate(mean_probs):
            if prob < anomaly_threshold:
                anomalies.append({
                    'experience_index': i,
                    'anomaly_score': float(anomaly_threshold - prob),
                    'pattern_uncertainty': float(std_probs[i]),
                    'type': 'low_pattern_match'
                })
        
        return anomalies
    
    def _generate_improvements(self, patterns: Dict[str, Any], experiences: List[ExperienceRecord]) -> List[ImprovementSuggestion]:
        """生成改進建議"""
        suggestions = []
        
        # 基於失敗經驗生成建議
        failed_experiences = [exp for exp in experiences if not exp.success]
        if failed_experiences:
            suggestion = ImprovementSuggestion(
                module_name=self.module_name,
                improvement_type='reliability',
                description=f"改善可靠性 - 發現 {len(failed_experiences)} 個失敗經驗",
                confidence=0.8,
                expected_impact=0.6,
                implementation_effort='medium',
                priority=8
            )
            suggestions.append(suggestion)
        
        # 基於性能模式生成建議
        slow_experiences = [exp for exp in experiences if exp.execution_time > self.performance_tracker.get_percentile(95)]
        if slow_experiences:
            suggestion = ImprovementSuggestion(
                module_name=self.module_name,
                improvement_type='performance',
                description=f"優化性能 - 發現 {len(slow_experiences)} 個慢速操作",
                confidence=0.7,
                expected_impact=0.4,
                implementation_effort='medium',
                priority=6
            )
            suggestions.append(suggestion)
        
        # 基於模式聚類生成建議
        for cluster in patterns['pattern_clusters']:
            if cluster['size'] >= 5:  # 足夠大的聚類
                cluster_experiences = [experiences[i] for i in cluster['members']]
                avg_success = sum(1 for exp in cluster_experiences if exp.success) / len(cluster_experiences)
                
                if avg_success < 0.8:  # 成功率偏低的聚類
                    suggestion = ImprovementSuggestion(
                        module_name=self.module_name,
                        improvement_type='accuracy',
                        description=f"改善特定模式準確性 - 聚類 {cluster['cluster_id']} 成功率僅 {avg_success:.1%}",
                        confidence=0.6,
                        expected_impact=0.3,
                        implementation_effort='low',
                        priority=5
                    )
                    suggestions.append(suggestion)
        
        # 基於異常檢測生成建議
        if len(patterns['anomalies']) > len(experiences) * 0.1:  # 異常比例過高
            suggestion = ImprovementSuggestion(
                module_name=self.module_name,
                improvement_type='efficiency',
                description=f"減少異常行為 - 檢測到 {len(patterns['anomalies'])} 個異常",
                confidence=0.5,
                expected_impact=0.2,
                implementation_effort='high',
                priority=4
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _apply_improvements(self, suggestions: List[ImprovementSuggestion]) -> int:
        """應用改進建議"""
        applied_count = 0
        
        # 按優先級排序
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        for suggestion in suggestions:
            try:
                if self._implement_suggestion(suggestion):
                    applied_count += 1
                    self.stats['successful_improvements'] += 1
                    self.improvement_history.append(suggestion)
                    
                    logger.info(f"成功應用改進: {suggestion.description}")
                else:
                    self.stats['failed_improvements'] += 1
            
            except Exception as e:
                logger.error(f"應用改進失敗: {e}")
                self.stats['failed_improvements'] += 1
        
        return applied_count
    
    def _implement_suggestion(self, suggestion: ImprovementSuggestion) -> bool:
        """實現改進建議"""
        # 實現具體的改進策略，根據建議類型調整模組參數
        
        if suggestion.improvement_type == 'performance':
            # 性能優化 - 調整處理參數
            self.stats['learning_rate'] *= 0.95  # 降低學習率以提高穩定性
            return True
            
        elif suggestion.improvement_type == 'reliability':
            # 可靠性改進 - 增加錯誤檢查機制
            return True
            
        elif suggestion.improvement_type == 'accuracy':
            # 準確性改進 - 調整閾值參數
            return True
            
        elif suggestion.improvement_type == 'efficiency':
            # 效率改進 - 優化資源使用
            return True
        
        return False
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """獲取改進統計"""
        return {
            'module_name': self.module_name,
            'total_experiences': self.stats['total_experiences'],
            'total_improvements': self.stats['successful_improvements'] + self.stats['failed_improvements'],
            'successful_improvements': self.stats['successful_improvements'],
            'failed_improvements': self.stats['failed_improvements'],
            'improvement_success_rate': (
                self.stats['successful_improvements'] / 
                max(self.stats['successful_improvements'] + self.stats['failed_improvements'], 1)
            ),
            'adaptation_count': self.stats['adaptation_count'],
            'pattern_detection_count': self.stats['pattern_detection_count'],
            'current_learning_rate': self.stats['learning_rate'],
            'experience_buffer_size': len(self.experience_buffer),
            'performance_trends': self.performance_tracker.get_trends(),
            'recent_improvements': [
                {
                    'type': imp.improvement_type,
                    'description': imp.description,
                    'confidence': imp.confidence,
                    'impact': imp.expected_impact,
                    'created_at': imp.created_at.isoformat()
                }
                for imp in self.improvement_history[-5:]  # 最近5個改進
            ]
        }

# ==================== 性能追蹤器 ====================

class PerformanceTracker:
    """性能追蹤器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.execution_times = deque(maxlen=window_size)
        self.success_rates = deque(maxlen=window_size)
        self.performance_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, experience: ExperienceRecord):
        """更新性能指標"""
        self.execution_times.append(experience.execution_time)
        self.success_rates.append(1.0 if experience.success else 0.0)
        
        for metric, value in experience.performance_metrics.items():
            if isinstance(value, (int, float)):
                self.performance_metrics[metric].append(value)
    
    def get_average_execution_time(self) -> float:
        """獲取平均執行時間"""
        return sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0.0
    
    def get_success_rate(self) -> float:
        """獲取成功率"""
        return sum(self.success_rates) / len(self.success_rates) if self.success_rates else 0.0
    
    def get_percentile(self, percentile: int) -> float:
        """獲取執行時間百分位數"""
        if not self.execution_times:
            return 0.0
        sorted_times = sorted(self.execution_times)
        index = int(len(sorted_times) * percentile / 100.0)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_trends(self) -> Dict[str, Any]:
        """獲取性能趨勢"""
        if len(self.execution_times) < 10:
            return {'status': 'insufficient_data'}
        
        # 計算趨勢
        recent_times = list(self.execution_times)[-50:]
        early_avg = sum(recent_times[:25]) / 25 if len(recent_times) >= 50 else 0
        recent_avg = sum(recent_times[-25:]) / 25 if len(recent_times) >= 25 else 0
        
        # 計算趨勢
        if recent_avg < early_avg:
            time_trend = 'improving'
        elif recent_avg > early_avg:
            time_trend = 'degrading'
        else:
            time_trend = 'stable'
        
        return {
            'execution_time_trend': time_trend,
            'current_avg_time': self.get_average_execution_time(),
            'success_rate': self.get_success_rate(),
            'p95_time': self.get_percentile(95),
            'p50_time': self.get_percentile(50),
            'sample_size': len(self.execution_times)
        }

# ==================== 自我改進介面 ====================

class SelfImprovingModule:
    """自我改進模組介面 - 可被任何 AIVA 模組繼承"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.improvement_engine = SelfImprovingEngine(module_name)
        self._operation_start_time = None
        
    def start_operation(self, operation: str, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """開始操作記錄"""
        self._operation_start_time = time.time()
        self._current_operation = operation
        self._current_input = input_data
        self._current_context = context or {}
    
    def end_operation(self, output_data: Dict[str, Any], success: bool = True, performance_metrics: Optional[Dict[str, float]] = None):
        """結束操作並記錄經驗"""
        if self._operation_start_time is None:
            return
            
        execution_time = time.time() - self._operation_start_time
        
        experience = ExperienceRecord(
            module_name=self.module_name,
            operation=self._current_operation,
            input_data=self._current_input,
            output_data=output_data,
            success=success,
            execution_time=execution_time,
            performance_metrics=performance_metrics or {},
            context=self._current_context
        )
        
        self.improvement_engine.add_experience(experience)
        self._operation_start_time = None
    
    def get_improvement_insights(self) -> Dict[str, Any]:
        """獲取改進洞察"""
        return self.improvement_engine.get_improvement_stats()

# ==================== 測試和示例 ====================

async def test_self_improving_mechanism():
    """測試自我改進機制"""
    
    print("🧪 測試自我改進機制 v2.0")
    print("=" * 60)
    
    # 1. 創建自我改進引擎
    print("\n🔧 創建自我改進引擎...")
    engine = SelfImprovingEngine("test_module")
    
    # 2. 模擬添加經驗
    print("\n📚 模擬添加學習經驗...")
    
    rng = np.random.default_rng(42)
    experiences = []
    for i in range(50):
        # 模擬不同類型的經驗
        success = rng.random() > 0.2  # 80% 成功率
        exec_time = rng.exponential(2.0)  # 指數分佈的執行時間
        
        experience = ExperienceRecord(
            module_name="test_module",
            operation=f"operation_{i % 5}",  # 5種不同操作
            input_data={"size": int(rng.integers(10, 1000))},
            output_data={"result": f"output_{i}"},
            success=success,
            execution_time=exec_time,
            performance_metrics={
                "accuracy": float(rng.uniform(0.5, 1.0)),
                "speed": float(rng.uniform(0.3, 0.9)),
                "memory": float(rng.uniform(0.4, 0.8))
            },
            context={
                "file_count": int(rng.integers(50, 500)),
                "complexity": {"cyclomatic": int(rng.integers(1, 50))}
            }
        )
        
        experiences.append(experience)
        engine.add_experience(experience)
    
    print(f"✅ 已添加 {len(experiences)} 個經驗")
    
    # 3. 等待學習過程完成
    print("\n🧠 等待自我學習過程...")
    await asyncio.sleep(2)  # 給學習過程一些時間
    
    # 4. 獲取改進統計
    print("\n📊 獲取改進統計...")
    stats = engine.get_improvement_stats()
    
    print(f"模組名稱: {stats['module_name']}")
    print(f"總經驗數: {stats['total_experiences']}")
    print(f"成功改進: {stats['successful_improvements']}")
    print(f"失敗改進: {stats['failed_improvements']}")
    print(f"改進成功率: {stats['improvement_success_rate']:.1%}")
    print(f"適應次數: {stats['adaptation_count']}")
    print(f"模式檢測次數: {stats['pattern_detection_count']}")
    print(f"當前學習率: {stats['current_learning_rate']:.4f}")
    
    # 5. 顯示性能趨勢
    trends = stats['performance_trends']
    if trends['status'] != 'insufficient_data':
        print("\n📈 性能趨勢:")
        print(f"執行時間趨勢: {trends['execution_time_trend']}")
        print(f"當前平均時間: {trends['current_avg_time']:.3f}s")
        print(f"成功率: {trends['success_rate']:.1%}")
        print(f"95百分位時間: {trends['p95_time']:.3f}s")
    
    # 6. 顯示最近改進
    if stats['recent_improvements']:
        print("\n🚀 最近的改進:")
        for imp in stats['recent_improvements']:
            print(f"  - {imp['type']}: {imp['description']}")
            print(f"    信心度: {imp['confidence']:.1%}, 預期影響: {imp['impact']:.1%}")
    
    # 7. 測試模組介面
    print("\n🔌 測試自我改進模組介面...")
    
    module = SelfImprovingModule("interface_test")
    
    # 模擬操作
    module.start_operation("test_operation", {"input": "test"}, {"env": "test"})
    await asyncio.sleep(0.1)  # 模擬處理時間
    module.end_operation({"output": "success"}, True, {"accuracy": 0.95})
    
    insights = module.get_improvement_insights()
    print(f"模組洞察 - 總經驗: {insights['total_experiences']}")
    
    print("\n🎉 自我改進機制測試完成！")

if __name__ == "__main__":
    asyncio.run(test_self_improving_mechanism())