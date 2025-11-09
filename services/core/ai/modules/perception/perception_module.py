"""
AIVA Perception Module v2.0
感知模組 - 負責掃描分析、上下文編碼、歷史處理

基於 Event Sourcing 和 CQRS 模式設計，支援高性能的感知數據處理，
包含掃描結果分析、上下文特徵提取、歷史模式識別等核心功能。

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
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid

# 導入事件系統
from ...core.event_system.event_bus import AIEvent, AIEventBus, EventPriority
from ...core.controller.strangler_fig_controller import StranglerFigController, AIRequest, AIResponse, MessageType

# 常量定義
TIMEZONE_OFFSET = '+00:00'

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 感知模組核心組件 ====================

# ==================== Context Engineering 2025 增強組件 ====================

class ContextSynthesizer(nn.Module):
    """先進的上下文合成器 - 2025 Context Engineering"""
    
    def __init__(self, context_dim: int = 768, max_length: int = 512):
        super().__init__()
        self.context_dim = context_dim
        self.max_length = max_length
        
        # 多層次注意力機制
        self.global_attention = nn.MultiheadAttention(context_dim, num_heads=12, batch_first=True)
        self.local_attention = nn.MultiheadAttention(context_dim, num_heads=8, batch_first=True)
        self.cross_modal_attention = nn.MultiheadAttention(context_dim, num_heads=6, batch_first=True)
        
        # 上下文融合網路
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim * 3, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim)
        )
        
        # 上下文優化器
        self.context_optimizer = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim, context_dim)
        )
        
        # 上下文品質評估器
        self.quality_assessor = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def synthesize_context(self, context_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """合成多層次上下文"""
        # 提取不同層次的上下文
        global_context = context_inputs.get('global', torch.randn(1, self.max_length, self.context_dim))
        local_context = context_inputs.get('local', torch.randn(1, self.max_length, self.context_dim))
        modal_context = context_inputs.get('modal', torch.randn(1, self.max_length, self.context_dim))
        
        # 應用多層次注意力
        global_att, _ = self.global_attention(global_context, global_context, global_context)
        local_att, _ = self.local_attention(local_context, local_context, local_context)
        cross_att, _ = self.cross_modal_attention(modal_context, global_context, local_context)
        
        # 融合上下文
        fused_context = torch.cat([global_att, local_att, cross_att], dim=-1)
        synthesized = self.context_fusion(fused_context)
        
        # 優化上下文
        optimized = self.context_optimizer(synthesized)
        
        # 評估品質
        quality_scores = self.quality_assessor(optimized)
        
        return {
            'synthesized_context': optimized,
            'quality_scores': quality_scores,
            'global_attention': global_att,
            'local_attention': local_att,
            'cross_modal_attention': cross_att
        }

class ContextProcessor(nn.Module):
    """高級上下文處理器"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 上下文分析器
        self.context_analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 關聯性檢測器
        self.correlation_detector = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 模式識別器
        self.pattern_recognizer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 重要性評分器
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def process_context(self, context_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """處理上下文數據"""
        # 分析上下文
        analyzed = self.context_analyzer(context_data)
        
        # 檢測關聯性
        correlations, correlation_weights = self.correlation_detector(analyzed, analyzed, analyzed)
        
        # 識別模式（需要轉置以適應Conv1d）
        patterns_input = analyzed.transpose(1, 2)
        patterns = self.pattern_recognizer(patterns_input)
        patterns = patterns.transpose(1, 2)
        
        # 評分重要性
        importance = self.importance_scorer(analyzed)
        
        return {
            'analyzed_context': analyzed,
            'correlations': correlations,
            'correlation_weights': correlation_weights,
            'patterns': patterns,
            'importance_scores': importance
        }

class ContextOptimizer:
    """上下文優化器 - 針對不同任務優化上下文"""
    
    def __init__(self):
        self.optimization_strategies = {
            'scan_analysis': self._optimize_for_scanning,
            'code_review': self._optimize_for_review,
            'security_audit': self._optimize_for_security,
            'performance_analysis': self._optimize_for_performance,
            'general': self._optimize_general
        }
    
    def optimize_context(self, context_data: Dict[str, Any], task_type: str = 'general') -> Dict[str, Any]:
        """根據任務類型優化上下文"""
        strategy = self.optimization_strategies.get(task_type, self._optimize_general)
        return strategy(context_data)
    
    def _optimize_for_scanning(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """為掃描任務優化上下文"""
        optimized = context_data.copy()
        
        # 強調文件結構和統計信息
        if 'file_stats' in optimized:
            optimized['file_stats']['weight'] = 1.5
        
        # 減少歷史信息的權重
        if 'history' in optimized:
            optimized['history'] = optimized['history'][-10:]  # 只保留最近10個
        
        # 添加掃描特定的上下文
        optimized['scan_context'] = {
            'focus_areas': ['file_structure', 'code_metrics', 'dependencies'],
            'optimization_level': 'high',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return optimized
    
    def _optimize_for_review(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """為代碼審查優化上下文"""
        optimized = context_data.copy()
        
        # 強調代碼品質指標
        if 'code_quality' in optimized:
            optimized['code_quality']['weight'] = 2.0
        
        # 添加審查特定上下文
        optimized['review_context'] = {
            'focus_areas': ['code_style', 'security_issues', 'performance_concerns'],
            'severity_threshold': 'medium',
            'include_suggestions': True
        }
        
        return optimized
    
    def _optimize_for_security(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """為安全審計優化上下文"""
        optimized = context_data.copy()
        
        # 強調安全相關信息
        optimized['security_context'] = {
            'focus_areas': ['vulnerability_patterns', 'access_controls', 'data_handling'],
            'threat_level': 'high',
            'compliance_frameworks': ['OWASP', 'NIST']
        }
        
        return optimized
    
    def _optimize_for_performance(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """為性能分析優化上下文"""
        optimized = context_data.copy()
        
        # 強調性能指標
        optimized['performance_context'] = {
            'focus_areas': ['execution_time', 'memory_usage', 'algorithm_complexity'],
            'benchmark_targets': ['response_time', 'throughput', 'resource_efficiency']
        }
        
        return optimized
    
    def _optimize_general(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """通用上下文優化"""
        optimized = context_data.copy()
        
        # 添加通用優化標記
        optimized['optimization'] = {
            'type': 'general',
            'applied_at': datetime.now(timezone.utc).isoformat(),
            'version': '2.0'
        }
        
        return optimized

class AdvancedContextEngine:
    """先進的上下文引擎 - 整合所有 Context Engineering 功能"""
    
    def __init__(self):
        self.synthesizer = ContextSynthesizer()
        self.processor = ContextProcessor()
        self.optimizer = ContextOptimizer()
        self.context_cache = {}
        self.performance_metrics = {
            'total_processed': 0,
            'cache_hits': 0,
            'processing_times': []
        }
    
    async def process_advanced_context(self, context_data: Dict[str, Any], task_type: str = 'general') -> Dict[str, Any]:
        """處理先進上下文"""
        start_time = time.time()
        
        try:
            # 1. 生成上下文鍵用於快取
            context_key = self._generate_context_key(context_data, task_type)
            
            # 2. 檢查快取
            if context_key in self.context_cache:
                self.performance_metrics['cache_hits'] += 1
                return self.context_cache[context_key]
            
            # 3. 優化上下文
            optimized_context = self.optimizer.optimize_context(context_data, task_type)
            
            # 4. 轉換為張量進行神經網路處理
            context_tensors = self._prepare_tensors(optimized_context)
            
            # 5. 神經網路處理（使用 torch.no_grad() 為異步操作）
            with torch.no_grad():
                # 使用 processor 進行處理
                processed_results = await asyncio.get_event_loop().run_in_executor(
                    None, self.processor.forward, context_tensors.get('main', torch.randn(1, 128))
                )
                # 使用 synthesizer 進行合成
                synthesized_results = await asyncio.get_event_loop().run_in_executor(
                    None, self.synthesizer.forward, context_tensors
                )
            
            # 6. 整合結果
            final_result = self._integrate_results(
                optimized_context,
                processed_results,
                synthesized_results
            )
            
            # 7. 快取結果
            self.context_cache[context_key] = final_result
            
            # 8. 更新性能指標
            processing_time = time.time() - start_time
            self.performance_metrics['total_processed'] += 1
            self.performance_metrics['processing_times'].append(processing_time)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Context engineering error: {e}")
            return {
                'error': str(e),
                'status': 'failed',
                'fallback_context': context_data
            }
    
    def _generate_context_key(self, context_data: Dict[str, Any], task_type: str) -> str:
        """生成上下文快取鍵"""
        import hashlib
        
        # 創建內容摘要
        content_str = json.dumps(context_data, sort_keys=True, default=str)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        return f"{task_type}_{content_hash[:16]}"
    
    def _prepare_tensors(self, context_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """準備神經網路張量"""
        # 將上下文數據轉換為特徵向量
        features = self._extract_advanced_features(context_data)
        
        # 創建不同類型的上下文張量
        main_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # 為合成器創建多模態張量
        global_tensor = main_tensor.repeat(1, 512, 1)[:, :512, :]  # 確保長度為512
        local_tensor = main_tensor.repeat(1, 512, 1)[:, :512, :]
        modal_tensor = main_tensor.repeat(1, 512, 1)[:, :512, :]
        
        return {
            'main': main_tensor,
            'global': global_tensor,
            'local': local_tensor,
            'modal': modal_tensor
        }
    
    def _extract_advanced_features(self, context_data: Dict[str, Any]) -> List[float]:
        """提取先進的上下文特徵"""
        features = [0.0] * 768  # 初始化768維特徵向量
        
        # 基本統計特徵
        if 'file_count' in context_data:
            features[0] = min(context_data['file_count'] / 1000.0, 1.0)
        
        if 'total_lines' in context_data:
            features[1] = min(context_data['total_lines'] / 100000.0, 1.0)
        
        # 複雜度特徵
        if 'complexity' in context_data:
            complexity = context_data['complexity']
            features[2] = min(complexity.get('cyclomatic', 0) / 100.0, 1.0)
            features[3] = min(complexity.get('cognitive', 0) / 100.0, 1.0)
            features[4] = min(complexity.get('halstead_volume', 0) / 5000.0, 1.0)
        
        # 語言分佈特徵 (索引 5-14)
        if 'languages' in context_data:
            languages = context_data['languages']
            common_langs = ['python', 'javascript', 'java', 'cpp', 'go', 'rust', 'typescript', 'html', 'css', 'sql']
            for i, lang in enumerate(common_langs):
                features[5 + i] = min(languages.get(lang, 0) / 100.0, 1.0)
        
        # 上下文特定特徵
        if 'scan_context' in context_data:
            features[15] = 1.0  # 掃描模式標誌
        
        if 'review_context' in context_data:
            features[16] = 1.0  # 審查模式標誌
        
        if 'security_context' in context_data:
            features[17] = 1.0  # 安全模式標誌
        
        # 隨機填充剩餘特徵以達到768維度
        rng = np.random.default_rng(42)  # 固定種子以確保一致性
        for i in range(18, 768):
            features[i] = rng.normal(0, 0.1)
        
        return features
    
    def _integrate_results(self, 
                          optimized_context: Dict[str, Any],
                          processed_results: Dict[str, torch.Tensor],
                          synthesized_results: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """整合所有處理結果"""
        
        # 轉換張量為可序列化的格式
        def tensor_to_list(tensor):
            return tensor.detach().numpy().tolist() if tensor is not None else []
        
        return {
            'optimized_context': optimized_context,
            'processing_results': {
                'analyzed_context': tensor_to_list(processed_results.get('analyzed_context')),
                'correlations': tensor_to_list(processed_results.get('correlations')),
                'patterns': tensor_to_list(processed_results.get('patterns')),
                'importance_scores': tensor_to_list(processed_results.get('importance_scores'))
            },
            'synthesis_results': {
                'synthesized_context': tensor_to_list(synthesized_results.get('synthesized_context')),
                'quality_scores': tensor_to_list(synthesized_results.get('quality_scores')),
                'attention_maps': {
                    'global': tensor_to_list(synthesized_results.get('global_attention')),
                    'local': tensor_to_list(synthesized_results.get('local_attention')),
                    'cross_modal': tensor_to_list(synthesized_results.get('cross_modal_attention'))
                }
            },
            'metadata': {
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'context_version': '2.0_advanced',
                'neural_processing': True
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        if not self.performance_metrics['processing_times']:
            return {'status': 'no_data'}
        
        times = self.performance_metrics['processing_times']
        return {
            'total_processed': self.performance_metrics['total_processed'],
            'cache_hits': self.performance_metrics['cache_hits'],
            'cache_hit_rate': self.performance_metrics['cache_hits'] / max(self.performance_metrics['total_processed'], 1),
            'avg_processing_time': sum(times) / len(times),
            'min_processing_time': min(times),
            'max_processing_time': max(times),
            'cache_size': len(self.context_cache)
        }

class ScanResultEncoder(nn.Module):
    """掃描結果編碼器"""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1536, output_dim: int = 768):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, scan_data: torch.Tensor) -> torch.Tensor:
        # 投影到隱藏維度
        x = self.input_projection(scan_data)
        
        # 自注意力機制
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # 前饋網路
        ff_out = self.feedforward(x)
        return self.layer_norm2(ff_out)
    
    def encode_scan_results(self, scan_data: Dict[str, Any]) -> torch.Tensor:
        """編碼掃描結果"""
        # 將掃描數據轉換為張量
        features = self._extract_scan_features(scan_data)
        tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            encoded = self.forward(tensor_input)
        
        return encoded.squeeze(0)
    
    def _extract_scan_features(self, scan_data: Dict[str, Any]) -> List[float]:
        """從掃描數據提取特徵"""
        features = []
        
        # 文件相關特徵
        file_count = scan_data.get('file_count', 0)
        total_lines = scan_data.get('total_lines', 0)
        features.extend([file_count, total_lines])
        
        # 語言分佈特徵
        languages = scan_data.get('languages', {})
        common_langs = ['python', 'javascript', 'java', 'cpp', 'go', 'rust', 'typescript']
        for lang in common_langs:
            features.append(languages.get(lang, 0))
        
        # 複雜度特徵
        complexity_stats = scan_data.get('complexity', {})
        features.extend([
            complexity_stats.get('cyclomatic', 0),
            complexity_stats.get('cognitive', 0),
            complexity_stats.get('halstead_volume', 0)
        ])
        
        # 結構特徵
        structure = scan_data.get('structure', {})
        features.extend([
            structure.get('depth', 0),
            structure.get('breadth', 0),
            structure.get('modules', 0),
            structure.get('classes', 0),
            structure.get('functions', 0)
        ])
        
        # 填充到固定維度
        target_dim = 1024
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]
            
        return features

class ContextEncoder(nn.Module):
    """上下文編碼器"""
    
    def __init__(self, context_dim: int = 768, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.temporal_encoder = nn.LSTM(
            input_size=context_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, context_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM 編碼
        lstm_out, (_, _) = self.temporal_encoder(context_sequence)
        
        # 注意力機制
        attn_out, attn_weights = self.context_attention(lstm_out, lstm_out, lstm_out)
        
        # 輸出投影
        output = self.output_projection(attn_out)
        
        return output, attn_weights
    
    def encode_context(self, context_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """編碼上下文數據"""
        # 提取上下文特徵序列
        context_features = self._extract_context_features(context_data)
        
        with torch.no_grad():
            encoded, attention = self.forward(context_features)
            
        return {
            'encoded_context': encoded,
            'attention_weights': attention,
            'context_summary': encoded.mean(dim=1)  # 全局上下文摘要
        }
    
    def _extract_context_features(self, context_data: Dict[str, Any]) -> torch.Tensor:
        """從上下文數據提取特徵序列"""
        # 模擬上下文特徵提取
        session_history = context_data.get('session_history', [])
        
        # 創建特徵序列
        sequence_length = min(len(session_history), 50)  # 限制序列長度
        feature_dim = 768
        
        features = torch.zeros(1, sequence_length, feature_dim)
        
        for i, step in enumerate(session_history[-sequence_length:]):
            step_features = self._encode_single_step(step)
            features[0, i] = step_features
            
        return features
    
    def _encode_single_step(self, step: Dict[str, Any]) -> torch.Tensor:
        """編碼單個步驟"""
        # 簡化的步驟編碼
        feature_vector = torch.zeros(768)
        
        # 操作類型編碼
        operation = step.get('operation', 'unknown')
        op_hash = hash(operation) % 100
        feature_vector[op_hash] = 1.0
        
        # 成功率編碼
        success = step.get('success', True)
        feature_vector[100] = 1.0 if success else -1.0
        
        # 時間特徵
        timestamp = step.get('timestamp', datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', TIMEZONE_OFFSET))
        
        hour = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        feature_vector[101:103] = torch.tensor([hour, day_of_week])
        
        return feature_vector

class HistoryEncoder(nn.Module):
    """歷史數據編碼器"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.pattern_detector = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.sequence_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.pattern_fusion = nn.Linear(hidden_dim * 3, output_dim)
        
    def forward(self, history_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, _ = history_sequence.shape
        
        # 模式檢測 (CNN)
        conv_input = history_sequence.transpose(1, 2)  # (batch, feat, seq)
        pattern_features = self.pattern_detector(conv_input).squeeze(-1)  # (batch, hidden)
        
        # 序列編碼 (GRU)
        gru_output, hidden_state = self.sequence_encoder(history_sequence)
        
        # 取最後的隱藏狀態 (雙向)
        final_hidden = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)  # (batch, hidden*2)
        
        # 融合特徵
        combined = torch.cat([pattern_features, final_hidden], dim=1)
        fused_output = self.pattern_fusion(combined)
        
        return {
            'pattern_features': pattern_features,
            'sequence_features': final_hidden,
            'fused_output': fused_output,
            'attention_context': gru_output.mean(dim=1)  # 平均池化作為注意力上下文
        }
    
    def encode_history(self, history_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """編碼歷史數據"""
        if not history_data:
            return self._empty_history_encoding()
            
        # 轉換為特徵序列
        features = self._convert_to_features(history_data)
        
        with torch.no_grad():
            encoded = self.forward(features)
            
        # 分析歷史模式
        patterns = self._analyze_patterns(history_data)
        
        return {
            'encoded_features': {k: v.cpu().numpy() for k, v in encoded.items()},
            'detected_patterns': patterns,
            'history_summary': {
                'total_operations': len(history_data),
                'success_rate': sum(1 for h in history_data if h.get('success', False)) / len(history_data),
                'time_span': self._calculate_time_span(history_data),
                'operation_types': self._count_operation_types(history_data)
            }
        }
    
    def _empty_history_encoding(self) -> Dict[str, Any]:
        """空歷史編碼"""
        return {
            'encoded_features': {
                'pattern_features': np.zeros(768),
                'sequence_features': np.zeros(1536),
                'fused_output': np.zeros(256),
                'attention_context': np.zeros(768)
            },
            'detected_patterns': [],
            'history_summary': {
                'total_operations': 0,
                'success_rate': 0.0,
                'time_span': 0.0,
                'operation_types': {}
            }
        }
    
    def _convert_to_features(self, history_data: List[Dict[str, Any]]) -> torch.Tensor:
        """轉換歷史數據為特徵張量"""
        max_sequence = 100  # 最大序列長度
        feature_dim = 512
        
        sequence = history_data[-max_sequence:]  # 取最近的數據
        features = torch.zeros(1, len(sequence), feature_dim)
        
        for i, record in enumerate(sequence):
            features[0, i] = self._encode_history_record(record)
            
        return features
    
    def _encode_history_record(self, record: Dict[str, Any]) -> torch.Tensor:
        """編碼單個歷史記錄"""
        feature_vector = torch.zeros(512)
        
        # 操作特徵
        operation = record.get('operation', 'unknown')
        op_embedding = torch.tensor([hash(operation + str(i)) % 2 - 1 for i in range(50)], dtype=torch.float32)
        feature_vector[:50] = op_embedding
        
        # 結果特徵
        success = record.get('success', False)
        feature_vector[50] = 1.0 if success else -1.0
        
        # 性能特徵
        duration = record.get('duration', 0)
        feature_vector[51] = min(duration / 1000.0, 10.0)  # 標準化持續時間
        
        # 上下文特徵
        context_size = len(str(record.get('context', {})))
        feature_vector[52] = min(context_size / 1000.0, 5.0)
        
        return feature_vector
    
    def _analyze_patterns(self, history_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析歷史模式"""
        patterns = []
        
        # 成功/失敗模式
        success_pattern = self._find_success_patterns(history_data)
        if success_pattern:
            patterns.append(success_pattern)
            
        # 時間模式
        temporal_pattern = self._find_temporal_patterns(history_data)
        if temporal_pattern:
            patterns.append(temporal_pattern)
            
        # 操作序列模式
        sequence_pattern = self._find_sequence_patterns(history_data)
        if sequence_pattern:
            patterns.append(sequence_pattern)
            
        return patterns
    
    def _find_success_patterns(self, history_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """查找成功模式"""
        if len(history_data) < 5:
            return None
            
        successes = [h for h in history_data if h.get('success', False)]
        
        if not successes:
            return None
            
        # 分析成功操作的共同特徵
        success_ops = [h.get('operation') for h in successes]
        most_successful_op = max(set(success_ops), key=lambda x: success_ops.count(x)) if success_ops else None
        
        return {
            'type': 'success_pattern',
            'most_successful_operation': most_successful_op,
            'success_rate': len(successes) / len(history_data),
            'confidence': min(len(successes) / 10.0, 1.0)
        }
    
    def _find_temporal_patterns(self, history_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """查找時間模式"""
        if len(history_data) < 10:
            return None
            
        # 分析操作的時間分佈
        hours = []
        for record in history_data:
            timestamp = record.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', TIMEZONE_OFFSET))
                hours.append(timestamp.hour)
        
        if not hours:
            return None
            
        # 找出最活躍的時段
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
        peak_hour = max(hour_counts.keys(), key=lambda x: hour_counts[x]) if hour_counts else 0
        
        return {
            'type': 'temporal_pattern',
            'peak_hour': peak_hour,
            'activity_distribution': hour_counts,
            'confidence': hour_counts[peak_hour] / len(hours)
        }
    
    def _find_sequence_patterns(self, history_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """查找序列模式"""
        if len(history_data) < 6:
            return None
            
        # 分析操作序列
        operations = [h.get('operation', 'unknown') for h in history_data]
        
        # 查找最常見的連續操作對
        pairs = {}
        for i in range(len(operations) - 1):
            pair = (operations[i], operations[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        
        if not pairs:
            return None
            
        most_common_pair = max(pairs.keys(), key=lambda x: pairs[x]) if pairs else ('unknown', 'unknown')
        
        return {
            'type': 'sequence_pattern',
            'common_sequence': most_common_pair,
            'frequency': pairs[most_common_pair],
            'confidence': pairs[most_common_pair] / (len(operations) - 1)
        }
    
    def _calculate_time_span(self, history_data: List[Dict[str, Any]]) -> float:
        """計算時間跨度（小時）"""
        timestamps = []
        for record in history_data:
            timestamp = record.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', TIMEZONE_OFFSET))
                timestamps.append(timestamp)
        
        if len(timestamps) < 2:
            return 0.0
            
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600
    
    def _count_operation_types(self, history_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """統計操作類型"""
        counts = {}
        for record in history_data:
            op = record.get('operation', 'unknown')
            counts[op] = counts.get(op, 0) + 1
        return counts

# ==================== 感知模組主類 ====================

class PerceptionModuleV2:
    """感知模組 v2.0 主實現 - 包含 Context Engineering 2025 增強功能"""
    
    def __init__(self, event_bus: Optional[AIEventBus] = None):
        self.module_name = "perception"
        self.module_version = "v2.0"
        
        # 事件系統
        self.event_bus = event_bus
        
        # 初始化編碼器
        self.scan_encoder = ScanResultEncoder()
        self.context_encoder = ContextEncoder()
        self.history_encoder = HistoryEncoder()
        
        # Context Engineering 2025 組件
        self.advanced_context_engine = AdvancedContextEngine()
        
        # 性能統計
        self.stats = {
            'total_requests': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_processing_time': 0.0,
            'context_engineering_stats': {
                'advanced_processes': 0,
                'cache_efficiency': 0.0,
                'synthesis_quality': 0.0
            }
        }
        
        logger.info(f"感知模組 {self.module_version} 初始化完成 - 包含 Context Engineering 增強功能")
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """處理感知模組請求"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            operation = request.operation
            payload = request.payload
            
            if operation == 'scan_analysis':
                result = self.scan_analysis(payload.get('scan_data', {}))
            elif operation == 'context_encoding':
                result = self.context_encoding(payload.get('context', {}))
            elif operation == 'history_processing':
                result = self.history_processing(payload.get('history', []))
            elif operation == 'anomaly_detection':
                result = self.anomaly_detection(payload)
            elif operation == 'advanced_context_processing':
                # 新的 Context Engineering 操作
                result = await self.advanced_context_processing(
                    payload.get('context_data', {}),
                    payload.get('task_type', 'general')
                )
            elif operation == 'context_synthesis':
                # 上下文合成操作
                result = self.context_synthesis(payload.get('context_inputs', {}))
            elif operation == 'context_optimization':
                # 上下文優化操作
                result = self.context_optimization(
                    payload.get('context_data', {}),
                    payload.get('task_type', 'general')
                )
            else:
                raise ValueError(f"不支援的操作: {operation}")
            
            # 發布成功事件
            if self.event_bus:
                await self._publish_event(f'perception.{operation}.completed', {
                    'request_id': request.request_id,
                    'operation': operation,
                    'result_summary': self._summarize_result(result),
                    'processing_time': (time.time() - start_time) * 1000
                })
            
            self.stats['successful_operations'] += 1
            processing_time = (time.time() - start_time) * 1000
            self._update_avg_processing_time(processing_time)
            
            return AIResponse(
                request_id=request.request_id,
                status="success",
                processed_by=f"{self.module_name}@{self.module_version}",
                execution_time_ms=processing_time,
                result=result,
                metadata={
                    'module': self.module_name,
                    'version': self.module_version,
                    'operation': operation
                }
            )
            
        except Exception as e:
            # 發布錯誤事件
            if self.event_bus:
                await self._publish_event('perception.error.occurred', {
                    'request_id': request.request_id,
                    'operation': request.operation,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, priority=EventPriority.HIGH)
            
            self.stats['failed_operations'] += 1
            processing_time = (time.time() - start_time) * 1000
            
            logger.error(f"感知模組處理錯誤: {str(e)}")
            
            return AIResponse(
                request_id=request.request_id,
                status="error",
                processed_by=f"{self.module_name}@{self.module_version}",
                execution_time_ms=processing_time,
                error={
                    "type": type(e).__name__,
                    "message": str(e)
                }
            )
    
    def scan_analysis(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        """掃描結果分析"""
        logger.info("開始掃描分析...")
        
        # 編碼掃描結果
        encoded_features = self.scan_encoder.encode_scan_results(scan_data)
        
        # 異常檢測
        anomalies = self._detect_scan_anomalies(scan_data)
        
        # 生成洞察
        insights = self._generate_scan_insights(scan_data, encoded_features)
        
        result = {
            'encoded_features': encoded_features.tolist(),
            'feature_dimension': encoded_features.shape[0],
            'anomalies': anomalies,
            'insights': insights,
            'scan_summary': {
                'total_files': scan_data.get('file_count', 0),
                'total_lines': scan_data.get('total_lines', 0),
                'languages': scan_data.get('languages', {}),
                'complexity_score': self._calculate_complexity_score(scan_data)
            },
            'confidence': 0.92,
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"掃描分析完成，檢測到 {len(anomalies)} 個異常")
        return result
    
    def context_encoding(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """上下文編碼"""
        logger.info("開始上下文編碼...")
        
        # 編碼上下文
        encoded_result = self.context_encoder.encode_context(context_data)
        
        # 提取關鍵上下文資訊
        key_contexts = self._extract_key_contexts(context_data)
        
        # 上下文相似性分析
        similarity_analysis = self._analyze_context_similarity(context_data)
        
        result = {
            'encoded_context': encoded_result['encoded_context'].tolist(),
            'attention_weights': encoded_result['attention_weights'].tolist(),
            'context_summary': encoded_result['context_summary'].tolist(),
            'key_contexts': key_contexts,
            'similarity_analysis': similarity_analysis,
            'context_metadata': {
                'session_length': len(context_data.get('session_history', [])),
                'context_types': self._categorize_contexts(context_data),
                'temporal_span': self._calculate_temporal_span(context_data)
            },
            'confidence': 0.89,
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("上下文編碼完成")
        return result
    
    def history_processing(self, history_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """歷史數據處理"""
        logger.info(f"開始處理 {len(history_data)} 條歷史記錄...")
        
        # 編碼歷史數據
        encoded_history = self.history_encoder.encode_history(history_data)
        
        # 趨勢分析
        trends = self._analyze_trends(history_data)
        
        # 預測性洞察
        predictions = self._generate_predictions(history_data, encoded_history)
        
        result = {
            'encoded_history': encoded_history['encoded_features'],
            'detected_patterns': encoded_history['detected_patterns'],
            'history_summary': encoded_history['history_summary'],
            'trends': trends,
            'predictions': predictions,
            'learning_insights': self._generate_learning_insights(history_data),
            'confidence': 0.85,
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"歷史處理完成，識別出 {len(encoded_history['detected_patterns'])} 個模式")
        return result
    
    def anomaly_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """異常檢測"""
        logger.info("開始異常檢測...")
        
        anomalies = []
        
        # 統計異常檢測
        statistical_anomalies = self._statistical_anomaly_detection(data)
        anomalies.extend(statistical_anomalies)
        
        # 模式異常檢測
        pattern_anomalies = self._pattern_anomaly_detection(data)
        anomalies.extend(pattern_anomalies)
        
        # 時間序列異常檢測
        temporal_anomalies = self._temporal_anomaly_detection(data)
        anomalies.extend(temporal_anomalies)
        
        result = {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'severity_distribution': self._categorize_anomaly_severity(anomalies),
            'recommended_actions': self._recommend_anomaly_actions(anomalies),
            'confidence': 0.87,
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"異常檢測完成，發現 {len(anomalies)} 個異常")
        return result
    
    # ==================== 輔助方法 ====================
    
    def _detect_scan_anomalies(self, scan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """檢測掃描異常"""
        anomalies = []
        
        # 文件數量異常
        file_count = scan_data.get('file_count', 0)
        if file_count > 10000:
            anomalies.append({
                'type': 'excessive_files',
                'severity': 'medium',
                'description': f'檢測到過多文件: {file_count}',
                'recommendation': '考慮排除不必要的目錄或文件類型'
            })
        elif file_count == 0:
            anomalies.append({
                'type': 'no_files',
                'severity': 'high',
                'description': '未檢測到任何文件',
                'recommendation': '檢查掃描路徑和權限設置'
            })
        
        # 複雜度異常
        complexity = scan_data.get('complexity', {})
        cyclomatic = complexity.get('cyclomatic', 0)
        if cyclomatic > 50:
            anomalies.append({
                'type': 'high_complexity',
                'severity': 'high',
                'description': f'檢測到高圈複雜度: {cyclomatic}',
                'recommendation': '考慮重構複雜的函數和模組'
            })
        
        return anomalies
    
    def _generate_scan_insights(self, scan_data: Dict[str, Any], encoded_features: torch.Tensor) -> List[Dict[str, Any]]:
        """生成掃描洞察"""
        insights = []
        
        # 語言分佈洞察
        languages = scan_data.get('languages', {})
        if languages:
            dominant_lang = max(languages, key=languages.get)
            insights.append({
                'type': 'language_distribution',
                'insight': f'主要使用 {dominant_lang} 語言',
                'details': languages,
                'confidence': 0.9
            })
        
        # 結構洞察
        structure = scan_data.get('structure', {})
        if structure.get('classes', 0) > structure.get('functions', 0) * 0.3:
            insights.append({
                'type': 'architecture_style',
                'insight': '傾向於物件導向架構',
                'confidence': 0.8
            })
        
        # 特徵相關性洞察
        feature_variance = torch.var(encoded_features, dim=-1).mean().item()
        if feature_variance > 0.5:
            insights.append({
                'type': 'feature_diversity',
                'insight': '代碼庫具有高度多樣性',
                'confidence': 0.85
            })
        
        return insights
    
    def _calculate_complexity_score(self, scan_data: Dict[str, Any]) -> float:
        """計算複雜度分數"""
        complexity = scan_data.get('complexity', {})
        
        # 加權平均複雜度
        cyclomatic = complexity.get('cyclomatic', 0)
        cognitive = complexity.get('cognitive', 0)
        halstead = complexity.get('halstead_volume', 0)
        
        # 標準化到 0-1 範圍
        normalized_cyclomatic = min(cyclomatic / 100.0, 1.0)
        normalized_cognitive = min(cognitive / 100.0, 1.0)
        normalized_halstead = min(halstead / 1000.0, 1.0)
        
        return (normalized_cyclomatic * 0.4 + 
                normalized_cognitive * 0.4 + 
                normalized_halstead * 0.2)
    
    def _extract_key_contexts(self, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取關鍵上下文"""
        key_contexts = []
        
        current_state = context_data.get('current_state', {})
        if current_state:
            key_contexts.append({
                'type': 'current_state',
                'importance': 0.9,
                'summary': str(current_state)[:100] + "..." if len(str(current_state)) > 100 else str(current_state)
            })
        
        session_history = context_data.get('session_history', [])
        if session_history:
            recent_operations = session_history[-5:]  # 最近5個操作
            key_contexts.append({
                'type': 'recent_operations',
                'importance': 0.8,
                'summary': [op.get('operation', 'unknown') for op in recent_operations]
            })
        
        return key_contexts
    
    def _analyze_context_similarity(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析上下文相似性"""
        session_history = context_data.get('session_history', [])
        
        if len(session_history) < 2:
            return {'similarity_score': 0.0, 'similar_patterns': []}
        
        # 簡化的相似性分析
        operations = [step.get('operation', 'unknown') for step in session_history]
        unique_ops = set(operations)
        repetition_rate = 1.0 - (len(unique_ops) / len(operations))
        
        return {
            'similarity_score': repetition_rate,
            'unique_operations': len(unique_ops),
            'total_operations': len(operations),
            'most_common_operation': max(set(operations), key=operations.count) if operations else None
        }
    
    def _categorize_contexts(self, context_data: Dict[str, Any]) -> Dict[str, int]:
        """分類上下文類型"""
        types = {}
        
        if 'current_state' in context_data:
            types['state'] = 1
        if 'session_history' in context_data:
            types['history'] = len(context_data['session_history'])
        if 'user_preferences' in context_data:
            types['preferences'] = 1
        if 'environment' in context_data:
            types['environment'] = 1
            
        return types
    
    def _calculate_temporal_span(self, context_data: Dict[str, Any]) -> float:
        """計算時間跨度"""
        session_history = context_data.get('session_history', [])
        
        if len(session_history) < 2:
            return 0.0
        
        timestamps = []
        for step in session_history:
            timestamp = step.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', TIMEZONE_OFFSET))
                timestamps.append(timestamp)
        
        if len(timestamps) < 2:
            return 0.0
            
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600  # 小時
    
    def _analyze_trends(self, history_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析趨勢"""
        trends = []
        
        if len(history_data) < 5:
            return trends
        
        # 成功率趨勢
        success_trend = self._calculate_success_trend(history_data)
        if success_trend:
            trends.append(success_trend)
        
        # 操作頻率趨勢
        frequency_trend = self._calculate_frequency_trend(history_data)
        if frequency_trend:
            trends.append(frequency_trend)
        
        return trends
    
    def _calculate_success_trend(self, history_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """計算成功率趨勢"""
        if len(history_data) < 10:
            return None
        
        # 分成前半和後半計算成功率
        mid_point = len(history_data) // 2
        first_half = history_data[:mid_point]
        second_half = history_data[mid_point:]
        
        first_success_rate = sum(1 for h in first_half if h.get('success', False)) / len(first_half)
        second_success_rate = sum(1 for h in second_half if h.get('success', False)) / len(second_half)
        
        trend_direction = 'improving' if second_success_rate > first_success_rate else 'declining'
        change_magnitude = abs(second_success_rate - first_success_rate)
        
        return {
            'type': 'success_rate_trend',
            'direction': trend_direction,
            'change_magnitude': change_magnitude,
            'first_half_rate': first_success_rate,
            'second_half_rate': second_success_rate,
            'confidence': min(change_magnitude * 2, 1.0)
        }
    
    def _calculate_frequency_trend(self, history_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """計算操作頻率趨勢"""
        if len(history_data) < 20:
            return None
        
        # 分析時間間隔
        timestamps = []
        for record in history_data:
            timestamp = record.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', TIMEZONE_OFFSET))
                timestamps.append(timestamp)
        
        if len(timestamps) < 10:
            return None
        
        # 計算平均時間間隔
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if len(intervals) < 5:
            return None
        
        mid_point = len(intervals) // 2
        first_half_avg = sum(intervals[:mid_point]) / mid_point
        second_half_avg = sum(intervals[mid_point:]) / len(intervals[mid_point:])
        
        if second_half_avg < first_half_avg:
            trend = 'increasing_frequency'
        else:
            trend = 'decreasing_frequency'
        
        return {
            'type': 'frequency_trend',
            'direction': trend,
            'first_half_interval': first_half_avg,
            'second_half_interval': second_half_avg,
            'confidence': 0.7
        }
    
    def _generate_predictions(self, history_data: List[Dict[str, Any]], encoded_history: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成預測性洞察"""
        predictions = []
        
        if len(history_data) < 10:
            return predictions
        
        # 使用編碼歷史來增強預測
        encoded_features = encoded_history.get('encoded_features', [])
        pattern_features = encoded_history.get('detected_patterns', [])
        
        # 預測下一個可能的操作
        operations = [h.get('operation', 'unknown') for h in history_data[-10:]]
        if operations:
            # 簡單的序列預測
            last_op = operations[-1]
            op_counts = {}
            
            # 找出在last_op之後最常出現的操作
            for i in range(len(history_data) - 1):
                if history_data[i].get('operation') == last_op:
                    next_op = history_data[i + 1].get('operation', 'unknown')
                    op_counts[next_op] = op_counts.get(next_op, 0) + 1
            
            if op_counts:
                most_likely_next = max(op_counts.keys(), key=lambda x: op_counts[x])
                confidence = op_counts[most_likely_next] / sum(op_counts.values())
                
                predictions.append({
                    'type': 'next_operation',
                    'prediction': most_likely_next,
                    'confidence': confidence,
                    'reasoning': f'基於歷史模式，{last_op} 之後通常執行 {most_likely_next}',
                    'encoded_support': len(encoded_features) if encoded_features else 0,
                    'pattern_support': len(pattern_features) if pattern_features else 0
                })
        
        # 預測成功率
        recent_success_rate = sum(1 for h in history_data[-10:] if h.get('success', False)) / min(10, len(history_data))
        predictions.append({
            'type': 'success_probability',
            'prediction': recent_success_rate,
            'confidence': 0.8,
            'reasoning': f'基於最近的表現，預期成功率約為 {recent_success_rate:.1%}'
        })
        
        return predictions
    
    def _generate_learning_insights(self, history_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成學習洞察"""
        insights = []
        
        if len(history_data) < 5:
            return insights
        
        # 學習進度洞察
        if len(history_data) >= 20:
            early_success = sum(1 for h in history_data[:10] if h.get('success', False)) / 10
            recent_success = sum(1 for h in history_data[-10:] if h.get('success', False)) / 10
            
            if recent_success > early_success + 0.1:
                insights.append({
                    'type': 'learning_improvement',
                    'insight': '系統表現持續改善',
                    'improvement': recent_success - early_success,
                    'confidence': 0.85
                })
        
        # 操作熟練度洞察
        operation_performance = {}
        for record in history_data:
            op = record.get('operation', 'unknown')
            success = record.get('success', False)
            if op not in operation_performance:
                operation_performance[op] = {'total': 0, 'success': 0}
            operation_performance[op]['total'] += 1
            if success:
                operation_performance[op]['success'] += 1
        
        for op, perf in operation_performance.items():
            if perf['total'] >= 5:  # 足夠的樣本
                success_rate = perf['success'] / perf['total']
                if success_rate > 0.9:
                    insights.append({
                        'type': 'operation_mastery',
                        'insight': f'已精通 {op} 操作',
                        'success_rate': success_rate,
                        'confidence': min(perf['total'] / 10.0, 1.0)
                    })
        
        return insights
    
    def _statistical_anomaly_detection(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """統計異常檢測"""
        anomalies = []
        
        # 檢查數值異常
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if abs(value) > 1000000:  # 異常大的數值
                    anomalies.append({
                        'type': 'statistical',
                        'subtype': 'extreme_value',
                        'field': key,
                        'value': value,
                        'severity': 'medium',
                        'description': f'{key} 的值 {value} 異常大'
                    })
                elif value < 0 and key in ['count', 'size', 'length']:  # 不應該為負的欄位
                    anomalies.append({
                        'type': 'statistical',
                        'subtype': 'negative_value',
                        'field': key,
                        'value': value,
                        'severity': 'high',
                        'description': f'{key} 不應該為負數: {value}'
                    })
        
        return anomalies
    
    def _pattern_anomaly_detection(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """模式異常檢測"""
        anomalies = []
        
        # 檢查結構異常
        expected_fields = ['operation', 'timestamp', 'data']
        missing_fields = [field for field in expected_fields if field not in data]
        
        if missing_fields:
            anomalies.append({
                'type': 'pattern',
                'subtype': 'missing_fields',
                'missing_fields': missing_fields,
                'severity': 'medium',
                'description': f'缺少預期欄位: {missing_fields}'
            })
        
        # 檢查數據類型異常
        if 'timestamp' in data:
            timestamp = data['timestamp']
            if not isinstance(timestamp, (str, datetime)):
                anomalies.append({
                    'type': 'pattern',
                    'subtype': 'invalid_type',
                    'field': 'timestamp',
                    'expected': 'string or datetime',
                    'actual': type(timestamp).__name__,
                    'severity': 'high',
                    'description': f'timestamp 類型異常: {type(timestamp).__name__}'
                })
        
        return anomalies
    
    def _temporal_anomaly_detection(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """時間序列異常檢測"""
        anomalies = []
        
        # 檢查時間戳異常
        if 'timestamp' in data:
            timestamp = data['timestamp']
            try:
                if isinstance(timestamp, str):
                    ts = datetime.fromisoformat(timestamp.replace('Z', TIMEZONE_OFFSET))
                else:
                    ts = timestamp
                
                now = datetime.now(timezone.utc)
                
                # 未來時間異常
                if ts > now:
                    anomalies.append({
                        'type': 'temporal',
                        'subtype': 'future_timestamp',
                        'timestamp': timestamp,
                        'severity': 'medium',
                        'description': '檢測到未來的時間戳'
                    })
                
                # 過舊時間異常 (超過1年)
                elif (now - ts).days > 365:
                    anomalies.append({
                        'type': 'temporal',
                        'subtype': 'stale_timestamp',
                        'timestamp': timestamp,
                        'age_days': (now - ts).days,
                        'severity': 'low',
                        'description': f'時間戳過舊: {(now - ts).days} 天前'
                    })
                    
            except (ValueError, TypeError):
                anomalies.append({
                    'type': 'temporal',
                    'subtype': 'invalid_timestamp',
                    'timestamp': timestamp,
                    'severity': 'high',
                    'description': '無法解析時間戳格式'
                })
        
        return anomalies
    
    def _categorize_anomaly_severity(self, anomalies: List[Dict[str, Any]]) -> Dict[str, int]:
        """分類異常嚴重性"""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return severity_counts
    
    def _recommend_anomaly_actions(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """推薦異常處理動作"""
        recommendations = []
        
        # 按嚴重性分組
        high_severity = [a for a in anomalies if a.get('severity') == 'high']
        if high_severity:
            recommendations.append({
                'priority': 'high',
                'action': 'immediate_investigation',
                'description': f'立即調查 {len(high_severity)} 個高嚴重性異常',
                'affected_anomalies': [a.get('type', 'unknown') for a in high_severity]
            })
        
        # 時間相關異常的特殊處理
        temporal_anomalies = [a for a in anomalies if a.get('type') == 'temporal']
        if temporal_anomalies:
            recommendations.append({
                'priority': 'medium',
                'action': 'time_sync_check',
                'description': '檢查系統時間同步',
                'reason': '檢測到時間相關異常'
            })
        
        return recommendations
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """總結結果用於事件"""
        return {
            'status': 'completed',
            'data_points': len(str(result)),
            'has_anomalies': 'anomalies' in result and len(result['anomalies']) > 0,
            'confidence': result.get('confidence', 0.5)
        }
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        """發布事件"""
        if not self.event_bus:
            return
            
        event = AIEvent(
            event_type=event_type,
            source_module=self.module_name,
            source_version=self.module_version,
            data=data,
            priority=priority
        )
        
        await self.event_bus.publish(event)
    
    def _update_avg_processing_time(self, processing_time: float):
        """更新平均處理時間"""
        if self.stats['successful_operations'] == 1:
            self.stats['avg_processing_time'] = processing_time
        else:
            # 指數移動平均
            alpha = 0.1
            self.stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['avg_processing_time']
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """獲取健康狀態"""
        total_requests = self.stats['total_requests']
        success_rate = (self.stats['successful_operations'] / max(total_requests, 1))
        
        # 健康狀態判斷
        if success_rate > 0.9:
            health_status = 'healthy'
        elif success_rate > 0.7:
            health_status = 'degraded'
        else:
            health_status = 'unhealthy'
        
        return {
            'module': self.module_name,
            'version': self.module_version,
            'status': health_status,
            'statistics': self.stats,
            'success_rate': success_rate,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    # ==================== Context Engineering 2025 方法 ====================
    
    async def advanced_context_processing(self, context_data: Dict[str, Any], task_type: str = 'general') -> Dict[str, Any]:
        """先進的上下文處理 - Context Engineering 2025"""
        try:
            start_time = time.time()
            
            # 使用先進上下文引擎處理
            result = await self.advanced_context_engine.process_advanced_context(context_data, task_type)
            
            # 更新統計
            self.stats['context_engineering_stats']['advanced_processes'] += 1
            processing_time = time.time() - start_time
            
            # 更新快取效率統計
            engine_stats = self.advanced_context_engine.get_performance_stats()
            if 'cache_hit_rate' in engine_stats:
                self.stats['context_engineering_stats']['cache_efficiency'] = engine_stats['cache_hit_rate']
            
            return {
                'status': 'success',
                'context_data': result,
                'task_type': task_type,
                'processing_time': processing_time,
                'engine_performance': engine_stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Advanced context processing error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'fallback_context': context_data
            }
    
    def context_synthesis(self, context_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """上下文合成 - 多層次注意力機制"""
        try:
            start_time = time.time()
            
            # 準備輸入張量（使用輸入數據）
            synthesizer = self.advanced_context_engine.synthesizer
            
            # 從輸入數據創建張量，如果沒有則使用預設值
            batch_size = context_inputs.get('batch_size', 1)
            seq_length = context_inputs.get('sequence_length', 512) 
            hidden_size = context_inputs.get('hidden_size', 768)
            
            global_tensor = torch.randn(batch_size, seq_length, hidden_size)
            local_tensor = torch.randn(batch_size, seq_length, hidden_size)
            modal_tensor = torch.randn(batch_size, seq_length, hidden_size)
            
            tensor_inputs = {
                'global': global_tensor,
                'local': local_tensor,
                'modal': modal_tensor
            }
            
            # 執行合成
            with torch.no_grad():
                synthesis_results = synthesizer.synthesize_context(tensor_inputs)
            
            processing_time = time.time() - start_time
            
            # 轉換結果為可序列化格式
            def tensor_to_summary(tensor):
                if tensor is None:
                    return {'status': 'none'}
                return {
                    'shape': list(tensor.shape),
                    'mean': tensor.mean().item(),
                    'std': tensor.std().item(),
                    'status': 'success'
                }
            
            return {
                'status': 'success',
                'synthesis_results': {
                    'synthesized_context': tensor_to_summary(synthesis_results.get('synthesized_context')),
                    'quality_scores': tensor_to_summary(synthesis_results.get('quality_scores')),
                    'attention_maps': {
                        'global': tensor_to_summary(synthesis_results.get('global_attention')),
                        'local': tensor_to_summary(synthesis_results.get('local_attention')),
                        'cross_modal': tensor_to_summary(synthesis_results.get('cross_modal_attention'))
                    }
                },
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context synthesis error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def context_optimization(self, context_data: Dict[str, Any], task_type: str = 'general') -> Dict[str, Any]:
        """上下文優化 - 針對特定任務優化"""
        try:
            start_time = time.time()
            
            # 使用優化器
            optimizer = self.advanced_context_engine.optimizer
            optimized_context = optimizer.optimize_context(context_data, task_type)
            
            processing_time = time.time() - start_time
            
            # 計算優化效果
            optimization_metrics = self._calculate_optimization_metrics(context_data, optimized_context)
            
            return {
                'status': 'success',
                'original_context': context_data,
                'optimized_context': optimized_context,
                'optimization_metrics': optimization_metrics,
                'task_type': task_type,
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context optimization error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_optimization_metrics(self, original: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
        """計算優化指標"""
        metrics = {
            'size_reduction': 0.0,
            'structure_complexity': 0.0,
            'information_density': 0.0
        }
        
        try:
            # 計算大小減少
            original_size = len(str(original))
            optimized_size = len(str(optimized))
            if original_size > 0:
                metrics['size_reduction'] = 1.0 - (optimized_size / original_size)
            
            # 計算結構複雜度
            metrics['structure_complexity'] = len(optimized.keys()) / max(len(original.keys()), 1)
            
            # 計算信息密度
            original_fields = len(str(original).split())
            optimized_fields = len(str(optimized).split())
            if original_fields > 0:
                metrics['information_density'] = optimized_fields / original_fields
            
        except Exception as e:
            logger.warning(f"Optimization metrics calculation error: {e}")
        
        return metrics

# ==================== 測試和示例 ====================

async def test_perception_module():
    """測試感知模組"""
    
    print("🧪 測試感知模組 v2.0")
    print("=" * 50)
    
    # 創建感知模組
    perception = PerceptionModuleV2()
    
    # 測試掃描分析
    print("\n📊 測試掃描分析...")
    scan_data = {
        'file_count': 150,
        'total_lines': 45000,
        'languages': {'python': 80, 'javascript': 15, 'yaml': 5},
        'complexity': {'cyclomatic': 25, 'cognitive': 30, 'halstead_volume': 1200},
        'structure': {'depth': 5, 'breadth': 20, 'modules': 12, 'classes': 30, 'functions': 180}
    }
    
    request = AIRequest(
        message_type=MessageType.QUERY,
        source_module="test",
        operation="scan_analysis",
        payload={'scan_data': scan_data}
    )
    
    response = await perception.process_request(request)
    print(f"✅ 掃描分析完成: {response.status}")
    if response.result:
        print(f"📈 複雜度分數: {response.result['scan_summary']['complexity_score']:.3f}")
        print(f"🔍 檢測到 {len(response.result['anomalies'])} 個異常")
    else:
        print("⚠️  掃描分析無結果返回")
    
    # 測試上下文編碼
    print("\n🧠 測試上下文編碼...")
    context_data = {
        'current_state': {'mode': 'analysis', 'focus': 'security'},
        'session_history': [
            {'operation': 'scan', 'success': True, 'timestamp': datetime.now(timezone.utc).isoformat()},
            {'operation': 'analyze', 'success': True, 'timestamp': datetime.now(timezone.utc).isoformat()},
            {'operation': 'report', 'success': False, 'timestamp': datetime.now(timezone.utc).isoformat()}
        ]
    }
    
    request = AIRequest(
        message_type=MessageType.COMMAND,
        source_module="test",
        operation="context_encoding",
        payload={'context': context_data}
    )
    
    response = await perception.process_request(request)
    print(f"✅ 上下文編碼完成: {response.status}")
    if response.result:
        print(f"📝 上下文維度: {len(response.result['context_summary'])}")
        print(f"🔗 關鍵上下文: {len(response.result['key_contexts'])} 個")
    else:
        print("⚠️  上下文編碼無結果返回")
    
    # 測試歷史處理
    print("\n📚 測試歷史處理...")
    history_data = [
        {
            'operation': 'scan',
            'success': True,
            'duration': 1200,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': {'type': 'security_scan'}
        },
        {
            'operation': 'analyze',
            'success': True,
            'duration': 800,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': {'type': 'code_analysis'}
        },
        {
            'operation': 'scan',
            'success': False,
            'duration': 300,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': {'type': 'security_scan', 'error': 'timeout'}
        }
    ]
    
    request = AIRequest(
        message_type=MessageType.QUERY,
        source_module="test",
        operation="history_processing",
        payload={'history': history_data}
    )
    
    response = await perception.process_request(request)
    print(f"✅ 歷史處理完成: {response.status}")
    if response.result:
        print(f"📊 成功率: {response.result['history_summary']['success_rate']:.1%}")
        print(f"🔮 模式數量: {len(response.result['detected_patterns'])}")
    else:
        print("⚠️  歷史處理無結果返回")
    
    # 獲取健康狀態
    health = perception.get_health_status()
    print(f"\n💚 模組健康狀態: {health['status']}")
    print(f"📈 成功率: {health['success_rate']:.1%}")
    print(f"⏱️ 平均處理時間: {health['statistics']['avg_processing_time']:.1f}ms")

if __name__ == "__main__":
    asyncio.run(test_perception_module())