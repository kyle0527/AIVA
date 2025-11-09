"""
AIVA Strangler Fig Controller v2.0
企業級漸進式遷移控制器

基於 Martin Fowler 的 Strangler Fig Pattern 設計，
支援從 bio_neuron_core.py (v1) 到新 5-模組架構 (v2) 的無縫遷移。

Author: AIVA Team
Created: 2025-11-09
Version: 2.0.0
"""

import asyncio
import time
import random
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import uuid
import traceback


class CircuitBreakerError(Exception):
    """熔斷器異常"""
    pass

# 匯入介面定義 (暫時內部定義，後續分離)
from typing import Protocol
from pydantic import BaseModel, Field

class MessageType(Enum):
    """消息類型枚舉"""
    COMMAND = "command"      # 命令操作 (CQRS-C)
    QUERY = "query"          # 查詢操作 (CQRS-Q)
    EVENT = "event"          # 事件通知
    RESPONSE = "response"    # 響應消息

class Priority(Enum):
    """優先級枚舉"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class AIRequest(BaseModel):
    """統一 AI 請求格式"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_module: str
    target_module: Optional[str] = None
    version: str = "v2.0"
    prefer_version: str = "auto"
    fallback_enabled: bool = True
    operation: str
    payload: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    priority: Priority = Priority.NORMAL
    timeout_seconds: float = 30.0
    retry_count: int = 0

class AIResponse(BaseModel):
    """統一 AI 響應格式"""
    request_id: str
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str
    processed_by: str
    execution_time_ms: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

class IAIModule(Protocol):
    """AI 模組標準介面"""
    @property
    def module_name(self) -> str: ...
    @property
    def module_version(self) -> str: ...

class IVersionRouter(Protocol):
    """版本路由介面"""
    async def route_request(self, request: AIRequest) -> AIResponse: ...
    async def should_use_v2(self, request: AIRequest) -> bool: ...
    async def get_migration_status(self) -> Dict[str, Any]: ...

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 核心枚舉和數據類型 ====================

class MigrationPhase(Enum):
    """遷移階段"""
    PHASE_1_GERMINATION = "phase_1"      # 發芽期：新功能在 v2
    PHASE_2_SPREADING = "phase_2"        # 擴展期：邊緣功能遷移
    PHASE_3_SURROUNDING = "phase_3"      # 包圍期：核心功能遷移
    PHASE_4_REPLACEMENT = "phase_4"      # 替換期：完全替換

class CircuitState(Enum):
    """熔斷器狀態"""
    CLOSED = "closed"           # 正常狀態
    OPEN = "open"               # 熔斷狀態
    HALF_OPEN = "half_open"     # 半開狀態

class HealthStatus(Enum):
    """健康狀態"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"

@dataclass
class MigrationConfig:
    """遷移配置"""
    current_phase: MigrationPhase = MigrationPhase.PHASE_1_GERMINATION
    traffic_ratios: Dict[str, float] = field(default_factory=dict)
    health_threshold: float = 0.8
    fallback_enabled: bool = True
    monitoring_enabled: bool = True
    auto_advance: bool = False
    
    def __post_init__(self):
        """初始化預設流量比例"""
        if not self.traffic_ratios:
            self.traffic_ratios = self._get_default_ratios()
    
    def _get_default_ratios(self) -> Dict[str, float]:
        """獲取預設流量比例"""
        phase_ratios = {
            MigrationPhase.PHASE_1_GERMINATION: {
                'new_features': 1.0,        # 新功能 100% 走 v2
                'existing_features': 0.0    # 既有功能 0% 走 v2
            },
            MigrationPhase.PHASE_2_SPREADING: {
                'enhanced_rag': 0.2,        # RAG 增強 20%
                'code_analysis': 0.5,       # 程式碼分析 50%
                'knowledge_search': 0.3,    # 知識搜索 30%
                'new_features': 1.0         # 新功能持續 100%
            },
            MigrationPhase.PHASE_3_SURROUNDING: {
                'decision_making': 0.7,     # 決策制定 70%
                'learning_engine': 0.8,     # 學習引擎 80%
                'strategy_planning': 0.6,   # 策略規劃 60%
                'enhanced_rag': 0.6,        # RAG 增強提高到 60%
                'code_analysis': 0.9,       # 程式碼分析提高到 90%
                'new_features': 1.0         # 新功能持續 100%
            },
            MigrationPhase.PHASE_4_REPLACEMENT: {
                'all_features': 1.0         # 所有功能 100% v2
            }
        }
        return phase_ratios.get(self.current_phase, {})

@dataclass  
class RoutingDecision:
    """路由決策結果"""
    use_v2: bool
    reason: str
    confidence: float
    fallback_available: bool
    estimated_latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthMetrics:
    """健康指標"""
    error_rate: float = 0.0
    avg_latency: float = 0.0
    throughput: float = 0.0
    success_rate: float = 1.0
    availability: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_health_score(self) -> float:
        """計算健康分數 (0.0 - 1.0)"""
        # 加權計算健康分數
        weights = {
            'success_rate': 0.4,
            'availability': 0.3, 
            'error_rate': -0.2,    # 錯誤率越高分數越低
            'latency': -0.1        # 延遲越高分數越低
        }
        
        # 正規化延遲 (假設 1000ms 為基準)
        normalized_latency = min(self.avg_latency / 1000.0, 1.0)
        
        score = (
            weights['success_rate'] * self.success_rate +
            weights['availability'] * self.availability +
            weights['error_rate'] * (1.0 - self.error_rate) +
            weights['latency'] * (1.0 - normalized_latency)
        )
        
        return max(0.0, min(1.0, score))

# ==================== 熔斷器實現 ====================

class CircuitBreaker:
    """熔斷器實現"""
    
    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs):
        """執行被熔斷器保護的函數"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            # 檢查是否是預期的異常類型
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """判斷是否應該嘗試重置"""
        if self.last_failure_time is None:
            return False
        return (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """成功回調"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """失敗回調"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# ==================== 健康監控器 ====================

class HealthMonitor:
    """健康監控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.v1_metrics = HealthMetrics()
        self.v2_metrics = HealthMetrics()
        
        # 滑動窗口記錄
        self.v1_requests = []
        self.v2_requests = []
        
    def record_request(self, version: str, success: bool, latency: float):
        """記錄請求結果"""
        timestamp = datetime.now(timezone.utc)
        record = {
            'timestamp': timestamp,
            'success': success,
            'latency': latency
        }
        
        if version == 'v1':
            self.v1_requests.append(record)
            if len(self.v1_requests) > self.window_size:
                self.v1_requests.pop(0)
            self._update_metrics(self.v1_requests, self.v1_metrics)
            
        elif version == 'v2':
            self.v2_requests.append(record)
            if len(self.v2_requests) > self.window_size:
                self.v2_requests.pop(0)
            self._update_metrics(self.v2_requests, self.v2_metrics)
    
    def _update_metrics(self, requests: List[Dict], metrics: HealthMetrics):
        """更新指標"""
        if not requests:
            return
        
        # 計算成功率
        successes = [r for r in requests if r['success']]
        metrics.success_rate = len(successes) / len(requests)
        metrics.error_rate = 1.0 - metrics.success_rate
        
        # 計算平均延遲
        metrics.avg_latency = sum(r['latency'] for r in requests) / len(requests)
        
        # 計算吞吐量 (請求/秒)
        if len(requests) > 1:
            time_span = (requests[-1]['timestamp'] - requests[0]['timestamp']).total_seconds()
            metrics.throughput = len(requests) / max(time_span, 1.0)
        
        metrics.availability = 1.0  # 假設服務可用
        metrics.last_updated = datetime.now(timezone.utc)
    
    def get_health_status(self, version: str) -> HealthStatus:
        """獲取健康狀態"""
        metrics = self.v1_metrics if version == 'v1' else self.v2_metrics
        health_score = metrics.calculate_health_score()
        
        if health_score >= 0.9:
            return HealthStatus.HEALTHY
        elif health_score >= 0.7:
            return HealthStatus.DEGRADED
        elif health_score >= 0.3:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.UNAVAILABLE
    
    def get_comparative_health(self) -> Dict[str, Any]:
        """獲取比較健康狀況"""
        v1_score = self.v1_metrics.calculate_health_score()
        v2_score = self.v2_metrics.calculate_health_score()
        
        return {
            'v1': {
                'score': v1_score,
                'status': self.get_health_status('v1'),
                'metrics': {
                    'success_rate': self.v1_metrics.success_rate,
                    'avg_latency': self.v1_metrics.avg_latency,
                    'error_rate': self.v1_metrics.error_rate,
                    'throughput': self.v1_metrics.throughput
                }
            },
            'v2': {
                'score': v2_score,
                'status': self.get_health_status('v2'),
                'metrics': {
                    'success_rate': self.v2_metrics.success_rate,
                    'avg_latency': self.v2_metrics.avg_latency,
                    'error_rate': self.v2_metrics.error_rate,
                    'throughput': self.v2_metrics.throughput
                }
            },
            'recommendation': 'v2' if v2_score > v1_score else 'v1',
            'confidence': abs(v2_score - v1_score)
        }

# ==================== 流量分析器 ====================

class TrafficAnalyzer:
    """流量分析器"""
    
    def __init__(self):
        self.feature_usage = {}
        self.user_patterns = {}
        
    def analyze_request(self, request: AIRequest) -> Dict[str, Any]:
        """分析請求特性"""
        feature = request.operation
        timestamp = request.timestamp
        
        # 記錄功能使用
        if feature not in self.feature_usage:
            self.feature_usage[feature] = []
        self.feature_usage[feature].append(timestamp)
        
        # 計算使用頻率
        recent_usage = self._get_recent_usage(feature, timedelta(hours=1))
        
        return {
            'feature': feature,
            'frequency': len(recent_usage),
            'is_peak_time': self._is_peak_time(timestamp),
            'user_pattern': self._analyze_user_pattern(request),
            'complexity_score': self._calculate_complexity_score(request)
        }
    
    def _get_recent_usage(self, feature: str, time_window: timedelta) -> List[datetime]:
        """獲取最近使用記錄"""
        if feature not in self.feature_usage:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - time_window
        return [t for t in self.feature_usage[feature] if t >= cutoff_time]
    
    def _is_peak_time(self, timestamp: datetime) -> bool:
        """判斷是否為峰值時間"""
        hour = timestamp.hour
        # 假設 9-12 和 14-18 為峰值時間
        return (9 <= hour <= 12) or (14 <= hour <= 18)
    
    def _analyze_user_pattern(self, request: AIRequest) -> str:
        """分析使用者模式"""
        payload_size = len(str(request.payload))
        
        if payload_size < 1000:
            return "simple_query"
        elif payload_size < 10000:
            return "complex_analysis"
        else:
            return "batch_processing"
    
    def _calculate_complexity_score(self, request: AIRequest) -> float:
        """計算複雜度分數"""
        factors = {
            'payload_size': len(str(request.payload)),
            'has_context': request.context is not None,
            'priority': request.priority.value,
            'timeout': request.timeout_seconds
        }
        
        # 簡單的複雜度計算
        score = (
            factors['payload_size'] / 10000 +  # 負載大小權重
            (0.2 if factors['has_context'] else 0.0) +  # 上下文權重
            factors['priority'] / 4.0 +  # 優先級權重
            (1.0 - min(factors['timeout'] / 60.0, 1.0))  # 超時權重
        )
        
        return min(score, 1.0)

# ==================== Strangler Fig Controller 主體 ====================

class StranglerFigController(IVersionRouter):
    """Strangler Fig Controller 主實現"""
    
    def __init__(self, config: Optional[MigrationConfig] = None):
        self.config = config or MigrationConfig()
        self.health_monitor = HealthMonitor()
        self.traffic_analyzer = TrafficAnalyzer()
        
        # 熔斷器
        self.v1_circuit_breaker = CircuitBreaker(failure_threshold=10)
        self.v2_circuit_breaker = CircuitBreaker(failure_threshold=5)
        
        # 功能分類
        self.new_features = {
            'static_analysis', 'function_exploration', 'cli_generation',
            'self_cognition', 'capability_assessment', 'architecture_analysis',
            'code_analysis', 'advanced_rag'
        }
        
        self.legacy_features = {
            'basic_decision', 'simple_execution', 'traditional_scan'
        }
        
        # 統計資訊
        self.routing_stats = {
            'total_requests': 0,
            'v1_requests': 0,
            'v2_requests': 0,
            'fallback_used': 0,
            'errors': 0
        }
        
        logger.info(f"Strangler Fig Controller 初始化完成，當前階段: {self.config.current_phase.value}")
    
    async def route_request(self, request: AIRequest) -> AIResponse:
        """智能路由請求 - 核心方法"""
        start_time = time.time()
        self.routing_stats['total_requests'] += 1
        
        try:
            # 1. 分析請求特性
            traffic_analysis = self.traffic_analyzer.analyze_request(request)
            
            # 2. 做出路由決策
            decision = await self._make_routing_decision(request, traffic_analysis)
            
            # 3. 執行請求
            if decision.use_v2:
                response = await self._execute_v2_with_monitoring(request, decision)
            else:
                response = await self._execute_v1_with_monitoring(request, decision)
            
            # 4. 記錄成功指標
            execution_time = (time.time() - start_time) * 1000
            version = 'v2' if decision.use_v2 else 'v1'
            self.health_monitor.record_request(version, True, execution_time)
            
            # 5. 增強響應元數據
            response.metadata.update({
                'routing_decision': {
                    'use_v2': decision.use_v2,
                    'reason': decision.reason,
                    'confidence': decision.confidence
                },
                'traffic_analysis': traffic_analysis,
                'migration_phase': self.config.current_phase.value
            })
            
            return response
            
        except Exception as e:
            # 錯誤處理和降級
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"路由錯誤: {str(e)}")
            
            # 嘗試降級
            if hasattr(self, '_last_decision') and self._last_decision.fallback_available:
                try:
                    fallback_response = await self._execute_fallback(request)
                    self.routing_stats['fallback_used'] += 1
                    return fallback_response
                except Exception as fallback_error:
                    logger.error(f"降級也失敗: {str(fallback_error)}")
            
            self.routing_stats['errors'] += 1
            
            # 返回錯誤響應
            return AIResponse(
                request_id=request.request_id,
                status="error",
                processed_by="strangler_fig_router",
                execution_time_ms=execution_time,
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    async def _make_routing_decision(
        self, 
        request: AIRequest, 
        traffic_analysis: Dict[str, Any]
    ) -> RoutingDecision:
        """做出路由決策"""
        operation = request.operation
        
        # 1. 新功能強制走 v2
        if operation in self.new_features:
            decision = RoutingDecision(
                use_v2=True,
                reason="new_feature_v2_only",
                confidence=1.0,
                fallback_available=False,
                estimated_latency=100.0  # 預設值
            )
            self._last_decision = decision
            return decision
        
        # 2. 明確的遺留功能走 v1
        if operation in self.legacy_features:
            decision = RoutingDecision(
                use_v2=False,
                reason="legacy_feature_v1_preferred",
                confidence=0.9,
                fallback_available=False,
                estimated_latency=50.0  # 預設值
            )
            self._last_decision = decision
            return decision
        
        # 3. 根據遷移階段和流量比例決策
        traffic_ratio = self.config.traffic_ratios.get(operation, 0.0)
        
        # 4. 考慮健康狀況
        health_comparison = self.health_monitor.get_comparative_health()
        
        # 5. 考慮負載因素
        load_factor = await self._calculate_load_factor(traffic_analysis)
        
        # 6. 綜合決策算法
        v2_score = self._calculate_v2_score(
            traffic_ratio,
            health_comparison,
            load_factor,
            traffic_analysis
        )
        
        # 7. 最終決策
        use_v2 = v2_score > 0.5
        
        decision = RoutingDecision(
            use_v2=use_v2,
            reason=f"algorithm_score_{v2_score:.2f}",
            confidence=abs(v2_score - 0.5) * 2,  # 距離0.5越遠信心度越高
            fallback_available=self.config.fallback_enabled,
            estimated_latency=health_comparison['v2']['metrics']['avg_latency'] if use_v2 
                            else health_comparison['v1']['metrics']['avg_latency']
        )
        
        self._last_decision = decision
        return decision
    
    def _calculate_v2_score(
        self,
        traffic_ratio: float,
        health_comparison: Dict[str, Any],
        load_factor: float,
        traffic_analysis: Dict[str, Any]
    ) -> float:
        """計算 v2 分數 (0.0 - 1.0)"""
        # 基礎遷移比例
        base_score = traffic_ratio
        
        # 健康狀況調整
        v2_health = health_comparison['v2']['score']
        v1_health = health_comparison['v1']['score']
        health_adjustment = (v2_health - v1_health) * 0.3
        
        # 負載調整 (低負載時更願意嘗試 v2)
        load_adjustment = (1.0 - load_factor) * 0.1
        
        # 複雜度調整 (高複雜度更適合 v2)
        complexity_adjustment = traffic_analysis['complexity_score'] * 0.1
        
        # 峰值時間調整 (峰值時間更保守)
        peak_adjustment = -0.1 if traffic_analysis['is_peak_time'] else 0.0
        
        final_score = base_score + health_adjustment + load_adjustment + complexity_adjustment + peak_adjustment
        
        return max(0.0, min(1.0, final_score))
    
    async def _calculate_load_factor(self, traffic_analysis: Dict[str, Any]) -> float:
        """計算負載因子"""
        # 基於請求頻率和系統負載計算
        frequency_factor = min(traffic_analysis['frequency'] / 100.0, 1.0)
        
        # 可以加入更多負載指標，如 CPU、記憶體等
        return frequency_factor
    
    async def _execute_v2_with_monitoring(
        self, 
        request: AIRequest, 
        decision: RoutingDecision
    ) -> AIResponse:
        """執行 v2 並監控"""
        self.routing_stats['v2_requests'] += 1
        
        # 使用熔斷器保護
        try:
            response = await self.v2_circuit_breaker.call(
                self._execute_v2_internal, 
                request
            )
            return response
        except Exception as e:
            # v2 失敗，嘗試降級到 v1
            if decision.fallback_available:
                logger.warning(f"v2 失敗，降級到 v1: {str(e)}")
                return await self._execute_v1_with_monitoring(request, decision)
            else:
                raise e
    
    async def _execute_v1_with_monitoring(
        self, 
        request: AIRequest, 
        decision: RoutingDecision
    ) -> AIResponse:
        """執行 v1 並監控"""
        self.routing_stats['v1_requests'] += 1
        
        # 使用熔斷器保護
        response = await self.v1_circuit_breaker.call(
            self._execute_v1_internal, 
            request
        )
        return response
    
    async def _execute_v2_internal(self, request: AIRequest) -> AIResponse:
        """執行 v2 內部邏輯"""
        # 這裡應該調用實際的 v2 模組
        # 暫時模擬實現
        await asyncio.sleep(0.1)  # 模擬 v2 處理時間
        
        return AIResponse(
            request_id=request.request_id,
            status="success",
            processed_by="ai_v2_modules",
            execution_time_ms=100.0,
            result={
                "message": "Processed by AI v2.0 modules",
                "operation": request.operation,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata={
                "version": "v2.0",
                "modules_used": ["perception", "knowledge", "cognition", "decision", "execution"]
            }
        )
    
    async def _execute_v1_internal(self, request: AIRequest) -> AIResponse:
        """執行 v1 內部邏輯"""
        # 這裡應該調用實際的 bio_neuron_core.py
        # 暫時模擬實現
        await asyncio.sleep(0.05)  # 模擬 v1 處理時間 (更快)
        
        return AIResponse(
            request_id=request.request_id,
            status="success", 
            processed_by="bio_neuron_core_v1",
            execution_time_ms=50.0,
            result={
                "message": "Processed by bio_neuron_core v1",
                "operation": request.operation,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata={
                "version": "v1.0",
                "legacy_system": True
            }
        )
    
    async def _execute_fallback(self, request: AIRequest) -> AIResponse:
        """執行降級策略"""
        return AIResponse(
            request_id=request.request_id,
            status="partial",
            processed_by="fallback_handler",
            execution_time_ms=10.0,
            result={
                "message": "Fallback response - limited functionality",
                "operation": request.operation,
                "fallback_reason": "primary_systems_unavailable"
            },
            metadata={
                "is_fallback": True,
                "limited_functionality": True
            }
        )
    
    async def should_use_v2(self, request: AIRequest) -> bool:
        """簡化的 v2 判斷接口"""
        traffic_analysis = self.traffic_analyzer.analyze_request(request)
        decision = await self._make_routing_decision(request, traffic_analysis)
        return decision.use_v2
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """獲取遷移狀態"""
        health_comparison = self.health_monitor.get_comparative_health()
        
        return {
            'current_phase': self.config.current_phase.value,
            'traffic_ratios': self.config.traffic_ratios,
            'routing_stats': self.routing_stats,
            'health_comparison': health_comparison,
            'circuit_breakers': {
                'v1_state': self.v1_circuit_breaker.state.value,
                'v2_state': self.v2_circuit_breaker.state.value
            },
            'recommendations': await self._generate_recommendations()
        }
    
    async def _generate_recommendations(self) -> List[str]:
        """生成建議"""
        recommendations = []
        
        # 分析健康狀況
        health_comparison = self.health_monitor.get_comparative_health()
        v2_health = health_comparison['v2']['score']
        v1_health = health_comparison['v1']['score']
        
        if v2_health > v1_health + 0.1:
            recommendations.append("v2 健康狀況良好，建議增加流量比例")
        elif v1_health > v2_health + 0.1:
            recommendations.append("v1 表現更穩定，建議暫緩遷移")
        
        # 分析錯誤率
        error_rate = self.routing_stats['errors'] / max(self.routing_stats['total_requests'], 1)
        if error_rate > 0.05:
            recommendations.append("錯誤率偏高，建議檢查系統穩定性")
        
        # 分析降級使用率
        fallback_rate = self.routing_stats['fallback_used'] / max(self.routing_stats['total_requests'], 1)
        if fallback_rate > 0.1:
            recommendations.append("降級使用率過高，建議優化主要系統")
        
        return recommendations

# ==================== 階段管理器 ====================

class MigrationPhaseManager:
    """遷移階段管理器"""
    
    def __init__(self, controller: StranglerFigController):
        self.controller = controller
        self.phase_history = []
    
    async def advance_to_next_phase(self) -> bool:
        """推進到下一階段"""
        current = self.controller.config.current_phase
        
        phase_sequence = [
            MigrationPhase.PHASE_1_GERMINATION,
            MigrationPhase.PHASE_2_SPREADING, 
            MigrationPhase.PHASE_3_SURROUNDING,
            MigrationPhase.PHASE_4_REPLACEMENT
        ]
        
        try:
            current_index = phase_sequence.index(current)
            if current_index >= len(phase_sequence) - 1:
                logger.info("已經是最終階段")
                return False
            
            next_phase = phase_sequence[current_index + 1]
            
            # 檢查是否滿足推進條件
            can_advance = await self._check_advancement_criteria()
            
            if can_advance:
                await self._execute_phase_transition(next_phase)
                return True
            else:
                logger.warning("不滿足階段推進條件")
                return False
                
        except ValueError:
            logger.error(f"未知的遷移階段: {current}")
            return False
    
    async def _check_advancement_criteria(self) -> bool:
        """檢查階段推進標準"""
        status = await self.controller.get_migration_status()
        
        criteria = {
            'health_score_v2': status['health_comparison']['v2']['score'] > 0.8,
            'error_rate': status['routing_stats']['errors'] / 
                         max(status['routing_stats']['total_requests'], 1) < 0.02,
            'v2_success_rate': status['health_comparison']['v2']['metrics']['success_rate'] > 0.95
        }
        
        return all(criteria.values())
    
    async def _execute_phase_transition(self, new_phase: MigrationPhase):
        """執行階段轉換"""
        old_phase = self.controller.config.current_phase
        
        logger.info(f"開始從 {old_phase.value} 遷移到 {new_phase.value}")
        
        # 記錄歷史
        self.phase_history.append({
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'timestamp': datetime.now(timezone.utc),
            'migration_stats': await self.controller.get_migration_status()
        })
        
        # 更新配置
        self.controller.config.current_phase = new_phase
        self.controller.config.traffic_ratios = self.controller.config._get_default_ratios()
        
        logger.info(f"成功遷移到 {new_phase.value}")

# ==================== 測試和示例 ====================

async def test_strangler_fig_controller():
    """測試 Strangler Fig Controller"""
    
    # 創建控制器
    controller = StranglerFigController()
    
    # 測試不同類型的請求
    test_requests = [
        AIRequest(
            message_type=MessageType.QUERY,
            source_module="test_client",
            operation="static_analysis",  # 新功能
            payload={"code": "def hello(): pass"}
        ),
        AIRequest(
            message_type=MessageType.COMMAND,
            source_module="test_client", 
            operation="basic_decision",   # 遺留功能
            payload={"context": "simple decision"}
        ),
        AIRequest(
            message_type=MessageType.QUERY,
            source_module="test_client",
            operation="enhanced_rag",     # 遷移中功能
            payload={"query": "find similar documents"}
        )
    ]
    
    print("🧪 測試 Strangler Fig Controller")
    print("=" * 50)
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n📤 測試請求 {i}: {request.operation}")
        
        response = await controller.route_request(request)
        
        print(f"✅ 響應狀態: {response.status}")
        print(f"🔧 處理者: {response.processed_by}")
        print(f"⏱️ 執行時間: {response.execution_time_ms:.1f}ms")
        
        if 'routing_decision' in response.metadata:
            decision = response.metadata['routing_decision']
            print(f"🎯 路由決策: {'v2' if decision['use_v2'] else 'v1'}")
            print(f"💭 原因: {decision['reason']}")
            print(f"🎲 信心度: {decision['confidence']:.2f}")
    
    # 顯示遷移狀態
    print("\n📊 遷移狀態報告")
    print("=" * 50)
    status = await controller.get_migration_status()
    
    print(f"當前階段: {status['current_phase']}")
    print(f"總請求數: {status['routing_stats']['total_requests']}")
    print(f"v1 請求: {status['routing_stats']['v1_requests']}")
    print(f"v2 請求: {status['routing_stats']['v2_requests']}")
    print(f"降級次數: {status['routing_stats']['fallback_used']}")
    
    print("\n💡 建議:")
    for rec in status['recommendations']:
        print(f"  • {rec}")

if __name__ == "__main__":
    asyncio.run(test_strangler_fig_controller())