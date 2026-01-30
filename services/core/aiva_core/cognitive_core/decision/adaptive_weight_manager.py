"""Adaptive Weight Manager - 自適應權重管理器

根據任務上下文和歷史效能動態調整 AI 決策權重

設計原則：
1. 情境感知：根據風險、經驗、時間等因素調整權重
2. 學習適應：基於歷史效能動態優化
3. 安全優先：高風險情境增加規則權重

實現日期: 2026-01-11
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
from datetime import datetime

# 使用 aiva_common 的統一枚舉
from aiva_common.enums import RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """權重配置"""
    neural: float = 0.5       # 神經網路 (AI 直覺)
    experience: float = 0.3   # 經驗學習 (歷史記憶)
    rule: float = 0.2         # 規則引擎 (安全邊界)
    
    def to_dict(self) -> dict[str, float]:
        return {
            "neural": self.neural,
            "experience": self.experience,
            "rule": self.rule
        }
    
    def normalize(self) -> "WeightConfig":
        """歸一化權重確保總和為 1"""
        total = self.neural + self.experience + self.rule
        if total > 0:
            return WeightConfig(
                neural=self.neural / total,
                experience=self.experience / total,
                rule=self.rule / total
            )
        return self


@dataclass
class ContextFactors:
    """情境調整因子"""
    high_risk: dict[str, float] = field(default_factory=lambda: {
        "neural": -0.15,   # 高風險時減少 AI 直覺
        "rule": +0.20      # 增加規則權重
    })
    
    low_experience: dict[str, float] = field(default_factory=lambda: {
        "neural": +0.10,   # 經驗不足時增加 AI
        "experience": -0.10
    })
    
    time_critical: dict[str, float] = field(default_factory=lambda: {
        "neural": +0.15,   # 時間緊迫時依賴 AI 快速決策
        "experience": -0.10,
        "rule": -0.05
    })
    
    known_target: dict[str, float] = field(default_factory=lambda: {
        "experience": +0.20,  # 已知目標時增加經驗權重
        "neural": -0.10
    })
    
    expert_mode: dict[str, float] = field(default_factory=lambda: {
        "neural": +0.20,      # 專家模式增加 AI 權重
        "rule": -0.10
    })
    
    novice_mode: dict[str, float] = field(default_factory=lambda: {
        "rule": +0.15,        # 新手模式增加規則權重
        "neural": -0.10
    })
    
    high_confidence: dict[str, float] = field(default_factory=lambda: {
        "neural": +0.10       # AI 高信心度時增加其權重
    })
    
    repeated_failure: dict[str, float] = field(default_factory=lambda: {
        "neural": -0.10,      # 重複失敗時減少當前策略
        "experience": +0.05,
        "rule": +0.05
    })


@dataclass
class PerformanceRecord:
    """效能記錄"""
    timestamp: datetime
    weights_used: dict[str, float]
    decision_source: str  # "neural", "experience", "rule"
    success: bool
    reward: float
    context_type: str  # 情境類型


class AdaptiveWeightManager:
    """自適應權重管理器
    
    根據任務上下文和歷史效能動態調整決策權重
    
    使用示例:
        manager = AdaptiveWeightManager()
        
        # 獲取動態權重
        weights = manager.get_weights(context)
        
        # 決策後更新
        manager.update_performance(weights_used, decision_source, success, reward)
    """
    
    def __init__(
        self,
        base_weights: Optional[WeightConfig] = None,
        context_factors: Optional[ContextFactors] = None,
        history_size: int = 100,
        learning_rate: float = 0.1
    ):
        """初始化自適應權重管理器
        
        Args:
            base_weights: 基礎權重配置
            context_factors: 情境調整因子
            history_size: 歷史記錄大小
            learning_rate: 學習率（用於基於效能的調整）
        """
        self.base_weights = base_weights or WeightConfig()
        self.context_factors = context_factors or ContextFactors()
        self.learning_rate = learning_rate
        
        # 效能歷史
        self._performance_history: deque[PerformanceRecord] = deque(maxlen=history_size)
        
        # 動態調整偏移量（基於學習）
        self._learned_offset: dict[str, float] = {
            "neural": 0.0,
            "experience": 0.0,
            "rule": 0.0
        }
        
        # 各來源的成功率統計
        self._source_stats: dict[str, dict[str, float]] = {
            "neural": {"success": 0, "total": 0},
            "experience": {"success": 0, "total": 0},
            "rule": {"success": 0, "total": 0}
        }
        
        logger.info(
            f"AdaptiveWeightManager initialized: "
            f"neural={self.base_weights.neural:.2f}, "
            f"experience={self.base_weights.experience:.2f}, "
            f"rule={self.base_weights.rule:.2f}"
        )
    
    def get_weights(self, context: Any) -> dict[str, float]:
        """根據上下文計算動態權重
        
        Args:
            context: 決策上下文 (DecisionContext)
            
        Returns:
            動態調整後的權重字典
        """
        weights = WeightConfig(
            neural=self.base_weights.neural + self._learned_offset["neural"],
            experience=self.base_weights.experience + self._learned_offset["experience"],
            rule=self.base_weights.rule + self._learned_offset["rule"]
        )
        
        # 應用情境因子
        applied_factors = []
        
        # 1. 風險等級調整
        if hasattr(context, 'risk_level'):
            if context.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self._apply_factor(weights, self.context_factors.high_risk)
                applied_factors.append("high_risk")
        
        # 2. 經驗不足調整
        if hasattr(context, 'attempts_without_success'):
            if context.attempts_without_success >= 3:
                self._apply_factor(weights, self.context_factors.repeated_failure)
                applied_factors.append("repeated_failure")
            elif context.attempts_without_success >= 5:
                self._apply_factor(weights, self.context_factors.low_experience)
                applied_factors.append("low_experience")
        
        # 3. 時間緊迫調整
        if hasattr(context, 'time_constraints') and context.time_constraints:
            if context.time_constraints < 300:  # 5 分鐘內
                self._apply_factor(weights, self.context_factors.time_critical)
                applied_factors.append("time_critical")
        
        # 4. 已知目標調整
        if hasattr(context, 'target_info') and context.target_info:
            if context.target_info.get("known", False):
                self._apply_factor(weights, self.context_factors.known_target)
                applied_factors.append("known_target")
        
        # 5. 模式調整
        if hasattr(context, 'mode_restrictions'):
            if "expert" in context.mode_restrictions:
                self._apply_factor(weights, self.context_factors.expert_mode)
                applied_factors.append("expert_mode")
            elif "novice" in context.mode_restrictions:
                self._apply_factor(weights, self.context_factors.novice_mode)
                applied_factors.append("novice_mode")
        
        # 歸一化
        normalized = weights.normalize()
        
        if applied_factors:
            logger.debug(
                f"Applied factors: {applied_factors} → "
                f"neural={normalized.neural:.2f}, "
                f"experience={normalized.experience:.2f}, "
                f"rule={normalized.rule:.2f}"
            )
        
        return normalized.to_dict()
    
    def _apply_factor(self, weights: WeightConfig, factor: dict[str, float]) -> None:
        """應用調整因子"""
        for key, delta in factor.items():
            if hasattr(weights, key):
                current = getattr(weights, key)
                # 限制在 [0.05, 0.85] 範圍內
                new_value = max(0.05, min(0.85, current + delta))
                setattr(weights, key, new_value)
    
    def update_performance(
        self,
        weights_used: dict[str, float],
        decision_source: str,
        success: bool,
        reward: float = 0.0,
        context_type: str = "general"
    ) -> None:
        """更新效能記錄並調整學習偏移量
        
        Args:
            weights_used: 使用的權重
            decision_source: 決策來源 ("neural", "experience", "rule")
            success: 是否成功
            reward: 獎勵值 (0.0-1.0)
            context_type: 情境類型
        """
        # 記錄效能
        record = PerformanceRecord(
            timestamp=datetime.now(),
            weights_used=weights_used,
            decision_source=decision_source,
            success=success,
            reward=reward,
            context_type=context_type
        )
        self._performance_history.append(record)
        
        # 更新來源統計
        if decision_source in self._source_stats:
            self._source_stats[decision_source]["total"] += 1
            if success:
                self._source_stats[decision_source]["success"] += 1
        
        # 基於效能調整學習偏移量
        self._update_learned_offset(decision_source, success, reward)
        
        logger.debug(
            f"Performance updated: source={decision_source}, "
            f"success={success}, reward={reward:.2f}"
        )
    
    def _update_learned_offset(
        self,
        decision_source: str,
        success: bool,
        reward: float
    ) -> None:
        """基於效能更新學習偏移量"""
        # 計算調整量
        if success:
            # 成功時增加該來源的權重
            adjustment = self.learning_rate * (reward - 0.5)
        else:
            # 失敗時減少該來源的權重
            adjustment = -self.learning_rate * 0.1
        
        # 應用調整
        if decision_source in self._learned_offset:
            current = self._learned_offset[decision_source]
            # 限制偏移量在 [-0.15, +0.15] 範圍
            new_offset = max(-0.15, min(0.15, current + adjustment))
            self._learned_offset[decision_source] = new_offset
        
        # 歸一化偏移量（確保總和接近 0）
        total_offset = sum(self._learned_offset.values())
        if abs(total_offset) > 0.01:
            correction = total_offset / 3
            for key in self._learned_offset:
                self._learned_offset[key] -= correction
    
    def get_source_success_rates(self) -> dict[str, float]:
        """獲取各來源的成功率"""
        rates = {}
        for source, stats in self._source_stats.items():
            total = stats["total"]
            if total > 0:
                rates[source] = stats["success"] / total
            else:
                rates[source] = 0.0
        return rates
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息"""
        success_rates = self.get_source_success_rates()
        
        return {
            "base_weights": self.base_weights.to_dict(),
            "learned_offset": self._learned_offset.copy(),
            "effective_weights": {
                key: self.base_weights.to_dict()[key] + self._learned_offset[key]
                for key in self._learned_offset
            },
            "source_success_rates": success_rates,
            "total_decisions": len(self._performance_history),
            "recent_success_rate": self._calculate_recent_success_rate()
        }
    
    def _calculate_recent_success_rate(self, window: int = 20) -> float:
        """計算最近的成功率"""
        if not self._performance_history:
            return 0.0
        
        recent = list(self._performance_history)[-window:]
        successes = sum(1 for r in recent if r.success)
        return successes / len(recent) if recent else 0.0
    
    def reset_learning(self) -> None:
        """重置學習狀態"""
        self._learned_offset = {
            "neural": 0.0,
            "experience": 0.0,
            "rule": 0.0
        }
        self._performance_history.clear()
        self._source_stats = {
            "neural": {"success": 0, "total": 0},
            "experience": {"success": 0, "total": 0},
            "rule": {"success": 0, "total": 0}
        }
        logger.info("Learning state reset")
    
    def export_state(self) -> dict[str, Any]:
        """導出狀態（用於持久化）"""
        return {
            "base_weights": self.base_weights.to_dict(),
            "learned_offset": self._learned_offset.copy(),
            "source_stats": self._source_stats.copy(),
            "learning_rate": self.learning_rate
        }
    
    def import_state(self, state: dict[str, Any]) -> None:
        """導入狀態（用於恢復）"""
        if "base_weights" in state:
            self.base_weights = WeightConfig(**state["base_weights"])
        if "learned_offset" in state:
            self._learned_offset = state["learned_offset"]
        if "source_stats" in state:
            self._source_stats = state["source_stats"]
        if "learning_rate" in state:
            self.learning_rate = state["learning_rate"]
        
        logger.info("State imported successfully")


# ==================== 便捷函數 ====================

_default_manager: Optional[AdaptiveWeightManager] = None


def get_adaptive_weight_manager() -> AdaptiveWeightManager:
    """獲取全局自適應權重管理器"""
    global _default_manager
    if _default_manager is None:
        _default_manager = AdaptiveWeightManager()
    return _default_manager


def get_dynamic_weights(context: Any) -> dict[str, float]:
    """便捷函數：獲取動態權重"""
    return get_adaptive_weight_manager().get_weights(context)


# ==================== 導出 ====================

__all__ = [
    "AdaptiveWeightManager",
    "WeightConfig",
    "ContextFactors",
    "PerformanceRecord",
    "get_adaptive_weight_manager",
    "get_dynamic_weights"
]

