"""
Training Module - 訓練和場景管理模組

提供 OWASP 靶場場景管理、訓練編排等功能

注意: TrainingOrchestrator 已移除，訓練功能整合至 ContinuousLearningEngine
"""

from aiva_common.schemas import Scenario, ScenarioResult

from .scenario_manager import ScenarioManager

__all__ = [
    "Scenario",
    "ScenarioManager",
    "ScenarioResult",
]

