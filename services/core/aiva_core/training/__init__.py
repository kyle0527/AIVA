"""
Training Module - 訓練和場景管理模組

提供 OWASP 靶場場景管理、訓練編排等功能
"""

from services.aiva_common.schemas import Scenario, ScenarioResult

from .scenario_manager import ScenarioManager
from .training_orchestrator import TrainingOrchestrator

__all__ = [
    "Scenario",
    "ScenarioManager",
    "ScenarioResult",
    "TrainingOrchestrator",
]
