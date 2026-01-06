"""
Learning Module - 機器學習和經驗管理模組

提供模型訓練等功能
注意: experience_manager 現在使用 aiva_common.ai.experience_manager 統一實現
"""

from .model_trainer import ModelTrainer
from .scalable_bio_trainer import ScalableBioTrainer, ScalableBioTrainingConfig

__all__ = ["ModelTrainer", "ScalableBioTrainer", "ScalableBioTrainingConfig"]
