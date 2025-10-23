"""
AIVA Reception Module

數據接收和處理層。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .data_reception_layer import DataReceptionLayer
    from .test_result_database import TestResultDatabase
    from .experience_repository import ExperienceRepository
    from .experience_models import (
        Base,
        ExperienceRecord,
        TrainingDataset,
        DatasetSample,
        ModelTrainingHistory
    )
    
    __all__ = [
        "DataReceptionLayer",
        "TestResultDatabase",
        "ExperienceRepository",
        "Base",
        "ExperienceRecord", 
        "TrainingDataset",
        "DatasetSample",
        "ModelTrainingHistory"
    ]
except ImportError:
    __all__ = []
