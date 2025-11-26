"""
功能模組基礎組件包

提供功能模組的基礎類和工具
"""

from .feature_registry import FeatureRegistry, get_global_registry
from .result_schema import FeatureResult, Finding
from .integration_helper import (
    IntegrationHelper,
    get_integration_helper,
    save_finding,
    record_experience,
)

__all__ = [
    "FeatureRegistry",
    "get_global_registry",
    "FeatureResult",
    "Finding",
    "IntegrationHelper",
    "get_integration_helper",
    "save_finding",
    "record_experience",
]
