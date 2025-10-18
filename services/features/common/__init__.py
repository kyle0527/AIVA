"""
Common services - 統一的檢測基礎設施
為所有功能模組提供統一的配置和智能檢測能力
"""

from .detection_config import (
    BaseDetectionConfig,
    DetectionStrategy,
    IDORConfig,
    SSRFConfig,
    XSSConfig,
)
from .unified_smart_detection_manager import (
    DetectionMetrics,
    UnifiedSmartDetectionManager,
)

__all__ = [
    "BaseDetectionConfig",
    "SSRFConfig",
    "XSSConfig",
    "IDORConfig",
    "DetectionStrategy",
    "UnifiedSmartDetectionManager",
    "DetectionMetrics",
]
