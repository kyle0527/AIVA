"""
Steganography Module

提供隱寫術與反隱寫術能力，支援圖片、音訊、視訊等多種媒體。

風險等級: L1 (資訊隱藏)
模組版本: 1.0.0
"""

from .manager import SteganographyManager
from .models import (
    # Enums
    SteganographyMethod,
    CarrierType,
    ImageFormat,
    EncryptionAlgorithm,
    
    # Data Models
    EmbedConfig,
    EmbedResult,
    ExtractConfig,
    ExtractResult,
    DetectionResult
)

__all__ = [
    "SteganographyManager",
    "SteganographyMethod",
    "CarrierType",
    "ImageFormat",
    "EncryptionAlgorithm",
    "EmbedConfig",
    "EmbedResult",
    "ExtractConfig",
    "ExtractResult",
    "DetectionResult",
]

__version__ = "1.0.0"
__risk_level__ = "L1"
