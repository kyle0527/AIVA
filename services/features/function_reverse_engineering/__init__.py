"""
Reverse Engineering Module

提供逆向工程能力，支援 Android APK、二進位文件分析等功能。

風險等級: L1 (分析工具)
模組版本: 1.0.0
"""

from .manager import ReverseEngineeringManager
from .models import (
    AnalysisMode,
    APKInfo,
    # Data Models
    BinaryInfo,
    # Enums
    BinaryType,
    CodeAnalysisResult,
    DecompileConfig,
    DecompileResult,
    DecompilerType,
    MalwareAnalysisResult,
    ThreatLevel,
)

__all__ = [
    "ReverseEngineeringManager",
    "BinaryType",
    "AnalysisMode",
    "DecompilerType",
    "ThreatLevel",
    "BinaryInfo",
    "APKInfo",
    "DecompileConfig",
    "DecompileResult",
    "MalwareAnalysisResult",
    "CodeAnalysisResult",
]

__version__ = "1.0.0"
__risk_level__ = "L1"
