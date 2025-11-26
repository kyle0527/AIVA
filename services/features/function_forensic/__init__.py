"""
Forensic Tools Module

提供數位鑑識分析能力，包括文件分析、記憶體取證、網路流量分析等功能。

風險等級: L0 (分析工具)
模組版本: 1.0.0
"""

from .manager import ForensicManager
from .models import (
    # Enums
    ForensicAnalysisType,
    EvidenceType,
    FileSystemType,
    ArtifactCategory,
    
    # Data Models
    CaseInfo,
    EvidenceItem,
    AnalysisConfig,
    AnalysisResult,
    TimelineEvent
)

__all__ = [
    "ForensicManager",
    "ForensicAnalysisType",
    "EvidenceType",
    "FileSystemType",
    "ArtifactCategory",
    "CaseInfo",
    "EvidenceItem",
    "AnalysisConfig",
    "AnalysisResult",
    "TimelineEvent",
]

__version__ = "1.0.0"
__risk_level__ = "L0"
