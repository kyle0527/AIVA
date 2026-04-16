"""
Forensic Tools Module

提供數位鑑識分析能力，包括文件分析、記憶體取證、網路流量分析等功能。

風險等級: L0 (分析工具)
模組版本: 1.0.0
"""

from .manager import ForensicManager
from .models import (
    AnalysisConfig,
    AnalysisResult,
    ArtifactCategory,
    # Data Models
    CaseInfo,
    EvidenceItem,
    EvidenceType,
    FileSystemType,
    # Enums
    ForensicAnalysisType,
    TimelineEvent,
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

__version__ = "3.0.0"
__status__ = "production"
__architecture__ = "manager-based"
__risk_level__ = "L0"
__last_updated__ = "2026-01-23"
