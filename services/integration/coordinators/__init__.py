"""Integration Coordinators - 協調器模組

提供雙閉環數據收集與處理協調器
"""

from .base_coordinator import (
    BaseCoordinator,
    BountyInfo,
    CoreFeedback,
    ErrorInfo,
    EvidenceData,
    FeatureResult,
    Finding,
    ImpactAssessment,
    OptimizationData,
    PerformanceMetrics,
    PoCData,
    RemediationAdvice,
    ReportData,
    StatisticsData,
    TargetInfo,
    VerificationResult,
)
from .xss_coordinator import XSSCoordinator

__all__ = [
    # Base classes
    "BaseCoordinator",
    # Data models
    "TargetInfo",
    "EvidenceData",
    "PoCData",
    "ImpactAssessment",
    "RemediationAdvice",
    "BountyInfo",
    "Finding",
    "StatisticsData",
    "PerformanceMetrics",
    "ErrorInfo",
    "FeatureResult",
    "OptimizationData",
    "ReportData",
    "VerificationResult",
    "CoreFeedback",
    # Specific coordinators
    "XSSCoordinator",
]
