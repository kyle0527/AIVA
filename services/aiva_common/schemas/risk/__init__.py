"""
AIVA Risk Domain Schemas  
========================

風險評估領域模型，包含：
- 風險評估模型
- 攻擊路徑分析
- 標準參考資料

此領域專注於風險量化、攻擊路徑建模和風險管理。
"""

from .assessment import *
from .attack_paths import *
from .references import *

__all__ = [
    # 風險評估 (risk.py)
    "RiskAssessmentContext",
    "RiskAssessmentResult",
    "RiskTrendAnalysis",
    "AttackPathNode",
    "AttackPathEdge", 
    "AttackPathPayload",
    "AttackPathRecommendation",
    # 標準參考 (references.py)
    "CAPECReference",
    "CVEReference",
    "CWEReference",
    "VulnerabilityDiscovery",
]