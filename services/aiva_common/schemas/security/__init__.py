"""
AIVA Security Domain Schemas
============================

安全檢測領域模型，包含：
- 安全事件統一標準
- 漏洞發現與細節
- 威脅情報模型
- 低價值漏洞檢測

此領域專注於安全檢測、威脅識別和漏洞管理。
"""

from .events import *
from .findings import *
from .threat_intel import *

__all__ = [
    # 安全事件 (security_events.py)
    "BaseSIEMEvent",
    "BaseAttackPathNode",
    "BaseAttackPathEdge",
    "BaseAttackPath",
    "EnhancedSIEMEvent",
    "EventStatus",
    "SkillLevel",
    "Priority",
    "AttackPathNodeType",
    "AttackPathEdgeType",
    "LegacySIEMEventAdapter",
    "LegacyAttackPathAdapter",
    # 漏洞發現 (findings.py) - 只包含真正存在的類別
    "Vulnerability",
    "Target",
    "FindingTarget",  # 別名指向Target
    "FindingEvidence",
    "FindingImpact",
    "FindingRecommendation",
    "FindingPayload",
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
    "VulnerabilityCorrelation",
    "VulnerabilityScorecard",
    "CodeLevelRootCause",
    "SASTDASTCorrelation",
    "AIVerificationRequest",
    "AIVerificationResult",
    # 威脅情報 (threat_intelligence.py)
    "STIXDomainObject",
    "STIXRelationshipObject",
    "AttackPattern",
    "Malware",
    "Indicator",
    "ThreatActor",
    "IntrusionSet",
    "Campaign",
    "CourseOfAction",
    "Vulnerability",
    "Tool",
    "ObservedData",
    "Report",
    "Relationship",
    "Sighting",
    "Bundle",
    "ExternalReference",
    "GranularMarking",
    "KillChainPhase",
    "TAXIICollection",
    "TAXIIManifest",
    "TAXIIManifestEntry",
    "TAXIIStatus",
    "TAXIIErrorMessage",
    "ThreatIntelligenceReport",
    "IOCEnrichment",
    "BugBountyIntelligence",
    "LowValueVulnerabilityPattern",
]
