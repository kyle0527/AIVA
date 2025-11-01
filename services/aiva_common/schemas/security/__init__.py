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
from .vulnerabilities import *
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
    # 漏洞發現 (findings.py)
    "Vulnerability",
    "Target",
    "FindingTarget",
    "FindingEvidence",
    "FindingImpact", 
    "FindingRecommendation",
    "FindingPayload",
    "SensitiveMatch",
    "VulnerabilityCorrelation",
    "VulnerabilityScorecard",
    "CodeLevelRootCause",
    "SASTDASTCorrelation",
    "AIVerificationRequest", 
    "AIVerificationResult",
    # 低價值漏洞 (low_value_vulnerabilities.py)
    "LowValueVulnerabilityType",
    "VulnerabilityPattern", 
    "InfoDisclosurePattern",
    "ErrorMessageDisclosure",
    "DebugInfoDisclosure",
    "XSSPattern",
    "ReflectedXSSBasic",
    "DOMXSSSimple",
    "CSRFPattern",
    "CSRFMissingToken",
    "CSRFJSONBypass",
    "IDORPattern",
    "IDORSimpleID", 
    "IDORUserData",
    "OpenRedirectPattern",
    "HostHeaderInjectionPattern",
    "CORSMisconfigurationPattern",
    "ClickjackingPattern",
    "LowValueVulnerabilityTest",
    "LowValueVulnerabilityResult",
    "BugBountyStrategy",
    "BountyPrediction",
    "ROIAnalysis",
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
    "STIXVulnerability",
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