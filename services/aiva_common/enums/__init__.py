"""
AIVA Common Enums Package

此套件提供了 AIVA 微服務生態系統中所有枚舉類型的統一介面。

使用方式:
    from aiva_common.enums import ModuleName, Severity, VulnerabilityType

架構說明:
    - common.py: 通用枚舉 (狀態、級別等)
    - modules.py: 模組相關枚舉 (模組名稱、主題)
    - security.py: 安全測試枚舉 (漏洞、攻擊類型等)
    - assets.py: 資產管理枚舉 (資產類型、環境等)
"""

# ==================== 通用枚舉 ====================
from .common import (
    Severity,
    Confidence,
    TaskStatus,
    TestStatus,
    ScanStatus,
    ThreatLevel,
    RiskLevel,
    RemediationStatus,
)

# ==================== 模組相關 ====================
from .modules import (
    ProgrammingLanguage,
    LanguageFramework,
    CodeQualityMetric,
    ModuleName,
    Topic,
)

# ==================== 安全測試 ====================
from .security import (
    VulnerabilityByLanguage,
    SecurityPattern,
    VulnerabilityType,
    VulnerabilityStatus,
    Location,
    SensitiveInfoType,
    IntelSource,
    IOCType,
    RemediationType,
    Permission,
    AccessDecision,
    PostExTestType,
    PersistenceType,
    Exploitability,
    AttackPathNodeType,
    AttackPathEdgeType,
)

# ==================== 資產管理 ====================
from .assets import (
    BusinessCriticality,
    Environment,
    AssetType,
    AssetStatus,
    DataSensitivity,
    AssetExposure,
    ComplianceFramework,
)

# 為了保持向後相容，明確匯出所有公開介面
__all__ = [
    "AccessDecision",
    "AssetExposure",
    "AssetStatus",
    "AssetType",
    "AttackPathEdgeType",
    "AttackPathNodeType",
    "BusinessCriticality",
    "CodeQualityMetric",
    "ComplianceFramework",
    "Confidence",
    "DataSensitivity",
    "Environment",
    "Exploitability",
    "IOCType",
    "IntelSource",
    "LanguageFramework",
    "Location",
    "ModuleName",
    "Permission",
    "PersistenceType",
    "PostExTestType",
    "ProgrammingLanguage",
    "RemediationStatus",
    "RemediationType",
    "RiskLevel",
    "ScanStatus",
    "SecurityPattern",
    "SensitiveInfoType",
    "Severity",
    "TaskStatus",
    "TestStatus",
    "ThreatLevel",
    "Topic",
    "VulnerabilityByLanguage",
    "VulnerabilityStatus",
    "VulnerabilityType",
]

# 版本資訊
__version__ = "2.1.0"