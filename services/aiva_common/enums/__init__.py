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
# ==================== 資產管理 ====================
from .assets import (
    AssetExposure,
    AssetStatus,
    AssetType,
    BusinessCriticality,
    ComplianceFramework,
    DataSensitivity,
    Environment,
)
from .common import (
    Confidence,
    RemediationStatus,
    RiskLevel,
    ScanStatus,
    Severity,
    TaskStatus,
    TestStatus,
    ThreatLevel,
)

# ==================== 模組相關 ====================
from .modules import (
    CodeQualityMetric,
    LanguageFramework,
    ModuleName,
    ProgrammingLanguage,
    Topic,
)

# ==================== 安全測試 ====================
from .security import (
    AccessDecision,
    AttackPathEdgeType,
    AttackPathNodeType,
    Exploitability,
    IntelSource,
    IOCType,
    Location,
    Permission,
    PersistenceType,
    PostExTestType,
    RemediationType,
    SecurityPattern,
    SensitiveInfoType,
    VulnerabilityByLanguage,
    VulnerabilityStatus,
    VulnerabilityType,
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
