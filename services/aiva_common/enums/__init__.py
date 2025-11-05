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

# ==================== 學術研究和知識管理 ====================
from .academic import (
    AccessLevel,
    CitationMetric,
    CollaborationType,
    ConferenceRank,
    ConferenceType,
    ConsentType,
    # DataFormat 已統一到 common.py 避免重複
    DataType,
    EthicsApprovalStatus,
    FundingType,
    ImpactMeasure,
    IntellectualPropertyType,
    JournalRanking,
    KnowledgeType,
    LicenseType,
    MetadataStandard,
    OntologyType,
    ParticipationType,
    PeerReviewType,
    PresentationFormat,
    PublicationType,
    ResearchDiscipline,
    ResearcherRole,
    ResearchEthicsCategory,
    ResearchMethodology,
    ResearchPhase,
    ResearchType,
    ReviewStatus,
    SemanticRelation,
)

# ==================== AI 系統相關 ====================
# from .ai import (
#     # AI 相關枚舉暫時註釋，需要時再啟用
# )

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

# ==================== 業務和商業邏輯 ====================
from .business import (
    AccountType,
    ApprovalStatus,
    AuditResult,
    BusinessProcessType,
    CampaignStatus,
    CampaignType,
    ComplianceFrameworkType,
    ControlType,
    CustomerSegment,
    CustomerStatus,
    CustomerType,
    DeliveryStatus,
    Department,
    EmployeeStatus,
    EmployeeType,
    GovernanceLevel,
    ImprovementMethodology,
    InteractionType,
    InvoiceStatus,
    LeadSource,
    PaymentMethod,
    PaymentStatus,
    PerformanceRating,
    PricingModel,
    ProcessStatus,
    ProductStatus,
    ProductType,
    PurchaseOrderStatus,
    QualityMetric,
    QualityStandard,
    RiskCategory,
    # RiskLevel 已統一到 common.py 避免重複
    RiskStatus,
    SalesStage,
    ServiceLevel,
    StrategicObjective,
    SupplierStatus,
    SupplierType,
    TaskType,
    TransactionType,
)
from .common import (
    AlertType,
    AsyncTaskStatus,
    Confidence,
    CVSSSeverity,      # 新的 CVSS v4.0 官方標準
    DataFormat,
    EncodingType,
    ErrorCategory,
    HttpMethod,
    HTTPStatusClass,   # 新的 RFC 7231 官方術語
    HttpStatusCodeRange,  # 向後相容別名
    LogLevel,
    NetworkProtocol,
    OperationResult,
    PluginStatus,
    PluginType,
    RemediationStatus,
    ResponseDataType,
    RiskLevel,
    ScanStatus,
    ScanStrategy,
    Severity,          # 向後相容別名
    StoppingReason,
    TaskStatus,
    TaskType,
    TestStatus,
    ThreatLevel,
    ValidationResult,
)

# ==================== 數據模型和格式 ====================
# from .data_models import (
#     # 數據模型相關枚舉暫時註釋
# )

# ==================== 基礎設施和網絡 ====================
# from .infrastructure import (
#     # 基礎設施相關枚舉暫時註釋
# )

# ==================== 模組相關 ====================
from .modules import (
    CodeQualityMetric,
    ECMAScriptVersion,
    JavaScriptFeature,
    LanguageFramework,
    ModuleName,
    ProgrammingLanguage,
    Topic,
)

# ==================== 系統操作和運維 ====================
# from .operations import (
#     # 運維相關枚舉暫時註釋
# )

# ==================== 滲透測試系統 ====================
# from .pentest import (
#     # 滲透測試相關枚舉暫時註釋
# )

# ==================== 安全測試 ====================
# from .security import (
#     # 安全測試相關枚舉暫時註釋
# )

# ==================== 用戶界面和用戶體驗 ====================
# from .ui_ux import (
#     # UI/UX 相關枚舉暫時註釋
# )

# ==================== Web API 標準 ====================
from .web_api_standards import (
    HTTPStatusCode,
    JSONSchemaKeyword,
    JWTAlgorithm,
    JWTClaim,
    OpenAPIFormat,
    OpenAPIParameterLocation,
    OpenAPISchemaType,
    OpenAPISecuritySchemeType,
    SARIFArtifactRoles,
    SARIFLevel,
    SARIFResultKind,
    WebStandard,
)

# __all__ 由於複雜性暫時移除，使用 import * 時會自動導入所有公開符號

# 版本資訊
__version__ = "2.1.0"
