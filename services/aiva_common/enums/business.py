"""
業務和商業邏輯枚舉

遵循以下標準：
- ISO 9001:2015 品質管理系統
- ISO/IEC 27001:2022 資訊安全管理
- COSO Enterprise Risk Management Framework
- ITIL v4 Service Value System
- TOGAF 9.2 企業架構框架
- Business Process Model and Notation (BPMN) 2.0
- Six Sigma 品質管理方法論
- Balanced Scorecard 戰略管理工具
"""

from enum import Enum

# ==================== 業務流程和工作流 ====================


class BusinessProcessType(Enum):
    """業務流程類型 - 基於 BPMN 2.0"""

    CORE = "CORE"  # 核心業務流程
    SUPPORT = "SUPPORT"  # 支援流程
    MANAGEMENT = "MANAGEMENT"  # 管理流程
    OPERATIONAL = "OPERATIONAL"  # 操作流程
    STRATEGIC = "STRATEGIC"  # 戰略流程
    TACTICAL = "TACTICAL"  # 戰術流程
    CUSTOMER_FACING = "CUSTOMER_FACING"  # 面向客戶流程
    INTERNAL = "INTERNAL"  # 內部流程


class ProcessStatus(Enum):
    """流程狀態"""

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    WAITING = "WAITING"
    ESCALATED = "ESCALATED"


class TaskType(Enum):
    """任務類型 - 基於 BPMN"""

    USER_TASK = "USER_TASK"
    SERVICE_TASK = "SERVICE_TASK"
    SCRIPT_TASK = "SCRIPT_TASK"
    BUSINESS_RULE_TASK = "BUSINESS_RULE_TASK"
    MANUAL_TASK = "MANUAL_TASK"
    RECEIVE_TASK = "RECEIVE_TASK"
    SEND_TASK = "SEND_TASK"
    SUBPROCESS = "SUBPROCESS"
    CALL_ACTIVITY = "CALL_ACTIVITY"


class ApprovalStatus(Enum):
    """審批狀態"""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DELEGATED = "DELEGATED"
    ESCALATED = "ESCALATED"
    WITHDRAWN = "WITHDRAWN"
    EXPIRED = "EXPIRED"
    ON_HOLD = "ON_HOLD"


# ==================== 客戶關係管理 ====================


class CustomerType(Enum):
    """客戶類型"""

    INDIVIDUAL = "INDIVIDUAL"  # 個人客戶
    ENTERPRISE = "ENTERPRISE"  # 企業客戶
    SMB = "SMB"  # 中小企業
    GOVERNMENT = "GOVERNMENT"  # 政府機構
    NON_PROFIT = "NON_PROFIT"  # 非營利組織
    PARTNER = "PARTNER"  # 合作夥伴
    RESELLER = "RESELLER"  # 經銷商
    DISTRIBUTOR = "DISTRIBUTOR"  # 分銷商


class CustomerStatus(Enum):
    """客戶狀態"""

    PROSPECT = "PROSPECT"  # 潛在客戶
    LEAD = "LEAD"  # 線索
    ACTIVE = "ACTIVE"  # 活躍客戶
    INACTIVE = "INACTIVE"  # 非活躍客戶
    SUSPENDED = "SUSPENDED"  # 暫停
    CHURNED = "CHURNED"  # 流失客戶
    VIP = "VIP"  # 重要客戶
    BLACKLISTED = "BLACKLISTED"  # 黑名單


class CustomerSegment(Enum):
    """客戶細分"""

    PREMIUM = "PREMIUM"
    STANDARD = "STANDARD"
    BASIC = "BASIC"
    ENTERPRISE = "ENTERPRISE"
    STARTUP = "STARTUP"
    EDUCATION = "EDUCATION"
    GOVERNMENT = "GOVERNMENT"
    NON_PROFIT = "NON_PROFIT"


class InteractionType(Enum):
    """互動類型"""

    PHONE_CALL = "PHONE_CALL"
    EMAIL = "EMAIL"
    CHAT = "CHAT"
    VIDEO_CALL = "VIDEO_CALL"
    MEETING = "MEETING"
    SUPPORT_TICKET = "SUPPORT_TICKET"
    SURVEY = "SURVEY"
    FEEDBACK = "FEEDBACK"
    COMPLAINT = "COMPLAINT"
    INQUIRY = "INQUIRY"


# ==================== 銷售和市場營銷 ====================


class SalesStage(Enum):
    """銷售階段 - 基於銷售漏斗模型"""

    AWARENESS = "AWARENESS"  # 認知階段
    INTEREST = "INTEREST"  # 興趣階段
    CONSIDERATION = "CONSIDERATION"  # 考慮階段
    INTENT = "INTENT"  # 意向階段
    EVALUATION = "EVALUATION"  # 評估階段
    PURCHASE = "PURCHASE"  # 購買階段
    RENEWAL = "RENEWAL"  # 續約階段
    UPSELL = "UPSELL"  # 向上銷售
    CROSS_SELL = "CROSS_SELL"  # 交叉銷售


class LeadSource(Enum):
    """線索來源"""

    WEBSITE = "WEBSITE"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    EMAIL_MARKETING = "EMAIL_MARKETING"
    SEARCH_ENGINE = "SEARCH_ENGINE"
    REFERRAL = "REFERRAL"
    ADVERTISEMENT = "ADVERTISEMENT"
    EVENT = "EVENT"
    WEBINAR = "WEBINAR"
    CONTENT_MARKETING = "CONTENT_MARKETING"
    DIRECT_SALES = "DIRECT_SALES"
    PARTNER = "PARTNER"
    TRADE_SHOW = "TRADE_SHOW"


class CampaignType(Enum):
    """活動類型"""

    EMAIL = "EMAIL"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    SEARCH_ENGINE = "SEARCH_ENGINE"
    DISPLAY = "DISPLAY"
    VIDEO = "VIDEO"
    CONTENT = "CONTENT"
    EVENT = "EVENT"
    WEBINAR = "WEBINAR"
    DIRECT_MAIL = "DIRECT_MAIL"
    TELEMARKETING = "TELEMARKETING"
    AFFILIATE = "AFFILIATE"
    INFLUENCER = "INFLUENCER"


class CampaignStatus(Enum):
    """活動狀態"""

    DRAFT = "DRAFT"
    SCHEDULED = "SCHEDULED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


# ==================== 產品和服務管理 ====================


class ProductType(Enum):
    """產品類型"""

    PHYSICAL = "PHYSICAL"  # 實體產品
    DIGITAL = "DIGITAL"  # 數位產品
    SERVICE = "SERVICE"  # 服務
    SUBSCRIPTION = "SUBSCRIPTION"  # 訂閱服務
    LICENSE = "LICENSE"  # 授權
    BUNDLE = "BUNDLE"  # 套裝產品
    ADDON = "ADDON"  # 附加產品
    CONSUMABLE = "CONSUMABLE"  # 消耗品


class ProductStatus(Enum):
    """產品狀態"""

    DEVELOPMENT = "DEVELOPMENT"  # 開發中
    BETA = "BETA"  # 測試版
    ACTIVE = "ACTIVE"  # 活躍
    MAINTENANCE = "MAINTENANCE"  # 維護中
    DEPRECATED = "DEPRECATED"  # 已棄用
    END_OF_LIFE = "END_OF_LIFE"  # 生命週期結束
    DISCONTINUED = "DISCONTINUED"  # 停產


class ServiceLevel(Enum):
    """服務級別"""

    BASIC = "BASIC"
    STANDARD = "STANDARD"
    PREMIUM = "PREMIUM"
    ENTERPRISE = "ENTERPRISE"
    CUSTOM = "CUSTOM"


class PricingModel(Enum):
    """定價模型"""

    FIXED = "FIXED"  # 固定價格
    USAGE_BASED = "USAGE_BASED"  # 使用量計費
    SUBSCRIPTION = "SUBSCRIPTION"  # 訂閱制
    FREEMIUM = "FREEMIUM"  # 免費增值
    TIERED = "TIERED"  # 分層定價
    VOLUME = "VOLUME"  # 量級定價
    DYNAMIC = "DYNAMIC"  # 動態定價
    AUCTION = "AUCTION"  # 拍賣制


# ==================== 財務和會計 ====================


class AccountType(Enum):
    """會計科目類型 - 基於會計準則"""

    ASSET = "ASSET"  # 資產
    LIABILITY = "LIABILITY"  # 負債
    EQUITY = "EQUITY"  # 所有者權益
    REVENUE = "REVENUE"  # 收入
    EXPENSE = "EXPENSE"  # 費用
    CONTRA_ASSET = "CONTRA_ASSET"  # 備抵資產
    CONTRA_LIABILITY = "CONTRA_LIABILITY"  # 備抵負債


class TransactionType(Enum):
    """交易類型"""

    SALE = "SALE"  # 銷售
    PURCHASE = "PURCHASE"  # 採購
    PAYMENT = "PAYMENT"  # 付款
    RECEIPT = "RECEIPT"  # 收款
    REFUND = "REFUND"  # 退款
    ADJUSTMENT = "ADJUSTMENT"  # 調整
    TRANSFER = "TRANSFER"  # 轉賬
    JOURNAL_ENTRY = "JOURNAL_ENTRY"  # 日記帳分錄


class PaymentMethod(Enum):
    """付款方式"""

    CASH = "CASH"
    CREDIT_CARD = "CREDIT_CARD"
    DEBIT_CARD = "DEBIT_CARD"
    BANK_TRANSFER = "BANK_TRANSFER"
    WIRE_TRANSFER = "WIRE_TRANSFER"
    PAYPAL = "PAYPAL"
    CRYPTOCURRENCY = "CRYPTOCURRENCY"
    CHECK = "CHECK"
    MOBILE_PAYMENT = "MOBILE_PAYMENT"
    DIGITAL_WALLET = "DIGITAL_WALLET"


class PaymentStatus(Enum):
    """付款狀態"""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"
    DISPUTED = "DISPUTED"
    EXPIRED = "EXPIRED"


class InvoiceStatus(Enum):
    """發票狀態"""

    DRAFT = "DRAFT"
    SENT = "SENT"
    VIEWED = "VIEWED"
    PAID = "PAID"
    OVERDUE = "OVERDUE"
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"
    DISPUTED = "DISPUTED"


# ==================== 人力資源管理 ====================


class EmployeeType(Enum):
    """員工類型"""

    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    TEMPORARY = "TEMPORARY"
    INTERN = "INTERN"
    CONSULTANT = "CONSULTANT"
    FREELANCER = "FREELANCER"
    VOLUNTEER = "VOLUNTEER"


class EmployeeStatus(Enum):
    """員工狀態"""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ON_LEAVE = "ON_LEAVE"
    TERMINATED = "TERMINATED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"
    PROBATION = "PROBATION"


class Department(Enum):
    """部門"""

    EXECUTIVE = "EXECUTIVE"
    SALES = "SALES"
    MARKETING = "MARKETING"
    FINANCE = "FINANCE"
    HUMAN_RESOURCES = "HUMAN_RESOURCES"
    OPERATIONS = "OPERATIONS"
    INFORMATION_TECHNOLOGY = "INFORMATION_TECHNOLOGY"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    RESEARCH_DEVELOPMENT = "RESEARCH_DEVELOPMENT"
    LEGAL = "LEGAL"
    PROCUREMENT = "PROCUREMENT"
    QUALITY_ASSURANCE = "QUALITY_ASSURANCE"


class PerformanceRating(Enum):
    """績效評級"""

    OUTSTANDING = "OUTSTANDING"  # 5 - 優秀
    EXCEEDS_EXPECTATIONS = "EXCEEDS_EXPECTATIONS"  # 4 - 超出預期
    MEETS_EXPECTATIONS = "MEETS_EXPECTATIONS"  # 3 - 符合預期
    BELOW_EXPECTATIONS = "BELOW_EXPECTATIONS"  # 2 - 低於預期
    UNSATISFACTORY = "UNSATISFACTORY"  # 1 - 不滿意


# ==================== 供應鏈管理 ====================


class SupplierType(Enum):
    """供應商類型"""

    MANUFACTURER = "MANUFACTURER"  # 製造商
    DISTRIBUTOR = "DISTRIBUTOR"  # 分銷商
    WHOLESALER = "WHOLESALER"  # 批發商
    RETAILER = "RETAILER"  # 零售商
    SERVICE_PROVIDER = "SERVICE_PROVIDER"  # 服務提供商
    CONTRACTOR = "CONTRACTOR"  # 承包商
    CONSULTANT = "CONSULTANT"  # 顧問


class SupplierStatus(Enum):
    """供應商狀態"""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    BLOCKED = "BLOCKED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    EVALUATION = "EVALUATION"
    TERMINATED = "TERMINATED"


class PurchaseOrderStatus(Enum):
    """採購訂單狀態"""

    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class DeliveryStatus(Enum):
    """交付狀態"""

    PENDING = "PENDING"
    SHIPPED = "SHIPPED"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"
    RETURNED = "RETURNED"
    DAMAGED = "DAMAGED"


# ==================== 風險管理和合規 ====================


class RiskCategory(Enum):
    """風險類別 - 基於 COSO ERM 框架"""

    STRATEGIC = "STRATEGIC"  # 戰略風險
    OPERATIONAL = "OPERATIONAL"  # 操作風險
    FINANCIAL = "FINANCIAL"  # 財務風險
    COMPLIANCE = "COMPLIANCE"  # 合規風險
    REPUTATIONAL = "REPUTATIONAL"  # 聲譽風險
    TECHNOLOGY = "TECHNOLOGY"  # 技術風險
    LEGAL = "LEGAL"  # 法律風險
    ENVIRONMENTAL = "ENVIRONMENTAL"  # 環境風險
    CYBER = "CYBER"  # 網絡風險
    MARKET = "MARKET"  # 市場風險


class RiskLevel(Enum):
    """風險級別"""

    VERY_LOW = "VERY_LOW"  # 1
    LOW = "LOW"  # 2
    MEDIUM = "MEDIUM"  # 3
    HIGH = "HIGH"  # 4
    VERY_HIGH = "VERY_HIGH"  # 5
    CRITICAL = "CRITICAL"  # 6


class RiskStatus(Enum):
    """風險狀態"""

    IDENTIFIED = "IDENTIFIED"
    ASSESSED = "ASSESSED"
    MITIGATED = "MITIGATED"
    ACCEPTED = "ACCEPTED"
    TRANSFERRED = "TRANSFERRED"
    AVOIDED = "AVOIDED"
    MONITORED = "MONITORED"
    CLOSED = "CLOSED"


class ControlType(Enum):
    """控制類型"""

    PREVENTIVE = "PREVENTIVE"  # 預防性控制
    DETECTIVE = "DETECTIVE"  # 偵測性控制
    CORRECTIVE = "CORRECTIVE"  # 糾正性控制
    COMPENSATING = "COMPENSATING"  # 補償性控制
    DIRECTIVE = "DIRECTIVE"  # 指導性控制


# ==================== 品質管理 ====================


class QualityStandard(Enum):
    """品質標準"""

    ISO_9001 = "ISO_9001"
    ISO_14001 = "ISO_14001"
    ISO_27001 = "ISO_27001"
    ISO_45001 = "ISO_45001"
    SIX_SIGMA = "SIX_SIGMA"
    LEAN = "LEAN"
    CMMI = "CMMI"
    ITIL = "ITIL"


class QualityMetric(Enum):
    """品質指標"""

    DEFECT_RATE = "DEFECT_RATE"
    CUSTOMER_SATISFACTION = "CUSTOMER_SATISFACTION"
    FIRST_PASS_YIELD = "FIRST_PASS_YIELD"
    CYCLE_TIME = "CYCLE_TIME"
    PROCESS_CAPABILITY = "PROCESS_CAPABILITY"
    SIGMA_LEVEL = "SIGMA_LEVEL"
    COST_OF_QUALITY = "COST_OF_QUALITY"
    CUSTOMER_COMPLAINTS = "CUSTOMER_COMPLAINTS"


class ImprovementMethodology(Enum):
    """改進方法論"""

    PDCA = "PDCA"  # Plan-Do-Check-Act
    DMAIC = "DMAIC"  # Define-Measure-Analyze-Improve-Control
    KAIZEN = "KAIZEN"  # 持續改進
    LEAN_SIX_SIGMA = "LEAN_SIX_SIGMA"
    AGILE = "AGILE"
    DESIGN_THINKING = "DESIGN_THINKING"


# ==================== 戰略管理和治理 ====================


class StrategicObjective(Enum):
    """戰略目標 - 基於平衡計分卡"""

    FINANCIAL = "FINANCIAL"  # 財務面向
    CUSTOMER = "CUSTOMER"  # 客戶面向
    INTERNAL_PROCESS = "INTERNAL_PROCESS"  # 內部流程面向
    LEARNING_GROWTH = "LEARNING_GROWTH"  # 學習與成長面向


class GovernanceLevel(Enum):
    """治理層級"""

    BOARD = "BOARD"  # 董事會
    EXECUTIVE = "EXECUTIVE"  # 高階主管
    MANAGEMENT = "MANAGEMENT"  # 管理層
    OPERATIONAL = "OPERATIONAL"  # 操作層


class ComplianceFrameworkType(Enum):
    """合規框架類型"""

    REGULATORY = "REGULATORY"  # 法規合規
    INDUSTRY = "INDUSTRY"  # 行業標準
    INTERNAL = "INTERNAL"  # 內部政策
    CONTRACTUAL = "CONTRACTUAL"  # 合約要求
    VOLUNTARY = "VOLUNTARY"  # 自願遵循


class AuditResult(Enum):
    """審計結果"""

    SATISFACTORY = "SATISFACTORY"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    UNSATISFACTORY = "UNSATISFACTORY"
    NON_COMPLIANT = "NON_COMPLIANT"
    MATERIAL_WEAKNESS = "MATERIAL_WEAKNESS"
    SIGNIFICANT_DEFICIENCY = "SIGNIFICANT_DEFICIENCY"
