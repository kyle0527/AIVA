"""
系統操作和運維枚舉

遵循以下標準：
- ITIL v4 Service Management Framework
- DevOps 最佳實踐和模式
- Site Reliability Engineering (SRE) 原則
- Agile 和 Scrum 框架
- COBIT 2019 治理框架
- ISO/IEC 20000-1:2018 Service Management
- NIST Cybersecurity Framework
- OWASP DevSecOps Guideline
"""

from enum import Enum

# ==================== DevOps 和 CI/CD ====================


class DevOpsStage(Enum):
    """DevOps 階段 - 基於 DevOps 生命週期"""

    PLAN = "PLAN"
    CODE = "CODE"
    BUILD = "BUILD"
    TEST = "TEST"
    RELEASE = "RELEASE"
    DEPLOY = "DEPLOY"
    OPERATE = "OPERATE"
    MONITOR = "MONITOR"
    FEEDBACK = "FEEDBACK"


class PipelineStage(Enum):
    """CI/CD 管道階段"""

    SOURCE = "SOURCE"
    BUILD = "BUILD"
    UNIT_TEST = "UNIT_TEST"
    INTEGRATION_TEST = "INTEGRATION_TEST"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    SECURITY_SCAN = "SECURITY_SCAN"
    PACKAGE = "PACKAGE"
    DEPLOY_STAGING = "DEPLOY_STAGING"
    ACCEPTANCE_TEST = "ACCEPTANCE_TEST"
    DEPLOY_PRODUCTION = "DEPLOY_PRODUCTION"
    SMOKE_TEST = "SMOKE_TEST"
    ROLLBACK = "ROLLBACK"


class BuildStatus(Enum):
    """構建狀態"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    UNSTABLE = "UNSTABLE"


class DeploymentEnvironment(Enum):
    """部署環境"""

    DEVELOPMENT = "DEVELOPMENT"
    TESTING = "TESTING"
    STAGING = "STAGING"
    UAT = "UAT"  # User Acceptance Testing
    PRODUCTION = "PRODUCTION"
    SANDBOX = "SANDBOX"
    INTEGRATION = "INTEGRATION"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    DEMO = "DEMO"


class ReleaseType(Enum):
    """發布類型 - 基於語義版本控制"""

    MAJOR = "MAJOR"  # 主版本 (不兼容更改)
    MINOR = "MINOR"  # 次版本 (向後兼容功能)
    PATCH = "PATCH"  # 補丁版本 (向後兼容修復)
    HOTFIX = "HOTFIX"  # 熱修復
    BETA = "BETA"  # 測試版
    ALPHA = "ALPHA"  # 內測版
    RC = "RC"  # Release Candidate
    SNAPSHOT = "SNAPSHOT"  # 快照版本


# ==================== 測試類型和策略 ====================


class TestType(Enum):
    """測試類型 - 基於測試金字塔模型"""

    UNIT = "UNIT"
    INTEGRATION = "INTEGRATION"
    SYSTEM = "SYSTEM"
    ACCEPTANCE = "ACCEPTANCE"
    SMOKE = "SMOKE"
    REGRESSION = "REGRESSION"
    PERFORMANCE = "PERFORMANCE"
    LOAD = "LOAD"
    STRESS = "STRESS"
    SECURITY = "SECURITY"
    PENETRATION = "PENETRATION"
    USABILITY = "USABILITY"
    COMPATIBILITY = "COMPATIBILITY"
    API = "API"
    UI = "UI"
    DATABASE = "DATABASE"
    MUTATION = "MUTATION"
    FUZZ = "FUZZ"
    CHAOS = "CHAOS"


class TestLevel(Enum):
    """測試層級"""

    COMPONENT = "COMPONENT"
    INTEGRATION = "INTEGRATION"
    SYSTEM = "SYSTEM"
    ACCEPTANCE = "ACCEPTANCE"


class TestingStrategy(Enum):
    """測試策略"""

    TDD = "TDD"  # Test-Driven Development
    BDD = "BDD"  # Behavior-Driven Development
    ATDD = "ATDD"  # Acceptance Test-Driven Development
    SHIFT_LEFT = "SHIFT_LEFT"  # 左移測試
    SHIFT_RIGHT = "SHIFT_RIGHT"  # 右移測試
    PYRAMID = "PYRAMID"  # 測試金字塔
    TROPHY = "TROPHY"  # 測試獎盃
    DIAMOND = "DIAMOND"  # 測試鑽石


class TestResult(Enum):
    """測試結果"""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


# ==================== 事件管理和操作 ====================


class IncidentSeverity(Enum):
    """事件嚴重程度 - 基於 ITIL 事件管理"""

    SEV1 = "SEV1"  # 嚴重 - 完全中斷
    SEV2 = "SEV2"  # 高 - 顯著影響
    SEV3 = "SEV3"  # 中 - 部分影響
    SEV4 = "SEV4"  # 低 - 最小影響
    SEV5 = "SEV5"  # 計劃性 - 無業務影響


class IncidentStatus(Enum):
    """事件狀態"""

    NEW = "NEW"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    ESCALATED = "ESCALATED"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    REOPENED = "REOPENED"


class IncidentCategory(Enum):
    """事件類別 - 基於 ITIL 分類"""

    HARDWARE = "HARDWARE"
    SOFTWARE = "SOFTWARE"
    NETWORK = "NETWORK"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    ACCESS = "ACCESS"
    DATA = "DATA"
    SERVICE = "SERVICE"
    PROCESS = "PROCESS"
    DOCUMENTATION = "DOCUMENTATION"


class ChangeType(Enum):
    """變更類型 - 基於 ITIL 變更管理"""

    STANDARD = "STANDARD"  # 標準變更
    NORMAL = "NORMAL"  # 普通變更
    EMERGENCY = "EMERGENCY"  # 緊急變更
    MAJOR = "MAJOR"  # 主要變更
    MINOR = "MINOR"  # 次要變更
    TECHNICAL = "TECHNICAL"  # 技術變更
    BUSINESS = "BUSINESS"  # 業務變更


class ChangeStatus(Enum):
    """變更狀態"""

    REQUESTED = "REQUESTED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    IMPLEMENTED = "IMPLEMENTED"
    TESTED = "TESTED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


# ==================== 性能和資源監控 ====================


class PerformanceMetric(Enum):
    """性能指標"""

    CPU_USAGE = "CPU_USAGE"
    MEMORY_USAGE = "MEMORY_USAGE"
    DISK_USAGE = "DISK_USAGE"
    NETWORK_THROUGHPUT = "NETWORK_THROUGHPUT"
    RESPONSE_TIME = "RESPONSE_TIME"
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"
    ERROR_RATE = "ERROR_RATE"
    UPTIME = "UPTIME"
    AVAILABILITY = "AVAILABILITY"
    CONCURRENT_USERS = "CONCURRENT_USERS"
    TRANSACTION_RATE = "TRANSACTION_RATE"
    QUEUE_LENGTH = "QUEUE_LENGTH"
    CONNECTION_COUNT = "CONNECTION_COUNT"


class CapacityMetric(Enum):
    """容量指標"""

    STORAGE_CAPACITY = "STORAGE_CAPACITY"
    NETWORK_BANDWIDTH = "NETWORK_BANDWIDTH"
    PROCESSING_POWER = "PROCESSING_POWER"
    MEMORY_CAPACITY = "MEMORY_CAPACITY"
    CONNECTION_POOL = "CONNECTION_POOL"
    THREAD_POOL = "THREAD_POOL"
    LICENSE_COUNT = "LICENSE_COUNT"
    USER_LIMIT = "USER_LIMIT"


class ThresholdType(Enum):
    """閾值類型"""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    NORMAL = "NORMAL"
    OPTIMAL = "OPTIMAL"
    UPPER_LIMIT = "UPPER_LIMIT"
    LOWER_LIMIT = "LOWER_LIMIT"
    BASELINE = "BASELINE"
    TARGET = "TARGET"


# ==================== 自動化和編排 ====================


class AutomationType(Enum):
    """自動化類型"""

    INFRASTRUCTURE = "INFRASTRUCTURE"  # Infrastructure as Code
    CONFIGURATION = "CONFIGURATION"  # Configuration Management
    DEPLOYMENT = "DEPLOYMENT"  # Deployment Automation
    TESTING = "TESTING"  # Test Automation
    MONITORING = "MONITORING"  # Monitoring Automation
    BACKUP = "BACKUP"  # Backup Automation
    SCALING = "SCALING"  # Auto Scaling
    RECOVERY = "RECOVERY"  # Disaster Recovery
    PROVISIONING = "PROVISIONING"  # Resource Provisioning
    ORCHESTRATION = "ORCHESTRATION"  # Workflow Orchestration


class WorkflowStatus(Enum):
    """工作流狀態"""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    RETRY = "RETRY"
    SKIPPED = "SKIPPED"


class ScheduleType(Enum):
    """排程類型"""

    CRON = "CRON"
    INTERVAL = "INTERVAL"
    ONE_TIME = "ONE_TIME"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    MANUAL = "MANUAL"
    CONDITIONAL = "CONDITIONAL"
    DEPENDENCY_BASED = "DEPENDENCY_BASED"


# ==================== 服務品質和SLA ====================


class ServiceLevel(Enum):
    """服務級別"""

    PLATINUM = "PLATINUM"  # 99.99% 可用性
    GOLD = "GOLD"  # 99.9% 可用性
    SILVER = "SILVER"  # 99.5% 可用性
    BRONZE = "BRONZE"  # 99% 可用性
    BASIC = "BASIC"  # 95% 可用性


class SLAMetric(Enum):
    """SLA 指標"""

    AVAILABILITY = "AVAILABILITY"
    RESPONSE_TIME = "RESPONSE_TIME"
    THROUGHPUT = "THROUGHPUT"
    ERROR_RATE = "ERROR_RATE"
    RESOLUTION_TIME = "RESOLUTION_TIME"
    MEAN_TIME_TO_REPAIR = "MEAN_TIME_TO_REPAIR"
    MEAN_TIME_BETWEEN_FAILURES = "MEAN_TIME_BETWEEN_FAILURES"
    SERVICE_QUALITY = "SERVICE_QUALITY"


class PriorityLevel(Enum):
    """優先級別 - 基於 ITIL 優先級矩陣"""

    CRITICAL = "CRITICAL"  # 緊急且影響大
    HIGH = "HIGH"  # 緊急或影響大
    MEDIUM = "MEDIUM"  # 中等緊急性和影響
    LOW = "LOW"  # 低緊急性和影響
    PLANNING = "PLANNING"  # 計劃性工作


# ==================== 運維工具和平台 ====================


class MonitoringTool(Enum):
    """監控工具"""

    PROMETHEUS = "PROMETHEUS"
    GRAFANA = "GRAFANA"
    NAGIOS = "NAGIOS"
    ZABBIX = "ZABBIX"
    DATADOG = "DATADOG"
    NEW_RELIC = "NEW_RELIC"
    SPLUNK = "SPLUNK"
    ELK_STACK = "ELK_STACK"
    JAEGER = "JAEGER"
    ZIPKIN = "ZIPKIN"
    PINGDOM = "PINGDOM"
    UPTIME_ROBOT = "UPTIME_ROBOT"


class ConfigurationTool(Enum):
    """配置管理工具"""

    ANSIBLE = "ANSIBLE"
    PUPPET = "PUPPET"
    CHEF = "CHEF"
    SALTSTACK = "SALTSTACK"
    TERRAFORM = "TERRAFORM"
    CLOUDFORMATION = "CLOUDFORMATION"
    HELM = "HELM"
    KUSTOMIZE = "KUSTOMIZE"


class ContainerTool(Enum):
    """容器工具"""

    DOCKER = "DOCKER"
    PODMAN = "PODMAN"
    CONTAINERD = "CONTAINERD"
    CRIO = "CRIO"
    LXC = "LXC"
    RUNC = "RUNC"


class OrchestrationTool(Enum):
    """編排工具"""

    KUBERNETES = "KUBERNETES"
    DOCKER_SWARM = "DOCKER_SWARM"
    NOMAD = "NOMAD"
    MESOS = "MESOS"
    OPENSHIFT = "OPENSHIFT"
    RANCHER = "RANCHER"


# ==================== 敏捷和項目管理 ====================


class AgileMethodology(Enum):
    """敏捷方法論"""

    SCRUM = "SCRUM"
    KANBAN = "KANBAN"
    EXTREME_PROGRAMMING = "EXTREME_PROGRAMMING"
    LEAN = "LEAN"
    SAFe = "SAFe"  # Scaled Agile Framework
    LESS = "LESS"  # Large-Scale Scrum
    SCRUMBAN = "SCRUMBAN"
    CRYSTAL = "CRYSTAL"
    DSDM = "DSDM"  # Dynamic Systems Development Method


class SprintStatus(Enum):
    """Sprint 狀態"""

    PLANNED = "PLANNED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    RETROSPECTIVE = "RETROSPECTIVE"


class AgileTaskStatus(Enum):
    """敏捷任務狀態"""

    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    CODE_REVIEW = "CODE_REVIEW"
    TESTING = "TESTING"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    ON_HOLD = "ON_HOLD"


class TeamRole(Enum):
    """團隊角色 - 基於 Scrum 和 DevOps 角色"""

    PRODUCT_OWNER = "PRODUCT_OWNER"
    SCRUM_MASTER = "SCRUM_MASTER"
    DEVELOPER = "DEVELOPER"
    TESTER = "TESTER"
    DEVOPS_ENGINEER = "DEVOPS_ENGINEER"
    SRE = "SRE"  # Site Reliability Engineer
    ARCHITECT = "ARCHITECT"
    TEAM_LEAD = "TEAM_LEAD"
    BUSINESS_ANALYST = "BUSINESS_ANALYST"
    UX_DESIGNER = "UX_DESIGNER"


# ==================== 品質保證和合規 ====================


class QualityGate(Enum):
    """品質門檻"""

    CODE_COVERAGE = "CODE_COVERAGE"
    SECURITY_SCAN = "SECURITY_SCAN"
    PERFORMANCE_TEST = "PERFORMANCE_TEST"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    VULNERABILITY_SCAN = "VULNERABILITY_SCAN"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    DOCUMENTATION = "DOCUMENTATION"
    PEER_REVIEW = "PEER_REVIEW"


class ComplianceStatus(Enum):
    """合規狀態"""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"
    EXCEPTION_GRANTED = "EXCEPTION_GRANTED"
    REMEDIATION_REQUIRED = "REMEDIATION_REQUIRED"


class AuditType(Enum):
    """審計類型"""

    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"
    PERFORMANCE = "PERFORMANCE"
    PROCESS = "PROCESS"
    FINANCIAL = "FINANCIAL"
    TECHNICAL = "TECHNICAL"
    OPERATIONAL = "OPERATIONAL"
    QUALITY = "QUALITY"


# ==================== 容量規劃和優化 ====================


class CapacityPlanningMethod(Enum):
    """容量規劃方法"""

    TREND_ANALYSIS = "TREND_ANALYSIS"
    BASELINE_MODELING = "BASELINE_MODELING"
    ANALYTICAL_MODELING = "ANALYTICAL_MODELING"
    SIMULATION_MODELING = "SIMULATION_MODELING"
    LOAD_TESTING = "LOAD_TESTING"
    BENCHMARK_TESTING = "BENCHMARK_TESTING"


class OptimizationType(Enum):
    """優化類型"""

    PERFORMANCE = "PERFORMANCE"
    COST = "COST"
    RESOURCE = "RESOURCE"
    AVAILABILITY = "AVAILABILITY"
    SCALABILITY = "SCALABILITY"
    SECURITY = "SECURITY"
    RELIABILITY = "RELIABILITY"
    MAINTAINABILITY = "MAINTAINABILITY"


class ResourceOptimizationStrategy(Enum):
    """資源優化策略"""

    RIGHT_SIZING = "RIGHT_SIZING"
    AUTO_SCALING = "AUTO_SCALING"
    LOAD_BALANCING = "LOAD_BALANCING"
    CACHING = "CACHING"
    COMPRESSION = "COMPRESSION"
    DEDUPLICATION = "DEDUPLICATION"
    ARCHIVING = "ARCHIVING"
    CONSOLIDATION = "CONSOLIDATION"
