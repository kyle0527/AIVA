"""
基礎設施和網絡相關枚舉

遵循以下標準：
- NIST Cybersecurity Framework v1.1
- ISO/IEC 27001:2022
- IEEE 802.11 標準（無線網絡）
- TCP/IP 協議族標準
- OSI 七層模型
- ITIL v4 Service Management
- AWS Well-Architected Framework
- 雲原生計算基金會(CNCF)標準
"""

from enum import Enum

# ==================== 網絡協議和服務 ====================


class NetworkProtocol(Enum):
    """網絡協議類型 - 基於 TCP/IP 協議族"""

    # 應用層協議
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    FTP = "FTP"
    FTPS = "FTPS"
    SFTP = "SFTP"
    SSH = "SSH"
    TELNET = "TELNET"
    SMTP = "SMTP"
    SMTPS = "SMTPS"
    POP3 = "POP3"
    POP3S = "POP3S"
    IMAP = "IMAP"
    IMAPS = "IMAPS"
    DNS = "DNS"
    DHCP = "DHCP"
    SNMP = "SNMP"
    LDAP = "LDAP"
    LDAPS = "LDAPS"
    NTP = "NTP"
    TFTP = "TFTP"

    # 傳輸層協議
    TCP = "TCP"
    UDP = "UDP"
    SCTP = "SCTP"

    # 網絡層協議
    IP = "IP"
    IPv4 = "IPv4"
    IPv6 = "IPv6"
    ICMP = "ICMP"
    ICMPv6 = "ICMPv6"
    IGMP = "IGMP"

    # 鏈路層協議
    ETHERNET = "ETHERNET"
    WIFI = "WIFI"
    BLUETOOTH = "BLUETOOTH"
    PPP = "PPP"

    # VPN 協議
    IPSEC = "IPSEC"
    L2TP = "L2TP"
    PPTP = "PPTP"
    OPENVPN = "OPENVPN"
    WIREGUARD = "WIREGUARD"


class PortType(Enum):
    """端口類型分類 - 基於 IANA 端口分配"""

    WELL_KNOWN = "WELL_KNOWN"  # 0-1023
    REGISTERED = "REGISTERED"  # 1024-49151
    DYNAMIC = "DYNAMIC"  # 49152-65535
    EPHEMERAL = "EPHEMERAL"  # 動態端口


class FirewallAction(Enum):
    """防火牆動作 - 基於網絡安全標準"""

    ALLOW = "ALLOW"
    DENY = "DENY"
    DROP = "DROP"
    REJECT = "REJECT"
    LOG = "LOG"
    RATE_LIMIT = "RATE_LIMIT"


class NetworkTopology(Enum):
    """網絡拓撲類型"""

    STAR = "STAR"
    BUS = "BUS"
    RING = "RING"
    MESH = "MESH"
    TREE = "TREE"
    HYBRID = "HYBRID"
    POINT_TO_POINT = "POINT_TO_POINT"
    POINT_TO_MULTIPOINT = "POINT_TO_MULTIPOINT"


# ==================== 雲端基礎設施 ====================


class CloudProvider(Enum):
    """雲端服務提供商"""

    AWS = "AWS"
    AZURE = "AZURE"
    GCP = "GCP"
    ALIBABA_CLOUD = "ALIBABA_CLOUD"
    IBM_CLOUD = "IBM_CLOUD"
    ORACLE_CLOUD = "ORACLE_CLOUD"
    DIGITALOCEAN = "DIGITALOCEAN"
    LINODE = "LINODE"
    VULTR = "VULTR"
    HEROKU = "HEROKU"
    PRIVATE_CLOUD = "PRIVATE_CLOUD"
    HYBRID_CLOUD = "HYBRID_CLOUD"
    MULTI_CLOUD = "MULTI_CLOUD"


class CloudServiceModel(Enum):
    """雲端服務模型 - 基於 NIST 定義"""

    IAAS = "IAAS"  # Infrastructure as a Service
    PAAS = "PAAS"  # Platform as a Service
    SAAS = "SAAS"  # Software as a Service
    FAAS = "FAAS"  # Function as a Service
    CAAS = "CAAS"  # Container as a Service
    DAAS = "DAAS"  # Desktop as a Service
    BAAS = "BAAS"  # Backend as a Service


class CloudDeploymentModel(Enum):
    """雲端部署模型 - 基於 NIST 定義"""

    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    HYBRID = "HYBRID"
    COMMUNITY = "COMMUNITY"


class ContainerOrchestrator(Enum):
    """容器編排平台 - 基於 CNCF 標準"""

    KUBERNETES = "KUBERNETES"
    DOCKER_SWARM = "DOCKER_SWARM"
    APACHE_MESOS = "APACHE_MESOS"
    NOMAD = "NOMAD"
    OPENSHIFT = "OPENSHIFT"
    RANCHER = "RANCHER"


class ServiceMeshType(Enum):
    """服務網格類型 - 基於 CNCF 生態"""

    ISTIO = "ISTIO"
    LINKERD = "LINKERD"
    CONSUL_CONNECT = "CONSUL_CONNECT"
    ENVOY = "ENVOY"
    TRAEFIK_MESH = "TRAEFIK_MESH"


# ==================== 存儲系統 ====================


class StorageType(Enum):
    """存儲類型 - 基於存儲技術標準"""

    BLOCK = "BLOCK"
    FILE = "FILE"
    OBJECT = "OBJECT"
    DATABASE = "DATABASE"
    CACHE = "CACHE"
    DISTRIBUTED = "DISTRIBUTED"


class StorageInterface(Enum):
    """存儲接口類型"""

    SATA = "SATA"
    SAS = "SAS"
    NVME = "NVME"
    SCSI = "SCSI"
    IDE = "IDE"
    USB = "USB"
    THUNDERBOLT = "THUNDERBOLT"
    FIBRE_CHANNEL = "FIBRE_CHANNEL"
    ISCSI = "ISCSI"
    NFS = "NFS"
    SMB = "SMB"
    S3 = "S3"


class RAIDLevel(Enum):
    """RAID 級別 - 基於 RAID 標準"""

    RAID0 = "RAID0"
    RAID1 = "RAID1"
    RAID5 = "RAID5"
    RAID6 = "RAID6"
    RAID10 = "RAID10"
    RAID50 = "RAID50"
    RAID60 = "RAID60"
    NO_RAID = "NO_RAID"


class BackupType(Enum):
    """備份類型 - 基於數據保護標準"""

    FULL = "FULL"
    INCREMENTAL = "INCREMENTAL"
    DIFFERENTIAL = "DIFFERENTIAL"
    SNAPSHOT = "SNAPSHOT"
    CONTINUOUS = "CONTINUOUS"
    MIRROR = "MIRROR"


# ==================== 安全基礎設施 ====================


class AuthenticationMethod(Enum):
    """認證方法 - 基於認證標準"""

    PASSWORD = "PASSWORD"
    MULTI_FACTOR = "MULTI_FACTOR"
    BIOMETRIC = "BIOMETRIC"
    CERTIFICATE = "CERTIFICATE"
    TOKEN = "TOKEN"
    OAUTH = "OAUTH"
    SAML = "SAML"
    LDAP = "LDAP"
    ACTIVE_DIRECTORY = "ACTIVE_DIRECTORY"
    SINGLE_SIGN_ON = "SINGLE_SIGN_ON"
    FEDERATED = "FEDERATED"


class EncryptionAlgorithm(Enum):
    """加密算法 - 基於密碼學標準"""

    AES_128 = "AES_128"
    AES_192 = "AES_192"
    AES_256 = "AES_256"
    RSA_1024 = "RSA_1024"
    RSA_2048 = "RSA_2048"
    RSA_4096 = "RSA_4096"
    ECC_256 = "ECC_256"
    ECC_384 = "ECC_384"
    DES = "DES"
    TRIPLE_DES = "TRIPLE_DES"
    BLOWFISH = "BLOWFISH"
    TWOFISH = "TWOFISH"
    CHACHA20 = "CHACHA20"


class CertificateType(Enum):
    """證書類型 - 基於 PKI 標準"""

    SSL_TLS = "SSL_TLS"
    CODE_SIGNING = "CODE_SIGNING"
    EMAIL = "EMAIL"
    USER = "USER"
    DEVICE = "DEVICE"
    ROOT_CA = "ROOT_CA"
    INTERMEDIATE_CA = "INTERMEDIATE_CA"
    SELF_SIGNED = "SELF_SIGNED"
    WILDCARD = "WILDCARD"
    EXTENDED_VALIDATION = "EXTENDED_VALIDATION"


# ==================== 監控和觀測 ====================


class MonitoringType(Enum):
    """監控類型 - 基於可觀測性標準"""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    APPLICATION = "APPLICATION"
    NETWORK = "NETWORK"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    AVAILABILITY = "AVAILABILITY"
    BUSINESS = "BUSINESS"
    USER_EXPERIENCE = "USER_EXPERIENCE"


class MetricType(Enum):
    """指標類型 - 基於監控標準"""

    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"
    SUMMARY = "SUMMARY"
    TIMER = "TIMER"
    RATE = "RATE"
    PERCENTAGE = "PERCENTAGE"


class AlertSeverity(Enum):
    """告警嚴重程度 - 基於 ITIL 事件管理"""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    WARNING = "WARNING"


class LogLevel(Enum):
    """日誌級別 - 基於 Syslog RFC 3164"""

    EMERGENCY = "EMERGENCY"  # 0
    ALERT = "ALERT"  # 1
    CRITICAL = "CRITICAL"  # 2
    ERROR = "ERROR"  # 3
    WARNING = "WARNING"  # 4
    NOTICE = "NOTICE"  # 5
    INFO = "INFO"  # 6
    DEBUG = "DEBUG"  # 7


# ==================== 負載均衡和高可用 ====================


class LoadBalancingAlgorithm(Enum):
    """負載均衡算法"""

    ROUND_ROBIN = "ROUND_ROBIN"
    WEIGHTED_ROUND_ROBIN = "WEIGHTED_ROUND_ROBIN"
    LEAST_CONNECTIONS = "LEAST_CONNECTIONS"
    WEIGHTED_LEAST_CONNECTIONS = "WEIGHTED_LEAST_CONNECTIONS"
    LEAST_RESPONSE_TIME = "LEAST_RESPONSE_TIME"
    HASH = "HASH"
    IP_HASH = "IP_HASH"
    RANDOM = "RANDOM"
    WEIGHTED_RANDOM = "WEIGHTED_RANDOM"


class HighAvailabilityPattern(Enum):
    """高可用模式"""

    ACTIVE_ACTIVE = "ACTIVE_ACTIVE"
    ACTIVE_PASSIVE = "ACTIVE_PASSIVE"
    HOT_STANDBY = "HOT_STANDBY"
    COLD_STANDBY = "COLD_STANDBY"
    CLUSTER = "CLUSTER"
    LOAD_BALANCED = "LOAD_BALANCED"
    FAILOVER = "FAILOVER"
    DISASTER_RECOVERY = "DISASTER_RECOVERY"


class ScalingType(Enum):
    """擴展類型"""

    HORIZONTAL = "HORIZONTAL"  # Scale out
    VERTICAL = "VERTICAL"  # Scale up
    AUTO_SCALING = "AUTO_SCALING"
    MANUAL_SCALING = "MANUAL_SCALING"
    SCHEDULED_SCALING = "SCHEDULED_SCALING"
    PREDICTIVE_SCALING = "PREDICTIVE_SCALING"


# ==================== 網絡架構模式 ====================


class ArchitecturePattern(Enum):
    """架構模式 - 基於企業架構標準"""

    MONOLITHIC = "MONOLITHIC"
    MICROSERVICES = "MICROSERVICES"
    SERVICE_ORIENTED = "SERVICE_ORIENTED"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    LAYERED = "LAYERED"
    HEXAGONAL = "HEXAGONAL"
    CLEAN_ARCHITECTURE = "CLEAN_ARCHITECTURE"
    SERVERLESS = "SERVERLESS"
    EDGE_COMPUTING = "EDGE_COMPUTING"


class DeploymentStrategy(Enum):
    """部署策略"""

    BLUE_GREEN = "BLUE_GREEN"
    CANARY = "CANARY"
    ROLLING = "ROLLING"
    RECREATE = "RECREATE"
    A_B_TESTING = "A_B_TESTING"
    SHADOW = "SHADOW"
    FEATURE_FLAGS = "FEATURE_FLAGS"


class NetworkSegmentation(Enum):
    """網絡分段類型"""

    VLAN = "VLAN"
    SUBNET = "SUBNET"
    VPC = "VPC"
    SECURITY_GROUP = "SECURITY_GROUP"
    NETWORK_ACL = "NETWORK_ACL"
    FIREWALL_ZONE = "FIREWALL_ZONE"
    DMZ = "DMZ"
    PRIVATE_NETWORK = "PRIVATE_NETWORK"
    PUBLIC_NETWORK = "PUBLIC_NETWORK"


# ==================== 資源管理 ====================


class ResourceType(Enum):
    """資源類型"""

    COMPUTE = "COMPUTE"
    MEMORY = "MEMORY"
    STORAGE = "STORAGE"
    NETWORK = "NETWORK"
    GPU = "GPU"
    CPU = "CPU"
    BANDWIDTH = "BANDWIDTH"
    IOPS = "IOPS"
    LICENSE = "LICENSE"
    SERVICE = "SERVICE"


class ResourceState(Enum):
    """資源狀態 - 基於雲端資源生命週期"""

    CREATING = "CREATING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    STOPPING = "STOPPING"
    STARTING = "STARTING"
    TERMINATED = "TERMINATED"
    TERMINATING = "TERMINATING"
    ERROR = "ERROR"
    PENDING = "PENDING"
    UPDATING = "UPDATING"
    DELETING = "DELETING"
    DELETED = "DELETED"


class PerformanceTier(Enum):
    """性能等級"""

    BASIC = "BASIC"
    STANDARD = "STANDARD"
    PREMIUM = "PREMIUM"
    ULTRA = "ULTRA"
    BURSTABLE = "BURSTABLE"
    PROVISIONED = "PROVISIONED"
    ON_DEMAND = "ON_DEMAND"


# ==================== 災難恢復 ====================


class DisasterRecoveryTier(Enum):
    """災難恢復等級 - 基於業務連續性標準"""

    TIER_0 = "TIER_0"  # 24x7 冗余站點
    TIER_1 = "TIER_1"  # 冷站點
    TIER_2 = "TIER_2"  # 溫站點
    TIER_3 = "TIER_3"  # 熱站點
    TIER_4 = "TIER_4"  # 鏡像站點


class RecoveryMetric(Enum):
    """恢復指標"""

    RTO = "RTO"  # Recovery Time Objective
    RPO = "RPO"  # Recovery Point Objective
    MTTR = "MTTR"  # Mean Time To Recovery
    MTBF = "MTBF"  # Mean Time Between Failures
    AVAILABILITY = "AVAILABILITY"
    RELIABILITY = "RELIABILITY"


# ==================== 合規和治理 ====================


class ComplianceFramework(Enum):
    """合規框架"""

    SOX = "SOX"  # Sarbanes-Oxley
    PCI_DSS = "PCI_DSS"  # Payment Card Industry
    HIPAA = "HIPAA"  # Health Insurance Portability
    GDPR = "GDPR"  # General Data Protection Regulation
    SOC2 = "SOC2"  # Service Organization Control 2
    ISO27001 = "ISO27001"  # Information Security Management
    NIST = "NIST"  # National Institute of Standards
    COBIT = "COBIT"  # Control Objectives for IT
    ITIL = "ITIL"  # IT Infrastructure Library
    FedRAMP = "FedRAMP"  # Federal Risk and Authorization


class GovernanceModel(Enum):
    """治理模型"""

    CENTRALIZED = "CENTRALIZED"
    DECENTRALIZED = "DECENTRALIZED"
    FEDERATED = "FEDERATED"
    HYBRID = "HYBRID"
    SELF_SERVICE = "SELF_SERVICE"
    POLICY_DRIVEN = "POLICY_DRIVEN"
