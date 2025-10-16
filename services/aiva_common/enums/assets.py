"""
資產管理相關枚舉 - 資產類型、環境、合規等
"""

from __future__ import annotations

from enum import Enum


class BusinessCriticality(str, Enum):
    """業務重要性等級"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Environment(str, Enum):
    """環境類型"""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


class AssetType(str, Enum):
    """資產類型"""

    URL = "url"
    REPOSITORY = "repository"
    HOST = "host"
    CONTAINER = "container"
    API_ENDPOINT = "api_endpoint"
    MOBILE_APP = "mobile_app"
    WEB_APPLICATION = "web_application"
    DATABASE = "database"
    API_SERVICE = "api_service"


class AssetStatus(str, Enum):
    """資產狀態"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class DataSensitivity(str, Enum):
    """資料敏感度等級"""

    HIGHLY_SENSITIVE = "highly_sensitive"  # 信用卡、健康資料、密碼
    SENSITIVE = "sensitive"  # PII（個人識別信息）
    INTERNAL = "internal"  # 內部資料
    PUBLIC = "public"  # 公開資料


class AssetExposure(str, Enum):
    """資產網路暴露度"""

    INTERNET_FACING = "internet_facing"  # 直接暴露於互聯網
    DMZ = "dmz"  # DMZ 區域
    INTERNAL_NETWORK = "internal_network"  # 內部網路
    ISOLATED = "isolated"  # 隔離網路


class ComplianceFramework(str, Enum):
    """合規框架標籤"""

    PCI_DSS = "pci-dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"
    CIS = "cis"

