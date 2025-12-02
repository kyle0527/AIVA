"""
Social Engineering Toolkit Module

提供社交工程攻擊能力，包括釣魚攻擊、憑證竊取、目標資訊收集等功能。

風險等級: L2 (需要授權)
模組版本: 1.0.0
"""

from .manager import SocialEngineeringManager
from .handler import SocialEngineeringCommandHandler
from .models import (
    # Enums
    PhishingType,
    TargetPlatform,
    DeliveryMethod,
    CredentialType,
    TemplateCategory,
    CampaignStatus,
    AnalyticsMetric,
    
    # Data Models
    PhishingConfig,
    PhishingResult,
    CampaignConfig,
    TargetInfo,
    CredentialData,
    AnalyticsData
)

__all__ = [
    # Manager
    "SocialEngineeringManager",
    "SocialEngineeringCommandHandler",
    
    # Enums
    "PhishingType",
    "TargetPlatform",
    "DeliveryMethod",
    "CredentialType",
    "TemplateCategory",
    "CampaignStatus",
    "AnalyticsMetric",
    
    # Data Models
    "PhishingConfig",
    "PhishingResult",
    "CampaignConfig",
    "TargetInfo",
    "CredentialData",
    "AnalyticsData",
]

__version__ = "1.0.0"
__risk_level__ = "L2"
__module_type__ = "social_engineering"
