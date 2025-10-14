"""
AIVA Integration - 整合模組

這是 AIVA 的整合模組包，負責與外部服務和系統的集成。

模組包含:
- aiva_integration: 主要整合功能
  - reception: 數據接收和處理
  - threat_intel: 威脅情報整合
  - perf_feedback: 性能反饋和優化
  - ai_engine: AI 引擎整合
  - storage: 存儲系統整合
"""

__version__ = "1.0.0"

# 從 aiva_common 導入共享基礎設施
from ..aiva_common.enums import (
    IntelSource,
    Severity,
)

# 從本模組導入集成相關模型
from .models import (
    EnhancedIOCRecord,
    NotificationPayload,
    SIEMEvent,
    SIEMEventPayload,
    ThreatIntelLookupPayload,
    ThreatIntelResultPayload,
    WebhookPayload,
)

__all__ = [
    # 來自 aiva_common
    "IntelSource",
    "Severity",
    # 來自本模組
    "EnhancedIOCRecord",
    "NotificationPayload",
    "SIEMEvent",
    "SIEMEventPayload",
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "WebhookPayload",
]
