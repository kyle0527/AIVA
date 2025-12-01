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

# ==================== 從 aiva_common 導入共享基礎設施 ====================
from ..aiva_common.enums import (
    IntelSource,
    Severity,
)
from ..aiva_common.schemas import (
    NotificationPayload,
    SIEMEventPayload,
    ThreatIntelLookupPayload,
    ThreatIntelResultPayload,
    WebhookPayload,
)

# ==================== 從本模組導入 Integration 專屬類 ====================
from .models import (
    EnhancedIOCRecord,
    SIEMEvent,
)

__all__ = [
    # ==================== 來自 aiva_common ====================
    # 枚舉類
    "IntelSource",
    "Severity",
    # 共享 Schema
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "SIEMEventPayload",
    "NotificationPayload",
    "WebhookPayload",
    # ==================== 來自本模組 (Integration 專屬) ====================
    "EnhancedIOCRecord",
    "SIEMEvent",
    # ==================== 註冊函數 ====================
    "register_search_handler_to_command_center",
]


def register_search_handler_to_command_center(config: dict = None) -> None:
    """
    註冊 Search 命令處理器到 AI 命令中心
    
    這個函數由外部調用（通常是 Core 模組初始化時），
    避免了跨模組的直接 import 依賴。
    
    Args:
        config: 搜索配置字典，包含各種 API Key
    
    使用示例:
        # 在 AI Commander 或主程式中
        from services import integration
        integration.register_search_handler_to_command_center({
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            ...
        })
    """
    from services.aiva_common.command_center import get_command_center
    from .search_command_handler import SearchCommandHandler
    
    command_center = get_command_center()
    search_handler = SearchCommandHandler(config=config or {})
    command_center.register_module("search", search_handler)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("✅ Search 命令處理器已註冊到 AI 命令中心")
