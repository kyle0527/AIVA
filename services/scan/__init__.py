"""
AIVA Scan - 多語言統一掃描引擎

支援:
- Python: 靜態內容爬取和表單分析
- TypeScript: 動態 JavaScript 渲染 (Playwright)
- Rust: 高性能敏感資訊掃描
- Go: 並發掃描和資產發現

新架構 (無 RabbitMQ):
- AI 直接指揮: 通過 AICommandCenter 下達命令
- 數據合約: 使用 aiva_common 統一數據模型
- 直接調用: 移除 RabbitMQ 消息隊列
"""

__version__ = "2.0.0"  # 無 RabbitMQ 版本

# 命令處理器 (AI 直接指揮接口)
from .command_handler import ScanCommandHandler

# 從 aiva_common 導入共享基礎設施 (標準優先)
from ..aiva_common.enums import (
    AssetType,
    ScanStatus,
    Severity,
)
from ..aiva_common.schemas import (
    Asset,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    Fingerprints,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
    Vulnerability,
)

# 從 coordinators 導入掃描特定擴展
from .coordinators.scan_models import (
    ScanCoordinationMetadata,
    EngineStatus,
    MultiEngineCoordinationResult,
)

from .coordinators.multi_engine_coordinator import (
    MultiEngineCoordinator,
)

# 導入各引擎模組（延遲導入以避免循環依賴）
try:
    from . import engines
    from . import coordinators
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"部分模組導入失敗（可能處於重構中）: {e}")

__all__ = [
    # 命令處理器 (新增)
    "ScanCommandHandler",
    "register_to_command_center",  # 新增: 註冊函數
    
    # ========== 從 aiva_common 導入（共享標準，優先使用） ==========
    # 枚舉
    "AssetType",
    "ScanStatus",
    "Severity",
    # Schema - 基礎模型
    "Asset",
    "Fingerprints",
    "Summary",
    # Schema - 掃描任務
    "ScanStartPayload", 
    "ScanCompletedPayload",
    # Schema - 漏洞
    "Vulnerability",
    # Schema - 標準引用
    "CVEReference",
    "CVSSv3Metrics",
    "CWEReference",
    
    # ========== 協調器 ==========
    "MultiEngineCoordinator",
    
    # ========== 協調器特有模型（只有新定義的） ==========
    "ScanCoordinationMetadata",
    "EngineStatus",
    "MultiEngineCoordinationResult",
    
    # ========== 模組架構 ==========
    "engines",         # 四個語言引擎模組
    "coordinators",    # 協調器模組
]


def register_to_command_center() -> None:
    """
    註冊 Scan 模組到 AI 命令中心
    
    這個函數由外部調用（通常是 Core 模組初始化時），
    避免了跨模組的直接 import 依賴。
    
    使用示例:
        # 在 AI Commander 或主程式中
        from services import scan
        scan.register_to_command_center()
    """
    from services.aiva_common.command_center import get_command_center
    from .command_handler import ScanCommandHandler
    
    command_center = get_command_center()
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("✅ Scan 模組已註冊到 AI 命令中心")
