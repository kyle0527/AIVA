"""
AIVA 能力註冊中心初始化模組
整合所有組件並提供統一的入口點

此模組遵循 aiva_common 規範:
- 統一的錯誤處理和日誌記錄
- 標準化的配置管理
- 完整的生命週期管理
- 豐富的監控和診斷功能
"""

from .registry import CapabilityRegistry, registry
from .models import (
    CapabilityRecord,
    CapabilityEvidence, 
    CapabilityScorecard,
    CapabilityType,
    CapabilityStatus,
    CLITemplate,
    ExecutionRequest,
    ExecutionResult,
    InputParameter,
    OutputParameter,
    create_sample_capability,
    validate_capability_id,
    create_capability_id
)
from .toolkit import CapabilityToolkit, toolkit
from .lifecycle import ToolLifecycleManager, ToolLifecycleEvent, InstallationResult
from .lifecycle_cli import LifecycleCLI
from .function_recon import (
    FunctionReconManager,
    NetworkScanner,
    DNSRecon,
    WebRecon,
    OSINTRecon,
    ReconCLI,
    ReconTarget,
    ReconTargetType,
    ReconStatus,
    register_recon_capabilities
)

__version__ = "1.0.0"
__author__ = "AIVA Development Team"
__description__ = "AIVA 統一能力註冊與管理系統"

# 匯出主要組件
__all__ = [
    # 核心組件
    "CapabilityRegistry",
    "CapabilityToolkit", 
    "registry",
    "toolkit",
    "app",
    
    # 生命週期管理
    "ToolLifecycleManager",
    "ToolLifecycleEvent",
    "InstallationResult",
    "LifecycleCLI",
    
    # 功能偵察模組
    "FunctionReconManager",
    "NetworkScanner", 
    "DNSRecon",
    "WebRecon",
    "OSINTRecon",
    "ReconCLI",
    "ReconTarget",
    "ReconTargetType",
    "ReconStatus",
    "register_recon_capabilities",
    
    # 資料模型
    "CapabilityRecord",
    "CapabilityEvidence",
    "CapabilityScorecard", 
    "CLITemplate",
    "ExecutionRequest",
    "ExecutionResult",
    "InputParameter",
    "OutputParameter",
    
    # 列舉類型
    "CapabilityType",
    "CapabilityStatus",
    
    # 工具函數
    "create_sample_capability",
    "validate_capability_id", 
    "create_capability_id",
    
    # 版本資訊
    "__version__",
    "__author__",
    "__description__"
]


def get_version() -> str:
    """獲取能力註冊中心版本"""
    return __version__


def get_info() -> dict:
    """獲取能力註冊中心資訊"""
    return {
        "name": "AIVA Capability Registry",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components": {
            "registry": "能力註冊與發現服務",
            "toolkit": "能力管理工具集",
            "models": "統一資料模型",
            "cli": "命令行管理介面"
        },
        "features": [
            "自動能力發現和註冊",
            "跨語言支援 (Python, Go, Rust, TypeScript)",
            "即時健康監控",
            "智能依賴管理",
            "豐富的API和CLI工具",
            "完整的文件生成",
            "性能分析和報告"
        ]
    }


# 快速啟動函數
async def quick_start():
    """快速啟動能力註冊中心"""
    from aiva_common.utils.logging import get_logger
    
    logger = get_logger(__name__)
    
    logger.info("🚀 AIVA 能力註冊中心快速啟動")
    
    # 執行自動發現
    logger.info("🔍 開始自動發現能力...")
    discovery_stats = await registry.discover_capabilities()
    
    logger.info(
        "發現完成",
        discovered_count=discovery_stats["discovered_count"],
        languages=discovery_stats["languages"]
    )
    
    # 顯示統計資訊  
    stats = await registry.get_capability_stats()
    logger.info(
        "系統統計",
        total_capabilities=stats["total_capabilities"],
        by_language=stats["by_language"],
        health_summary=stats["health_summary"]
    )
    
    logger.info("✅ 能力註冊中心已就緒")
    
    return {
        "status": "ready",
        "discovery_stats": discovery_stats,
        "system_stats": stats
    }