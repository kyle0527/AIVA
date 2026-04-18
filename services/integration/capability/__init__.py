"""
AIVA 能力註冊中心初始化模組
整合所有組件並提供統一的入口點

此模組遵循 aiva_common 規範:
- 統一的錯誤處理和日誌記錄
- 標準化的配置管理
- 完整的生命週期管理
- 豐富的監控和診斷功能
"""

from .function_recon import (
    DNSRecon,
    FunctionReconManager,
    NetworkScanner,
    OSINTRecon,
    ReconCLI,
    ReconStatus,
    ReconTarget,
    ReconTargetType,
    WebRecon,
)
from .lifecycle import InstallationResult, ToolLifecycleEvent, ToolLifecycleManager
from .lifecycle_cli import LifecycleCLI
from .models import (
    CapabilityEvidence,
    CapabilityRecord,
    CapabilityScorecard,
    CapabilityStatus,
    CapabilityType,
    CLITemplate,
    ExecutionRequest,
    ExecutionResult,
    InputParameter,
    OutputParameter,
    create_capability_id,
    create_sample_capability,
    validate_capability_id,
)
from .toolkit import CapabilityToolkit, toolkit

__version__ = "1.0.0"
__author__ = "AIVA Development Team"
__description__ = "AIVA 統一能力註冊與管理系統"

# 匯出主要組件
__all__ = [
    # 核心組件
    "CapabilityToolkit", 
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

