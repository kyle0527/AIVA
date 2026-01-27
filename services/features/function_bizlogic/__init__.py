"""
BizLogic Module - 業務邏輯漏洞檢測模組

檢測應用程式的業務邏輯缺陷，這類漏洞無法被傳統掃描器發現。

檢測能力:
- 價格操控 (Price Manipulation)
- 工作流程繞過 (Workflow Bypass)
- 競態條件 (Race Condition)
- 優惠券/折扣濫用 (Coupon Abuse)
- 業務規則繞過 (Business Rule Bypass)

整合狀態 (2025-12-17):
✅ 同步接口: BizLogicManager.scan_sync()
✅ CommandHandler: BizLogicCommandHandler
⏳ AI Commander: 準備整合
"""

# 導入所有掃描器類別
from .price_manipulation_scanner import PriceManipulationScanner
from .workflow_bypass_scanner import WorkflowBypassScanner
from .race_condition_scanner import RaceConditionScanner

# 導入整合工具
from .integration_tools.bizlogic_tools import BizLogicManager

# 安全導入命令處理器
try:
    from .command_handler import BizLogicCommandHandler
except ImportError:
    BizLogicCommandHandler = None

__version__ = "3.0.0"
__status__ = "production"
__architecture__ = "manager-based"
__last_updated__ = "2026-01-23"

__all__ = [
    "PriceManipulationScanner",
    "WorkflowBypassScanner", 
    "RaceConditionScanner",
    "BizLogicManager",
    "BizLogicCommandHandler",
]
