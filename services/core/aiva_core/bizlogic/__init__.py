"""
BizLogic Module - 業務邏輯漏洞測試模組

測試應用程式的業務邏輯缺陷,這類漏洞無法被傳統掃描器發現。

測試類型:
- 價格操縱 (Price Manipulation)
- 工作流程繞過 (Workflow Bypass)
- 優惠券/折扣濫用 (Coupon Abuse)
- 競爭條件 (Race Condition)
- 投票/評分系統操縱 (Voting Manipulation)
"""

__version__ = "1.0.0"

__all__ = [
    "PriceManipulationTester",
    "WorkflowBypassTester",
    "RaceConditionTester",
]
