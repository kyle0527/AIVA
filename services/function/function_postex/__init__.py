"""
PostEx Module - 後滲透測試模組 (僅供授權測試使用)

⚠️ 警告: 此模組包含後滲透測試功能,僅應在獲得明確授權的環境中使用。
未經授權的使用可能違反法律。

提供權限提升、橫向移動、數據外洩檢測、持久化檢查等功能。
"""

from .data_exfiltration_tester import DataExfiltrationTester
from .lateral_movement import LateralMovementTester
from .persistence_checker import PersistenceChecker
from .privilege_escalator import PrivilegeEscalator

__all__ = [
    "PrivilegeEscalator",
    "LateralMovementTester",
    "DataExfiltrationTester",
    "PersistenceChecker",
]

__version__ = "1.0.0"
__warning__ = "FOR AUTHORIZED SECURITY TESTING ONLY"
