"""
function_postex.engines
後滲透測試引擎模組

提供兩組引擎:
- 結構化引擎 (privilege_escalation/lateral_movement/persistence): 回傳 dataclass vector
- 整合引擎 (privilege_engine/lateral_engine/persistence_engine): 回傳 dict，與 AIVA FindingPayload 整合
"""

__all__ = [
    # 結構化引擎 + 資料類型
    "PrivilegeEscalationEngine",
    "LateralMovementEngine",
    "PersistenceEngine",
    "PrivEscVector",
    "LateralMovementVector",
    "PersistenceVector",
    # 整合引擎 (detector/postex_detector.py 使用)
    "PrivilegeEscalationTester",
    "LateralMovementTester",
    "PersistenceChecker",
]

from .lateral_engine import LateralMovementTester
from .lateral_movement import LateralMovementEngine, LateralMovementVector
from .persistence import PersistenceEngine, PersistenceVector
from .persistence_engine import PersistenceChecker
from .privilege_engine import PrivilegeEscalator as PrivilegeEscalationTester
from .privilege_escalation import PrivEscVector, PrivilegeEscalationEngine
