"""
Post-Exploitation Module

後滲透測試模組

⚠️ 警告: 僅用於授權測試
架構: Detector + 多引擎
模組版本: 2.0.0

使用方式:
    from services.features.function_postex import PostExDetector
    detector = PostExDetector()
    findings = detector.analyze(
        test_type="privilege_escalation",
        target="localhost",
        task_id="task-1",
        scan_id="scan-1",
        safe_mode=True,
    )
"""

from services.features.function_postex.detector.postex_detector import (
    PostExDetector,
)
from services.features.function_postex.engines import (
    LateralMovementEngine,
    PersistenceEngine,
    PrivilegeEscalationEngine,
)

__all__ = [
    "PostExDetector",
    "PrivilegeEscalationEngine",
    "LateralMovementEngine",
    "PersistenceEngine",
]

__version__ = "2.0.0"
__status__ = "in_development"
__architecture__ = "detector-based"
__last_updated__ = "2026-03-17"

# 架構說明 (2026-03-17)
# ✅ Detector-based 架構（PostExDetector 為主入口）
# ✅ 三引擎: PrivilegeEscalation / LateralMovement / Persistence
# ✅ detectors/ 路徑已統一至 detector/（重新匯出以維持相容）
# ✅ engines/__init__.py 匯出兩組引擎（結構化 + 整合）
# ✅ Windows 權限提升與持久化已實作
# ❌ postex_manager.py 已廢棄 (v1.3.0 移除)
