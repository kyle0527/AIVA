"""
AIVA XSS Detection Module

跨站腳本 (XSS) 檢測模組，支援四種檢測引擎:
- Traditional XSS: 反射型 XSS 檢測
- DOM XSS: DOM-based XSS 檢測  
- Stored XSS: 儲存型 XSS 檢測
- Blind XSS: 盲注 XSS 檢測

架構: Worker-based (高並發)
風險等級: L2 (需要授權)
模組版本: 2.0.0
"""

from .command_handler import XSSCommandHandler
from .worker import XssWorkerService
from .traditional_detector import TraditionalXssDetector
from .dom_xss_detector import DomXssDetector
from .stored_detector import StoredXssDetector
from .blind_xss_listener_validator import BlindXssListenerValidator

__all__ = [
    "XSSCommandHandler",
    "XssWorkerService",
    "TraditionalXssDetector",
    "DomXssDetector",
    "StoredXssDetector",
    "BlindXssListenerValidator",
]

__version__ = "2.1.0"
__architecture__ = "worker-based"
__risk_level__ = "L2"
__status__ = "ready_for_integration"
__last_updated__ = "2025-12-17"

# 整合狀態 (2025-12-17)
# ✅ CommandHandler 已完成 (XSSCommandHandler)
# ✅ Worker 架構已完成 (XssWorkerService)
# ✅ 4種檢測器已完成
# ⏳ AI Commander 整合待開發
