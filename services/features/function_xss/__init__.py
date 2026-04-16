"""
AIVA XSS Detection Module

跨站腳本 (XSS) 檢測模組，支援四種檢測引擎:
- Traditional XSS: 反射型 XSS 檢測
- DOM XSS: DOM-based XSS 檢測  
- Stored XSS: 儲存型 XSS 檢測
- Blind XSS: 盲注 XSS 檢測

架構: Detector-based (直接調用核心檢測器)
風險等級: L2 (需要授權)
模組版本: 3.0.0
"""

from .blind_xss_listener_validator import BlindXssListenerValidator
from .command_handler import XSSCommandHandler
from .dom_xss_detector import DomXssDetector
from .payload_generator import XssPayloadGenerator
from .stored_detector import StoredXssDetector
from .traditional_detector import TraditionalXssDetector

__all__ = [
    "XSSCommandHandler",
    "TraditionalXssDetector",
    "DomXssDetector",
    "StoredXssDetector",
    "BlindXssListenerValidator",
    "XssPayloadGenerator",
]

__version__ = "3.0.0"
__architecture__ = "detector-based"
__risk_level__ = "L2"
__status__ = "production_ready"
__last_updated__ = "2026-02-03"

# 整合狀態 (2026-02-03)
# ✅ CommandHandler 保留中 (XSSCommandHandler - 過渡期)
# ✅ 核心檢測器架構 (直接調用)
# ✅ 4種檢測器已完成
# ✅ AI Commander 透過 CLI 執行器調用
# ⚠️ 逐步遷移至純 CLI 驅動架構
