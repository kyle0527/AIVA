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

__version__ = "2.0.0"
__architecture__ = "worker-based"
__risk_level__ = "L2"
