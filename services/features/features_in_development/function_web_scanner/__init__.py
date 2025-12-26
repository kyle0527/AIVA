"""
AIVA Web Scanner Module

綜合 Web 掃描模組，提供多種掃描能力:
- Subdomain Enumeration: 子域名枚舉
- Directory Scanning: 目錄掃描
- Vulnerability Detection: 漏洞檢測
- Technology Fingerprinting: 技術指紋識別
- Port Scanning: 端口掃描

架構: 工具集成型 + 統一管理器
風險等級: L1 (資訊收集)
模組版本: 1.1.0

使用方式:
    from services.features.function_web_scanner import WebScannerManager
    manager = WebScannerManager()
    result = manager.scan("https://example.com")
"""

from services.features.function_web_scanner.scanner_manager import (
    WebScannerManager,
    scan_target
)

__all__ = ["WebScannerManager", "scan_target"]

__version__ = "1.2.0"
__status__ = "in_development"
__risk_level__ = "L1"
__last_updated__ = "2025-12-17"

# 整合狀態 (2025-12-17)
# ✅ WebScannerManager 架構已完成
# ✅ scan_target 函數已完成
# ⏳ 檢測邏輯需要增強
# ⏳ README.md 文檔待補充
# ⏳ CommandHandler 待開發
# ⏳ AI Commander 整合待開發
