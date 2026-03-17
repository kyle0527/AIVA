"""
AIVA Web Scanner Module

綜合 Web 掃描模組，提供多種掃描能力:
- Subdomain Enumeration: 子域名枚舉
- Directory Scanning: 目錄掃描
- Vulnerability Detection: 漏洞檢測
- Technology Fingerprinting: 技術指紋識別
- Port Scanning: 端口掃描

架構: 工具集成型 (WebAttackManager)
風險等級: L1 (資訊收集)
模組版本: 1.3.0

使用方式:
    from services.features.function_web_scanner import WebAttackManager
    manager = WebAttackManager()
    result = await manager.comprehensive_scan(
        target="https://example.com",
        options={"subdomain_scan": True, "directory_scan": True}
    )
"""

from services.features.function_web_scanner.integration_tools import (
    WebAttackManager,
    SubdomainEnumerator,
    DirectoryScanner,
    WebVulnerabilityScanner,
)
from services.features.function_web_scanner.scanners import (
    SubdomainScanner,
    DirectoryBruteforcer,
    TechDetector,
    PortScanner,
    WebCrawler,
)

__all__ = [
    "WebAttackManager",
    "SubdomainEnumerator",
    "DirectoryScanner",
    "WebVulnerabilityScanner",
    "SubdomainScanner",
    "DirectoryBruteforcer",
    "TechDetector",
    "PortScanner",
    "WebCrawler",
]

__version__ = "1.3.0"
__status__ = "in_development"
__risk_level__ = "L1"
__last_updated__ = "2026-03-17"

# 架構說明 (2026-03-17)
# ✅ WebAttackManager 為主入口 (integration_tools/web_tools.py)
# ✅ 5 個獨立掃描引擎 (scanners/)
# ❌ scanner_manager.py 已廢棄 (v1.3.0 移除)
