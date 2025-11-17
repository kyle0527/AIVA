"""XSS Integration Tools - 整合工具模組

提供與其他工具（Dalfox等）的整合功能
"""

from .xss_tools import (
    XSSTarget,
    XSSVulnerability,
    DalfoxIntegration,
    ReflectedXSSScanner,
    DOMXSSScanner,
    StoredXSSScanner,
    BlindXSSScanner,
    XSSManager,
)

__all__ = [
    "XSSTarget",
    "XSSVulnerability",
    "DalfoxIntegration",
    "ReflectedXSSScanner",
    "DOMXSSScanner",
    "StoredXSSScanner",
    "BlindXSSScanner",
    "XSSManager",
]
