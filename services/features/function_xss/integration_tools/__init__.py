"""XSS Integration Tools - 整合工具模組

提供與其他工具（Dalfox等）的整合功能
"""

from .xss_tools import (
    BlindXSSDetector,
    DalfoxIntegration,
    DOMXSSDetector,
    StoredXSSDetector,
    XSSManager,
    XSSPayloadGenerator,
    XSSTarget,
    XSSVulnerability,
)

__all__ = [
    "XSSTarget",
    "XSSVulnerability",
    "DalfoxIntegration",
    "XSSPayloadGenerator",
    "DOMXSSDetector",
    "StoredXSSDetector",
    "BlindXSSDetector",
    "XSSManager",
]
