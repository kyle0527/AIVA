"""
API Testing Models

暫時從原api_standards.py導入API測試相關模型
"""

from ..interfaces.api_standards import (
    APISecurityTest,
    APIVulnerabilityFinding,
)

__all__ = [
    "APISecurityTest", 
    "APIVulnerabilityFinding",
]