# Python Scanning Engine
# OWASP A05/A08/A02 檢測能力

"""
Python 掃描引擎 - 專注於深度安全檢測

核心功能:
- XXE (XML External Entity) 檢測 (OWASP A05)
- 不安全反序列化檢測 (OWASP A08)
- 被動流量分析 (OWASP A02, A05)
"""

__version__ = "1.0.0"
__all__ = ["XXEDetector", "DeserializationDetector", "PassiveAnalyzer"]

from .xxe_detector import XXEDetector
from .deserialization_detector import DeserializationDetector
from .passive_analyzer import PassiveAnalyzer
