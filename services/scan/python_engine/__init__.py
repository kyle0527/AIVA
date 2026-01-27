# Python Scanning Engine - OWASP 檢測引擎

"""
Python 掃描引擎 - 專注於深度安全檢測

✅ **核心檢測器**:
- XXEDetector: XML External Entity 注入檢測 (OWASP A05:2021)
- DeserializationDetector: 不安全反序列化檢測 (OWASP A08:2021)  
- PassiveAnalyzer: 被動流量分析 (OWASP A02/A05:2021)

✅ **特點**:
- 無外部依賴（除 requests, lxml）
- 完整的 dataclass 結果結構
- OWASP 標準映射
- 詳細證據收集

使用示例:
    from services.scan.python_engine import XXEDetector
    
    detector = XXEDetector(callback_server="http://callback.com")
    findings = detector.test_xxe("http://target.com/api", "xml_param")
    
    for finding in findings:
        print(f"[{finding.severity}] {finding.type}: {finding.evidence}")
"""

__version__ = "1.0.0"
__all__ = ["XXEDetector", "DeserializationDetector", "PassiveAnalyzer"]

from .xxe_detector import XXEDetector, XXEFinding
from .deserialization_detector_v2 import DeserializationDetector, DeserializationFinding
from .passive_analyzer import PassiveAnalyzer, PassiveFinding
