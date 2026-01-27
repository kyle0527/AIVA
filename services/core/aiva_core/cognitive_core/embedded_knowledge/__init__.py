"""
AIVA 內嵌安全知識模組 (Embedded Security Knowledge)

此套件提供了 AIVA 攻擊決策系統所需的內建專業知識，
無需 RAG 搜索即可直接調用進行漏洞判斷和策略決策。

模組結構:
    - base.py: 基礎數據類型和通用結構
    - vulnerability_detection.py: 漏洞檢測判斷邏輯 (SQLi/XSS/SSRF/IDOR)
    - cve_identification.py: 高危 CVE 識別模組
    - waf_bypass.py: WAF 繞過技術字典
    - web_architecture.py: 現代架構漏洞檢測 (GraphQL/JWT/WebSocket/REST)

使用方式:
    from services.core.aiva_core.cognitive_core.embedded_knowledge import (
        VulnerabilityDetector,
        CVEIdentifier,
        WAFBypassEngine,
        WebArchitectureAnalyzer,
    )
    
    # 判斷 SQL 注入是否成功
    result = VulnerabilityDetector.check_sqli(response_text, response_time)
    
    # 識別目標是否存在已知高危 CVE
    cve_matches = CVEIdentifier.identify(target_fingerprint)
    
    # 獲取針對特定 WAF 的繞過技術
    bypass_payloads = WAFBypassEngine.get_bypass_payloads("cloudflare", "sqli")

架構原則:
    1. 零搜索延遲: 所有知識直接編碼為 Python 類和方法
    2. 結構化輸出: 所有方法返回 dataclass 或 TypedDict，便於 AI 推理
    3. 可擴展設計: 預留 register_* 方法支援動態添加新知識
    4. 置信度評分: 每個判斷結果都包含置信度分數 (0.0-1.0)

Version: 1.0.0 (2026-01-19)
"""

from .base import (
    # 基礎數據類型
    DetectionResult,
    ConfidenceLevel,
    VulnerabilityType,
    AttackContext,
    ResponseAnalysis,
    # 通用工具
    KnowledgeRegistry,
)

from .vulnerability_detection import VulnerabilityDetector

from .cve_identification import CVEIdentifier

from .waf_bypass import WAFBypassEngine

from .web_architecture import WebArchitectureAnalyzer

__all__ = [
    # 核心檢測器
    "VulnerabilityDetector",
    "CVEIdentifier", 
    "WAFBypassEngine",
    "WebArchitectureAnalyzer",
    # 數據類型
    "DetectionResult",
    "ConfidenceLevel",
    "VulnerabilityType",
    "AttackContext",
    "ResponseAnalysis",
    # 擴展接口
    "KnowledgeRegistry",
]

__version__ = "1.0.0"
