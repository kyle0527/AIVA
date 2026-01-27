"""
AIVA 基礎類型 Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class MessageHeader(BaseModel):
    """訊息標頭 - V2統一架構增強版，支援分散式追蹤與路由"""

    message_id: str
    """唯一訊息識別碼（UUID格式）"""

    trace_id: str
    """分散式追蹤識別碼（用於跨服務追蹤）"""

    correlation_id: Optional[str] = None
    """關聯識別碼（用於請求響應配對）"""

    source_module: str
    """來源模組名稱（發送者識別）"""

    target_module: Optional[str] = None
    """目標模組名稱（接收者識別）"""

    timestamp: datetime
    """訊息創建時間戳（ISO 8601格式）"""

    version: str = "1.0"
    """"""

    session_id: Optional[str] = None
    """會話識別碼（用於會話相關訊息群組）"""

    user_context: Optional[str] = None
    """使用者上下文（用於權限與審計）"""


class Target(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any
    """"""

    parameter: Optional[str] = None
    """"""

    method: Optional[str] = None
    """"""

    headers: Optional[Dict[str, Any]] = None
    """"""

    params: Optional[Dict[str, Any]] = None
    """"""

    body: Optional[str] = None
    """"""


class Vulnerability(BaseModel):
    """漏洞基本資訊 - 用於 Finding 中的漏洞描述。符合標準：CWE、CVE、CVSS v3.1/v4.0、OWASP"""

    name: Any
    """"""

    cwe: Optional[str] = None
    """CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/"""

    cve: Optional[str] = None
    """CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/"""

    severity: Any
    """"""

    confidence: Any
    """"""

    description: Optional[str] = None
    """"""

    cvss_score: Any = None
    """CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/"""

    cvss_vector: Optional[str] = None
    """CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"""

    owasp_category: Optional[str] = None
    """OWASP Top 10 分類，例如: A03:2021-Injection"""


class Asset(BaseModel):
    """資產基本資訊"""

    asset_id: str
    """"""

    type: str
    """"""

    value: str
    """"""

    parameters: Optional[List[str]] = None
    """"""

    has_form: bool = False
    """"""


class Authentication(BaseModel):
    """認證資訊"""

    method: str = "none"
    """"""

    credentials: Optional[Dict[str, Any]] = None
    """"""


class ExecutionError(BaseModel):
    """執行錯誤統一格式"""

    error_id: str
    """"""

    error_type: str
    """"""

    message: str
    """"""

    payload: Optional[str] = None
    """"""

    vector: Optional[str] = None
    """"""

    timestamp: Optional[datetime] = None
    """"""

    attempts: int = 1
    """"""


class Fingerprints(BaseModel):
    """技術指紋"""

    web_server: Optional[Dict[str, Any]] = None
    """"""

    framework: Optional[Dict[str, Any]] = None
    """"""

    language: Optional[Dict[str, Any]] = None
    """"""

    waf_detected: bool = False
    """"""

    waf_vendor: Optional[str] = None
    """"""


class RateLimit(BaseModel):
    """速率限制"""

    requests_per_second: int = 25
    """"""

    burst: int = 50
    """"""


class RiskFactor(BaseModel):
    """風險因子"""

    factor_name: str
    """風險因子名稱"""

    weight: float
    """權重"""

    value: float
    """因子值"""

    description: Optional[str] = None
    """因子描述"""


class ScanScope(BaseModel):
    """掃描範圍"""

    exclusions: Optional[List[str]] = None
    """"""

    include_subdomains: bool = True
    """"""

    allowed_hosts: Optional[List[str]] = None
    """"""


class Summary(BaseModel):
    """掃描摘要"""

    urls_found: int = 0
    """"""

    forms_found: int = 0
    """"""

    apis_found: int = 0
    """"""

    scan_duration_seconds: int = 0
    """"""


class TaskDependency(BaseModel):
    """任務依賴"""

    dependency_type: str
    """依賴類型"""

    dependent_task_id: str
    """依賴任務ID"""

    condition: Optional[str] = None
    """依賴條件"""

    required: bool = True
    """是否必需"""


class AIVerificationRequest(BaseModel):
    """AI 驅動漏洞驗證請求"""

    verification_id: str
    """"""

    finding_id: str
    """"""

    scan_id: str
    """"""

    vulnerability_type: Any
    """"""

    target: Any
    """"""

    evidence: Any
    """"""

    verification_mode: str = "non_destructive"
    """"""

    context: Optional[Dict[str, Any]] = None
    """"""


class AIVerificationResult(BaseModel):
    """AI 驅動漏洞驗證結果"""

    verification_id: str
    """"""

    finding_id: str
    """"""

    verification_status: str
    """"""

    confidence_score: float
    """"""

    verification_method: str
    """"""

    test_steps: Optional[List[str]] = None
    """"""

    observations: Optional[List[str]] = None
    """"""

    recommendations: Optional[List[str]] = None
    """"""

    timestamp: Optional[datetime] = None
    """"""


class CodeLevelRootCause(BaseModel):
    """程式碼層面根因分析結果"""

    analysis_id: str
    """"""

    vulnerable_component: str
    """"""

    affected_findings: List[str]
    """"""

    code_location: Optional[str] = None
    """"""

    vulnerability_pattern: Optional[str] = None
    """"""

    fix_recommendation: Optional[str] = None
    """"""


class FindingTarget(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any
    """"""

    parameter: Optional[str] = None
    """"""

    method: Optional[str] = None
    """"""

    headers: Optional[Dict[str, Any]] = None
    """"""

    params: Optional[Dict[str, Any]] = None
    """"""

    body: Optional[str] = None
    """"""


class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""

    analysis_id: str
    """"""

    url: str
    """"""

    source_size_bytes: int
    """"""

    dangerous_functions: Optional[List[str]] = None
    """"""

    external_resources: Optional[List[str]] = None
    """"""

    data_leaks: Optional[Dict[str, Any]] = None
    """"""

    findings: Optional[List[str]] = None
    """"""

    apis_called: Optional[List[str]] = None
    """"""

    ajax_endpoints: Optional[List[str]] = None
    """"""

    suspicious_patterns: Optional[List[str]] = None
    """"""

    risk_score: float = 0.0
    """"""

    security_score: int = 100
    """"""

    timestamp: Optional[datetime] = None
    """"""


class SASTDASTCorrelation(BaseModel):
    """SAST-DAST 資料流關聯結果"""

    correlation_id: str
    """"""

    sast_finding_id: str
    """"""

    dast_finding_id: str
    """"""

    data_flow_path: List[str]
    """"""

    verification_status: str
    """"""

    confidence_score: float
    """"""

    explanation: Optional[str] = None
    """"""


class SensitiveMatch(BaseModel):
    """敏感資訊匹配結果"""

    match_id: str
    """"""

    pattern_name: str
    """"""

    matched_text: str
    """"""

    context: str
    """"""

    confidence: float
    """"""

    line_number: Any = None
    """"""

    file_path: Optional[str] = None
    """"""

    url: Optional[str] = None
    """"""

    severity: Any = "medium"
    """"""


class VulnerabilityCorrelation(BaseModel):
    """漏洞關聯分析結果"""

    correlation_id: str
    """"""

    correlation_type: str
    """"""

    related_findings: List[str]
    """"""

    confidence_score: float
    """"""

    root_cause: Optional[str] = None
    """"""

    common_components: Optional[List[str]] = None
    """"""

    explanation: Optional[str] = None
    """"""

    timestamp: Optional[datetime] = None
    """"""

