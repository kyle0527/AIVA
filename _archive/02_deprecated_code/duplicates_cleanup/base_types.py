"""
AIVA 基礎類型 Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MessageHeader(BaseModel):
    """訊息標頭 - 用於所有訊息的統一標頭格式"""

    message_id: str
    """"""

    trace_id: str
    """"""

    correlation_id: str | None = None
    """"""

    source_module: str
    """來源模組名稱"""

    timestamp: datetime = None
    """"""

    version: str = "1.0"
    """"""


class Target(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any
    """"""

    parameter: str | None = None
    """"""

    method: str | None = None
    """"""

    headers: dict[str, Any] = None
    """"""

    params: dict[str, Any] = None
    """"""

    body: str | None = None
    """"""


class Vulnerability(BaseModel):
    """漏洞基本資訊 - 用於 Finding 中的漏洞描述。符合標準：CWE、CVE、CVSS v3.1/v4.0、OWASP"""

    name: Any
    """"""

    cwe: str | None = None
    """CWE ID (格式: CWE-XXX)，參考 https://cwe.mitre.org/"""

    cve: str | None = None
    """CVE ID (格式: CVE-YYYY-NNNNN)，參考 https://cve.mitre.org/"""

    severity: Any
    """"""

    confidence: Any
    """"""

    description: str | None = None
    """"""

    cvss_score: Any = None
    """CVSS v3.1 Base Score (0.0-10.0)，參考 https://www.first.org/cvss/"""

    cvss_vector: str | None = None
    """CVSS v3.1 Vector String，例如: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"""

    owasp_category: str | None = None
    """OWASP Top 10 分類，例如: A03:2021-Injection"""


class Asset(BaseModel):
    """資產基本資訊"""

    asset_id: str
    """"""

    type: str
    """"""

    value: str
    """"""

    parameters: list[str] = None
    """"""

    has_form: bool = False
    """"""


class Authentication(BaseModel):
    """認證資訊"""

    method: str = "none"
    """"""

    credentials: dict[str, Any] = None
    """"""


class ExecutionError(BaseModel):
    """執行錯誤統一格式"""

    error_id: str
    """"""

    error_type: str
    """"""

    message: str
    """"""

    payload: str | None = None
    """"""

    vector: str | None = None
    """"""

    timestamp: datetime = None
    """"""

    attempts: int = 1
    """"""


class Fingerprints(BaseModel):
    """技術指紋"""

    web_server: dict[str, Any] = None
    """"""

    framework: dict[str, Any] = None
    """"""

    language: dict[str, Any] = None
    """"""

    waf_detected: bool = False
    """"""

    waf_vendor: str | None = None
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

    description: str | None = None
    """因子描述"""


class ScanScope(BaseModel):
    """掃描範圍"""

    exclusions: list[str] = None
    """"""

    include_subdomains: bool = True
    """"""

    allowed_hosts: list[str] = None
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

    condition: str | None = None
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

    context: dict[str, Any] = None
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

    test_steps: list[str] = None
    """"""

    observations: list[str] = None
    """"""

    recommendations: list[str] = None
    """"""

    timestamp: datetime = None
    """"""


class CodeLevelRootCause(BaseModel):
    """程式碼層面根因分析結果"""

    analysis_id: str
    """"""

    vulnerable_component: str
    """"""

    affected_findings: list[str]
    """"""

    code_location: str | None = None
    """"""

    vulnerability_pattern: str | None = None
    """"""

    fix_recommendation: str | None = None
    """"""


class FindingEvidence(BaseModel):
    """漏洞證據"""

    payload: str | None = None
    """"""

    response_time_delta: Any = None
    """"""

    db_version: str | None = None
    """"""

    request: str | None = None
    """"""

    response: str | None = None
    """"""

    proof: str | None = None
    """"""


class FindingImpact(BaseModel):
    """漏洞影響描述"""

    description: str | None = None
    """"""

    business_impact: str | None = None
    """"""

    technical_impact: str | None = None
    """"""

    affected_users: Any = None
    """"""

    estimated_cost: Any = None
    """"""


class FindingPayload(BaseModel):
    """漏洞發現 Payload - 統一的漏洞報告格式"""

    finding_id: str
    """"""

    task_id: str
    """"""

    scan_id: str
    """"""

    status: str
    """"""

    vulnerability: Any
    """"""

    target: Any
    """"""

    strategy: str | None = None
    """"""

    evidence: Any = None
    """"""

    impact: Any = None
    """"""

    recommendation: Any = None
    """"""

    metadata: dict[str, Any] = None
    """"""

    created_at: datetime = None
    """"""

    updated_at: datetime = None
    """"""


class FindingRecommendation(BaseModel):
    """漏洞修復建議"""

    fix: str | None = None
    """"""

    priority: str | None = None
    """"""

    remediation_steps: list[str] = None
    """"""

    references: list[str] = None
    """"""


class FindingTarget(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any
    """"""

    parameter: str | None = None
    """"""

    method: str | None = None
    """"""

    headers: dict[str, Any] = None
    """"""

    params: dict[str, Any] = None
    """"""

    body: str | None = None
    """"""


class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""

    analysis_id: str
    """"""

    url: str
    """"""

    source_size_bytes: int
    """"""

    dangerous_functions: list[str] = None
    """"""

    external_resources: list[str] = None
    """"""

    data_leaks: dict[str, Any] = None
    """"""

    findings: list[str] = None
    """"""

    apis_called: list[str] = None
    """"""

    ajax_endpoints: list[str] = None
    """"""

    suspicious_patterns: list[str] = None
    """"""

    risk_score: float = 0.0
    """"""

    security_score: int = 100
    """"""

    timestamp: datetime = None
    """"""


class SASTDASTCorrelation(BaseModel):
    """SAST-DAST 資料流關聯結果"""

    correlation_id: str
    """"""

    sast_finding_id: str
    """"""

    dast_finding_id: str
    """"""

    data_flow_path: list[str]
    """"""

    verification_status: str
    """"""

    confidence_score: float
    """"""

    explanation: str | None = None
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

    file_path: str | None = None
    """"""

    url: str | None = None
    """"""

    severity: Any = "medium"
    """"""


class VulnerabilityCorrelation(BaseModel):
    """漏洞關聯分析結果"""

    correlation_id: str
    """"""

    correlation_type: str
    """"""

    related_findings: list[str]
    """"""

    confidence_score: float
    """"""

    root_cause: str | None = None
    """"""

    common_components: list[str] = None
    """"""

    explanation: str | None = None
    """"""

    timestamp: datetime = None
    """"""
