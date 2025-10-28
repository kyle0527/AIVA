"""
AIVA 基礎類型 Schema - 自動生成 (相容版本)
=====================================

此檔案基於手動維護的 Schema 定義自動生成，確保完全相容

⚠️  此檔案由 core_schema_sot.yaml 自動生成，請勿手動修改
📅 最後更新: 2025-10-28T10:55:40.858296
🔄 Schema 版本: 1.0.0
🎯 相容性: 完全相容手動維護版本
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from pydantic import BaseModel, Field

# 導入枚舉類型以保持相容性
try:
    from ...enums import ModuleName
except ImportError:
    from services.aiva_common.enums import ModuleName



class MessageHeader(BaseModel):
    """訊息標頭 - 用於所有訊息的統一標頭格式"""

    message_id: str
    trace_id: str
    correlation_id: Optional[str] = None
    source_module: ModuleName
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = '1.0'


class Target(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any
    parameter: Optional[str] = None
    method: Optional[str] = None
    headers: Dict[str, Any] | None = None
    params: Dict[str, Any] | None = None
    body: Optional[str] = None


class Vulnerability(BaseModel):
    """漏洞基本資訊 - 用於 Finding 中的漏洞描述

符合標準：
- CWE: Common Weakness Enumeration (MITRE)
- CVE: Common Vulnerabilities and Exposures
- CVSS: Common Vulnerability Scoring System v3.1/v4.0
- OWASP: Open Web Application Security Project"""

    name: Any
    cwe: Optional[str] = None
    cve: Optional[str] = None
    severity: Any
    confidence: Any
    description: Optional[str] = None
    cvss_score: Any | None = None
    cvss_vector: Optional[str] = None
    owasp_category: Optional[str] = None


class Asset(BaseModel):
    """資產基本資訊"""

    asset_id: str
    type: str
    value: str
    parameters: List[str] | None = None
    has_form: bool | None = None


class Authentication(BaseModel):
    """認證資訊"""

    method: str = 'none'
    credentials: Dict[str, Any] | None = None


class ExecutionError(BaseModel):
    """執行錯誤統一格式"""

    error_id: str
    error_type: str
    message: str
    payload: Optional[str] = None
    vector: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempts: int = 1


class Fingerprints(BaseModel):
    """技術指紋"""

    web_server: Dict[str, Any] | None = None
    framework: Dict[str, Any] | None = None
    language: Dict[str, Any] | None = None
    waf_detected: bool | None = None
    waf_vendor: Optional[str] = None


class RateLimit(BaseModel):
    """速率限制"""

    requests_per_second: int = 25
    burst: int = 50


class RiskFactor(BaseModel):
    """風險因子"""

    factor_name: str
    weight: float
    value: float
    description: Optional[str] = None


class ScanScope(BaseModel):
    """掃描範圍"""

    exclusions: List[str] | None = None
    include_subdomains: bool = True
    allowed_hosts: List[str] | None = None


class Summary(BaseModel):
    """掃描摘要"""

    urls_found: int | None = None
    forms_found: int | None = None
    apis_found: int | None = None
    scan_duration_seconds: int | None = None


class TaskDependency(BaseModel):
    """任務依賴"""

    dependency_type: str
    dependent_task_id: str
    condition: Optional[str] = None
    required: bool = True


class AIVerificationRequest(BaseModel):
    """AI 驅動漏洞驗證請求"""

    verification_id: str
    finding_id: str
    scan_id: str
    vulnerability_type: Any
    target: Any
    evidence: Any
    verification_mode: str = 'non_destructive'
    context: Dict[str, Any] | None = None


class AIVerificationResult(BaseModel):
    """AI 驅動漏洞驗證結果"""

    verification_id: str
    finding_id: str
    verification_status: str
    confidence_score: float
    verification_method: str
    test_steps: List[str] | None = None
    observations: List[str] | None = None
    recommendations: List[str] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CodeLevelRootCause(BaseModel):
    """程式碼層面根因分析結果"""

    analysis_id: str
    vulnerable_component: str
    affected_findings: List[str]
    code_location: Optional[str] = None
    vulnerability_pattern: Optional[str] = None
    fix_recommendation: Optional[str] = None


class FindingEvidence(BaseModel):
    """漏洞證據"""

    payload: Optional[str] = None
    response_time_delta: Any | None = None
    db_version: Optional[str] = None
    request: Optional[str] = None
    response: Optional[str] = None
    proof: Optional[str] = None


class FindingImpact(BaseModel):
    """漏洞影響描述"""

    description: Optional[str] = None
    business_impact: Optional[str] = None
    technical_impact: Optional[str] = None
    affected_users: Any | None = None
    estimated_cost: Any | None = None


class FindingPayload(BaseModel):
    """漏洞發現 Payload - 統一的漏洞報告格式"""

    finding_id: str
    task_id: str
    scan_id: str
    status: str
    vulnerability: Any
    target: Any
    strategy: Optional[str] = None
    evidence: Any | None = None
    impact: Any | None = None
    recommendation: Any | None = None
    metadata: Dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FindingRecommendation(BaseModel):
    """漏洞修復建議"""

    fix: Optional[str] = None
    priority: Optional[str] = None
    remediation_steps: List[str] | None = None
    references: List[str] | None = None


class FindingTarget(BaseModel):
    """目標資訊 - 漏洞所在位置"""

    url: Any
    parameter: Optional[str] = None
    method: Optional[str] = None
    headers: Dict[str, Any] | None = None
    params: Dict[str, Any] | None = None
    body: Optional[str] = None


class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""

    analysis_id: str
    url: str
    source_size_bytes: int
    dangerous_functions: List[str] | None = None
    external_resources: List[str] | None = None
    data_leaks: Dict[str, Any] | None = None
    findings: List[str] | None = None
    apis_called: List[str] | None = None
    ajax_endpoints: List[str] | None = None
    suspicious_patterns: List[str] | None = None
    risk_score: float | None = None
    security_score: int = 100
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SASTDASTCorrelation(BaseModel):
    """SAST-DAST 資料流關聯結果"""

    correlation_id: str
    sast_finding_id: str
    dast_finding_id: str
    data_flow_path: List[str]
    verification_status: str
    confidence_score: float
    explanation: Optional[str] = None


class SensitiveMatch(BaseModel):
    """敏感資訊匹配結果"""

    match_id: str
    pattern_name: str
    matched_text: str
    context: str
    confidence: float
    line_number: Any | None = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    severity: Any = 'medium'


class VulnerabilityCorrelation(BaseModel):
    """漏洞關聯分析結果"""

    correlation_id: str
    correlation_type: str
    related_findings: List[str]
    confidence_score: float
    root_cause: Optional[str] = None
    common_components: List[str] | None = None
    explanation: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
