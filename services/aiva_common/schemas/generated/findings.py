"""
AIVA 漏洞發現 Schema - 自動生成 (相容版本)
====================================

此檔案基於手動維護的 Schema 定義自動生成，確保完全相容

⚠️  此檔案由 core_schema_sot.yaml 自動生成，請勿手動修改
📅 最後更新: 2025-10-28T10:55:40.861473
🔄 Schema 版本: 1.0.0
🎯 相容性: 完全相容手動維護版本
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from pydantic import BaseModel, Field

# 導入基礎類型以保持相容性
try:
    from .base_types import Target, Vulnerability
except ImportError:
from services.aiva_common.schemas.base import Target, Vulnerability



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
