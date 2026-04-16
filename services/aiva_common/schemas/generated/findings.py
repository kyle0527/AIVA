"""
AIVA Findings Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .base_types import *


class FindingPayload(BaseModel):
    """漏洞發現載荷 - 掃描結果的標準格式"""

    finding_id: str
    """發現識別碼"""

    task_id: str
    """任務識別碼"""

    scan_id: str
    """掃描識別碼"""

    status: Literal['new', 'confirmed', 'false_positive', 'fixed', 'ignored'] = Field(default='new')
    """發現狀態"""

    vulnerability: Vulnerability
    """漏洞資訊"""

    target: Target
    """目標資訊"""

    strategy: str | None = None
    """使用的策略"""

    evidence: FindingEvidence | None = None
    """證據資料"""

    impact: FindingImpact | None = None
    """影響評估"""

    recommendation: FindingRecommendation | None = None
    """修復建議"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """中繼資料"""

    created_at: datetime
    """建立時間"""

    updated_at: datetime
    """更新時間"""


class FindingEvidence(BaseModel):
    """漏洞證據"""

    payload: str | None = None
    """攻擊載荷"""

    response_time_delta: float | None = None
    """響應時間差異"""

    db_version: str | None = None
    """資料庫版本"""

    request: str | None = None
    """HTTP請求"""

    response: str | None = None
    """HTTP響應"""

    proof: str | None = None
    """證明資料"""


class FindingImpact(BaseModel):
    """漏洞影響評估"""

    description: str | None = None
    """影響描述"""

    business_impact: str | None = None
    """業務影響"""

    technical_impact: str | None = None
    """技術影響"""

    affected_users: int | None = Field(ge=0, default=None)
    """受影響用戶數"""

    estimated_cost: float | None = Field(ge=0.0, default=None)
    """估計成本"""


class FindingRecommendation(BaseModel):
    """漏洞修復建議"""

    fix: str | None = None
    """修復方法"""

    priority: Literal['critical', 'high', 'medium', 'low'] | None = Field(default=None)
    """修復優先級"""

    remediation_steps: list[str] = Field(default_factory=list)
    """修復步驟"""

    references: list[str] = Field(default_factory=list)
    """參考資料"""


class TokenTestResult(BaseModel):
    """Token 測試結果"""

    vulnerable: bool
    """是否存在漏洞"""

    token_type: Literal['jwt', 'session', 'api', 'bearer', 'oauth', 'custom'] = Field(default='custom')
    """Token 類型 (jwt, session, api, etc.)"""

    issue: str
    """發現的問題"""

    details: str
    """詳細描述"""

    decoded_payload: dict[str, Any] | None = None
    """解碼後的載荷內容"""

    severity: Literal['HIGH', 'MEDIUM', 'LOW', 'INFO'] = Field(default="MEDIUM")
    """漏洞嚴重程度"""

    test_type: Literal['weak_token', 'expired_token', 'malformed_token', 'privilege_escalation', 'session_fixation'] = Field(default='weak_token')
    """測試類型"""

