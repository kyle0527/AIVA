"""
AIVA Findings Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義

⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
📅 最後更新: 2025-10-27T08:15:28.157056
🔄 Schema 版本: 1.0.0
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
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

    status: str = Field(choices=['new', 'confirmed', 'false_positive', 'fixed', 'ignored'])
    """發現狀態"""

    vulnerability: Vulnerability
    """漏洞資訊"""

    target: Target
    """目標資訊"""

    strategy: Optional[str] = None
    """使用的策略"""

    evidence: Optional[FindingEvidence] = None
    """證據資料"""

    impact: Optional[FindingImpact] = None
    """影響評估"""

    recommendation: Optional[FindingRecommendation] = None
    """修復建議"""

    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    """中繼資料"""

    created_at: datetime
    """建立時間"""

    updated_at: datetime
    """更新時間"""



class FindingEvidence(BaseModel):
    """漏洞證據"""

    payload: Optional[str] = None
    """攻擊載荷"""

    response_time_delta: Optional[float] = None
    """響應時間差異"""

    db_version: Optional[str] = None
    """資料庫版本"""

    request: Optional[str] = None
    """HTTP請求"""

    response: Optional[str] = None
    """HTTP響應"""

    proof: Optional[str] = None
    """證明資料"""



class FindingImpact(BaseModel):
    """漏洞影響評估"""

    description: Optional[str] = None
    """影響描述"""

    business_impact: Optional[str] = None
    """業務影響"""

    technical_impact: Optional[str] = None
    """技術影響"""

    affected_users: Optional[int] = None
    """受影響用戶數"""

    estimated_cost: Optional[float] = None
    """估計成本"""



class FindingRecommendation(BaseModel):
    """漏洞修復建議"""

    fix: Optional[str] = None
    """修復方法"""

    priority: Optional[str] = None
    """修復優先級"""

    remediation_steps: Optional[List[str]] = Field(default_factory=list)
    """修復步驟"""

    references: Optional[List[str]] = Field(default_factory=list)
    """參考資料"""



