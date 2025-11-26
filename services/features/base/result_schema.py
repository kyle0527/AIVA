"""
功能模組結果數據結構

定義功能模組執行結果的標準格式
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class FindingSeverity(str, Enum):
    """發現的嚴重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingConfidence(str, Enum):
    """發現的置信度"""
    CONFIRMED = "confirmed"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Finding(BaseModel):
    """
    發現的漏洞或問題
    
    Attributes:
        finding_id: 發現的唯一標識符
        vulnerability_type: 漏洞類型（如 "sqli", "xss"）
        severity: 嚴重程度
        confidence: 置信度
        title: 標題
        description: 詳細描述
        affected_url: 受影響的 URL
        affected_parameter: 受影響的參數
        payload: 使用的 payload
        evidence: 證據（如響應內容）
        remediation: 修復建議
        references: 參考鏈接
        timestamp: 發現時間
        metadata: 額外元數據
    """
    finding_id: str = Field(..., description="發現的唯一標識符")
    vulnerability_type: str = Field(..., description="漏洞類型")
    severity: FindingSeverity = Field(..., description="嚴重程度")
    confidence: FindingConfidence = Field(..., description="置信度")
    
    title: str = Field(..., description="標題")
    description: str = Field(..., description="詳細描述")
    
    affected_url: str = Field(..., description="受影響的 URL")
    affected_parameter: Optional[str] = Field(None, description="受影響的參數")
    
    payload: Optional[str] = Field(None, description="使用的 payload")
    evidence: Optional[Dict[str, Any]] = Field(None, description="證據")
    
    remediation: Optional[str] = Field(None, description="修復建議")
    references: Optional[List[str]] = Field(None, description="參考鏈接")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="發現時間")
    metadata: Optional[Dict[str, Any]] = Field(None, description="額外元數據")


class FeatureExecutionStatus(str, Enum):
    """功能執行狀態"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class FeatureResult(BaseModel):
    """
    功能模組執行結果
    
    Attributes:
        feature_name: 功能模組名稱
        task_id: 任務 ID
        status: 執行狀態
        execution_time: 執行時間（秒）
        findings: 發現的漏洞列表
        statistics: 統計信息
        error_message: 錯誤訊息（如果有）
        metadata: 額外元數據
    """
    feature_name: str = Field(..., description="功能模組名稱")
    task_id: str = Field(..., description="任務 ID")
    
    status: FeatureExecutionStatus = Field(..., description="執行狀態")
    execution_time: float = Field(..., description="執行時間（秒）")
    
    findings: List[Finding] = Field(default_factory=list, description="發現的漏洞列表")
    
    statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="統計信息（如請求數、payload 數等）"
    )
    
    error_message: Optional[str] = Field(None, description="錯誤訊息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="額外元數據")
    
    @property
    def has_findings(self) -> bool:
        """是否有發現"""
        return len(self.findings) > 0
    
    @property
    def critical_findings_count(self) -> int:
        """嚴重漏洞數量"""
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)
    
    @property
    def high_findings_count(self) -> int:
        """高危漏洞數量"""
        return sum(1 for f in self.findings if f.severity == FindingSeverity.HIGH)
    
    def get_findings_by_severity(self, severity: FindingSeverity) -> List[Finding]:
        """
        根據嚴重程度獲取發現
        
        Args:
            severity: 嚴重程度
            
        Returns:
            符合條件的發現列表
        """
        return [f for f in self.findings if f.severity == severity]
    
    def get_findings_by_confidence(self, confidence: FindingConfidence) -> List[Finding]:
        """
        根據置信度獲取發現
        
        Args:
            confidence: 置信度
            
        Returns:
            符合條件的發現列表
        """
        return [f for f in self.findings if f.confidence == confidence]
