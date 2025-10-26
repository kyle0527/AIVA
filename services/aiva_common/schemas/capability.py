"""
AIVA 能力管理相關 Schema

此模組定義能力管理系統中使用的統一數據模型。
包含能力定義、評分卡、執行記錄等核心結構。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

from ..enums import ProgrammingLanguage, TaskStatus


class InputParameter(BaseModel):
    """輸入參數定義"""
    
    name: str = Field(..., description="參數名稱")
    type: str = Field(..., description="參數類型")
    required: bool = Field(default=True, description="是否必需")
    description: str = Field(..., description="參數描述")
    default: Optional[Any] = Field(None, description="默認值")


class OutputParameter(BaseModel):
    """輸出參數定義"""
    
    name: str = Field(..., description="輸出名稱")
    type: str = Field(..., description="輸出類型")
    description: str = Field(..., description="輸出描述")


class CapabilityInfo(BaseModel):
    """能力信息"""
    
    id: str = Field(..., description="能力唯一標識符")
    name: str = Field(..., description="能力顯示名稱")
    description: Optional[str] = Field(None, description="能力詳細描述")
    version: str = Field(default="1.0.0", description="能力版本")
    
    # 技術信息
    language: ProgrammingLanguage = Field(..., description="實現語言")
    entrypoint: str = Field(..., description="入口點路徑")
    topic: str = Field(..., description="主題分類")
    
    # 接口定義
    inputs: Optional[List[InputParameter]] = Field(None, description="輸入參數列表")
    outputs: Optional[List[OutputParameter]] = Field(None, description="輸出參數列表")
    
    # 依賴與前置條件
    prerequisites: Optional[List[str]] = Field(None, description="前置條件列表")
    dependencies: Optional[List[str]] = Field(None, description="依賴的其他能力ID")
    
    # 元數據
    tags: Optional[List[str]] = Field(None, description="標籤列表")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="能力狀態")
    
    # 時間戳
    created_at: Optional[datetime] = Field(None, description="創建時間")
    updated_at: Optional[datetime] = Field(None, description="更新時間")


class CapabilityScorecard(BaseModel):
    """能力評分卡"""
    
    capability_id: str = Field(..., description="能力ID")
    
    # 7日性能指標
    success_rate_7d: float = Field(default=0.0, description="7日成功率", ge=0, le=1)
    avg_latency_ms: float = Field(default=0.0, description="平均延遲(毫秒)", ge=0)
    availability_7d: float = Field(default=1.0, description="7日可用性", ge=0, le=1)
    usage_count_7d: int = Field(default=0, description="7日使用次數", ge=0)
    
    # 時間戳
    last_used_at: Optional[datetime] = Field(None, description="最後使用時間")
    last_updated_at: Optional[datetime] = Field(None, description="最後更新時間")
    
    # 額外指標
    error_count_7d: int = Field(default=0, description="7日錯誤次數", ge=0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="其他元數據")