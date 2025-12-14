# -*- coding: utf-8 -*-
"""
API Response Schemas

提供標準化的 API 回應格式
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """標準 API 回應格式"""
    
    success: bool = Field(..., description="請求是否成功")
    message: str = Field(..., description="回應訊息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="回應資料")
    trace_id: Optional[str] = Field(default=None, description="追蹤 ID")
    errors: Optional[List[str]] = Field(default=None, description="錯誤列表")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元資料")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"result": "example_data"},
                "trace_id": "abc123",
                "errors": None,
                "metadata": {"timestamp": "2025-12-10T00:00:00Z"}
            }
        }
