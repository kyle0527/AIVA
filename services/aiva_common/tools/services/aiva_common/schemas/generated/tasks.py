"""
AIVA Tasks Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義

⚠️  此檔案由core_schema_sot.yaml自動生成，請勿手動修改
📅 最後更新: 2025-10-23T00:00:00Z
🔄 Schema 版本: 1.0.0
"""


from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field




class FunctionTaskPayload(BaseModel):
    """功能任務載荷 - 掃描任務的標準格式"""

    task_id: str
    """任務識別碼"""

    scan_id: str
    """掃描識別碼"""

    priority: int = Field(ge=0, le=10)
    """任務優先級"""

    target: FunctionTaskTarget
    """掃描目標"""

    context: FunctionTaskContext
    """任務上下文"""

    strategy: str = Field(values=['fast', 'deep', 'aggressive', 'stealth'])
    """掃描策略"""

    custom_payloads: List[str] = Field(default_factory=list)
    """自訂載荷"""

    test_config: FunctionTaskTestConfig
    """測試配置"""


class FunctionTaskTarget(BaseModel):
    """功能任務目標"""

    # 繼承自: Target

    parameter_location: str = Field(values=['url', 'query', 'form', 'json', 'header', 'cookie'])
    """參數位置"""

    cookies: Dict[str, str] = Field(default_factory=dict)
    """Cookie資料"""

    form_data: Dict[str, Any] = Field(default_factory=dict)
    """表單資料"""

    json_data: Optional[Dict[str, Any]] = None
    """JSON資料"""


class FunctionTaskContext(BaseModel):
    """功能任務上下文"""

    db_type_hint: Optional[str] = Field(values=['mysql', 'postgresql', 'mssql', 'oracle', 'sqlite', 'mongodb'], default=None)
    """資料庫類型提示"""

    waf_detected: bool = Field(default=False)
    """是否檢測到WAF"""

    related_findings: List[str] = Field(default_factory=list)
    """相關發現"""


class FunctionTaskTestConfig(BaseModel):
    """功能任務測試配置"""

    payloads: List[str]
    """標準載荷列表"""

    custom_payloads: List[str] = Field(default_factory=list)
    """自訂載荷列表"""

    blind_xss: bool = Field(default=False)
    """是否進行Blind XSS測試"""

    dom_testing: bool = Field(default=False)
    """是否進行DOM測試"""

    timeout: Optional[float] = Field(ge=0.1, le=60.0, default=None)
    """請求逾時(秒)"""


class ScanTaskPayload(BaseModel):
    """掃描任務載荷 - 用於SCA/SAST等需要項目URL的掃描任務"""

    task_id: str
    """任務識別碼"""

    scan_id: str
    """掃描識別碼"""

    priority: int = Field(ge=0, le=10)
    """任務優先級"""

    target: Target
    """掃描目標 (包含URL)"""

    scan_type: str = Field(values=['sca', 'sast', 'secret', 'license', 'dependency'])
    """掃描類型"""

    repository_info: Optional[Dict[str, Any]] = None
    """代碼倉庫資訊 (分支、commit等)"""

    timeout: Optional[int] = Field(ge=60, le=3600, default=None)
    """掃描逾時(秒)"""

