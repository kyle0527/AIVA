"""
AIVA Plugins Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""


from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field




class PluginManifest(BaseModel):
    """插件清單"""

    plugin_id: str
    """插件唯一標識符"""

    name: str
    """插件名稱"""

    version: str
    """插件版本"""

    author: str
    """插件作者"""

    description: str
    """插件描述"""

    plugin_type: PluginType
    """插件類型"""

    dependencies: List[str] = Field(default_factory=list)
    """依賴插件列表"""

    permissions: List[str] = Field(default_factory=list)
    """所需權限列表"""

    config_schema: Optional[Dict[str, Any]] = None
    """配置 Schema"""

    min_aiva_version: str
    """最低AIVA版本要求"""

    max_aiva_version: Optional[str] = None
    """最高AIVA版本要求"""

    entry_point: str
    """插件入口點"""

    homepage: Optional[str] = None
    """插件主頁"""

    repository: Optional[str] = None
    """源碼倉庫"""

    license: str = Field(default="MIT")
    """許可證"""

    keywords: List[str] = Field(default_factory=list)
    """關鍵詞"""

    created_at: datetime
    """創建時間"""

    updated_at: datetime
    """更新時間"""


class PluginExecutionContext(BaseModel):
    """插件執行上下文"""

    plugin_id: str
    """插件ID"""

    execution_id: str
    """執行ID"""

    input_data: Dict[str, Any]
    """輸入數據"""

    context: Dict[str, Any] = Field(default_factory=dict)
    """執行上下文"""

    timeout_seconds: int = Field(ge=1, le=600, default=60)
    """執行超時時間(秒)"""

    environment: Dict[str, str] = Field(default_factory=dict)
    """環境變數"""

    working_directory: Optional[str] = None
    """工作目錄"""

    user_id: Optional[str] = None
    """執行用戶ID"""

    session_id: Optional[str] = None
    """會話ID"""

    trace_id: Optional[str] = None
    """追蹤ID"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """元數據"""

    created_at: datetime
    """創建時間"""


class PluginExecutionResult(BaseModel):
    """插件執行結果"""

    execution_id: str
    """執行ID"""

    plugin_id: str
    """插件ID"""

    success: bool
    """執行是否成功"""

    result_data: Optional[Dict[str, Any]] = None
    """結果數據"""

    error_message: Optional[str] = None
    """錯誤信息"""

    error_code: Optional[str] = None
    """錯誤代碼"""

    execution_time_ms: float = Field(ge=0)
    """執行時間(毫秒)"""

    memory_usage_mb: Optional[float] = None
    """內存使用量(MB)"""

    output_logs: List[str] = Field(default_factory=list)
    """輸出日誌"""

    warnings: List[str] = Field(default_factory=list)
    """警告信息"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """結果元數據"""

    created_at: datetime
    """創建時間"""


class PluginConfig(BaseModel):
    """插件配置"""

    plugin_id: str
    """插件ID"""

    enabled: bool = Field(default=True)
    """是否啟用"""

    configuration: Dict[str, Any] = Field(default_factory=dict)
    """配置參數"""

    priority: int = Field(ge=1, le=10, default=5)
    """執行優先級"""

    auto_start: bool = Field(default=False)
    """是否自動啟動"""

    max_instances: int = Field(ge=1, le=10, default=1)
    """最大實例數"""

    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    """資源限制"""

    environment_variables: Dict[str, str] = Field(default_factory=dict)
    """環境變數"""

    created_at: datetime
    """創建時間"""

    updated_at: datetime
    """更新時間"""


class PluginRegistry(BaseModel):
    """插件註冊表"""

    registry_id: str
    """註冊表ID"""

    name: str
    """註冊表名稱"""

    plugins: Dict[str, PluginManifest] = Field(default_factory=dict)
    """已註冊插件"""

    total_plugins: int = Field(ge=0, default=0)
    """插件總數"""

    active_plugins: int = Field(ge=0, default=0)
    """活躍插件數"""

    registry_version: str
    """註冊表版本"""

    created_at: datetime
    """創建時間"""

    updated_at: datetime
    """更新時間"""


class PluginHealthCheck(BaseModel):
    """插件健康檢查"""

    plugin_id: str
    """插件ID"""

    status: PluginStatus
    """插件狀態"""

    last_check_time: datetime
    """最後檢查時間"""

    response_time_ms: Optional[float] = None
    """響應時間(毫秒)"""

    error_message: Optional[str] = None
    """錯誤信息"""

    health_score: float = Field(ge=0.0, le=100.0, default=100.0)
    """健康分數"""

    uptime_percentage: float = Field(ge=0.0, le=100.0, default=100.0)
    """運行時間百分比"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """健康檢查元數據"""

