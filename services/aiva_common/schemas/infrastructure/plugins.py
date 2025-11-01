"""
插件管理 Schema 定義

此模組定義了插件清單、執行上下文、配置等相關的資料模型。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...enums import PluginStatus, PluginType


class PluginManifest(BaseModel):
    """插件清單"""

    plugin_id: str = Field(description="插件唯一標識符")
    name: str = Field(description="插件名稱")
    version: str = Field(description="插件版本")
    author: str = Field(description="插件作者")
    description: str = Field(description="插件描述")
    plugin_type: PluginType = Field(description="插件類型")
    dependencies: list[str] = Field(default_factory=list, description="依賴插件列表")
    permissions: list[str] = Field(default_factory=list, description="所需權限列表")
    config_schema: dict[str, Any] | None = Field(
        default=None, description="配置 Schema"
    )
    min_aiva_version: str = Field(description="最低AIVA版本要求")
    max_aiva_version: str | None = Field(
        default=None, description="最高AIVA版本要求"
    )
    entry_point: str = Field(description="插件入口點")
    homepage: str | None = Field(default=None, description="插件主頁")
    repository: str | None = Field(default=None, description="源碼倉庫")
    license: str = Field(default="MIT", description="許可證")
    keywords: list[str] = Field(default_factory=list, description="關鍵詞")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="創建時間"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="更新時間"
    )

    @field_validator("plugin_id")
    @classmethod
    def validate_plugin_id(cls, v: str) -> str:
        """驗證插件ID"""
        if not v.strip():
            raise ValueError("插件ID不能為空")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("插件ID只能包含字母、數字、連字符和下劃線")
        if len(v) > 50:
            raise ValueError("插件ID長度不能超過50個字符")
        return v.strip().lower()

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """驗證版本號"""
        import re

        if not re.match(r"^(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9-]+)?$", v):
            raise ValueError("版本號必須遵循語義化版本格式 (x.y.z)")
        return v

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v: list[str]) -> list[str]:
        """驗證權限列表"""
        allowed_permissions = {
            "read_config",
            "write_config",
            "read_data",
            "write_data",
            "network_access",
            "file_system",
            "execute_commands",
            "admin_access",
        }
        for permission in v:
            if permission not in allowed_permissions:
                raise ValueError(f"無效的權限: {permission}")
        return v


class PluginExecutionContext(BaseModel):
    """插件執行上下文"""

    plugin_id: str = Field(description="插件ID")
    execution_id: str = Field(description="執行ID")
    input_data: dict[str, Any] = Field(description="輸入數據")
    context: dict[str, Any] = Field(default_factory=dict, description="執行上下文")
    timeout_seconds: int = Field(
        default=60, ge=1, le=600, description="執行超時時間(秒)"
    )
    environment: dict[str, str] = Field(default_factory=dict, description="環境變數")
    working_directory: str | None = Field(default=None, description="工作目錄")
    user_id: str | None = Field(default=None, description="執行用戶ID")
    session_id: str | None = Field(default=None, description="會話ID")
    trace_id: str | None = Field(default=None, description="追蹤ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="創建時間"
    )

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, v: str) -> str:
        """驗證執行ID"""
        if not v.strip():
            raise ValueError("執行ID不能為空")
        return v.strip()


class PluginExecutionResult(BaseModel):
    """插件執行結果"""

    plugin_id: str = Field(description="插件ID")
    execution_id: str = Field(description="執行ID")
    status: PluginStatus = Field(description="執行狀態")
    result: dict[str, Any] | None = Field(default=None, description="執行結果")
    error_message: str | None = Field(default=None, description="錯誤信息")
    error_code: str | None = Field(default=None, description="錯誤代碼")
    execution_time_ms: float = Field(ge=0, description="執行時間(毫秒)")
    memory_usage_mb: float | None = Field(
        default=None, ge=0, description="內存使用(MB)"
    )
    output_logs: list[str] = Field(default_factory=list, description="輸出日誌")
    warnings: list[str] = Field(default_factory=list, description="警告信息")
    artifacts: dict[str, str] = Field(default_factory=dict, description="產出物件")
    metadata: dict[str, Any] = Field(default_factory=dict, description="執行元數據")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="開始時間"
    )
    end_time: datetime | None = Field(default=None, description="結束時間")

    @field_validator("error_message")
    @classmethod
    def validate_error_message(cls, v: str | None) -> str | None:
        """驗證錯誤信息"""
        if v is not None and len(v) > 2000:
            return v[:2000] + "... (截斷)"
        return v


class PluginConfig(BaseModel):
    """插件配置"""

    plugin_id: str = Field(description="插件ID")
    enabled: bool = Field(default=True, description="是否啟用")
    auto_start: bool = Field(default=False, description="是否自動啟動")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    max_instances: int = Field(default=1, ge=1, le=10, description="最大實例數")
    resource_limits: dict[str, Any] = Field(
        default_factory=dict, description="資源限制"
    )
    environment_variables: dict[str, str] = Field(
        default_factory=dict, description="環境變數"
    )
    custom_config: dict[str, Any] = Field(
        default_factory=dict, description="自定義配置"
    )
    tags: list[str] = Field(default_factory=list, description="標籤")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="創建時間"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="更新時間"
    )

    @field_validator("custom_config")
    @classmethod
    def validate_custom_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        """驗證自定義配置"""
        # 確保配置值可序列化
        import json

        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"自定義配置必須可JSON序列化: {e}")
        return v


class PluginRegistry(BaseModel):
    """插件註冊表"""

    registry_id: str = Field(description="註冊表ID")
    name: str = Field(description="註冊表名稱")
    description: str = Field(description="註冊表描述")
    plugins: list[PluginManifest] = Field(default_factory=list, description="插件列表")
    total_plugins: int = Field(default=0, ge=0, description="插件總數")
    active_plugins: int = Field(default=0, ge=0, description="活躍插件數")
    registry_url: str | None = Field(default=None, description="註冊表URL")
    last_sync: datetime | None = Field(default=None, description="最後同步時間")
    metadata: dict[str, Any] = Field(default_factory=dict, description="註冊表元數據")

    @field_validator("plugins")
    @classmethod
    def validate_plugins(cls, v: list[PluginManifest]) -> list[PluginManifest]:
        """驗證插件列表"""
        plugin_ids = [plugin.plugin_id for plugin in v]
        if len(plugin_ids) != len(set(plugin_ids)):
            raise ValueError("插件ID不能重複")
        return v


class PluginHealthCheck(BaseModel):
    """插件健康檢查"""

    plugin_id: str = Field(description="插件ID")
    check_id: str = Field(description="檢查ID")
    status: str = Field(description="健康狀態")
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="最後檢查時間"
    )
    response_time_ms: float = Field(ge=0, description="響應時間(毫秒)")
    error_count: int = Field(default=0, ge=0, description="錯誤計數")
    warning_count: int = Field(default=0, ge=0, description="警告計數")
    details: dict[str, Any] = Field(default_factory=dict, description="檢查詳情")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """驗證健康狀態"""
        allowed = {"healthy", "degraded", "unhealthy", "unknown"}
        if v not in allowed:
            raise ValueError(f"健康狀態必須是 {allowed} 之一")
        return v
