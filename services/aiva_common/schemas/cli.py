"""
CLI 界面 Schema 定義

此模組定義了命令行界面相關的資料模型，包括命令定義、參數、執行結果等。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CLIParameter(BaseModel):
    """CLI 參數定義"""

    name: str = Field(description="參數名稱")
    type: str = Field(description="參數類型")
    description: str = Field(description="參數描述")
    required: bool = Field(default=False, description="是否必需")
    default_value: str | int | float | bool | None = Field(
        default=None, description="默認值"
    )
    choices: list[str] | None = Field(default=None, description="可選值列表")
    min_value: int | float | None = Field(default=None, description="最小值")
    max_value: int | float | None = Field(default=None, description="最大值")
    pattern: str | None = Field(default=None, description="正則表達式模式")
    help_text: str | None = Field(default=None, description="幫助文本")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """驗證參數類型"""
        allowed_types = {
            "string",
            "integer",
            "float",
            "boolean",
            "choice",
            "file",
            "directory",
        }
        if v not in allowed_types:
            raise ValueError(f"參數類型必須是 {allowed_types} 之一")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """驗證參數名稱"""
        if not v.strip():
            raise ValueError("參數名稱不能為空")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("參數名稱只能包含字母、數字、連字符和下劃線")
        return v.strip().lower()


class CLICommand(BaseModel):
    """CLI 命令定義"""

    command_name: str = Field(description="命令名稱")
    description: str = Field(description="命令描述")
    category: str = Field(default="general", description="命令分類")
    parameters: list[CLIParameter] = Field(
        default_factory=list, description="命令參數列表"
    )
    examples: list[str] = Field(default_factory=list, description="使用示例")
    aliases: list[str] = Field(default_factory=list, description="命令別名")
    deprecated: bool = Field(default=False, description="是否已棄用")
    min_args: int = Field(default=0, ge=0, description="最少參數數量")
    max_args: int | None = Field(default=None, ge=0, description="最多參數數量")
    requires_auth: bool = Field(default=False, description="是否需要認證")
    permissions: list[str] = Field(default_factory=list, description="所需權限")
    tags: list[str] = Field(default_factory=list, description="標籤")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="創建時間"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="更新時間"
    )

    @field_validator("command_name")
    @classmethod
    def validate_command_name(cls, v: str) -> str:
        """驗證命令名稱"""
        if not v.strip():
            raise ValueError("命令名稱不能為空")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("命令名稱只能包含字母、數字、連字符和下劃線")
        return v.strip().lower()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """驗證命令分類"""
        allowed_categories = {
            "general",
            "scan",
            "security",
            "analysis",
            "reporting",
            "config",
            "admin",
            "debug",
            "plugin",
            "utility",
        }
        if v not in allowed_categories:
            raise ValueError(f"命令分類必須是 {allowed_categories} 之一")
        return v


class CLIExecutionResult(BaseModel):
    """CLI 執行結果"""

    command: str = Field(description="執行的命令")
    arguments: list[str] = Field(default_factory=list, description="命令參數")
    exit_code: int = Field(description="退出代碼")
    stdout: str = Field(default="", description="標準輸出")
    stderr: str = Field(default="", description="標準錯誤")
    execution_time_ms: float = Field(ge=0, description="執行時間(毫秒)")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="開始時間"
    )
    end_time: datetime | None = Field(default=None, description="結束時間")
    user_id: str | None = Field(default=None, description="執行用戶ID")
    session_id: str | None = Field(default=None, description="會話ID")
    working_directory: str | None = Field(default=None, description="工作目錄")
    environment: dict[str, str] = Field(default_factory=dict, description="環境變數")
    resource_usage: dict[str, Any] = Field(
        default_factory=dict, description="資源使用情況"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="執行元數據")

    @field_validator("exit_code")
    @classmethod
    def validate_exit_code(cls, v: int) -> int:
        """驗證退出代碼"""
        if not -128 <= v <= 127:
            raise ValueError("退出代碼必須在 -128 到 127 之間")
        return v

    @field_validator("stdout", "stderr")
    @classmethod
    def validate_output(cls, v: str) -> str:
        """驗證輸出內容"""
        # 限制輸出長度以避免過大的對象
        if len(v) > 100000:  # 100KB 限制
            return v[:100000] + "... (截斷)"
        return v


class CLISession(BaseModel):
    """CLI 會話"""

    session_id: str = Field(description="會話ID")
    user_id: str | None = Field(default=None, description="用戶ID")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="開始時間"
    )
    end_time: datetime | None = Field(default=None, description="結束時間")
    commands_executed: list[str] = Field(
        default_factory=list, description="已執行命令列表"
    )
    current_directory: str = Field(default="/", description="當前目錄")
    environment: dict[str, str] = Field(
        default_factory=dict, description="會話環境變數"
    )
    settings: dict[str, Any] = Field(default_factory=dict, description="會話設置")
    history: list[CLIExecutionResult] = Field(
        default_factory=list, description="命令歷史"
    )
    status: str = Field(default="active", description="會話狀態")
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="最後活動時間"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """驗證會話狀態"""
        allowed = {"active", "inactive", "terminated", "suspended"}
        if v not in allowed:
            raise ValueError(f"會話狀態必須是 {allowed} 之一")
        return v


class CLIConfiguration(BaseModel):
    """CLI 配置"""

    config_id: str = Field(description="配置ID")
    name: str = Field(description="配置名稱")
    description: str | None = Field(default=None, description="配置描述")
    global_settings: dict[str, Any] = Field(
        default_factory=dict, description="全局設置"
    )
    command_settings: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="命令特定設置"
    )
    aliases: dict[str, str] = Field(default_factory=dict, description="命令別名映射")
    default_parameters: dict[str, Any] = Field(
        default_factory=dict, description="默認參數"
    )
    security_settings: dict[str, Any] = Field(
        default_factory=dict, description="安全設置"
    )
    ui_settings: dict[str, Any] = Field(default_factory=dict, description="界面設置")
    logging_settings: dict[str, Any] = Field(
        default_factory=dict, description="日誌設置"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="創建時間"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="更新時間"
    )
    version: str = Field(default="1.0.0", description="配置版本")

    @field_validator("config_id")
    @classmethod
    def validate_config_id(cls, v: str) -> str:
        """驗證配置ID"""
        if not v.strip():
            raise ValueError("配置ID不能為空")
        return v.strip()


class CLIMetrics(BaseModel):
    """CLI 指標統計"""

    metric_id: str = Field(description="指標ID")
    time_period: str = Field(description="統計時間段")
    total_commands: int = Field(default=0, ge=0, description="總命令數")
    successful_commands: int = Field(default=0, ge=0, description="成功命令數")
    failed_commands: int = Field(default=0, ge=0, description="失敗命令數")
    average_execution_time_ms: float = Field(
        default=0, ge=0, description="平均執行時間(毫秒)"
    )
    most_used_commands: list[dict[str, Any]] = Field(
        default_factory=list, description="最常用命令"
    )
    error_rates: dict[str, float] = Field(
        default_factory=dict, description="錯誤率統計"
    )
    user_activity: dict[str, int] = Field(
        default_factory=dict, description="用戶活動統計"
    )
    resource_usage: dict[str, Any] = Field(
        default_factory=dict, description="資源使用統計"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="統計開始時間"
    )
    end_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="統計結束時間"
    )

    @field_validator("time_period")
    @classmethod
    def validate_time_period(cls, v: str) -> str:
        """驗證時間段"""
        allowed = {"hourly", "daily", "weekly", "monthly", "yearly", "custom"}
        if v not in allowed:
            raise ValueError(f"時間段必須是 {allowed} 之一")
        return v
