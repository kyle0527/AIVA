"""
AIVA Cli Schema - 自動生成
=====================================

AIVA跨語言Schema統一定義 - 以手動維護版本為準

⚠️  此配置已同步手動維護的Schema定義，確保單一事實原則
📅 最後更新: 2025-10-30T00:00:00.000000
🔄 Schema 版本: 1.1.0
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .base_types import *


class CLIParameter(BaseModel):
    """CLI 參數定義"""

    name: str
    """參數名稱"""

    type: Literal['string', 'integer', 'float', 'boolean', 'choice', 'file', 'directory'] = Field(default='string')
    """參數類型"""

    description: str
    """參數描述"""

    required: bool = Field(default=False)
    """是否必需"""

    default_value: Optional[Any] = None
    """默認值"""

    choices: Optional[List[str]] = None
    """可選值列表"""

    min_value: Optional[float] = None
    """最小值"""

    max_value: Optional[float] = None
    """最大值"""

    pattern: Optional[str] = None
    """正則表達式模式"""

    help_text: Optional[str] = None
    """幫助文本"""


class CLICommand(BaseModel):
    """CLI 命令定義"""

    command_name: str
    """命令名稱"""

    description: str
    """命令描述"""

    category: Literal['general', 'scan', 'security', 'analysis', 'reporting', 'config', 'admin', 'debug', 'plugin', 'utility'] = Field(default="general")
    """命令分類"""

    parameters: List[CLIParameter] = Field(default_factory=list)
    """命令參數列表"""

    examples: List[str] = Field(default_factory=list)
    """使用示例"""

    aliases: List[str] = Field(default_factory=list)
    """命令別名"""

    deprecated: bool = Field(default=False)
    """是否已棄用"""

    min_args: int = Field(ge=0, default=0)
    """最少參數數量"""

    max_args: Optional[int] = Field(ge=0, default=None)
    """最多參數數量"""

    requires_auth: bool = Field(default=False)
    """是否需要認證"""

    permissions: List[str] = Field(default_factory=list)
    """所需權限"""

    tags: List[str] = Field(default_factory=list)
    """標籤"""

    created_at: datetime
    """創建時間"""

    updated_at: datetime
    """更新時間"""


class CLIExecutionResult(BaseModel):
    """CLI 執行結果"""

    command: str
    """執行的命令"""

    arguments: List[str] = Field(default_factory=list)
    """命令參數"""

    exit_code: int
    """退出代碼"""

    stdout: str = Field(default="")
    """標準輸出"""

    stderr: str = Field(default="")
    """標準錯誤"""

    execution_time_ms: float = Field(ge=0)
    """執行時間(毫秒)"""

    start_time: datetime
    """開始時間"""

    end_time: Optional[datetime] = None
    """結束時間"""

    user_id: Optional[str] = None
    """執行用戶ID"""

    session_id: Optional[str] = None
    """會話ID"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """執行元數據"""


class CLISession(BaseModel):
    """CLI 會話"""

    session_id: str
    """會話ID"""

    user_id: Optional[str] = None
    """用戶ID"""

    start_time: datetime
    """開始時間"""

    end_time: Optional[datetime] = None
    """結束時間"""

    command_history: List[str] = Field(default_factory=list)
    """命令歷史"""

    environment: Dict[str, str] = Field(default_factory=dict)
    """環境變數"""

    working_directory: str
    """工作目錄"""

    active: bool = Field(default=True)
    """會話是否活躍"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """會話元數據"""


class CLIConfiguration(BaseModel):
    """CLI 配置"""

    config_id: str
    """配置ID"""

    name: str
    """配置名稱"""

    settings: Dict[str, Any] = Field(default_factory=dict)
    """配置設定"""

    auto_completion: bool = Field(default=True)
    """是否啟用自動完成"""

    history_size: int = Field(ge=1, le=10000, default=1000)
    """歷史記錄大小"""

    prompt_style: str = Field(default="default")
    """提示符樣式"""

    color_scheme: str = Field(default="default")
    """顏色方案"""

    timeout_seconds: int = Field(ge=1, le=3600, default=300)
    """命令超時時間(秒)"""

    created_at: datetime
    """創建時間"""

    updated_at: datetime
    """更新時間"""


class CLIMetrics(BaseModel):
    """CLI 使用指標"""

    metric_id: str
    """指標ID"""

    command_count: int = Field(ge=0, default=0)
    """命令執行總數"""

    successful_commands: int = Field(ge=0, default=0)
    """成功執行的命令數"""

    failed_commands: int = Field(ge=0, default=0)
    """失敗的命令數"""

    average_execution_time_ms: float = Field(ge=0, default=0.0)
    """平均執行時間(毫秒)"""

    most_used_commands: List[str] = Field(default_factory=list)
    """最常用命令列表"""

    peak_usage_time: Optional[datetime] = None
    """峰值使用時間"""

    collection_period_start: datetime
    """統計開始時間"""

    collection_period_end: Optional[datetime] = None
    """統計結束時間"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """統計元數據"""

