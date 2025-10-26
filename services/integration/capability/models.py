"""
AIVA 能力註冊系統核心模組
基於 aiva_common 規範的統一數據結構設計

遵循 aiva_common 設計原則:
- 使用統一的枚舉定義，禁止重複定義
- 基於 Pydantic v2 的強類型數據模型
- 符合國際標準和官方規範
- 完整的類型標註和驗證邏輯
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

# 遵循 aiva_common 規範 - 使用統一的枚舉定義
from aiva_common.enums import (
    ProgrammingLanguage,
    Severity,
    Confidence,
    TaskStatus,
    Environment,
    ModuleName,
    Topic
)


class CapabilityStatus(str, Enum):
    """能力狀態枚舉"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CapabilityType(str, Enum):
    """能力類型枚舉"""
    SCANNER = "scanner"
    DETECTOR = "detector"
    ANALYZER = "analyzer"
    REPORTER = "reporter"
    UTILITY = "utility"


class InputParameter(BaseModel):
    """輸入參數定義"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(..., description="參數名稱")
    type: str = Field(..., description="參數類型")
    required: bool = Field(default=True, description="是否必需")
    description: str = Field(..., description="參數描述")
    default: Optional[Any] = Field(None, description="默認值")
    validation_rules: Optional[Dict[str, Any]] = Field(None, description="驗證規則")


class OutputParameter(BaseModel):
    """輸出參數定義"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(..., description="輸出名稱")
    type: str = Field(..., description="輸出類型")
    description: str = Field(..., description="輸出描述")
    sample_value: Optional[Any] = Field(None, description="示例值")


class CapabilityRecord(BaseModel):
    """統一能力記錄格式"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    # 基本信息
    id: str = Field(..., description="能力唯一標識符", min_length=1)
    name: str = Field(..., description="能力顯示名稱")
    description: str = Field(..., description="能力詳細描述")
    version: str = Field(default="1.0.0", description="能力版本")
    
    # 技術信息
    module: str = Field(..., description="所屬模組名稱")
    language: ProgrammingLanguage = Field(..., description="實現語言")
    entrypoint: str = Field(..., description="入口點路徑")
    capability_type: CapabilityType = Field(..., description="能力類型")
    
    # 接口定義
    inputs: List[InputParameter] = Field(default_factory=list, description="輸入參數列表")
    outputs: List[OutputParameter] = Field(default_factory=list, description="輸出參數列表")
    
    # 依賴與前置條件
    prerequisites: List[str] = Field(default_factory=list, description="前置條件列表")
    dependencies: List[str] = Field(default_factory=list, description="依賴的其他能力ID")
    
    # 元數據
    tags: List[str] = Field(default_factory=list, description="標籤列表")
    category: Optional[str] = Field(None, description="分類")
    priority: int = Field(default=50, description="優先級 (0-100)", ge=0, le=100)
    
    # 運行時信息
    topic: Optional[str] = Field(None, description="消息隊列主題")
    timeout_seconds: int = Field(default=300, description="超時時間(秒)", gt=0)
    retry_count: int = Field(default=3, description="重試次數", ge=0)
    
    # 狀態信息
    status: CapabilityStatus = Field(default=CapabilityStatus.UNKNOWN, description="當前狀態")
    last_probe: Optional[datetime] = Field(None, description="最後探測時間")
    last_success: Optional[datetime] = Field(None, description="最後成功時間")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="創建時間")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新時間")
    
    # 配置信息
    config: Optional[Dict[str, Any]] = Field(None, description="額外配置")
    environment_vars: Optional[Dict[str, str]] = Field(None, description="環境變量")


class CapabilityEvidence(BaseModel):
    """能力探測證據"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    capability_id: str = Field(..., description="能力ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="探測時間")
    probe_type: str = Field(..., description="探測類型")
    success: bool = Field(..., description="是否成功")
    
    # 性能指標
    latency_ms: Optional[int] = Field(None, description="延遲時間(毫秒)", ge=0)
    memory_usage_mb: Optional[float] = Field(None, description="內存使用(MB)", ge=0)
    cpu_usage_percent: Optional[float] = Field(None, description="CPU使用率(%)", ge=0, le=100)
    
    # 測試數據
    sample_input: Optional[Dict[str, Any]] = Field(None, description="測試輸入")
    sample_output: Optional[Dict[str, Any]] = Field(None, description="測試輸出")
    
    # 錯誤信息
    error_message: Optional[str] = Field(None, description="錯誤消息")
    error_code: Optional[str] = Field(None, description="錯誤代碼")
    stack_trace: Optional[str] = Field(None, description="堆棧跟蹤")
    
    # 追蹤信息
    trace_id: Optional[str] = Field(None, description="追蹤ID")
    span_id: Optional[str] = Field(None, description="跨度ID")
    
    # 額外上下文
    environment: Optional[Dict[str, str]] = Field(None, description="運行環境")
    metadata: Optional[Dict[str, Any]] = Field(None, description="額外元數據")


class CapabilityScorecard(BaseModel):
    """能力記分卡"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    capability_id: str = Field(..., description="能力ID")
    evaluation_period: str = Field(..., description="評估週期")
    
    # 可用性指標
    availability_percent: float = Field(..., description="可用性百分比", ge=0, le=100)
    success_rate_percent: float = Field(..., description="成功率百分比", ge=0, le=100)
    
    # 性能指標
    avg_latency_ms: float = Field(..., description="平均延遲(毫秒)", ge=0)
    p95_latency_ms: float = Field(..., description="95%延遲(毫秒)", ge=0)
    p99_latency_ms: float = Field(..., description="99%延遲(毫秒)", ge=0)
    
    # 資源使用
    avg_memory_mb: Optional[float] = Field(None, description="平均內存使用(MB)", ge=0)
    avg_cpu_percent: Optional[float] = Field(None, description="平均CPU使用(%)", ge=0, le=100)
    
    # 錯誤統計
    total_executions: int = Field(..., description="總執行次數", ge=0)
    error_count: int = Field(..., description="錯誤次數", ge=0)
    timeout_count: int = Field(..., description="超時次數", ge=0)
    
    # 錯誤分類
    error_categories: Dict[str, int] = Field(default_factory=dict, description="錯誤分類統計")
    recent_errors: List[str] = Field(default_factory=list, description="最近錯誤列表")
    
    # 趨勢分析
    performance_trend: str = Field(..., description="性能趨勢 (improving/stable/degrading)")
    reliability_score: float = Field(..., description="可靠性評分 (0-100)", ge=0, le=100)
    
    # 時間戳
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最後更新時間")
    next_evaluation: datetime = Field(..., description="下次評估時間")


class CLITemplate(BaseModel):
    """CLI模板定義"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    capability_id: str = Field(..., description="能力ID")
    command: str = Field(..., description="CLI命令")
    description: str = Field(..., description="命令描述")
    
    # 參數定義
    arguments: List[Dict[str, Any]] = Field(default_factory=list, description="命令參數")
    options: List[Dict[str, Any]] = Field(default_factory=list, description="命令選項")
    
    # 示例和幫助
    examples: List[str] = Field(default_factory=list, description="使用示例")
    help_text: str = Field(..., description="幫助文本")
    
    # 生成信息
    template_version: str = Field(default="1.0", description="模板版本")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="生成時間")


class ExecutionRequest(BaseModel):
    """執行請求"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    capability_id: str = Field(..., description="要執行的能力ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="執行參數")
    
    # 執行選項
    timeout_seconds: Optional[int] = Field(None, description="超時時間", gt=0)
    retry_count: Optional[int] = Field(None, description="重試次數", ge=0)
    priority: int = Field(default=50, description="執行優先級", ge=0, le=100)
    
    # 上下文信息
    context: Optional[Dict[str, Any]] = Field(None, description="執行上下文")
    trace_id: Optional[str] = Field(None, description="追蹤ID")
    user_id: Optional[str] = Field(None, description="用戶ID")
    
    # 回調配置
    callback_url: Optional[str] = Field(None, description="回調URL")
    webhook_config: Optional[Dict[str, Any]] = Field(None, description="Webhook配置")


class ExecutionResult(BaseModel):
    """執行結果"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    request_id: str = Field(..., description="請求ID")
    capability_id: str = Field(..., description="能力ID")
    
    # 執行狀態
    success: bool = Field(..., description="是否成功")
    status: str = Field(..., description="執行狀態")
    
    # 結果數據
    result: Optional[Dict[str, Any]] = Field(None, description="執行結果")
    output: Optional[str] = Field(None, description="標準輸出")
    error_output: Optional[str] = Field(None, description="錯誤輸出")
    
    # 錯誤信息
    error_message: Optional[str] = Field(None, description="錯誤消息")
    error_code: Optional[str] = Field(None, description="錯誤代碼")
    
    # 性能指標
    execution_time_ms: int = Field(..., description="執行時間(毫秒)", ge=0)
    memory_peak_mb: Optional[float] = Field(None, description="內存峰值(MB)", ge=0)
    
    # 時間戳
    started_at: datetime = Field(..., description="開始時間")
    completed_at: datetime = Field(..., description="完成時間")
    
    # 追蹤信息
    trace_id: Optional[str] = Field(None, description="追蹤ID")
    span_id: Optional[str] = Field(None, description="跨度ID")
    
    # 元數據
    metadata: Optional[Dict[str, Any]] = Field(None, description="額外元數據")


# 常用的驗證函數
def validate_capability_id(capability_id: str) -> bool:
    """驗證能力ID格式"""
    if not capability_id:
        return False
    
    # 格式：category.module.function
    parts = capability_id.split('.')
    return len(parts) >= 2 and all(part.isidentifier() for part in parts)


def create_capability_id(category: str, module: str, function: str) -> str:
    """創建標準格式的能力ID"""
    return f"{category}.{module}.{function}"


# 示例數據創建函數
def create_sample_capability() -> CapabilityRecord:
    """創建示例能力記錄"""
    return CapabilityRecord(
        id="security.sqli.boolean_detection",
        name="SQL注入布爾盲注檢測",
        description="檢測Web應用中的SQL注入布爾盲注漏洞",
        module="function_sqli",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="services.features.function_sqli.worker:run_boolean_sqli",
        capability_type=CapabilityType.SCANNER,
        inputs=[
            InputParameter(
                name="url",
                type="str",
                required=True,
                description="目標URL",
                validation_rules={"format": "url"}
            ),
            InputParameter(
                name="timeout",
                type="int",
                required=False,
                description="超時時間(秒)",
                default=30,
                validation_rules={"min": 1, "max": 300}
            )
        ],
        outputs=[
            OutputParameter(
                name="vulnerabilities",
                type="List[Dict]",
                description="發現的漏洞列表",
                sample_value=[{"type": "sqli", "severity": "high"}]
            )
        ],
        tags=["security", "sqli", "web", "injection"],
        category="vulnerability_scanner",
        prerequisites=["network.connectivity"],
        timeout_seconds=300
    )