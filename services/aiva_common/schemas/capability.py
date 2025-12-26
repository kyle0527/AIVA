"""
AIVA 能力管理相關 Schema

此模組定義能力管理系統中使用的統一數據模型。
包含能力定義、評分卡、執行記錄等核心結構。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..enums import ProgrammingLanguage, TaskStatus


class InputParameter(BaseModel):
    """輸入參數定義"""

    name: str = Field(..., description="參數名稱")
    type: str = Field(..., description="參數類型")
    required: bool = Field(default=True, description="是否必需")
    description: str = Field(..., description="參數描述")
    default: Any | None = Field(None, description="默認值")


class OutputParameter(BaseModel):
    """輸出參數定義"""

    name: str = Field(..., description="輸出名稱")
    type: str = Field(..., description="輸出類型")
    description: str = Field(..., description="輸出描述")


class CapabilityInfo(BaseModel):
    """能力信息"""

    id: str = Field(..., description="能力唯一標識符")
    name: str = Field(..., description="能力顯示名稱")
    description: str | None = Field(None, description="能力詳細描述")
    version: str = Field(default="1.0.0", description="能力版本")

    # 技術信息
    language: ProgrammingLanguage = Field(..., description="實現語言")
    entrypoint: str = Field(..., description="入口點路徑")
    topic: str = Field(..., description="主題分類")

    # 接口定義
    inputs: list[InputParameter] | None = Field(None, description="輸入參數列表")
    outputs: list[OutputParameter] | None = Field(None, description="輸出參數列表")

    # 依賴與前置條件
    prerequisites: list[str] | None = Field(None, description="前置條件列表")
    dependencies: list[str] | None = Field(None, description="依賴的其他能力ID")

    # 元數據
    tags: list[str] | None = Field(None, description="標籤列表")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="能力狀態")

    # 時間戳
    created_at: datetime | None = Field(None, description="創建時間")
    updated_at: datetime | None = Field(None, description="更新時間")


class CapabilityScorecard(BaseModel):
    """能力評分卡"""

    capability_id: str = Field(..., description="能力ID")

    # 7日性能指標
    success_rate_7d: float = Field(default=0.0, description="7日成功率", ge=0, le=1)
    avg_latency_ms: float = Field(default=0.0, description="平均延遲(毫秒)", ge=0)
    availability_7d: float = Field(default=1.0, description="7日可用性", ge=0, le=1)
    usage_count_7d: int = Field(default=0, description="7日使用次數", ge=0)

    # 時間戳
    last_used_at: datetime | None = Field(None, description="最後使用時間")
    last_updated_at: datetime | None = Field(None, description="最後更新時間")

    # 額外指標
    error_count_7d: int = Field(default=0, description="7日錯誤次數", ge=0)
    metadata: dict[str, Any] | None = Field(None, description="其他元數據")


# ==================== Manifest 架構 (v1.0) ====================


from enum import Enum


class ParameterMappingStrategy(str, Enum):
    """參數映射策略"""
    LINEAR = "linear"  # intensity * (max - min) + min
    INVERSE = "inverse"  # max - intensity * (max - min)
    EXPONENTIAL = "exponential"  # min * (max/min) ^ intensity
    LOGARITHMIC = "logarithmic"  # log-based scaling


class TunableParameter(BaseModel):
    """可調參數定義 (B類參數)
    
    參數值由 AI 強度等級 (0.0-1.0) 映射生成。
    例如: intensity=0.9 → threads=9 (range 1-10, linear)
    """
    
    type: str = Field(..., description="參數類型 (integer/float/boolean/string)")
    default: Any = Field(..., description="默認值")
    min: Any | None = Field(None, description="最小值 (數值類型)")
    max: Any | None = Field(None, description="最大值 (數值類型)")
    mapping_strategy: ParameterMappingStrategy = Field(
        default=ParameterMappingStrategy.LINEAR,
        description="映射策略"
    )
    description: str = Field(..., description="參數描述")


class AITriggerTags(BaseModel):
    """AI 觸發標籤 (Hard Mask 過濾)
    
    標籤用於能力過濾和觸發判斷。
    - required: 必須全部匹配
    - boost: 匹配會提高優先級
    - exclude: 匹配會排除此能力
    """
    
    required: list[str] = Field(default_factory=list, description="必需標籤")
    boost: list[str] = Field(default_factory=list, description="加分標籤")
    exclude: list[str] = Field(default_factory=list, description="排除標籤")


class ManifestMetadata(BaseModel):
    """Manifest 元數據"""
    
    flow_id: int = Field(..., description="流程 ID (對應 840 flows)")
    script_name: str = Field(..., description="腳本名稱 (從 840 flows path 中提取)")
    script_file: str = Field(..., description="腳本文件名 (例如: monitoring.py)")
    script_path: str = Field(..., description="腳本相對路徑 (從項目根目錄)")
    tool_name: str = Field(..., description="工具顯示名稱")
    description: str = Field(..., description="工具功能描述")
    language: ProgrammingLanguage = Field(..., description="實現語言")
    version: str = Field(default="1.0.0", description="Manifest 版本")
    module: str | None = Field(None, description="所屬模組 (service_backbone/cognitive_core/etc.)")


class ManifestExecution(BaseModel):
    """執行配置"""
    
    binary_path: str = Field(..., description="可執行文件路徑 (相對於項目根目錄)")
    command_template: str = Field(
        ...,
        description="CLI 命令模板 (使用 {param_name} 佔位符)"
    )
    working_dir: str = Field(default=".", description="工作目錄")
    timeout_seconds: int = Field(default=300, description="超時時間(秒)", ge=1)


class ManifestParameters(BaseModel):
    """參數配置
    
    分為兩類:
    - context_mapping (A類): 從上下文直接獲取，key=manifest參數名, value=context鍵名
    - tunable (B類): 由 AI 強度等級映射生成
    """
    
    context_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="A類參數: 上下文映射 {manifest_param: context_key}"
    )
    tunable: dict[str, TunableParameter] = Field(
        default_factory=dict,
        description="B類參數: 可調參數定義"
    )


class AICognitive(BaseModel):
    """AI 認知配置"""
    
    primary_capability: str = Field(..., description="主要能力類別")
    risk_level: str = Field(default="low", description="風險等級 (low/medium/high/critical)")
    tags: AITriggerTags = Field(default_factory=AITriggerTags, description="觸發標籤")


class ToolManifest(BaseModel):
    """工具 Manifest 完整定義
    
    遵循「這是一個非常關鍵的架構轉折點.md」規範，
    提供 AI 與 CLI 工具之間的靜態映射。
    
    設計理念:
    1. Static Registry - 每個能力一個 JSON 文件
    2. AI → Flow ID → Manifest → CommandBuilder → CLI 執行
    3. 參數分離 (A類: 上下文, B類: AI調節)
    4. Hard Mask 過濾 (基於標籤)
    """
    
    meta: ManifestMetadata = Field(..., description="元數據")
    ai_cognitive: AICognitive = Field(..., description="AI 認知配置")
    execution: ManifestExecution = Field(..., description="執行配置")
    parameters: ManifestParameters = Field(..., description="參數配置")
    
    # 可選擴展
    dependencies: list[str] = Field(default_factory=list, description="依賴的其他 flow_id")
    examples: list[dict[str, Any]] = Field(default_factory=list, description="使用示例")
    notes: str | None = Field(None, description="額外註記")
