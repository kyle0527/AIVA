"""
AIVA 能力管理 Schema - v2.0 極簡版

專為 AI 動態執行設計，遵循「協議先行」策略。
只保留 AI 理解和執行命令所需的最小資訊集。

設計原則：
1. CLI 為中心 - 一切圍繞命令執行
2. 參數動態化 - 支援 AI 根據情境調整參數
3. 語言無關 - 支援 Python/Rust/Go/TypeScript/Docker
4. 去除雜訊 - 移除監控、維運等非執行相關資訊

參考：AIVA_系統架構升級計畫.md
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ==================== v2.0 極簡能力模型 ====================


class ParameterType(str, Enum):
    """參數類型枚舉"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"


class ExecutableParameter(BaseModel):
    """可執行參數定義
    
    告訴 AI 如何正確填充參數值。
    包含類型約束、預設值和可選值列表。
    """
    
    description: str = Field(..., description="參數用途說明（簡短功能描述）")
    type: ParameterType = Field(
        default=ParameterType.STRING,
        description="參數類型：string/integer/float/boolean/array"
    )
    required: bool = Field(default=False, description="是否必填")
    default: Any | None = Field(None, description="預設值")
    enum: list[str] | None = Field(None, description="允許的值列表（枚舉）")
    
    # 數值範圍約束（用於掃描強度等可調參數）
    min_value: float | None = Field(None, description="最小值（數值類型）")
    max_value: float | None = Field(None, description="最大值（數值類型）")
    
    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str | ParameterType) -> ParameterType:
        """驗證並轉換參數類型"""
        if isinstance(v, str):
            return ParameterType(v.lower())
        return v


class ExecutableCapability(BaseModel):
    """可執行能力清單
    
    這是 AI 與底層工具之間的統一協議。
    AI 通過此協議理解能力、生成命令、調用工具。
    
    關鍵特性：
    - 語言無關：底層可以是 Python/Rust/Go/TypeScript/Docker
    - 動態參數化：支援 AI 根據情境調整參數
    - 類型安全：Pydantic 自動驗證參數類型和範圍
    - RAG 友好：description 和 tags 用於語義檢索
    """
    
    # === 核心識別 ===
    id: str = Field(
        ...,
        description="唯一標識符，格式建議：{category}.{action}.{target}（例：xss.scan.web, sqli.detect.api）"
    )
    name: str = Field(..., description="顯示名稱（人類可讀）")
    
    # === 功能描述（AI 理解用）===
    description: str = Field(
        ...,
        description="功能性描述。格式：動詞+對象|特徵1|特徵2。例：'掃描網頁XSS漏洞|支援DOM分析|檢測反射型存儲型'"
    )
    
    # === CLI 執行配置 ===
    command_template: str = Field(
        ...,
        description="CLI 命令模板，使用 {param_name} 作為佔位符。例：'python -m xss_scan --url {target} --depth {depth}'"
    )
    
    # === 參數定義 ===
    parameters: dict[str, ExecutableParameter] = Field(
        default_factory=dict,
        description="可調參數字典。Key 為模板中的變數名"
    )
    
    # === 分類標籤（RAG 檢索用）===
    tags: list[str] = Field(
        default_factory=list,
        description="功能標籤，用於快速過濾和檢索。例：['web', 'vulnerability', 'attack']"
    )
    
    # === 執行配置 ===
    timeout: int = Field(
        default=300,
        description="執行超時（秒）",
        gt=0,
        le=3600
    )
    working_dir: str | None = Field(
        None,
        description="執行目錄（相對於專案根目錄）"
    )
    
    # === 元數據（可選）===
    version: str = Field(default="1.0.0", description="能力版本")
    created_at: datetime | None = Field(None, description="創建時間")
    updated_at: datetime | None = Field(None, description="更新時間")
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """驗證 ID 格式"""
        if not v or not v.strip():
            raise ValueError("ID 不能為空")
        if " " in v:
            raise ValueError("ID 不能包含空格")
        return v.strip().lower()
    
    @field_validator("command_template")
    @classmethod
    def validate_command_template(cls, v: str) -> str:
        """驗證命令模板"""
        if not v or not v.strip():
            raise ValueError("命令模板不能為空")
        return v.strip()
    
    def generate_command(self, params: dict[str, Any]) -> str:
        """
        根據參數值生成實際 CLI 命令
        
        Args:
            params: 參數值字典
            
        Returns:
            完整的 CLI 命令字符串
            
        Raises:
            ValueError: 缺少必填參數或參數值不合法
        """
        # 驗證必填參數
        for param_name, param_def in self.parameters.items():
            if param_def.required and param_name not in params:
                raise ValueError(f"缺少必填參數: {param_name}")
        
        # 填充參數（包含預設值）
        final_params = {}
        for param_name, param_def in self.parameters.items():
            if param_name in params:
                final_params[param_name] = params[param_name]
            elif param_def.default is not None:
                final_params[param_name] = param_def.default
            else:
                final_params[param_name] = ""
        
        # 生成命令
        return self.command_template.format(**final_params)
    
    def validate_params(self, params: dict[str, Any]) -> dict[str, str]:
        """
        驗證參數值是否符合約束
        
        Args:
            params: 待驗證的參數字典
            
        Returns:
            錯誤字典 {param_name: error_message}，無錯誤則返回空字典
        """
        errors = {}
        
        for param_name, value in params.items():
            if param_name not in self.parameters:
                errors[param_name] = f"未知參數: {param_name}"
                continue
            
            param_def = self.parameters[param_name]
            
            # 檢查枚舉值
            if param_def.enum and value not in param_def.enum:
                errors[param_name] = f"值必須是 {param_def.enum} 之一"
            
            # 檢查數值範圍
            if param_def.type in (ParameterType.INTEGER, ParameterType.FLOAT):
                try:
                    num_value = float(value)
                    if param_def.min_value is not None and num_value < param_def.min_value:
                        errors[param_name] = f"值必須 >= {param_def.min_value}"
                    if param_def.max_value is not None and num_value > param_def.max_value:
                        errors[param_name] = f"值必須 <= {param_def.max_value}"
                except (ValueError, TypeError):
                    errors[param_name] = f"必須是 {param_def.type.value} 類型"
        
        return errors


# ==================== 輔助函數 ====================


def validate_capability(capability_dict: dict) -> ExecutableCapability:
    """驗證並創建能力清單"""
    return ExecutableCapability(**capability_dict)


def create_capability_from_analysis(
    analysis_data: dict,
    mmd_filename: str
) -> ExecutableCapability:
    """
    從分析工具產出的數據創建能力清單
    
    Args:
        analysis_data: 分析結果中的單個函數數據
        mmd_filename: Mermaid 圖表文件名（去掉 .mmd）
        
    Returns:
        ExecutableCapability 實例
    """
    # 從圖表文件名提取 ID
    capability_id = mmd_filename
    
    # 從 analysis_data 提取資訊
    function_name = analysis_data.get("functionName", analysis_data.get("function_name", "unknown"))
    category = analysis_data.get("category", "other")
    
    # 構建參數定義
    parameters = {}
    for input_param in analysis_data.get("inputs", []):
        parameters[input_param] = ExecutableParameter(
            description=f"參數 {input_param}",
            type=ParameterType.STRING,
            required=True,
            default=None,
            enum=None,
            min_value=None,
            max_value=None
        )
    
    # 構建標籤
    tags = [category]
    if "call_targets" in analysis_data:
        tags.extend(analysis_data["call_targets"][:3])  # 取前3個調用目標作為標籤
    
    return ExecutableCapability(
        id=capability_id,
        name=function_name,
        description=analysis_data.get("description", f"執行 {function_name} 功能"),
        command_template="# 待補充 CLI 命令模板",
        parameters=parameters,
        tags=tags,
        timeout=300,
        working_dir=None,
        created_at=None,
        updated_at=None
    )


# ==================== 範例數據 ====================


EXAMPLE_XSS_SCANNER = {
    "id": "xss.scan.web",
    "name": "XSS 漏洞掃描器",
    "description": "掃描網頁XSS漏洞|支援反射型存儲型DOM型|可調深度和並發數",
    "command_template": "python -m services.features.function_xss.xss_worker --url {target} --depth {depth} --threads {threads}",
    "parameters": {
        "target": {
            "description": "目標URL",
            "type": "string",
            "required": True
        },
        "depth": {
            "description": "爬蟲深度",
            "type": "integer",
            "default": 2,
            "min_value": 1,
            "max_value": 5
        },
        "threads": {
            "description": "並發執行緒",
            "type": "integer",
            "default": 10,
            "min_value": 1,
            "max_value": 50
        }
    },
    "tags": ["web", "vulnerability", "xss", "scan"],
    "timeout": 600
}


EXAMPLE_CRYPTO_ANALYZER = {
    "id": "crypto.analyze.tls",
    "name": "加密分析器",
    "description": "分析TLS/SSL配置|檢測加密套件強度|驗證憑證鏈",
    "command_template": "./bin/crypto_analyzer --target {host} --mode {mode}",
    "parameters": {
        "host": {
            "description": "目標主機",
            "type": "string",
            "required": True
        },
        "mode": {
            "description": "分析模式",
            "type": "string",
            "default": "fast",
            "enum": ["fast", "full", "deep"]
        }
    },
    "tags": ["network", "crypto", "analysis"],
    "timeout": 60,
    "working_dir": "services/crypto_tools"
}


# ==================== 廢棄說明 ====================


# v1.0 模型（ToolManifest, CapabilityInfo 等）已廢棄
# 不再使用包含監控、評分卡等維運資訊的複雜模型
# 請使用 ExecutableCapability 作為唯一標準

