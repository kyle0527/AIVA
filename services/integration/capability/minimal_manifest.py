"""
AIVA 極簡能力清單 (Minimal Capability Manifest)

⚠️ 已棄用 (DEPRECATED) - 2026-01-04
=======================================
此模組已被 latest_classification.json（自動產出）取代。

原因：
1. 路徑 A（自動產出）已可提供 AI 所需的所有資訊
2. 手動維護的 Manifest 格式與自動產出不一致
3. 5M 特化 AI 不需要自然語言描述，只需要結構化特徵

替代方案：
- 能力定義：使用 aiva_flow_classifier.py 自動產出
- 數據源：data/internal_exploration/latest_classification.json
- 編碼器：使用 capability_encoder.py 將能力轉為 512 維向量

保留此文件僅為向後相容，不應用於新開發。
=======================================

原說明（歷史紀錄）：
專為 AI 執行設計，去除所有監控、維運等雜訊資訊。
只保留 AI 理解能力和執行命令所需的最小資訊集。

設計原則：
1. CLI 為中心 - 一切圍繞命令執行
2. 極簡描述 - 功能性描述，非自然語言
3. 統一參數化 - 所有能力都支援參數調整
4. 跨語言通用 - Python/Rust/Go/TypeScript 統一格式
"""
import warnings

warnings.warn(
    "minimal_manifest.py 已棄用，請使用 latest_classification.json 自動產出的能力定義",
    DeprecationWarning,
    stacklevel=2
)

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ParameterType(str, Enum):
    """參數類型"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"


class CLIParameter(BaseModel):
    """CLI 參數定義 - 告訴 AI 如何填空"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    description: str = Field(..., description="參數用途 (簡短功能描述)")
    type: ParameterType = Field(default=ParameterType.STRING, description="參數類型")
    required: bool = Field(default=False, description="是否必填")
    default: Optional[Any] = Field(None, description="預設值")
    enum: Optional[List[str]] = Field(None, description="允許的值列表")
    
    # 參數調整相關（用於掃描強度等）
    min_value: Optional[float] = Field(None, description="最小值（數值類型用）")
    max_value: Optional[float] = Field(None, description="最大值（數值類型用）")


class MinimalManifest(BaseModel):
    """極簡能力清單 - AI 執行專用"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # === 核心識別 ===
    id: str = Field(..., description="唯一標識符 (e.g., xss.scan, sqli.detect)")
    name: str = Field(..., description="顯示名稱")
    
    # === 功能描述（AI 理解用）===
    description: str = Field(
        ..., 
        description="功能性描述。格式：動詞+對象+特徵。例：'掃描網頁XSS漏洞|支援DOM分析|檢測反射型存儲型'"
    )
    
    # === CLI 執行 ===
    command_template: str = Field(
        ...,
        description="CLI 命令模板。使用 {param_name} 佔位符。例：'python -m xss_scan --url {target} --depth {depth}'"
    )
    
    # === 參數定義 ===
    parameters: Dict[str, CLIParameter] = Field(
        default_factory=dict,
        description="可調參數字典。Key 為模板中的變數名"
    )
    
    # === 分類標籤（RAG 檢索用）===
    tags: List[str] = Field(
        default_factory=list,
        description="功能標籤。例：['web', 'vulnerability', 'attack']"
    )
    
    # === 可選配置 ===
    timeout: int = Field(default=300, description="執行超時（秒）", gt=0)
    working_dir: Optional[str] = Field(None, description="執行目錄（相對於專案根目錄）")


# ==================== 使用範例 ====================

# 範例 1: Python 掃描工具
EXAMPLE_XSS_SCANNER = {
    "id": "xss.scan.web",
    "name": "XSS掃描器",
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

# 範例 2: Rust 高性能工具
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

# 範例 3: 端口掃描（Go 工具）
EXAMPLE_PORT_SCANNER = {
    "id": "recon.scan.port",
    "name": "端口掃描器",
    "description": "掃描目標端口|支援TCP和UDP|可調速度和範圍",
    "command_template": "./bin/port_scanner -host {target} -ports {ports} -speed {speed}",
    "parameters": {
        "target": {
            "description": "目標IP或域名",
            "type": "string",
            "required": True
        },
        "ports": {
            "description": "端口範圍",
            "type": "string",
            "default": "1-1000"
        },
        "speed": {
            "description": "掃描速度",
            "type": "string",
            "default": "normal",
            "enum": ["slow", "normal", "fast", "aggressive"]
        }
    },
    "tags": ["recon", "network", "port"],
    "timeout": 300
}

# 範例 4: SQL注入檢測
EXAMPLE_SQLI_DETECTOR = {
    "id": "sqli.detect.web",
    "name": "SQL注入檢測器",
    "description": "檢測SQL注入漏洞|支援多種資料庫|自動Payload變異",
    "command_template": "python -m services.features.function_sqli.sqli_worker --url {target} --level {level}",
    "parameters": {
        "target": {
            "description": "目標URL",
            "type": "string",
            "required": True
        },
        "level": {
            "description": "檢測級別(影響Payload數量)",
            "type": "integer",
            "default": 3,
            "min_value": 1,
            "max_value": 5
        },
        "dbms": {
            "description": "目標資料庫類型",
            "type": "string",
            "default": "auto",
            "enum": ["auto", "mysql", "postgresql", "mssql", "oracle"]
        }
    },
    "tags": ["web", "vulnerability", "sqli", "database"],
    "timeout": 900
}


def validate_manifest(manifest_dict: dict) -> MinimalManifest:
    """驗證並創建 Manifest"""
    return MinimalManifest(**manifest_dict)


def generate_cli_command(manifest: MinimalManifest, params: dict[str, Any]) -> str:
    """
    根據 Manifest 和參數生成實際 CLI 命令
    
    Args:
        manifest: 能力清單
        params: 參數值字典
        
    Returns:
        完整的 CLI 命令字符串
    """
    # 驗證必填參數
    for param_name, param_def in manifest.parameters.items():
        if param_def.required and param_name not in params:
            raise ValueError(f"缺少必填參數: {param_name}")
    
    # 填充預設值
    final_params = {}
    for param_name, param_def in manifest.parameters.items():
        if param_name in params:
            final_params[param_name] = params[param_name]
        elif param_def.default is not None:
            final_params[param_name] = param_def.default
        else:
            final_params[param_name] = ""
    
    # 生成命令
    return manifest.command_template.format(**final_params)


# ==================== 使用示範 ====================

if __name__ == "__main__":
    # 載入 Manifest
    manifest = validate_manifest(EXAMPLE_XSS_SCANNER)
    
    # AI 提供的參數
    ai_params = {
        "target": "https://example.com",
        "depth": 3,  # AI 決定加深掃描
        "threads": 20  # AI 決定提高並發
    }
    
    # 生成命令
    command = generate_cli_command(manifest, ai_params)
    print(f"生成的命令: {command}")
    
    # 輸出：
    # python -m services.features.function_xss.xss_worker --url https://example.com --depth 3 --threads 20
