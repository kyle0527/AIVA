---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 跨語言 Schema 完整使用指南

## 📑 目錄

- [🎯 概述](#-概述)
- [📋 Schema 分類](#-schema-分類)
- [🔧 使用指南](#-使用指南)
- [⚡ 快速開始](#-快速開始)
- [🔄 工具鏈使用](#-工具鏈使用)
- [📊 進階功能](#-進階功能)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

## 🎯 概述

AIVA 跨語言 Schema 系統提供統一的數據模型定義，支援 Python、Go、Rust 三種語言的自動代碼生成和 AI 組件智能操作。

### ✨ 核心特色

- **🔄 跨語言一致性**: 單一 YAML 定義，自動生成三種語言代碼
- **🤖 AI 友好接口**: 專為 AI 組件設計的智能操作接口
- **📝 自動代碼生成**: 支援 Pydantic v2、Go structs、Rust Serde
- **🔍 一致性驗證**: 自動檢測跨語言差異和問題
- **⚡ 零配置使用**: 開箱即用的完整工具鏈

## 📋 Schema 分類

### 📊 當前 Schema 統計
- **總計**: 54 個 Schema 類別，398 個字段
- **支援語言**: Python、Go、Rust
- **版本**: 1.1.0

### 🗂️ Schema 分類詳情

| 分類 | Schema 數量 | 描述 | 主要用途 |
|------|-------------|------|----------|
| **base_types** | 24 個 | 基礎資料類型 | 通用數據結構、錯誤處理、資產管理 |
| **messaging** | 3 個 | 訊息通訊格式 | 跨服務通訊、請求響應、事件處理 |
| **tasks** | 5 個 | 任務管理結構 | 掃描任務、功能任務、任務配置 |
| **findings** | 4 個 | 漏洞發現格式 | 安全掃描結果、證據記錄、影響評估 |
| **async_utils** | 6 個 | 異步工具模組 | 異步任務、重試策略、資源管理 |
| **plugins** | 6 個 | 插件管理系統 | 插件清單、執行上下文、健康檢查 |
| **cli** | 6 個 | 命令行界面 | CLI 參數、命令定義、執行結果 |

## 🚀 快速開始

### 1. 基本使用

```python
# Python 使用範例
from services.aiva_common.tools.cross_language_interface import CrossLanguageSchemaInterface

# 初始化接口
interface = CrossLanguageSchemaInterface()

# 獲取所有 Schema
all_schemas = interface.get_all_schemas()
print(f"總共 {len(all_schemas)} 個 Schema")

# 查找特定 Schema
async_config = interface.get_schema_by_name("AsyncTaskConfig")
print(f"找到 Schema: {async_config.name} - {async_config.description}")
```

### 2. 跨語言代碼生成

```python
# 生成三種語言的代碼
schema_name = "AsyncTaskConfig"

python_code = interface.generate_schema_code(schema_name, "python")
go_code = interface.generate_schema_code(schema_name, "go")
rust_code = interface.generate_schema_code(schema_name, "rust")

print("Python 代碼:")
print(python_code)
print("\nGo 代碼:")
print(go_code)
print("\nRust 代碼:")
print(rust_code)
```

### 3. AI 友好信息獲取

```python
# 獲取 AI 可理解的結構化信息
ai_info = interface.get_ai_friendly_schema_info("PluginManifest")
print(json.dumps(ai_info, indent=2, ensure_ascii=False))
```

## 📝 詳細使用範例

### 🔄 異步工具模組 (async_utils)

#### AsyncTaskConfig 使用範例

**Python 範例**:
```python
from services.aiva_common.schemas.async_utils import AsyncTaskConfig, RetryConfig

# 創建異步任務配置
task_config = AsyncTaskConfig(
    task_name="security_scan",
    timeout_seconds=300,
    retry_config=RetryConfig(
        max_attempts=3,
        backoff_base=2.0,
        exponential_backoff=True
    ),
    priority=8,
    tags=["security", "async"],
    metadata={"scan_type": "comprehensive"}
)

print(f"任務配置: {task_config.task_name}, 超時: {task_config.timeout_seconds}s")
```

**Go 範例**:
```go
package main

import (
    "encoding/json"
    "fmt"
    "github.com/aiva/schemas"
)

func main() {
    // 創建異步任務配置
    taskConfig := schemas.AsyncTaskConfig{
        TaskName:       "security_scan",
        TimeoutSeconds: 300,
        RetryConfig: schemas.RetryConfig{
            MaxAttempts:        3,
            BackoffBase:        2.0,
            ExponentialBackoff: true,
        },
        Priority: 8,
        Tags:     []string{"security", "async"},
        Metadata: map[string]interface{}{
            "scan_type": "comprehensive",
        },
    }
    
    jsonData, _ := json.Marshal(taskConfig)
    fmt.Printf("任務配置: %s\n", string(jsonData))
}
```

**Rust 範例**:
```rust
use crate::schemas::{AsyncTaskConfig, RetryConfig};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 創建異步任務配置
    let mut metadata = HashMap::new();
    metadata.insert("scan_type".to_string(), serde_json::Value::String("comprehensive".to_string()));
    
    let task_config = AsyncTaskConfig {
        task_name: "security_scan".to_string(),
        timeout_seconds: 300,
        retry_config: RetryConfig {
            max_attempts: 3,
            backoff_base: 2.0,
            exponential_backoff: true,
            ..Default::default()
        },
        priority: 8,
        tags: vec!["security".to_string(), "async".to_string()],
        metadata,
        ..Default::default()
    };
    
    let json = serde_json::to_string_pretty(&task_config)?;
    println!("任務配置: {}", json);
    Ok(())
}
```

### 🔌 插件管理模組 (plugins)

#### PluginManifest 使用範例

**Python 範例**:
```python
from services.aiva_common.schemas.plugins import PluginManifest, PluginType

# 創建插件清單
plugin_manifest = PluginManifest(
    plugin_id="sql-injection-scanner",
    name="SQL Injection Scanner",
    version="1.2.0",
    author="AIVA Security Team",
    description="Advanced SQL injection vulnerability detection plugin",
    plugin_type=PluginType.SCANNER,
    dependencies=["base-scanner", "db-connector"],
    permissions=["network_access", "read_data"],
    min_aiva_version="2.0.0",
    entry_point="scanner.main:SQLIScanner",
    homepage="https://github.com/aiva/plugins/sqli-scanner",
    keywords=["security", "sql", "injection", "scanner"]
)

print(f"插件: {plugin_manifest.name} v{plugin_manifest.version}")
print(f"類型: {plugin_manifest.plugin_type}")
```

### 🖥️ CLI 界面模組 (cli)

#### CLICommand 使用範例

**Python 範例**:
```python
from services.aiva_common.schemas.cli import CLICommand, CLIParameter

# 創建 CLI 命令定義
scan_command = CLICommand(
    command_name="scan",
    description="執行安全掃描",
    category="security",
    parameters=[
        CLIParameter(
            name="target",
            type="string",
            description="掃描目標 URL",
            required=True,
            help_text="要掃描的目標網站 URL"
        ),
        CLIParameter(
            name="depth",
            type="integer",
            description="掃描深度",
            required=False,
            default_value=3,
            min_value=1,
            max_value=10
        ),
        CLIParameter(
            name="output-format",
            type="choice",
            description="輸出格式",
            required=False,
            default_value="json",
            choices=["json", "xml", "csv", "html"]
        )
    ],
    examples=[
        "aiva scan --target https://example.com",
        "aiva scan --target https://example.com --depth 5 --output-format html"
    ],
    requires_auth=True,
    permissions=["scan", "network_access"]
)

print(f"命令: {scan_command.command_name}")
print(f"參數數量: {len(scan_command.parameters)}")
```

## 🛠️ 開發工具

### 1. Schema 代碼生成器

```bash
# 生成所有語言的 Schema
python services/aiva_common/tools/schema_codegen_tool.py --generate-all

# 只生成特定語言
python services/aiva_common/tools/schema_codegen_tool.py --lang python
python services/aiva_common/tools/schema_codegen_tool.py --lang go  
python services/aiva_common/tools/schema_codegen_tool.py --lang rust
```

### 2. 跨語言一致性驗證

```bash
# 執行完整驗證
python test_cross_language_validation.py

# 檢查驗證報告
cat cross_language_validation_report.json
```

### 3. AI 組件接口測試

```bash
# 測試 AI 組件功能
python services/aiva_common/tools/cross_language_interface.py
```

## 🔍 AI 組件操作指南

### AI 如何使用跨語言 Schema

#### 1. 理解 Schema 結構

```python
# AI 可以使用此接口理解所有 Schema
interface = CrossLanguageSchemaInterface()

# 獲取完整的 AI 友好信息
all_info = interface.get_ai_friendly_schema_info()

# 輸出包含:
# - 總 Schema 數量和分類統計
# - 每個 Schema 的詳細字段信息
# - 三種語言的類型映射
# - 完整代碼生成示例
```

#### 2. 動態代碼生成

```python
# AI 可以為任何 Schema 生成任意語言的代碼
def generate_code_for_ai(schema_name: str, target_language: str) -> str:
    interface = CrossLanguageSchemaInterface()
    return interface.generate_schema_code(schema_name, target_language)

# 示例：AI 生成插件配置的 Go 代碼
go_code = generate_code_for_ai("PluginConfig", "go")
```

#### 3. 類型轉換助手

```python
# AI 可以理解不同語言間的類型對應關係
def convert_types_for_ai(source_type: str) -> Dict[str, str]:
    interface = CrossLanguageSchemaInterface()
    return {
        "python": interface.convert_type_to_language(source_type, "python"),
        "go": interface.convert_type_to_language(source_type, "go"),
        "rust": interface.convert_type_to_language(source_type, "rust")
    }

# 示例：AI 了解 Optional[str] 在各語言中的表示
type_mappings = convert_types_for_ai("Optional[str]")
# 結果: {"python": "Optional[str]", "go": "*string", "rust": "Option<String>"}
```

## 🔧 擴展和自定義

### 添加新的 Schema

1. **更新 YAML SOT**:
   ```yaml
   # 在 core_schema_sot.yaml 中添加新分類
   my_new_category:
     NewSchema:
       description: "新 Schema 描述"
       fields:
         field_name:
           type: str
           required: true
           description: "字段描述"
   ```

2. **重新生成代碼**:
   ```bash
   python services/aiva_common/tools/schema_codegen_tool.py --generate-all
   ```

3. **驗證一致性**:
   ```bash
   python test_cross_language_validation.py
   ```

### 添加新的語言支援

1. **更新生成配置**:
   ```yaml
   generation_config:
     new_language:
       target_dir: "path/to/generated/schemas"
       base_imports:
         - "import statements"
       field_mapping:
         str: "native_string_type"
         int: "native_int_type"
   ```

2. **實現生成器**:
   ```python
   def generate_new_language_schemas(self) -> List[str]:
       # 實現新語言的代碼生成邏輯
       pass
   ```

## 🚨 常見問題和故障排除

### Q1: 代碼生成失敗
**A1**: 檢查環境變數設置和 YAML 文件格式
```bash
export AIVA_RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
export AIVA_POSTGRES_URL="postgresql://user:pass@localhost:5432/aiva"
```

### Q2: 跨語言類型不一致
**A2**: 運行一致性驗證找出具體問題
```bash
python test_cross_language_validation.py
```

### Q3: AI 組件無法理解 Schema  
**A3**: 使用 AI 友好接口獲取結構化信息
```python
info = interface.get_ai_friendly_schema_info("SchemaName")
```

### Q4: 新增 Schema 後編譯錯誤
**A4**: 確保更新了所有語言的代碼生成配置

## 📈 性能和最佳實踐

### 1. Schema 設計原則
- ✅ 使用清晰描述性的字段名稱
- ✅ 提供完整的字段描述和驗證規則
- ✅ 保持跨語言類型的一致性
- ✅ 合理使用可選字段和默認值

### 2. 代碼生成最佳實踐
- 🔄 定期重新生成所有語言代碼
- 🔍 使用一致性驗證工具檢查問題
- 📝 維護清晰的 Schema 文檔
- 🚀 自動化構建和測試流程

### 3. AI 組件使用建議
- 🤖 使用結構化接口而非直接解析
- 📊 利用統計信息了解 Schema 分布
- 🔧 使用代碼生成功能而非手動編寫
- ✅ 定期驗證 AI 組件的理解準確性

## 📚 相關資源

- **Schema 定義文件**: `services/aiva_common/core_schema_sot.yaml`
- **生成工具**: `services/aiva_common/tools/schema_codegen_tool.py`
- **AI 接口**: `services/aiva_common/tools/cross_language_interface.py`
- **驗證工具**: `services/aiva_common/tools/cross_language_validator.py`
- **Python Schema**: `services/aiva_common/schemas/generated/`
- **Go Schema**: `services/features/common/go/aiva_common_go/schemas/generated/`
- **Rust Schema**: `services/scan/info_gatherer_rust/src/schemas/generated/`

## 🎯 總結

AIVA 跨語言 Schema 系統提供了完整的多語言數據模型解決方案，支援：

- **54 個** 統一 Schema 定義
- **3 種** 程式語言自動生成
- **AI 友好** 的智能操作接口
- **自動化** 的一致性驗證
- **完整** 的開發工具鏈

通過這個系統，AI 組件可以無縫理解和操作不同程式語言的數據結構，實現真正的跨語言統一架構。