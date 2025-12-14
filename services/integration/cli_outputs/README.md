# CLI Outputs Directory

此目錄用於存放各語言 CLI 工具生成的輸出文件。

## 目錄結構

```
cli_outputs/
├── python/           # Python 工具輸出
│   ├── CLI_COMMANDS_REFERENCE.md
│   ├── cli_commands_db.json
│   └── classification_data.json
├── typescript/       # TypeScript 工具輸出
│   ├── CLI_COMMANDS_REFERENCE.md
│   ├── classification.json
│   └── *.mmd (Mermaid 圖)
├── go/              # Go 工具輸出
│   ├── CLI_COMMANDS_REFERENCE.md
│   ├── classification.json
│   └── *.mmd
└── rust/            # Rust 工具輸出
    ├── CLI_COMMANDS_REFERENCE.md
    ├── classification.json
    └── *.mmd
```

## 使用方式

所有 CLI 工具的輸出都會自動重定向到此目錄的相應語言子目錄中。

### 配置輸出路徑

在 `cli_tools_config.json` 中配置輸出路徑：

```json
{
  "output_config": {
    "base_dir": "services/integration/cli_outputs",
    "language_subdirs": true
  }
}
```

### 訪問輸出文件

通過 CLI Registry 訪問：

```python
from services.integration.capability.cli_registry import get_cli_registry

registry = get_cli_registry()

# 獲取 Python 工具的 CLI 參考手冊路徑
python_cli_ref = registry.get_output_path("python", "CLI_COMMANDS_REFERENCE.md")
```

## 文件說明

### CLI_COMMANDS_REFERENCE.md
Markdown 格式的 CLI 指令參考手冊，適合人類閱讀。

### cli_commands_db.json / classification.json
JSON 格式的分類數據，供 AI 系統檢索和使用。

### *.mmd
Mermaid 格式的流程圖文件，可視化代碼結構和數據流。

## 自動清理

舊的輸出文件會定期清理，只保留最新的版本。清理策略：
- 保留最近 7 天的輸出
- 保留最新的 5 個版本
- 手動標記為重要的文件不會被清理
