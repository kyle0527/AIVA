# CLI 指令整合分析報告

## 📊 現有架構分析

### 1. **已存在的 Integration 模組 CLI 基礎設施**

#### ✅ 已完成部分

1. **cli_registry.py** - CLI 註冊中心 (390行)
   - `CLICommand` 數據類 - 完整的指令定義模型
   - `CLIToolConfig` - 工具配置模型
   - `CLIRegistry` - 註冊中心類
   - 已實現指令註冊、查詢、執行功能

2. **cli_tools_config.json** - CLI 工具配置 (459行)
   - 已定義 Python 工具的完整配置
   - 包含 4 個 Python 指令的詳細定義
   - 標準化的 JSON 結構

3. **cli_outputs/** 目錄結構
   ```
   cli_outputs/
   ├── python/
   ├── typescript/
   ├── go/
   ├── rust/
   └── README.md
   ```

4. **models.py** - CLITemplate 模型
   - 已定義 CLI 模板數據結構
   - 與 cli_registry.py 中的 CLICommand 功能重疊

### 2. **各語言 CLI 工具分析**

#### Python Tool (aiva_cli_implementation.py)

**CLI 參數**:
```bash
--generate-doc {md|json}    # 生成文檔
--list                       # 列出流程
--flow <id>                  # 執行流程
--dry-run                    # 預覽執行
--data <path>                # 指定數據源
```

**輸出文件**:
- `CLI_COMMANDS_REFERENCE.md` - Markdown 手冊
- `cli_commands_db.json` - JSON 資料庫
- `classification_data.json` - 分類數據 (282 flows)

**輸出目錄配置**:
- 已支持環境變數 `AIVA_CLI_OUTPUT_DIR`
- 默認路徑: `services/integration/cli_outputs/python`

#### TypeScript Tool (ts2mermaid.ts)

**CLI 參數** (從代碼推斷):
```bash
--file <path>                # 分析單個文件
--project <dir>              # 分析整個項目
--output <path>              # 輸出文件路徑
--output-dir <dir>           # 輸出目錄
--stitch                     # 跨文件串接
--classify                   # 功能分類
--generate-cli               # 生成 CLI 手冊
--analyze-bottleneck         # 瓶頸分析
```

**輸出文件**:
- `*.mmd` - Mermaid 流程圖
- `CLI_COMMANDS_REFERENCE.md` - CLI 手冊
- `classification_data.json` - 分類數據
- `system_flow.mmd` - 系統架構圖
- `bottleneck_report.json` - 瓶頸報告

#### Go Tool (go2mermaid.go)

**CLI 參數** (從代碼推斷):
```bash
-file <path>                 # 分析單個文件
-dir <path>                  # 分析目錄
-output <path>               # 輸出文件
-output-dir <path>           # 輸出目錄
-stitch                      # 跨文件串接
-classify                    # 功能分類
-gen-cli                     # 生成 CLI 手冊
-stats                       # 統計報告
```

**輸出文件**:
- `*.mmd` - Mermaid 流程圖
- `CLI_COMMANDS_REFERENCE.md` - CLI 手冊
- `classification_data.json` - 分類數據
- `system_flow.mmd` - 系統架構圖
- `stats_report.json` - 統計報告

#### Rust Tool (rs2mermaid / main.rs)

**CLI 參數** (從代碼推斷):
```bash
--file <path>                # 分析單個文件
--input <dir>                # 輸入目錄
--output <dir>               # 輸出目錄
--project <dir>              # 分析項目
--stitch                     # 跨文件串接
--classify                   # 功能分類
--gen-cli                    # 生成 CLI 手冊
--analyze                    # 分析報告
```

**輸出文件**:
- `*.mmd` - Mermaid 流程圖
- `CLI_COMMANDS_REFERENCE.md` - CLI 手冊
- `classification_data.json` - 分類數據
- `system_flow.mmd` - 系統架構圖
- `analysis_report.json` - 分析報告

---

## 🔍 關鍵問題與差異

### 1. **CLI 參數命名不統一**

| 功能 | Python | TypeScript | Go | Rust |
|------|--------|------------|-----|------|
| 單文件分析 | N/A | `--file` | `-file` | `--file` |
| 項目分析 | N/A | `--project` | `-dir` | `--input` |
| 輸出目錄 | 環境變數 | `--output-dir` | `-output-dir` | `--output` |
| 生成 CLI | `--generate-doc` | `--generate-cli` | `-gen-cli` | `--gen-cli` |
| 分類 | N/A | `--classify` | `-classify` | `--classify` |
| 串接 | N/A | `--stitch` | `-stitch` | `--stitch` |

### 2. **輸出文件格式差異**

| 輸出類型 | Python | TS/Go/Rust |
|---------|--------|------------|
| CLI 手冊 | ✅ Markdown | ✅ Markdown |
| CLI 資料庫 | ✅ JSON | ❌ 無 (僅 MD) |
| 分類數據 | ✅ JSON | ✅ JSON |
| 流程圖 | ❌ 無 | ✅ Mermaid |
| 系統架構 | ❌ 無 | ✅ Mermaid |
| 統計報告 | ❌ 無 | ✅ JSON |

### 3. **功能差異**

| 功能 | Python | TypeScript | Go | Rust |
|------|--------|------------|-----|------|
| 流程執行 | ✅ | ❌ | ❌ | ❌ |
| 代碼分析 | ✅ | ✅ | ✅ | ✅ |
| 流程圖生成 | ❌ | ✅ | ✅ | ✅ |
| 跨文件串接 | ❌ | ✅ | ✅ | ✅ |
| 分類統計 | ✅ | ✅ | ✅ | ✅ |
| CLI 生成 | ✅ | ✅ | ✅ | ✅ |
| 瓶頸分析 | ❌ | ✅ | ✅ | ✅ |

---

## 🎯 統一 CLI 接口設計

### 標準化命令格式

為了讓 AI 和 internal_loop_connector.py 能夠統一理解所有語言的 CLI，我們需要定義標準化的命令類別：

#### 1. **分析類命令** (Analysis)
- **功能**: 分析代碼文件或項目
- **統一參數**:
  - `--input <path>` - 輸入文件或目錄
  - `--output-dir <dir>` - 輸出目錄
- **對應**:
  - Python: N/A (自動分析)
  - TS: `--file` / `--project`
  - Go: `-file` / `-dir`
  - Rust: `--file` / `--input`

#### 2. **生成類命令** (Generation)
- **功能**: 生成文檔、報告、圖表
- **統一參數**:
  - `--format <type>` - 輸出格式 (md, json, mermaid)
  - `--output <path>` - 輸出文件路徑
- **對應**:
  - Python: `--generate-doc {md|json}`
  - TS: `--generate-cli`, `--output`
  - Go: `-gen-cli`, `-output`
  - Rust: `--gen-cli`, `--output`

#### 3. **執行類命令** (Execution)
- **功能**: 執行流程或任務
- **統一參數**:
  - `--flow <id>` - 流程 ID
  - `--dry-run` - 預覽模式
- **對應**:
  - Python: `--flow`, `--dry-run`
  - TS/Go/Rust: N/A

#### 4. **分類類命令** (Classification)
- **功能**: 功能分類與統計
- **統一參數**:
  - `--classify` - 執行分類
  - `--output <path>` - 輸出分類結果
- **對應**:
  - Python: 自動執行
  - TS: `--classify`
  - Go: `-classify`
  - Rust: `--classify`

#### 5. **串接類命令** (Stitching)
- **功能**: 跨文件數據流串接
- **統一參數**:
  - `--stitch` - 執行串接
  - `--output <path>` - 輸出系統架構圖
- **對應**:
  - Python: N/A
  - TS: `--stitch`
  - Go: `-stitch`
  - Rust: `--stitch`

---

## 📋 建議的統一 CLI 配置結構

### cli_tools_config.json 完整結構

```json
{
  "version": "2.0.0",
  "description": "AIVA CLI 工具統一配置",
  "output_base_dir": "services/integration/cli_outputs",
  
  "tools": {
    "python": {
      "language": "python",
      "tool_name": "AIVA Python CLI Tool",
      "tool_path": "services/core/aiva_core/internal_exploration/python_tools",
      "executable": "aiva_cli_implementation.py",
      "execution_prefix": "python -m aiva_core.internal_exploration.python_tools.aiva_cli_implementation",
      "version": "1.0.0",
      "requires_compilation": false,
      "setup_commands": [],
      
      "capabilities": {
        "analysis": true,
        "generation": true,
        "execution": true,
        "classification": true,
        "stitching": false,
        "flowchart": false
      },
      
      "commands": {
        "generate_doc_md": {
          "id": "python.generate.doc.md",
          "name": "生成 Markdown 文檔",
          "category": "generation",
          "template": "--generate-doc md",
          "args": {
            "output_dir": {
              "flag": "--output-dir",
              "required": false,
              "default": "${output_base_dir}/python"
            }
          },
          "outputs": ["CLI_COMMANDS_REFERENCE.md"]
        },
        "generate_doc_json": {
          "id": "python.generate.doc.json",
          "name": "生成 JSON 資料庫",
          "category": "generation",
          "template": "--generate-doc json",
          "args": {
            "output_dir": {
              "flag": "--output-dir",
              "required": false,
              "default": "${output_base_dir}/python"
            }
          },
          "outputs": ["cli_commands_db.json"]
        },
        "list_flows": {
          "id": "python.list.flows",
          "name": "列出可用流程",
          "category": "analysis",
          "template": "--list",
          "args": {},
          "outputs": ["stdout"]
        },
        "execute_flow": {
          "id": "python.execute.flow",
          "name": "執行指定流程",
          "category": "execution",
          "template": "--flow {flow_id}",
          "args": {
            "flow_id": {
              "required": true,
              "type": "integer"
            },
            "dry_run": {
              "flag": "--dry-run",
              "required": false,
              "type": "boolean"
            }
          },
          "outputs": ["stdout"]
        }
      }
    },
    
    "typescript": {
      "language": "typescript",
      "tool_name": "AIVA TypeScript CLI Tool",
      "tool_path": "services/core/aiva_core/internal_exploration/typescript_tools",
      "executable": "ts2mermaid.ts",
      "execution_prefix": "npx ts-node",
      "version": "2.0.0",
      "requires_compilation": false,
      "setup_commands": ["npm install"],
      
      "capabilities": {
        "analysis": true,
        "generation": true,
        "execution": false,
        "classification": true,
        "stitching": true,
        "flowchart": true
      },
      
      "commands": {
        "analyze_file": {
          "id": "typescript.analyze.file",
          "name": "分析單個文件",
          "category": "analysis",
          "template": "--file {input_file} --output {output_file}",
          "args": {
            "input_file": {
              "flag": "--file",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/typescript/output.mmd"
            }
          },
          "outputs": ["*.mmd"]
        },
        "analyze_project": {
          "id": "typescript.analyze.project",
          "name": "分析整個項目",
          "category": "analysis",
          "template": "--project {project_dir} --output-dir {output_dir}",
          "args": {
            "project_dir": {
              "flag": "--project",
              "required": true,
              "type": "path"
            },
            "output_dir": {
              "flag": "--output-dir",
              "required": false,
              "default": "${output_base_dir}/typescript"
            }
          },
          "outputs": ["*.mmd", "classification_data.json"]
        },
        "generate_cli": {
          "id": "typescript.generate.cli",
          "name": "生成 CLI 手冊",
          "category": "generation",
          "template": "--generate-cli --output {output_file}",
          "args": {
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/typescript/CLI_COMMANDS_REFERENCE.md"
            }
          },
          "outputs": ["CLI_COMMANDS_REFERENCE.md"]
        },
        "classify": {
          "id": "typescript.classify",
          "name": "功能分類",
          "category": "classification",
          "template": "--classify --project {project_dir} --output {output_file}",
          "args": {
            "project_dir": {
              "flag": "--project",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/typescript/classification_data.json"
            }
          },
          "outputs": ["classification_data.json"]
        },
        "stitch": {
          "id": "typescript.stitch",
          "name": "跨文件串接",
          "category": "stitching",
          "template": "--stitch --project {project_dir} --output {output_file}",
          "args": {
            "project_dir": {
              "flag": "--project",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/typescript/system_flow.mmd"
            }
          },
          "outputs": ["system_flow.mmd"]
        }
      }
    },
    
    "go": {
      "language": "go",
      "tool_name": "AIVA Go CLI Tool",
      "tool_path": "services/core/aiva_core/internal_exploration/go_tools",
      "executable": "go2mermaid",
      "execution_prefix": "./",
      "version": "2.0.0",
      "requires_compilation": true,
      "setup_commands": ["go build go2mermaid.go"],
      
      "capabilities": {
        "analysis": true,
        "generation": true,
        "execution": false,
        "classification": true,
        "stitching": true,
        "flowchart": true
      },
      
      "commands": {
        "analyze_file": {
          "id": "go.analyze.file",
          "name": "分析單個文件",
          "category": "analysis",
          "template": "-file {input_file} -output {output_file}",
          "args": {
            "input_file": {
              "flag": "-file",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "-output",
              "required": false,
              "default": "${output_base_dir}/go/output.mmd"
            }
          },
          "outputs": ["*.mmd"]
        },
        "analyze_project": {
          "id": "go.analyze.project",
          "name": "分析整個項目",
          "category": "analysis",
          "template": "-dir {project_dir} -output-dir {output_dir}",
          "args": {
            "project_dir": {
              "flag": "-dir",
              "required": true,
              "type": "path"
            },
            "output_dir": {
              "flag": "-output-dir",
              "required": false,
              "default": "${output_base_dir}/go"
            }
          },
          "outputs": ["*.mmd", "classification_data.json"]
        },
        "generate_cli": {
          "id": "go.generate.cli",
          "name": "生成 CLI 手冊",
          "category": "generation",
          "template": "-gen-cli -output {output_file}",
          "args": {
            "output_file": {
              "flag": "-output",
              "required": false,
              "default": "${output_base_dir}/go/CLI_COMMANDS_REFERENCE.md"
            }
          },
          "outputs": ["CLI_COMMANDS_REFERENCE.md"]
        },
        "classify": {
          "id": "go.classify",
          "name": "功能分類",
          "category": "classification",
          "template": "-classify -dir {project_dir} -output {output_file}",
          "args": {
            "project_dir": {
              "flag": "-dir",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "-output",
              "required": false,
              "default": "${output_base_dir}/go/classification_data.json"
            }
          },
          "outputs": ["classification_data.json"]
        },
        "stitch": {
          "id": "go.stitch",
          "name": "跨文件串接",
          "category": "stitching",
          "template": "-stitch -dir {project_dir} -output {output_file}",
          "args": {
            "project_dir": {
              "flag": "-dir",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "-output",
              "required": false,
              "default": "${output_base_dir}/go/system_flow.mmd"
            }
          },
          "outputs": ["system_flow.mmd"]
        }
      }
    },
    
    "rust": {
      "language": "rust",
      "tool_name": "AIVA Rust CLI Tool",
      "tool_path": "services/core/aiva_core/internal_exploration/rust_tools",
      "executable": "rs2mermaid",
      "execution_prefix": "./target/release/",
      "version": "2.0.0",
      "requires_compilation": true,
      "setup_commands": ["cargo build --release"],
      
      "capabilities": {
        "analysis": true,
        "generation": true,
        "execution": false,
        "classification": true,
        "stitching": true,
        "flowchart": true
      },
      
      "commands": {
        "analyze_file": {
          "id": "rust.analyze.file",
          "name": "分析單個文件",
          "category": "analysis",
          "template": "--file {input_file} --output {output_file}",
          "args": {
            "input_file": {
              "flag": "--file",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/rust/output.mmd"
            }
          },
          "outputs": ["*.mmd"]
        },
        "analyze_project": {
          "id": "rust.analyze.project",
          "name": "分析整個項目",
          "category": "analysis",
          "template": "--input {project_dir} --output {output_dir}",
          "args": {
            "project_dir": {
              "flag": "--input",
              "required": true,
              "type": "path"
            },
            "output_dir": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/rust"
            }
          },
          "outputs": ["*.mmd", "classification_data.json"]
        },
        "generate_cli": {
          "id": "rust.generate.cli",
          "name": "生成 CLI 手冊",
          "category": "generation",
          "template": "--gen-cli --output {output_file}",
          "args": {
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/rust/CLI_COMMANDS_REFERENCE.md"
            }
          },
          "outputs": ["CLI_COMMANDS_REFERENCE.md"]
        },
        "classify": {
          "id": "rust.classify",
          "name": "功能分類",
          "category": "classification",
          "template": "--classify --input {project_dir} --output {output_file}",
          "args": {
            "project_dir": {
              "flag": "--input",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/rust/classification_data.json"
            }
          },
          "outputs": ["classification_data.json"]
        },
        "stitch": {
          "id": "rust.stitch",
          "name": "跨文件串接",
          "category": "stitching",
          "template": "--stitch --input {project_dir} --output {output_file}",
          "args": {
            "project_dir": {
              "flag": "--input",
              "required": true,
              "type": "path"
            },
            "output_file": {
              "flag": "--output",
              "required": false,
              "default": "${output_base_dir}/rust/system_flow.mmd"
            }
          },
          "outputs": ["system_flow.mmd"]
        }
      }
    }
  }
}
```

---

## 🔧 AI 視角的統一接口

### 為 internal_loop_connector.py 設計的簡化接口

AI 只需要知道以下信息：

```python
# 命令分類 (Category)
CATEGORIES = {
    "analysis": "代碼分析",
    "generation": "文檔生成",
    "execution": "流程執行",
    "classification": "功能分類",
    "stitching": "跨文件串接"
}

# 可用命令 (按語言)
AVAILABLE_COMMANDS = {
    "python": [
        "generate_doc_md",      # 生成 Markdown 文檔
        "generate_doc_json",    # 生成 JSON 資料庫
        "list_flows",           # 列出流程
        "execute_flow"          # 執行流程
    ],
    "typescript": [
        "analyze_file",         # 分析文件
        "analyze_project",      # 分析項目
        "generate_cli",         # 生成 CLI
        "classify",             # 分類
        "stitch"                # 串接
    ],
    "go": [
        "analyze_file",
        "analyze_project",
        "generate_cli",
        "classify",
        "stitch"
    ],
    "rust": [
        "analyze_file",
        "analyze_project",
        "generate_cli",
        "classify",
        "stitch"
    ]
}

# 命令功能映射 (AI 理解層)
COMMAND_PURPOSES = {
    "generate_doc_md": "生成人類可讀的 CLI 指令手冊",
    "generate_doc_json": "生成機器可讀的 CLI 指令資料庫",
    "analyze_file": "分析單個代碼文件並生成流程圖",
    "analyze_project": "分析整個項目並生成多個流程圖",
    "classify": "對代碼功能進行自動分類",
    "stitch": "串接跨文件調用關係生成系統架構圖",
    "execute_flow": "執行預定義的數據流程",
    "list_flows": "列出所有可用的數據流程"
}
```

---

## ✅ 實施建議

### 階段 1: 配置統一 (優先)

1. 完善 `cli_tools_config.json`
   - 添加 TypeScript, Go, Rust 配置
   - 標準化參數命名
   - 定義輸出路徑模板

2. 更新 Python 工具
   - 修改 `aiva_cli_implementation.py` 讀取統一配置
   - 移除硬編碼路徑
   - 支持 `--output-dir` 參數

### 階段 2: 註冊中心增強

1. 擴展 `cli_registry.py`
   - 從配置文件自動加載所有工具
   - 實現命令查詢 API
   - 提供命令執行包裝器

2. 創建統一執行器
   - 根據語言自動選擇執行前綴
   - 處理編譯需求 (Go/Rust)
   - 統一輸出收集

### 階段 3: Internal Loop 整合

1. 更新 `internal_loop_connector.py`
   - 導入 `cli_registry`
   - 使用統一的命令查詢接口
   - 通過命令 ID 而非直接調用

2. 提供 AI 友好的接口
   - 按類別查詢命令
   - 按功能描述搜索
   - 獲取命令使用範例

---

## 📦 目錄結構規劃

```
services/integration/
├── capability/
│   ├── cli_registry.py          # ⭐ 核心註冊中心
│   ├── cli_executor.py          # ⭐ 新增：統一執行器
│   ├── cli_tools_config.json    # ⭐ 完整配置
│   └── models.py                # CLITemplate (考慮合併到 cli_registry)
│
├── cli_outputs/                 # ⭐ 所有語言的輸出
│   ├── python/
│   │   ├── CLI_COMMANDS_REFERENCE.md
│   │   ├── cli_commands_db.json
│   │   └── classification_data.json
│   ├── typescript/
│   │   ├── CLI_COMMANDS_REFERENCE.md
│   │   ├── classification_data.json
│   │   ├── system_flow.mmd
│   │   └── *.mmd
│   ├── go/
│   │   └── (同 typescript)
│   ├── rust/
│   │   └── (同 typescript)
│   └── README.md
│
└── executable_commands/         # 便捷腳本 (可選)
    ├── analyze_python.sh
    ├── analyze_typescript.sh
    └── ...
```
