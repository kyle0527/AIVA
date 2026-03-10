# 🔌 AIVA 插件與工具指南

> 最後更新: 2025-12

本指南記錄 AIVA 專案中所有可用的插件、擴充功能、開發工具及依賴項目，協助開發者快速了解可用資源。

---

## 📑 目錄

- [🔹 VS Code 擴充功能](#vs-code-擴充功能)
- [🐍 Python 套件與工具](#python-套件與工具)
- [🔧 自定義 AIVA 插件](#自定義-aiva-插件)
- [🔨 多語言開發工具](#多語言開發工具)
- [📦 專案配置工具](#專案配置工具)

---

## 🔹 VS Code 擴充功能

### ✨ AI & 程式碼協助

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `openai.chatgpt` | 0.4.26 | OpenAI ChatGPT 整合 |
| `visualstudioexptteam.vscodeintellicode` | 1.3.2 | IntelliCode 智能建議 |
| `visualstudioexptteam.intellicode-api-usage-examples` | 0.2.9 | API 使用範例 |
| `sourcery.sourcery` | 1.39.0 | Python 程式碼自動優化 |

### 🐍 Python 開發

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `ms-python.python` | 2025.20.0 | Python 官方擴充套件 |
| `ms-python.vscode-pylance` | 2025.10.4 | Pylance 語言伺服器 |
| `ms-python.debugpy` | 2025.16.0 | Python 除錯工具 |
| `ms-python.autopep8` | 2025.2.0 | PEP8 自動格式化 |
| `ms-python.black-formatter` | 2025.2.0 | Black 格式化工具 |
| `ms-python.isort` | 2025.0.0 | 導入排序工具 |
| `ms-python.vscode-python-envs` | 1.12.0 | Python 環境管理 |
| `charliermarsh.ruff` | 2025.28.0 | Ruff 快速檢查工具 |
| `njpwerner.autodocstring` | 0.6.1 | 自動生成文檔字串 |
| `njqdev.vscode-python-typehint` | 1.5.1 | 類型提示支援 |
| `kevinrose.vsc-python-indent` | 1.21.0 | Python 縮排增強 |
| `cstrap.python-snippets` | 0.1.2 | Python 程式碼片段 |
| `kaih2o.python-resource-monitor` | 0.3.0 | 資源監控工具 |
| `demystifying-javascript.python-extensions-pack` | 1.0.3 | Python 擴充包 |
| `xirider.livecode` | 1.3.10 | 即時程式碼執行 |
| `almenon.arepl` | 3.0.0 | 即時 Python REPL |
| `wbd2023.vscode-pylance-workspace-folder-scope` | 0.2.1 | Pylance 工作區範圍 |

### 🧪 測試工具

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `ms-toolsai.jupyter` | 2025.9.1 | Jupyter Notebook 支援 |
| `ms-toolsai.jupyter-keymap` | 1.1.2 | Jupyter 鍵盤映射 |
| `ms-toolsai.jupyter-renderers` | 1.3.0 | Jupyter 渲染器 |
| `ms-toolsai.vscode-jupyter-cell-tags` | 0.1.9 | Jupyter 儲存格標籤 |
| `ms-toolsai.vscode-jupyter-slideshow` | 0.1.6 | Jupyter 投影片 |
| `hbenl.vscode-test-explorer` | 2.22.1 | 測試總管 |
| `littlefoxteam.vscode-python-test-adapter` | 0.8.2 | Python 測試適配器 |
| `ms-vscode.test-adapter-converter` | 0.2.1 | 測試適配器轉換器 |

### 🦀 Rust 開發

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `rust-lang.rust-analyzer` | 0.3.2711 | Rust 語言伺服器 |
| `tamasfe.even-better-toml` | 0.21.2 | TOML 語法支援 |

### 🐹 Go 開發

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `golang.go` | 0.50.0 | Go 官方擴充套件 |

### 🟦 TypeScript/JavaScript 開發

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `dbaeumer.vscode-eslint` | 3.0.16 | ESLint 檢查工具 |
| `esbenp.prettier-vscode` | 11.0.0 | Prettier 格式化工具 |
| `stylelint.vscode-stylelint` | 1.5.3 | CSS/SCSS 檢查工具 |

### 🐳 容器與遠端開發

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `ms-azuretools.vscode-docker` | 2.0.0 | Docker 支援 |
| `ms-azuretools.vscode-containers` | 2.2.0 | 容器開發 |
| `ms-vscode-remote.remote-containers` | 0.434.0 | 遠端容器 |
| `ms-vscode-remote.remote-ssh` | 0.122.0 | SSH 遠端連線 |
| `ms-vscode-remote.remote-ssh-edit` | 0.87.0 | SSH 配置編輯 |
| `ms-vscode.remote-explorer` | 0.5.0 | 遠端總管 |
| `ms-vscode.remote-server` | 1.5.3 | 遠端伺服器 |
| `github.codespaces` | 1.18.4 | GitHub Codespaces |

### 🗄️ 資料庫工具

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `cweijan.vscode-redis-client` | 8.4.2 | Redis 客戶端 |
| `mtxr.sqltools` | 0.28.5 | SQL 工具 |
| `mtxr.sqltools-driver-pg` | 0.5.7 | PostgreSQL 驅動 |
| `cweijan.dbclient-jdbc` | 1.4.6 | JDBC 資料庫客戶端 |

### 📝 文檔與標記

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `yzhang.markdown-all-in-one` | 3.6.3 | Markdown 增強功能 |
| `davidanson.vscode-markdownlint` | 0.60.0 | Markdown 檢查工具 |
| `bierner.markdown-mermaid` | 1.29.0 | Mermaid 圖表預覽 |
| `bpruitt-goddard.mermaid-markdown-syntax-highlighting` | 1.7.5 | Mermaid 語法高亮 |
| `mermaidchart.vscode-mermaid-chart` | 2.5.6 | Mermaid 圖表編輯器 |
| `chrischinchilla.vscode-pandoc` | 0.6.2 | Pandoc 轉換工具 |
| `mintlify.document` | 2.2.2 | 自動文檔生成 |
| `tomoki1207.pdf` | 1.2.2 | PDF 檢視器 |

### 🌐 Web 開發

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `batisteo.vscode-django` | 1.15.0 | Django 開發支援 |
| `thebarkman.vscode-djaneiro` | 1.4.2 | Django 程式碼片段 |
| `humao.rest-client` | 0.25.1 | REST API 客戶端 |
| `ms-edgedevtools.vscode-edge-devtools` | 2.1.10 | Edge 開發者工具 |

### 🎨 Git 工具

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `eamodio.gitlens` | 17.6.2 | GitLens - Git 超級增強 |
| `donjayamanne.githistory` | 0.6.20 | Git 歷史記錄 |
| `mhutchie.git-graph` | 1.30.0 | Git 圖形化介面 |
| `donjayamanne.git-extension-pack` | 0.1.3 | Git 擴充包 |
| `github.vscode-pull-request-github` | 0.124.0 | GitHub Pull Request |
| `ziyasal.vscode-open-in-github` | 1.3.6 | 在 GitHub 中開啟 |

### 🎯 品質與除錯

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `sonarsource.sonarlint-vscode` | 4.37.0 | SonarLint 程式碼品質檢查 |
| `usernamehw.errorlens` | 3.26.0 | 錯誤顯示增強 |
| `streetsidesoftware.code-spell-checker` | 4.2.6 | 拼字檢查 |
| `aaron-bond.better-comments` | 3.0.2 | 註解增強 |
| `gruntfuggly.todo-tree` | 0.0.226 | TODO 樹狀檢視 |

### 🛠️ 專案管理與實用工具

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `alefragnani.project-manager` | 12.8.0 | 專案管理器 |
| `fill-labs.dependi` | 0.7.15 | 依賴管理工具 |
| `formulahendry.code-runner` | 0.12.2 | 程式碼執行器 |
| `codezombiech.gitignore` | 0.10.0 | .gitignore 生成器 |
| `christian-kohler.path-intellisense` | 2.10.0 | 路徑智能提示 |
| `mechatroner.rainbow-csv` | 3.23.0 | CSV 檔案彩色顯示 |
| `oderwat.indent-rainbow` | 8.3.1 | 縮排彩虹色 |
| `redhat.vscode-yaml` | 1.19.1 | YAML 支援 |

### 🌍 語言與主題

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `ms-ceintl.vscode-language-pack-zh-hant` | 1.107.2025121009 | 繁體中文語言包 |
| `ms-vscode.vscode-speech` | 0.16.0 | 語音支援 |
| `ms-vscode.vscode-speech-language-pack-zh-tw` | 0.5.0 | 中文語音包 |
| `pkief.material-icon-theme` | 5.29.0 | Material 圖示主題 |
| `vscode-icons-team.vscode-icons` | 12.15.0 | vscode-icons 圖示主題 |

### 🔧 其他工具

| 擴充功能 | 版本 | 說明 |
|---------|------|------|
| `ms-vscode.powershell` | 2025.4.0 | PowerShell 支援 |
| `deerawan.vscode-dash` | 2.4.0 | Dash 文檔檢視 |
| `trabpukcip.wolf` | 0.4.3 | 程式碼安全掃描 |

**總計**: 88 個擴充功能

---

## 🐍 Python 套件與工具

### 🤖 核心 AI & 機器學習

| 套件 | 版本 | 說明 |
|------|------|------|
| `torch` | >=2.1.0 | PyTorch 深度學習框架 |
| `torchvision` | >=0.16.0 | 電腦視覺工具 |
| `transformers` | >=4.30.0 | Hugging Face 轉換器 |
| `sentence-transformers` | >=2.2.0 | 語義文本嵌入 |
| `scikit-learn` | >=1.3.0 | 機器學習工具集 |
| `numpy` | >=1.24.0 | 數值計算 |
| `scipy` | >=1.11.0 | 科學計算 |

### 🧠 多代理 & RAG 系統

| 套件 | 版本 | 說明 |
|------|------|------|
| `langchain` | >=0.1.0 | LangChain 框架 |
| `chromadb` | >=0.4.0 | 向量資料庫 |
| `openai` | >=1.0.0 | OpenAI API 客戶端 |
| `nltk` | >=3.8.0 | 自然語言處理工具包 |
| `spacy` | >=3.6.0 | 工業級 NLP |

### ⚡ Web 框架 & API

| 套件 | 版本 | 說明 |
|------|------|------|
| `fastapi` | >=0.115.0 | 現代化 Web 框架 |
| `uvicorn` | >=0.30.0 | ASGI 伺服器 |
| `pydantic` | >=2.7.0 | 資料驗證 |
| `websockets` | >=12.0 | WebSocket 支援 |

### 📡 訊息佇列 & 任務處理

| 套件 | 版本 | 說明 |
|------|------|------|
| `aio-pika` | >=9.4.0 | 非同步 RabbitMQ 客戶端 |
| `celery` | >=5.3.0 | 分散式任務佇列 |
| `kombu` | >=5.3.0 | 訊息傳遞庫 |

### 🌐 HTTP 客戶端

| 套件 | 版本 | 說明 |
|------|------|------|
| `httpx` | >=0.27.0 | 非同步 HTTP 客戶端 |
| `requests` | >=2.31.0 | 同步 HTTP 客戶端 |
| `aiohttp` | >=3.8.0 | 非同步 HTTP 框架 |
| `beautifulsoup4` | >=4.12.2 | HTML 解析工具 |
| `lxml` | >=5.0.0 | XML 處理工具 |

### 🗄️ 資料庫

| 套件 | 版本 | 說明 |
|------|------|------|
| `redis` | >=5.0.0 | Redis 客戶端 |
| `neo4j` | >=5.23.0 | Neo4j 圖資料庫 |
| `sqlalchemy` | >=2.0.31 | SQL ORM |
| `asyncpg` | >=0.29.0 | 非同步 PostgreSQL |
| `psycopg2-binary` | >=2.9.0 | PostgreSQL 適配器 |
| `alembic` | >=1.13.2 | 資料庫遷移工具 |

### 🔒 安全性 & 認證

| 套件 | 版本 | 說明 |
|------|------|------|
| `PyJWT` | >=2.8.0 | JWT 處理 |
| `python-jose[cryptography]` | >=3.3.0 | JWT 加密 |
| `passlib[bcrypt]` | >=1.7.4 | 密碼雜湊 |
| `cryptography` | >=42.0.0 | 加密操作 |
| `python-multipart` | >=0.0.6 | 檔案上傳支援 |

### 📊 資料處理

| 套件 | 版本 | 說明 |
|------|------|------|
| `pandas` | >=2.0.0 | 資料分析 |
| `aiofiles` | >=23.2.1 | 非同步檔案操作 |
| `orjson` | >=3.10.0 | 高效 JSON 處理 |
| `toml` | >=0.10.2 | TOML 解析器 |
| `PyYAML` | >=6.0 | YAML 解析器 |

### 📝 日誌與監控

| 套件 | 版本 | 說明 |
|------|------|------|
| `structlog` | >=24.1.0 | 結構化日誌 |
| `rich` | >=13.0.0 | 豐富的終端輸出 |
| `click` | >=8.1.0 | 命令列介面 |
| `prometheus-client` | >=0.17.0 | 指標收集 |
| `psutil` | >=5.9.6 | 系統監控 |

### 🔄 跨語言通訊

| 套件 | 版本 | 說明 |
|------|------|------|
| `grpcio` | >=1.60.0 | gRPC 框架 |
| `grpcio-tools` | >=1.60.0 | gRPC 工具 |
| `protobuf` | >=4.25.0 | Protocol Buffers |

### 🎯 韌性與實用工具

| 套件 | 版本 | 說明 |
|------|------|------|
| `tenacity` | >=8.3.0 | 重試與韌性模式 |
| `python-dotenv` | >=1.0.1 | 環境變數管理 |

### 🎮 強化學習

| 套件 | 版本 | 說明 |
|------|------|------|
| `gymnasium` | >=0.29.0 | RL 環境介面 |

### 🛠️ 開發工具

| 套件 | 版本 | 說明 |
|------|------|------|
| `pytest` | >=8.0.0 | 測試框架 |
| `pytest-cov` | >=4.0.0 | 測試覆蓋率 |
| `pytest-asyncio` | >=0.23.0 | 非同步測試支援 |
| `black` | >=24.0.0 | 程式碼格式化 |
| `ruff` | >=0.3.0 | 快速檢查工具 |
| `mypy` | >=1.8.0 | 類型檢查 |
| `pre-commit` | >=3.6.0 | Git pre-commit 鉤子 |
| `types-requests` | >=2.31.0 | requests 類型存根 |

**總計**: 70+ 個 Python 套件

---

## 🔧 自定義 AIVA 插件

### 1. aiva-contracts-tooling

**位置**: `tools/integration/aiva-contracts-tooling/`

**功能**: JSON Schema 和 TypeScript 類型生成工具
- 從 `aiva_schemas_plugin` 自動匯出 JSON Schema
- 生成 TypeScript `.d.ts` 類型定義
- 支援 CLI 操作和 CI/CD 整合

**主要命令**:
```bash
# 列出所有模型
aiva-contracts list-models

# 匯出 JSON Schema
aiva-contracts export-jsonschema --out ./schemas/aiva_schemas.json

# 生成 TypeScript 定義
aiva-contracts gen-ts --json ./schemas/aiva_schemas.json --out ./schemas/aiva_schemas.d.ts
```

**應用場景**:
- 前後端類型同步
- API 契約定義
- 多語言類型轉換

---

### 2. aiva-enums-plugin

**位置**: `tools/integration/aiva-enums-plugin/`

**功能**: 集中管理和導出枚舉類型
- Python 端：轉接 `aiva_common.enums`
- TypeScript 端：生成 `enums.ts` 檔案
- 統一的枚舉管理入口

**主要功能**:
```bash
# 生成 TypeScript 枚舉
python scripts/gen_ts_enums.py --out ./schemas/enums.ts
```

**應用場景**:
- 狀態碼統一管理
- 多語言枚舉同步
- 類型安全增強

---

### 3. aiva-schemas-plugin

**位置**: `tools/integration/aiva-schemas-plugin/`

**功能**: 統一的 Schema 插件系統
- 轉接層：re-export `aiva_common.schemas` 
- 批量重構：統一導入路徑
- 清理工具：移除重複的 schemas.py

**重構工具**:
```bash
# 批量改寫匯入並清理檔案
python scripts/refactor_imports_and_cleanup.py --repo-root ./services

# 複製到自含插件
python scripts/copy_into_plugin.py --repo-root ./services
```

**應用場景**:
- Schema 集中管理
- 導入路徑統一
- 重構輔助工具

---

### 4. aiva-go-plugin

**位置**: `tools/integration/aiva-go-plugin/`

**功能**: Go 語言結構體生成
- 從 Python schemas 生成 Go 結構體
- 支援類型映射和標記生成
- Go FFI 整合支援

**應用場景**:
- Python-Go 類型同步
- 跨語言資料交換
- FFI 介面生成

---

### 🎯 整合開發流程

1. **修改 Python schemas** → 運行 `contracts-tooling`
2. **更新枚舉定義** → 運行 `enums-plugin`  
3. **重構 schema 結構** → 運行 `schemas-plugin`
4. **需要 Go 整合** → 運行 `go-plugin`

---

## 🔨 多語言開發工具

### 🦀 Rust 工具鏈

**配置**: `Cargo.toml`

**依賴**:
```toml
[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.35", features = ["full"] }
uuid = { version = "1.6", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

**工作區成員**:
- `services/features/function_crypto/rust_core` - 加密功能核心
- `tools/common/development` - 開發工具

**應用**:
- 高性能加密運算
- 系統層級操作
- 記憶體安全要求高的模組

---

### 🐹 Go 工具鏈

**模組管理**: 10 個 `go.mod` 檔案

**主要模組**:
1. `services/features/function_authn_go` - Go 認證功能
2. `services/scan/engines/go_engine/internal/ssrf` - SSRF 掃描
3. `services/scan/engines/go_engine/internal/sca` - SCA 分析
4. `services/scan/engines/go_engine/internal/cspm` - CSPM 檢查
5. `services/features/common/go/aiva_common_go` - Go 共用庫

**核心依賴**:
```go
require github.com/rabbitmq/amqp091-go v1.10.0
```

**應用**:
- 高並發掃描引擎
- 訊息佇列客戶端
- 雲安全掃描工具

---

### 🟦 TypeScript/Node.js 工具

**配置**: 5 個 `package.json` 檔案

#### ts2mermaid 工具
**位置**: `tools/common/development/package.json`

**功能**: TypeScript AST 解析與 Mermaid 流程圖產生

**依賴**:
```json
{
  "typescript": "^5.3.3",
  "@types/node": "^20.10.6",
  "ts-node": "^10.9.2"
}
```

**應用**:
- 程式碼結構視覺化
- 流程圖自動生成
- 架構文檔輔助

---

## 📦 專案配置工具

### 🔧 Python 建置系統

**配置**: `pyproject.toml`

#### Build System
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"
```

#### Tool Configurations

**Black** (程式碼格式化):
```toml
[tool.black]
line-length = 88
target-version = ['py313']
include = '\.pyi?$'
extend-exclude = '''
/(
  # 排除目錄
  \.eggs
  | \.git
  | \.venv
  | _build
  | build
  | dist
  | node_modules
)/
'''
```

**Ruff** (快速檢查):
```toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501", "B008", "N802", "N803", "N806"]
```

**Mypy** (類型檢查):
```toml
[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Pytest** (測試框架):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"
asyncio_mode = "auto"
```

---

### ⚙️ VS Code 工作區設定

**配置**: `.vscode/settings.json`

#### Python/Pylance 最佳化
```json
{
  "python.analysis.diagnosticMode": "openFilesOnly",
  "python.analysis.backgroundAnalysis": "off",
  "python.analysis.indexing": false,
  "python.analysis.autoImportCompletions": false
}
```

#### TypeScript 驗證
```json
{
  "typescript.validate.enable": true,
  "javascript.validate.enable": true
}
```

#### Rust Analyzer
```json
{
  "rust-analyzer.checkOnSave": false,
  "rust-analyzer.diagnostics.enable": false
}
```

#### Go (gopls)
```json
{
  "go.lintOnSave": "off",
  "gopls": {
    "diagnosticsTrigger": "Save",
    "ui.diagnostic.analyses": {
      "unusedparams": false,
      "unusedvariable": false
    }
  }
}
```

---

## 🔗 相關資源

### 📚 開發指南
- [開發環境總覽](./README.md)
- [多語言環境標準](./MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md)
- [工具集使用手冊](../../tools/README.md)

### 🏗️ 架構文檔
- [AIVA Common 套件文檔](../../services/aiva_common/README.md)
- [Features 模組文檔](../../services/features/README.md)
- [簡化架構文檔](../../services/features/SIMPLE_ARCHITECTURE.md)

### 🔌 插件開發
- [插件整合指南](../../tools/integration/README.md)
- [Schema 管理最佳實踐](../architecture/SCHEMA_GUIDE.md)

---

## 📋 版本資訊

- **Python**: 3.13+
- **Node.js**: 20+
- **Go**: 1.21+
- **Rust**: 1.70+
- **AIVA Core**: v6.1

---

## 🎯 使用建議

### 新手入門順序
1. ✅ 安裝 Python 擴充功能包 (8 個核心工具)
2. ✅ 配置 Python 環境 (`requirements.txt`)
3. ✅ 啟用 Pylance 與 Ruff
4. ✅ 安裝 Git 工具組 (6 個擴充)
5. ✅ 配置 Docker 與容器工具
6. ✅ 根據需要安裝特定語言工具

### 效能最佳化建議
- ⚡ 使用 `.vscode/settings.json` 的最佳化配置
- ⚡ 關閉不需要的語言伺服器背景分析
- ⚡ 排除大型目錄 (`logs/`, `models/`, `_out/`)
- ⚡ 僅在需要時啟用格式化工具

### CI/CD 整合
- 🔄 在 PR 中自動運行 `contracts-tooling`
- 🔄 檢查 TypeScript 定義是否最新
- 🔄 驗證多語言類型一致性
- 🔄 運行 `pytest` 與 `ruff` 檢查

---

**維護者**: AIVA Team  
**更新週期**: 每月或重大變更時  
**問題回報**: 請至專案 Issue 區塊提出
