# 🧭 Internal Exploration - 內部探索

> **路徑**: `internal_exploration/`  
> **狀態**: ✅ 正常 | **最後更新**: 2026-01-13  
> **子模組**: 5 個 (Python, Go, Rust, TypeScript, Self-Healing) | **總文件數**: 20  
> **資料位置**: `services/integration/data/internal_exploration/` | **最新分析**: v15 (Core) + Features分析  
> **4語言分析**: ✅ 已完成 (Go: 4 flows, TS: 3 flows, Python: 150 flows, Rust: 0 flows[正常])

## 概述

**Internal Exploration** 是 AIVA 的自我認知系統，提供**4種語言**的 AST 分析、數據流追蹤、自動化分類和自我修復能力。支援 **Python、Go、Rust、TypeScript** 的完整代碼分析與JSON輸出統一格式。

**核心職責**：
- 🔍 **多語言代碼分析** - 支援 4 種語言的 AST 解析與統一JSON輸出
- 📊 **數據流視覺化** - 自動生成 Mermaid 流程圖
- 🏷️ **智能分類系統** - 六大類別自動分類
- 🔧 **自我修復診斷** - 自動檢測數據流斷點和架構問題
- 📝 **CLI 指令生成** - 自動產生可執行命令腳本
- 🔗 **JSON格式統一** - schema v3.3，相容 FlowExecutor ⭐

---

## 🎯 最新成果 (2026-01-13)

### **Phase 1 完成：多語言JSON格式統一**

✅ **4語言工具改進完成**：
- **TypeScript**: 修改 `ts2mermaid.ts`，添加 `metadata` + `flows` 欄位
- **Go**: 修改 `go2mermaid.go`，添加 `metadata` + `flows` 欄位  
- **Rust**: 修改 `main.rs`，添加 `metadata` + `flows` 欄位
- **Python**: 使用 `aiva_flow_classifier_final.py` 轉換舊格式

✅ **實際功能模組分析**：
| 語言 | 模組 | Flows | 檔案 | 狀態 |
|------|------|-------|------|------|
| **Go** | function_authn_go | 4 | 4 | ✅ 已驗證 |
| **TypeScript** | typescript_engine | 3 | 11 | ✅ 已驗證 |
| **Python** | features_ready | 150 | 119 | ✅ 已完成 |
| **Rust** | function_crypto | 0* | 5 | ✅ 正常(工具集) |

*Rust 0 flows 為正常：crypto模組是4個獨立analyzer工具的集合，無跨檔案調用

✅ **統一JSON Schema v3.3**：
```json
{
  "metadata": {
    "tool": "工具名",
    "version": "2.0",
    "language": "語言",
    "generated_at": "ISO時間",
    "total_flows": 數量,
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [...],  // 新增：統一格式
  "functions": [...],  // 保留：原始數據
  // ... 其他原始欄位全部保留
}
```

---

## ✅ 模組驗證狀態 (2026-01-13)

**完整性檢查**:
- ✅ **無測試文件** - 所有 16 個 Python 文件均為功能代碼（無 test*.py, *mock*.py）
- ✅ **無編譯錯誤** - 全模組編譯通過，無語法錯誤
- ✅ **功能完整** - 所有文件均為核心能力實現
- ✅ **多語言支援** - Python (6), Go (2), Rust (2), TypeScript (2)

**FlowExecutor 核心整合** ⭐:
- ✅ `aiva_cli_implementation.py` - FlowExecutor 類實現 (Line 99-650)
- ✅ `aiva_exploration_pipeline.py` - 動態調用 FlowExecutor 生成文檔 (Line 442-443)
- ✅ `latest_classification.json` - 系統指針，始終指向最新版本 (Line 31, 74, 85)

**核心模組整合**:
- ✅ `core_capabilities.cli.aiva_cli` 導入 FlowExecutor (Line 112, 198)
- ✅ `cognitive_core.internal_loop_connector` 導入 ExplorationPipeline, FlowExecutor (Line 565, 823)
- ✅ `core_capabilities.capability_registry` 從 internal_exploration 加載能力 (Line 148)
- ✅ `task_planning.dispatcher` 請求 internal_exploration 分析 (多個調用點)

**資料存放位置** (2026-01-10 更新):
- ✅ **正式路徑**: `services/integration/data/internal_exploration/`
- ✅ **最新版本**: v6 (2026-01-10 09:38)
- ✅ **latest_classification.json**: 659KB
- ✅ **CLI 指令文檔**: `analysis_history/v6/CLI_COMMANDS_REFERENCE.md`

---

## 架構

### 子模組結構

| 子模組 | 功能 | 文件數 | 文檔 |
|--------|------|--------|------|
| python_tools/ | Python AST 分析、FlowExecutor、CLI 生成 ⭐ | 6 | [README](python_tools/README.md) |
| self_healing/ | 數據流斷點檢測、缺失連接分析、智能過濾 | 8 | [README](self_healing/README.md) |
| go_tools/ | Go AST 分析、go2mermaid | 2 | [README](go_tools/README.md) |
| rust_tools/ | Rust AST 分析、Cargo 集成 | 2 | [README](rust_tools/README.md) |
| typescript_tools/ | TypeScript AST 分析、ts2mermaid | 2 | [README](typescript_tools/README.md) |

### 根目錄組件 (3 個文件)

- `dispatcher.py` - 內部探索發送器，跨模組通信
- `modules_config.json` - 模組配置文件
- `__init__.py` - 模組初始化，導出 CoreAnalyzer

---

## 主要類別

| 類別 | 文件 | 說明 |
|------|------|------|
| **`FlowExecutor`** ⭐ | **python_tools/aiva_cli_implementation.py** | **313-318 個 CLI flows 執行器** |
| `ExplorationDispatcher` | dispatcher.py | 內部探索統一發送器 |
| `AIVAFlowAnalyzer` | python_tools/aiva_flow_analyzer.py | 流程圖生成與智能組合 |
| `AIVAFlowClassifier` | python_tools/aiva_flow_classifier.py | 數據流分類分析 |
| `AIVAExplorationPipeline` | python_tools/aiva_exploration_pipeline.py | 認知更新管線總控 |
| `CoreAnalyzer` | self_healing/core_analyzer.py | 統一診斷入口 |
| `PracticalAnalyzer` | self_healing/practical_analyzer.py | 智能過濾和分級 |

---

## 依賴關係

**外部依賴**：
- `ast` - Python AST 解析
- `json` - 配置和數據文件

**內部依賴**：
- `service_backbone.messaging` - 消息代理
- `cognitive_core.learning_system` - 學習系統
- `service_backbone.storage` - 報告存儲

**被依賴於** ⭐:
- `core_capabilities.cli.aiva_cli` - 導入 FlowExecutor 執行 313-318 個 flows
- `cognitive_core.internal_loop_connector` - 導入 ExplorationPipeline, FlowExecutor
- `core_capabilities.capability_registry` - 從 internal_exploration 加載能力
- `task_planning.dispatcher` - 請求 internal_exploration 進行分析

---

**導航**: [← 返回 AIVA Core](../README.md)

---

## 📑 詳細目錄

- [🎯 模組概述](#-模組概述)
- [🏗️ 目錄結構](#️-目錄結構)
- [🛠️ 核心工具套件](#️-核心工具套件)
- [🚀 快速開始](#-快速開始)
- [📚 詳細文檔](#-詳細文檔)
- [📊 統計資訊](#-統計資訊)

---

## 🎯 模組概述

**AIVA Internal Exploration** 是 AIVA 專案的核心代碼分析與自我診斷系統,提供多語言 AST 分析、數據流追蹤、自動化分類和自我修復能力。

### 主要功能

1. **多語言代碼分析** - 支援 Python、TypeScript、Go、Rust 四種語言的 AST 解析
2. **數據流視覺化** - 自動生成 Mermaid 流程圖,追蹤跨檔案函數調用關係
3. **智能分類系統** - 按功能將代碼分為六大類別(偵察/攻擊/分析/報告/持久化/其他)
4. **自我修復診斷** - 自動檢測數據流斷點、缺失連接和架構問題
5. **CLI 指令生成** - 自動產生可執行的命令腳本和操作手冊

### 應用場景

- **代碼架構分析** - 理解複雜專案的模組依賴和數據流向
- **重構輔助** - 識別高耦合模組和潛在瓶頸點
- **文檔自動化** - 生成函數調用圖和 CLI 操作手冊
- **健康度診斷** - 檢測代碼中的斷點、缺失連接和異常流程

---

## 🏗️ 目錄結構

```
internal_exploration/
├── 📁 python_tools/                        # Python AST 分析工具套件
│   ├── aiva_flow_analyzer.py              # 流程圖生成與智能組合
│   ├── aiva_flow_classifier.py            # 數據流分類分析
│   ├── aiva_cli_implementation.py         # 動態執行與文檔生成
│   ├── aiva_exploration_pipeline.py       # 認知更新管線總控
│   └── README.md                          # 📘 Python 工具詳細文檔
│
│   ⚠️ **重要：統一資料存放路徑 (2026-01-10)**
│   
│   **當前路徑結構**: services/integration/data/internal_exploration/
│   
│   輸出文件：
│   - latest_classification.json (最新分類數據)
│   - analysis_history/v{N}/ (版本歷史記錄)
│     - analysis_results.json
│     - classification_data.json
│     - CLI_COMMANDS_REFERENCE.md
│     - cli_commands_db.json
│     - classification_summary.md
│     - complete_flow_details.md
│     - multi_path_analysis.md
│     - diff_report.md
│   
│   詳見: [Python 工具完整文檔](python_tools/README.md)
│
├── 📁 typescript_tools/                    # TypeScript AST 分析工具
│   ├── ts2mermaid.ts                      # TypeScript 統一分析工具
│   ├── package.json                       # Node.js 依賴配置
│   └── README.md                          # 📘 TypeScript 工具詳細文檔
│
├── 📁 go_tools/                            # Go AST 分析工具
│   ├── go2mermaid.go                      # Go 統一分析工具
│   ├── go.mod                             # Go 模組配置
│   └── README.md                          # 📘 Go 工具詳細文檔
│
├── 📁 rust_tools/                          # Rust AST 分析工具
│   ├── src/main.rs                        # Rust 統一分析工具
│   ├── Cargo.toml                         # Rust 依賴配置
│   └── README.md                          # 📘 Rust 工具詳細文檔
│
├── 📁 self_healing/                        # 自我修復診斷模組 ⭐
│   ├── core_analyzer.py                   # 統一診斷入口
│   ├── analyze_dataflow_breakpoints.py    # 數據流斷點檢測
│   ├── analyze_missing_function_connections.py  # 缺失連接分析
│   ├── practical_analyzer.py              # 智能過濾和分級
│   └── README.md                          # 📘 Self-Healing 詳細文檔
│
└── 📁 analysis_history/                    # 歷史分析結果
    └── v4/                                # 版本 4 分析數據
```

---

## 🛠️ 核心工具套件

### Python 工具套件

**位置**: `python_tools/`  
**主要語言**: Python 3.10+

Python 工具套件是 AIVA Internal Exploration 的核心基礎,提供最完整的分析功能:

- ✅ **四大模組**: 從底層 AST 解析到高階管線編排
- ✅ **智能組合**: 跨檔案數據流自動串接
- ✅ **動態執行**: Pipeline 數據傳遞與流程驗證
- ✅ **版本管理**: 自動版本控制與差異比對

**主要模組**:
- `aiva_flow_analyzer.py` - AST 解析與 Mermaid 流程圖生成
- `aiva_flow_classifier.py` - 五大模組架構分類與多路徑分析
- `aiva_cli_implementation.py` - 動態流程執行與 CLI 手冊生成
- `aiva_exploration_pipeline.py` - 完整自動化管線編排

**詳細文檔**: [Python 工具操作手冊](python_tools/README.md)

---

### TypeScript 工具

**位置**: `typescript_tools/`  
**主要語言**: TypeScript/JavaScript

TypeScript 工具與 Python 版本功能完全對等,支援 TS/TSX/JS/JSX 多種格式:

- ✅ **統一工具**: 單一 `ts2mermaid.ts` 整合 5 大功能模組
- ✅ **跨檔案追蹤**: 自動識別 import/export 關係
- ✅ **分類系統**: 六大類別自動標記
- ✅ **CLI 生成**: 產生 Bash/PowerShell 執行腳本

**核心功能**:
- AST 解析與圖形生成
- 跨檔案數據流串接 (Stitcher)
- 系統架構圖生成
- 自動分類系統 (Classifier)
- CLI 指令手冊生成

**詳細文檔**: [TypeScript 工具操作手冊](typescript_tools/README.md)

---

### Go 工具

**位置**: `go_tools/`  
**主要語言**: Go 1.21+

Go 工具提供原生 Go 代碼的高效能分析,整合 5 大功能模組:

- ✅ **原生解析**: 使用 `go/ast` 標準庫
- ✅ **Package 分析**: 完整支援 Go module 結構
- ✅ **並發安全**: 適用於大型 Go 專案
- ✅ **JSON 輸出**: 標準化分析報告格式

**核心功能**:
- AST 解析與流程圖生成
- 跨檔案數據流串接
- 功能分類與統計
- CLI 指令手冊生成
- 系統瓶頸分析

**詳細文檔**: [Go 工具使用手冊](go_tools/README.md)

---

### Rust 工具

**位置**: `rust_tools/`  
**主要語言**: Rust 1.70+

Rust 工具完整對標 Python 和 Go 版本,提供類型安全的代碼分析:

- ✅ **Syn Crate**: 使用強大的 `syn` 庫解析 Rust AST
- ✅ **模組系統**: 完整支援 Rust module/crate 結構
- ✅ **方法分析**: 支援 impl 區塊和 trait 方法
- ✅ **類型推導**: 識別結構體方法調用

**核心功能**:
- AST 解析與流程圖生成
- 跨檔案數據流串接
- 功能分類與統計
- CLI 指令手冊生成
- 系統瓶頸分析

**詳細文檔**: [Rust 工具使用手冊](rust_tools/README.md)

---

### Self-Healing 自我修復模組

**位置**: `self_healing/`  
**主要語言**: Python 3.10+

Self-Healing 模組是 AIVA 的自我診斷系統,提供統一的代碼健康度檢查入口:

- ✅ **統一診斷**: 單一 API 完成所有健康度檢查
- ✅ **三大分析器**: 斷點檢測 / 缺失連接 / 智能過濾
- ✅ **分級報告**: Critical / Warning / Info 三級問題分類
- ✅ **零錯誤**: 所有工具通過完整驗證

**核心模組**:
- `core_analyzer.py` - 統一診斷入口 (推薦使用)
- `analyze_dataflow_breakpoints.py` - 數據流斷點檢測
- `analyze_missing_function_connections.py` - 缺失連接分析
- `practical_analyzer.py` - 智能過濾和分級

**詳細文檔**: [Self-Healing 模組說明](self_healing/README.md)

---

## 🚀 快速開始

### 📋 如何更換分析目標/範圍

所有4語言工具都支援靈活的目標調整，以下說明如何切換分析範圍：

#### **1. TypeScript 工具** (`ts2mermaid.ts`)

```bash
# 分析整個目錄
cd /目標專案路徑
npx ts-node /path/to/ts2mermaid.ts "./src"

# 分析特定模組
npx ts-node /path/to/ts2mermaid.ts "./src/services/specific-module"

# 分析單一檔案（需在該目錄下）
npx ts-node /path/to/ts2mermaid.ts "."
```

**輸出位置**: 自動在當前目錄創建 `analysis_output/analysis_results.json`

#### **2. Go 工具** (`go2mermaid.go`)

```bash
# 分析整個專案
cd /Go專案根目錄
go run /path/to/go2mermaid.go .

# 分析特定package
cd /Go專案/specific_package
go run /path/to/go2mermaid.go .

# 分析子目錄
go run /path/to/go2mermaid.go ./sub/directory
```

**輸出位置**: 
- 預設: `/path/to/go_tools/output/analysis_results.json`  
- 可在當前目錄創建 `analysis_output/` 存放結果

#### **3. Rust 工具** (`main.rs`)

```bash
# 分析整個Rust專案
cd /Rust專案目錄
cargo run --manifest-path /path/to/Cargo.toml -- .

# 分析特定模組目錄
cd /Rust專案/src/modules/specific
cargo run --manifest-path /path/to/Cargo.toml -- .
```

**輸出位置**: 固定路徑 `services/integration/data/internal_exploration/analysis_results/rust/`

**⚠️ 注意**: Rust工具輸出路徑固定，若要改為相對路徑需修改 `src/paths_config.rs`

#### **4. Python 工具** (`aiva_flow_analyzer.py`)

```bash
# 分析整個目錄
python _dev_tools/common/development/aiva_flow_analyzer.py \
  --target "services/features/features_ready" \
  --output "./features_analysis"

# 分析核心模組
python _dev_tools/common/development/aiva_flow_analyzer.py \
  --target "services/core/aiva_core" \
  --output "./core_analysis"

# 調整深度和路徑數
python _dev_tools/common/development/aiva_flow_analyzer.py \
  --target "任意目錄" \
  --depth 5 \
  --max-paths 20 \
  --output "./custom_output"
```

**參數說明**:
- `--target`: 分析目標目錄（相對或絕對路徑）
- `--depth`: 最大追蹤深度（預設3）
- `--max-paths`: 每個入口點的最大路徑數（預設10）
- `--output`: 輸出目錄

**輸出位置**: 使用者指定的 `--output` 目錄

#### **5. 分類器** (將舊格式轉為統一JSON)

```bash
# 對Python分析結果進行分類
python _dev_tools/common/development/aiva_flow_classifier_final.py \
  --input "./your_analysis_output" \
  --output "./classification_results"
```

### 環境準備

```powershell
# 設定 Python 環境變數 (Python 工具必需)
$env:PYTHONPATH="C:\D\fold7\AIVA-git\services\common;C:\D\fold7\AIVA-git\services\core"
```

### 選擇適合的工具

根據你的目標語言選擇對應工具:

| 分析目標 | 推薦工具 | 目標參數 | 輸出控制 |
|---------|---------|---------|---------|
| **Python 代碼** | `aiva_flow_analyzer.py` | `--target <路徑>` | `--output <目錄>` ✅ |
| **TypeScript** | `ts2mermaid.ts` | 命令行參數 | 當前目錄固定 ⚠️ |
| **Go 代碼** | `go2mermaid.go` | 命令行參數 | 固定路徑 ⚠️ |
| **Rust 代碼** | `main.rs` | 命令行參數 | 固定路徑 ⚠️ |
| **分類轉換** | `aiva_flow_classifier_final.py` | `--input <目錄>` | `--output <目錄>` ✅ |

### 典型工作流程

#### **完整分析流程 (以Features模組為例)**

```bash
# 步驟1: Python代碼分析
cd C:\D\fold7\AIVA-git
python _dev_tools/common/development/aiva_flow_analyzer.py \
  --target "services/features/features_ready" \
  --output "./features_ready_analysis" \
  --verbose

# 步驟2: 分類轉換為統一JSON格式
python _dev_tools/common/development/aiva_flow_classifier_final.py \
  --input "./features_ready_analysis" \
  --output "./features_classification"

# 步驟3: Go模組分析 (在模組目錄執行)
cd services/features/features_in_development/function_authn_go
go run ../../../../core/aiva_core/internal_exploration/go_tools/go2mermaid.go .

# 步驟4: TypeScript模組分析
cd services/scan/typescript_engine
npx ts-node --esm ../../../services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts "./src"

# 步驟5: Rust模組分析
cd services/features/features_ready/function_crypto/rust_core
cargo run --manifest-path ../../../../core/aiva_core/internal_exploration/rust_tools/Cargo.toml -- .
```

#### **結果檢視**

```bash
# Python結果
cat features_classification/classification_data.json

# Go結果  
cat services/features/features_in_development/function_authn_go/analysis_output/analysis_results.json

# TypeScript結果
cat services/scan/typescript_engine/analysis_output/analysis_results.json

# Rust結果
cat services/integration/data/internal_exploration/analysis_results/rust/analysis_results.json
```

#### **輸出格式對比**

所有工具現在都輸出統一的JSON Schema v3.3格式：

```json
{
  "metadata": {
    "tool": "ts2mermaid",        // 工具名稱
    "version": "2.0",             // 版本
    "language": "typescript",     // 語言
    "generated_at": "2026-01-13T...",
    "total_flows": 3,             // flows數量
    "total_files": 11,            // 檔案數
    "schema_version": "3.3",      // ✅ 統一版本
    "ai_compatible": true         // ✅ AI可讀
  },
  "flows": [                      // ✅ 統一欄位
    {
      "id": 1,
      "path": ["從", "到"],
      "full_path": ["完整路徑1", "完整路徑2"],
      "classifications": [...],
      "language": "typescript",
      "cli_command": "..."
    }
  ],
  "functions": [...],             // 保留原始數據
  "flow_chains": [...],           // 保留原始數據（Python）
  "summary": {...}                // 保留原始數據
}
```

---

## 📚 詳細文檔

### 工具操作手冊

每個工具套件都有完整的操作手冊,包含安裝、使用、範例和疑難排解:

- 📘 **[Python 工具操作手冊](python_tools/README.md)**
  - 四大核心模組詳解
  - 17+ 實際使用範例
  - 完整 API 參考
  - 故障排除指南

- 📘 **[TypeScript 工具操作手冊](typescript_tools/README.md)**
  - 統一工具使用說明
  - 輸出檔案格式詳解
  - 三大實戰場景
  - 效能優化建議

- 📘 **[Go 工具使用手冊](go_tools/README.md)**
  - Go AST 分析原理
  - 編譯與執行方式
  - 參數詳細說明
  - 輸出檔案結構

- 📘 **[Rust 工具使用手冊](rust_tools/README.md)**
  - Rust 特有功能說明
  - Cargo 編譯流程
  - 模組系統分析
  - Trait 方法處理

- 📘 **[Self-Healing 模組說明](self_healing/README.md)**
  - 統一診斷 API
  - 三大分析器詳解
  - 診斷報告解讀
  - 整合使用範例

---

## 📊 統計資訊

### 當前成果記錄 (2026-01-13)

**已完成的4語言分析**:

| 語言 | 模組 | 位置 | Flows | 檔案數 | 狀態 |
|------|------|------|-------|--------|------|
| **Go** | function_authn_go | features/features_in_development/ | 4 | 4 | ✅ |
| **TypeScript** | typescript_engine | services/scan/ | 3 | 11 | ✅ |
| **Python** | features_ready | services/features/ | 150 | 119 | ✅ |
| **Rust** | function_crypto | features/features_ready/ | 0* | 5 | ✅ |

*Rust 0 flows 為正常情況：crypto模組是4個獨立analyzer的工具集合，無跨檔案調用鏈

**統計總覽**:
- 總Flows: **157條** (Go: 4 + TS: 3 + Python: 150 + Rust: 0)
- 總檔案: **139個** (Go: 4 + TS: 11 + Python: 119 + Rust: 5)
- 支援語言: **4種** (Python, TypeScript, Go, Rust)
- JSON Schema: **v3.3統一格式**

### 子模組統計

| 子模組 | 檔案數 | 代碼行數 | 說明 | 文檔 |
|--------|--------|---------|------|------|
| **python_tools** | 6 | 4,450 | Python AST 分析工具套件 | [README](python_tools/README.md) |
| **self_healing** | 8 | 3,711 | 自我診斷與健康度檢測 | [README](self_healing/README.md) |
| **go_tools** | 2 | 891 | Go AST 分析工具 (✅ 已改進) | [README](go_tools/README.md) |
| **rust_tools** | 2 | 864 | Rust AST 分析工具 (✅ 已改進) | [README](rust_tools/README.md) |
| **typescript_tools** | 2 | 865 | TypeScript AST 分析工具 (✅ 已改進) | [README](typescript_tools/README.md) |
| **總計** | **20** | **10,781** | - | - |

### 分析能力

- **支援語言**: 4 種主流語言 (Python/TypeScript/Go/Rust)
- **輸出格式**: Mermaid 圖 + **統一JSON Schema v3.3** + CLI 腳本
- **統一欄位**: metadata, flows, functions (保留所有原始欄位)
- **AI相容**: ai_compatible=true，可直接供FlowExecutor使用

---

## ❓ 常見問題 (FAQ)

### Q1: 為什麼Rust的flows是0？

**A**: 這是**正常的**！`function_crypto/rust_core` 是4個獨立分析工具的集合：
- `cookie_analyzer.rs` - Cookie安全分析
- `header_analyzer.rs` - HTTP標頭分析  
- `js_crypto_analyzer.rs` - JavaScript密碼學分析
- `tls_analyzer.rs` - TLS/SSL配置分析

這些analyzer不互相調用，所以沒有跨檔案數據流，0 flows是正確結果。

### Q2: 如何更換分析目標？

**A**: 每個工具都支援目標調整，參數不同：
- **Python**: `--target <路徑>` 參數
- **TypeScript/Go/Rust**: 命令行參數指定目錄

詳見上方「📋 如何更換分析目標/範圍」章節。

### Q3: 輸出結果存放在哪裡？

**A**: 
- **Python**: 自訂 `--output` 目錄
- **TypeScript**: 當前目錄的 `analysis_output/`
- **Go**: `go_tools/output/`  
- **Rust**: 固定路徑 `services/integration/data/internal_exploration/analysis_results/rust/`

### Q4: 如何驗證JSON格式正確？

**A**: 檢查3個關鍵欄位：
```bash
# 使用PowerShell檢查
$json = Get-Content "路徑/analysis_results.json" | ConvertFrom-Json
$json.metadata.schema_version  # 應為 "3.3"
$json.metadata.ai_compatible   # 應為 true
$json.flows.Count              # 應有數字(Python/TS/Go)或0(Rust工具集)
```

### Q5: 各語言工具可以分析其他專案嗎？

**A**: ✅ **完全可以！**所有工具都是通用的：
```bash
# 分析任何Python專案
python aiva_flow_analyzer.py --target "/path/to/any/python/project"

# 分析任何Go專案
cd /any/go/project && go run /path/to/go2mermaid.go .

# 分析任何TypeScript專案  
cd /any/ts/project && npx ts-node /path/to/ts2mermaid.ts "./src"

# 分析任何Rust專案
cd /any/rust/project && cargo run --manifest-path /path/to/Cargo.toml -- .
```

---
- **分類系統**: 6 大功能類別 (偵察/攻擊/分析/報告/持久化/其他)
- **跨檔案追蹤**: 所有工具支援多檔案數據流串接
- **自我診斷**: Self-Healing 模組提供零錯誤健康度檢查

### AIVA Core 分析統計

基於 Python 工具的 AIVA Core 代碼分析結果:

- **總數據流**: 360 條 (含分支)
- **真實連接**: 228 條
- **分析檔案**: 125 個 Python 檔案
- **分析函數**: 1,321 個函數
- **主要模組**: service_backbone (佔比 64.9%)

---

## 🤝 貢獻與支援

### 回報問題

發現問題請到 GitHub Issues 回報,包含:
- 使用的工具與版本
- 完整錯誤訊息
- 重現步驟
- 預期行為

### 改進建議

歡迎提交 Pull Request:
- 新增功能或改進現有功能
- 修正錯誤或優化效能
- 補充文檔或範例
- 翻譯文檔

---

**版本**: v3.2.0  
**最後更新**: 2026-01-07  
**維護者**: AIVA Team  
**授權**: AIVA 專案授權條款
