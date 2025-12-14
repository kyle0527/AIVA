# 多語言完整功能對照表

本文檔說明四種語言版本 (Python, Go, TypeScript, Rust) 如何完整實現 AIVA 數據流分析的三個階段。

## 📋 總覽

| 階段 | Python | Go | TypeScript | Rust |
|------|--------|-----|-----------|------|
| **階段一: AST 分析** | ✅ `aiva_flow_analyzer.py` | ✅ `go2mermaid.go` | ✅ `ts2mermaid.ts` | ✅ `rs2mermaid.rs` |
| **階段二: 數據流分類** | ✅ `aiva_flow_classifier.py` | ✅ `go_classifier.go` | ✅ `ts_classifier.ts` | ✅ `rs_classifier.rs` |
| **階段三: CLI 執行** | ✅ `aiva_cli_implementation.py` | ✅ `go_cli_implementation.go` | ✅ `ts_cli_implementation.ts` | ✅ `rs_cli_implementation.rs` |

---

## 🔍 階段一: AST 分析與流程圖生成

### 核心功能

所有版本都實現以下功能：

1. **AST 解析** - 分析源代碼的抽象語法樹
2. **函數調用追蹤** - 識別函數之間的調用關係
3. **Mermaid 圖生成** - 產生視覺化流程圖
4. **函數分類** - 將函數分為六大類別
5. **CLI 指令生成** - 自動生成命令列指令
6. **JSON 輸出** - 結構化數據輸出

### 使用方式對照

#### Python 版本
```powershell
python aiva_flow_analyzer.py --target-dir="services/scan" --output="analysis"
```

#### Go 版本
```powershell
go run go2mermaid.go --input="services/scan" --output="analysis"
```

#### TypeScript 版本
```powershell
npm run analyze -- --input="services/scan" --output="analysis"
# 或
ts-node ts2mermaid.ts --input="services/scan" --output="analysis"
```

#### Rust 版本
```powershell
cargo run --bin rs2mermaid -- --input="services/scan" --output="analysis"
```

### 輸出檔案對照

| 檔案 | Python | Go | TypeScript | Rust |
|------|--------|-----|-----------|------|
| 完整數據流圖 | `flow_results.json` | `graph.json` | `graph.json` | `graph.json` |
| 分類函數列表 | (在 JSON 中) | `classified_functions.json` | `classified_functions.json` | `classified_functions.json` |
| CLI 指令清單 | (在 JSON 中) | `cli_commands.txt` | `cli_commands.txt` | `cli_commands.txt` |
| Mermaid 圖 | `*.mmd` | `*.mmd` | `*.mmd` | `*.mmd` |
| 統計報告 | `analysis_report.txt` | (在 JSON 中) | (在 JSON 中) | (在 JSON 中) |

---

## 📊 階段二: 數據流分類與統計分析

### 核心功能

所有版本都實現以下功能：

1. **模組分類** - 將數據流分配到六大模組
   - 認知核心模組 (cognitive_core)
   - 內探模組 (internal_exploration)
   - 任務規劃模組 (task_planning)
   - 外學模組 (external_learning)
   - 核心能力模組 (core_capabilities)
   - 服務骨幹模組 (service_backbone)

2. **組件類型識別** - 標記為 AI組件/程式組件/混合組件
3. **路徑分析** - 分析多路徑到達相同終點的情況
4. **統計報告** - 生成詳細的分布統計
5. **數據增強** - 添加完整路徑、主模組等資訊

### 使用方式對照

#### Python 版本
```powershell
python aiva_flow_classifier.py --input="flow_results.json" --output="classified"
```

#### Go 版本
```powershell
go run go_classifier.go --input="flow_results.json" --output="classified"
```

#### TypeScript 版本
```powershell
npm run classify -- --input="flow_results.json" --output="classified"
# 或
ts-node ts_classifier.ts --input="flow_results.json" --output="classified"
```

#### Rust 版本
```powershell
cargo run --bin rs_classifier -- --input="flow_results.json" --output="classified"
```

### 輸出檔案對照

| 檔案 | 所有版本 | 說明 |
|------|---------|------|
| `classification_data.json` | ✅ | 主要輸出：完整分類後的數據流 (282 flows) |
| `classification_report.txt` | ✅ | 人類可讀的統計報告 |
| `module_distribution.json` | ✅ | 模組分布與多路徑分析 |

---

## 🚀 階段三: 動態執行與文檔生成

### 核心功能

所有版本都實現以下功能：

1. **流程列表** - 顯示所有可用的數據流
2. **流程執行** - 模擬執行指定的數據流 (Go/TS/Rust 為模擬模式)
3. **Dry Run 模式** - 預覽執行計畫而不實際執行
4. **Markdown 文檔生成** - 生成人類可讀的 CLI 參考手冊
5. **JSON 資料庫生成** - 生成 AI 可檢索的指令資料庫
6. **按模組分組** - 依據六大模組組織流程

### 使用方式對照

#### 列出所有流程

```powershell
# Python
python aiva_cli_implementation.py --list

# Go
go run go_cli_implementation.go --list

# TypeScript
npm run cli:list
# 或
ts-node ts_cli_implementation.ts --list

# Rust
cargo run --bin rs_cli_implementation -- --list
```

#### 執行特定流程

```powershell
# Python (真實執行)
python aiva_cli_implementation.py --flow=1

# Go (模擬執行)
go run go_cli_implementation.go --flow=1

# TypeScript (模擬執行)
ts-node ts_cli_implementation.ts --flow=1

# Rust (模擬執行)
cargo run --bin rs_cli_implementation -- --flow=1
```

#### Dry Run 預覽

```powershell
# Python
python aiva_cli_implementation.py --flow=1 --dry-run

# Go
go run go_cli_implementation.go --flow=1 --dry-run

# TypeScript
ts-node ts_cli_implementation.ts --flow=1 --dry-run

# Rust
cargo run --bin rs_cli_implementation -- --flow=1 --dry-run
```

#### 生成 Markdown 文檔

```powershell
# Python
python aiva_cli_implementation.py --generate-doc md

# Go
go run go_cli_implementation.go --generate-doc=md

# TypeScript
npm run cli:doc
# 或
ts-node ts_cli_implementation.ts --generate-doc=md

# Rust
cargo run --bin rs_cli_implementation -- --generate-doc=md
```

#### 生成 JSON 資料庫

```powershell
# Python
python aiva_cli_implementation.py --generate-doc json

# Go
go run go_cli_implementation.go --generate-doc=json

# TypeScript
ts-node ts_cli_implementation.ts --generate-doc=json

# Rust
cargo run --bin rs_cli_implementation -- --generate-doc=json
```

### 輸出檔案對照

| 檔案 | 所有版本 | 說明 |
|------|---------|------|
| `CLI_COMMANDS_REFERENCE.md` | ✅ | Markdown 格式的 CLI 參考手冊 (人類閱讀) |
| `cli_commands_db.json` | ✅ | JSON 格式的指令資料庫 (AI 檢索) |

---

## 🔄 完整工作流程範例

### 場景: 分析混合語言專案

```powershell
# 設定輸出目錄
$OUTPUT = "C:\D\fold7\AIVA-git\services\scan\analysis_output"

# ===== 階段一: AST 分析 =====

# Python 代碼分析
python aiva_flow_analyzer.py --target-dir="services\scan" --output="$OUTPUT\python"

# Go 代碼分析
go run go2mermaid.go --input="services\scan" --output="$OUTPUT\go"

# TypeScript 代碼分析
npm run analyze -- --input="services\scan" --output="$OUTPUT\typescript"

# Rust 代碼分析
cargo run --bin rs2mermaid -- --input="services\scan" --output="$OUTPUT\rust"

# ===== 階段二: 數據流分類 =====

# 分類 Python 分析結果
python aiva_flow_classifier.py `
    --input="$OUTPUT\python\flow_results.json" `
    --output="$OUTPUT\python\classified"

# 分類 Go 分析結果
go run go_classifier.go `
    --input="$OUTPUT\go\graph.json" `
    --output="$OUTPUT\go\classified"

# 分類 TypeScript 分析結果
ts-node ts_classifier.ts `
    --input="$OUTPUT\typescript\graph.json" `
    --output="$OUTPUT\typescript\classified"

# 分類 Rust 分析結果
cargo run --bin rs_classifier -- `
    --input="$OUTPUT\rust\graph.json" `
    --output="$OUTPUT\rust\classified"

# ===== 階段三: 生成文檔與執行 =====

# 生成 Python 流程文檔
cd "$OUTPUT\python\classified"
python ..\..\..\..\..\core\aiva_core\internal_exploration\aiva_cli_implementation.py `
    --input="classification_data.json" `
    --generate-doc md

# 生成 Go 流程文檔
cd "$OUTPUT\go\classified"
go run ..\..\..\..\..\core\aiva_core\internal_exploration\go_cli_implementation.go `
    --input="classification_data.json" `
    --generate-doc=md

# 生成 TypeScript 流程文檔
cd "$OUTPUT\typescript\classified"
ts-node ..\..\..\..\..\core\aiva_core\internal_exploration\ts_cli_implementation.ts `
    --input="classification_data.json" `
    --generate-doc=md

# 生成 Rust 流程文檔
cd "$OUTPUT\rust\classified"
cargo run --bin rs_cli_implementation -- `
    --input="classification_data.json" `
    --generate-doc=md

# ===== 查看和執行流程 =====

# 列出 Python 流程
python aiva_cli_implementation.py --list

# 執行 Go 流程 (模擬)
go run go_cli_implementation.go --flow=1 --dry-run

# 執行 TypeScript 流程 (模擬)
ts-node ts_cli_implementation.ts --flow=5

# 執行 Rust 流程 (模擬)
cargo run --bin rs_cli_implementation -- --flow=10 --dry-run
```

---

## 📝 功能差異說明

### 執行模式

| 語言 | 執行模式 | 說明 |
|------|---------|------|
| **Python** | 真實執行 | 可以動態導入模組並實際執行 Python 代碼 |
| **Go** | 模擬執行 | 顯示執行步驟，但不實際執行 Go 代碼 |
| **TypeScript** | 模擬執行 | 顯示執行步驟，但不實際執行 TS 代碼 |
| **Rust** | 模擬執行 | 顯示執行步驟，但不實際執行 Rust 代碼 |

### 為何 Go/TS/Rust 是模擬執行？

1. **動態載入限制** - 編譯語言不支援像 Python 那樣的動態模組導入
2. **類型安全** - 需要在編譯時確定所有類型
3. **設計目的** - 這些工具主要用於分析和文檔生成，而非實際執行

### 各語言的優勢

| 語言 | 主要優勢 | 最佳使用場景 |
|------|---------|-------------|
| **Python** | 動態執行、生態豐富 | 實際執行 Python 數據流，開發測試 |
| **Go** | 高效能、並發處理 | 大規模代碼分析、生產環境部署 |
| **TypeScript** | 類型安全、前端整合 | 與 Node.js 項目整合、Web 應用 |
| **Rust** | 極致效能、記憶體安全 | 系統級分析、關鍵性能路徑 |

---

## 🎯 快速參考

### 完整三階段命令 (Python)

```powershell
# 一鍵執行完整流程
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration

# 階段一
python aiva_flow_analyzer.py --target-dir="..\..\..\.." --output="full_analysis"

# 階段二
python aiva_flow_classifier.py --input="full_analysis\flow_results.json" --output="full_analysis\classified"

# 階段三
python aiva_cli_implementation.py --input="full_analysis\classified\classification_data.json" --generate-doc md
python aiva_cli_implementation.py --input="full_analysis\classified\classification_data.json" --generate-doc json
python aiva_cli_implementation.py --input="full_analysis\classified\classification_data.json" --list
```

### 完整三階段命令 (Go)

```powershell
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration

# 階段一
go run go2mermaid.go --input="..\..\..\.." --output="go_analysis"

# 階段二
go run go_classifier.go --input="go_analysis\graph.json" --output="go_analysis\classified"

# 階段三
go run go_cli_implementation.go --input="go_analysis\classified\classification_data.json" --generate-doc=md
go run go_cli_implementation.go --input="go_analysis\classified\classification_data.json" --generate-doc=json
go run go_cli_implementation.go --input="go_analysis\classified\classification_data.json" --list
```

### 完整三階段命令 (TypeScript)

```powershell
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration

# 階段一
npm run analyze -- --input="..\..\..\.." --output="ts_analysis"

# 階段二
npm run classify -- --input="ts_analysis\graph.json" --output="ts_analysis\classified"

# 階段三
ts-node ts_cli_implementation.ts --input="ts_analysis\classified\classification_data.json" --generate-doc=md
ts-node ts_cli_implementation.ts --input="ts_analysis\classified\classification_data.json" --generate-doc=json
ts-node ts_cli_implementation.ts --input="ts_analysis\classified\classification_data.json" --list
```

### 完整三階段命令 (Rust)

```powershell
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration

# 階段一
cargo run --bin rs2mermaid -- --input="..\..\..\.." --output="rust_analysis"

# 階段二
cargo run --bin rs_classifier -- --input="rust_analysis\graph.json" --output="rust_analysis\classified"

# 階段三
cargo run --bin rs_cli_implementation -- --input="rust_analysis\classified\classification_data.json" --generate-doc=md
cargo run --bin rs_cli_implementation -- --input="rust_analysis\classified\classification_data.json" --generate-doc=json
cargo run --bin rs_cli_implementation -- --input="rust_analysis\classified\classification_data.json" --list
```

---

## 📦 檔案清單

### Python 版本
- `aiva_flow_analyzer.py` (1355 行) - 階段一
- `aiva_flow_classifier.py` (665 行) - 階段二
- `aiva_cli_implementation.py` (598 行) - 階段三

### Go 版本
- `go2mermaid.go` (596 行) - 階段一
- `go_classifier.go` (465 行) - 階段二
- `go_cli_implementation.go` (402 行) - 階段三
- `go.mod` - Go 模組配置

### TypeScript 版本
- `ts2mermaid.ts` (510 行) - 階段一
- `ts_classifier.ts` (351 行) - 階段二
- `ts_cli_implementation.ts` (308 行) - 階段三
- `package.json` - Node.js 配置

### Rust 版本
- `rs2mermaid.rs` (556 行) - 階段一
- `rs_classifier.rs` (445 行) - 階段二
- `rs_cli_implementation.rs` (418 行) - 階段三
- `Cargo.toml` - Rust 專案配置

---

## ✅ 功能完整性確認

| 功能 | Python | Go | TypeScript | Rust |
|------|--------|-----|-----------|------|
| AST 解析 | ✅ | ✅ | ✅ | ✅ |
| Mermaid 圖生成 | ✅ | ✅ | ✅ | ✅ |
| 函數分類 (6類) | ✅ | ✅ | ✅ | ✅ |
| 模組分類 (6模組) | ✅ | ✅ | ✅ | ✅ |
| 組件類型識別 | ✅ | ✅ | ✅ | ✅ |
| 多路徑分析 | ✅ | ✅ | ✅ | ✅ |
| CLI 指令生成 | ✅ | ✅ | ✅ | ✅ |
| 流程列表顯示 | ✅ | ✅ | ✅ | ✅ |
| Dry Run 模式 | ✅ | ✅ | ✅ | ✅ |
| Markdown 文檔 | ✅ | ✅ | ✅ | ✅ |
| JSON 資料庫 | ✅ | ✅ | ✅ | ✅ |
| 統計報告 | ✅ | ✅ | ✅ | ✅ |
| **真實執行** | ✅ | ❌ | ❌ | ❌ |

**總結**: 所有四種語言版本都具備完整的三階段功能，差異僅在於 Python 可以真實執行流程，而其他語言採用模擬執行模式。

---

**文檔版本**: 1.0.0  
**更新日期**: 2025-12-10  
**維護者**: AIVA Team
