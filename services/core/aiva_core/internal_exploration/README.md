# 🧭 Internal Exploration - 內部探索

> **路徑**: `internal_exploration/`  
> **狀態**: ✅ Production Ready | **最後更新**: 2026-01-21  
> **子模組**: 2 個 | **總文件數**: 16 | **架構版本**: v3.1（多語言工具已優化）  
> **父模組**: [AIVA Core](../README.md)

## 概述

**Internal Exploration** 是 AIVA 五大核心模組之一，作為自我認知系統。提供**4種語言**的 AST 分析、數據流追蹤、自動化分類和自我執行能力。

**核心特色**：
- 🔍 **多語言 AST 解析** - Python, Go, Rust, TypeScript 統一 JSON 輸出
- 📊 **數據流視覺化** - 自動生成 Mermaid 流程圖
- 🏷️ **智能分類系統** - 內部模組/外部模組自動分類
- 🔧 **自我修復診斷** - 自動檢測數據流斷點和架構問題
- ⚡ **動態執行系統** - 支援多語言 subprocess 執行

---

## 語言工具狀態總覽

| 語言 | 分析能力 | struct/CLI 支援 | 狀態 |
|------|----------|----------------|------|
| **Python** | ✅ 完整 AST | ✅ 函數參數 | 207 flows |
| **Go** | ✅ 完整 AST + 語義 | ✅ struct tags | 5 flows |
| **Rust** | ⚠️ 語法解析 | ❌ Clap macros | 1 flow |
| **TypeScript** | ✅ 完整 AST | ✅ interface/type | 待測試 |

---

## 🎯 架構設計 (v3.0)

### 雙層架構

```
語言層 (Language Layer) - 只做 AST 解析
  ├── python_tools/aiva_flow_analyzer.py
  ├── go_tools/go2mermaid.go
  ├── rust_tools/src/main.rs
  └── typescript_tools/ts2mermaid.ts
        ↓ 輸出統一 JSON
        
業務邏輯層 (Business Logic Layer) - 分類與執行
  ├── aiva_internal_classifier.py    (AI Core 分類)
  ├── aiva_internal_executor.py      (AI Core 執行)
  ├── aiva_external_classifier.py    (Features/Scan 分類)
  └── aiva_external_executor.py      (Features/Scan 執行)
```

### Internal vs External CLI

| 類型 | 目標模組 | 通信方式 | 分類器 | 執行器 |
|------|---------|---------|--------|--------|
| **Internal CLI** | AI Core 模組 | 直接導入 | `aiva_internal_classifier.py` | `aiva_internal_executor.py` |
| **External CLI** | Features/Scan | subprocess + JSON | `aiva_external_classifier.py` | `aiva_external_executor.py` |

---

## 📁 目錄結構

```
internal_exploration/
├── aiva_internal_classifier.py      # AI Core 分類器 ⭐
├── aiva_internal_executor.py        # AI Core 執行器 ⭐
├── aiva_external_classifier.py      # Features/Scan 分類器 ⭐
├── aiva_external_executor.py        # Features/Scan 執行器 ⭐
├── modules_config.json               # 模組配置
├── __init__.py                       # 模組初始化
│
├── python_tools/                     # Python AST 工具
│   ├── aiva_flow_analyzer.py        # Python AST 解析器（核心）
│   ├── aiva_flow_classifier.py      # 📚 範本（測試參考）
│   ├── aiva_cli_implementation.py   # 📚 範本（測試參考）
│   ├── data/                         # RAG 知識庫（重要）
│   └── README.md
│
├── go_tools/                         # Go AST 工具
│   ├── go2mermaid.go                 # Go AST 解析器
│   ├── go.mod
│   └── README.md
│
├── rust_tools/                       # Rust AST 工具
│   ├── src/main.rs                   # Rust AST 解析器
│   ├── Cargo.toml
│   └── README.md
│
├── typescript_tools/                 # TypeScript AST 工具
│   ├── ts2mermaid.ts                 # TS/JS AST 解析器
│   ├── package.json
│   └── README.md
│
└── self_healing/                     # 自我診斷模組
    ├── core_analyzer.py              # 統一診斷入口
    ├── analyze_dataflow_breakpoints.py
    ├── analyze_missing_function_connections.py
    ├── practical_analyzer.py
    └── README.md
```

---

## 🛠️ 核心組件

### 主要類別

| 類別 | 文件 | 說明 |
|------|------|------|
| **`AIVAFlowClassifier`** | aiva_internal_classifier.py | AI Core 6大模組分類器 |
| **`FlowExecutor`** | aiva_internal_executor.py | AI Core 流程執行器 |
| **`MultiLanguageClassifier`** | aiva_external_classifier.py | 多語言整合分類器 |
| **`MultiLangExecutor`** | aiva_external_executor.py | 多語言 subprocess 執行器 |
| `AIVAFlowAnalyzer` | python_tools/aiva_flow_analyzer.py | Python AST 分析器 |
| `CoreAnalyzer` | self_healing/core_analyzer.py | 自我診斷入口 |

### 語言工具對比

| 語言 | 核心檔案 | 輸出格式 | 編譯/執行 | 參數提取 | 狀態 |
|------|---------|---------|-----------|----------|------|
| Python | `aiva_flow_analyzer.py` | JSON | `python` 直接執行 | ✅ 函數參數 | ✅ 207 flows |
| Go | `go2mermaid.go` | JSON | `go run` 或編譯 | ✅ struct + JSON tags | ✅ 5 flows |
| Rust | `src/main.rs` | JSON | `cargo run` | ⚠️ stdin JSON only | ⚠️ 1 flow |
| TypeScript | `ts2mermaid.ts` | JSON | `npx ts-node` | ✅ interface/type | 🔄 待測試 |

**統一輸出**: 所有工具輸出 `analysis_results.json`（JSON Schema v3.3*）

**注意事項**：
- **Go**: v3.1 已支援 struct 欄位提取（適用於微服務 stdin JSON 模式）
- **Rust**: 目前僅支援 stdin JSON 模式，Clap CLI 框架的參數無法提取（`syn` crate 限制）
- **Python**: 使用 `ast` 標準庫，完整支援所有函數定義
- **TypeScript**: 理論支援 interface/type，待實際測試驗證

## 📊 JSON Schema 說明

### 統一輸出格式

所有語言工具輸出格式：

```json
{
  "metadata": {
    "tool": "python_analyzer | go2mermaid | rs2mermaid | ts2mermaid",
    "version": "3.0",
    "language": "python | go | rust | typescript",
    "generated_at": "2026-01-18T10:30:00+08:00",
    "total_flows": 157,
    "total_files": 119,
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["from_function", "to_function"],
      "full_path": ["module.from_func", "module.to_func"],
      "func_names": ["from_func", "to_func"],
      "length": 2,
      "start": "from_function",
      "end": "to_function",
      "from_script": "path/to/file1.py",
      "to_script": "path/to/file2.py"
    }
  ],
  "functions": { ... }
}
```

### ⚠️ 關於 schema_version 和 ai_compatible

**重要說明**：
- `"schema_version": "3.3"` - **自定義版本標記**，無正式規範文件
- `"ai_compatible": true` - **標記可被分類器處理**，無具體驗證機制

這些欄位是在 2026-01-13 的 Phase 1 改進中加入，目的是統一多語言工具輸出格式。目前：
- ✅ Go/Rust/TypeScript 工具有輸出這些欄位
- ⚠️ Python 工具使用舊格式（無這些欄位）
- ⚠️ 分類器沒有強制驗證版本

---

## 🚀 快速開始

### 環境準備

```powershell
# Python 環境變數（必要）
$env:PYTHONPATH="C:\D\fold7\AIVA-git\services\common;C:\D\fold7\AIVA-git\services\core"
```

### Internal CLI 工作流程（AI Core 模組）

```bash
# 1. AST 分析 - 使用 Python 工具
python python_tools/aiva_flow_analyzer.py --target services/core/aiva_core --output ./analysis

# 2. 分類 - 讀取 analysis_results.json
python aiva_internal_classifier.py

# 3. 執行 - 讀取 classification_data.json
python aiva_internal_executor.py --flow 11
```

### External CLI 工作流程（Features/Scan 模組）

```bash
# 1. 多語言 AST 分析

# Python 模組
python python_tools/aiva_flow_analyzer.py --target services/features/function_xss

# Go 模組
cd services/features/function_authn_go
go run ../../core/aiva_core/internal_exploration/go_tools/go2mermaid.go .

# Rust 模組
cd services/features/function_crypto
cargo run --manifest-path ../../core/aiva_core/internal_exploration/rust_tools/Cargo.toml -- .

# TypeScript 模組
cd services/scan/typescript_engine
npx ts-node ../../core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts ./src

# 2. 整合分類 - 讀取所有 analysis_results.json
python aiva_external_classifier.py

# 3. 多語言執行 - subprocess + JSON
python aiva_external_executor.py --lang python --flow 1
```

---

## 📚 各子模組說明

### 1. Python Tools

**功能**: Python 代碼 AST 解析  
**核心文件**: `python_tools/aiva_flow_analyzer.py`

**當前狀態**:
- ✅ AST 解析完整
- ⚠️ 輸出為舊格式（`flow_chains` + `function_details`）
- ⚠️ 缺少 `metadata.schema_version` 和 `ai_compatible`

**範本文件** (將移除):
- `aiva_flow_classifier.py` - 早期分類器範本
- `aiva_cli_implementation.py` - 早期執行器範本

**重要**: `python_tools/data/` 包含 RAG 知識庫，不可刪除。

📖 詳見: [python_tools/README.md](python_tools/README.md)

### 2. Go Tools

**功能**: Go 代碼 AST 解析  
**核心文件**: `go_tools/go2mermaid.go`

**特色**:
- ✅ 原生 `go/ast` 解析
- ✅ 支援 Go module 結構
- ✅ 輸出 JSON Schema v3.3
- ✅ 並發安全

📖 詳見: [go_tools/README.md](go_tools/README.md)

### 3. Rust Tools

**功能**: Rust 代碼 AST 解析  
**核心文件**: `rust_tools/src/main.rs`

**特色**:
- ✅ 使用 `syn` crate 解析
- ✅ 支援 Rust module/crate
- ✅ 輸出 JSON Schema v3.3
- ✅ 類型安全

📖 詳見: [rust_tools/README.md](rust_tools/README.md)

### 4. TypeScript Tools

**功能**: TypeScript/JavaScript AST 解析  
**核心文件**: `typescript_tools/ts2mermaid.ts`

**特色**:
- ✅ 支援 TS/TSX/JS/JSX
- ✅ ES6 模組追蹤
- ✅ 輸出 JSON Schema v3.3
- ✅ Async/await 支援

📖 詳見: [typescript_tools/README.md](typescript_tools/README.md)

### 5. Self-Healing

**功能**: 自我診斷與健康度檢測  
**核心文件**: `self_healing/core_analyzer.py`

**特色**:
- ✅ 統一診斷 API
- ✅ 三級問題分類（Critical/Warning/Info）
- ✅ 斷點檢測
- ✅ 缺失連接分析

📖 詳見: [self_healing/README.md](self_healing/README.md)

---

## 📊 統計資訊

### 多語言分析成果 (2026-01-13)

| 語言 | 模組 | Flows | 檔案數 | 狀態 |
|------|------|-------|--------|------|
| **Python** | features_ready | 150 | 119 | ✅ 完成 |
| **Go** | function_authn_go | 4 | 4 | ✅ 完成 |
| **Rust** | function_crypto | 0* | 5 | ✅ 完成 |
| **TypeScript** | typescript_engine | 3 | 11 | ✅ 完成 |
| **總計** | - | **157** | **139** | - |

*Rust 0 flows 為正常：crypto 模組是 4 個獨立 analyzer 工具集合，無跨檔案調用。

### 子模組統計

| 子模組 | 檔案數 | 代碼行數 | 說明 |
|--------|--------|---------|------|
| python_tools | 3 核心 + 2 範本 | ~4,000 | Python AST 工具 |
| go_tools | 1 | 891 | Go AST 工具 |
| rust_tools | 1 | 864 | Rust AST 工具 |
| typescript_tools | 1 | 865 | TypeScript AST 工具 |
| self_healing | 8 | 3,711 | 自我診斷系統 |
| **業務邏輯層** | 4 | ~2,500 | 分類器+執行器 |

---

## ❓ 常見問題

### Q1: 為什麼 Rust 的 flows 是 0？

**A**: 正常情況。`function_crypto/rust_core` 是 4 個獨立分析工具：
- cookie_analyzer.rs
- header_analyzer.rs
- js_crypto_analyzer.rs
- tls_analyzer.rs

這些工具不互相調用，所以無跨檔案數據流。

### Q2: schema_version "3.3" 有正式規範嗎？

**A**: 沒有。這是 Phase 1 改進時自定義的版本標記，用於：
- 標識多語言工具使用統一格式
- 讓分類器識別新格式的 JSON

目前沒有：
- ❌ 正式的 JSON Schema 規範文件
- ❌ 版本驗證機制
- ❌ "ai_compatible" 的具體定義

### Q3: python_tools/ 的範本文件要刪除嗎？

**A**: 暫時保留，待測試完成後移除：
- `aiva_flow_classifier.py` - 已被 `aiva_internal_classifier.py` 取代
- `aiva_cli_implementation.py` - 已被 `aiva_internal_executor.py` 取代

**重要**: `data/` 目錄包含 RAG 知識庫，**不可刪除**。

### Q4: 如何切換分析目標？

**A**: 各工具參數不同：

```bash
# Python - 使用 --target 參數
python python_tools/aiva_flow_analyzer.py --target <路徑> --output <目錄>

# Go - 命令行參數
cd <目標目錄>
go run <路徑>/go2mermaid.go .

# Rust - 命令行參數
cd <目標目錄>
cargo run --manifest-path <路徑>/Cargo.toml -- .

# TypeScript - 命令行參數
cd <目標目錄>
npx ts-node <路徑>/ts2mermaid.ts ./src
```

### Q5: 分類器和執行器的區別？

**A**:

| 組件 | 職責 | 輸入 | 輸出 |
|------|------|------|------|
| **Classifier** | 分類數據流 | `analysis_results.json` | `classification_data.json` |
| **Executor** | 執行流程 | `classification_data.json` | 執行結果 |

---

## 🔄 架構演進歷史

### v3.0 (2026-01-18) - 語言層與業務邏輯層分離

- ✅ 建立 4 個正式業務邏輯腳本（internal/external × classifier/executor）
- ✅ 語言工具專注於 AST 解析
- ✅ 移除根目錄多餘檔案（exploration_pipeline, capability_cli, dispatcher）
- ✅ 更新所有文檔反映新架構

### v2.x (2026-01-13) - JSON 格式統一 (Phase 1)

- ✅ Go/Rust/TypeScript 工具添加 `metadata` + `flows` 欄位
- ✅ 制定 JSON Schema v3.3（自定義）
- ✅ 完成 4 語言模組分析

### v1.x - 早期單體設計

- Python 工具包含分析+分類+執行
- 各語言工具輸出格式不統一

---

## 📝 待辦事項

### 高優先級

- [ ] 更新 Python 工具輸出格式（對齊 Go/Rust/TypeScript）
- [ ] 測試並移除範本文件（aiva_flow_classifier.py, aiva_cli_implementation.py）
- [ ] 建立 JSON Schema 正式規範文件（如需要）

### 中優先級

- [ ] 統一所有工具的輸出路徑配置
- [ ] 添加分類器的版本驗證機制
- [ ] 完善 "ai_compatible" 的具體定義

### 低優先級

- [ ] 考慮 data/ 目錄重定位
- [ ] 優化 Rust 工具輸出路徑（目前為硬編碼）

---

## 🤝 貢獻

### 回報問題

發現問題請提供：
- 使用的工具與版本
- 完整錯誤訊息
- 重現步驟

### 改進建議

歡迎提交 Pull Request 改進：
- 新功能開發
- 文檔完善
- Bug 修復

---

**導航**: [← 返回 AIVA Core](../README.md)

**相關文檔**:
- [Python Tools README](python_tools/README.md)
- [Go Tools README](go_tools/README.md)
- [Rust Tools README](rust_tools/README.md)
- [TypeScript Tools README](typescript_tools/README.md)
- [Self-Healing README](self_healing/README.md)
- [架構重構報告](ARCHITECTURE_REFACTOR.md)
- [多語言工具說明](MULTILANG_TOOLS_README.md)
- [Phase 1 完成報告](PHASE1_COMPLETION_REPORT.md)
