# 🧭 Internal Exploration - 內部探索

> **版本**: v3.1.0  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2026-01-06  
> **角色**: AIVA 的自我認知系統 - 代碼分析與能力發現  
> **架構**: 三階段管道產出 v3.3 格式 latest_classification.json  
> **程式檔案**: 23 個 (Python 17 + Go 2 + Rust 2 + TypeScript 2)  
> **代碼行數**: 12,063 行  
> **能力數**: 201 flows (23.9%) - 最大模組  
> **支援語言**: Python, Go, Rust, TypeScript (4 語言 AST 分析)

**導航**: [← 返回 AIVA Core](../README.md)

---

> **🔔 重要更新 (2025-12-15)**: 分析結果輸出路徑已重構！  
> 📖 查看 [更新摘要](python_tools/UPDATE_SUMMARY.md) | [詳細變更日誌](python_tools/CHANGELOG_PATH_MIGRATION.md) | [使用指南](python_tools/README.md#輸出路徑配置)

## 📑 目錄

- [🎯 模組概述](#-模組概述)
- [🏗️ 目錄結構](#️-目錄結構)
- [🛠️ 核心工具套件](#️-核心工具套件)
  - [Python 工具套件](#python-工具套件)
  - [TypeScript 工具](#typescript-工具)
  - [Go 工具](#go-工具)
  - [Rust 工具](#rust-工具)
  - [Self-Healing 自我修復模組](#self-healing-自我修復模組)
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
│   ⚠️ **重要：輸出路徑架構變更 (2025-12-15)**
│   
│   **新路徑結構**: services/integration/analysis_data/{module}/{category}/
│   - module: core, features, scan, integration
│   - category: capabilities, flows, classifications
│   
│   **舊路徑**: services/integration/data/internal_exploration/ (保留)
│   **新路徑**: services/integration/analysis_data/ (統一儲存)
│   
│   詳見: 
│   - [Python 工具實作說明](python_tools/README.md#輸出路徑配置)
│   - [統一路徑文檔](../../../docs/UNIFIED_OUTPUT_PATHS.md)
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

### 環境準備

```powershell
# 設定 Python 環境變數 (Python 工具必需)
$env:PYTHONPATH="C:\D\fold7\AIVA-git\services\common;C:\D\fold7\AIVA-git\services\core"
```

### 選擇適合的工具

根據你的目標語言選擇對應工具:

| 分析目標 | 推薦工具 | 快速開始命令 |
|---------|---------|------------|
| **Python 代碼** | Python 工具套件 | [查看 Python 工具文檔](python_tools/README.md#-快速開始) |
| **TypeScript/JavaScript** | TypeScript 工具 | [查看 TypeScript 工具文檔](typescript_tools/README.md#-快速開始) |
| **Go 代碼** | Go 工具 | [查看 Go 工具文檔](go_tools/README.md#-快速開始) |
| **Rust 代碼** | Rust 工具 | [查看 Rust 工具文檔](rust_tools/README.md#-快速開始) |
| **代碼健康度診斷** | Self-Healing 模組 | [查看 Self-Healing 文檔](self_healing/README.md#-快速開始) |

### 典型工作流程

1. **代碼分析階段** - 使用對應語言的工具生成流程圖
2. **結果檢視階段** - 查看生成的 Mermaid 圖和 JSON 報告
3. **問題診斷階段** - 使用 Self-Healing 模組檢測健康度問題
4. **持續整合階段** - 整合到 CI/CD 流程中自動化執行

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

### 工具覆蓋範圍

| 語言 | 工具狀態 | 代碼行數 | 核心功能 |
|------|---------|---------|---------|
| Python | ✅ 完整 | ~3000 行 | 4 個獨立模組 |
| TypeScript | ✅ 完整 | 769 行 | 統一工具 |
| Go | ✅ 完整 | 782 行 | 統一工具 |
| Rust | ✅ 完整 | 739 行 | 統一工具 |
| Self-Healing | ✅ 完整 | ~1000 行 | 3 個分析器 |

### 分析能力

- **支援語言**: 4 種主流語言 (Python/TypeScript/Go/Rust)
- **輸出格式**: Mermaid 圖 + JSON 報告 + CLI 腳本
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

**版本**: v10.0.0  
**最後更新**: 2025-12-10  
**維護者**: AIVA Team  
**授權**: AIVA 專案授權條款
