# AIVA Internal Exploration 操作手冊

> **版本**: v11.0.0
> **更新日期**: 2026-01-21
> **狀態**: ✅ 生產就緒  
> **架構**: 雙層架構 (語言層 + 業務邏輯層)
> **代碼品質**: Zero Errors - 所有工具通過 Pylance + SonarLint 檢查

## 📑 目錄

- [系統概述](#系統概述)
- [快速開始](#快速開始)
- [核心組件](#核心組件)
- [使用指南](#使用指南)
- [故障排除](#故障排除)
- [開發指南](#開發指南)

---

## 系統概述

### 架構說明

AIVA Internal Exploration 是一個**雙層自動化管道系統**，用於分析、分類和執行 AIVA 系統的內部能力流程。

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

## 快速開始

### 環境準備

```powershell
# Python 環境變數（必要）
# 請根據實際路徑調整
$env:PYTHONPATH="C:\Path\To\AIVA\services\common;C:\Path\To\AIVA\services\core"
```

### 方式一：Internal CLI 工作流程（AI Core 模組）

```bash
cd services/core/aiva_core/internal_exploration

# 1. AST 分析 - 使用 Python 工具
python python_tools/aiva_flow_analyzer.py --target ../ --output ./analysis

# 2. 分類 - 讀取 analysis_results.json
python aiva_internal_classifier.py

# 3. 執行 - 讀取 classification_data.json
python aiva_internal_executor.py --flow 11
```

### 方式二：External CLI 工作流程（Features/Scan 模組）

```bash
cd services/core/aiva_core/internal_exploration

# 1. AST 分析 (以 Python 模組為例)
python python_tools/aiva_flow_analyzer.py --target ../../../features/function_xss

# 2. 整合分類
python aiva_external_classifier.py

# 3. 多語言執行
python aiva_external_executor.py --lang python --flow 1
```

---

## 核心組件

### 1. 語言層工具 (AST Analyzers)

負責解析各語言源碼並生成統一格式的 JSON 數據流。

- **Python**: `python_tools/aiva_flow_analyzer.py`
- **Go**: `go_tools/go2mermaid.go`
- **Rust**: `rust_tools/src/main.rs`
- **TypeScript**: `typescript_tools/ts2mermaid.ts`

### 2. 業務邏輯層 - Internal (AI Core)

負責 AI Core 內部模組的分類與直接執行。

- **Classifier**: `aiva_internal_classifier.py`
  - 讀取 `analysis_results.json`
  - 輸出 `latest_classification.json`
- **Executor**: `aiva_internal_executor.py`
  - 直接導入 Python 類別並執行
  - 用於 AI Core 內部流程

### 3. 業務邏輯層 - External (Features/Scan)

負責外部功能模組的分類與 subprocess 執行。

- **Classifier**: `aiva_external_classifier.py`
  - 整合多語言分析結果
  - 輸出 `classification_data.json`
- **Executor**: `aiva_external_executor.py`
  - 通過 subprocess 調用外部 CLI
  - 支援 Python, Go, Rust, TypeScript

---

## 使用指南

### 場景 1：分析 AI Core 並更新能力

當修改了 AI Core 代碼後，需要更新內部能力列表：

```bash
# 1. 重新分析
python python_tools/aiva_flow_analyzer.py --target ../ --output ./analysis

# 2. 重新分類
python aiva_internal_classifier.py

# 3. 驗證新能力
python aiva_internal_executor.py --list
```

### 場景 2：測試 External Feature (如 XSS)

```bash
# 1. 分析目標模組
python python_tools/aiva_flow_analyzer.py --target ../../../features/function_xss

# 2. 更新分類
python aiva_external_classifier.py

# 3. 執行測試
python aiva_external_executor.py --lang python --flow 1
```

### 場景 3：多語言模組分析 (Go/Rust/TS)

請參考 `services/core/aiva_core/internal_exploration/README.md` 中的詳細指令，各語言工具使用方式略有不同。

---

## 故障排除

### 問題 1：找不到模組

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'aiva_common'
```

**解決方法**:
確保 `PYTHONPATH` 包含 `services/common` 和 `services/core`。

### 問題 2：分類結果為空

**解決方法**:
1. 檢查 `analysis_results.json` 是否生成。
2. 檢查 `aiva_internal_classifier.py` (或 external) 的輸入路徑配置是否正確。

### 問題 3：Rust/Go 工具執行失敗

**解決方法**:
確保已安裝相應的編譯器 (`go`, `cargo`) 和依賴。
Go 需要 `go mod tidy`。
Rust 需要 `cargo build`。

---

## 開發指南

### 添加新的分類規則

編輯 `aiva_internal_classifier.py` 或 `aiva_external_classifier.py`，在 `MODULES` 字典中添加新模組映射。

### JSON Schema 規範

所有語言工具輸出格式應遵循 JSON Schema v3.3 (詳見 `internal_exploration/README.md`)。

```json
{
  "metadata": {
    "tool": "tool_name",
    "version": "3.0",
    "schema_version": "3.3"
  },
  "flows": [...]
}
```

---

**最後更新**: 2026-01-21
**維護者**: AIVA Development Team
