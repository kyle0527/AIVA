# 🤖 Core - AI 核心分析腳本

> **所屬層級**: `scripts/core/`
> **上一層級**: [Scripts 根目錄](../README.md)
> **檔案數量**: 13 個腳本

---

## 📋 目錄概述

Core 目錄包含 AIVA 系統的核心分析工具，專門用於能力分析、流程追蹤、模組連接性檢測等。這些工具幫助開發者與架構師洞察系統的內部結構。

---

## 📂 腳本詳細說明

### 🔍 能力分析工具

| 腳本 | 功能說明 |
|------|----------|
| `analyze_attack_scan_capabilities.py` | 分析攻擊掃描相關的能力定義與覆蓋率 |
| `analyze_module_capabilities_distribution.py` | 分析各模組的能力分布情況 |
| `analyze_uncovered_capabilities.py` | 識別尚未被覆蓋的能力項目 |
| `classify_existing_capabilities.py` | 對現有能力進行分類整理 |
| `classify_internal_external_capabilities.py` | 區分內部與外部能力來源 |
| `run_capability_analysis.py` | 執行完整的能力分析流程總結 |

### 🔗 模組連接分析

| 腳本 | 功能說明 |
|------|----------|
| `analyze_module_connectivity.py` | 分析模組間的連接關係與依賴 |
| `analyze_unique_endpoints_by_module.py` | 識別各模組的獨特端點 |

### 📊 流程分析工具

| 腳本 | 功能說明 |
|------|----------|
| `analyze_multi_path_flows.py` | 分析多路徑流程的執行情況 |
| `analyze_unknown_flows.py` | 識別並分析未知的流程路徑 |
| `enrich_flows_with_capabilities.py` | 將能力資訊注入流程定義 |
| `find_executable_flows.py` | 搜尋可執行的流程定義 |

### 🛠️ 外部模組分析

| 腳本 | 功能說明 |
|------|----------|
| `analyze_rust_output.py` | 分析 Rust 模組引擎的輸出結果格式 |

---

## 🚀 快速開始

### 執行能力分析
```bash
python run_capability_analysis.py
```

### 分析模組連接性
```bash
python analyze_module_connectivity.py
```

---

## 💡 最佳實踐

- 所有腳本設計為獨立執行，可透過終端機直接呼叫。
- 部分腳本依賴 `data/internal_exploration/` 中的分類 JSON 資料夾，請確保該目錄擁有最新資料。
