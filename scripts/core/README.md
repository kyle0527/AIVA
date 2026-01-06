# 🤖 Core - AI 核心分析腳本

> **版本**: v1.0  
> **更新日期**: 2026年1月6日  
> **檔案數量**: 14 個腳本

---

## 📋 目錄概述

Core 目錄包含 AIVA 系統的核心分析工具，專門用於能力分析、流程追蹤、模組連接性檢測等 AI 核心功能。

---

## 📂 腳本說明

### 🔍 能力分析工具

| 腳本 | 功能說明 |
|------|----------|
| `analyze_attack_scan_capabilities.py` | 分析攻擊掃描相關的能力定義與覆蓋率 |
| `analyze_module_capabilities_distribution.py` | 分析各模組的能力分布情況 |
| `analyze_uncovered_capabilities.py` | 識別尚未被覆蓋的能力項目 |
| `classify_existing_capabilities.py` | 對現有能力進行分類整理 |
| `classify_internal_external_capabilities.py` | 區分內部與外部能力來源 |

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

### 🛠️ 其他核心工具

| 腳本 | 功能說明 |
|------|----------|
| `analyze_rust_output.py` | 分析 Rust 模組的輸出結果 |
| `run_capability_analysis.py` | 執行完整的能力分析流程 |
| `scan_bizlogic_real.py` | 掃描業務邏輯的真實實現 |

---

## 🚀 使用方式

```bash
# 執行能力分析
python run_capability_analysis.py

# 分析模組連接性
python analyze_module_connectivity.py

# 分析未覆蓋的能力
python analyze_uncovered_capabilities.py
```

---

## 📝 注意事項

- 所有腳本需在 AIVA 專案根目錄執行
- 部分腳本依賴 `data/internal_exploration/` 的分類資料
- 建議先執行 `run_capability_analysis.py` 建立基礎分析
