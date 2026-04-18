# 🛠️ Utilities - 實用工具腳本

> **所屬層級**: `scripts/utilities/`
> **上一層級**: [Scripts 根目錄](../README.md)
> **檔案數量**: 12 個腳本

---

## 📋 目錄概述

Utilities 目錄包含 AIVA 系統的特定實用工具，專注於資料驗證、文件自動生成、資料庫工具與一次性的操作腳本。此目錄為相對獨立的工具集。

---

## 📂 腳本詳細說明

### ✅ 驗證檢查工具

| 腳本 | 功能說明 |
|------|----------|
| `check_data_version.py` | 檢查資料版本的一致性 |
| `check_latest_classification.py` | 驗證最新分類資料的正確性 |
| `check_path_format.py` | 檢查路徑格式是否符合系統規範 |

### 📝 生成工具

| 腳本 | 功能說明 |
|------|----------|
| `generate_ai_capability_reference.py` | 生成 AI 能力參考文檔 |
| `generate_bidirectional_improvement_plan.py` | 生成雙向改進計畫 |
| `generate_direct_connection_plan.py` | 生成直接連接計畫 |
| `fill_cli_documentation.py` | 自動填充 CLI 文檔內容 |

### 🗄️ 資料庫工具

| 腳本 | 功能說明 |
|------|----------|
| `backfill_capabilities.py` | 回填能力資料至資料庫 (PostgreSQL/Chroma) |
| `create_capability_db.py` | 建立能力資料庫表結構 |

### 🧪 其他工具

| 腳本 | 功能說明 |
|------|----------|
| `select_test_flows.py` | 選擇測試用的流程樣本 |
| `DELETE_OPTIONS.ps1` | 清理選項的 PowerShell 腳本 |

---

## 🚀 快速開始

### 執行資料庫回填
```bash
python create_capability_db.py
python backfill_capabilities.py
```

### 產生改進計畫
```bash
python generate_bidirectional_improvement_plan.py
```

---

## 💡 最佳實踐

- 生成工具的輸出通常存放於專案根目錄的 `reports/` 或 `docs/` 目錄。
- 執行資料庫操作前，請確保本地的 PostgreSQL 或對應依賴服務已啟動。
