# 🛠️ Utilities - 實用工具腳本

> **版本**: v1.0  
> **更新日期**: 2026年1月6日  
> **檔案數量**: 14 個腳本

---

## 📋 目錄概述

Utilities 目錄包含 AIVA 系統的各種實用工具，包括驗證檢查、生成工具、資料庫遷移等輔助腳本。

---

## 📂 腳本說明

### ✅ 驗證檢查工具

| 腳本 | 功能說明 |
|------|----------|
| `check_data_version.py` | 檢查資料版本的一致性 |
| `check_latest_classification.py` | 驗證最新分類資料的正確性 |
| `check_path_format.py` | 檢查路徑格式是否符合規範 |
| `verify_orchestrator.py` | 驗證協調器的運作狀態 |
| `verify_system_authenticity.py` | 驗證系統元件的真實性 |
| `health_check.py` | 系統健康狀態檢查 |

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
| `backfill_capabilities.py` | 回填能力資料至資料庫 |
| `create_capability_db.py` | 建立能力資料庫結構 |

### 🧪 測試工具

| 腳本 | 功能說明 |
|------|----------|
| `select_test_flows.py` | 選擇測試用的流程樣本 |

### 🔧 維護工具

| 腳本 | 功能說明 |
|------|----------|
| `DELETE_OPTIONS.ps1` | 清理選項的 PowerShell 腳本 |

---

## 🚀 使用方式

### 驗證檢查

```bash
# 系統健康檢查
python health_check.py

# 檢查資料版本
python check_data_version.py

# 驗證最新分類
python check_latest_classification.py
```

### 生成文檔

```bash
# 生成能力參考
python generate_ai_capability_reference.py

# 生成改進計畫
python generate_bidirectional_improvement_plan.py
```

### 資料庫操作

```bash
# 建立能力資料庫
python create_capability_db.py

# 回填能力資料
python backfill_capabilities.py
```

---

## 📝 注意事項

- 驗證工具建議定期執行以確保系統完整性
- 生成工具的輸出通常存放於 `reports/` 或 `docs/` 目錄
- 資料庫工具操作前建議先備份現有資料
