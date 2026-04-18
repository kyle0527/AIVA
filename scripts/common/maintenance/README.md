# 🛠️ 維護工具 (Maintenance)

> **所屬層級**: `scripts/common/maintenance/`
> **上一層級**: [Common 通用服務](../README.md)
> **檔案數量**: 4 個腳本

---

## 📋 目錄概述

此目錄包含 AIVA 系統的深入維護與分析腳本。這些工具提供程式碼庫健康檢查、依賴路徑修復、報表生成及核心模組優化，屬於最底層詳細的技術工具集。

---

## 📂 腳本詳細說明

### 1. `fix_import_paths.py`
**功能：**
自動修復專案中因重構或移動檔案而導致的無效或錯誤的 Python import 路徑。
**使用情境：**
當您在執行 Python 工具時遇到 `ModuleNotFoundError`，可執行此腳本來掃描並修復導入路徑。

### 2. `generate_project_report.ps1`
**功能：**
自動掃描 AIVA 專案結構，產出專案的架構報告。
**輸出：**
包含資料夾大小、檔案數統計、語言分佈比例等整體專案健康與規模報表。

### 3. `generate_tree_ultimate_chinese.ps1`
**功能：**
進階版專案目錄樹生成器。
**特色：**
- 支援產出帶有中文註解的 ASCII 樹狀圖
- 可追蹤版本差異，標示出新增/修改/刪除的檔案（搭配歷史快照比對）

### 4. `optimize_core_modules.ps1`
**功能：**
針對 AIVA Core 核心模組執行自動化的重構與效能優化建議。
**特色：**
- 統一 AI 引擎引用
- 重構依賴注入架構
- 提示效能監控的掛載點

---

## 🚀 快速開始

### 修復 Python Import 路徑
```bash
python fix_import_paths.py
```

### 產生專案樹狀圖 (附中文註解)
```powershell
.\generate_tree_ultimate_chinese.ps1 -ShowColorInTerminal -AddChineseComments
```

### 執行核心模組優化
```powershell
.\optimize_core_modules.ps1 all -DryRun
```

---

## 💡 最佳實踐

- 定期於每週或每次大版本發布前，執行 `generate_project_report.ps1` 與 `generate_tree_ultimate_chinese.ps1` 進行紀錄。
- 如果移動了 `services/core` 下的任何子模組，必須執行 `fix_import_paths.py`。
- `optimize_core_modules.ps1` 建議在開發分支上執行，並進行完整單元測試以防破壞既有邏輯。
