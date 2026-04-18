# 📜 AIVA Scripts - 腳本工具集

> **版本**: v8.0  
> **層級**: 根目錄 (`scripts/`)
> **狀態**: 目錄結構整合完成，採用階層式設計。

---

## 📋 概述

AIVA Scripts 是 AIVA 系統的營運、診斷、與分析中心。
這裡包含了所有用於啟動系統、排查錯誤、執行測試、與產生系統架構報告的命令列與批次腳本。

為了保持清晰，工具依照其核心功能被歸類於 4 個主要的子模組目錄中。此外，根目錄下直接提供最常使用的系統啟動捷徑與跨模組協調腳本。

---

## 🗂️ 子模組目錄索引

請點擊下方連結進入各子目錄以獲取更詳細的說明：

| 目錄 | 說明 | 連結 |
|------|------|------|
| 🤖 **core/** | AI 核心與模組連接性的深度分析腳本 | [Core 說明](./core/README.md) |
| 🔗 **common/** | 通用服務、啟動腳本、CLI 介面及維護工具 | [Common 說明](./common/README.md) |
| 🛠️ **utilities/** | 特定資料處理、生成及獨立驗證工具 | [Utilities 說明](./utilities/README.md) |
| 🗃️ **_archive/** | 歷史過時與棄用的腳本存檔區 | [Archive 說明](./_archive/README.md) |

---

## 📂 根目錄腳本導覽

為了方便開發者快速存取，根目錄保留了主要的系統進入點與高層次的協調腳本。

### 🚀 系統啟動捷徑
這些腳本通常是封裝了底層邏輯，讓您可以一鍵啟動 AIVA 環境：
- `啟動AIVA系統.bat` / `啟動AIVA.ps1` - 啟動 AIVA 完整系統或核心介面
- `start_aiva_core.ps1` - 啟動 AIVA 核心 API
- `start_dashboard.py` - 啟動 AIVA 儀表板
- `啟動能力選單.bat` / `啟動外部能力選單.bat` - 啟動各類能力選單

### ⚙️ 執行與協調工具
- `planning_execution_coordinator.py` - 負責策劃與執行的協調器
- `scan_with_constraints.py` - 執行受限安全掃描的工具
- `validate_rag_p1.py` - RAG 模組的 P1 等級驗證
- `執行Flow.bat` / `預覽Flow.bat` - 流程執行與預覽批次檔
- `分類外部模組.bat` / `執行外部模組.bat` - 外部模組管理工具
- `啟動統一執行器.bat` / `快速執行能力.bat` - 執行器介面啟動

---

## 📚 相關文檔

- [FIX_MODULE_IMPORT_GUIDE.md](./FIX_MODULE_IMPORT_GUIDE.md) - 模組導入修復指南
