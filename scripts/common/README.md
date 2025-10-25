# 🔧 AIVA 通用腳本集合 (Common Scripts)

本目錄是 AIVA 專案的通用腳本集合，包含系統層級的自動化腳本和工具，支援跨模組的部署、啟動、維護和驗證等功能。

## 🎯 目錄定位

`scripts/common/` 是 AIVA 五大模組架構中的 **通用系統腳本** 區域，提供：
- 🏠 跨模組通用的系統級腳本
- � 統一的系統啟動和部署工具
- 🔍 系統維護和監控腳本
- ✅ 項目驗證和質量保證工具

## �📁 目錄結構

### 🚀 launcher/ - 啟動器腳本 (3個)
系統統一啟動入口，支援不同啟動模式和場景

- **aiva_launcher.py** ✅ - AIVA 五大模組統一啟動器
- **start_ai_continuous_training.py** ✅ - AI 持續學習系統啟動器
- **smart_communication_selector.py** ✅ - 智能通訊模式選擇器

### � deployment/ - 部署腳本 (6個)
自動化系統部署和服務管理腳本

- **start_all.ps1** ✅ - 一鍵啟動所有 AIVA 服務
- **start_all_multilang.ps1** ✅ - 啟動多語言支援服務
- **start_dev.bat** ✅ - 開發環境快速啟動
- **start_ui_auto.ps1** ✅ - 自動啟動 Web UI 介面
- **stop_all.ps1** ✅ - 停止所有運行服務
- **stop_all_multilang.ps1** ✅ - 停止多語言服務

### ⚙️ setup/ - 環境設置腳本 (2個)
初始化和配置開發環境

- **setup_env.bat** ✅ - Python 環境和依賴包設置
- **setup_multilang.ps1** ✅ - 多語言環境配置 (Go, Rust, Node.js)

### 🔍 maintenance/ - 維護腳本 (10個)
系統監控、診斷和維護工具

- **check_status.ps1** ✅ - 系統服務狀態檢查
- **diagnose_system.ps1** ✅ - 系統診斷和問題檢測
- **health_check_multilang.ps1** ✅ - 多語言服務健康檢查
- **generate_project_report.ps1** ✅ - 生成項目狀態報告
- **generate_stats.ps1** ✅ - 生成項目統計數據
- **generate_tree_ultimate_chinese.ps1** ✅ - 生成中文項目樹狀圖
- **generate_tree_ultimate_chinese_backup.ps1** ✅ - 樹狀圖生成備份版本
- **fix_import_paths.py** ✅ - Python 導入路徑自動修復
- **optimize_core_modules.ps1** ✅ - 核心模組性能優化
- **system_repair_tool.py** ✅ - 系統自動修復工具

### ✅ validation/ - 驗證腳本 (1個)
項目完整性和質量驗證工具

- **aiva_package_validator.py** ✅ - AIVA 補包完整性驗證器

---

## 🚀 快速使用

### 首次部署
```powershell
.\scripts\setup\setup_env.bat
.\scripts\setup\setup_multilang.ps1
.\scripts\deployment\start_all_multilang.ps1
```

### 測試驗證
```powershell
python scripts\testing\comprehensive_test.py
python scripts\validation\aiva_package_validator.py
```

### 系統維護
```powershell
.\scripts\maintenance\check_status.ps1
.\scripts\maintenance\diagnose_system.ps1
python scripts\maintenance\system_repair_tool.py
```

---

## 📊 統計資訊
- **總計**: 37 個腳本
- **Python**: 22 個
- **PowerShell**: 13 個  
- **Batch**: 2 個

---

**維護者**: AIVA DevOps Team  
**最後更新**: 2025-10-24  
**驗證狀態**: ✅ 所有 37 個腳本已驗證
