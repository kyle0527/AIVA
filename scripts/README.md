# 📜 AIVA Scripts - 按五大模組重組# 🔧 AIVA 腳本集合



## 🎯 新組織結構本目錄包含 AIVA 專案的各種自動化腳本和工具,支援系統的部署、測試、維護和監控等各項功能。



### 🏠 **common/** - 通用系統腳本## 📁 目錄結構

- **launcher/** - 系統啟動器

- **deployment/** - 部署腳本  ### 🚀 launcher/ - 啟動器腳本 (3個)

- **setup/** - 環境設置- **aiva_launcher.py** ✅ - AIVA 統一啟動入口

- **maintenance/** - 系統維護- **start_ai_continuous_training.py** ✅ - AI 持續訓練啟動器

- **validation/** - 套件驗證- **smart_communication_selector.py** ✅ - 智能通訊選擇器



### 🧠 **core/** - 核心模組腳本### 🧪 testing/ - 測試相關腳本 (8個)

- **reporting/** - 核心業務報告- **comprehensive_test.py** ✅ - 全功能測試腳本

- **ai_system_connectivity_check.py** ✅ - AI 系統連接檢查

### 🔍 **scan/** - 掃描模組腳本  - **aiva_full_worker_live_test.py** ✅ - 完整工作者實時測試

- **reporting/** - 掃描結果報告- **aiva_module_status_checker.py** ✅ - 模組狀態檢查器

- **aiva_system_connectivity_sop_check.py** ✅ - 系統連接 SOP 檢查

### 🔗 **integration/** - 整合模組腳本- **enhanced_real_ai_attack_system.py** ✅ - 增強型 AI 攻擊測試

- **cross_language_bridge.py** - 跨語言橋接- **juice_shop_real_attack_test.py** ✅ - Juice Shop 攻擊測試

- **ffi_integration.py** - FFI 整合- **real_attack_executor.py** ✅ - 真實攻擊執行器

- **graalvm_integration.py** - GraalVM 整合  

- **wasm_integration.py** - WebAssembly 整合### ✅ validation/ - 驗證相關腳本 (1個)

- **reporting/** - 整合狀態報告- **aiva_package_validator.py** ✅ - 套件驗證器



### ⚙️ **features/** - 功能模組腳本### 🔗 integration/ - 整合相關腳本 (4個)

- **conversion/** - 文檔轉換工具- **cross_language_bridge.py** ✅ - 跨語言橋接器

- **ffi_integration.py** ✅ - FFI 整合

## 🚀 使用指南- **graalvm_integration.py** ✅ - GraalVM 整合

- **wasm_integration.py** ✅ - WebAssembly 整合

### 系統部署

```bash### 📊 reporting/ - 報告生成腳本 (3個)

# 環境設置- **aiva_enterprise_security_report.py** ✅ - 企業安全報告生成器

scripts/common/setup/setup_env.bat- **final_report.py** ✅ - 最終報告生成器

- **aiva_crosslang_unified.py** ✅ - 跨語言統一報告工具

# 啟動系統

scripts/common/launcher/aiva_launcher.py### 🔄 conversion/ - 轉換工具腳本 (1個)

- **docx_to_md_converter.py** ✅ - DOCX 轉 Markdown 轉換器

# 部署服務

scripts/common/deployment/start_all.ps1### 🚀 deployment/ - 部署腳本 (6個)

```- **start_all.ps1** ✅ - 啟動所有服務

- **start_all_multilang.ps1** ✅ - 啟動多語言服務

### 模組測試- **start_dev.bat** ✅ - 開發環境啟動

```bash- **start_ui_auto.ps1** ✅ - 自動啟動 UI

# 核心模組測試 - 在 testing/core/- **stop_all.ps1** ✅ - 停止所有服務

# 掃描模組測試 - 在 testing/scan/  - **stop_all_multilang.ps1** ✅ - 停止多語言服務

# 整合模組測試 - 在 testing/integration/

# 功能模組測試 - 在 testing/features/### ⚙️ setup/ - 環境設置腳本 (2個)

```- **setup_env.bat** ✅ - 環境設置

- **setup_multilang.ps1** ✅ - 多語言環境設置

---

### 🔍 maintenance/ - 維護腳本 (9個)

**重組完成**: 2025-10-24  - **check_status.ps1** ✅ - 檢查系統狀態

**架構**: 五大模組對應- **diagnose_system.ps1** ✅ - 系統診斷
- **health_check_multilang.ps1** ✅ - 多語言健康檢查
- **generate_project_report.ps1** ✅ - 生成專案報告
- **generate_stats.ps1** ✅ - 生成統計資料
- **generate_tree_ultimate_chinese.ps1** ✅ - 生成專案樹狀圖
- **fix_import_paths.py** ✅ - 修復導入路徑
- **optimize_core_modules.ps1** ✅ - 優化核心模組
- **system_repair_tool.py** ✅ - 系統修復工具

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
