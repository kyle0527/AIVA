# AIVA 歸檔目錄索引

## 📁 目錄

- [📂 目錄結構](#目錄結構)
  - [01_completed_projects](#01_completed_projects)
  - [02_deprecated_code](#02_deprecated_code)
  - [03_historical_reports](#03_historical_reports)
  - [04_scripts_completed](#04_scripts_completed)
  - [05_backups](#05_backups)
  - [06_documentation_archive](#06_documentation_archive)
  - [07_configuration_archive](#07_configuration_archive)
  - [08_tool_archive](#08_tool_archive)
  - [09_integration_archive](#09_integration_archive)
- [📝 使用說明](#使用說明)

---

此目錄包含 AIVA 專案的歷史文件、已完成項目和棄用代碼的歸檔。

## 目錄結構

- **01_completed_projects/**: 已完成的項目和功能
- **02_deprecated_code/**: 已棄用的代碼和組件
- **03_historical_reports/**: 歷史報告和分析文件
- **04_scripts_completed/**: 已完成的腳本和工具
- **05_backups/**: 重要文件的備份
- **06_documentation_archive/**: 歸檔的文件檔案
- **07_configuration_archive/**: 歸檔的配置文件
- **08_tool_archive/**: 歸檔的工具和實用程序
- **09_integration_archive/**: 集成相關的歷史文件

## 使用說明

1. 將完成的項目移動到對應的目錄中
2. 為每個歸檔項目添加時間戳
3. 保持清晰的命名約定
4. 定期清理和組織歸檔內容

更新時間: 2026-02-11
**檢查日期**: 2026年2月11日
**狀態**: ✅ 目錄結構已驗證，9個子目錄均存在

---

## 最近封存紀錄

### 2026-02-11: 整合層清理
- 封存 `services/integration/alembic/` (未使用的 PostgreSQL 遷移)
- 封存 4 個已廢棄的 Manager 類別:
  - `minimal_manifest.py` (已由自動產出取代)
  - `scanner_manager.py` (請使用 WebAttackManager)
  - `postex_manager.py` (請使用 PostExDetector)
  - `authn_manager.py` (請直接調用 Go 引擎)
- 詳見: [09_integration_archive/ARCHIVE_INDEX.md](09_integration_archive/ARCHIVE_INDEX.md)
