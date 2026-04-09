# AIVA 歸檔目錄索引

## 📁 目錄

- [📂 目前目錄結構](#目前目錄結構)
  - [09_integration_archive](#09_integration_archive)
  - [services](#services)
- [📝 使用說明](#使用說明)

---

此目錄包含 AIVA 專案的歷史文件、已完成項目和棄用代碼的歸檔。

## 目前目錄結構

- **09_integration_archive/**: 集成相關的歷史文件（廢棄的 Manager 類別、Alembic 遷移）
- **services/reports/2026-02/**: 2026-02 服務架構分析與報告

## 使用說明

1. 將完成的項目移動到對應的目錄中
2. 為每個歸檔項目添加時間戳
3. 保持清晰的命名約定
4. 定期清理和組織歸檔內容

更新時間: 2026-04-09
**狀態**: ✅ 已清理過時檔案（2026-04-09）

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

### 2026-04-09: 歸檔清理
刪除以下已無殘餘價值的目錄（共 266 個檔案，約 5.9MB）：
- `03_historical_reports/` — CLI 診斷、備份分析、舊版測試報告（2026-01/02）
- `docs/` — Mock 移除報告、修復報告（2026-01，已完成動作記錄）
- `06_documentation_archive/` — 舊版 CLI 文件（2026-01/02）
- `07_documentation_archive/` — 舊版 CLI 指南（2026-01）
- `docs_20260310/` — 2026-03-10 已過期的架構/技術文件（含 9 個 Nov-2025 JSON 測試輸出）
- `guides_20260310/` — 2026-03-10 已過期的指南
- `cognitive_core_cleanup_20260405/` — April 2026 清理歸檔的廢棄 Python 程式碼
- `core_capabilities_cleanup_20260405/` — 同上
- `service_backbone_cleanup_20260405/` — 同上
- `base_feature_infrastructure/` — 已被取代的基礎設施程式碼
- `validation/` — 舊版驗證腳本（已歸檔）
- `services/core/aiva_core/fixed_issues/` — 已修復的 issue 記錄
- `services/core/aiva_core/cleanup_reports/` — 已完成的清理報告
- `services/core/aiva_core/reports/` — 舊版 issue 報告（2026-01）
- `services/core/aiva_core/task_planning/` — 廢棄的 mode_manager.py
- `services/reports/2026-02/classification_results/analysis_results/` — 重複資料
- `services/reports/2026-02/classification_results/internal_exploration/` — 重複資料
