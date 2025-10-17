# 掃描模組歸檔文檔

> **歸檔日期**: 2025-10-17  
> **原因**: 文檔整合統一，保留 `SCAN_MODULE_GUIDE.md` 作為唯一指南

## 📁 歸檔內容

### 🏗️ 架構設計文檔
- `scan_orchestrator_migration_plan.md` - 掃描編排器遷移計劃
- `migration_plan_dynamic_engine.md` - 動態引擎遷移計劃  
- `multi_language_architecture_plan.md` - 多語言架構計劃
- `persistent_queue_design.md` - 持久化佇列設計

### 📊 流程圖和可視化
- `01_scan_diagrams.html` - 掃描模組流程圖合集
- `scan_flow.md` - CLI 掃描流程文檔

### 🧪 測試文件
- `test_scan.ps1` - 掃描功能測試腳本

## 📋 整合狀態

所有重要內容已整合到 `services/scan/SCAN_MODULE_GUIDE.md`：

- ✅ **架構原則** - 統一的開發規範
- ✅ **技術架構** - 多語言掃描引擎設計
- ✅ **開發指南** - 完整的開發流程
- ✅ **編碼規範** - 跨語言一致性要求

## 🔄 移轉記錄

| 原始位置 | 歸檔位置 | 狀態 |
|---------|---------|------|
| `services/scan/aiva_scan/scan_orchestrator_migration_plan.md` | `_archive/scan_module_docs/` | ✅ 已移動 |
| `services/scan/aiva_scan/migration_plan_dynamic_engine.md` | `_archive/scan_module_docs/` | ✅ 已移動 |
| `services/scan/aiva_scan/multi_language_architecture_plan.md` | `_archive/scan_module_docs/` | ✅ 已移動 |
| `services/scan/aiva_scan/persistent_queue_design.md` | `_archive/scan_module_docs/` | ✅ 已移動 |
| `_out/combined_diagrams/01_scan_diagrams.html` | `_archive/scan_module_docs/` | ✅ 已移動 |
| `services/cli/generated/flow_diagrams/scan_flow.md` | `_archive/scan_module_docs/` | ✅ 已移動 |
| `tests/test_scan.ps1` | `_archive/scan_module_docs/` | ✅ 已移動 |

---

**注意**: 如需參考這些歷史文檔，請查閱此歸檔目錄。所有現行開發請參考 `services/scan/SCAN_MODULE_GUIDE.md`。