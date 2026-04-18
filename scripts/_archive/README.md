# 🗃️ 歷史存檔目錄 (_archive)

> **所屬層級**: `scripts/_archive/`
> **上一層級**: [Scripts 根目錄](../README.md)
> **檔案數量**: 13 個腳本（含子目錄）

---

## 📋 目錄概述

此目錄包含從 `scripts/` 主目錄歸檔的老舊、過時或功能已整合的工具腳本。
保留這些腳本主要是作為**參考、備份與歷史追溯**的用途。這些腳本**不再被主動維護**。

---

## 🗂️ 子目錄導覽

| 子目錄 | 說明 |
|--------|------|
| `analysis/` | 舊版系統依賴與服務結構分析腳本 |
| `migration/` | 過去用於檔案遷移、連結修復的單次性腳本 |
| `utilities/` | 舊版文件與依賴生成腳本 |

---

## 📂 腳本歸檔列表

### 📁 根目錄歸檔
- `analyze_weights.py`
- `check_missing_files.py`
- `classify_aiva_core_only.py`
- `extract_internal_commands.py`

### 📁 `analysis/`
- `_analyze_dependencies_detail.py`
- `_analyze_services_structure.py`
- `_analyze_services_structure_deep.py`

### 📁 `migration/`
- `_execute_file_moves.py`
- `_find_missing_files.py`
- `_fix_moved_file_links.py`
- `_fix_wrong_links.py`

### 📁 `utilities/`
- `_generate_complete_guide.py`
- `_generate_dependencies_guide.py`

---

## 💡 使用建議

- **不建議直接執行**：這些腳本可能依賴過時的路徑結構或已不存在的套件。
- 若有需要相關功能，請優先查看 `scripts/core/` 或 `scripts/common/` 中的現代化替代工具。
