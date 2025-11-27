# 🔗 內部連結修正報告

---
**執行時間**: 2025年11月27日

## 📑 目錄

- [修正摘要](#修正摘要)
- [文件移動映射](#文件移動映射)
- [修正詳情](#修正詳情)

---

## 📊 修正摘要

- **檢查文件數**: 437
- **修改文件數**: 26
- **修正連結數**: 69

## 📂 文件移動映射

以下文件已移動到新位置:

- `services/scan/engines/rust_engine/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
- `services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
- `services/core/aiva_core/USAGE_GUIDE.md` → `docs/guides/services/aiva_core_USAGE_GUIDE.md`
- `services/features/DEVELOPMENT_STANDARDS.md` → `docs/development/services_DEVELOPMENT_STANDARDS.md`

## 🔧 修正詳情

### FINAL_COMPLETION_VERIFICATION.md

修正 6 個連結:

-  `DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `bash
# 在整合指南中搜尋
grep -n "playwright" services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md

# 或在 VS Code 中使用 Ctrl+F
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `bash
# 方法1: 在 VS Code 中預覽
code services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md

# 方法2: 在終端查看
cat services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md

# 方法3: 在瀏覽器查看
start services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `
主要文檔:
└─ services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md (31 KB)

數據文件:
└─ _node_modules_complete_inventory.json (42 KB)

分析報告:
├─ NODE_MODULES_ANALYSIS_REPORT.md (12 KB)
├─ NODE_MODULES_DELETION_DECISION_REPORT.md (14 KB)
└─ NODE_MODULES_CONSOLIDATION_REPORT.md (7 KB)

本報告:
└─ FINAL_COMPLETION_VERIFICATION.md (本文件)

工具腳本:
├─ _extract_node_modules_docs.py
├─ _generate_dependencies_guide.py
├─ _generate_complete_guide.py
├─ _verify_extraction.py
└─ _find_missing_files.py
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `
輸出文件: services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md
文件大小: 31,251 bytes (約 31 KB)
總行數: 1,427 行
涵蓋範圍: 100% (全部 439 個 MD 文件)
涵蓋套件: 229 個
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`

### MD_FILES_COMPLETE_CHECK_REPORT.md

修正 10 個連結:

-  `services\scan\engines\typescript_engine\DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `services\scan\engines\rust_engine\USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `services\features\DEVELOPMENT_STANDARDS.md` → `docs/development/services_DEVELOPMENT_STANDARDS.md`
-  `services\core\aiva_core\USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `reports\implementation\INTEGRATION_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `reports\architecture\PYTHON_ENGINE_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `reports\architecture\METRICS_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `reports\architecture\DEVELOPMENT_STANDARDS.md` → `docs/development/services_DEVELOPMENT_STANDARDS.md`
-  `reports\architecture\COORDINATOR_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `reports\architecture\USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### NODE_MODULES_CONSOLIDATION_REPORT.md

修正 6 個連結:

-  `DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `bash
# 1. 安裝新套件
npm install <package-name>

# 2. 重新提取文檔
python _extract_node_modules_docs.py

# 3. 重新生成指南
python _generate_dependencies_guide.py

# 4. 檢查更新
git diff services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `bash
# 在指南中搜尋特定套件
grep -n "playwright" services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `bash
# 1. 閱讀使用指南
cat services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md

# 2. 安裝所有依賴（如果還沒有）
cd services/scan/engines/typescript_engine
npm install

# 3. 開始開發
npm run dev
` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`

### SERVICES_MD_REORGANIZATION_PLAN.md

修正 9 個連結:

-  `powershell
Move-Item -Path "C:\D\fold7\AIVA-git\services/scan\engines\rust_engine\USAGE_GUIDE.md" -Destination "C:\D\fold7\AIVA-git\docs/guides/services/rust_engine_USAGE_GUIDE.md" -Force
Move-Item -Path "C:\D\fold7\AIVA-git\services/scan\engines\typescript_engine\DEPENDENCIES_GUIDE.md" -Destination "C:\D\fold7\AIVA-git\docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md" -Force
Move-Item -Path "C:\D\fold7\AIVA-git\services/core\aiva_core\USAGE_GUIDE.md" -Destination "C:\D\fold7\AIVA-git\docs/guides/services/aiva_core_USAGE_GUIDE.md" -Force
Move-Item -Path "C:\D\fold7\AIVA-git\services/features\DEVELOPMENT_STANDARDS.md" -Destination "C:\D\fold7\AIVA-git\docs/development/services_DEVELOPMENT_STANDARDS.md" -Force
` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `

**services/features\DEVELOPMENT_STANDARDS.md**
  - → ` → `docs/development/services_DEVELOPMENT_STANDARDS.md`
-  `
  - 原因: 服務使用指南應統一放在 docs/guides/services/

**services/core\aiva_core\USAGE_GUIDE.md**
  - → ` → `docs/guides/services/aiva_core_USAGE_GUIDE.md`
-  `
  - 原因: 服務使用指南應統一放在 docs/guides/services/

**services/scan\engines\typescript_engine\DEPENDENCIES_GUIDE.md**
  - → ` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `

**services/scan\engines\rust_engine\USAGE_GUIDE.md**
  - → ` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `features\DEVELOPMENT_STANDARDS.md` → `docs/development/services_DEVELOPMENT_STANDARDS.md`
-  `scan\engines\typescript_engine\DEPENDENCIES_GUIDE.md` → `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
-  `scan\engines\rust_engine\USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `core\aiva_core\USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### guides\README.md

修正 2 個連結:

-  [`development/METRICS_USAGE_GUIDE.md`](development/METRICS_USAGE_GUIDE.md) → [`development/METRICS_USAGE_GUIDE.md`](../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  `development/METRICS_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### services\features\README.md

修正 1 個連結:

-  [📊 **性能監控規範**](../../guides/development/METRICS_USAGE_GUIDE.md) → [📊 **性能監控規範**](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### services\scan\README.md

修正 1 個連結:

-  `COORDINATOR_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### services\scan\coordinators\README.md

修正 4 個連結:

-  [PYTHON_ENGINE_USAGE_GUIDE.md](./PYTHON_ENGINE_USAGE_GUIDE.md) → [PYTHON_ENGINE_USAGE_GUIDE.md](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  [COORDINATOR_USAGE_GUIDE.md](./COORDINATOR_USAGE_GUIDE.md) → [COORDINATOR_USAGE_GUIDE.md](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  [COORDINATOR_USAGE_GUIDE.md](./COORDINATOR_USAGE_GUIDE.md) → [COORDINATOR_USAGE_GUIDE.md](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  [完整指南 →](./COORDINATOR_USAGE_GUIDE.md) → [完整指南 →](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### services\integration\docs\README.md

修正 3 個連結:

-  [Features 模組整合指南](../../features/INTEGRATION_USAGE_GUIDE.md) → [Features 模組整合指南](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  [[Features 模組整合指南](../../features/INTEGRATION_USAGE_GUIDE.md) → [[Features 模組整合指南](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  `services/features/INTEGRATION_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### services\features\function_payload_generator\README.md

修正 1 個連結:

-  [功能模組標準](../DEVELOPMENT_STANDARDS.md) → [功能模組標準](../../../docs/development/services_DEVELOPMENT_STANDARDS.md)

### services\features\docs\issues\README.md

修正 1 個連結:

-  [開發規範](../DEVELOPMENT_STANDARDS.md) → [開發規範](../../../../docs/development/services_DEVELOPMENT_STANDARDS.md)

### services\core\aiva_core\README.md

修正 1 個連結:

-  [🚀 使用指南](USAGE_GUIDE.md) → [🚀 使用指南](../../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### reports\analysis\EXPLORATION_SYSTEM_MISUNDERSTANDING_ANALYSIS.md

修正 1 個連結:

-  `
c:\D\fold7\AIVA-git\現有腳本快速使用指南.md:
- real_web_reconnaissance('http://example.com')
- 'reconnaissance': recon_result

c:\D\fold7\AIVA-git\services\core\aiva_core\USAGE_GUIDE.md:
- "id": "reconnaissance"
- phases = ["reconnaissance", "vulnerability_discovery", ...]
` → `docs/guides/services/aiva_core_USAGE_GUIDE.md`

### reports\architecture\COORDINATOR_USAGE_GUIDE.md

修正 3 個連結:

-  [Go Engine 指南](../engines/go_engine/USAGE_GUIDE.md) → [Go Engine 指南](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  [Rust Engine 指南](../engines/rust_engine/USAGE_GUIDE.md) → [Rust Engine 指南](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)
-  [Python Engine 指南](../engines/python_engine/USAGE_GUIDE.md) → [Python Engine 指南](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### reports\architecture\ENGINES_DOCUMENTATION_INDEX.md

修正 3 個連結:

-  `rust_engine/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `go_engine/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `rust_engine/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### reports\architecture\GO_ENGINE_STATUS.md

修正 1 個連結:

-  [USAGE_GUIDE.md](./USAGE_GUIDE.md) → [USAGE_GUIDE.md](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### reports\architecture\GUIDES_CONSOLIDATION_REPORT.md

修正 5 個連結:

-  `services/scan/engines/rust_engine/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `services/scan/engines/go_engine/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `services/scan/coordinators/PYTHON_ENGINE_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `services/scan/coordinators/COORDINATOR_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `services/core/aiva_core/USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### reports\architecture\GUIDES_DIRECTORY_UPDATE_REPORT.md

修正 1 個連結:

-  `guides/development/METRICS_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### reports\architecture\GUIDES_DIRECTORY_UPDATE_SUMMARY.md

修正 1 個連結:

-  `METRICS_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### reports\architecture\PYTHON_ENGINE_USAGE_GUIDE.md

修正 1 個連結:

-  [協調器使用指南](./COORDINATOR_USAGE_GUIDE.md) → [協調器使用指南](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### reports\implementation\PYTHON_ENGINE_REWRITE_COMPLETION_REPORT_2025-11-23.md

修正 2 個連結:

-  `PYTHON_ENGINE_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`
-  `PYTHON_ENGINE_USAGE_GUIDE.md` → `docs/guides/services/rust_engine_USAGE_GUIDE.md`

### reports\project_status\IMPORT_FIX_PROGRESS.md

修正 1 個連結:

-  `DEVELOPMENT_STANDARDS.md` → `docs/development/services_DEVELOPMENT_STANDARDS.md`

### reports\testing\INSTALLATION_GUIDE.md

修正 2 個連結:

-  [DEVELOPMENT_STANDARDS.md](./docs/DEVELOPMENT_STANDARDS.md) → [DEVELOPMENT_STANDARDS.md](../../docs/development/services_DEVELOPMENT_STANDARDS.md)
-  [USAGE_GUIDE.md](./services/core/aiva_core/USAGE_GUIDE.md) → [USAGE_GUIDE.md](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

### reports\architecture\core_analysis\AIVA_ANALYSIS_EXPLORATION_ARCHITECTURE_REPAIR_PLAN.md

修正 1 個連結:

-  [Core 模組開發規範](./DEVELOPMENT_STANDARDS.md) → [Core 模組開發規範](../../../docs/development/services_DEVELOPMENT_STANDARDS.md)

### reports\analysis\dependencies\DEPENDENCY_ANALYSIS.md

修正 1 個連結:

-  `DEVELOPMENT_STANDARDS.md` → `docs/development/services_DEVELOPMENT_STANDARDS.md`

### guides\development\README.md

修正 1 個連結:

-  [Metrics Usage Guide](./METRICS_USAGE_GUIDE.md) → [Metrics Usage Guide](../../docs/guides/services/rust_engine_USAGE_GUIDE.md)

---

*報告生成時間: 2025年11月27日*
