# 📚 AIVA 架構設計文檔中心

**最後更新**: 2025-12-31  
**文檔總數**: 28 份  
**總計大小**: ~530 KB

---

## 📑 目錄總覽

本目錄集中收錄了所有 AIVA 專案的架構設計、模組能力與發展計劃相關文檔。

| 目錄 | 說明 | 文件數 |
|------|------|--------|
| [01_core_architecture/](01_core_architecture/) | 核心架構設計 | 5 |
| [02_dual_loop/](02_dual_loop/) | 雙閉環架構 | 5 |
| [03_modules_capabilities/](03_modules_capabilities/) | 模組與能力 | 7 |
| [04_cli_system/](04_cli_system/) | CLI 系統設計 | 4 |
| [05_implementation_plans/](05_implementation_plans/) | 實施計劃 | 2 |
| [06_design_philosophy/](06_design_philosophy/) | 設計哲學 | 4 |

---

## 🏗️ 01_core_architecture/ - 核心架構設計

定義 AIVA 系統的整體架構與目錄結構。

| 文件 | 大小 | 說明 |
|------|------|------|
| [SIMPLE_ARCHITECTURE.md](01_core_architecture/SIMPLE_ARCHITECTURE.md) | 20.2 KB | 功能模組簡化架構設計 v5.0 |
| [AI_MODULE_INTEGRATION_ARCHITECTURE.md](01_core_architecture/AI_MODULE_INTEGRATION_ARCHITECTURE.md) | 37.5 KB | AI 模組整合架構設計 |
| [_PROJECT_ROOT_STRUCTURE_GUIDE.md](01_core_architecture/_PROJECT_ROOT_STRUCTURE_GUIDE.md) | 20.8 KB | 專案根目錄結構指南 |
| [_CORE_DIRECTORIES_FUNCTIONALITY_GUIDE.md](01_core_architecture/_CORE_DIRECTORIES_FUNCTIONALITY_GUIDE.md) | 29.3 KB | 核心目錄功能說明 |
| [ARCHITECTURE_EVOLUTION_HISTORY.md](01_core_architecture/ARCHITECTURE_EVOLUTION_HISTORY.md) | 2.5 KB | 架構演進歷史 |

---

## 🔄 02_dual_loop/ - 雙閉環架構

AIVA 的核心運作機制：內閉環（自我認知）+ 外閉環（執行學習）。

| 文件 | 大小 | 說明 |
|------|------|------|
| [13_STEPS_WORKFLOW_VERIFICATION.md](02_dual_loop/13_STEPS_WORKFLOW_VERIFICATION.md) | 21.3 KB | 完整 13 步驟工作流程（Phase 0-4） |
| [INTERNAL_LOOP_EXECUTION_GUIDE.md](02_dual_loop/INTERNAL_LOOP_EXECUTION_GUIDE.md) | 29.1 KB | 內閉環執行操作手冊 |
| [EXTERNAL_LOOP_ACTIVATION_PLAN.md](02_dual_loop/EXTERNAL_LOOP_ACTIVATION_PLAN.md) | 20.7 KB | 外閉環啟動實施方案 |
| [EXPERIENCE_LEARNING_DESIGN.md](02_dual_loop/EXPERIENCE_LEARNING_DESIGN.md) | 19.6 KB | 經驗學習規劃設計 |
| [INTEGRATION_DUAL_LOOP_DESIGN.md](02_dual_loop/INTEGRATION_DUAL_LOOP_DESIGN.md) | 8.7 KB | Integration 模組雙閉環設計 |

**核心概念**:
```
內閉環 (Internal Loop)     外閉環 (External Loop)
------------------------   ------------------------
探索自身代碼 → RAG        用戶請求 → AI決策
分析能力 → 註冊          任務編排 → 執行
自我認知 → 查詢          結果反饋 → 學習優化
```

---

## 📦 03_modules_capabilities/ - 模組與能力

六大核心模組的能力分析與 CLI 整合。

| 文件 | 大小 | 說明 |
|------|------|------|
| [SIX_MODULES_CAPABILITIES_AND_CLI_GUIDE.md](03_modules_capabilities/SIX_MODULES_CAPABILITIES_AND_CLI_GUIDE.md) | 30.4 KB | 六大模組能力與 CLI 指南（核心） |
| [00_FIVE_MODULES_SUMMARY.md](03_modules_capabilities/00_FIVE_MODULES_SUMMARY.md) | 17.3 KB | 五大模組總覽 |
| [01_CORE_MODULE_ANALYSIS.md](03_modules_capabilities/01_CORE_MODULE_ANALYSIS.md) | 19.8 KB | Core 模組分析 |
| [02_AIVA_COMMON_MODULE_ANALYSIS.md](03_modules_capabilities/02_AIVA_COMMON_MODULE_ANALYSIS.md) | 19.2 KB | AIVA Common 模組分析 |
| [03_FEATURES_MODULE_ANALYSIS.md](03_modules_capabilities/03_FEATURES_MODULE_ANALYSIS.md) | 8.8 KB | Features 模組分析 |
| [04_SCAN_MODULE_ANALYSIS.md](03_modules_capabilities/04_SCAN_MODULE_ANALYSIS.md) | 10.1 KB | Scan 模組分析 |
| [05_INTEGRATION_MODULE_ANALYSIS.md](03_modules_capabilities/05_INTEGRATION_MODULE_ANALYSIS.md) | 9.7 KB | Integration 模組分析 |

**六大模組**:
1. **Core** - AI 決策與調度
2. **AIVA Common** - 共享組件與 Schema
3. **Features** - 安全測試功能
4. **Scan** - 掃描引擎
5. **Integration** - 數據整合
6. **External Tools** - 外部工具整合

---

## 💻 04_cli_system/ - CLI 系統設計

AIVA 的命令行接口設計與使用指南。

| 文件 | 大小 | 說明 |
|------|------|------|
| [CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md](04_cli_system/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md) | 57.2 KB | CLI 命令架構完整分析（最詳細） |
| [CLI_COMPLETE_GUIDE.md](04_cli_system/CLI_COMPLETE_GUIDE.md) | 13.9 KB | CLI 完整使用指南 |
| [CLI_GUIDE.md](04_cli_system/CLI_GUIDE.md) | 13.2 KB | CLI 基礎指南 |
| [AIVA_CLI_USAGE_GUIDE.md](04_cli_system/AIVA_CLI_USAGE_GUIDE.md) | 5.9 KB | CLI 使用指南 |

---

## 📋 05_implementation_plans/ - 實施計劃

從設計到部署的完整實施路線圖。

| 文件 | 大小 | 說明 |
|------|------|------|
| [AI_MODULE_INTEGRATION_IMPLEMENTATION_PLAN.md](05_implementation_plans/AI_MODULE_INTEGRATION_IMPLEMENTATION_PLAN.md) | 38.6 KB | AI 模組整合實施計劃 |
| [AI_MODULE_INTEGRATION_QUICKSTART.md](05_implementation_plans/AI_MODULE_INTEGRATION_QUICKSTART.md) | 18.9 KB | 快速開始指南 |

---

## 🧠 06_design_philosophy/ - 設計哲學

AIVA 的設計理念與模組化思想。

| 文件 | 大小 | 說明 |
|------|------|------|
| [MODULE_DESIGN_PHILOSOPHY.md](06_design_philosophy/MODULE_DESIGN_PHILOSOPHY.md) | 17.2 KB | 模組設計理念（python_tools vs self_healing） |
| [CORRECT_DESIGN_UNDERSTANDING.md](06_design_philosophy/CORRECT_DESIGN_UNDERSTANDING.md) | 9.4 KB | 設計理解釐清 |
| [DESIGN_EVALUATION_REPORT.md](06_design_philosophy/DESIGN_EVALUATION_REPORT.md) | 12.6 KB | 設計評估報告 |
| [COMPLEXITY_REDUCTION_GUIDE.md](06_design_philosophy/COMPLEXITY_REDUCTION_GUIDE.md) | 21.9 KB | 複雜度降低指南 |

---

## 📖 建議閱讀順序

### 新手入門
1. [SIMPLE_ARCHITECTURE.md](01_core_architecture/SIMPLE_ARCHITECTURE.md) - 了解整體架構
2. [00_FIVE_MODULES_SUMMARY.md](03_modules_capabilities/00_FIVE_MODULES_SUMMARY.md) - 了解模組分布
3. [CLI_GUIDE.md](04_cli_system/CLI_GUIDE.md) - 學習 CLI 使用

### 理解雙閉環
1. [13_STEPS_WORKFLOW_VERIFICATION.md](02_dual_loop/13_STEPS_WORKFLOW_VERIFICATION.md) - 完整流程
2. [INTERNAL_LOOP_EXECUTION_GUIDE.md](02_dual_loop/INTERNAL_LOOP_EXECUTION_GUIDE.md) - 內閉環操作
3. [EXTERNAL_LOOP_ACTIVATION_PLAN.md](02_dual_loop/EXTERNAL_LOOP_ACTIVATION_PLAN.md) - 外閉環啟動

### 深入開發
1. [SIX_MODULES_CAPABILITIES_AND_CLI_GUIDE.md](03_modules_capabilities/SIX_MODULES_CAPABILITIES_AND_CLI_GUIDE.md) - 模組能力
2. [CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md](04_cli_system/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md) - CLI 架構
3. [MODULE_DESIGN_PHILOSOPHY.md](06_design_philosophy/MODULE_DESIGN_PHILOSOPHY.md) - 設計哲學

---

## 📊 文檔來源

| 來源 | 說明 |
|------|------|
| `docs/technical/` | 原始技術文檔目錄 |
| `services/.../` | 模組內嵌文檔 |
| `guides/` | 操作指南目錄 |
| `docs/reports/` | 分析報告目錄 |
| `docs/analysis/` | 模組分析報告 |

**注意**: 本目錄為文檔集中副本，原始文件仍保留在各自位置。
