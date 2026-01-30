# AIVA 根目錄文檔整理計劃（完整版）

**日期**: 2026-01-28  
**依據**: CLI 架構討論、Services 架構分析、系統規劃  
**分析範圍**: 根目錄 31 個 .md 文件 + _archive 目錄結構

---

## 📁 _archive 目錄現狀

```
_archive/
├── ARCHIVE_INDEX.md              # 索引文件
├── README.md
├── 03_historical_reports/        # 歷史報告
│   ├── 2026-01/                  # 月份子目錄（已創建）
│   └── cli-changelog-2026-01-10.md
├── 06_documentation_archive/     # 文檔歸檔
│   ├── 2026-01/                  # 月份子目錄（已創建）
│   └── cli-data-2026-01-10/
├── 07_configuration_archive/     # 配置歸檔
├── 07_documentation_archive/     # 舊文檔歸檔（有內容）
│   ├── CLI架構重構指南.md
│   ├── 跨語言CLI設計指南.md
│   ├── 雙閉環可行性分析指南_v1.0_archived.md
│   └── ...
├── 08_tool_archive/              # 工具歸檔
├── 09_integration_archive/       # 整合歸檔
└── validation/                   # 驗證腳本
```

---

## 📊 根目錄文檔分類結果（31個文件）

### ✅ 保留在根目錄（3個）

| 檔名 | 理由 |
|------|------|
| `README.md` | 專案入口文檔，必須保留 |
| `CHANGELOG.md` | 版本歷史，標準專案文檔 |
| `DOCUMENTATION_REORGANIZATION_PLAN.md` | 當前整理計劃，暫保留待整理完成後歸檔 |

---

### 📥 移動到 guides/（4個指南文檔）

| 檔名 | 目標路徑 | 說明 |
|------|----------|------|
| `EXTERNAL_CAPABILITIES_USAGE_GUIDE.md` | `guides/general/` | 外部能力使用指南 |
| `FUNCTION_CALLABLE_JUDGMENT_GUIDE.md` | `guides/development/` | 功能可調用判斷指南 |
| `RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md` | `guides/technical/` | RAG觸發與通知指南 |
| `QUICK_REFERENCE.md` | `guides/general/QUICK_REFERENCE_GUIDE.md` | 快速參考（改名） |

---

### 📁 移動到 docs/（10個架構與系統文檔）

| 檔名 | 目標路徑 | 說明 |
|------|----------|------|
| `AI_CAPABILITY_SELECTION_MECHANISM_REPORT.md` | `docs/01_architecture/` | ✨ 今日生成，AI能力選擇機制（重要，保留） |
| `SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md` | `docs/01_architecture/` | ✨ 今日生成，Services架構分析（重要，保留） |
| `COMMANDER_CLI_ARCHITECTURE_UPDATE.md` | `docs/01_architecture/` | ✨ 今日生成，CLI架構更新（重要，保留） |
| `UNIFIED_NAMING_CONVENTION.md` | `docs/01_architecture/` | 命名規範（重要，保留） |
| `LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md` | `docs/learning_system/` | 學習系統架構 |
| `AI_LEARNING_DATA_FLOW.md` | `docs/learning_system/` | AI學習數據流 |
| `RAG_CLI_COMMAND_DECISION_SYSTEM.md` | `docs/rag_system/` | RAG CLI決策系統 |
| `RAG_INTERNAL_EXPLORATION_INTEGRATION.md` | `docs/rag_system/` | RAG內部探索整合 |
| `VECTOR_STORE_AND_RAG_ARCHITECTURE.md` | `docs/rag_system/` | 向量庫與RAG架構 |
| `DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md` | `docs/learning_system/` | 雙循環實現計劃 |

---

### 🗄️ 直接歸檔到 _archive/（14個過時/已完成文檔）

#### 歸檔到 `_archive/03_historical_reports/2026-01/`

| 檔名 | 內容摘要 | 歸檔理由 |
|------|----------|----------|
| `22_flows_test_report.md` | 22個XSS入口點測試報告(2026-01-21) | ✅ 測試已完成，歷史記錄 |
| `CLI問題診斷報告.md` | CLI導入問題診斷 | ✅ 問題已解決 |
| `CLI導入路徑錯誤分析.md` | 導入路徑錯誤根因分析 | ✅ 錯誤已修復 |
| `BACKUP_ANALYSIS_REPORT.md` | 備份文件分析報告 | ✅ 分析已完成 |
| `BACKUP_CLEANUP_EXECUTION_REPORT.md` | 備份清理執行報告 | ✅ 清理已執行 |
| `DATA_DIFFERENCE_ANALYSIS.md` | 274 graphs vs 107 flows差異分析 | ✅ 已理解並整合 |
| `USAGE_GUIDE_VALIDATION_REPORT.md` | 使用指南驗證報告 | ✅ 驗證已完成 |
| `GUIDES_REORGANIZATION_ANALYSIS.md` | 指南重組分析(今日) | ✅ 已轉化為執行計劃 |

#### 歸檔到 `_archive/06_documentation_archive/2026-01/`

| 檔名 | 內容摘要 | 歸檔理由 |
|------|----------|----------|
| `CLI架構實現總結.md` | CLI三層架構實現總結 | 🔄 被 COMMANDER_CLI_ARCHITECTURE_UPDATE.md 取代 |
| `services_directory_structure_analysis.md` | Services目錄結構分析 | 🔄 被 SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md 取代 |
| `XSS_174_FLOWS_ANALYSIS_REPORT.md` | XSS 174流程分析 | 🔄 已整合到架構報告 |
| `WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE.md` | 171個內部函數不可調用原因 | 🔄 已在架構報告中說明 |
| `CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md` | Classifier vs Executor階段對比 | 🔄 架構概念已整合 |
| `AI_EXECUTOR_INTEGRATION_COMPLETE.md` | AI執行器整合完成報告 | ✅ MVP完成報告，歷史記錄 |

---

### ⚠️ 需確認後處理（4個）

| 檔名 | 內容摘要 | 待確認事項 |
|------|----------|------------|
| `CLI_AI_INTEGRATION_IMPLEMENTATION.md` | CLI AI整合實現 | 是否已被新報告完全覆蓋？ |
| `AI_MODULES_CAPABILITY_CHECK.md` | AI模組能力檢查 | 是否有未整合的內容？ |
| `CLI_vs_DirectImport_對比.md` | CLI vs 直接導入對比 | 設計決策文檔，建議重寫為ADR格式 |
| `function_xss_operable_classification.md` | XSS可操作功能分類(101個) | 移至 `features_classification/` 或歸檔？ |

---

## 📋 執行摘要

| 操作 | 數量 | 說明 |
|------|------|------|
| **保留根目錄** | 3 | README, CHANGELOG, 當前計劃 |
| **移動到 guides/** | 4 | 指南類文檔 |
| **移動到 docs/** | 10 | 架構與系統文檔 |
| **歸檔到 _archive/** | 14 | 過時/已完成文檔 |
| **需確認** | 4 | 待您確認後處理 |
| **總計** | 35 | |

---

## 🎯 整理後根目錄預覽

```
C:\D\fold7\AIVA-git\
├── README.md                              # 專案說明
├── CHANGELOG.md                           # 版本歷史
├── DOCUMENTATION_REORGANIZATION_PLAN.md   # 整理計劃（暫留）
├── Cargo.toml                             # Rust配置
├── pyproject.toml                         # Python配置
├── requirements.txt                       # Python依賴
└── ... (其他非 .md 文件)
```

**整理後：根目錄只保留 3 個 .md 文件（從 31 個減少到 3 個）**

---

## ❓ 需要您確認的 4 個文件

### 1. `CLI_AI_INTEGRATION_IMPLEMENTATION.md` (8.11 KB)
**內容**: CLI AI整合實現細節  
**問題**: 內容是否已被 `COMMANDER_CLI_ARCHITECTURE_UPDATE.md` 完全覆蓋？  
**建議**: 如果是 → 歸檔；如果有獨特內容 → 移至 docs/

### 2. `AI_MODULES_CAPABILITY_CHECK.md` (11.24 KB)
**內容**: AI模組能力檢查報告  
**問題**: 是否有未整合到新架構報告的重要內容？  
**建議**: 如果是一次性檢查 → 歸檔；如果持續更新 → 移至 docs/

### 3. `CLI_vs_DirectImport_對比.md` (9.45 KB)
**內容**: CLI vs 直接導入的設計決策對比  
**問題**: 這是重要的設計決策文檔  
**建議**: 重寫為 ADR 格式放到 `docs/02_design_decisions/ADR_001_CLI_vs_DirectImport.md`，原文歸檔

### 4. `function_xss_operable_classification.md` (25.77 KB)
**內容**: XSS 模組 101 個可操作功能的詳細分類  
**問題**: 這是功能分類參考文檔  
**建議**: 移至 `features_classification/` 目錄（保持可查閱）

---

**請確認上述 4 個文件的處理方式，我再繼續執行整理。**

