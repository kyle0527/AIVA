# AIVA 架構文檔整理與歸檔計劃

> **整理日期**: 2026-01-28  
> **目的**: 統一管理架構相關文檔，避免重複內容，建立清晰的文檔結構  
> **使用現有**: 利用 `_archive/` 已有的目錄結構

---

## 📊 一、文檔分類統計

### 1.1 根目錄文檔統計

| 類別 | 數量 | 說明 |
|------|------|------|
| **架構設計** | 8 | CLI架構、Services架構、Core架構 |
| **功能分析** | 7 | XSS分析、SQLi分析、能力檢查 |
| **系統整合** | 6 | RAG整合、學習系統、執行器整合 |
| **問題診斷** | 5 | CLI問題、導入錯誤、備份清理 |
| **使用指南** | 6 | 快速參考、使用驗證、能力指南 |
| **總計** | **32** | 不包括 README |

---

## 🗂️ 二、文檔分類詳細列表

### 2.1 ✅ 最新架構文檔（保留）

這些是最新生成的核心架構文檔，應該保留在根目錄：

| 檔案 | 大小 | 日期 | 狀態 | 說明 |
|------|------|------|------|------|
| `AI_CAPABILITY_SELECTION_MECHANISM_REPORT.md` | 20.61 KB | 2026-01-28 | ✅ 最新 | AI如何選擇能力的完整機制 |
| `SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md` | 18.13 KB | 2026-01-28 | ✅ 最新 | Services目錄完整架構分析 |
| `COMMANDER_CLI_ARCHITECTURE_UPDATE.md` | 46.05 KB | 2026-01-28 | ✅ 最新 | Commander CLI架構更新報告 |

---

### 2.2 🔄 需要更新的文檔（部分內容過時）

這些文檔包含有用信息，但部分內容已被新文檔覆蓋：

| 檔案 | 大小 | 日期 | 問題 | 建議處理 |
|------|------|------|------|----------|
| `CLI架構實現總結.md` | 18.84 KB | 2026-01-19 | 部分內容被 COMMANDER_CLI_ARCHITECTURE_UPDATE.md 覆蓋 | ⚠️ 合併到新報告或更新 |
| `CLI_vs_DirectImport_對比.md` | 9.45 KB | 2026-01-19 | 設計決策仍有價值，但執行細節已過時 | ⚠️ 保留設計理念部分 |
| `services_directory_structure_analysis.md` | 8.00 KB | 2026-01-21 | 被 SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md 完全覆蓋 | 🗄️ 歸檔 |

---

### 2.3 🗄️ 應該歸檔的文檔（已過時或重複）

這些文檔的內容已經被新報告完全覆蓋，建議移至 `_archive/` 目錄：

#### 2.3.1 CLI 問題相關（已解決）

| 檔案 | 大小 | 日期 | 原因 |
|------|------|------|------|
| `CLI問題診斷報告.md` | 6.56 KB | 2026-01-19 | 問題已在 COMMANDER_CLI_ARCHITECTURE_UPDATE.md 解決 |
| `CLI導入路徑錯誤分析.md` | 7.73 KB | 2026-01-19 | 錯誤已修復，歷史記錄 |

#### 2.3.2 分析報告（內容重複）

| 檔案 | 大小 | 日期 | 原因 |
|------|------|------|------|
| `services_directory_structure_analysis.md` | 8.00 KB | 2026-01-21 | 完全被 SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md 覆蓋 |
| `BACKUP_ANALYSIS_REPORT.md` | 11.82 KB | 2026-01-21 | 備份分析已完成 |
| `BACKUP_CLEANUP_EXECUTION_REPORT.md` | 10.44 KB | 2026-01-21 | 清理執行已完成 |
| `DATA_DIFFERENCE_ANALYSIS.md` | 1.60 KB | 2026-01-20 | 數據差異分析已完成 |

#### 2.3.3 特定功能分析（已整合）

| 檔案 | 大小 | 日期 | 原因 |
|------|------|------|------|
| `22_flows_test_report.md` | 5.61 KB | 2026-01-21 | 特定測試報告，歷史記錄 |
| `XSS_174_FLOWS_ANALYSIS_REPORT.md` | 10.47 KB | 2026-01-21 | XSS流程分析，已整合到新報告 |
| `WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE.md` | 13.06 KB | 2026-01-21 | 特定問題分析，已解決 |
| `function_xss_operable_classification.md` | 25.77 KB | 2026-01-21 | 功能分類，應移至 features_classification/ |

---

### 2.4 📚 保留的參考文檔（有長期價值）

這些文檔包含重要的設計理念或長期參考資料：

| 檔案 | 大小 | 日期 | 保留原因 |
|------|------|------|----------|
| `UNIFIED_NAMING_CONVENTION.md` | 9.26 KB | 2026-01-20 | 命名規範，長期參考 |
| `QUICK_REFERENCE.md` | 5.35 KB | 2026-01-19 | 快速參考，實用 |
| `CHANGELOG.md` | 9.65 KB | 2026-01-19 | 變更日誌，必須保留 |
| `EXTERNAL_CAPABILITIES_USAGE_GUIDE.md` | 12.87 KB | 2026-01-23 | 使用指南，實用 |
| `FUNCTION_CALLABLE_JUDGMENT_GUIDE.md` | 14.15 KB | 2026-01-21 | 判斷指南，參考價值 |
| `USAGE_GUIDE_VALIDATION_REPORT.md` | 10.78 KB | 2026-01-21 | 驗證報告，品質保證 |

---

### 2.5 🔬 RAG/學習系統文檔（專門領域）

這些文檔屬於特定子系統，建議移至 `docs/` 對應子目錄：

| 檔案 | 大小 | 日期 | 建議位置 |
|------|------|------|----------|
| `RAG_CLI_COMMAND_DECISION_SYSTEM.md` | 21.84 KB | 2026-01-20 | `docs/rag_system/` |
| `RAG_INTERNAL_EXPLORATION_INTEGRATION.md` | 8.94 KB | 2026-01-20 | `docs/rag_system/` |
| `RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md` | 15.28 KB | 2026-01-20 | `docs/rag_system/` |
| `VECTOR_STORE_AND_RAG_ARCHITECTURE.md` | 14.43 KB | 2026-01-20 | `docs/rag_system/` |
| `LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md` | 7.60 KB | 2026-01-20 | `docs/learning_system/` |
| `AI_LEARNING_DATA_FLOW.md` | 17.86 KB | 2026-01-20 | `docs/learning_system/` |
| `DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md` | 18.96 KB | 2026-01-20 | `docs/learning_system/` |

---

### 2.6 🔗 整合相關文檔（整合層）

| 檔案 | 大小 | 日期 | 建議位置 |
|------|------|------|----------|
| `AI_EXECUTOR_INTEGRATION_COMPLETE.md` | 10.66 KB | 2026-01-23 | `docs/integration/` |
| `CLI_AI_INTEGRATION_IMPLEMENTATION.md` | 8.11 KB | 2026-01-20 | `docs/integration/` |
| `CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md` | 10.70 KB | 2026-01-20 | `docs/integration/` |
| `AI_MODULES_CAPABILITY_CHECK.md` | 11.24 KB | 2026-01-20 | `docs/integration/` |

---

## 📋 三、建議的目錄結構重組（使用現有 _archive）

```
C:\D\fold7\AIVA-git\
│
├── README.md                                          # 專案主文檔
├── CHANGELOG.md                                       # 變更日誌  
├── QUICK_REFERENCE.md                                 # 快速參考
│
├── docs/                                              # 文檔目錄
│   ├── 01_architecture/                              # 🏗️ 架構文檔 (新建)
│   ├── 02_design_decisions/                          # 🎯 設計決策 (新建)
│   ├── 03_analysis_reports/                          # 📊 分析報告 (已存在)
│   ├── 04_user_guides/                               # 📖 使用指南 (新建)
│   ├── 05_api_reference/                             # 🔌 API參考 (新建)
│   ├── 06_deployment/                                # 🚀 部署文檔 (新建)
│   ├── rag_system/                                   # 🔍 RAG系統 (移動)
│   └── learning_system/                              # 🧠 學習系統 (移動)
│
└── _archive/                                         # 🗄️ 歷史歸檔 (使用現有)
    ├── 03_historical_reports/                        # 歷史報告
    │   └── 2026-01/                                  # 按月份子目錄
    │       ├── CLI問題診斷報告_20260119.md
    │       ├── CLI導入路徑錯誤分析_20260119.md
    │       ├── services_directory_structure_analysis_20260121.md
    │       └── ...
    │
    ├── 06_documentation_archive/                     # 文檔歸檔
    │   └── 2026-01/
    │       ├── CLI架構實現總結_20260119.md
    │       └── XSS_174_FLOWS_ANALYSIS_REPORT_20260121.md
    │
    └── ARCHIVE_INDEX.md                              # 歸檔索引 (更新)
```

---

## � 四、應該建立的文件清單（後續經營依據）

### 4.1 核心架構文檔（必須建立）

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `SYSTEM_OVERVIEW.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 系統整體概覽、核心概念說明 |
| `ARCHITECTURE_PRINCIPLES.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 架構設計原則與理念 |
| `MODULE_DEPENDENCY_MAP.md` | `docs/01_architecture/` | ❌ 缺少 | 🟡 P1 | 模組依賴關係圖 |
| `DATA_FLOW_DIAGRAM.md` | `docs/01_architecture/` | ❌ 缺少 | 🟡 P1 | 完整數據流向圖 |
| `AI_CAPABILITY_SELECTION_MECHANISM.md` | `docs/01_architecture/` | ✅ 已有 | - | AI能力選擇機制（已生成） |
| `SERVICES_ARCHITECTURE_ANALYSIS.md` | `docs/01_architecture/` | ✅ 已有 | - | Services架構分析（已生成） |
| `COMMANDER_CLI_ARCHITECTURE.md` | `docs/01_architecture/` | ✅ 已有 | - | Commander CLI架構（已生成） |

### 4.2 設計決策文檔（記錄關鍵決策）

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `ADR_001_CLI_vs_DirectImport.md` | `docs/02_design_decisions/` | ⚠️ 部分 | 🟡 P1 | 為何選擇CLI架構（需重寫為ADR格式） |
| `ADR_002_RAG_Integration_Strategy.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟡 P1 | RAG整合策略決策 |
| `ADR_003_Learning_System_Design.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟡 P1 | 學習系統設計決策 |
| `ADR_004_Multi_Language_Engine.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟢 P2 | 多語言引擎選擇決策 |
| `ADR_005_Security_Model.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🔴 P0 | 安全模型設計決策 |
| `DESIGN_DECISIONS_INDEX.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟡 P1 | 設計決策索引 |

### 4.3 API 參考文檔（開發必備）

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `API_OVERVIEW.md` | `docs/05_api_reference/` | ❌ 缺少 | 🔴 P0 | API總覽與版本說明 |
| `CLI_COMMANDS_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🔴 P0 | 所有CLI命令參考 |
| `CORE_API_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟡 P1 | Core模組API文檔 |
| `FEATURES_API_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟡 P1 | Features模組API文檔 |
| `RAG_API_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟢 P2 | RAG系統API文檔 |
| `ERROR_CODES_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟡 P1 | 錯誤碼對照表 |

### 4.4 使用指南（用戶文檔）

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `GETTING_STARTED.md` | `docs/04_user_guides/` | ❌ 缺少 | 🔴 P0 | 快速開始指南 |
| `INSTALLATION_GUIDE.md` | `docs/04_user_guides/` | ❌ 缺少 | 🔴 P0 | 詳細安裝指南 |
| `CONFIGURATION_GUIDE.md` | `docs/04_user_guides/` | ❌ 缺少 | 🟡 P1 | 配置指南 |
| `COMMON_WORKFLOWS.md` | `docs/04_user_guides/` | ❌ 缺少 | 🟡 P1 | 常見工作流程 |
| `TROUBLESHOOTING_GUIDE.md` | `docs/04_user_guides/` | ❌ 缺少 | 🟡 P1 | 故障排除指南 |
| `FAQ.md` | `docs/04_user_guides/` | ❌ 缺少 | 🟢 P2 | 常見問題 |
| `EXTERNAL_CAPABILITIES_USAGE_GUIDE.md` | `docs/04_user_guides/` | ✅ 已有 | - | 外部能力使用指南（已有） |
| `FUNCTION_CALLABLE_JUDGMENT_GUIDE.md` | `docs/04_user_guides/` | ✅ 已有 | - | 功能可調用判斷（已有） |

### 4.5 部署與運維文檔

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `DEPLOYMENT_GUIDE.md` | `docs/06_deployment/` | ❌ 缺少 | 🔴 P0 | 部署指南 |
| `DOCKER_SETUP.md` | `docs/06_deployment/` | ❌ 缺少 | 🟡 P1 | Docker部署說明 |
| `PRODUCTION_CHECKLIST.md` | `docs/06_deployment/` | ❌ 缺少 | 🟡 P1 | 生產環境檢查清單 |
| `MONITORING_AND_LOGGING.md` | `docs/06_deployment/` | ❌ 缺少 | 🟡 P1 | 監控與日誌 |
| `BACKUP_AND_RECOVERY.md` | `docs/06_deployment/` | ❌ 缺少 | 🟢 P2 | 備份與恢復 |
| `PERFORMANCE_TUNING.md` | `docs/06_deployment/` | ❌ 缺少 | 🟢 P2 | 性能優化 |

### 4.6 開發文檔（貢獻者必讀）

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `CONTRIBUTING.md` | 根目錄 | ❌ 缺少 | 🔴 P0 | 貢獻指南 |
| `DEVELOPMENT_SETUP.md` | `docs/` | ❌ 缺少 | 🔴 P0 | 開發環境設置 |
| `CODE_STYLE_GUIDE.md` | `docs/` | ❌ 缺少 | 🟡 P1 | 代碼風格指南 |
| `TESTING_GUIDE.md` | `docs/` | ❌ 缺少 | 🟡 P1 | 測試指南 |
| `COMMIT_CONVENTIONS.md` | `docs/` | ❌ 缺少 | 🟢 P2 | 提交規範 |
| `UNIFIED_NAMING_CONVENTION.md` | `docs/` | ✅ 已有 | - | 命名規範（已有） |

### 4.7 RAG 與學習系統文檔

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `RAG_SYSTEM_OVERVIEW.md` | `docs/rag_system/` | ❌ 缺少 | 🟡 P1 | RAG系統總覽 |
| `RAG_CLI_COMMAND_DECISION_SYSTEM.md` | `docs/rag_system/` | ✅ 已有 | - | RAG CLI決策（已有） |
| `RAG_INTERNAL_EXPLORATION_INTEGRATION.md` | `docs/rag_system/` | ✅ 已有 | - | RAG內部探索（已有） |
| `VECTOR_STORE_AND_RAG_ARCHITECTURE.md` | `docs/rag_system/` | ✅ 已有 | - | 向量庫架構（已有） |
| `LEARNING_SYSTEM_OVERVIEW.md` | `docs/learning_system/` | ❌ 缺少 | 🟡 P1 | 學習系統總覽 |
| `LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md` | `docs/learning_system/` | ✅ 已有 | - | 學習系統架構（已有） |
| `AI_LEARNING_DATA_FLOW.md` | `docs/learning_system/` | ✅ 已有 | - | 學習數據流（已有） |

### 4.8 安全與合規文檔

| 文件名 | 位置 | 狀態 | 優先級 | 說明 |
|--------|------|------|--------|------|
| `SECURITY.md` | 根目錄 | ❌ 缺少 | 🔴 P0 | 安全政策 |
| `SECURITY_ARCHITECTURE.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 安全架構設計 |
| `VULNERABILITY_HANDLING.md` | `docs/` | ❌ 缺少 | 🟡 P1 | 漏洞處理流程 |
| `COMPLIANCE_CHECKLIST.md` | `docs/` | ❌ 缺少 | 🟢 P2 | 合規檢查清單 |
| `LICENSE.md` | 根目錄 | ❌ 缺少 | 🔴 P0 | 授權條款 |

---

### 📊 4.9 文檔缺口統計

| 優先級 | 說明 | 數量 | 佔比 |
|--------|------|------|------|
| 🔴 **P0 - 必須** | 核心文檔，影響專案使用 | **18** | 32% |
| 🟡 **P1 - 重要** | 重要文檔，影響開發效率 | **24** | 43% |
| 🟢 **P2 - 建議** | 補充文檔，提升完整性 | **8** | 14% |
| ✅ **已完成** | 現有文檔 | **14** | 25% |
| **總計** | | **56** | 100% |

---

## 🔧 五、執行計劃

### Phase 1: 創建目錄結構

```powershell
# 創建新的文檔目錄結構（docs/ 下）
cd "C:\D\fold7\AIVA-git"

# 架構文檔
New-Item -ItemType Directory -Path "docs\01_architecture" -Force

# 設計決策
New-Item -ItemType Directory -Path "docs\02_design_decisions" -Force

# docs\03_analysis_reports 已存在

# 使用指南
New-Item -ItemType Directory -Path "docs\04_user_guides" -Force

# API 參考
New-Item -ItemType Directory -Path "docs\05_api_reference" -Force

# 部署文檔
New-Item -ItemType Directory -Path "docs\06_deployment" -Force

# RAG 系統（保持現有）
New-Item -ItemType Directory -Path "docs\rag_system" -Force

# 學習系統（保持現有）
New-Item -ItemType Directory -Path "docs\learning_system" -Force

# 在 _archive 下創建月份子目錄
New-Item -ItemType Directory -Path "_archive\03_historical_reports\2026-01" -Force
New-Item -ItemType Directory -Path "_archive\06_documentation_archive\2026-01" -Force
```

### Phase 2: 移動現有文檔到正確位置

#### 2.1 最新架構文檔 → docs/01_architecture/

```powershell
# 移動最新架構文檔
Move-Item "AI_CAPABILITY_SELECTION_MECHANISM_REPORT.md" "docs\01_architecture\AI_CAPABILITY_SELECTION_MECHANISM.md"
Move-Item "SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md" "docs\01_architecture\SERVICES_ARCHITECTURE_ANALYSIS.md"
Move-Item "COMMANDER_CLI_ARCHITECTURE_UPDATE.md" "docs\01_architecture\COMMANDER_CLI_ARCHITECTURE.md"
```

#### 2.2 設計決策 → docs/02_design_decisions/

```powershell
# CLI設計決策（需要重寫為ADR格式）
Copy-Item "CLI_vs_DirectImport_對比.md" "docs\02_design_decisions\ADR_001_CLI_vs_DirectImport.md"
# 保留原文件暫時不刪除，等重寫完成後再歸檔
```

#### 2.3 使用指南 → docs/04_user_guides/

```powershell
Move-Item "EXTERNAL_CAPABILITIES_USAGE_GUIDE.md" "docs\04_user_guides\"
Move-Item "FUNCTION_CALLABLE_JUDGMENT_GUIDE.md" "docs\04_user_guides\"
Move-Item "USAGE_GUIDE_VALIDATION_REPORT.md" "docs\04_user_guides\"
Move-Item "QUICK_REFERENCE.md" "docs\04_user_guides\"
```

#### 2.4 RAG 系統文檔 → docs/rag_system/

```powershell
Move-Item "RAG_CLI_COMMAND_DECISION_SYSTEM.md" "docs\rag_system\"
Move-Item "RAG_INTERNAL_EXPLORATION_INTEGRATION.md" "docs\rag_system\"
Move-Item "RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md" "docs\rag_system\"
Move-Item "VECTOR_STORE_AND_RAG_ARCHITECTURE.md" "docs\rag_system\"
```

#### 2.5 學習系統文檔 → docs/learning_system/

```powershell
Move-Item "LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md" "docs\learning_system\"
Move-Item "AI_LEARNING_DATA_FLOW.md" "docs\learning_system\"
Move-Item "DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md" "docs\learning_system\"
```

#### 2.6 整合層文檔 → docs/03_analysis_reports/

```powershell
# 這些屬於分析報告，放在現有的 03_analysis_reports
Move-Item "AI_EXECUTOR_INTEGRATION_COMPLETE.md" "docs\03_analysis_reports\"
Move-Item "CLI_AI_INTEGRATION_IMPLEMENTATION.md" "docs\03_analysis_reports\"
Move-Item "CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md" "docs\03_analysis_reports\"
Move-Item "AI_MODULES_CAPABILITY_CHECK.md" "docs\03_analysis_reports\"
```

#### 2.7 命名規範 → docs/01_architecture/

```powershell
Move-Item "UNIFIED_NAMING_CONVENTION.md" "docs\01_architecture\"
```

#### 2.8 歸檔過時文檔 → _archive/

```powershell
# 歸檔已解決的問題報告 → 03_historical_reports
Move-Item "CLI問題診斷報告.md" "_archive\03_historical_reports\2026-01\CLI問題診斷報告_20260119.md"
Move-Item "CLI導入路徑錯誤分析.md" "_archive\03_historical_reports\2026-01\CLI導入路徑錯誤分析_20260119.md"
Move-Item "BACKUP_ANALYSIS_REPORT.md" "_archive\03_historical_reports\2026-01\BACKUP_ANALYSIS_REPORT_20260121.md"
Move-Item "BACKUP_CLEANUP_EXECUTION_REPORT.md" "_archive\03_historical_reports\2026-01\BACKUP_CLEANUP_EXECUTION_REPORT_20260121.md"
Move-Item "DATA_DIFFERENCE_ANALYSIS.md" "_archive\03_historical_reports\2026-01\DATA_DIFFERENCE_ANALYSIS_20260120.md"
Move-Item "22_flows_test_report.md" "_archive\03_historical_reports\2026-01\22_flows_test_report_20260121.md"

# 歸檔已被新報告取代的文檔 → 06_documentation_archive
Move-Item "CLI架構實現總結.md" "_archive\06_documentation_archive\2026-01\CLI架構實現總結_20260119.md"
Move-Item "services_directory_structure_analysis.md" "_archive\06_documentation_archive\2026-01\services_directory_structure_analysis_20260121.md"
Move-Item "XSS_174_FLOWS_ANALYSIS_REPORT.md" "_archive\06_documentation_archive\2026-01\XSS_174_FLOWS_ANALYSIS_REPORT_20260121.md"
Move-Item "WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE.md" "_archive\06_documentation_archive\2026-01\WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE_20260121.md"

# 移動功能分類到正確位置
Move-Item "function_xss_operable_classification.md" "features_classification\"
```

### Phase 3: 更新 _archive/ARCHIVE_INDEX.md

在 `_archive/ARCHIVE_INDEX.md` 中添加新的歸檔記錄：

```markdown
## 2026-01 文檔整理歸檔

### 03_historical_reports/2026-01/
- `CLI問題診斷報告_20260119.md` - Commander初始化問題診斷 → 已在新架構中解決
- `CLI導入路徑錯誤分析_20260119.md` - 導入路徑錯誤分析 → 已修復
- `BACKUP_ANALYSIS_REPORT_20260121.md` - 備份分析報告 → 已完成
- `BACKUP_CLEANUP_EXECUTION_REPORT_20260121.md` - 清理執行報告 → 已完成
- `DATA_DIFFERENCE_ANALYSIS_20260120.md` - 數據差異分析 → 已完成
- `22_flows_test_report_20260121.md` - 22個flows測試報告 → 歷史記錄

### 06_documentation_archive/2026-01/
- `CLI架構實現總結_20260119.md` → 被 `COMMANDER_CLI_ARCHITECTURE.md` 取代
- `services_directory_structure_analysis_20260121.md` → 被 `SERVICES_ARCHITECTURE_ANALYSIS.md` 取代
- `XSS_174_FLOWS_ANALYSIS_REPORT_20260121.md` → XSS流程分析，已整合到新報告
- `WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE_20260121.md` → 特定問題分析，已解決
```

---

## 📊 六、重組後的效益

| 項目 | 重組前 | 重組後 | 改善 |
|------|--------|--------|------|
| **根目錄文檔數** | 32 個 | 3 個 | ⬇️ 90% |
| **架構文檔集中度** | 分散 | 集中在 docs/architecture/ | ⬆️ 清晰 |
| **過時文檔** | 混雜 | 歸檔至 _archive/ | ⬆️ 整潔 |
| **查找效率** | 需要搜索 | 按主題分類 | ⬆️ 快速 |

---

## ✅ 七、驗證清單

完成重組後，確認：

- [ ] 目錄結構已創建 (docs/01-06 + rag_system + learning_system)
- [ ] 最新架構文檔在 `docs/01_architecture/`
- [ ] 設計決策文檔在 `docs/02_design_decisions/`
- [ ] 分析報告在 `docs/03_analysis_reports/`
- [ ] 使用指南在 `docs/04_user_guides/`
- [ ] RAG 系統文檔在 `docs/rag_system/`
- [ ] 學習系統文檔在 `docs/learning_system/`
- [ ] 過時文檔已歸檔至 `_archive/03_historical_reports/2026-01/`
- [ ] 被取代文檔已歸檔至 `_archive/06_documentation_archive/2026-01/`
- [ ] 根目錄只保留 README、CHANGELOG、CONTRIBUTING、SECURITY、LICENSE
- [ ] `_archive/ARCHIVE_INDEX.md` 已更新
- [ ] README.md 已更新文檔結構鏈接

---

## 🚀 八、後續建立文檔的優先順序

### 第一階段（立即建立）- P0 優先級

```markdown
1. ✅ SECURITY.md - 安全政策（必須）
2. ✅ LICENSE.md - 授權條款（必須）
3. ✅ CONTRIBUTING.md - 貢獻指南（必須）
4. ✅ docs/01_architecture/SYSTEM_OVERVIEW.md - 系統總覽
5. ✅ docs/01_architecture/ARCHITECTURE_PRINCIPLES.md - 架構原則
6. ✅ docs/01_architecture/SECURITY_ARCHITECTURE.md - 安全架構
7. ✅ docs/04_user_guides/GETTING_STARTED.md - 快速開始
8. ✅ docs/04_user_guides/INSTALLATION_GUIDE.md - 安裝指南
9. ✅ docs/05_api_reference/API_OVERVIEW.md - API總覽
10. ✅ docs/05_api_reference/CLI_COMMANDS_REFERENCE.md - CLI命令參考
11. ✅ docs/06_deployment/DEPLOYMENT_GUIDE.md - 部署指南
12. ✅ docs/DEVELOPMENT_SETUP.md - 開發環境設置
```

### 第二階段（重要）- P1 優先級

```markdown
13. docs/01_architecture/MODULE_DEPENDENCY_MAP.md - 模組依賴圖
14. docs/01_architecture/DATA_FLOW_DIAGRAM.md - 數據流向圖
15. docs/02_design_decisions/ADR_001_CLI_vs_DirectImport.md - CLI設計決策（重寫）
16. docs/02_design_decisions/ADR_002_RAG_Integration_Strategy.md - RAG整合決策
17. docs/02_design_decisions/ADR_005_Security_Model.md - 安全模型決策
18. docs/04_user_guides/CONFIGURATION_GUIDE.md - 配置指南
19. docs/04_user_guides/TROUBLESHOOTING_GUIDE.md - 故障排除
20. docs/05_api_reference/ERROR_CODES_REFERENCE.md - 錯誤碼表
21. docs/rag_system/RAG_SYSTEM_OVERVIEW.md - RAG總覽
22. docs/learning_system/LEARNING_SYSTEM_OVERVIEW.md - 學習系統總覽
```

### 第三階段（建議）- P2 優先級

其餘 P2 優先級文檔可以根據實際需求逐步建立。

---

## 📝 九、文檔模板建議

### 9.1 ADR（架構決策記錄）模板

```markdown
# ADR-XXX: [決策標題]

**狀態**: [提議中 | 已接受 | 已棄用 | 已取代]  
**日期**: YYYY-MM-DD  
**決策者**: [團隊/個人]

## 背景

描述需要做出決策的背景和問題。

## 決策

我們將會 [決策內容]。

## 理由

- 原因 1
- 原因 2
- 原因 3

## 後果

### 優點
- 優點 1
- 優點 2

### 缺點
- 缺點 1
- 缺點 2

## 替代方案

### 方案 A
[描述和理由]

### 方案 B
[描述和理由]

## 相關文檔
- [相關ADR]
- [相關架構文檔]
```

### 9.2 架構文檔模板

```markdown
# [模組名稱] 架構文檔

> **版本**: v1.0 | **日期**: YYYY-MM-DD | **狀態**: 草稿/審查/已發布

## 概覽

[一段話描述這個模組的核心功能]

## 目錄

- [架構圖](#架構圖)
- [核心組件](#核心組件)
- [數據流向](#數據流向)
- [API接口](#api接口)
- [配置說明](#配置說明)
- [部署指南](#部署指南)

## 架構圖

```
[ASCII圖或Mermaid圖]
```

## 核心組件

### 組件A
- **職責**: 
- **依賴**: 
- **接口**: 

## 數據流向

1. 步驟1
2. 步驟2
3. 步驟3

## 相關文檔
- [相關架構文檔]
- [相關API文檔]
```

---

## 🎯 十、執行時間表

| 階段 | 任務 | 預計時間 | 負責人 |
|------|------|----------|--------|
| **Phase 1** | 創建目錄結構 | 10 分鐘 | - |
| **Phase 2** | 移動現有文檔 | 30 分鐘 | - |
| **Phase 3** | 更新 ARCHIVE_INDEX | 15 分鐘 | - |
| **驗證** | 檢查所有鏈接 | 30 分鐘 | - |
| **第一階段文檔** | 建立 P0 文檔 (12個) | 2-3 天 | 團隊 |
| **第二階段文檔** | 建立 P1 文檔 (10個) | 1-2 週 | 團隊 |
| **第三階段文檔** | 建立 P2 文檔 (8個) | 根據需求 | 團隊 |

---

**整理日期**: 2026-01-28  
**下一步**: 執行 Phase 1-3，然後開始建立優先級 P0 文檔
