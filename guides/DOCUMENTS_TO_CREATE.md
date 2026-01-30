# AIVA 待建立文檔清單

**最後更新**: 2026-01-28  
**目的**: 列出所有需要建立的文檔，供自動化工具或手動建立使用  
**來源**: DOCUMENTATION_REORGANIZATION_PLAN.md 第四章

---

## 📋 清單說明

本清單包含 AIVA 專案所有待建立的文檔，共 **56 個文檔**：
- 🔴 **P0 必須** (18個) - 核心文檔，影響專案使用
- 🟡 **P1 重要** (24個) - 重要文檔，影響開發效率
- 🟢 **P2 建議** (8個) - 補充文檔，提升完整性
- ✅ **已完成** (6個) - 現有文檔

---

## 📊 統計摘要

| 優先級 | 說明 | 數量 | 佔比 |
|--------|------|------|------|
| 🔴 **P0 - 必須** | 核心文檔，影響專案使用 | **18** | 32% |
| 🟡 **P1 - 重要** | 重要文檔，影響開發效率 | **24** | 43% |
| 🟢 **P2 - 建議** | 補充文檔，提升完整性 | **8** | 14% |
| ✅ **已完成** | 現有文檔 | **6** | 11% |
| **總計** | | **56** | 100% |

---

## 📑 一、核心架構文檔 (7個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `SYSTEM_OVERVIEW.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 系統整體概覽、核心概念說明 |
| `ARCHITECTURE_PRINCIPLES.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 架構設計原則與理念 |
| `SECURITY_ARCHITECTURE.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 安全架構設計 |
| `MODULE_DEPENDENCY_MAP.md` | `docs/01_architecture/` | ❌ 缺少 | 🟡 P1 | 模組依賴關係圖 |
| `DATA_FLOW_DIAGRAM.md` | `docs/01_architecture/` | ❌ 缺少 | 🟡 P1 | 完整數據流向圖 |
| `AI_CAPABILITY_SELECTION_MECHANISM.md` | `docs/01_architecture/` | ✅ 已有 | - | AI能力選擇機制（已生成） |
| `SERVICES_ARCHITECTURE_ANALYSIS.md` | `docs/01_architecture/` | ✅ 已有 | - | Services架構分析（已生成） |

**P0 必須 (3個)**: SYSTEM_OVERVIEW, ARCHITECTURE_PRINCIPLES, SECURITY_ARCHITECTURE  
**P1 重要 (2個)**: MODULE_DEPENDENCY_MAP, DATA_FLOW_DIAGRAM  
**已完成 (2個)**: AI_CAPABILITY_SELECTION_MECHANISM, SERVICES_ARCHITECTURE_ANALYSIS

---

## 📑 二、設計決策文檔 ADR (6個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `ADR_005_Security_Model.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🔴 P0 | 安全模型設計決策 |
| `ADR_001_CLI_vs_DirectImport.md` | `docs/02_design_decisions/` | ⚠️ 部分 | 🟡 P1 | 為何選擇CLI架構（需重寫為ADR格式） |
| `ADR_002_RAG_Integration_Strategy.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟡 P1 | RAG整合策略決策 |
| `ADR_003_Learning_System_Design.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟡 P1 | 學習系統設計決策 |
| `DESIGN_DECISIONS_INDEX.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟡 P1 | 設計決策索引 |
| `ADR_004_Multi_Language_Engine.md` | `docs/02_design_decisions/` | ❌ 缺少 | 🟢 P2 | 多語言引擎選擇決策 |

**P0 必須 (1個)**: ADR_005_Security_Model  
**P1 重要 (4個)**: ADR_001~003, DESIGN_DECISIONS_INDEX  
**P2 建議 (1個)**: ADR_004_Multi_Language_Engine

---

## 📑 三、API 參考文檔 (6個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `API_OVERVIEW.md` | `docs/05_api_reference/` | ❌ 缺少 | 🔴 P0 | API總覽與版本說明 |
| `CLI_COMMANDS_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🔴 P0 | 所有CLI命令完整參考 |
| `CORE_API_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟡 P1 | Core模組API文檔 |
| `FEATURES_API_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟡 P1 | Features模組API文檔 |
| `ERROR_CODES_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟡 P1 | 錯誤碼對照表 |
| `RAG_API_REFERENCE.md` | `docs/05_api_reference/` | ❌ 缺少 | 🟢 P2 | RAG系統API文檔 |

**P0 必須 (2個)**: API_OVERVIEW, CLI_COMMANDS_REFERENCE  
**P1 重要 (3個)**: CORE_API, FEATURES_API, ERROR_CODES  
**P2 建議 (1個)**: RAG_API_REFERENCE

---

## 📑 四、使用指南 (8個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `GETTING_STARTED.md` | `guides/general/` | ❌ 缺少 | 🔴 P0 | 快速開始指南 |
| `INSTALLATION_GUIDE.md` | `guides/general/` | ❌ 缺少 | 🔴 P0 | 詳細安裝指南 |
| `CONFIGURATION_GUIDE.md` | `guides/general/` | ❌ 缺少 | 🟡 P1 | 系統配置指南 |
| `COMMON_WORKFLOWS.md` | `guides/general/` | ❌ 缺少 | 🟡 P1 | 常見工作流程 |
| `TROUBLESHOOTING_GUIDE.md` | `guides/troubleshooting/` | ❌ 缺少 | 🟡 P1 | 故障排除指南 |
| `FAQ.md` | `guides/general/` | ❌ 缺少 | 🟢 P2 | 常見問題解答 |
| `EXTERNAL_CAPABILITIES_USAGE_GUIDE.md` | `guides/general/` | ✅ 已有 | - | 外部能力使用指南 |
| `FUNCTION_CALLABLE_JUDGMENT_GUIDE.md` | `guides/development/` | ✅ 已有 | - | 功能可調用判斷指南 |

**P0 必須 (2個)**: GETTING_STARTED, INSTALLATION_GUIDE  
**P1 重要 (3個)**: CONFIGURATION, COMMON_WORKFLOWS, TROUBLESHOOTING  
**P2 建議 (1個)**: FAQ  
**已完成 (2個)**: EXTERNAL_CAPABILITIES_USAGE, FUNCTION_CALLABLE_JUDGMENT

---

## 📑 五、部署與運維文檔 (6個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `DEPLOYMENT_GUIDE.md` | `guides/deployment/` | ❌ 缺少 | 🔴 P0 | 完整部署指南 |
| `DOCKER_SETUP.md` | `guides/deployment/` | ❌ 缺少 | 🟡 P1 | Docker容器化部署 |
| `PRODUCTION_CHECKLIST.md` | `guides/deployment/` | ❌ 缺少 | 🟡 P1 | 生產環境檢查清單 |
| `MONITORING_AND_LOGGING.md` | `guides/deployment/` | ❌ 缺少 | 🟡 P1 | 監控與日誌配置 |
| `BACKUP_AND_RECOVERY.md` | `guides/deployment/` | ❌ 缺少 | 🟢 P2 | 備份與災難恢復 |
| `PERFORMANCE_TUNING.md` | `guides/deployment/` | ❌ 缺少 | 🟢 P2 | 性能調優指南 |

**P0 必須 (1個)**: DEPLOYMENT_GUIDE  
**P1 重要 (3個)**: DOCKER_SETUP, PRODUCTION_CHECKLIST, MONITORING_AND_LOGGING  
**P2 建議 (2個)**: BACKUP_AND_RECOVERY, PERFORMANCE_TUNING

---

## 📑 六、開發文檔 (6個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `CONTRIBUTING.md` | 根目錄 | ❌ 缺少 | 🔴 P0 | 貢獻者指南 |
| `DEVELOPMENT_SETUP.md` | `guides/development/` | ❌ 缺少 | 🔴 P0 | 開發環境設置 |
| `CODE_STYLE_GUIDE.md` | `guides/development/` | ❌ 缺少 | 🟡 P1 | 代碼風格指南 |
| `TESTING_GUIDE.md` | `guides/development/` | ❌ 缺少 | 🟡 P1 | 測試指南與規範 |
| `COMMIT_CONVENTIONS.md` | `guides/development/` | ❌ 缺少 | 🟢 P2 | Git提交規範 |
| `UNIFIED_NAMING_CONVENTION.md` | `docs/01_architecture/` | ✅ 已有 | - | 統一命名規範 |

**P0 必須 (2個)**: CONTRIBUTING, DEVELOPMENT_SETUP  
**P1 重要 (2個)**: CODE_STYLE_GUIDE, TESTING_GUIDE  
**P2 建議 (1個)**: COMMIT_CONVENTIONS  
**已完成 (1個)**: UNIFIED_NAMING_CONVENTION

---

## 📑 七、RAG 與學習系統文檔 (7個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `RAG_SYSTEM_OVERVIEW.md` | `docs/rag_system/` | ❌ 缺少 | 🟡 P1 | RAG系統總覽 |
| `LEARNING_SYSTEM_OVERVIEW.md` | `docs/learning_system/` | ❌ 缺少 | 🟡 P1 | 學習系統總覽 |
| `RAG_CLI_COMMAND_DECISION_SYSTEM.md` | `docs/rag_system/` | ✅ 已有 | - | RAG CLI決策系統 |
| `RAG_INTERNAL_EXPLORATION_INTEGRATION.md` | `docs/rag_system/` | ✅ 已有 | - | RAG內部探索整合 |
| `RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md` | `guides/technical/` | ✅ 已有 | - | RAG觸發與通知指南 |
| `VECTOR_STORE_AND_RAG_ARCHITECTURE.md` | `docs/rag_system/` | ✅ 已有 | - | 向量庫與RAG架構 |
| `LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md` | `docs/learning_system/` | ✅ 已有 | - | 學習系統完整架構 |

**P1 重要 (2個)**: RAG_SYSTEM_OVERVIEW, LEARNING_SYSTEM_OVERVIEW  
**已完成 (5個)**: RAG相關4個 + LEARNING_SYSTEM_COMPLETE_ARCHITECTURE

---

## 📑 八、安全與合規文檔 (5個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `SECURITY.md` | 根目錄 | ❌ 缺少 | 🔴 P0 | 安全政策聲明 |
| `LICENSE.md` | 根目錄 | ❌ 缺少 | 🔴 P0 | 開源授權條款 |
| `VULNERABILITY_HANDLING.md` | `docs/` | ❌ 缺少 | 🟡 P1 | 漏洞處理流程 |
| `COMPLIANCE_CHECKLIST.md` | `docs/` | ❌ 缺少 | 🟢 P2 | 合規檢查清單 |
| `SECURITY_ARCHITECTURE.md` | `docs/01_architecture/` | ❌ 缺少 | 🔴 P0 | 安全架構設計（重複，見第一章） |

**P0 必須 (2個)**: SECURITY, LICENSE  
**P1 重要 (1個)**: VULNERABILITY_HANDLING  
**P2 建議 (1個)**: COMPLIANCE_CHECKLIST

---

## 📑 九、其他核心文檔 (5個)

| 檔名 | 目標路徑 | 狀態 | 優先級 | 說明 |
|------|----------|------|--------|------|
| `README.md` | 根目錄 | ✅ 已有 | - | 專案說明文檔 |
| `CHANGELOG.md` | 根目錄 | ✅ 已有 | - | 版本變更歷史 |
| `ROADMAP.md` | 根目錄 | ❌ 缺少 | 🟡 P1 | 專案路線圖 |
| `ARCHITECTURE_OVERVIEW.md` | `docs/01_architecture/` | ❌ 缺少 | 🟡 P1 | 架構總覽（整合架構文檔） |
| `CLI_USAGE_GUIDE.md` | `guides/general/` | ❌ 缺少 | 🟡 P1 | CLI使用指南 |

**P1 重要 (3個)**: ROADMAP, ARCHITECTURE_OVERVIEW, CLI_USAGE_GUIDE  
**已完成 (2個)**: README, CHANGELOG

---

## 🎯 優先級建議執行順序

### 第一階段：P0 必須文檔（18個，預計2-3天）

#### 根目錄 (5個)
1. `SECURITY.md` - 安全政策
2. `LICENSE.md` - 授權條款
3. `CONTRIBUTING.md` - 貢獻指南

#### 架構文檔 (3個)
4. `docs/01_architecture/SYSTEM_OVERVIEW.md` - 系統總覽
5. `docs/01_architecture/ARCHITECTURE_PRINCIPLES.md` - 架構原則
6. `docs/01_architecture/SECURITY_ARCHITECTURE.md` - 安全架構

#### 使用指南 (2個)
7. `guides/general/GETTING_STARTED.md` - 快速開始
8. `guides/general/INSTALLATION_GUIDE.md` - 安裝指南

#### API 參考 (2個)
9. `docs/05_api_reference/API_OVERVIEW.md` - API總覽
10. `docs/05_api_reference/CLI_COMMANDS_REFERENCE.md` - CLI命令參考

#### 部署運維 (1個)
11. `guides/deployment/DEPLOYMENT_GUIDE.md` - 部署指南

#### 開發文檔 (2個)
12. `guides/development/DEVELOPMENT_SETUP.md` - 開發環境

#### 設計決策 (1個)
13. `docs/02_design_decisions/ADR_005_Security_Model.md` - 安全模型決策

---

### 第二階段：P1 重要文檔（24個，預計1-2週）

#### 架構文檔 (2個)
- `docs/01_architecture/MODULE_DEPENDENCY_MAP.md`
- `docs/01_architecture/DATA_FLOW_DIAGRAM.md`

#### 設計決策 (4個)
- `docs/02_design_decisions/ADR_001_CLI_vs_DirectImport.md` (重寫)
- `docs/02_design_decisions/ADR_002_RAG_Integration_Strategy.md`
- `docs/02_design_decisions/ADR_003_Learning_System_Design.md`
- `docs/02_design_decisions/DESIGN_DECISIONS_INDEX.md`

#### API 參考 (3個)
- `docs/05_api_reference/CORE_API_REFERENCE.md`
- `docs/05_api_reference/FEATURES_API_REFERENCE.md`
- `docs/05_api_reference/ERROR_CODES_REFERENCE.md`

#### 使用指南 (3個)
- `guides/general/CONFIGURATION_GUIDE.md`
- `guides/general/COMMON_WORKFLOWS.md`
- `guides/troubleshooting/TROUBLESHOOTING_GUIDE.md`

#### 部署運維 (3個)
- `guides/deployment/DOCKER_SETUP.md`
- `guides/deployment/PRODUCTION_CHECKLIST.md`
- `guides/deployment/MONITORING_AND_LOGGING.md`

#### 開發文檔 (2個)
- `guides/development/CODE_STYLE_GUIDE.md`
- `guides/development/TESTING_GUIDE.md`

#### RAG與學習系統 (2個)
- `docs/rag_system/RAG_SYSTEM_OVERVIEW.md`
- `docs/learning_system/LEARNING_SYSTEM_OVERVIEW.md`

#### 安全合規 (1個)
- `docs/VULNERABILITY_HANDLING.md`

#### 其他 (3個)
- `ROADMAP.md`
- `docs/01_architecture/ARCHITECTURE_OVERVIEW.md`
- `guides/general/CLI_USAGE_GUIDE.md`

---

### 第三階段：P2 建議文檔（8個，按需建立）

#### 設計決策 (1個)
- `docs/02_design_decisions/ADR_004_Multi_Language_Engine.md`

#### API 參考 (1個)
- `docs/05_api_reference/RAG_API_REFERENCE.md`

#### 使用指南 (1個)
- `guides/general/FAQ.md`

#### 部署運維 (2個)
- `guides/deployment/BACKUP_AND_RECOVERY.md`
- `guides/deployment/PERFORMANCE_TUNING.md`

#### 開發文檔 (1個)
- `guides/development/COMMIT_CONVENTIONS.md`

#### 安全合規 (1個)
- `docs/COMPLIANCE_CHECKLIST.md`

---

## 📝 文檔模板參考

### ADR (Architecture Decision Record) 模板

```markdown
# ADR-XXX: [決策標題]

**日期**: YYYY-MM-DD  
**狀態**: [提議 | 接受 | 已棄用 | 被取代]  
**決策者**: [決策者列表]

## 背景與問題

[描述需要做決策的技術背景和要解決的問題]

## 考慮的選項

### 選項 1: [選項名稱]
- **優點**:
- **缺點**:

### 選項 2: [選項名稱]
- **優點**:
- **缺點**:

## 決策

[描述最終選擇的選項及原因]

## 後果

### 正面影響
- [列舉正面影響]

### 負面影響
- [列舉負面影響]

### 風險與緩解
- [列舉風險及緩解措施]

## 相關文檔

- [相關ADR連結]
- [相關架構文檔]
```

---

### 架構文檔模板

```markdown
# [模組/系統名稱] 架構文檔

**版本**: v1.0  
**最後更新**: YYYY-MM-DD  
**負責人**: [負責人]

## 📋 概述

[簡要描述模組用途和核心功能]

## 🏗️ 架構設計

### 核心組件

[描述主要組件及其職責]

### 模組依賴

[列出依賴的其他模組]

### 數據流向

[描述數據如何在系統中流動]

## 🔧 技術實現

### 技術棧

- **語言**: [程式語言]
- **框架**: [使用的框架]
- **資料庫**: [資料庫類型]

### 關鍵設計決策

[列出重要的設計決策並連結到相關ADR]

## 📊 性能考量

[描述性能特性和限制]

## 🔒 安全考量

[描述安全機制和注意事項]

## 📚 相關文檔

- [API文檔連結]
- [使用指南連結]
- [相關ADR連結]
```

---

## 🔄 維護說明

### 更新此清單

當以下情況發生時需要更新此清單：
1. ✅ 完成文檔建立 - 將狀態改為「✅ 已有」
2. ➕ 新增待建文檔 - 加入對應分類並標註優先級
3. 🔄 變更文檔位置 - 更新「目標路徑」欄位
4. 📝 調整優先級 - 根據專案需求調整 P0/P1/P2

### 驗證方式

```powershell
# 檢查文檔是否存在
$checkList = @(
    "docs/01_architecture/SYSTEM_OVERVIEW.md",
    "guides/general/GETTING_STARTED.md",
    # ... 其他文檔路徑
)

foreach ($doc in $checkList) {
    $exists = Test-Path "C:\D\fold7\AIVA-git\$doc"
    Write-Host "$(if ($exists) {'✅'} else {'❌'}) $doc"
}
```

---

## 📞 聯繫方式

如對此清單有任何疑問或建議，請聯繫專案維護者。

---

**文檔結束**
