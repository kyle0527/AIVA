# 📚 AIVA 所有指南和手冊分類說明

## 📑 目錄

- [📊 快速統計](#快速統計)
- [🎯 1. 使用者手冊 (7個) - **面向最終使用者**](#1-使用者手冊-7個-面向最終使用者)
  - [✅ 已整理到 `docs/user_guides/` (6個)](#已整理到-docsuserguides-6個)
    - [00_general - 通用指南](#00general-通用指南)
    - [01_core - Core 模組手冊](#01core-core-模組手冊)
  - [✅ 保留在原位 (1個)](#保留在原位-1個)
- [👨‍💻 2. 開發指南 (14個) - **面向開發者**](#2-開發指南-14個-面向開發者)
  - [📁 `guides/development/` (12個)](#guidesdevelopment-12個)
  - [📁 其他位置 (2個)](#其他位置-2個)
- [🏗️ 3. 架構指南 (5個) - **面向架構師**](#3-架構指南-5個-面向架構師)
  - [📁 `guides/architecture/`](#guidesarchitecture)
- [🚀 4. 部署指南 (4個) - **面向運維**](#4-部署指南-4個-面向運維)
  - [📁 `guides/deployment/`](#guidesdeployment)
- [🔧 5. 故障排除指南 (4個) - **面向技術支持**](#5-故障排除指南-4個-面向技術支持)
  - [📁 `guides/troubleshooting/`](#guidestroubleshooting)
- [📦 6. 模組開發指南 (8個) - **面向模組開發者**](#6-模組開發指南-8個-面向模組開發者)
  - [📁 `guides/modules/`](#guidesmodules)
- [📄 7. 技術文檔/報告 (23個) - **參考和記錄**](#7-技術文檔報告-23個-參考和記錄)
  - [🔹 Services 內部文檔 (5個)](#services-內部文檔-5個)
  - [🔹 Coordinators 文檔 (2個)](#coordinators-文檔-2個)
  - [🔹 項目級文檔 (7個)](#項目級文檔-7個)
  - [🔹 開發者文檔 (2個)](#開發者文檔-2個)
  - [🔹 驗證報告 (3個)](#驗證報告-3個)
  - [🔹 整合文檔 (2個)](#整合文檔-2個)
  - [🔹 修復指南 (2個)](#修復指南-2個)
  - [🔹 存檔文檔 (1個)](#存檔文檔-1個)
- [🎯 核心區分標準](#核心區分標準)
  - [✅ 使用者手冊 (User Manual/Guide)](#使用者手冊-user-manualguide)
  - [📖 開發/技術指南 (Development/Technical Guide)](#開發技術指南-developmenttechnical-guide)
  - [📄 技術文檔/報告 (Documentation/Report)](#技術文檔報告-documentationreport)
- [📊 總結](#總結)
  - [真正的使用者手冊: **7個**](#真正的使用者手冊-7個)
  - [開發/技術指南: **31個**](#開發技術指南-31個)
  - [技術文檔/報告: **27個**](#技術文檔報告-27個)
- [🔍 如何識別使用者手冊](#如何識別使用者手冊)

---
---
---
---

## 📊 快速統計

| 類別 | 數量 | 說明 |
|------|------|------|
| **使用者手冊** | 7 | 面向最終使用者的完整手冊 |
| **開發指南** | 14 | 面向開發者的技術指南 |
| **架構指南** | 5 | 系統架構和 Schema 設計 |
| **部署指南** | 4 | 安裝、配置、Docker/K8s |
| **故障排除指南** | 4 | 問題診斷和修復 |
| **模組開發指南** | 8 | 各語言模組開發 |
| **技術文檔/報告** | 23 | 更新記錄、驗證報告等 |

---

## 🎯 1. 使用者手冊 (7個) - **面向最終使用者**

> **特徵**: 完整的端到端使用說明，包含安裝、配置、使用、故障排除

### ✅ 已整理到 `docs/user_guides/` (6個)

#### 00_general - 通用指南
1. **AIVA_USER_MANUAL.md** (144KB)
   - 📍 `docs/user_guides/00_general/`
   - 📝 AIVA 系統整體使用手冊
   - 👥 適用對象: 所有使用者
   - 📦 內容: 系統簡介、快速開始、AI功能、API使用、故障排除

2. **AIVA_MODEL_GUIDE.md** (10KB)
   - 📍 `docs/user_guides/00_general/`
   - 📝 AI 模型權重管理與載入指南
   - 👥 適用對象: AI 工程師
   - 📦 內容: 模型權重管理、載入機制、性能優化

#### 01_core - Core 模組手冊
3. **AIVA_CORE_使用者手冊.md** (100KB)
   - 📍 `docs/user_guides/01_core/`
   - 📝 Core 模組完整使用指南
   - 👥 適用對象: Core 開發者、系統管理員
   - 📦 內容: 六大模組測試、常見問題、進階操作

4. **REAL_AI_CORE_OPERATIONS_MANUAL.md** (34KB)
   - 📍 `docs/user_guides/01_core/`
   - 📝 建立真實 AI 核心的操作手冊
   - 👥 適用對象: AI 架構師
   - 📦 內容: AI 能力評估、系統需求、建制流程、性能優化

5. **AIVA_AI_USER_MANUAL.md** (151KB)
   - 📍 `docs/user_guides/01_core/`
   - 📝 AI 功能使用指南
   - 👥 適用對象: AI 使用者
   - 📦 內容: AI 決策、RAG 檢索、分析掃描、API 使用

6. **AI_SERVICES_USER_GUIDE.md** (101KB)
   - 📍 `docs/user_guides/01_core/`
   - 📝 AI 系統實際使用指南 (v6.0-dev)
   - 👥 適用對象: 開發者、研究員
   - 📦 內容: 雙重閉環架構、術語規範、實際功能說明

### ✅ 保留在原位 (1個)
7. **SCAN_USER_GUIDE.md** (26KB)
   - 📍 `services/scan/`
   - 📝 Scan 模組使用者手冊 (v2.1)
   - 👥 適用對象: 安全測試人員
   - 📦 內容: 兩階段掃描流程、AI 命令接口、監控和結果查看
   - ⚠️ 保留原因: 技術文檔，與代碼緊密相關

---

## 👨‍💻 2. 開發指南 (14個) - **面向開發者**

> **特徵**: 具體開發任務的操作指南

### 📁 `guides/development/` (12個)

1. **DEVELOPMENT_QUICK_START_GUIDE.md**
   - 快速開始開發指南
   - 內容: 環境設置、基本工作流程

2. **DEVELOPMENT_TASKS_GUIDE.md**
   - 開發任務指南
   - 內容: 常見開發任務的操作步驟

3. **API_VERIFICATION_GUIDE.md**
   - API 驗證指南
   - 內容: API 測試和驗證方法

4. **DEPENDENCY_MANAGEMENT_GUIDE.md**
   - 依賴管理指南
   - 內容: Python 依賴、虛擬環境管理

5. **DATA_STORAGE_GUIDE.md**
   - 數據存儲指南
   - 內容: 數據持久化、存儲策略

6. **SCHEMA_IMPORT_GUIDE.md**
   - Schema 導入指南
   - 內容: 數據 Schema 的導入和使用

7. **UI_LAUNCH_GUIDE.md**
   - UI 啟動指南
   - 內容: Web 界面啟動和配置

8. **METRICS_USAGE_GUIDE.md**
   - 指標使用指南
   - 內容: 性能指標收集和分析

9. **TOKEN_OPTIMIZATION_GUIDE.md**
   - Token 優化指南
   - 內容: LLM Token 使用優化

10. **LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md**
    - 語言服務器優化指南
    - 內容: Python/TypeScript LSP 優化

11. **EXTENSIONS_INSTALL_GUIDE.md**
    - 擴展安裝指南
    - 內容: VS Code 擴展安裝

12. **GIT_PUSH_GUIDELINES.md**
    - Git 推送準則
    - 內容: Git 工作流程、提交規範

### 📁 其他位置 (2個)

13. **AIVA_AI_MANUAL_UPDATE_LOG.md**
    - 📍 `guides/development/`
    - AI 手冊更新日誌
    - 內容: 手冊更新記錄

14. **MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md**
    - 📍 `guides/development/`
    - 多語言環境標準
    - 內容: 跨語言開發規範

---

## 🏗️ 3. 架構指南 (5個) - **面向架構師**

> **特徵**: 系統架構設計和 Schema 規範

### 📁 `guides/architecture/`

1. **SCHEMA_GUIDE.md**
   - Schema 設計指南
   - 內容: 數據 Schema 設計原則

2. **SCHEMA_GENERATION_GUIDE.md**
   - Schema 生成指南
   - 內容: 自動生成 Schema 的方法

3. **SCHEMA_COMPLIANCE_GUIDE.md**
   - Schema 合規指南
   - 內容: Schema 驗證和合規性檢查

4. **CROSS_LANGUAGE_SCHEMA_GUIDE.md**
   - 跨語言 Schema 指南
   - 內容: 多語言 Schema 統一

5. **CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md**
   - 跨語言兼容性指南
   - 內容: 多語言互操作性

---

## 🚀 4. 部署指南 (4個) - **面向運維**

> **特徵**: 系統安裝、配置、部署

### 📁 `guides/deployment/`

1. **INSTALLATION_GUIDE.md**
   - 安裝指南
   - 內容: 系統安裝步驟、依賴安裝

2. **ENVIRONMENT_CONFIG_GUIDE.md**
   - 環境配置指南
   - 內容: 環境變數、配置文件

3. **BUILD_GUIDE.md**
   - 構建指南
   - 內容: 項目構建流程

4. **DOCKER_KUBERNETES_GUIDE.md**
   - Docker/K8s 指南
   - 內容: 容器化部署、K8s 配置

---

## 🔧 5. 故障排除指南 (4個) - **面向技術支持**

> **特徵**: 問題診斷和修復方案

### 📁 `guides/troubleshooting/`

1. **IMPORT_ISSUES_RESOLUTION_GUIDE.md**
   - 導入問題解決指南
   - 內容: Python 導入錯誤診斷和修復

2. **FORWARD_REFERENCE_REPAIR_GUIDE.md**
   - 前向引用修復指南
   - 內容: Python 類型前向引用問題

3. **PERFORMANCE_OPTIMIZATION_GUIDE.md**
   - 性能優化指南
   - 內容: 系統性能診斷和優化

4. **TESTING_REPRODUCTION_GUIDE.md**
   - 測試重現指南
   - 內容: Bug 重現和測試方法

---

## 📦 6. 模組開發指南 (8個) - **面向模組開發者**

> **特徵**: 各語言/模組的開發指南

### 📁 `guides/modules/`

1. **PYTHON_DEVELOPMENT_GUIDE.md**
   - Python 開發指南
   - 內容: Python 模組開發規範

2. **GO_DEVELOPMENT_GUIDE.md**
   - Go 開發指南
   - 內容: Go 引擎開發規範

3. **RUST_DEVELOPMENT_GUIDE.md**
   - Rust 開發指南
   - 內容: Rust 引擎開發規範

4. **AI_ENGINE_GUIDE.md**
   - AI 引擎指南
   - 內容: AI 引擎開發和集成

5. **FEATURE_MODULES_DEVELOPMENT_GUIDE.md**
   - 功能模組開發指南
   - 內容: Features 模組開發

6. **ANALYSIS_FUNCTIONS_GUIDE.md**
   - 分析功能指南
   - 內容: 代碼分析功能開發

7. **SUPPORT_FUNCTIONS_GUIDE.md**
   - 支持功能指南
   - 內容: 輔助功能開發

8. **MODULE_MIGRATION_GUIDE.md**
   - 模組遷移指南
   - 內容: 模組升級和遷移

---

## 📄 7. 技術文檔/報告 (23個) - **參考和記錄**

> **特徵**: 更新記錄、驗證報告、使用說明

### 🔹 Services 內部文檔 (5個)

1. **USAGE_GUIDE.md**
   - 📍 `services/core/aiva_core/`
   - Core 模組使用說明

2. **SYSTEM_STARTUP_GUIDE.md**
   - 📍 `services/core/aiva_core/service_backbone/`
   - 系統啟動指南

3. **USAGE_GUIDE.md** (Go)
   - 📍 `services/scan/engines/go_engine/`
   - Go 引擎使用

4. **USAGE_GUIDE.md** (Rust)
   - 📍 `services/scan/engines/rust_engine/`
   - Rust 引擎使用

5. **OPERATION_GUIDE.md**
   - 📍 `services/scan/engines/typescript_engine/docs/`
   - TypeScript 引擎操作

### 🔹 Coordinators 文檔 (2個)

6. **PYTHON_ENGINE_USAGE_GUIDE.md**
   - 📍 `services/scan/coordinators/`
   - Python 引擎使用

7. **COORDINATOR_USAGE_GUIDE.md**
   - 📍 `services/scan/coordinators/`
   - 協調器使用

### 🔹 項目級文檔 (7個)

8. **GUIDES_UPDATE_SUMMARY_2025-11-22.md**
   - 📍 根目錄
   - 指南更新總結

9. **GUIDES_CLEANUP_SUMMARY_2025-11-22.md**
   - 📍 根目錄
   - 指南清理總結

10. **GUIDES_CLEANUP_ROUND2_SUMMARY_2025-11-22.md**
    - 📍 根目錄
    - 指南清理第二輪總結

11. **GUIDES_DIRECTORY_UPDATE_SUMMARY.md**
    - 📍 `guides/`
    - 指南目錄更新摘要

12. **GUIDES_DIRECTORY_UPDATE_REPORT.md**
    - 📍 `guides/`
    - 指南目錄更新報告

13. **EXTERNAL_GUIDES_INTEGRATION_PLAN.md**
    - 📍 `guides/`
    - 外部指南整合計畫

14. **FILE_ORGANIZATION_MAINTENANCE_GUIDE.md**
    - 📍 `reports/documentation/`
    - 文件組織維護指南

### 🔹 開發者文檔 (2個)

15. **DEVELOPER_GUIDE.md**
    - 📍 `reports/documentation/`
    - 開發者指南

16. **NODE_MODULES_GUIDE.md**
    - 📍 `services/scan/engines/typescript_engine/`
    - Node.js 模組指南

### 🔹 驗證報告 (3個)

17. **MANUAL_VALIDATION_REPORT.md**
    - 📍 `reports/analysis/`
    - 手冊驗證報告

18. **DOCKER_GUIDE_VALIDATION_REPORT.md**
    - 📍 `reports/architecture/`
    - Docker 指南驗證報告

19. **ARCHITECTURE_FIXES_VALIDATION_GUIDE.md**
    - 📍 `reports/architecture/`
    - 架構修復驗證指南

### 🔹 整合文檔 (2個)

20. **AIVA_WEB_RESEARCH_INTEGRATION_GUIDE.md**
    - 📍 `docs/guides/integration/`
    - Web 研究整合指南

21. **AIVA_5M_REPLACEMENT_IMPLEMENTATION_GUIDE.md**
    - 📍 `docs/guides/integration/`
    - 5M 模型替換實施指南

### 🔹 修復指南 (2個)

22. **AIVA_AI_REPAIR_GUIDE.md**
    - 📍 `docs/guides/`
    - AI 修復指南

23. **MERMAID_SMART_REPAIR_GUIDE.md**
    - 📍 `docs/reports/mermaid/`
    - Mermaid 圖表智能修復指南

### 🔹 存檔文檔 (1個)

24. **MULTI_ENGINE_COORDINATION_GUIDE.md**
    - 📍 `services/scan/archived_docs/`
    - 多引擎協調指南（已存檔）

---

## 🎯 核心區分標準

### ✅ 使用者手冊 (User Manual/Guide)
- **特徵**: 
  - 完整的端到端使用說明
  - 面向最終使用者
  - 包含安裝、配置、使用、故障排除
  - 通常 50KB+ 大小
  - 有完整的目錄結構

- **典型標題模式**:
  - `*USER_MANUAL.md`
  - `*USER_GUIDE.md`
  - `*使用者手冊.md`
  - `*OPERATIONS_MANUAL.md`

### 📖 開發/技術指南 (Development/Technical Guide)
- **特徵**:
  - 針對特定開發任務
  - 面向開發者/架構師
  - 聚焦於具體技術問題
  - 通常 10-30KB 大小
  - 結構較簡單

- **典型標題模式**:
  - `*DEVELOPMENT_GUIDE.md`
  - `*INSTALLATION_GUIDE.md`
  - `*OPTIMIZATION_GUIDE.md`
  - `SCHEMA_*.md`

### 📄 技術文檔/報告 (Documentation/Report)
- **特徵**:
  - 更新記錄、驗證報告
  - 參考性質，非操作指南
  - 通常較短
  - 歷史記錄性質

- **典型標題模式**:
  - `*SUMMARY.md`
  - `*REPORT.md`
  - `*UPDATE_LOG.md`
  - `*VALIDATION_*.md`

---

## 📊 總結

### 真正的使用者手冊: **7個**
這些是面向最終使用者的完整手冊，應該集中在 `docs/user_guides/` 管理。

### 開發/技術指南: **31個**
這些是面向開發者的技術指南，按主題分類在 `guides/` 目錄下。

### 技術文檔/報告: **27個**
這些是參考文檔和記錄，分散在各模組和 `reports/` 目錄。

---

## 🔍 如何識別使用者手冊

當遇到新文件時，檢查:

1. **文件名**包含 `USER` 或 `使用者`
2. **大小** > 50KB (通常)
3. **目錄結構**包含多個主要章節
4. **內容**涵蓋安裝→配置→使用→故障排除完整流程
5. **適用對象**是最終使用者而非開發者

符合 3 個以上標準 = 使用者手冊 ✅

---

**維護建議**:
- 使用者手冊應集中在 `docs/user_guides/`
- 開發指南保持在 `guides/` 分類目錄
- 技術文檔/報告可定期存檔
- 定期審核文件分類和內容更新
