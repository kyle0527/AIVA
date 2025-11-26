# 📚 AIVA 指南中心

> **📋 指南分類**: 按功能和領域分類的完整指南集合  
> **🎯 使用目的**: 為不同角色提供專業的技術文檔和操作手冊  
> **📅 最後更新**: 2025-11-25  
> **🏗️ 架構版本**: v2.0 數據合約驅動架構  
> **🐍 Python 環境**: 全域環境 (341 個套件)  
> **✅ 組織狀態**: 主要指南已更新為全域環境配置  
> **🔗 連結狀態**: 所有指南相互連結，支持快速跳轉

---

## 🏗️ 指南架構總覽

```
guides/
├── README.md                    # 📋 本索引文件
├── development/                 # 🛠️ 開發相關指南
├── architecture/                # 🏗️ 架構設計指南
├── modules/                     # ⚙️ 模組專業指南
├── deployment/                  # 🚀 部署運維指南
├── troubleshooting/            # 🔧 疑難排解指南
└── contracts/                   # 📋 數據合約文檔
```

### 🎯 **AIVA v2.0 系統架構** (2025-11-22)

#### **六大核心服務**
```
🏗️ AIVA 服務架構
├── 🤖 Core          # AI 驅動核心引擎 (23 AI components)
├── 🔗 Common        # 共享基礎設施 (100+ 模組)
├── 🎯 Features      # 多語言安全功能 (10+ 模組)
├── 🔄 Integration   # 企業級整合中樞
├── 🔍 Scan          # 多語言統一掃描引擎
└── 📁 Services      # 服務管理層
```

#### **雙閉環自我優化系統**
```
🔄 雙閉環架構
├── 內部閉環 (Know Thyself)
│   ├── 探索系統 (Exploration) - 自我診斷和代碼分析
│   ├── RAG 增強 - 知識庫檢索和上下文理解
│   └── 能力評估 - 了解自身能力與缺口
│
└── 外部閉環 (Learn from Battle)
    ├── 掃描系統 (Scan) - 目標系統偵測
    ├── 攻擊系統 (Attack) - 實戰測試和反饋
    └── 持續進化 - 從實戰中學習優化
```

#### **五大程式模組 + 六大核心服務**
- **五大程式模組**: Core, Features, Scan, Integration, Common
- **六大核心服務**: 包含上述五個 + Services 管理層
- **架構升級**: v2.0 移除 RabbitMQ，採用數據合約直接通信
- **類型安全**: Pydantic 數據合約，零外部依賴

---

## 📖 指南分類目錄

### 🏆 **核心綜合指南** (頂級參考)

| 文檔名稱 | 路徑 | 適用對象 | 完整度 |
|---------|------|----------|--------|
| **AIVA 系統架構文檔** | [`../README.md`](../README.md) | 🎯 所有開發者、架構師 | ✅ **v2.0** |
| **Services 架構總覽** | [`../services/README.md`](../services/README.md) | 🎯 服務架構師、核心開發者 | ✅ **六大服務** |
| **開發者指南** | [`../reports/documentation/DEVELOPER_GUIDE.md`](../reports/documentation/DEVELOPER_GUIDE.md) | 🛠️ 開發者、貢獻者、新手入門 | ✅ 完整 |
| **AIVA 綜合技術手冊** | [`../reports/documentation/AIVA_COMPREHENSIVE_GUIDE.md`](../reports/documentation/AIVA_COMPREHENSIVE_GUIDE.md) | 🎯 AI工程師、系統架構師 | ✅ 完整 |

### 🔗 **架構與設計文檔**

| 文檔名稱 | 路徑 | 核心概念 | 狀態 |
|---------|------|----------|------|
| **完整架構設計** | [`../docs/ARCHITECTURE_COMPLETE_DESIGN.md`](../docs/ARCHITECTURE_COMPLETE_DESIGN.md) | 🏗️ 系統架構完整設計理念 | ✅ **必讀** |
| **完整工作流程圖表** | [`../docs/COMPLETE_WORKFLOW_VISUALIZATION.md`](../docs/COMPLETE_WORKFLOW_VISUALIZATION.md) | 🔄 所有模組運作流程視覺化 | ✅ **必讀** |
| **AI 雙閉環自我優化** | [`../docs/AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../docs/AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md) | 🔄 內部探索與外部實戰雙閉環 | ✅ **核心文檔** |
| **掃描工作流程與數據流** | [`../docs/SCAN_WORKFLOW_AND_DATA_FLOW.md`](../docs/SCAN_WORKFLOW_AND_DATA_FLOW.md) | 🔍 多引擎掃描協同細節 | ✅ **最新** |
| **數據合約架構** | [`../SCAN_MODULE_RABBITMQ_REMOVAL.md`](../SCAN_MODULE_RABBITMQ_REMOVAL.md) | 🔄 v2.0 架構升級詳解 | ✅ **最新** |
| **術語對照表** | [`../TERMINOLOGY_GLOSSARY.md`](../TERMINOLOGY_GLOSSARY.md) | 📖 統一術語規範 | ✅ **必讀** |
| **實用工具遷移** | [`../UTILITY_TOOLS_MIGRATION.md`](../UTILITY_TOOLS_MIGRATION.md) | 🛠️ 工具位置和使用 | ✅ **2025-11-22** |

### 🛠️ **實用工具**

| 工具位置 | 功能 | 用途 |
|---------|------|------|
| `services/core/tools/` | AI 系統連接檢查 | 每日健康檢查 |
| `services/aiva_common/tools/` | 模組通連性檢查 | 每週系統檢查 |
| `services/integration/tools/` | SOP 合規性檢查 | 每週合規檢查 |
| `services/features/common/testers/` | 安全測試器 | 漏洞檢測工具 |

### 🛠️ **開發相關指南** (`development/`)

| 指南類型 | 文檔路徑 | 專業領域 | 狀態 |
|---------|----------|----------|------|
| 開發環境快速設置 | [`development/DEVELOPMENT_QUICK_START_GUIDE.md`](development/DEVELOPMENT_QUICK_START_GUIDE.md) | 🚀 環境初始化 | ✅ 完整 |
| 開發任務流程手冊 | [`development/DEVELOPMENT_TASKS_GUIDE.md`](development/DEVELOPMENT_TASKS_GUIDE.md) | ✅ 日常開發流程 | ✅ 完整 |
| 依賴管理操作手冊 | [`development/DEPENDENCY_MANAGEMENT_GUIDE.md`](development/DEPENDENCY_MANAGEMENT_GUIDE.md) | 📦 依賴管理策略 | ✅ 完整 |
| API 驗證操作手冊 | [`development/API_VERIFICATION_GUIDE.md`](development/API_VERIFICATION_GUIDE.md) | 🔐 密鑰驗證配置 | ✅ 完整 |
| AI 服務使用手冊 | [`development/AI_SERVICES_USER_GUIDE.md`](development/AI_SERVICES_USER_GUIDE.md) | 🤖 AI 功能使用 | ✅ 完整 |
| Schema 導入規範 | [`development/SCHEMA_IMPORT_GUIDE.md`](development/SCHEMA_IMPORT_GUIDE.md) | 📝 Schema 使用規範 | ✅ 必讀 |
| **Token 最佳化指南** | [`development/TOKEN_OPTIMIZATION_GUIDE.md`](development/TOKEN_OPTIMIZATION_GUIDE.md) | 🎯 開發效率最佳化 | ✅ 完整 |
| **統計收集系統** | [`development/METRICS_USAGE_GUIDE.md`](development/METRICS_USAGE_GUIDE.md) | 📊 系統監控與統計 | ✅ 完整 |
| **數據存儲指南** | [`development/DATA_STORAGE_GUIDE.md`](development/DATA_STORAGE_GUIDE.md) | 💾 數據存儲架構 | ✅ 完整 |
| **UI 啟動指南** | [`development/UI_LAUNCH_GUIDE.md`](development/UI_LAUNCH_GUIDE.md) | 🖥️ 界面管理 | ✅ 完整 |
| **擴充功能安裝** | [`development/EXTENSIONS_INSTALL_GUIDE.md`](development/EXTENSIONS_INSTALL_GUIDE.md) | 🔌 開發工具配置 | ✅ 完整 |
| **多語言環境標準** | [`development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md`](development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) | 🌐 Python/TS/Go/Rust 統一配置 | ✅ 完整 |
| **VS Code 配置最佳化** | [`development/VSCODE_CONFIGURATION_OPTIMIZATION.md`](development/VSCODE_CONFIGURATION_OPTIMIZATION.md) | ⚙️ IDE 性能優化 | ✅ 完整 |
| **語言伺服器優化** | [`development/LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md`](development/LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md) | ⚡ IDE 性能優化配置 | ✅ 完整 |
| **Git 推送規範** | [`development/GIT_PUSH_GUIDELINES.md`](development/GIT_PUSH_GUIDELINES.md) | 🔒 代碼安全推送 | ✅ 完整 |

### 🏗️ **架構設計指南** (`architecture/`)

| 指南類型 | 文檔路徑 | 技術重點 | 狀態 |
|---------|----------|----------|------|
| **v2.0 數據合約架構** | [`../README.md`](../README.md#架構總覽) | 📋 數據合約驅動設計 | ✅ 核心文檔 |
| **架構演進歷程** | [`../_archive/ARCHITECTURE_EVOLUTION_HISTORY.md`](../_archive/ARCHITECTURE_EVOLUTION_HISTORY.md) | 🔄 系統發展軌跡 | ✅ 完整 |
| **跨語言模組同步指南** | [`architecture/CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md`](architecture/CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md) | 🌐 多語言協同 | ✅ 完整 |
| **Schema 生成操作指南** | [`architecture/SCHEMA_GENERATION_GUIDE.md`](architecture/SCHEMA_GENERATION_GUIDE.md) | 🧬 數據結構標準化 | ✅ 完整 |
| **Schema 統一指南** | [`architecture/SCHEMA_GUIDE.md`](architecture/SCHEMA_GUIDE.md) | 📋 Schema 架構總覽 | ✅ 完整 |
| **Schema 合規規範** | [`architecture/SCHEMA_COMPLIANCE_GUIDE.md`](architecture/SCHEMA_COMPLIANCE_GUIDE.md) | ⚖️ 標準化開發規範 | ✅ 完整 |
| **跨語言 Schema 指南** | [`architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md`](architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md) | 🌐 完整跨語言實現 | ✅ 完整 |
| **跨語言兼容性指南** | [`architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md`](architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md) | 📊 多語言支援分析 | ✅ 完整 |

### ⚙️ **模組專業指南** (`modules/`) - 五大程式模組

#### 🎯 **Core 模組** - AI 引擎核心
| 指南類型 | 文檔路徑 | 內容重點 | 狀態 |
|---------|----------|----------|------|
| 開發規範手冊 | [`../services/core/docs/README_DEVELOPMENT.md`](../services/core/docs/README_DEVELOPMENT.md) | 🐍 Python 開發最佳實踐 | ✅ 完整 |
| AI 引擎操作手冊 | [`../services/core/docs/README_AI_ENGINE.md`](../services/core/docs/README_AI_ENGINE.md) | 🤖 AI 配置與優化 | ✅ 完整 |
| 執行引擎操作手冊 | [`../services/core/docs/README_EXECUTION.md`](../services/core/docs/README_EXECUTION.md) | ⚡ 性能優化策略 | ✅ 完整 |
| 學習系統操作手冊 | [`../services/core/docs/README_LEARNING.md`](../services/core/docs/README_LEARNING.md) | 🧠 ML 工程實踐 | ✅ 完整 |
| 測試策略手冊 | [`../services/core/docs/README_TESTING.md`](../services/core/docs/README_TESTING.md) | 🧪 測試框架使用 | ✅ 完整 |
| AI 引擎操作指南 | [`modules/AI_ENGINE_GUIDE.md`](modules/AI_ENGINE_GUIDE.md) | 🤖 AI 配置與優化 | ✅ 完整 |

#### ⚙️ **Features 模組** - 多語言業務功能
| 指南類型 | 文檔路徑 | 語言專精 | 狀態 |
|---------|----------|----------|------|
| Python 開發指南 | [`modules/PYTHON_DEVELOPMENT_GUIDE.md`](modules/PYTHON_DEVELOPMENT_GUIDE.md) | 🐍 核心業務邏輯 | ✅ 完整 |
| Go 開發指南 | [`modules/GO_DEVELOPMENT_GUIDE.md`](modules/GO_DEVELOPMENT_GUIDE.md) | 🐹 高效能服務 | ✅ 完整 |
| Rust 開發指南 | [`modules/RUST_DEVELOPMENT_GUIDE.md`](modules/RUST_DEVELOPMENT_GUIDE.md) | 🦀 安全分析引擎 | ✅ 完整 |
| 支援功能操作指南 | [`modules/SUPPORT_FUNCTIONS_GUIDE.md`](modules/SUPPORT_FUNCTIONS_GUIDE.md) | 🔧 運維工具集 | ✅ 完整 |

#### 🔍 **Scan 模組** - 掃描與偵測
- 參見 Features 模組中的掃描功能實現

#### 🔗 **Integration 模組** - 整合與協調
- 透過 [`../services/integration/README.md`](../services/integration/README.md) 查看完整整合操作手冊 ✅

#### 🏗️ **Common 模組** - 共用基礎設施
- 透過 [`../services/aiva_common/README.md`](../services/aiva_common/README.md) 查看標準化開發規範 ✅

#### 📋 **功能模組需求文件** (2025-11-06 完成)
> 五大程式模組中 Features 模組的詳細需求分析

| 報告編號 | 文檔路徑 | 涵蓋模組 | 狀態 |
|---------|----------|----------|------|
| 01 | [`../reports/features_modules/01_CRYPTO_POSTEX_急需實現報告.md`](../reports/features_modules/01_CRYPTO_POSTEX_急需實現報告.md) | 🚨 CRYPTO + POSTEX | ✅ 完整 |
| 02 | [`../reports/features_modules/02_SQLI_AUTHN_GO_架構完善報告.md`](../reports/features_modules/02_SQLI_AUTHN_GO_架構完善報告.md) | ⏳ SQLI + AUTHN_GO | ✅ 完整 |
| 03 | [`../reports/features_modules/03_架構重新定位_Go模組歸屬分析.md`](../reports/features_modules/03_架構重新定位_Go模組歸屬分析.md) | 🔄 GO模組分析 | ✅ 完整 |
| 04 | [`../reports/features_modules/04_GO模組遷移整合方案.md`](../reports/features_modules/04_GO模組遷移整合方案.md) | 🚀 GO模組遷移 | ✅ 完整 |
| 05 | [`../reports/features_modules/05_IDOR_SSRF_組件補強報告.md`](../reports/features_modules/05_IDOR_SSRF_組件補強報告.md) | 🔧 IDOR + SSRF | ✅ 完整 |
| 06 | [`../reports/features_modules/06_XSS_最佳實踐架構參考報告.md`](../reports/features_modules/06_XSS_最佳實踐架構參考報告.md) | 🌟 XSS架構範本 | ✅ 完整 |

#### 📋 **模組專用指南**
| 指南類型 | 文檔路徑 | 適用模組 | 狀態 |
|---------|----------|----------|------|
| **功能模組開發指南** | [`modules/FEATURE_MODULES_DEVELOPMENT_GUIDE.md`](modules/FEATURE_MODULES_DEVELOPMENT_GUIDE.md) | 🎯 Features 模組實作 | ✅ 完整 |
| 模組遷移操作指南 | [`modules/MODULE_MIGRATION_GUIDE.md`](modules/MODULE_MIGRATION_GUIDE.md) | 🔄 模組升級遷移 | ✅ 完整 |
| 分析功能架構指南 | [`modules/ANALYSIS_FUNCTIONS_GUIDE.md`](modules/ANALYSIS_FUNCTIONS_GUIDE.md) | 🔍 分析功能架構 | ✅ 完整 |

### 🚀 **部署運維指南** (`deployment/`)

| 指南類型 | 文檔路徑 | 部署重點 | 狀態 |
|---------|----------|----------|------|
| **系統安裝指南** | [`deployment/SYSTEM_INSTALLATION_GUIDE.md`](deployment/SYSTEM_INSTALLATION_GUIDE.md) | 🖥️ 完整系統環境安裝 | ✅ **全域環境** |
| **安裝指南** | [`deployment/INSTALLATION_GUIDE.md`](deployment/INSTALLATION_GUIDE.md) | 📦 Python 專案安裝 | ✅ **全域環境** |
| 構建流程操作指南 | [`deployment/BUILD_GUIDE.md`](deployment/BUILD_GUIDE.md) | 🔨 多語言構建自動化 | ✅ 完整 |
| Docker 容器化手冊 | [`deployment/DOCKER_GUIDE.md`](deployment/DOCKER_GUIDE.md) | 🐳 容器化部署實踐 | ✅ 完整 |
| Kubernetes 微服務部署 | [`deployment/DOCKER_KUBERNETES_GUIDE.md`](deployment/DOCKER_KUBERNETES_GUIDE.md) | ☸️ 微服務編排方案 | ✅ 完整 |
| 生產環境配置指南 | [`deployment/ENVIRONMENT_CONFIG_GUIDE.md`](deployment/ENVIRONMENT_CONFIG_GUIDE.md) | ⚙️ 生產配置管理 | ✅ 完整 |

> **📌 重要變更** (2025-11-25): 已移除虛擬環境,全面改用全域 Python 環境。詳見 [全域環境遷移報告](./GLOBAL_ENVIRONMENT_MIGRATION_2025-11-25.md)

### 🔧 **疑難排解指南** (`troubleshooting/`)

| 指南類型 | 文檔路徑 | 解決領域 | 狀態 |
|---------|----------|----------|----------|
| 開發環境故障排除 | [`troubleshooting/DEVELOPMENT_ENVIRONMENT_TROUBLESHOOTING.md`](troubleshooting/DEVELOPMENT_ENVIRONMENT_TROUBLESHOOTING.md) | 🚨 多語言環境診斷 | ✅ 完整 |
| Pydantic 模型修復指南 | [`troubleshooting/FORWARD_REFERENCE_REPAIR_GUIDE.md`](troubleshooting/FORWARD_REFERENCE_REPAIR_GUIDE.md) | 🔗 向前引用修復 | ✅ 完整 |
| 性能優化配置指南 | [`troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md`](troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md) | ⚡ 性能優化 | ✅ 完整 |
| 測試環境重現指南 | [`troubleshooting/TESTING_REPRODUCTION_GUIDE.md`](troubleshooting/TESTING_REPRODUCTION_GUIDE.md) | 🧪 測試環境重現 | ✅ 完整 |

### 🤖 **AI 與功能手冊**

| 指南類型 | 文檔路徑 | 功能重點 | 狀態 |
|---------|----------|----------|----------|
| **AI 雙閉環自我優化設計** | [`../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md) | 🔄 AI 雙閉環核心設計 | ✅ **核心文檔** |
| **術語對照表** | [`../TERMINOLOGY_GLOSSARY.md`](../TERMINOLOGY_GLOSSARY.md) | 📖 統一術語規範 | ✅ **必讀** |
| **22 個 AI 組件詳細說明** | [`../reports/ai_analysis/AIVA_22_AI_COMPONENTS_DETAILED_GUIDE.md`](../reports/ai_analysis/AIVA_22_AI_COMPONENTS_DETAILED_GUIDE.md) | 🤖 完整 AI 組件架構 | ✅ 完整 |
| AI 服務使用手冊 | [`development/AI_SERVICES_USER_GUIDE.md`](development/AI_SERVICES_USER_GUIDE.md) | 🧠 AI 功能使用實戰 | ✅ 完整 |
| API 驗證操作手冊 | [`development/API_VERIFICATION_GUIDE.md`](development/API_VERIFICATION_GUIDE.md) | 🔐 密鑰驗證功能 | ✅ 完整 |

### 🛠️ **工具與測試手冊**

| 指南類型 | 文檔路徑 | 工具類型 | 狀態 |
|---------|----------|----------|----------|
| 工具集使用手冊 | [`../tools/README.md`](../tools/README.md) | 🔧 專業工具操作 | ✅ 完整 |
| 測試框架手冊 | [`../testing/README.md`](../testing/README.md) | 🧪 測試策略與實踐 | ✅ 完整 |
| **實用工具位置** | `services/{module}/tools/` | 📍 工具位置參考 | ✅ 見上方表格 |

---

## 🎯 **使用建議與學習路徑**

### 🧠 **AI 核心設計理念** (必讀)
在開始任何其他學習路徑之前，請先了解 AIVA 的核心 AI 設計理念：

**📖 雙閉環自我優化設計**
- **內部閉環 (Know Thyself)**: 探索(對內自省) + RAG(知識增強) → 了解自身能力與缺口
- **外部閉環 (Learn from Battle)**: 掃描(對外偵測) + 攻擊(實戰反饋) → 持續進化優化

📚 **核心文檔閱讀順序**:
1. [`AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md) - 理解完整設計理念
2. [`TERMINOLOGY_GLOSSARY.md`](../TERMINOLOGY_GLOSSARY.md) - 掌握術語規範
3. [`README.md`](../README.md) - 了解 v2.0 架構總覽

⚠️ **重要提醒**:
- 「**探索 (Exploration)**」= AIVA 系統**自我診斷** (對內)
- 「**掃描 (Scan/Reconnaissance)**」= **目標系統偵測** (對外)
- 絕對不要混淆這兩個概念

---

### 📚 **新手入門路徑** (按順序學習)
1. **AIVA 系統架構文檔** ([`../README.md`](../README.md)) - 了解 v2.0 架構與六大核心服務
2. **雙閉環設計文檔** - 理解 AI 自我優化機制
3. **開發環境快速設置** - 掌握開發環境和標準流程
4. **依賴管理操作手冊** - 理解環境配置和包管理
5. **功能模組開發指南** - 掌握標準開發流程

### 🔧 **功能模組開發者路徑**
1. **功能模組開發指南** - 掌握標準架構 (Worker + Detector + Engine + Config)
2. **XSS最佳實踐參考報告** - 學習完整模組範本
3. 選擇對應優先級模組: CRYPTO/POSTEX (緊急) → SQLI/AUTHN_GO (補強) → IDOR/SSRF (標準)
4. 選擇對應語言的開發手冊 (Python/Go/Rust)
5. 學習 GO 模組遷移指南 (若需要)

### 💻 **傳統開發者專業路徑**
1. 選擇對應模組的專業指南 (Core/Features/Scan/Integration/Common)
2. 選擇對應語言的開發手冊 (Python/Go/Rust)
3. 學習 Schema 導入規範與代碼規範
4. 根據問題查閱疑難排解指南

### 🏭 **運維部署路徑**
1. **構建流程操作指南** - 了解構建與部署策略
2. **Docker 容器化手冊** - 容器化部署實踐
3. **Kubernetes 微服務部署** - 理解微服務架構
4. **疑難排解指南** - 解決運維問題

### 🤖 **AI 功能專家路徑**
1. **AI 自我優化雙重閉環設計** - 理解核心 AI 設計理念
2. **術語對照表** - 掌握統一術語規範
3. **AI 引擎操作手冊** - AI 系統配置
4. **22 個 AI 組件詳細說明** - 完整組件架構
5. **學習系統操作手冊** - ML 工程實踐
6. **AI 用戶操作手冊** - 實戰案例學習
7. **執行引擎操作手冊** - 性能優化

---

## 📝 **文檔維護原則**

### ✅ **命名規範**
- **技術手冊/操作手冊/使用手冊**: 實用性文檔，面向操作和使用
- **指南**: 僅限特殊情況使用，避免與報告混淆
- **報告**: 分析性文檔，面向總結和評估

### 🔄 **更新維護**
- 每個指南都應有明確的適用對象和完整度標示
- 定期檢查連結有效性和內容時效性
- 新增指南時更新本索引文件
- **最新檢查**: [Guides 一致性檢查報告](./GUIDES_CONSISTENCY_CHECK_REPORT.md) (2025-11-22)

### 📋 **品質保證**
- 每個指南都應包含目錄、適用場景、實際操作步驟
- 提供清晰的學習路徑和使用建議
- 保持與實際系統狀態同步
- **文檔符合度**: 99.8% ✅

---

## 📊 **相關報告**

### 📈 **重要更新與遷移報告**

| 報告名稱 | 路徑 | 檢查範圍 | 狀態 |
|---------|------|----------|------|
| **全域環境遷移報告** | [`./GLOBAL_ENVIRONMENT_MIGRATION_2025-11-25.md`](./GLOBAL_ENVIRONMENT_MIGRATION_2025-11-25.md) | 虛擬環境→全域環境 | ✅ **最新** 2025-11-25 |
| **Guides 一致性檢查** | [`./GUIDES_CONSISTENCY_CHECK_REPORT.md`](./GUIDES_CONSISTENCY_CHECK_REPORT.md) | guides 目錄全面檢查 | ✅ 2025-11-22 |
| **指南整合報告** | [`./GUIDES_CONSOLIDATION_REPORT.md`](./GUIDES_CONSOLIDATION_REPORT.md) | 指南文件整理完成 | ✅ 完成 |
| **指南更新報告** | [`./GUIDES_UPDATE_COMPLETE_2025-11-22.md`](./GUIDES_UPDATE_COMPLETE_2025-11-22.md) | 48 份指南全面更新 | ✅ 2025-11-22 |

---

**📋 文檔資訊**
- **維護者**: AIVA 核心團隊
- **創建日期**: 2025-10-31
- **最後更新**: 2025-11-25
- **架構版本**: v2.0 (數據合約驅動架構)
- **Python 環境**: 全域環境 (341 個套件)
- **模組架構**: 五大程式模組 + 六大核心服務
- **分類原則**: 按功能領域和使用對象分類
- **更新頻率**: 隨系統演進即時更新