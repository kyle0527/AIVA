# 📚 AIVA 指南中心

> **📋 指南分類**: 按功能和領域分類的完整指南集合  
> **🎯 使用目的**: 為不同角色提供專業的技術文檔和操作手冊  
> **📅 最後更新**: 2025-10-31  
> **✅ 組織狀態**: 完全分類整理，便於查找和維護

---

## 🏗️ 指南架構總覽

```
guides/
├── README.md                    # 📋 本索引文件
├── development/                 # 🛠️ 開發相關指南
├── architecture/                # 🏗️ 架構設計指南  
├── modules/                     # ⚙️ 模組專業指南
├── deployment/                  # 🚀 部署運維指南
└── troubleshooting/            # 🔧 疑難排解指南
```

---

## 📖 指南分類目錄

### 🏆 **核心綜合指南** (頂級參考)

| 文檔名稱 | 路徑 | 適用對象 | 完整度 |
|---------|------|----------|--------|
| **AIVA 綜合技術手冊** | [`../reports/documentation/AIVA_COMPREHENSIVE_GUIDE.md`](../reports/documentation/AIVA_COMPREHENSIVE_GUIDE.md) | 🎯 AI工程師、系統架構師、核心開發者 | ✅ 完整 |
| **開發者指南** | [`../reports/documentation/DEVELOPER_GUIDE.md`](../reports/documentation/DEVELOPER_GUIDE.md) | 🛠️ 開發者、貢獻者、新手入門 | ✅ 完整 |

### 🛠️ **開發相關指南** (`development/`)

| 指南類型 | 文檔路徑 | 專業領域 | 狀態 |
|---------|----------|----------|------|
| 開發環境快速設置 | [`development/DEVELOPMENT_QUICK_START_GUIDE.md`](development/DEVELOPMENT_QUICK_START_GUIDE.md) | 🚀 環境初始化 | ✅ 完整 (10/31實測驗證) |
| 開發任務流程手冊 | [`development/DEVELOPMENT_TASKS_GUIDE.md`](development/DEVELOPMENT_TASKS_GUIDE.md) | ✅ 日常開發流程 | ✅ 完整 (10/31實測驗證) |
| 依賴管理操作手冊 | [`development/DEPENDENCY_MANAGEMENT_GUIDE.md`](development/DEPENDENCY_MANAGEMENT_GUIDE.md) | 📦 深度依賴管理策略 + **ML依賴混合狀態** | ✅ 完整 (10/31實測驗證) |
| API 驗證操作手冊 | [`development/API_VERIFICATION_GUIDE.md`](development/API_VERIFICATION_GUIDE.md) | 🔐 密鑰驗證功能配置 | ✅ 完整 (10/31實測驗證) |
| AI 服務使用手冊 | [`development/AI_SERVICES_USER_GUIDE.md`](development/AI_SERVICES_USER_GUIDE.md) | 🤖 AI 功能使用實戰 | ✅ 完整 (10/31實測驗證) |
| Schema 導入規範 | [`development/SCHEMA_IMPORT_GUIDE.md`](development/SCHEMA_IMPORT_GUIDE.md) | 📝 Schema 使用規範 | ✅ 必讀 (10/31實測驗證) |
| **Token 最佳化指南** | [`development/TOKEN_OPTIMIZATION_GUIDE.md`](development/TOKEN_OPTIMIZATION_GUIDE.md) | 🎯 開發效率最佳化 | ✅ **新增** |
| **統計收集系統** | [`development/METRICS_USAGE_GUIDE.md`](development/METRICS_USAGE_GUIDE.md) | 📊 系統監控與統計 | ✅ **新增** |
| **數據存儲指南** | [`development/DATA_STORAGE_GUIDE.md`](development/DATA_STORAGE_GUIDE.md) | 💾 數據存儲架構 | ✅ **新增** |
| **UI 啟動指南** | [`development/UI_LAUNCH_GUIDE.md`](development/UI_LAUNCH_GUIDE.md) | 🖥️ 界面管理 | ✅ **新增** |
| **擴充功能安裝** | [`development/EXTENSIONS_INSTALL_GUIDE.md`](development/EXTENSIONS_INSTALL_GUIDE.md) | 🔌 開發工具配置 | ✅ **新增** |
| **多語言環境標準** | [`development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md`](development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) | 🌐 Python/TS/Go/Rust 統一配置 | ✅ **新增** (10/31驗證) |
| **VS Code 配置最佳化** | [`development/VSCODE_CONFIGURATION_OPTIMIZATION.md`](development/VSCODE_CONFIGURATION_OPTIMIZATION.md) | ⚙️ IDE 性能優化詳解 | ✅ **新增** (10/31驗證) |
| **語言轉換指南** | [`development/LANGUAGE_CONVERSION_GUIDE.md`](development/LANGUAGE_CONVERSION_GUIDE.md) | 🔄 跨語言代碼轉換完整指南 | ✅ **新增** (10/31驗證) |
| **語言伺服器優化** | [`development/LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md`](development/LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md) | ⚡ IDE 性能優化配置 | ✅ **新增** |
| **Git 推送規範** | [`development/GIT_PUSH_GUIDELINES.md`](development/GIT_PUSH_GUIDELINES.md) | 🔒 代碼安全推送 | ✅ **新增** |
| VS Code 插件完整清單 | [`../_out/VSCODE_EXTENSIONS_INVENTORY.md`](../_out/VSCODE_EXTENSIONS_INVENTORY.md) | 🛠️ 開發工具配置 (88個插件) | ✅ 完整 |

### 🏗️ **架構設計指南** (`architecture/`)

| 指南類型 | 文檔路徑 | 技術重點 | 狀態 |
|---------|----------|----------|------|
| 架構演進歷程 | [`../_archive/ARCHITECTURE_EVOLUTION_HISTORY.md`](../_archive/ARCHITECTURE_EVOLUTION_HISTORY.md) | 🔄 系統發展軌跡 | ✅ 完整 |
| 跨語言模組同步指南 | [`architecture/CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md`](architecture/CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md) | 🌐 多語言協同 | ✅ 完整 |
| Schema 生成操作指南 | [`architecture/SCHEMA_GENERATION_GUIDE.md`](architecture/SCHEMA_GENERATION_GUIDE.md) | 🧬 數據結構標準化 | ✅ 完整 |
| **Schema 統一指南** | [`architecture/SCHEMA_GUIDE.md`](architecture/SCHEMA_GUIDE.md) | 📋 Schema 架構總覽 | ✅ **新增** |
| **Schema 合規規範** | [`architecture/SCHEMA_COMPLIANCE_GUIDE.md`](architecture/SCHEMA_COMPLIANCE_GUIDE.md) | ⚖️ 標準化開發規範 | ✅ **新增** |
| **跨語言 Schema 指南** | [`architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md`](architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md) | 🌐 完整跨語言實現 | ✅ **新增** |
| 跨語言兼容性指南 | [`architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md`](architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md) | 📊 多語言支援分析 | ✅ 完整 |

### ⚙️ **模組專業指南** (`modules/`)

#### 🎯 **核心模組** (Core)
| 指南類型 | 文檔路徑 | 內容重點 | 狀態 |
|---------|----------|----------|------|
| 開發規範手冊 | [`../services/core/docs/README_DEVELOPMENT.md`](../services/core/docs/README_DEVELOPMENT.md) | 🐍 Python 開發最佳實踐 | ✅ 完整 |
| AI 引擎操作手冊 | [`../services/core/docs/README_AI_ENGINE.md`](../services/core/docs/README_AI_ENGINE.md) | 🤖 AI 配置與優化 | ✅ 完整 |
| 執行引擎操作手冊 | [`../services/core/docs/README_EXECUTION.md`](../services/core/docs/README_EXECUTION.md) | ⚡ 性能優化策略 | ✅ 完整 |
| 學習系統操作手冊 | [`../services/core/docs/README_LEARNING.md`](../services/core/docs/README_LEARNING.md) | 🧠 ML 工程實踐 | ✅ 完整 |
| 測試策略手冊 | [`../services/core/docs/README_TESTING.md`](../services/core/docs/README_TESTING.md) | 🧪 測試框架使用 | ✅ 完整 |
| AI 引擎操作指南 | [`modules/AI_ENGINE_GUIDE.md`](modules/AI_ENGINE_GUIDE.md) | 🤖 AI 配置與優化 | ✅ 完整 |

#### ⚙️ **功能模組** (Features)
| 指南類型 | 文檔路徑 | 語言專精 | 狀態 |
|---------|----------|----------|------|
| Python 開發指南 | [`modules/PYTHON_DEVELOPMENT_GUIDE.md`](modules/PYTHON_DEVELOPMENT_GUIDE.md) | 🐍 723 組件的核心業務邏輯 | ✅ 完整 |
| Go 開發指南 | [`modules/GO_DEVELOPMENT_GUIDE.md`](modules/GO_DEVELOPMENT_GUIDE.md) | 🐹 165 組件的高效能服務 | ✅ 完整 |
| Rust 開發指南 | [`modules/RUST_DEVELOPMENT_GUIDE.md`](modules/RUST_DEVELOPMENT_GUIDE.md) | 🦀 1,804 組件的安全分析 | ✅ 完整 |
| 支援功能操作指南 | [`modules/SUPPORT_FUNCTIONS_GUIDE.md`](modules/SUPPORT_FUNCTIONS_GUIDE.md) | 🔧 運維工具集 | ✅ 完整 |

#### 🔗 **整合模組** (Integration)
- 透過 [`../services/integration/README.md`](../services/integration/README.md) 查看完整整合操作手冊 ✅

#### 🏗️ **共用模組** (AIVA Common)
- 透過 [`../services/aiva_common/README.md`](../services/aiva_common/README.md) 查看標準化開發規範 ✅

#### 📋 **模組專用指南**
| 指南類型 | 文檔路徑 | 適用模組 | 狀態 |
|---------|----------|----------|------|
| 模組遷移操作指南 | [`modules/MODULE_MIGRATION_GUIDE.md`](modules/MODULE_MIGRATION_GUIDE.md) | 🔄 Features 模組升級 | ✅ 完整 |
| 分析功能架構指南 | [`modules/ANALYSIS_FUNCTIONS_GUIDE.md`](modules/ANALYSIS_FUNCTIONS_GUIDE.md) | 🔍 分析功能架構 | ✅ 完整 |

### 🚀 **部署運維指南** (`deployment/`)

| 指南類型 | 文檔路徑 | 部署重點 | 狀態 |
|---------|----------|----------|------|
| 構建流程操作指南 | [`deployment/BUILD_GUIDE.md`](deployment/BUILD_GUIDE.md) | 🔨 多語言構建自動化 | ✅ 完整 |
| Docker 基礎設施指南 | [`deployment/DOCKER_GUIDE.md`](deployment/DOCKER_GUIDE.md) | 🐳 容器化部署實踐 | ✅ 完整 (10/31實測驗證) |
| Docker & Kubernetes 部署 | [`deployment/DOCKER_KUBERNETES_GUIDE.md`](deployment/DOCKER_KUBERNETES_GUIDE.md) | ☸️ 微服務編排方案 | ✅ 完整 |
| 環境變數配置指南 | [`deployment/ENVIRONMENT_CONFIG_GUIDE.md`](deployment/ENVIRONMENT_CONFIG_GUIDE.md) | ⚙️ 環境配置管理 | ✅ 完整 |

| 指南類型 | 文檔路徑 | 部署場景 |
|---------|----------|----------|
| 部署操作手冊 | [`../docs/README_DEPLOYMENT.md`](../docs/README_DEPLOYMENT.md) | 🏭 生產環境部署 |
| Docker 容器化手冊 | [`deployment/DOCKER_GUIDE.md`](deployment/DOCKER_GUIDE.md) | 🐳 容器化部署策略 |
| 構建流程手冊 | [`deployment/BUILD_GUIDE.md`](deployment/BUILD_GUIDE.md) | 🔨 構建和打包流程 |
| **環境配置指南** | [`deployment/ENVIRONMENT_CONFIG_GUIDE.md`](deployment/ENVIRONMENT_CONFIG_GUIDE.md) | ⚙️ 環境變數配置 | ✅ **新增** |
| **微服務部署指南** | [`deployment/DOCKER_KUBERNETES_GUIDE.md`](deployment/DOCKER_KUBERNETES_GUIDE.md) | ☸️ 微服務完整部署 | ✅ **新增** |

### 🔧 **疑難排解指南** (`troubleshooting/`)

| 指南類型 | 文檔路徑 | 解決領域 | 驗證狀態 |
|---------|----------|----------|----------|
| **開發環境配置故障排除** | [`troubleshooting/DEVELOPMENT_ENVIRONMENT_TROUBLESHOOTING.md`](troubleshooting/DEVELOPMENT_ENVIRONMENT_TROUBLESHOOTING.md) | 🚨 多語言環境快速診斷 | ✅ **新增** (10/31驗證) |
| 向前引用修復指南 | [`troubleshooting/FORWARD_REFERENCE_REPAIR_GUIDE.md`](troubleshooting/FORWARD_REFERENCE_REPAIR_GUIDE.md) | 🔗 Pydantic 模型修復 | |
| 性能優化配置指南 | [`troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md`](troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md) | ⚡ 性能優化配置 | |
| **測試重現指南** | [`troubleshooting/TESTING_REPRODUCTION_GUIDE.md`](troubleshooting/TESTING_REPRODUCTION_GUIDE.md) | 🧪 測試環境快速重現 | ✅ **新增** |

### 🤖 **AI 與功能手冊**

| 指南類型 | 文檔路徑 | 功能重點 | 驗證狀態 |
|---------|----------|----------|----------|
| **22 個 AI 組件詳細說明** | [`../reports/ai_analysis/AIVA_22_AI_COMPONENTS_DETAILED_GUIDE.md`](../reports/ai_analysis/AIVA_22_AI_COMPONENTS_DETAILED_GUIDE.md) | 🤖 完整 AI 組件架構說明 | |
| AI 用戶操作手冊 | [`../AI_USER_GUIDE.md`](../AI_USER_GUIDE.md) | 🧠 AI 功能實戰案例 | |
| API 驗證操作手冊 | [`../API_VERIFICATION_GUIDE.md`](../API_VERIFICATION_GUIDE.md) | 🔐 密鑰驗證功能 | (10/31驗證) |

### 🛠️ **工具與測試手冊**

| 指南類型 | 文檔路徑 | 工具類型 | 驗證狀態 |
|---------|----------|----------|----------|
| 工具集使用手冊 | [`../tools/README.md`](../tools/README.md) | 🔧 專業工具操作 | |
| 測試框架手冊 | [`../testing/README.md`](../testing/README.md) | 🧪 測試策略與實踐 | |

---

## 🎯 **使用建議與學習路徑**

### 📚 **新手入門路徑** (按順序學習)
1. **AIVA 綜合技術手冊** - 理解整體架構和核心概念
2. **開發者指南** - 掌握開發環境和標準流程
3. **VS Code 插件清單** - 配置完整開發工具鏈
4. **依賴管理操作手冊** - 理解環境配置和包管理

### 🔧 **開發者專業路徑**
1. 選擇對應模組的專業指南 (Core/Scan/Features/Integration)
2. 選擇對應語言的開發手冊 (Python/Go/Rust/TypeScript)
3. 學習代碼規範與使用標準 (IMPORT_GUIDELINES.md)
4. 根據問題查閱疑難排解指南

### 🏭 **運維部署路徑**
1. **部署操作手冊** - 了解部署策略
2. **Docker 容器化手冊** - 容器化部署實踐
3. **架構設計指南** - 理解系統架構
4. **疑難排解指南** - 解決運維問題

### 🤖 **AI 功能專家路徑**
1. **AI 引擎操作手冊** - AI 系統配置
2. **學習系統操作手冊** - ML 工程實踐
3. **AI 用戶操作手冊** - 實戰案例學習
4. **執行引擎操作手冊** - 性能優化

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

### 📋 **品質保證**
- 每個指南都應包含目錄、適用場景、實際操作步驟
- 提供清晰的學習路徑和使用建議
- 保持與實際系統狀態同步

---

**📝 文檔資訊**
- **維護者**: AIVA 核心團隊
- **創建日期**: 2025-10-31
- **分類原則**: 按功能領域和使用對象分類
- **更新頻率**: 隨系統演進即時更新