# 📁 AIVA 項目根目錄結構完整說明

**生成日期**: 2025-11-27  
**參考來源**: VS Code 資源管理器截圖  
**目的**: 為所有開發者提供完整的目錄結構理解

---

## 📊 架構核心理解

> **⚠️ 關鍵認知**  
> **services/** 目錄包含 **93.8%** 的程式本體 (165,443 行, 557 檔案)  
> 其他所有目錄合計只佔 **6.2%** (約 11,085 行)  
> 詳見: [_SERVICES_IS_THE_REAL_CORE.md](./_SERVICES_IS_THE_REAL_CORE.md)

---

## 📑 目錄

- [🏆 核心程式目錄 (93.8%)](#核心程式目錄-938)
- [🌐 對外介面層 (1.4%)](#對外介面層-14)
- [🛠️ 開發工具層 (2.8%)](#開發工具層-28)
- [📊 運行時支援層 (0.6%)](#運行時支援層-06)
- [🧠 AI 實作層 (1.4%)](#ai-實作層-14)
- [📚 文檔與指南](#文檔與指南)
- [⚙️ 配置與環境](#配置與環境)
- [📦 編譯產物與快取](#編譯產物與快取)
- [🗄️ 資料與模型](#資料與模型)
- [🔧 工具與腳本](#工具與腳本)
- [📄 根目錄檔案](#根目錄檔案)

---

## 🏆 核心程式目錄 (93.8%)

### 📂 **services/** 🎯 **程式本體**

**代碼量**: 165,443 行 (93.8%)  
**檔案數**: 557 個 Python 檔案  
**地位**: AIVA 的真正核心，包含所有主要業務邏輯

#### 子模組結構

```
services/
├── aiva_common/          # 共享基礎設施 (~20,000 行)
│   ├── schemas/          # 數據模型定義 (Pydantic)
│   ├── enums/            # 枚舉類型 (Severity, Confidence 等)
│   ├── config/           # 統一配置系統
│   └── command_center.py # v2.0 命令路由中心
│
├── core/                 # AI 決策引擎 (~30,000 行)
│   ├── aiva_core/
│   │   ├── cognitive_core/      # 認知核心 (AI 大腦)
│   │   ├── core_capabilities/   # 核心能力 (攻擊執行、對話)
│   │   ├── internal_exploration/# 內部探索 (自省)
│   │   ├── external_learning/   # 外部學習 (攻擊反饋)
│   │   └── task_planning/       # 任務規劃 (AST 驅動)
│   └── README.md
│
├── features/             # 19+ 安全測試模組 (~80,000 行)
│   ├── function_sqli/           # SQL 注入檢測
│   ├── function_xss/            # XSS 跨站腳本
│   ├── function_ssrf/           # SSRF 伺服器端請求偽造
│   ├── function_idor/           # IDOR 不安全直接物件參考
│   ├── function_mass_assignment/# 大量賦值漏洞
│   ├── function_jwt_confusion/  # JWT 混淆攻擊
│   ├── function_oauth_confusion/# OAuth 配置錯誤
│   ├── function_graphql_authz/  # GraphQL 權限檢測
│   ├── function_ssrf_oob/       # SSRF OOB 檢測
│   └── (還有 10+ 個高價值模組...)
│
├── scan/                 # 掃描協調器 (~20,000 行)
│   ├── unified_scan_engine.py   # 統一掃描引擎
│   ├── command_handler.py       # 掃描命令處理器
│   └── scan_coordinator.py      # 掃描協調邏輯
│
└── integration/          # 統一資料管理 (~15,000 行)
    ├── unified_data_manager.py  # 統一資料管理器
    ├── experience_repository.py # 經驗記錄資料庫
    ├── finding_repository.py    # 漏洞發現資料庫
    └── attack_path_engine.py    # 攻擊路徑圖引擎
```

**商業價值**: 
- 19+ 高價值安全測試模組
- 單次成功漏洞發現潛在價值: $10.5K-$41K+
- Bug Bounty 特化設計

**技術特色**:
- ✅ v2.1.1 能力元數據驅動架構 (CapabilityRegistry + PostgreSQL + ChromaDB)
- ✅ 5M 參數神經網路 (100% 離線運行)
- ✅ RAG 增強決策
- ✅ 雙重閉環自我優化

---

## 🌐 對外介面層 (1.4%)

### 📂 **api/** 🌐 **REST API 包裝器**

**代碼量**: ~1,500 行 (0.8%)  
**檔案數**: 9 個 Python 檔案  
**地位**: services/features/ 的 REST API 包裝層

#### 結構

```
api/
├── main.py              # FastAPI 主應用 (625 行)
├── start_api.py         # API 啟動腳本
├── requirements.txt     # Python 依賴清單
├── README.md            # API 文檔 (已添加定位說明)
└── routers/             # 路由模組
    ├── auth.py          # JWT 認證端點
    ├── security.py      # 高價值掃描端點
    └── admin.py         # 系統管理端點
```

**功能**:
- 🔐 JWT 認證系統 (3 種用戶角色)
- 📡 5 個高價值功能模組 API
- 📊 掃描管理和系統監控
- 📖 Swagger UI 自動文檔

**用途**:
- CI/CD 整合安全檢測
- Bug Bounty 批量掃描
- 企業內部安全評估平台

**啟動**:
```bash
python api/start_api.py
# 訪問: http://localhost:8000/docs
```

---

### 📂 **web/** 🌐 **前端介面**

**代碼量**: ~1,000 行 JS/TS (0.6%)  
**技術棧**: Vue.js / React  
**地位**: 用戶界面層

**功能**:
- 🖥️ Web 控制台介面
- 📊 掃描結果可視化
- ⚙️ 配置管理界面
- 📈 統計報表展示

---

## 🛠️ 開發工具層 (2.8%)

### 📂 **plugins/** 🔌 **代碼生成工具**

**代碼量**: ~5,000 行 (2.8%)  
**地位**: 開發時工具，不是運行時依賴

#### 結構

```
plugins/
├── aiva_converters/          # 多語言轉換器插件包 (v1.1.0)
│   ├── converters/           # 格式轉換器
│   │   ├── sarif_converter.py        # SARIF 安全報告轉換
│   │   ├── task_converter.py         # AST 任務序列轉換
│   │   └── docx_to_md_converter.py   # Word → Markdown
│   │
│   ├── core/                 # 代碼生成引擎
│   │   ├── schema_codegen_tool.py    # 多語言 Schema 生成器 (1585行)
│   │   ├── typescript_generator.py   # TypeScript 專用生成器
│   │   └── cross_language_validator.py # 跨語言一致性驗證
│   │
│   ├── templates/            # Jinja2 代碼模板
│   │   ├── typescript/       # TS 介面模板
│   │   ├── rust/             # Rust 結構體模板
│   │   ├── go/               # Go 語言模板
│   │   └── python/           # Python 模板
│   │
│   ├── scripts/              # 自動化生成腳本
│   │   ├── generate-contracts.ps1
│   │   └── generate-official-contracts.ps1
│   │
│   └── tests/                # 測試框架
│
├── test_imports.py           # 導入測試 (6/6 通過)
└── README.md                 # 插件文檔
```

**主要功能**:
1. **格式轉換**: SARIF, AST, DOCX → MD
2. **代碼生成**: Python → TypeScript/Rust/Go
3. **Schema 同步**: 跨語言一致性保證

**使用場景**:
- 🔄 生成跨語言數據合約
- 📄 轉換安全報告格式
- 🛠️ 輔助多語言開發

---

### 📂 **utilities/** 🔧 **工具集 (規劃中)**

**代碼量**: 0 行 (0%)  
**狀態**: 目錄存在但無實際代碼  
**用途**: 規劃中的輔助工具集

---

## 📊 運行時支援層 (0.6%)

### 📂 **observability/** 📊 **監控框架**

**代碼量**: 538 行 (0.3%)  
**技術**: Prometheus + Grafana  
**地位**: 運行時性能監控

#### 功能

```
observability/
├── prometheus.yml       # Prometheus 配置
├── metrics/             # 指標收集器
│   ├── system_metrics.py
│   └── app_metrics.py
└── dashboards/          # Grafana 儀表板
    └── aiva_dashboard.json
```

**監控項目**:
- 📈 系統健康度 (95%)
- ⏱️ 響應時間 (<50ms)
- 🔢 API 調用統計
- 💾 資源使用情況

---

### 📂 **security/** 🔒 **安全框架**

**代碼量**: 547 行 (0.3%)  
**地位**: RBAC 安全控制框架

#### 功能

```
security/
├── rbac/                # 角色基礎訪問控制
│   ├── roles.py
│   └── permissions.py
├── auth/                # 認證機制
│   └── jwt_handler.py
└── encryption/          # 加密工具
    └── crypto_utils.py
```

**安全特性**:
- 🔐 JWT 令牌認證
- 👥 多角色權限控制
- 🔒 敏感數據加密
- 🛡️ API 訪問控制

---

## 🧠 AI 實作層 (1.4%)

### 📂 **src/** 🧠 **AI 引擎實作細節**

**代碼量**: ~2,500 行 (1.4%)  
**地位**: services/core/ 的底層實作

#### 結構

```
src/
├── core/                # AI 核心實作
│   ├── neural_network/  # 5M 參數神經網路實作
│   ├── rag_engine/      # RAG 檢索增強生成
│   └── decision_logic/  # 決策邏輯實作
├── models/              # 模型定義
└── utils/               # 工具函數
```

**關係**:
- src/core/ 被 services/core/ 使用
- 包含 AI 引擎的底層實作細節
- 神經網路訓練和推理邏輯

---

## 📚 文檔與指南

### 📂 **docs/** 📖 **技術文檔**

**內容**: 26 個文檔檔案  
**分類**: 
- 🏗️ 架構設計文檔
- 📘 API 參考文檔
- 🧑‍💻 開發者指南
- 🔧 部署說明

**主要文檔**:
- `INSTALLATION_GUIDE.md` - 安裝指南
- `DEPLOYMENT.md` - 部署指南
- `API_REFERENCE.md` - API 參考

---

### 📂 **guides/** 📚 **使用指南**

**內容**: 15 個指南文檔  
**類型**:
- 👤 用戶手冊
- 🎯 快速入門
- 💡 最佳實踐
- 🔍 故障排除

---

### 📂 **examples/** 💡 **示例代碼**

**內容**: 實際使用範例  
**包含**:
- Python 示例
- API 調用示例
- 配置範例
- 整合範例

---

### 📂 **reports/** 📊 **報告目錄**

**內容**: 263+ 個報告文檔  
**最近整理**: 已分類為 3 個子目錄

#### 結構 (已優化)

```
reports/
├── architecture/        # 架構分析報告
│   ├── _ARCHITECTURAL_MISCONCEPTIONS_AUDIT.md (新增)
│   ├── ARCHITECTURE_OPTIMIZATION_REPORT.md
│   ├── ARCHITECTURE_SUMMARY.md
│   └── (21+ 個架構文檔)
│
├── analysis/            # 代碼/文檔分析
│   ├── _CODE_FILES_DISTRIBUTION_ANALYSIS.md (新增)
│   ├── _DOCS_DIRECTORY_ANALYSIS.md (新增)
│   ├── SERVICES_STRUCTURE_ANALYSIS_REPORT.md (新增)
│   └── (6+ 個分析報告)
│
└── maintenance/         # 整理/維護記錄
    ├── _ARCHIVE_CONSOLIDATION_COMPLETION_REPORT.md (新增)
    ├── _CACHE_AND_LOGS_CLEANUP_COMPLETION_REPORT.md (新增)
    └── (14+ 個整理報告)
```

**用途**:
- 📈 追蹤項目演進
- 🔍 記錄技術決策
- 📝 保存整理歷史

---

## ⚙️ 配置與環境

### 📂 **config/** ⚙️ **配置目錄**

**內容**: 12 個配置檔案  
**類型**:
- 🐳 Docker 配置
- ☸️ Kubernetes 配置
- 🔧 應用配置
- 🌍 環境變數範本

**主要檔案**:
```
config/
├── docker-compose.yml   # 本地開發環境
├── k8s/                 # Kubernetes 配置
├── .env.example         # 環境變數範本
└── settings.yaml        # 應用配置
```

---

### 📂 **docker/** 🐳 **Docker 配置**

**內容**: 31 個 Docker 相關檔案  
**包含**:
- `Dockerfile.core` - 核心服務容器
- `Dockerfile.component` - 功能組件容器
- `docker-compose.yml` - 完整開發環境
- 健康檢查腳本

**特色**:
- 🏗️ 分層架構設計
- 🔄 Profile 機制 (按需啟動)
- 💚 健康檢查確保穩定性

---

### 📂 **.vscode/** 💻 **VS Code 配置**

**內容**: 編輯器配置  
**包含**:
- `settings.json` - 工作區設定
- `launch.json` - 調試配置
- `tasks.json` - 任務配置
- `extensions.json` - 推薦擴展

---

### 📄 **環境配置檔案**

```
.env                     # 本地環境變數
.env.docker              # Docker 環境變數
.env.example             # 環境變數範本
.env.local               # 本地覆蓋配置
.gitignore               # Git 忽略規則
.dockerignore            # Docker 忽略規則
.editorconfig            # 編輯器配置
.pylintrc                # Python Linter 配置
```

---

## 📦 編譯產物與快取

### 📂 **target/** 🎯 **Rust 編譯產物**

**內容**: 366 個編譯檔案  
**大小**: 數百 MB  
**狀態**: 應該被 .gitignore

**說明**:
- Rust 項目編譯輸出
- debug/ 和 release/ 目錄
- 可以安全刪除並重新編譯

**建議**: 添加到 .gitignore
```gitignore
/target/
```

---

### 📂 **_out/** 📤 **輸出目錄**

**用途**: 臨時輸出檔案  
**內容**: 生成的報告、日誌等  
**狀態**: 應該被忽略

---

### 📂 **cli_generated/** 🤖 **CLI 生成檔案**

**用途**: 命令行工具生成的檔案  
**內容**: 自動生成的代碼或配置  
**來源**: plugins/aiva_converters/

---

### 📂 **_archive/** 🗄️ **歸檔目錄**

**內容**: 已整理的歷史檔案  
**結構**: 5 個分類子目錄
```
_archive/
├── deprecated/          # 已棄用代碼
├── old_reports/         # 舊報告
├── backup/              # 備份檔案
├── experiments/         # 實驗性代碼
└── legacy/              # 遺留系統
```

**狀態**: ✅ 已完成整理 (2025-11-27)

---

## 🗄️ 資料與模型

### 📂 **data/** 📊 **資料目錄**

**內容**: 21 個資料檔案  
**類型**:
- 🗄️ 資料庫檔案 (SQLite)
- 📋 測試資料集
- 📊 統計資料
- 🧪 範例資料

**主要檔案**:
```
data/
├── experience.db        # 經驗記錄資料庫
├── findings.db          # 漏洞發現資料庫
├── attack_graph.json    # 攻擊路徑圖
└── test_data/           # 測試資料集
```

---

### 📂 **models/** 🧠 **模型目錄**

**內容**: 神經網路模型檔案  
**包含**:
- 🧠 5M 參數神經網路模型
- 📚 RAG 向量資料庫
- 🎯 訓練好的權重
- 📊 模型配置

---

### 📂 **weights/** ⚖️ **模型權重**

**內容**: 神經網路權重檔案  
**格式**: PyTorch .pth / TensorFlow .h5  
**大小**: 數十 MB 到數百 MB

---

### 📂 **analysis_results/** 📈 **分析結果**

**用途**: 保存分析輸出  
**內容**:
- 掃描結果
- 統計分析
- 性能報告
- 漏洞發現記錄

---

## 🔧 工具與腳本

### 📂 **scripts/** 🔨 **自動化腳本**

**內容**: 各種自動化腳本  
**分類**:
```
scripts/
├── setup/               # 環境設定腳本
├── deployment/          # 部署腳本
├── maintenance/         # 維護腳本
├── testing/             # 測試腳本
└── utils/               # 工具腳本
```

**常用腳本**:
- `setup_environment.ps1` - 環境初始化
- `run_tests.ps1` - 運行測試
- `deploy.sh` - 部署到生產環境

---

### 📂 **tools/** 🛠️ **開發工具**

**內容**: 開發輔助工具  
**包含**:
- 🔍 代碼分析工具
- 🧹 代碼清理工具
- 📊 統計工具
- 🔧 配置生成器

---

### 📂 **testing/** 🧪 **測試目錄**

**內容**: 測試相關檔案  
**結構**:
```
testing/
├── unit/                # 單元測試
├── integration/         # 整合測試
├── e2e/                 # 端到端測試
├── fixtures/            # 測試固定資料
└── mocks/               # 模擬對象
```

---

### 📂 **logs/** 📋 **日誌目錄**

**內容**: 6 個重要日誌檔案 (已清理)  
**原始**: 255 個檔案  
**清理後**: 6 個有用日誌

**保留的日誌**:
- 錯誤日誌 (errors.log)
- 系統日誌 (system.log)
- API 日誌 (api.log)
- 掃描日誌 (scan.log)
- AI 決策日誌 (ai_decisions.log)
- 性能日誌 (performance.log)

---

## 📄 根目錄檔案

### 🏠 **核心文檔 (保留的 4 個 MD)**

#### 1. **README.md** 📖
- **大小**: 13.14 KB
- **用途**: 項目主要說明文檔
- **狀態**: ✅ 已修正架構錯誤
- **內容**:
  - 項目介紹
  - 核心特色
  - 快速開始
  - 技術指標
  - 系統架構

#### 2. **_SERVICES_IS_THE_REAL_CORE.md** 🏆
- **大小**: 17.28 KB
- **用途**: 架構真相說明
- **重要性**: ⭐⭐⭐⭐⭐
- **內容**:
  - services/ 佔 93.8% 的統計證明
  - 架構誤解 vs 真相對比
  - 各目錄正確定位
  - 商業價值分析

#### 3. **_CORE_DIRECTORIES_FUNCTIONALITY_GUIDE.md** 📚
- **大小**: 29.98 KB
- **用途**: 輔助系統功能指南
- **狀態**: ✅ 已修正為"輔助系統"
- **內容**:
  - api/ 功能說明
  - plugins/ 工具說明
  - observability/ 監控說明
  - security/ 安全框架說明
  - src/, utilities/, web/ 說明

#### 4. **_MD_FILES_REORGANIZATION_PLAN.md** 📋
- **大小**: 9.01 KB
- **用途**: MD 檔案重組計畫
- **記錄**: 24 → 4 個的整理過程

---

### ⚙️ **配置檔案**

```
Cargo.toml               # Rust 項目配置
pyproject.toml           # Python 項目配置 (PEP 518)
requirements.txt         # Python 依賴清單
package.json             # Node.js 依賴 (如有前端)
```

---

### 🗄️ **資料庫檔案**

```
capability_registry.db   # 能力註冊資料庫
experience.db            # 經驗記錄資料庫 (可能)
```

---

### 📜 **腳本檔案**

```
啟動AI服務.bat            # Windows 快速啟動腳本
DELETE_OPTIONS.ps1       # 清理選項腳本
```

---

### 🔧 **工具配置**

```
AIVA.code-workspace      # VS Code 工作區配置
schema_codegen.log       # Schema 生成日誌
```

---

### 📝 **Git 相關**

```
.github/                 # GitHub Actions 配置
.gitignore               # Git 忽略規則
.gitattributes           # Git 屬性配置
```

---

## 📊 統計總結

### 代碼分布

| 目錄 | 代碼量 | 百分比 | 地位 |
|------|--------|--------|------|
| **services/** | 165,443 行 | 93.8% | 🏆 程式本體 |
| plugins/ | ~5,000 行 | 2.8% | 🔌 開發工具 |
| src/ | ~2,500 行 | 1.4% | 🧠 AI 實作 |
| api/ | ~1,500 行 | 0.8% | 🌐 API 包裝 |
| web/ | ~1,000 行 | 0.6% | 🌐 前端介面 |
| observability/ | 538 行 | 0.3% | 📊 監控 |
| security/ | 547 行 | 0.3% | 🔒 安全 |
| utilities/ | 0 行 | 0% | 🔧 規劃中 |
| **總計** | **~176,528 行** | **100%** | |

---

### 檔案數量統計

| 類型 | 數量 | 說明 |
|------|------|------|
| Python 檔案 | 557+ | 主要在 services/ |
| Markdown 文檔 | 300+ | 分散在各目錄 |
| 配置檔案 | 50+ | 各種配置 |
| 測試檔案 | 100+ | 單元/整合測試 |
| 腳本檔案 | 30+ | 自動化腳本 |

---

## 🎯 關鍵理解

### 1️⃣ **程式本體**: services/ (93.8%)

所有核心業務邏輯都在這裡:
- ✅ AI 決策引擎
- ✅ 19+ 安全測試模組
- ✅ 掃描協調器
- ✅ 數據管理

### 2️⃣ **輔助系統**: 其他目錄 (6.2%)

支援核心運行的基礎設施:
- 🌐 API/Web 對外介面
- 🔌 開發工具 (代碼生成)
- 📊 運行時支援 (監控/安全)
- 🧠 AI 底層實作

### 3️⃣ **開發優先級**

**核心開發**: 專注 services/ (93.8%)  
**工具開發**: plugins/, scripts/  
**介面開發**: api/, web/  
**運維開發**: observability/, docker/

---

## 💡 最佳實踐

### 新人上手路徑

1. 📖 閱讀 `README.md` - 了解項目概況
2. 🏆 閱讀 `_SERVICES_IS_THE_REAL_CORE.md` - 理解架構真相
3. 📚 閱讀 `_CORE_DIRECTORIES_FUNCTIONALITY_GUIDE.md` - 了解輔助系統
4. 🔍 探索 `services/` - 深入核心業務邏輯
5. 📊 查看 `reports/` - 了解技術決策和演進

### 開發注意事項

- ✅ 核心功能開發 → `services/`
- ✅ API 端點添加 → `api/routers/`
- ✅ 前端界面開發 → `web/`
- ✅ 工具腳本編寫 → `scripts/` 或 `tools/`
- ✅ 新文檔編寫 → 對應的 `reports/` 子目錄

### 資料夾使用規則

| 用途 | 存放位置 |
|------|---------|
| 核心業務代碼 | `services/` |
| API 端點 | `api/` |
| 前端代碼 | `web/` |
| 開發工具 | `plugins/`, `tools/` |
| 自動化腳本 | `scripts/` |
| 測試代碼 | `testing/` |
| 配置檔案 | `config/` |
| 文檔 | `docs/`, `guides/` |
| 架構報告 | `reports/architecture/` |
| 分析報告 | `reports/analysis/` |
| 整理記錄 | `reports/maintenance/` |
| 臨時輸出 | `_out/`, `analysis_results/` |
| 歷史歸檔 | `_archive/` |

---

## 📞 相關文檔

- [_SERVICES_IS_THE_REAL_CORE.md](./_SERVICES_IS_THE_REAL_CORE.md) - 架構真相
- [_CORE_DIRECTORIES_FUNCTIONALITY_GUIDE.md](./_CORE_DIRECTORIES_FUNCTIONALITY_GUIDE.md) - 輔助系統詳解
- [_ARCHITECTURAL_MISCONCEPTIONS_AUDIT.md](./reports/architecture/_ARCHITECTURAL_MISCONCEPTIONS_AUDIT.md) - 架構錯誤審計
- [README.md](./README.md) - 項目主文檔

---

**最後更新**: 2025-11-27  
**維護者**: AIVA 開發團隊  
**版本**: v2.1.1
