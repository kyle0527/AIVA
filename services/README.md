# 🏗️ AIVA Services - 企業級 Bug Bounty 平台服務架構

> **🎯 Bug Bounty 專業化 v6.2**: 五大核心服務協同，AI 驅動智能游透測試平台  
> **✅ 系統狀態**: 100% Bug Bounty 就緒，RAG1徹底移除完成，Core模組架構修復完成，跨語言 gRPC 整合完成，整合模組資料儲存標準化完成  
> **🔄 最後更新**: 2025年11月16日

---

## 📑 目錄

- [📋 概述](#-概述)
  - [🎯 核心特性](#-核心特性)
- [🏗️ 服務架構總覽](#️-服務架構總覽)
- [📦 五大核心服務](#-五大核心服務)
  - [🤖 Core - AI 驅動核心引擎](#-core---ai-驅動核心引擎)
  - [🔗 Common - Bug Bounty 共享庫](#-common---bug-bounty-共享庫)
  - [🎯 Features - 多語言安全功能](#-features---多語言安全功能)
  - [🔄 Integration - 企業級整合中樞](#-integration---企業級整合中樞)
  - [🔍 Scan - 多語言統一掃描引擎](#-scan---多語言統一掃描引擎)
- [📊 整體統計](#-整體統計)
  - [代碼規模](#代碼規模)
  - [技術棧分布](#技術棧分布)
  - [數據模型](#數據模型)
- [🚀 快速開始](#-快速開始)
  - [環境要求](#環境要求)
  - [安裝流程](#安裝流程)
- [🏗️ 架構設計原則](#️-架構設計原則)
- [🔗 服務間協作流程](#-服務間協作流程)
- [📚 文檔導航](#-文檔導航)
- [🛠️ 開發指南](#️-開發指南)
- [🎯 2025年11月更新摘要](#-2025年11月更新摘要)
- [🔐 安全性](#-安全性)
- [📊 監控與可觀測性](#-監控與可觀測性)
- [🤝 貢獻指南](#-貢獻指南)
- [📄 授權](#-授權)
- [📞 支援與聯繫](#-支援與聯繫)

---

## 📋 概述

**AIVA Services** 是一個現代化的企業級 Bug Bounty 平台，採用**微服務架構**設計，整合 **Python**、**TypeScript**、**Rust**、**Go** 四種語言技術棧，專精於動態漏洞檢測、黑盒滲透測試和智能攻擊策略規劃。

### 🎯 核心特性

- ✅ **AI 驅動**: 智能攻擊策略規劃、語義分析、自動化決策
- ✅ **多語言協同**: Python (AI/協調) + Rust (性能) + Go (並發) + TypeScript (前端)
- ✅ **企業級架構**: 微服務設計、分散式整合、統一數據標準
- ✅ **Bug Bounty 專業化**: 動態檢測、黑盒測試、實戰滲透
- ✅ **國際標準支援**: CVSS v3.1、MITRE ATT&CK、SARIF v2.1.0、CVE/CWE/CAPEC

---

## 📂 README 架構圖

### 📚 完整文檔結構樹

```
services/
│
├── README.md (本文檔) ⭐ 總覽索引
│   ├─ 📋 概述與核心特性
│   ├─ 🏗️ 服務架構總覽
│   ├─ 📦 五大核心服務介紹
│   ├─ 📊 整體統計數據
│   ├─ 🚀 快速開始指南
│   ├─ 🏗️ 架構設計原則
│   ├─ 🔗 服務間協作流程
│   ├─ 📚 文檔導航系統
│   ├─ 🛠️ 開發指南
│   ├─ 🎯 2025年11月更新摘要
│   ├─ 🔐 安全性與合規
│   └─ 📊 監控與可觀測性
│
├── 🤖 core/
│   ├── README.md (v6.1) 📖 AI 驅動核心引擎完整文檔
│   │   ├─ 導航: ← Services 總覽 | 文檔中心
│   │   ├─ � 總目錄 (13個主要章節)
│   │   ├─ 🚀 2025年11月架構修復摘要 (P0-P2)
│   │   ├─ 🏗️ 核心架構總覽
│   │   ├─ 📁 Core 子目錄結構
│   │   │   ├─ 🎯 AIVA Core (主引擎)
│   │   │   ├─ 🧠 AI Core (智慧增強)
│   │   │   └─ ⚡ AIVA Core v1 (工作流引擎)
│   │   ├─ 🔗 核心模組整合分析
│   │   ├─ 🛠️ Core 模組開發工具
│   │   ├─ � 模組規模一覽
│   │   ├─ 🚀 快速開始指南
│   │   ├─ 🧠 AI 系統運作機制詳解
│   │   ├─ ⚡ 執行引擎架構
│   │   ├─ 🧠 學習系統架構
│   │   ├─ 📊 分析決策系統
│   │   └─ 💾 存儲與狀態管理
│   │
│   ├── aiva_core/ (主引擎實現)
│   │   ├── ai_engine/ (AI 引擎核心)
│   │   ├── execution/ (執行引擎)
│   │   ├── learning/ (學習系統)
│   │   ├── analysis/ (分析決策)
│   │   ├── storage/ (存儲管理)
│   │   └── ... (50+ 子模組)
│   │
│   └── docs/ (空 - 內容已整合至 README)
│
├── 🔗 aiva_common/
│   ├── README.md (v6.1) 📖 Bug Bounty 共享庫完整文檔
│   │   ├─ 導航: ← Services 總覽 | 文檔中心
│   │   ├─ 📑 目錄 (14個主要章節)
│   │   ├─ 📋 概述
│   │   ├─ 🚀 核心特性
│   │   ├─ 🔧 快速安裝
│   │   ├─ 📊 數據模型 (Schema)
│   │   ├─ ⚙️ 配置管理
│   │   ├─ 📈 可觀測性
│   │   ├─ 🔨 異步工具
│   │   ├─ 🧩 插件架構
│   │   ├─ 🛡️ 安全特性
│   │   ├─ 🧪 測試指南
│   │   ├─ 📚 API 文檔
│   │   ├─ 🔄 開發指南
│   │   └─ 🚧 故障排除
│   │
│   ├── schemas/ (78+ Pydantic 模型)
│   ├── config/ (統一配置系統)
│   ├── observability/ (監控追蹤)
│   ├── async_utils/ (異步工具)
│   ├── plugins/ (插件架構)
│   ├── cross_language/ (跨語言支援)
│   │   ├── adapters/ (Go/Rust 適配器)
│   │   └── ... (Schema 生成器)
│   └── ... (50+ Python 檔案)
│
├── 🎯 features/
│   ├── README.md (v6.1) 📖 多語言安全功能架構文檔
│   │   ├─ 導航: ← Services 總覽 | 文檔中心
│   │   ├─ 📑 目錄
│   │   ├─ 🔧 修復規範
│   │   ├─ 🔗 功能模組導航
│   │   │   ├─ 📚 核心功能模組 (7個)
│   │   │   │   ├─ ✅ 完整實現 (5個)
│   │   │   │   └─ 🔹 部分實現 (2個)
│   │   │   ├─ 🎯 專業工具模組
│   │   │   └─ 🌐 多語言架構支援
│   │   ├─ 📊 模組統計
│   │   ├─ 🛠️ 開發指南
│   │   ├─ 🧪 測試
│   │   └─ 📚 文檔資源
│   │
│   ├── function_sqli/ 📖 README.md (SQL 注入檢測)
│   ├── function_xss/ 📖 README.md (XSS 檢測)
│   ├── function_ssrf/ 📖 README.md (SSRF 檢測)
│   ├── function_idor/ 📖 README.md (IDOR 檢測)
│   ├── function_authn_go/ 📖 README.md (認證檢測 - Go)
│   ├── function_crypto/ 📖 README.md (密碼學檢測 - Python+Rust)
│   ├── function_postex/ 📖 README.md (後滲透模組)
│   └── ... (70+ 多語言檔案)
│
├── 🔄 integration/
│   ├── README.md (v6.1) 📖 企業級整合中樞完整文檔
│   │   ├─ 導航: ← Services 總覽 | 文檔中心
│   │   ├─ 📑 目錄
│   │   ├─ 🎯 核心文檔
│   │   │   └─ 📖 Integration Core 核心模組
│   │   ├─ � 本文檔內容
│   │   │   ├─ 🚀 快速開始
│   │   │   ├─ 🔧 環境變數配置
│   │   │   ├─ 🛠️ 開發工具與環境
│   │   │   ├─ 🏗️ 整合架構深度分析
│   │   │   ├─ 📊 效能基準與全方位監控
│   │   │   ├─ 💡 使用方式與最佳實踐
│   │   │   ├─ 🔮 發展方向與路線圖
│   │   │   ├─ 🛡️ 安全性與合規
│   │   │   ├─ 🔧 故障排除與維護
│   │   │   ├─ 📚 API 參考
│   │   │   ├─ 👨‍💻 開發規範與最佳實踐
│   │   │   └─ 🤝 貢獻指南
│   │   └─ 📄 授權與支援
│   │
│   ├── aiva_integration/
│   │   ├── README.md 📖 核心實現詳解 (7層架構)
│   │   ├── analysis/ (合規檢查、風險評估)
│   │   ├── attack_path_analyzer/ (攻擊路徑分析)
│   │   ├── reception/ (數據接收層)
│   │   ├── remediation/ (修復建議)
│   │   ├── reporting/ (報告生成)
│   │   ├── threat_intel/ (威脅情報)
│   │   └── ... (40+ Python 檔案)
│   │
│   └── capability/ (功能工具集)
│
└── 🔍 scan/
    ├── README.md (v6.1) 📖 多語言統一掃描引擎文檔
    │   ├─ 導航: ← Services 總覽 | 文檔中心
    │   ├─ 📑 目錄
    │   ├─ 🔧 修復規範
    │   ├─ 📊 模組統計
    │   ├─ 🏗️ 核心架構
    │   │   ├─ 多語言協同設計
    │   │   ├─ 掃描引擎架構
    │   │   └─ 統一API層
    │   ├─ 🎯 子模組詳解
    │   │   ├─ AIVA Scan核心
    │   │   └─ Rust資訊收集器
    │   ├─ 🚀 快速開始
    │   ├─ 🛠️ 開發指南
    │   ├─ 🔍 掃描功能
    │   ├─ 📊 性能指標
    │   └─ 🧪 測試
    │
    ├── aiva_scan/ (Python 掃描核心)
    ├── aiva_scan_node/ (TypeScript 動態掃描)
    ├── go_scanners/ (Go 掃描器: CSPM, SCA, SSRF)
    ├── info_gatherer_rust/ (Rust 資訊收集器)
    └── ... (60+ 多語言檔案)
```

### 🔗 階層關聯關係

```
📄 services/README.md (主索引)
    │
    ├─ 🔗 雙向連結 → core/README.md
    │   └─ 📚 子文檔: aiva_core/, ai/, docs/
    │
    ├─ 🔗 雙向連結 → aiva_common/README.md
    │   └─ 📚 子文檔: schemas/, config/, cross_language/
    │
    ├─ 🔗 雙向連結 → features/README.md
    │   └─ 📚 子文檔: 7個功能模組 README
    │       ├─ function_sqli/README.md
    │       ├─ function_xss/README.md
    │       ├─ function_ssrf/README.md
    │       ├─ function_idor/README.md
    │       ├─ function_authn_go/README.md
    │       ├─ function_crypto/README.md
    │       └─ function_postex/README.md
    │
    ├─ 🔗 雙向連結 → integration/README.md
    │   └─ 📚 子文檔: aiva_integration/README.md
    │       └─ 7層架構詳解
    │
    └─ 🔗 雙向連結 → scan/README.md
        └─ 📚 子文檔: 多語言掃描引擎組件

🌐 所有子模組 → ../../docs/README.md (文檔中心)
```

### 📊 README 統計

| 模組 | README 文件 | 版本 | 更新日期 | 主要章節數 | 狀態 |
|------|------------|------|---------|-----------|------|
| **Services** | README.md | v6.1 | 2025-11-13 | 17 | ✅ 完整 |
| **Core** | README.md | v6.1 | 2025-11-13 | 13 | ✅ 完整 |
| **Common** | README.md | v6.1 | 2025-11-13 | 14 | ✅ 完整 |
| **Features** | README.md | v6.1 | 2025-11-13 | 10 | ✅ 完整 |
| **Integration** | README.md | v6.1 | 2025-11-13 | 15 | ✅ 完整 |
| **Scan** | README.md | v6.1 | 2025-11-13 | 12 | ✅ 完整 |
| **子模組總計** | 8+ 個 | - | - | - | ✅ 完整 |

### 🎯 文檔導航快速鍵

- **🏠 總覽首頁**: [services/README.md](README.md) ← 您在這裡
- **🤖 AI 核心**: [core/README.md](core/README.md) - 3層架構、AI引擎、執行系統
- **🔗 共享庫**: [aiva_common/README.md](aiva_common/README.md) - 78+ 模型、跨語言支援
- **🎯 功能集**: [features/README.md](features/README.md) - 7大檢測模組
- **🔄 整合層**: [integration/README.md](integration/README.md) - 7層架構、AI協調
- **🔍 掃描引擎**: [scan/README.md](scan/README.md) - 4語言協同掃描

---

## 🏗️ 服務架構總覽

```
┌─────────────────────────────────────────────────────────────────┐
│                        AIVA Services                            │
│                    五大核心服務協同                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─── 🤖 Core (AI 核心引擎)
         │    └─── AI 驅動、智能決策、策略規劃
         │
         ├─── 🔗 Common (共享庫)
         │    └─── 數據標準、工具集、跨語言支援
         │
         ├─── 🎯 Features (功能模組)
         │    └─── 漏洞檢測、攻擊功能、專業工具
         │
         ├─── 🔄 Integration (整合中樞)
         │    └─── 服務協調、監控、通信管理
         │
         └─── 🔍 Scan (掃描引擎)
              └─── 多語言掃描、性能優化、動態檢測
```

---

## 📦 五大核心服務

### 🤖 [Core - AI 驅動核心引擎](core/README.md)

**定位**: AI 智能決策中樞、攻擊策略規劃引擎

**核心能力**:
- 🧠 **AI 引擎**: sentence-transformers 語義編碼、智能分析
- ⚡ **執行引擎**: 動態計劃執行、實時決策調整
- 📊 **學習系統**: 經驗積累、模式識別、策略優化
- 🎯 **Bug Bounty**: 動態檢測專精、黑盒滲透測試

**技術棧**: Python 3.11+、FastAPI、Pydantic v2、sentence-transformers

**關鍵特性**:
- ✅ P0-P2 架構修復完成 (依賴注入、RAG 簡化、命令安全)
- ✅ RAG1徹底移除 (第五次確認完成，無殘留引用)
- ✅ execution_tracer模組修復 (建立trace_recorder.py，解決缺失組件)
- ✅ AI 語義編碼升級 (384維向量、相似度分析)
- ✅ 三層架構: AI Core + AIVA Core + AIVA Core v1

**📖 詳細文檔**: [services/core/README.md](core/README.md)

---

### 🔗 [Common - Bug Bounty 共享庫](aiva_common/README.md)

**定位**: 跨服務統一數據標準、工具集、配置管理

**核心能力**:
- 📊 **數據模型**: 78+ Pydantic 模型、48+ 標準枚舉
- 🌐 **跨語言支援**: TypeScript/Go 生成器、Schema 統一
- 📡 **Protocol Buffers**: gRPC 跨語言通信、pb2.py 生成
- ⚙️ **配置管理**: 統一配置系統、環境變數管理
- 📈 **可觀測性**: 日誌、指標、追蹤、監控

**技術棧**: Python 3.11+、Pydantic v2、asyncio、py.typed

**關鍵特性**:
- ✅ 50+ Python 檔案、8,500+ 行代碼
- ✅ 符合 CVSS v3.1、MITRE ATT&CK、SARIF v2.1.0 標準
- ✅ 完整類型支援、靜態檢查、PEP 8 規範
- ✅ 異步工具、插件架構、訊息處理

**📖 詳細文檔**: [services/aiva_common/README.md](aiva_common/README.md)

---

### 🎯 [Features - 多語言安全功能](features/README.md)

**定位**: 漏洞檢測功能集、攻擊工具模組

**核心能力**:
- ⚡ **SQL 注入**: 20 個 Python 文件、6 個檢測引擎
- 🎭 **XSS 檢測**: DOM/存儲型/反射型、10 個文件
- 🌐 **SSRF 檢測**: 內網探測、12 個文件
- 🔐 **IDOR 檢測**: 權限升級測試、12 個文件
- 🔑 **認證檢測**: Go 語言實現、高性能、5 個文件

**技術棧**: Python、Go、Rust、TypeScript 混合架構

**關鍵特性**:
- ✅ 5 個完整實現模組、2 個部分實現模組
- ✅ 多語言協同設計 (Python + Go + Rust)
- ✅ 專業工具模組 (密碼學、命令注入、文件上傳)
- ✅ 可擴展架構、模組化設計

**📖 詳細文檔**: [services/features/README.md](features/README.md)

---

### 🔄 [Integration - 企業級整合中樞](integration/README.md)

**定位**: 服務協調器、通信管理、效能監控、資料儲存管理

**核心能力**:
- 🤖 **AI Operation Recorder**: 智能協調核心
- 📡 **多層整合架構**: 7 層分散式設計
- 📊 **效能監控**: 實時指標、追蹤、告警
- 🔗 **服務通信**: RabbitMQ、PostgreSQL (Neo4j → NetworkX 遷移完成)
- 💾 **資料儲存**: 標準化資料儲存結構 (攻擊路徑、經驗記錄、訓練資料集、模型檢查點)
- 🔧 **維護工具**: 自動備份與清理腳本

**技術棧**: Python 3.11+、FastAPI、PostgreSQL、Redis、RabbitMQ、NetworkX

**關鍵特性**:
- ✅ 企業級整合中樞、AI 驅動協調
- ✅ 7 層架構深度、模組化設計
- ✅ 統一配置系統、環境變數管理
- ✅ 完整監控體系、故障排除工具
- ✅ **NEW**: 資料儲存標準化 (2025-11-16)
- ✅ **NEW**: Neo4j → NetworkX 遷移 (零外部依賴)
- ✅ **NEW**: 自動備份與清理維護腳本

**📖 詳細文檔**: [services/integration/README.md](integration/README.md)

---

### 🔍 [Scan - 多語言統一掃描引擎](scan/README.md)

**定位**: 高性能掃描引擎、動態檢測、黑盒測試

**核心能力**:
- 🚀 **多語言協同**: Python + TypeScript + Rust + Go
- 🔍 **主動掃描**: 智能爬蟲、漏洞探測
- 📡 **被動監聽**: 流量分析、異常檢測
- 🧠 **智能分析**: AI 輔助、模式識別

**技術棧**: Python、TypeScript、Rust、Go 四語言協同

**關鍵特性**:
- ✅ 四語言技術棧整合 (性能 + 並發 + AI)
- ✅ 統一 API 層、跨語言通信
- ✅ Rust 資訊收集器 (高性能、低延遲)
- ✅ Bug Bounty 專精、實戰滲透測試

**📖 詳細文檔**: [services/scan/README.md](scan/README.md)

---

## 📊 整體統計

### 代碼規模
```
Core:        50+ Python 檔案, 12,000+ 行代碼
Common:      50+ Python 檔案,  8,500+ 行代碼
Features:    70+ 多語言檔案, 15,000+ 行代碼
Integration: 40+ Python 檔案, 10,000+ 行代碼
Scan:        60+ 多語言檔案, 13,000+ 行代碼
─────────────────────────────────────────────
總計:       270+ 檔案,       58,500+ 行代碼
```

### 技術棧分布
- **Python**: 核心語言 (AI、協調、業務邏輯)
- **Rust**: 性能關鍵路徑 (掃描、資訊收集)
- **Go**: 高並發場景 (認證檢測、並行處理)
- **TypeScript**: 前端整合、API 生成

### 數據模型
- **Pydantic 模型**: 150+ 個
- **標準枚舉**: 80+ 個
- **API 端點**: 100+ 個

---

## 🚀 快速開始

### 環境要求
```bash
# 核心依賴
Python 3.11+
Node.js 18+
Rust 1.70+
Go 1.21+

# 資料庫與中間件
PostgreSQL 15+ (with pgvector)
Redis 7.0+
RabbitMQ 3.12+
Neo4j 5.0+
```

### 安裝流程

#### 1. 克隆專案
```bash
git clone https://github.com/kyle0527/AIVA.git
cd AIVA/services
```

#### 2. 設置 Python 環境
```bash
# 創建虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt
```

#### 3. 配置環境變數
```bash
# 複製環境變數範本
cp .env.example .env

# 編輯配置 (資料庫連接、API 金鑰等)
notepad .env  # Windows
vim .env      # Linux/Mac
```

#### 4. 啟動服務

**選項 A: 使用 Docker Compose (推薦)**
```bash
cd ..
docker-compose up -d
```

**選項 B: 手動啟動**
```bash
# 啟動 Core 服務
cd core
python -m aiva_core.main

# 啟動 Integration 服務
cd ../integration
python -m aiva_integration.main

# 啟動 Scan 服務
cd ../scan
python -m aiva_scan.main
```

---

## 🏗️ 架構設計原則

### 1. 微服務架構
- **服務獨立**: 每個服務獨立部署、擴展、維護
- **鬆耦合**: 通過標準 API 和訊息隊列通信
- **高內聚**: 相關功能聚合在同一服務內

### 2. 多語言協同
- **語言選型**: 根據場景選擇最優技術
- **統一標準**: Common 模組提供跨語言 Schema
- **互操作性**: gRPC、REST API、訊息隊列

### 3. 數據驅動
- **統一模型**: Pydantic v2 強類型檢查
- **標準支援**: CVSS、MITRE、SARIF、CVE/CWE
- **Schema First**: 先定義數據結構、再實現邏輯

### 4. AI 增強
- **語義分析**: sentence-transformers 384 維向量
- **智能決策**: 經驗學習、模式識別
- **自動化**: AI 輔助攻擊策略規劃

---

## 🔗 服務間協作流程

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Scan    │────▶│Integration│────▶│   Core   │
│  掃描    │     │  整合     │     │  決策    │
└──────────┘     └──────────┘     └──────────┘
     │                 │                 │
     │                 ▼                 ▼
     │           ┌──────────┐     ┌──────────┐
     └──────────▶│ Common   │     │ Features │
                 │ 共享庫    │     │ 功能集    │
                 └──────────┘     └──────────┘

流程說明:
1. Scan 執行掃描 → 發現目標
2. Integration 協調服務 → 分配任務
3. Core 分析決策 → 制定策略
4. Features 執行攻擊 → 漏洞驗證
5. Common 提供支援 → 數據標準、工具集
```

---

## 📚 文檔導航

### 核心文檔
- 📖 **[Core 模組文檔](core/README.md)** - AI 引擎架構、開發指南
- 📖 **[Common 模組文檔](aiva_common/README.md)** - 數據模型、API 參考
- 📖 **[Features 模組文檔](features/README.md)** - 功能模組、檢測引擎
- 📖 **[Integration 模組文檔](integration/README.md)** - 整合架構、監控系統
- 📖 **[Scan 模組文檔](scan/README.md)** - 掃描引擎、多語言協同

### 專題文檔
> **📌 注意**: 各模組的詳細文檔請參考各自的 README.md
- 📖 **Core 模組**: 詳細的 AI 引擎、執行引擎、學習系統文檔請見 [core/README.md](core/README.md)
- 📖 **Integration 模組**: 整合架構、API 參考請見 [integration/README.md](integration/README.md) 及 [aiva_integration/README.md](integration/aiva_integration/README.md)
- 📖 **Features 模組**: 各功能模組文檔請見 [features/README.md](features/README.md) 及各功能子目錄

---

## 🛠️ 開發指南

### 代碼規範
- **Python**: Black、Ruff、mypy、PEP 8
- **TypeScript**: ESLint、Prettier、TSConfig strict
- **Rust**: rustfmt、clippy、cargo check
- **Go**: gofmt、golangci-lint、go vet

### 提交規範
```bash
# Conventional Commits
feat: 新功能
fix: 修復
docs: 文檔
refactor: 重構
test: 測試
chore: 雜項
```

### 測試要求
- **單元測試**: 覆蓋率 ≥ 80%
- **整合測試**: 關鍵路徑 100%
- **性能測試**: 基準測試 + 負載測試
- **安全測試**: SAST + DAST

---

## 🎯 2025年11月更新摘要

### ✅ Core 模組架構修復完成 (2025-11-15)
- **RAG1徹底移除**: 第五次確認，所有RAG引用已指向當前版本，無legacy殘留
- **execution_tracer修復**: 建立`execution/trace_recorder.py`解決缺失模組問題
- **導入路徑修復**: 修正`services/core/__init__.py`相對導入錯誤
- **P0-P2架構優化**: 移除Mock邏輯、實施依賴注入、RAG簡化、命令安全

### 🔄 跨語言 gRPC 整合完成 (2025-11-15)
- **Protocol Buffers 生成**: aiva_services.proto → pb2.py 自動化編譯
- **multilang_coordinator 修正**: 38 個 Pylance 錯誤 → 0 個錯誤
- **Type Ignore 註釋**: 符合 Google gRPC Python 官方標準
- **跨語言通信**: Python ↔ Go ↔ Rust ↔ TypeScript gRPC 通道就緒

### 🧠 AI 語義編碼升級 (2025-11-15)
- **sentence-transformers 5.1.1**: 384 維語義向量
- **real_neural_core.py**: 替換字符累加為語義編碼
- **相似度分析**: 區分代碼語義 (0.25-0.59 vs 閑值 0.7)

### 💾 整合模組資料儲存標準化完成 (2025-11-16)
- **目錄結構建立**: `data/integration/` 標準化資料儲存 (攻擊路徑、經驗記錄、訓練資料集、模型檢查點)
- **Neo4j → NetworkX 遷移**: 攻擊路徑分析引擎完全遷移至 NetworkX (零外部依賴)
- **統一配置系統**: `config.py` 集中管理所有資料儲存路徑
- **維護腳本建立**: `backup.py` (自動備份) + `cleanup.py` (舊資料清理)
- **環境變數配置**: `.env` 新增整合模組專用配置 (AIVA_INTEGRATION_DATA_DIR 等)
- **依賴簡化**: 資料庫依賴從 4 個減少至 2 個 (PostgreSQL + RabbitMQ)

### 📦 版本同步
- **所有模組**: 升級至 v6.2
- **日期統一**: 2025年11月16日
- **文檔更新**: RAG1移除確認、execution_tracer修復記錄、gRPC 整合說明、資料儲存標準化說明

---

## 🔐 安全性

### 安全特性
- 🔒 **命令安全**: shlex.split 防注入
- 🛡️ **輸入驗證**: Pydantic 嚴格模式
- 🔑 **密鑰管理**: 環境變數、密鑰輪換
- 📊 **審計日誌**: 完整操作記錄

### 合規支援
- ✅ **CVSS v3.1**: 漏洞評分標準
- ✅ **MITRE ATT&CK**: 攻擊技術框架
- ✅ **SARIF v2.1.0**: 靜態分析格式
- ✅ **CVE/CWE/CAPEC**: 漏洞分類標準

---

## 📊 監控與可觀測性

### 監控層級
- **應用層**: API 延遲、錯誤率、吞吐量
- **服務層**: 服務健康、資源使用、依賴狀態
- **基礎設施層**: CPU、記憶體、磁碟、網路

### 工具鏈
- **日誌**: 結構化日誌、集中收集
- **指標**: Prometheus、Grafana
- **追蹤**: OpenTelemetry、分散式追蹤
- **告警**: 智能告警、異常檢測

---

## 🤝 貢獻指南

### 如何貢獻
1. **Fork 專案** → GitHub 右上角 Fork 按鈕
2. **創建分支** → `git checkout -b feature/your-feature`
3. **提交代碼** → `git commit -m "feat: add feature"`
4. **推送分支** → `git push origin feature/your-feature`
5. **提交 PR** → GitHub Pull Request

### 開發流程
1. 閱讀相關模組文檔
2. 配置開發環境
3. 編寫代碼 + 測試
4. 通過 CI/CD 檢查
5. Code Review
6. 合併至主分支

---

## 📄 授權

本專案採用 **MIT License** 授權。詳見 [LICENSE](../LICENSE) 檔案。

---

## 📞 支援與聯繫

- **GitHub Issues**: [提交問題](https://github.com/kyle0527/AIVA/issues)
- **文檔中心**: [docs/README.md](../docs/README.md)
- **專案主頁**: [AIVA GitHub](https://github.com/kyle0527/AIVA)

---

<div align="center">

**🎯 AIVA Services v6.1 - Bug Bounty 專業化平台**

*五大核心服務協同，AI 驅動智能滲透測試*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-cyan.svg)](https://golang.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**最後更新**: 2025年11月16日

</div>
