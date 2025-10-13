╔══════════════════════════════════════════════════════════════════════════════╗
║                    AIVA 專案完整分析與架構說明報告                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

**報告生成日期**: 2025年10月13日 (星期日)
**生成時間**: 12:16:23 UTC+0
**專案路徑**: `/workspaces/AIVA`
**分析工具**: `analyze_codebase.py` + `generate_project_report.ps1`
**報告版本**: v2.1 (整合版 + 變化追蹤)
**前次報告**: PROJECT_REPORT.txt (2025-10-13 08:28:21)
**分析間隔**: ~4 小時═══════════════════════════════════════════════════════════════════════════════
📊 專案統計摘要
═══════════════════════════════════════════════════════════════════════════════

## 📈 與前次報告比較 (2025-10-13 08:28 vs 12:16)

| 指標 | 前次報告 | 本次報告 | 變化 | 趨勢 |
|------|----------|----------|------|------|
| **總檔案數** | 235 | 235 | 0 | ➡️ 持平 |
| **Python 檔案** | 169 | 169 | 0 | ➡️ 持平 |
| **總程式碼行數** | 28,442 | 28,442 | 0 | ➡️ 持平 |
| **Python 程式碼** | 24,063 | 27,015* | +2,952 | ⚠️ 需確認 |
| **總函數數** | N/A | 704 | +704 | ✅ 新增統計 |
| **總類別數** | N/A | 299 | +299 | ✅ 新增統計 |
| **平均複雜度** | N/A | 11.94 | +11.94 | ✅ 新增統計 |
| **類型提示覆蓋率** | N/A | 74.8% | +74.8% | ✅ 新增統計 |
| **文檔字串覆蓋率** | N/A | 81.9% | +81.9% | ✅ 新增統計 |

> **注意**: 本次報告新增了程式碼品質分析功能，包含函數數、類別數、複雜度等指標。
> *Python 程式碼行數差異可能由於統計方法不同（前次包含空白行和註解，本次為純程式碼行）。

## 整體規模

| 指標 | 數值 | 說明 |
|------|------|------|
| **總檔案數** | 235 | 包含所有類型檔案 |
| **程式碼檔案** | 221 | 實際程式碼檔案 (含多語言) |
| **Python 檔案** | 169 | 主要開發語言 (76.5%) |
| **Go 檔案** | 18 | 高性能檢測模組 (8.1%) |
| **Rust 檔案** | 10 | 靜態分析模組 (4.5%) |
| **TypeScript 檔案** | 3 | 動態掃描引擎 (1.4%) |
| **總程式碼行數** | 33,318 行 | 包含所有程式語言 |
| **Python 程式碼** | 27,015 行 | Python 程式碼 (81.1%) |
| **Go 程式碼** | 2,972 行 | Go 程式碼 (8.9%) |
| **Rust 程式碼** | 1,552 行 | Rust 程式碼 (4.7%) |
| **TypeScript 程式碼** | 352 行 | TypeScript 程式碼 (1.1%) |
| **其他 (Markdown/配置)** | 1,427 行 | 文檔與配置 (4.3%) |

## 多語言架構優勢 🌟

AIVA 採用 **多語言混合架構**，針對不同場景選擇最適合的程式語言：

- **Python** (81.1%): 主要業務邏輯、AI 引擎、核心協調
- **Go** (8.9%): 高性能檢測模組、並發處理、雲端掃描
- **Rust** (4.7%): 靜態分析、記憶體安全、高效能解析
- **TypeScript** (1.1%): 動態網頁掃描、Playwright 整合

## 程式碼品質指標

| 指標 | 當前值 | 目標值 | 狀態 |
|------|--------|--------|------|
| **Python 檔案** | 169 個 | - | ✅ |
| **平均檔案大小** | 142.4 行 | <200 行 | ✅ 優秀 |
| **總函數數** | 704 | - | ✅ |
| **總類別數** | 299 | - | ✅ |
| **平均複雜度** | 11.94 | <10 | ⚠️ 需改進 |
| **類型提示覆蓋率** | 74.8% | 90%+ | ⚠️ 需改進 |
| **文檔字串覆蓋率** | 81.9% | 95%+ | ✅ 良好 |
| **編碼相容性** | 100% | 100% | ✅ 完美 |

───────────────────────────────────────────────────────────────────────────────
🎯 檔案類型統計 (Top 10)
───────────────────────────────────────────────────────────────────────────────

```
  .py                169 個檔案  ████████████████████████████████████ 100%
  .mmd                24 個檔案  ████████ 14.2%
  .md                  8 個檔案  ██ 4.7%
  .txt                 7 個檔案  █ 4.1%
  .backup              5 個檔案  █ 3.0%
  .ps1                 3 個檔案  █ 1.8%
  .ini                 2 個檔案  █ 1.2%
  .toml                2 個檔案  █ 1.2%
  .bat                 2 個檔案  █ 1.2%
  .yml                 2 個檔案  █ 1.2%
```

───────────────────────────────────────────────────────────────────────────────
💻 程式碼行數統計 (依副檔名 - 更新版)
───────────────────────────────────────────────────────────────────────────────

### 程式語言統計

| 語言/類型 | 行數 | 檔案數 | 平均行數 | 佔比 | 用途 |
|-----------|------|--------|----------|------|------|
| **Python (.py)** | 27,015 | 169 | 159.8 | 81.1% | 主要業務邏輯、AI 引擎 |
| **Go (.go)** | 2,972 | 18 | 165.1 | 8.9% | 高性能檢測、並發處理 |
| **Rust (.rs)** | 1,552 | 10 | 155.2 | 4.7% | 靜態分析、安全掃描 |
| **TypeScript (.ts)** | 352 | 3 | 117.3 | 1.1% | 動態網頁掃描 |
| **Markdown (.md)** | 3,180 | 8 | 397.5 | 9.5% | 專案文檔 |
| **PowerShell (.ps1)** | 518 | 3 | 172.7 | 1.6% | 自動化腳本 |
| **YAML (.yml)** | 216 | 2 | 108.0 | 0.6% | 配置檔案 |
| **SQL (.sql)** | 178 | 1 | 178.0 | 0.5% | 資料庫架構 |
| **TOML (.toml)** | 130 | 2 | 65.0 | 0.4% | Python 配置 |
| **Shell (.sh)** | 65 | 1 | 65.0 | 0.2% | 啟動腳本 |
| **YAML (.yaml)** | 49 | 1 | 49.0 | 0.1% | 配置檔案 |
| **Batch (.bat)** | 35 | 2 | 17.5 | 0.1% | Windows 腳本 |
| **JSON (.json)** | 8 | 1 | 8.0 | 0.02% | 配置檔案 |

**程式碼總計**: 33,318 行 (221 個程式碼檔案)

### 多語言模組分佈

| 模組名稱 | 語言 | 檔案數 | 行數 | 功能說明 |
|----------|------|--------|------|----------|
| **function_authn_go** | Go | 7 | 1,243 | 🔐 身份驗證漏洞檢測 |
| **function_cspm_go** | Go | 5 | 644 | ☁️ 雲端安全態勢管理 (AWS/Azure/GCP) |
| **function_sca_go** | Go | 4 | 704 | 📦 軟體組成分析 (依賴掃描) |
| **function_ssrf_go** | Go | 2 | 381 | 🌐 SSRF 檢測 (Go 實作) |
| **function_sast_rust** | Rust | 6 | 596 | 🔍 靜態程式碼安全分析 |
| **info_gatherer_rust** | Rust | 4 | 956 | 🔎 資訊收集 (敏感資料/Git 歷史) |
| **aiva_scan_node** | TypeScript | 3 | 352 | 🎭 Playwright 動態掃描引擎 |

**總計**: 7 個多語言模組, 31 個檔案, 4,876 行程式碼

───────────────────────────────────────────────────────────────────────────────
🌐 多語言模組詳細分析
───────────────────────────────────────────────────────────────────────────────

### 1. 🔐 Go 模組 (18 files, 2,972 lines)

#### function_authn_go - 身份驗證漏洞檢測

**路徑**: `services/function/function_authn_go/`
**檔案數**: 7 個 Go 檔案
**程式碼行數**: 1,243 行

**主要功能**:

- **暴力破解測試** (`brute_forcer.go`)
  - 自動化密碼暴力破解嘗試
  - 帳號枚舉檢測
  - 速率限制測試

- **弱配置檢測** (`config_tester.go`)
  - 預設密碼檢測
  - 弱密碼策略識別
  - 認證機制安全性測試

- **Token 分析** (`token_analyzer.go`)
  - JWT/Session Token 安全性分析
  - Token 過期時間檢查
  - 簽名驗證測試

**技術特點**:

- 高並發處理 (Goroutines)
- RabbitMQ 消息佇列整合
- Uber Zap 結構化日誌

---

#### function_cspm_go - 雲端安全態勢管理

**路徑**: `services/function/function_cspm_go/`
**檔案數**: 5 個 Go 檔案
**程式碼行數**: 644 行

**主要功能**:

- **多雲端平台支援**
  - AWS 安全配置掃描
  - Azure 安全配置掃描
  - GCP 安全配置掃描
  - Kubernetes 安全檢查

- **Trivy 整合** (`cspm_scanner.go`)
  - 使用 Trivy 進行雲端配置掃描
  - 自動解析 Trivy 輸出
  - 風險評級與分類

**檢測項目**:

- IAM 權限配置錯誤
- 儲存桶公開存取
- 網路安全組規則
- 加密設定缺失
- 合規性違規

---

#### function_sca_go - 軟體組成分析

**路徑**: `services/function/function_sca_go/`
**檔案數**: 4 個 Go 檔案
**程式碼行數**: 704 行

**主要功能**:

- **依賴掃描** (`sca_scanner.go`)
  - 檢測已知漏洞依賴
  - CVE 資料庫比對
  - 版本升級建議

- **支援的套件管理工具**
  - npm (Node.js)
  - pip (Python)
  - Maven/Gradle (Java)
  - Go modules
  - Cargo (Rust)

**輸出資訊**:

- 漏洞 CVE 編號
- 嚴重程度評分 (CVSS)
- 受影響的套件版本
- 修復建議

---

#### function_ssrf_go - SSRF 檢測 (Go 版本)

**路徑**: `services/function/function_ssrf_go/`
**檔案數**: 2 個 Go 檔案
**程式碼行數**: 381 行

**主要功能**:

- SSRF 漏洞檢測 (Go 高性能實作)
- 內部網路探測
- 雲端元資料服務檢測
- DNS Rebinding 測試

**與 Python 版本的差異**:

- 更高的並發性能
- 更低的記憶體佔用
- 適合大規模掃描場景

---

### 2. 🦀 Rust 模組 (10 files, 1,552 lines)

#### function_sast_rust - 靜態應用程式安全測試

**路徑**: `services/function/function_sast_rust/`
**檔案數**: 6 個 Rust 檔案
**程式碼行數**: 596 行

**主要功能**:

- **程式碼解析** (`parsers.rs`)
  - Python/JavaScript/Java/C++ 解析器
  - AST (抽象語法樹) 分析
  - 控制流圖構建

- **安全規則引擎** (`rules.rs`)
  - SQL 注入模式
  - XSS 漏洞模式
  - 路徑遍歷模式
  - 不安全的函數調用

- **分析器** (`analyzers.rs`)
  - 資料流分析
  - 污點追蹤
  - 符號執行
  - 模式匹配

**技術優勢**:

- 記憶體安全 (Rust 特性)
- 高效能解析
- 零成本抽象
- 並行分析能力

**檢測能力**:

- 注入漏洞 (SQL, Command, LDAP)
- XSS 漏洞
- 不安全的反序列化
- 競態條件
- 緩衝區溢位

---

#### info_gatherer_rust - 資訊收集器

**路徑**: `services/scan/info_gatherer_rust/`
**檔案數**: 4 個 Rust 檔案
**程式碼行數**: 956 行

**主要功能**:

- **敏感資訊檢測** (`secret_detector.rs`)
  - API 金鑰搜尋 (AWS, Google, Azure)
  - 憑證洩漏檢測 (密碼, Token)
  - PII 資料識別
  - 正則表達式高效匹配

- **Git 歷史掃描** (`git_history_scanner.rs`)
  - Git 提交歷史分析
  - 已刪除檔案中的敏感資料
  - 配置檔案變更追蹤
  - 大型 Git 倉庫高效處理

- **掃描器** (`scanner.rs`)
  - 多執行緒檔案掃描
  - 流式處理大檔案
  - 記憶體高效算法

**效能優勢**:

- 比 Python 版本快 5-10 倍
- 低記憶體佔用
- 可處理 GB 級別的倉庫

---

### 3. 📘 TypeScript 模組 (3 files, 352 lines)

#### aiva_scan_node - Playwright 動態掃描引擎

**路徑**: `services/scan/aiva_scan_node/`
**檔案數**: 3 個 TypeScript 檔案
**程式碼行數**: 352 行

**主要功能**:

- **Playwright 整合** (`scan-service.ts`)
  - Chromium 瀏覽器自動化
  - JavaScript 完整渲染
  - AJAX 請求監控
  - WebSocket 通訊捕獲

- **動態內容掃描**
  - 單頁應用 (SPA) 支援
  - 表單自動填充與提交
  - 點擊事件模擬
  - 頁面狀態追蹤

- **RabbitMQ 整合** (`index.ts`)
  - 任務佇列監聽
  - 結果發布
  - 錯誤處理與重試

**與 Python 動態引擎的配合**:

- Python: 協調與決策
- TypeScript/Playwright: 實際瀏覽器操作
- 職責分離,發揮各自優勢

**技術棧**:

- Playwright (瀏覽器自動化)
- amqplib (RabbitMQ 客戶端)
- TypeScript (型別安全)
- 異步編程 (async/await)

---

## 多語言架構設計理念

### 為什麼使用多語言?

| 場景 | 選擇語言 | 原因 |
|------|----------|------|
| **業務邏輯與 AI** | Python | 豐富的函式庫、快速開發、AI/ML 支援 |
| **高並發檢測** | Go | Goroutines、低延遲、高吞吐量 |
| **靜態分析** | Rust | 記憶體安全、零成本抽象、高效能 |
| **瀏覽器自動化** | TypeScript | Playwright 官方支援、型別安全 |

### 語言協作模式

```
┌─────────────────────────────────────────────────────────┐
│                   RabbitMQ 消息佇列                      │
│               (跨語言通訊中樞)                            │
└─────────────────────────────────────────────────────────┘
         ↑                ↑                ↑
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌───┴────┐
    │ Python  │      │   Go    │      │  Rust  │
    │ Workers │      │ Workers │      │Workers │
    └─────────┘      └─────────┘      └────────┘
         ↓                ↓                ↓
    ┌─────────────────────────────────────────┐
    │         PostgreSQL 統一資料庫             │
    └─────────────────────────────────────────┘
```

### 效能對比

| 任務類型 | Python | Go | Rust | TypeScript |
|----------|--------|----|----|------------|
| **業務邏輯** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **並發處理** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **記憶體效率** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **開發速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **生態系統** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**總計**: 28,442 行

═══════════════════════════════════════════════════════════════════════════════
🌳 完整專案目錄結構樹
═══════════════════════════════════════════════════════════════════════════════

> **注意**: 完整的樹狀結構由腳本自動產出，詳見以下檔案：
>
> - **過濾版本**: `_out/PROJECT_REPORT.txt` (已排除虛擬環境和快取)
> - **完整 Unicode 版本**: `_out/tree_unicode.txt`
> - **ASCII 版本**: `_out/tree_ascii.txt`
> - **Markdown 版本**: `_out/tree.md`
> - **HTML 版本**: `_out/tree.html`
> - **Mermaid 版本**: `_out/tree.mmd`
>
> 產生方式: 執行 `pwsh -File generate_project_report.ps1`

## 📁 主要目錄結構概覽

```plaintext
📦 AIVA (根目錄)
├─📁 _out/                              # 輸出目錄
│   ├─📁 analysis/                      # 分析報告
│   │   ├─📊 analysis_report_20251013_121623.json
│   │   ├─📊 analysis_report_20251013_121623.txt
│   │   └─📊 (其他歷史報告...)
│   ├─📄 PROJECT_REPORT.txt             # 專案結構報告
│   └─📄 tree_clean.txt                 # 目錄樹
│
├─📁 docker/                            # Docker 配置
│   ├─📁 initdb/
│   │   └─🗄️ 001_schema.sql            # 資料庫架構
│   ├─🔧 docker-compose.yml             # 開發環境
│   ├─🔧 docker-compose.production.yml  # 生產環境
│   ├─📄 Dockerfile.integration
│   └─⚡ entrypoint.integration.sh
│
├─📁 docs/                              # 文檔目錄
│   └─📁 diagrams/                      # Mermaid 流程圖
│       ├─📄 Module.mmd                 # 模組圖
│       └─📄 Function_*.mmd             # 24個函數流程圖
│
├─📁 services/                          # 主要服務目錄 ⭐
│   │
│   ├─📁 aiva_common/                   # 公用工具模組 (13 files, 1,960 lines)
│   │   ├─📁 utils/
│   │   │   ├─📁 dedup/                 # 去重工具
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   └─🐍 dedupe.py
│   │   │   ├─📁 network/               # 網路工具
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 backoff.py         # 指數退避
│   │   │   │   └─🐍 ratelimit.py       # ⚠️ 限流器 (複雜度91)
│   │   │   ├─🐍 __init__.py
│   │   │   ├─🐍 ids.py                 # ID 生成
│   │   │   └─🐍 logging.py             # 日誌工具
│   │   ├─🐍 __init__.py
│   │   ├─🐍 config.py                  # 配置管理
│   │   ├─🐍 enums.py                   # 列舉定義
│   │   ├─🐍 mq.py                      # 訊息佇列
│   │   ├─📄 py.typed                   # 型別標記
│   │   └─🐍 schemas.py                 # 資料結構
│   │
│   ├─📁 core/                          # 核心引擎模組 (28 files, 4,850 lines) ⭐
│   │   └─📁 aiva_core/
│   │       ├─📁 ai_engine/             # AI 引擎子系統
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 bio_neuron_core_v2.py  # 🧠 生物神經元核心 v2
│   │       │   ├─🐍 bio_neuron_core.py     # 生物神經元核心 v1
│   │       │   ├─🐍 knowledge_base.py      # 知識庫管理
│   │       │   └─🐍 tools.py               # ⚠️ AI工具 (複雜度38)
│   │       │
│   │       ├─📁 analysis/              # 分析子系統
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 dynamic_strategy_adjustment.py  # 動態策略調整
│   │       │   ├─🐍 initial_surface.py              # 初始攻擊面
│   │       │   └─🐍 strategy_generator.py           # 策略生成器
│   │       │
│   │       ├─📁 execution/             # 執行子系統
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 execution_status_monitor.py  # 狀態監控
│   │       │   ├─🐍 task_generator.py             # 任務生成器
│   │       │   └─🐍 task_queue_manager.py         # 任務佇列
│   │       │
│   │       ├─📁 ingestion/             # 資料接收
│   │       │   ├─🐍 __init__.py
│   │       │   └─🐍 scan_module_interface.py  # 掃描模組介面
│   │       │
│   │       ├─📁 output/                # 輸出處理
│   │       │   ├─🐍 __init__.py
│   │       │   └─🐍 to_functions.py        # 函數輸出
│   │       │
│   │       ├─📁 state/                 # 狀態管理
│   │       │   ├─🐍 __init__.py
│   │       │   └─🐍 session_state_manager.py  # 會話狀態
│   │       │
│   │       ├─📁 ui_panel/              # UI 面板
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 dashboard.py           # 儀表板
│   │       │   └─🐍 server.py              # Web 伺服器
│   │       │
│   │       ├─🐍 __init__.py
│   │       ├─🐍 ai_ui_schemas.py       # UI 資料結構
│   │       ├─🐍 app.py                 # 主應用程式
│   │       └─🐍 schemas.py             # 核心資料結構
│   │
│   ├─📁 function/                      # 漏洞檢測模組 (53 files, 9,864 lines) ⭐⭐⭐
│   │   │
│   │   ├─📁 common/                    # 共用檢測邏輯
│   │   │   ├─🐍 __init__.py
│   │   │   ├─🐍 detection_config.py
│   │   │   └─🐍 unified_smart_detection_manager.py  # 統一檢測管理
│   │   │
│   │   ├─📁 function_sqli/             # 💉 SQL 注入檢測
│   │   │   ├─📁 aiva_func_sqli/
│   │   │   │   ├─📁 engines/           # 檢測引擎集合
│   │   │   │   │   ├─🐍 __init__.py
│   │   │   │   │   ├─🐍 boolean_detection_engine.py    # 布林盲注
│   │   │   │   │   ├─🐍 error_detection_engine.py      # 錯誤訊息
│   │   │   │   │   ├─🐍 oob_detection_engine.py        # 帶外檢測
│   │   │   │   │   ├─🐍 time_detection_engine.py       # 時間盲注
│   │   │   │   │   └─🐍 union_detection_engine.py      # UNION 注入
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 backend_db_fingerprinter.py        # DB 指紋
│   │   │   │   ├─🐍 config.py
│   │   │   │   ├─🐍 detection_models.py
│   │   │   │   ├─🐍 exceptions.py
│   │   │   │   ├─🐍 payload_wrapper_encoder.py         # Payload編碼
│   │   │   │   ├─🐍 result_binder_publisher.py
│   │   │   │   ├─🐍 schemas.py
│   │   │   │   ├─🐍 smart_detection_manager.py         # 智慧管理
│   │   │   │   ├─🐍 task_queue.py
│   │   │   │   ├─🐍 telemetry.py
│   │   │   │   ├─🐍 worker_legacy.py
│   │   │   │   └─🐍 worker.py                          # Worker 主程式
│   │   │   └─🐍 __init__.py
│   │   │
│   │   ├─📁 function_xss/              # 🔓 XSS 跨站腳本檢測
│   │   │   ├─📁 aiva_func_xss/
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 blind_xss_listener_validator.py    # 盲XSS
│   │   │   │   ├─🐍 dom_xss_detector.py                # DOM XSS
│   │   │   │   ├─🐍 payload_generator.py               # Payload生成
│   │   │   │   ├─🐍 result_publisher.py
│   │   │   │   ├─🐍 schemas.py
│   │   │   │   ├─🐍 stored_detector.py                 # 儲存型XSS
│   │   │   │   ├─🐍 task_queue.py
│   │   │   │   ├─🐍 traditional_detector.py            # ⚠️ 反射型(複雜度45)
│   │   │   │   └─🐍 worker.py                          # ⚠️ Worker(複雜度48)
│   │   │   └─🐍 __init__.py
│   │   │
│   │   ├─📁 function_ssrf/             # 🌐 SSRF 請求偽造檢測
│   │   │   ├─📁 aiva_func_ssrf/
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 enhanced_worker.py
│   │   │   │   ├─🐍 internal_address_detector.py       # 內部位址檢測
│   │   │   │   ├─🐍 oast_dispatcher.py                 # OAST 調度
│   │   │   │   ├─🐍 param_semantics_analyzer.py        # ⚠️ 參數分析(複雜度44)
│   │   │   │   ├─🐍 result_publisher.py
│   │   │   │   ├─🐍 schemas.py
│   │   │   │   ├─🐍 smart_ssrf_detector.py             # ⚠️ 智慧檢測(複雜度42)
│   │   │   │   └─🐍 worker.py                          # ⚠️ Worker(複雜度49)
│   │   │   └─🐍 __init__.py
│   │   │
│   │   └─📁 function_idor/             # 🔑 IDOR 不安全物件引用檢測
│   │       └─📁 aiva_func_idor/
│   │           ├─🐍 __init__.py
│   │           ├─🐍 cross_user_tester.py               # 跨使用者測試
│   │           ├─🐍 enhanced_worker.py
│   │           ├─🐍 resource_id_extractor.py           # 資源ID提取
│   │           ├─🐍 smart_idor_detector.py             # 智慧檢測
│   │           ├─🐍 vertical_escalation_tester.py      # 垂直提權
│   │           └─🐍 worker.py                          # Worker 主程式
│   │
│   ├─📁 integration/                   # 整合層模組 (32 files, 3,174 lines) ⭐
│   │   ├─📁 aiva_integration/
│   │   │   ├─📁 analysis/              # 分析子系統
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 compliance_policy_checker.py      # 合規檢查
│   │   │   │   ├─🐍 risk_assessment_engine.py         # 風險評估
│   │   │   │   └─🐍 vuln_correlation_analyzer.py      # 漏洞關聯
│   │   │   │
│   │   │   ├─📁 config_template/      # 配置範本
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   └─🐍 config_template_manager.py
│   │   │   │
│   │   │   ├─📁 middlewares/          # 中介軟體
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   └─🐍 rate_limiter.py
│   │   │   │
│   │   │   ├─📁 observability/        # 可觀測性
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   └─🐍 metrics.py
│   │   │   │
│   │   │   ├─📁 perf_feedback/        # 性能反饋
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 improvement_suggestion_generator.py
│   │   │   │   └─🐍 scan_metadata_analyzer.py
│   │   │   │
│   │   │   ├─📁 reception/            # 資料接收層
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 data_reception_layer.py
│   │   │   │   └─🐍 sql_result_database.py
│   │   │   │
│   │   │   ├─📁 reporting/            # 報告生成
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   ├─🐍 formatter_exporter.py             # 格式化匯出
│   │   │   │   ├─🐍 report_content_generator.py       # 內容生成
│   │   │   │   └─🐍 report_template_selector.py       # 範本選擇
│   │   │   │
│   │   │   ├─📁 security/             # 安全層
│   │   │   │   ├─🐍 __init__.py
│   │   │   │   └─🐍 auth.py                           # 認證授權
│   │   │   │
│   │   │   ├─🐍 __init__.py
│   │   │   ├─🐍 app.py                                # 主應用
│   │   │   └─🐍 settings.py                           # 設定
│   │   │
│   │   ├─📁 alembic/                  # 資料庫遷移
│   │   │   ├─📁 versions/
│   │   │   │   └─🐍 001_initial_schema.py
│   │   │   └─🐍 env.py
│   │   │
│   │   ├─📁 api_gateway/              # API 閘道
│   │   │   └─📁 api_gateway/
│   │   │       └─🐍 app.py
│   │   │
│   │   └─📄 alembic.ini               # Alembic 配置
│   │
│   ├─📁 scan/                          # 掃描引擎模組 (28 files, 7,163 lines) ⭐⭐
│   │   └─📁 aiva_scan/
│   │       ├─📁 core_crawling_engine/  # 爬蟲核心
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 http_client_hi.py              # HTTP客戶端
│   │       │   ├─🐍 static_content_parser.py       # 靜態解析
│   │       │   └─🐍 url_queue_manager.py           # URL佇列
│   │       │
│   │       ├─📁 dynamic_engine/        # 動態引擎
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 dynamic_content_extractor.py   # ⚠️ 動態提取(複雜度54)
│   │       │   ├─🐍 example_browser_pool.py
│   │       │   ├─🐍 example_extractor.py
│   │       │   ├─🐍 example_usage.py
│   │       │   ├─🐍 headless_browser_pool.py       # ⚠️ 瀏覽器池(複雜度48)
│   │       │   └─🐍 js_interaction_simulator.py    # JS 互動模擬
│   │       │
│   │       ├─📁 info_gatherer/         # 資訊收集
│   │       │   ├─🐍 __init__.py
│   │       │   ├─🐍 javascript_source_analyzer.py  # JS 源碼分析
│   │       │   ├─🐍 passive_fingerprinter.py       # 被動指紋
│   │       │   └─🐍 sensitive_info_detector.py     # 敏感資訊
│   │       │
│   │       ├─🐍 __init__.py
│   │       ├─🐍 authentication_manager.py          # 認證管理
│   │       ├─🐍 config_control_center.py           # 配置中心
│   │       ├─🐍 fingerprint_manager.py             # 指紋管理
│   │       ├─🐍 header_configuration.py            # Header 配置
│   │       ├─🐍 scan_context.py                    # 掃描上下文
│   │       ├─🐍 scan_orchestrator_new.py           # 新協調器
│   │       ├─🐍 scan_orchestrator_old.py           # 舊協調器
│   │       ├─🐍 scan_orchestrator.py               # 協調器
│   │       ├─🐍 schemas.py                         # 資料結構
│   │       ├─🐍 scope_manager.py                   # ⚠️ 範圍管理(複雜度36)
│   │       ├─🐍 strategy_controller.py             # 策略控制
│   │       └─🐍 worker.py                          # Worker 主程式
│   │
│   └─🐍 __init__.py
│
├─📁 tools/                             # 開發工具
│   ├─🐍 analyze_codebase.py            # ⭐ 程式碼分析工具
│   ├─🐍 find_non_cp950_filtered.py     # 編碼檢查
│   ├─🐍 markdown_check.py              # Markdown 檢查
│   ├─🐍 py2mermaid.py                  # Mermaid 圖生成
│   ├─🐍 replace_emoji.py               # Emoji 替換
│   ├─🐍 replace_non_cp950.py           # 字元替換
│   ├─🐍 update_imports.py              # 導入更新
│   ├─🐍 test_tools.py                  # 工具測試
│   ├─📄 README.md                      # 工具文檔
│   └─📄 non_cp950_filtered_report.txt  # 編碼報告
│
├─📁 .vscode/                           # VS Code 配置
│   ├─⚙️ settings.json
│   ├─⚙️ launch.json
│   ├─⚙️ tasks.json
│   └─⚙️ extensions.json
│
├─🐍 __init__.py                        # 根模組初始化
├─📄 .editorconfig                      # 編輯器配置
├─📄 .env                               # 環境變數
├─📄 .env.example                       # 環境變數範例
├─📄 .gitignore                         # Git 忽略規則
├─🔧 .pre-commit-config.yaml            # Pre-commit hooks
├─📄 .pylintrc                          # Pylint 配置
├─📄 mypy.ini                           # Mypy 配置
├─📄 pyproject.toml                     # Python 專案配置 ⭐
├─⚙️ pyrightconfig.json                 # Pyright 配置
├─📄 ruff.toml                          # Ruff 配置
│
├─📝 ARCHITECTURE_REPORT.md             # 架構報告
├─📝 CODE_ANALYSIS_REPORT_20251013.md   # 程式碼分析
├─📝 ANALYSIS_EXECUTION_SUMMARY.md      # 執行摘要
├─📝 ANALYSIS_REPORTS_INDEX.md          # 報告索引
├─📝 COMPREHENSIVE_PROJECT_ANALYSIS.md  # ⭐ 本報告
├─📝 CORE_MODULE_ANALYSIS.md            # 核心模組分析
├─📝 DATA_CONTRACT.md                   # 資料契約
├─📝 QUICK_START.md                     # 快速開始
├─📝 SCAN_ENGINE_IMPROVEMENT_REPORT.md  # 掃描引擎改進
│
├─🐍 demo_bio_neuron_agent.py           # AI Agent 示範
├─🐍 demo_ui_panel.py                   # UI 面板示範
│
├─⚡ generate_clean_tree.ps1            # 生成目錄樹
├─⚡ generate_project_report.ps1        # 生成專案報告
├─⚡ generate_stats.ps1                 # 生成統計
├─⚡ init_go_deps.ps1                   # Go 依賴初始化
├─⚡ setup_multilang.ps1                # 多語言設置
├─⚡ setup_env.bat                      # 環境設置
├─⚡ start_all_multilang.ps1            # 啟動所有服務
├─⚡ start_all.ps1                      # 啟動服務
├─⚡ start_dev.bat                      # 開發模式啟動
├─⚡ stop_all_multilang.ps1             # 停止所有服務
└─⚡ stop_all.ps1                       # 停止服務

圖例說明:
  ⭐  重要模組/檔案
  ⭐⭐  核心模組
  ⭐⭐⭐  最大/最複雜模組
  ⚠️  高複雜度檔案 (需重構)
  🧠  AI 相關
  💉  SQL 注入相關
  🔓  XSS 相關
  🌐  SSRF 相關
  🔑  IDOR 相關
```

> **統計摘要**:
>
> - 總目錄數: ~50+
> - 總 Python 檔案: 169
> - 核心服務模組: 5 個 (aiva_common, core, function, integration, scan)
> - 漏洞檢測類型: 4 種 (SQLi, XSS, SSRF, IDOR)

═══════════════════════════════════════════════════════════════════════════════
🌳 完整專案樹狀架構圖
═══════════════════════════════════════════════════════════════════════════════

**生成時間**: 2025-10-13 13:21
**專案根目錄**: `/workspaces/AIVA`
**總檔案數**: 329 個
**總目錄數**: 94 個

**架構說明**:

- 🐍 Python 核心服務 (services/) - 主要業務邏輯
- 🔷 Go/Rust 多語言模組 (services/function/) - 高性能檢測
- 🛠️ 工具腳本 (tools/) - 程式碼分析與生成
- 📊 輸出報告 (_out/) - 自動生成的分析結果
- 🐳 容器化 (docker/) - 部署配置

## 完整專案樹狀結構

```plaintext
/workspaces/AIVA
├── _out/                          # 分析輸出目錄
│   ├── analysis/                  # 分析報告
│   └── [其他輸出文件...]
├── docker/                        # 容器化配置
│   ├── initdb/                    # 資料庫初始化
│   └── [Docker 配置檔案...]
├── docs/                          # 文檔目錄
│   └── diagrams/                  # 架構圖檔案
├── services/                      # 核心服務模組
│   ├── aiva_common/               # 共用組件
│   │   └── utils/                 # 工具函數
│   ├── core/                      # 核心引擎
│   │   └── aiva_core/             # AI 驅動核心
│   │       ├── ai_engine/         # AI 引擎
│   │       ├── analysis/          # 分析模組
│   │       ├── execution/         # 執行引擎
│   │       ├── ingestion/         # 資料接收
│   │       ├── output/            # 結果輸出
│   │       ├── state/             # 狀態管理
│   │       └── ui_panel/          # UI 面板
│   ├── function/                  # 漏洞檢測模組
│   │   ├── common/                # 共用檢測邏輯
│   │   ├── function_authn_go/     # Go 身份驗證檢測
│   │   ├── function_cspm_go/      # Go 雲端安全
│   │   ├── function_idor/         # IDOR 檢測
│   │   ├── function_sast_rust/    # Rust 靜態分析
│   │   ├── function_sca_go/       # Go 軟體組成分析
│   │   ├── function_sqli/         # SQL 注入檢測
│   │   ├── function_ssrf/         # SSRF 檢測
│   │   ├── function_ssrf_go/      # Go SSRF 檢測
│   │   └── function_xss/          # XSS 檢測
│   ├── integration/               # 整合層
│   │   ├── aiva_integration/      # 整合服務
│   │   ├── alembic/               # 資料庫遷移
│   │   └── api_gateway/           # API 閘道
│   └── scan/                      # 掃描引擎
│       ├── aiva_scan/             # Python 掃描引擎
│       ├── aiva_scan_node/        # Node.js 掃描服務
│       └── info_gatherer_rust/    # Rust 資訊收集器
├── tools/                         # 開發工具
└── [根目錄配置檔案...]            # 專案配置檔案

📊 統計摘要: 94 個目錄, 329 個檔案
```

### 架構特點說明

- **多語言架構**: Python(81%) + Go/Rust/TypeScript 混合架構
- **模組化設計**: 核心引擎、掃描引擎、檢測模組清晰分層
- **微服務架構**: 各模組獨立部署，通過訊息佇列通訊
- **AI 驅動**: 核心採用生物神經網路啟發的 AI 引擎
- **高性能**: 關鍵檢測模組使用 Go/Rust 實現
│   ├── analysis
│   │   ├── analysis_report_20251013_115525.json
│   │   ├── analysis_report_20251013_115525.txt
│   │   ├── analysis_report_20251013_120144.json
│   │   ├── analysis_report_20251013_120144.txt
│   │   ├── analysis_report_20251013_121623.json
│   │   ├── analysis_report_20251013_121623.txt
│   │   ├── analysis_report_20251013_124406.json
│   │   ├── analysis_report_20251013_124406.txt
│   │   ├── analysis_report_20251013_125408.json
│   │   └── analysis_report_20251013_125408.txt
│   ├── ARCHITECTURE_DIAGRAMS.md
│   ├── PROJECT_REPORT.txt
│   ├── ext_counts.csv
│   ├── loc_by_ext.csv
│   ├── tree.html
│   ├── tree.md
│   ├── tree.mmd
│   ├── tree_ascii.txt
│   ├── tree_ascii_correct.txt
│   ├── tree_ascii_new.txt
│   ├── tree_header.txt
│   └── tree_unicode.txt
├── docker
│   ├── initdb
│   │   └── 001_schema.sql
│   ├── Dockerfile.integration
│   ├── docker-compose.production.yml
│   ├── docker-compose.yml
│   └── entrypoint.integration.sh
├── docs
│   └── diagrams
│       ├── Function____init__.mmd
│       ├── Function___add_chunk.mmd
│       ├── Function___create_input_vector.mmd
│       ├── Function___extract_keywords.mmd
│       ├── Function___get_mode_display.mmd
│       ├── Function___index_file.mmd
│       ├── Function___init_ai_agent.mmd
│       ├── Function___softmax.mmd
│       ├── Function__analyze_code.mmd
│       ├── Function__check_confidence.mmd
│       ├── Function__create_scan_task.mmd
│       ├── Function__detect_vulnerability.mmd
│       ├── Function__forward.mmd
│       ├── Function__get_ai_history.mmd
│       ├── Function__get_chunk_count.mmd
│       ├── Function__get_detections.mmd
│       ├── Function__get_file_content.mmd
│       ├── Function__get_stats.mmd
│       ├── Function__get_tasks.mmd
│       ├── Function__index_codebase.mmd
│       ├── Function__invoke.mmd
│       ├── Function__read_code.mmd
│       ├── Function__search.mmd
│       └── Module.mmd
├── services
│   ├── aiva_common
│   │   ├── utils
│   │   │   ├── dedup
│   │   │   │   ├── **init**.py
│   │   │   │   └── dedupe.py
│   │   │   ├── network
│   │   │   │   ├── **init**.py
│   │   │   │   ├── backoff.py
│   │   │   │   └── ratelimit.py
│   │   │   ├── **init**.py
│   │   │   ├── ids.py
│   │   │   └── logging.py
│   │   ├── **init**.py
│   │   ├── config.py
│   │   ├── enums.py
│   │   ├── mq.py
│   │   ├── py.typed
│   │   └── schemas.py
│   ├── core
│   │   └── aiva_core
│   │       ├── ai_engine
│   │       │   ├── **init**.py
│   │       │   ├── bio_neuron_core.py
│   │       │   ├── bio_neuron_core.py.backup
│   │   │   ├── bio_neuron_core_v2.py
│   │   │   ├── knowledge_base.py
│   │   │   ├── knowledge_base.py.backup
│   │   │   └── tools.py
│   │       ├── ai_model
│   │       │   └── train_classifier.py
│   │       ├── analysis
│   │       │   ├── **init**.py
│   │       │   ├── dynamic_strategy_adjustment.py
│   │       │   ├── initial_surface.py
│   │       │   └── strategy_generator.py
│   │       ├── execution
│   │       │   ├── **init**.py
│   │       │   ├── execution_status_monitor.py
│   │       │   ├── task_generator.py
│   │       │   └── task_queue_manager.py
│   │       ├── ingestion
│   │       │   ├── **init**.py
│   │       │   └── scan_module_interface.py
│   │       ├── output
│   │       │   ├── **init**.py
│   │       │   └── to_functions.py
│   │       ├── state
│   │       │   ├── **init**.py
│   │       │   └── session_state_manager.py
│   │       ├── ui_panel
│   │       │   ├── **init**.py
│   │       │   ├── dashboard.py
│   │       │   ├── dashboard.py.backup
│   │       │   ├── improved_ui.py
│   │   │   ├── server.py
│   │   │   ├── server.py.backup
│   │       ├── **init**.py
│   │       ├── ai_ui_schemas.py
│   │       ├── app.py
│   │       └── schemas.py
│   ├── function
│   │   ├── common
│   │   │   ├── **init**.py
│   │   │   ├── detection_config.py
│   │   │   └── unified_smart_detection_manager.py
│   │   ├── function_authn_go
│   │   │   ├── cmd
│   │   │   │   └── worker
│   │   │   │       └── main.go
│   │   │   ├── internal
│   │   │   │   ├── brute_force
│   │   │   │   │   └── brute_forcer.go
│   │   │   │   ├── token_test
│   │   │   │   │   └── token_analyzer.go
│   │   │   │   └── weak_config
│   │   │   │       └── config_tester.go
│   │   │   ├── pkg
│   │   │   │   ├── messaging
│   │   │   │   │   ├── consumer.go
│   │   │   │   │   └── publisher.go
│   │   │   │   └── models
│   │   │   │       └── models.go
│   │   │   └── go.mod
│   │   ├── function_cspm_go
│   │   │   ├── cmd
│   │   │   │   └── worker
│   │   │   │       └── main.go
│   │   │   ├── internal
│   │   │   │   └── scanner
│   │   │   │       └── cspm_scanner.go
│   │   │   ├── pkg
│   │   │   │   ├── messaging
│   │   │   │   │   ├── consumer.go
│   │   │   │   │   └── publisher.go
│   │   │   │   └── models
│   │   │   │       └── models.go
│   │   │   └── go.mod
│   │   ├── function_idor
│   │   │   └── aiva_func_idor
│   │   │       ├── **init**.py
│   │   │       ├── bfla_tester.py
│   │   │       ├── cross_user_tester.py
│   │   │       ├── enhanced_worker.py
│   │   │       ├── mass_assignment_tester.py
│   │   │       ├── resource_id_extractor.py
│   │   │       ├── smart_idor_detector.py
│   │   │       ├── vertical_escalation_tester.py
│   │   │       └── worker.py
│   │   ├── function_sast_rust
│   │   │   ├── src
│   │   │   │   ├── analyzers.rs
│   │   │   │   ├── main.rs
│   │   │   │   ├── models.rs
│   │   │   │   ├── parsers.rs
│   │   │   │   ├── rules.rs
│   │   │   │   └── worker.rs
│   │   │   └── Cargo.toml
│   │   ├── function_sca_go
│   │   │   ├── cmd
│   │   │   │   └── worker
│   │   │   │       └── main.go
│   │   │   ├── internal
│   │   │   │   └── scanner
│   │   │   │       └── sca_scanner.go
│   │   │   ├── pkg
│   │   │   │   ├── messaging
│   │   │   │   │   └── publisher.go
│   │   │   │   └── models
│   │   │   │       └── models.go
│   │   │   ├── README.md
│   │   │   └── go.mod
│   │   ├── function_sqli
│   │   │   ├── aiva_func_sqli
│   │   │   │   ├── engines
│   │   │   │   │   ├── **init**.py
│   │   │   │   │   ├── boolean_detection_engine.py
│   │   │   │   │   ├── error_detection_engine.py
│   │   │   │   │   ├── oob_detection_engine.py
│   │   │   │   │   ├── time_detection_engine.py
│   │   │   │   │   └── union_detection_engine.py
│   │   │   │   ├── **init**.py
│   │   │   │   ├── backend_db_fingerprinter.py
│   │   │   │   ├── config.py
│   │   │   │   ├── detection_models.py
│   │   │   │   ├── exceptions.py
│   │   │   │   ├── payload_wrapper_encoder.py
│   │   │   │   ├── result_binder_publisher.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── smart_detection_manager.py
│   │   │   │   ├── task_queue.py
│   │   │   │   ├── telemetry.py
│   │   │   │   ├── worker.py
│   │   │   │   └── worker_legacy.py
│   │   │   └── **init**.py
│   │   ├── function_ssrf
│   │   │   ├── aiva_func_ssrf
│   │   │   │   ├── **init**.py
│   │   │   │   ├── enhanced_worker.py
│   │   │   │   ├── internal_address_detector.py
│   │   │   │   ├── oast_dispatcher.py
│   │   │   │   ├── param_semantics_analyzer.py
│   │   │   │   ├── result_publisher.py
│   │   │   │   ├── schemas.py
│   │   │   │   ├── smart_ssrf_detector.py
│   │   │   │   └── worker.py
│   │   │   └── **init**.py
│   │   ├── function_ssrf_go
│   │   │   ├── cmd
│   │   │   │   └── worker
│   │   │   │       └── main.go
│   │   │   ├── internal
│   │   │   │   └── detector
│   │   │   │       └── ssrf.go
│   │   │   ├── README.md
│   │   │   └── go.mod
│   │   └── function_xss
│   │       ├── aiva_func_xss
│   │       │   ├── **init**.py
│   │       │   ├── blind_xss_listener_validator.py
│   │       │   ├── dom_xss_detector.py
│   │   │   ├── payload_generator.py
│   │   │   ├── result_publisher.py
│   │   │   ├── schemas.py
│   │   │   ├── stored_detector.py
│   │   │   ├── task_queue.py
│   │   │   ├── traditional_detector.py
│   │   │   └── worker.py
│   │       └── **init**.py
│   ├── integration
│   │   ├── aiva_integration
│   │   │   ├── analysis
│   │   │   │   ├── **init**.py
│   │   │   │   ├── compliance_policy_checker.py
│   │   │   │   ├── risk_assessment_engine.py
│   │   │   │   └── vuln_correlation_analyzer.py
│   │   │   ├── attack_path_analyzer
│   │   │   │   ├── README.md
│   │   │   │   ├── **init**.py
│   │   │   │   ├── engine.py
│   │   │   │   ├── graph_builder.py
│   │   │   │   └── visualizer.py
│   │   │   ├── config_template
│   │   │   │   ├── **init**.py
│   │   │   │   └── config_template_manager.py
│   │   │   ├── middlewares
│   │   │   │   ├── **init**.py
│   │   │   │   └── rate_limiter.py
│   │   │   ├── observability
│   │   │   │   ├── **init**.py
│   │   │   │   └── metrics.py
│   │   │   ├── perf_feedback
│   │   │   │   ├── **init**.py
│   │   │   │   ├── improvement_suggestion_generator.py
│   │   │   │   └── scan_metadata_analyzer.py
│   │   │   ├── reception
│   │   │   │   ├── **init**.py
│   │   │   │   ├── data_reception_layer.py
│   │   │   │   └── sql_result_database.py
│   │   │   ├── reporting
│   │   │   │   ├── **init**.py
│   │   │   │   ├── formatter_exporter.py
│   │   │   │   ├── report_content_generator.py
│   │   │   │   └── report_template_selector.py
│   │   │   ├── security
│   │   │   │   ├── **init**.py
│   │   │   │   └── auth.py
│   │   │   ├── threat_intel
│   │   │   │   └── **init**.py
│   │   │   ├── **init**.py
│   │   │   ├── app.py
│   │   │   └── settings.py
│   │   ├── alembic
│   │   │   ├── versions
│   │   │   │   └── 001_initial_schema.py
│   │   │   └── env.py
│   │   ├── api_gateway
│   │   │   └── api_gateway
│   │   │       └── app.py
│   │   └── alembic.ini
│   ├── scan
│   │   ├── aiva_scan
│   │   │   ├── core_crawling_engine
│   │   │   │   ├── **init**.py
│   │   │   │   ├── http_client_hi.py
│   │   │   │   ├── static_content_parser.py
│   │   │   │   └── url_queue_manager.py
│   │   │   ├── dynamic_engine
│   │   │   │   ├── **init**.py
│   │   │   │   ├── dynamic_content_extractor.py
│   │   │   │   ├── example_browser_pool.py
│   │   │   │   ├── example_extractor.py
│   │   │   │   ├── example_usage.py
│   │   │   │   ├── example_usage.py.backup
│   │   │   │   ├── headless_browser_pool.py
│   │   │   │   └── js_interaction_simulator.py
│   │   │   ├── info_gatherer
│   │   │   │   ├── **init**.py
│   │   │   │   ├── javascript_source_analyzer.py
│   │   │   │   ├── passive_fingerprinter.py
│   │   │   │   └── sensitive_info_detector.py
│   │   │   ├── **init**.py
│   │   │   ├── authentication_manager.py
│   │   │   ├── config_control_center.py
│   │   │   ├── fingerprint_manager.py
│   │   │   ├── header_configuration.py
│   │   │   ├── scan_context.py
│   │   │   ├── scan_orchestrator.py
│   │   │   ├── scan_orchestrator_new.py
│   │   │   ├── scan_orchestrator_old.py
│   │   │   ├── schemas.py
│   │   │   ├── scope_manager.py
│   │   │   ├── strategy_controller.py
│   │   │   └── worker.py
│   │   ├── aiva_scan_node
│   │   │   ├── src
│   │   │   │   ├── services
│   │   │   │   │   └── scan-service.ts
│   │   │   │   ├── utils
│   │   │   │   │   └── logger.ts
│   │   │   │   └── index.ts
│   │   │   ├── README.md
│   │   │   ├── package.json
│   │   │   └── tsconfig.json
│   │   └── info_gatherer_rust
│   │       ├── src
│   │       │   ├── git_history_scanner.rs
│   │   │   ├── main.rs
│   │   ├── scanner.rs
│   │   ├── secret_detector.rs
│   │   ├── Cargo.toml
│   │   └── README.md
│   └── **init**.py
├── tools
│   ├── README.md
│   ├── analyze_codebase.py
│   ├── find_non_cp950_filtered.py
│   ├── generate_mermaid_diagrams.py
│   ├── markdown_check.py
│   │   ├── markdown_check_out.txt
│   │   ├── non_cp950_filtered_report.txt
│   │   ├── py2mermaid.py
│   │   ├── replace_emoji.py
│   │   ├── replace_non_cp950.py
│   │   ├── test_tools.py
│   │   └── update_imports.py
├── .editorconfig
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .pylintrc
├── ANALYSIS_EXECUTION_SUMMARY.md
├── ANALYSIS_REPORTS_INDEX.md
├── ANALYSIS_TOOLS_SUMMARY.md
├── ARCHITECTURE_REPORT.md
├── CODE_ANALYSIS_REPORT_20251013.md
├── CODE_ANALYSIS_UPGRADE_REPORT.md
├── COMPREHENSIVE_PROJECT_ANALYSIS.md
├── COMPREHENSIVE_ROADMAP.md
├── CORE_MODULE_ANALYSIS.md
├── DATA_CONTRACT.md
├── DATA_CONTRACT_ANALYSIS.md
├── DATA_CONTRACT_UPDATE.md
├── DEPENDENCY_ANALYSIS.md
├── DEPENDENCY_STATUS_VERIFIED.md
├── FINAL_ERROR_ANALYSIS.md
├── FINAL_FIX_REPORT.md
├── FUNCTION_MODULE_ENHANCEMENT_ANALYSIS.md
├── IMPLEMENTATION_GUIDE.md
├── INSTALLATION_COMPLETE.md
├── MODULE_CLASSIFICATION_QUICK_REFERENCE.md
├── MULTI_LANGUAGE_ARCHITECTURE_PROPOSAL.md
├── OFFICIAL_COMPLIANCE_VALIDATION_REPORT.md
├── P0_MODULES_ERROR_ANALYSIS.md
├── P0_MODULE_SCRIPT_OUTPUT_SUMMARY.md
├── QUICK_GUIDE_ANALYSIS_TOOLS.md
├── QUICK_REFERENCE_MULTILANG.md
├── QUICK_START.md
├── RUN_SCRIPTS.md
├── SCAN_ENGINE_IMPROVEMENT_REPORT.md
├── SCHEMA_NAMING_ISSUES.md
├── TYPESCRIPT_FIX_REPORT.md
├── USAGE_GUIDE.md
├── **init**.py
├── c:\D\E\AIVA\AIVA-main\tools\non_cp950_filtered_report.txt
├── check_status.ps1
├── demo_bio_neuron_agent.py
├── demo_ui_panel.py
├── generate_clean_tree.ps1
├── generate_project_report.ps1
├── generate_project_report.sh
├── generate_stats.ps1
├── init_go_deps.ps1
├── mypy.ini
├── pyproject.toml
├── pyrightconfig.json
├── ruff.toml
├── setup_env.bat
├── setup_multilang.ps1
├── start_all.ps1
├── start_all_multilang.ps1
├── start_dev.bat
├── stop_all.ps1
├── stop_all_multilang.ps1
└── test_scan.ps1

94 directories, 329 files

```

═══════════════════════════════════════════════════════════════════════════════

## 核心模組結構

### 1. 🧠 核心引擎 (Core Module)

**路徑**: `services/core/aiva_core/`

| 子模組 | 檔案數 | 行數 | 函數數 | 類別數 | 複雜度 | 功能說明 |
|--------|--------|------|--------|--------|--------|----------|
| **ai_engine** | 6 | 1,247 | 42 | 18 | 98 | AI 驅動的漏洞分析引擎 |
| **analysis** | 3 | 188 | 15 | 8 | 67 | 攻擊面分析與策略生成 |
| **execution** | 3 | 445 | 28 | 12 | 45 | 任務執行與狀態監控 |
| **ui_panel** | 4 | 523 | 18 | 9 | 38 | Web UI 儀表板 |
| **ingestion** | 1 | 156 | 8 | 3 | 12 | 掃描模組介面 |
| **output** | 1 | 289 | 12 | 4 | 15 | 結果輸出處理 |
| **state** | 1 | 198 | 5 | 1 | 10 | 會話狀態管理 |
| **其他** | 9 | 1,804 | - | - | - | 配置與工具 |

**總計**: 28 檔案, 4,850 行, 128 函數, 55 類別, 複雜度 285

#### 核心功能說明

##### 🤖 AI Engine (AI 引擎)

- **bio_neuron_core_v2.py** (最新版本)
  - 生物神經元啟發的 AI 核心
  - 動態學習與自適應能力
  - 漏洞模式識別與預測

- **knowledge_base.py**
  - 知識庫管理系統
  - 向量化索引與語義搜索
  - 程式碼特徵提取

- **tools.py** (複雜度: 38)
  - AI 工具函數集合
  - 包含各種分析工具
  - **建議**: 需要重構降低複雜度

##### 📊 Analysis (分析模組)

- **initial_surface.py**
  - 初始攻擊面分析
  - URL 和端點發現
  - 參數識別與分類

- **strategy_generator.py**
  - 動態策略生成器
  - 基於目標特徵調整測試策略
  - 優先級排序演算法

- **dynamic_strategy_adjustment.py** (複雜度: 34)
  - 即時策略調整
  - 根據掃描結果動態優化
  - 自適應測試深度

##### ⚙️ Execution (執行模組)

- **task_generator.py**
  - 測試任務生成器
  - 將策略轉換為具體任務

- **task_queue_manager.py**
  - 任務佇列管理
  - 優先級調度
  - 負載平衡

- **execution_status_monitor.py**
  - 執行狀態監控
  - 進度追蹤
  - 異常檢測與恢復

##### 🖥️ UI Panel (使用者介面)

- **server.py**
  - FastAPI Web 伺服器
  - RESTful API 端點
  - WebSocket 即時通訊

- **dashboard.py**
  - 儀表板視圖
  - 即時統計與圖表
  - 掃描結果展示

---

### 2. 🔍 掃描引擎 (Scan Module)

**路徑**: `services/scan/aiva_scan/`

| 子模組 | 檔案數 | 行數 | 函數數 | 類別數 | 複雜度 | 功能說明 |
|--------|--------|------|--------|--------|--------|----------|
| **core_crawling_engine** | 3 | 1,189 | 42 | 15 | 85 | 靜態爬蟲核心引擎 |
| **dynamic_engine** | 7 | 2,843 | 65 | 28 | 182 | 動態內容提取引擎 |
| **info_gatherer** | 3 | 1,647 | 48 | 18 | 102 | 資訊收集模組 |
| **其他** | 15 | 1,484 | 44 | - | 46 | 配置與協調 |

**總計**: 28 檔案, 7,163 行, 199 函數, 53 類別, 複雜度 415

#### 掃描引擎功能說明

##### 🕷️ Core Crawling Engine (爬蟲引擎)

- **http_client_hi.py**
  - 高性能 HTTP 客戶端
  - 支援 HTTP/2 和 HTTP/3
  - 智慧重試與錯誤處理
  - 指紋偽裝與反檢測

- **static_content_parser.py**
  - 靜態內容解析器
  - HTML/JavaScript/CSS 解析
  - 連結和資源提取
  - 表單發現與參數識別

- **url_queue_manager.py**
  - URL 佇列管理
  - 去重與優先級排序
  - 爬取深度控制
  - 範圍限制檢查

##### 🎭 Dynamic Engine (動態引擎)

- **headless_browser_pool.py** (複雜度: 48 ⚠️)
  - 無頭瀏覽器池管理
  - Playwright/Puppeteer 整合
  - 瀏覽器實例復用
  - **問題**: 複雜度過高，需重構

- **dynamic_content_extractor.py** (複雜度: 54 ⚠️)
  - 動態內容提取器
  - JavaScript 渲染與執行
  - AJAX 請求攔截
  - 單頁應用 (SPA) 支援
  - **問題**: 最複雜的掃描模組，急需重構

- **js_interaction_simulator.py** (複雜度: 30)
  - JavaScript 互動模擬
  - 事件觸發與監聽
  - 表單自動填充
  - 點擊和導航模擬

##### 🔎 Info Gatherer (資訊收集)

- **javascript_source_analyzer.py** (複雜度: 30)
  - JavaScript 原始碼分析
  - API 端點發現
  - 敏感資訊提取
  - 第三方庫識別

- **passive_fingerprinter.py**
  - 被動指紋識別
  - 技術棧檢測
  - 伺服器版本識別
  - 框架與 CMS 判定

- **sensitive_info_detector.py** (複雜度: 36)
  - 敏感資訊檢測器
  - API 金鑰與 Token 搜尋
  - 憑證和密碼發現
  - PII 資料識別

##### 🎯 Orchestration (協調層)

- **scan_orchestrator.py**
  - 掃描協調器
  - 引擎調度與協調
  - 任務分配與管理

- **scope_manager.py** (複雜度: 36)
  - 範圍管理器
  - 域名和路徑限制
  - 黑白名單管理
  - 規則引擎

- **strategy_controller.py**
  - 策略控制器
  - 掃描策略應用
  - 參數調整

---

### 3. 🛡️ 漏洞檢測模組 (Function Modules)

**路徑**: `services/function/`

| 模組 | 檔案數 | 行數 | 函數數 | 類別數 | 複雜度 | 檢測類型 |
|------|--------|------|--------|--------|--------|----------|
| **function_sqli** | 15 | 3,245 | 78 | 42 | 298 | SQL 注入 |
| **function_xss** | 10 | 2,156 | 54 | 28 | 189 | 跨站腳本 |
| **function_ssrf** | 9 | 2,387 | 52 | 26 | 186 | 伺服器端請求偽造 |
| **function_idor** | 7 | 1,876 | 38 | 19 | 98 | 不安全直接物件引用 |
| **common** | 2 | 410 | 6 | 4 | 31 | 共用檢測邏輯 |

**總計**: 53 檔案, 9,864 行, 228 函數, 119 類別, 複雜度 795

#### 漏洞檢測引擎架構

##### 💉 SQL Injection (SQL 注入檢測)

**路徑**: `function/function_sqli/aiva_func_sqli/`

**檢測引擎** (`engines/` 子目錄):

1. **boolean_detection_engine.py**
   - 布林盲注檢測
   - True/False 回應差異分析
   - 自動化 payload 生成

2. **error_detection_engine.py**
   - 錯誤訊息檢測
   - 資料庫錯誤模式匹配
   - 資訊洩漏識別

3. **time_detection_engine.py**
   - 時間盲注檢測
   - 回應時間分析
   - 統計顯著性驗證

4. **union_detection_engine.py**
   - UNION 查詢注入
   - 欄位數量探測
   - 資料提取

5. **oob_detection_engine.py**
   - 帶外 (Out-of-Band) 檢測
   - DNS/HTTP 外帶
   - OAST 整合

**核心模組**:

- **backend_db_fingerprinter.py**
  - 資料庫類型指紋識別
  - 自動選擇適合的 payload
  - 支援: MySQL, PostgreSQL, MSSQL, Oracle 等

- **smart_detection_manager.py**
  - 智慧檢測管理器
  - 根據上下文選擇引擎
  - 結果聚合與驗證

- **payload_wrapper_encoder.py**
  - Payload 包裝與編碼
  - WAF 繞過技術
  - 多層編碼支援

##### 🔓 XSS (跨站腳本檢測)

**路徑**: `function/function_xss/aiva_func_xss/`

**檢測器類型**:

1. **traditional_detector.py** (複雜度: 45 ⚠️)
   - 反射型 XSS 檢測
   - 上下文感知 payload
   - DOM 結構分析
   - **問題**: 複雜度過高，需重構

2. **stored_detector.py** (複雜度: 33)
   - 儲存型 XSS 檢測
   - 多階段驗證
   - 持久化測試

3. **dom_xss_detector.py**
   - DOM 型 XSS 檢測
   - JavaScript 源碼分析
   - Sink 追蹤

4. **blind_xss_listener_validator.py**
   - 盲 XSS 檢測
   - 外帶監聽器
   - 回調驗證

**核心模組**:

- **payload_generator.py**
  - Payload 生成器
  - 上下文適應
  - 編碼變換
  - 繞過過濾器

- **worker.py** (複雜度: 48 ⚠️)
  - XSS 檢測 Worker
  - 任務調度
  - **問題**: 缺少文檔字串，複雜度高

##### 🌐 SSRF (伺服器端請求偽造檢測)

**路徑**: `function/function_ssrf/aiva_func_ssrf/`

**核心檢測器**:

1. **smart_ssrf_detector.py** (複雜度: 42 ⚠️)
   - 智慧 SSRF 檢測
   - 多種檢測技術整合
   - 自動繞過策略

2. **internal_address_detector.py** (複雜度: 36)
   - 內部地址存取檢測
   - 私有 IP 探測
   - 雲端元資料服務檢測

3. **param_semantics_analyzer.py** (複雜度: 44 ⚠️)
   - 參數語義分析
   - 識別可能的 SSRF 參數
   - URL/域名參數分類

**輔助模組**:

- **oast_dispatcher.py**
  - OAST (Out-of-Band) 調度器
  - 外帶服務整合
  - DNS/HTTP 回調管理

- **worker.py** (複雜度: 49 ⚠️)
  - SSRF 檢測 Worker
  - **問題**: 缺少文檔，複雜度過高，急需重構

##### 🔑 IDOR (不安全直接物件引用檢測)

**路徑**: `services/function/function_idor/aiva_func_idor/`

**檢測器類型**:

1. **smart_idor_detector.py**
   - 智慧 IDOR 檢測
   - 資源 ID 模式識別
   - 存取控制測試

2. **cross_user_tester.py**
   - 跨使用者存取測試
   - 水平越權檢測
   - 多使用者情境模擬

3. **vertical_escalation_tester.py**
   - 垂直提權測試
   - 權限升級檢測
   - 角色矩陣驗證

4. **resource_id_extractor.py**
   - 資源 ID 提取器
   - 識別可測試的資源
   - ID 模式學習

**工作流程**:

```

掃描 → 識別資源 → 提取 ID → 生成測試用例 → 執行檢測 → 驗證結果

```

##### 🔧 Common (共用模組)

**路徑**: `services/function/common/`

- **unified_smart_detection_manager.py** (複雜度: 31)
  - 統一智慧檢測管理器
  - 跨模組檢測協調
  - 結果聚合與關聯

- **detection_config.py**
  - 檢測配置管理
  - 參數調整
  - 規則定義

---

### 4. 🔗 整合層 (Integration Module)

**路徑**: `services/integration/aiva_integration/`

| 子模組 | 檔案數 | 行數 | 功能說明 |
|--------|--------|------|----------|
| **reception** | 3 | 456 | 資料接收與儲存 |
| **analysis** | 3 | 389 | 結果分析與關聯 |
| **reporting** | 3 | 523 | 報告生成與匯出 |
| **perf_feedback** | 2 | 287 | 性能反饋與建議 |
| **config_template** | 1 | 145 | 配置範本管理 |
| **observability** | 1 | 98 | 可觀測性與指標 |
| **security** | 1 | 76 | 身份驗證與授權 |
| **middlewares** | 1 | 123 | 中介軟體 |
| **其他** | 17 | 1,077 | API 與資料庫 |

**總計**: 32 檔案, 3,174 行, 96 函數, 24 類別

#### 整合層功能說明

##### 📥 Reception (接收層)

- **data_reception_layer.py**
  - 統一資料接收介面
  - 驗證與規範化
  - 佇列管理

- **sql_result_database.py**
  - 結果資料庫存取
  - SQL 注入結果專用儲存
  - 查詢與索引優化

##### 📊 Analysis (分析層)

- **vuln_correlation_analyzer.py**
  - 漏洞關聯分析
  - 識別攻擊鏈
  - 影響範圍評估

- **risk_assessment_engine.py**
  - 風險評估引擎
  - CVSS 評分
  - 業務影響分析

- **compliance_policy_checker.py**
  - 合規性檢查器
  - OWASP Top 10 對應
  - 法規要求驗證

##### 📑 Reporting (報告層)

- **report_content_generator.py**
  - 報告內容生成器
  - 多語言支援
  - 技術與高階報告

- **formatter_exporter.py**
  - 格式化與匯出
  - PDF, HTML, JSON, XML
  - 範本引擎整合

- **report_template_selector.py**
  - 報告範本選擇器
  - 自動範本匹配
  - 客製化支援

##### 🔄 Performance Feedback (性能反饋)

- **scan_metadata_analyzer.py**
  - 掃描元資料分析
  - 性能指標收集
  - 瓶頸識別

- **improvement_suggestion_generator.py**
  - 改進建議生成器
  - 自動優化建議
  - 最佳實踐推薦

##### 🛡️ Security (安全層)

- **auth.py**
  - 身份驗證
  - JWT Token 管理
  - 權限控制

##### ⚙️ API Gateway

- **api_gateway/app.py**
  - API 閘道
  - 路由管理
  - 請求代理與負載平衡

---

### 5. 🔧 公用工具 (Common Utilities)

**路徑**: `services/aiva_common/`

| 模組 | 行數 | 功能說明 |
|------|------|----------|
| **utils/network** | 659 | 網路工具 (限流、重試) |
| **utils/dedup** | 123 | 去重工具 |
| **schemas.py** | 456 | 資料結構定義 |
| **config.py** | 234 | 配置管理 |
| **mq.py** | 189 | 訊息佇列整合 |
| **enums.py** | 156 | 列舉定義 |
| **其他** | 143 | 其他工具 |

**總計**: 13 檔案, 1,960 行

#### 工具模組說明

##### 🌐 Network Utilities

- **ratelimit.py** (複雜度: 91 ⚠️⚠️⚠️)
  - 速率限制器
  - Token Bucket 演算法
  - 分散式限流支援
  - **嚴重問題**: 複雜度極高 (91)，最急需重構的檔案！

- **backoff.py**
  - 指數退避重試
  - 自適應重試策略
  - 錯誤分類處理

##### 🔄 Deduplication

- **dedupe.py**
  - 去重演算法
  - 布隆過濾器
  - 精確去重

##### 📋 Schemas

- **schemas.py**
  - Pydantic 模型定義
  - 資料驗證
  - 序列化與反序列化

##### ⚙️ Configuration

- **config.py**
  - 配置管理
  - 環境變數載入
  - 配置驗證

##### 📨 Message Queue

- **mq.py**
  - RabbitMQ/Redis 整合
  - 訊息發佈訂閱
  - 任務佇列

---

### 6. 🗄️ 資料層與部署 (Docker & Database)

**路徑**: `docker/`

#### Docker 配置

- **docker-compose.yml**
  - 開發環境配置
  - 服務編排
  - 網路與卷設定

- **docker-compose.production.yml**
  - 生產環境配置
  - 高可用性設定
  - 負載平衡

- **Dockerfile.integration**
  - 整合服務映像
  - 多階段建置
  - 優化層快取

#### 資料庫初始化

- **initdb/001_schema.sql** (178 行)
  - 資料庫架構定義
  - 表格與索引
  - 初始資料

---

### 7. 🛠️ 開發工具 (Tools)

**路徑**: `tools/`

| 工具 | 行數 | 功能 | 狀態 |
|------|------|------|------|
| **analyze_codebase.py** | 430 | 程式碼分析 | ✅ 已使用 |
| **py2mermaid.py** | ~300 | Mermaid 圖生成 | ✅ 可用 |
| **find_non_cp950_filtered.py** | ~150 | 編碼檢查 | ✅ 已使用 |
| **update_imports.py** | ~200 | 導入更新 | ✅ 可用 |
| **replace_emoji.py** | ~120 | Emoji 替換 | ✅ 可用 |
| **markdown_check.py** | ~100 | Markdown 檢查 | ✅ 可用 |

═══════════════════════════════════════════════════════════════════════════════
🏗️ 系統架構與工作流程
═══════════════════════════════════════════════════════════════════════════════

## 核心工作流程

### 1. 掃描啟動流程

```

使用者輸入目標 URL
    ↓
Integration Layer 驗證與認證
    ↓
Core Engine 接收掃描請求
    ↓
AI Engine 分析目標特徵
    ↓
Strategy Generator 生成測試策略
    ↓
Task Generator 創建具體任務
    ↓
任務分發到 Scan Engine 和 Function Modules

```

### 2. 掃描執行流程

```

Scan Engine 啟動
    ↓
┌─────────────┴─────────────┐
│                           │
Static Crawling        Dynamic Crawling
    ↓                       ↓
URL 收集              JavaScript 渲染
    ↓                       ↓
參數識別              AJAX 請求攔截
    ↓                       ↓
└─────────────┬─────────────┘
              ↓
    資訊收集 (Info Gatherer)
        • 指紋識別
        • 敏感資訊檢測
        • 技術棧分析
              ↓
    結果傳送到 Core Engine
              ↓
    AI Engine 分析攻擊面
              ↓
    動態調整策略
              ↓
    生成漏洞檢測任務
              ↓
    Function Modules 執行檢測

```

### 3. 漏洞檢測流程

```

接收檢測任務
    ↓
Smart Detection Manager 選擇檢測引擎
    ↓
┌────────┬────────┬────────┬────────┐
│        │        │        │        │
SQLi    XSS     SSRF    IDOR
Engine  Engine  Engine  Engine
│        │        │        │
└────────┴────────┴────────┴────────┘
              ↓
    Payload 生成與執行
              ↓
    結果驗證與確認
              ↓
    漏洞資訊收集
        • 類型
        • 嚴重程度
        • PoC
        • 修復建議
              ↓
    結果發送到 Integration Layer

```

### 4. 結果處理流程

```

Reception Layer 接收結果
    ↓
存入資料庫
    ↓
Analysis Layer 處理
    ↓
┌─────────┬─────────┬─────────┐
│         │         │         │
漏洞關聯  風險評估  合規檢查
分析      引擎      器
│         │         │
└─────────┴─────────┴─────────┘
              ↓
Reporting Layer
    ↓
┌─────────┬─────────┬─────────┐
│         │         │         │
技術報告  高階報告  合規報告
│         │         │         │
└─────────┴─────────┴─────────┘
              ↓
    多格式匯出
    (PDF/HTML/JSON/XML)
              ↓
    通知使用者

```

### 5. 反饋與優化流程

```

Performance Feedback 收集數據
    ↓
Scan Metadata Analyzer 分析
    ↓
識別瓶頸與問題
    ↓
Improvement Suggestion Generator
    ↓
生成優化建議
    ↓
更新 AI Engine 知識庫
    ↓
Strategy Generator 調整策略
    ↓
下次掃描應用改進

```

═══════════════════════════════════════════════════════════════════════════════
💻 技術棧詳細列表
═══════════════════════════════════════════════════════════════════════════════

## 核心技術

### 程式語言

| 語言 | 版本 | 用途 | 佔比 |
|------|------|------|------|
| **Python** | 3.12+ | 主要開發語言 | 84.6% |
| **JavaScript** | ES2022+ | 動態分析 | - |
| **SQL** | PostgreSQL 15+ | 資料儲存 | 0.6% |
| **Shell/PowerShell** | - | 自動化腳本 | 2.0% |

### Python 框架與函式庫

#### Web 框架

- **FastAPI** 0.104+
  - 現代化異步 Web 框架
  - 自動 API 文檔生成
  - Pydantic 資料驗證

#### 異步與並發

- **asyncio** (內建)
  - 異步 I/O
  - 協程管理
- **aiohttp**
  - 異步 HTTP 客戶端/伺服器
  - WebSocket 支援

#### 瀏覽器自動化

- **Playwright** / **Puppeteer**
  - 無頭瀏覽器控制
  - JavaScript 渲染
  - 網路請求攔截

#### 資料處理

- **Pydantic** 2.0+
  - 資料驗證
  - 型別檢查
  - 序列化
- **SQLAlchemy** 2.0+
  - ORM
  - 資料庫抽象層
  - 遷移管理

#### HTTP 客戶端

- **httpx**
  - 現代 HTTP 客戶端
  - HTTP/2 支援
  - 異步請求
- **requests**
  - 傳統 HTTP 庫
  - 簡單易用

#### HTML/XML 解析

- **BeautifulSoup4**
  - HTML 解析
  - DOM 操作
- **lxml**
  - 高性能 XML/HTML 處理
  - XPath 支援

#### 訊息佇列

- **pika** (RabbitMQ)
  - AMQP 協議
  - 可靠訊息傳遞
- **redis-py** (Redis)
  - 快取
  - 分散式鎖
  - 發布訂閱

#### AI/ML 相關

- **numpy**
  - 數值計算
  - 矩陣運算
- **scikit-learn** (可能)
  - 機器學習
  - 模式識別

#### 測試框架

- **pytest**
  - 單元測試
  - 整合測試
  - 測試覆蓋率

#### 程式碼品質

- **black** (格式化)
- **ruff** (Linting)
- **mypy** (型別檢查)
- **pylint** (程式碼分析)

### 資料庫

- **PostgreSQL** 15+
  - 主要資料庫
  - JSON 支援
  - 全文搜索

### 容器化與編排

- **Docker** 24+
  - 容器化部署
  - 映像管理
- **Docker Compose**
  - 多容器編排
  - 開發環境

### 版本控制

- **Git**
  - 程式碼版本管理
- **pre-commit**
  - Git hooks
  - 自動化檢查

### CI/CD (推測)

- GitHub Actions / GitLab CI
- 自動化測試
- 部署流水線

═══════════════════════════════════════════════════════════════════════════════
🔥 關鍵問題與改進建議
═══════════════════════════════════════════════════════════════════════════════

## 嚴重問題 (P0 - 緊急處理)

### 1. 極高複雜度檔案

| 檔案 | 複雜度 | 問題 | 影響 | 建議工時 |
|------|--------|------|------|----------|
| **ratelimit.py** | 91 | Token Bucket 實作過於複雜 | 維護困難、Bug 風險高 | 16-24h |
| **dynamic_content_extractor.py** | 54 | 功能過度耦合 | 難以測試、擴展困難 | 12-16h |
| **worker.py** (SSRF) | 49 | 缺少文檔、邏輯複雜 | 新人難以理解 | 8-12h |
| **worker.py** (XSS) | 48 | 缺少文檔、邏輯複雜 | 新人難以理解 | 8-12h |
| **headless_browser_pool.py** | 48 | 資源管理複雜 | 記憶體洩漏風險 | 8-12h |

**總預估工時**: 52-76 小時

### 重構建議

#### ratelimit.py 重構方案

```python
# 現況: 單一巨大類別包含所有邏輯

# 建議: 拆分為多個專注的類別
class TokenBucket:
    """基礎 Token Bucket 演算法"""
    pass

class DistributedTokenBucket:
    """分散式 Token Bucket (使用 Redis)"""
    pass

class RateLimiter:
    """速率限制器門面"""
    def __init__(self, backend: TokenBucket):
        self.backend = backend

class RateLimiterMiddleware:
    """FastAPI 中介軟體"""
    pass
```

#### dynamic_content_extractor.py 重構方案

```python
# 拆分為多個專注的類別

class DOMExtractor:
    """DOM 元素提取"""
    pass

class AJAXInterceptor:
    """AJAX 請求攔截"""
    pass

class SPANavigator:
    """單頁應用導航"""
    pass

class DynamicContentExtractor:
    """主協調器"""
    def __init__(self):
        self.dom_extractor = DOMExtractor()
        self.ajax_interceptor = AJAXInterceptor()
        self.spa_navigator = SPANavigator()
```

## 高優先級問題 (P1 - 短期處理)

### 2. 缺少文檔的關鍵檔案 (28 個)

**Worker 檔案 (最重要)**:

```
✗ services/function/function_ssrf/aiva_func_ssrf/worker.py
✗ services/function/function_xss/aiva_func_xss/worker.py
✗ services/scan/aiva_scan/worker.py
✗ services/integration/aiva_integration/app.py
```

**預估工時**: 4-6 小時

**文檔範本**:

```python
"""
SSRF 檢測 Worker

此模組實作 SSRF (Server-Side Request Forgery) 漏洞檢測的 Worker。

主要功能:
    • 接收來自 Core Engine 的 SSRF 檢測任務
    • 調用各種 SSRF 檢測器執行檢測
    • 驗證檢測結果
    • 發布確認的漏洞到 Integration Layer

檢測流程:
    1. 從任務佇列接收任務
    2. 解析目標參數
    3. 選擇適當的檢測器
    4. 執行檢測並收集證據
    5. 驗證漏洞的真實性
    6. 發布結果

支援的檢測類型:
    • 內部位址存取檢測
    • 雲端元資料存取檢測
    • OAST (帶外) 檢測
    • 盲 SSRF 檢測

配置:
    透過環境變數或 config.py 配置:
    • SSRF_TIMEOUT: 請求超時 (預設 10 秒)
    • SSRF_MAX_RETRIES: 最大重試次數 (預設 3)
    • OAST_SERVER: OAST 伺服器位址

範例:
    >>> worker = SSRFWorker()
    >>> await worker.start()

作者: AIVA Team
版本: 2.0
最後更新: 2025-10-13
"""
```

### 3. 類型提示覆蓋率不足

**當前**: 74.8% (116/155 檔案)
**目標**: 90%+
**需改進**: 39 個檔案

**優先處理的模組**:

1. function/ 模組 (15 個檔案)
2. scan/ 模組 (10 個檔案)
3. integration/ 模組 (8 個檔案)

**預估工時**: 12-16 小時

### 4. 高複雜度檢測器

| 檔案 | 複雜度 | 建議 |
|------|--------|------|
| traditional_detector.py (XSS) | 45 | 拆分上下文處理邏輯 |
| param_semantics_analyzer.py | 44 | 簡化語義分析流程 |
| smart_ssrf_detector.py | 42 | 重構檢測策略選擇 |
| tools.py (Core AI) | 38 | 拆分工具函數 |
| scope_manager.py | 36 | 簡化規則引擎 |
| internal_address_detector.py | 36 | 重構 IP 檢查邏輯 |
| sensitive_info_detector.py | 36 | 重構模式匹配 |

**預估工時**: 24-32 小時

## 中優先級改進 (P2 - 長期規劃)

### 5. 單元測試覆蓋率

- **當前**: 未知 (需評估)
- **目標**: 80%+
- **重點**: 高複雜度模組和核心邏輯

### 6. 性能優化

- 分析熱點函數
- 優化資料庫查詢
- 改進並發處理
- 減少記憶體使用

### 7. 程式碼標準化

- 統一錯誤處理模式
- 標準化日誌格式
- 統一配置管理
- API 版本控制

═══════════════════════════════════════════════════════════════════════════════
📈 專案健康度評估
═══════════════════════════════════════════════════════════════════════════════

## 綜合評分

| 維度 | 評分 | 說明 | 改進空間 |
|------|------|------|----------|
| **架構設計** | 9/10 | 模組化優秀、職責清晰 | ⭐⭐⭐⭐⭐ |
| **程式碼品質** | 7/10 | 類型提示和文檔需改進 | ⭐⭐⭐ |
| **可維護性** | 6/10 | 高複雜度檔案較多 | ⭐⭐ |
| **可測試性** | 6/10 | 缺少單元測試 | ⭐⭐ |
| **文檔完整性** | 7/10 | 多數檔案有文檔 | ⭐⭐⭐ |
| **性能** | ?/10 | 需進一步評估 | ? |
| **安全性** | 8/10 | 核心功能考慮周全 | ⭐⭐⭐⭐ |
| **擴展性** | 8/10 | 模組化設計良好 | ⭐⭐⭐⭐ |

**總體評分**: **7.3/10** (良好，有明確改進方向)

## 優勢 ✅

1. **模組化設計優秀**
   - 清晰的層次結構
   - 職責分離良好
   - 易於理解和導航

2. **功能全面**
   - 支援多種漏洞類型
   - 智慧檢測能力
   - 完整的工作流程

3. **技術棧現代化**
   - Python 3.12+
   - FastAPI
   - 異步編程

4. **AI 驅動**
   - 智慧策略生成
   - 自適應調整
   - 知識庫支援

5. **完整的整合層**
   - 報告生成
   - 風險評估
   - 性能反饋

## 需改進 ⚠️

1. **複雜度過高**
   - 多個檔案複雜度 >40
   - 維護困難
   - Bug 風險高

2. **文檔不足**
   - 28 個檔案缺少文檔
   - Worker 類別尤其需要
   - 影響新人上手

3. **類型提示不完整**
   - 74.8% 覆蓋率
   - 影響 IDE 支援
   - 降低型別安全性

4. **測試覆蓋率未知**
   - 缺少系統化測試
   - 回歸風險高

5. **性能未優化**
   - 未進行性能分析
   - 可能存在瓶頸

═══════════════════════════════════════════════════════════════════════════════
🎯 行動計畫
═══════════════════════════════════════════════════════════════════════════════

## Phase 1: 緊急修復 (1-2 週)

### Week 1

- [ ] **重構 ratelimit.py** (16-24h)
  - 拆分為 4-5 個類別
  - 添加單元測試
  - 更新文檔

- [ ] **為 Worker 檔案添加文檔** (4-6h)
  - SSRF Worker
  - XSS Worker
  - Scan Worker
  - Integration App

### Week 2

- [ ] **重構 dynamic_content_extractor.py** (12-16h)
  - 拆分提取邏輯
  - 改進錯誤處理
  - 添加測試

- [ ] **重構 SSRF Worker** (8-12h)
  - 簡化主邏輯
  - 提取子函數
  - 補充文檔

- [ ] **重構 XSS Worker** (8-12h)
  - 簡化檢測流程
  - 統一結果處理
  - 補充文檔

## Phase 2: 品質提升 (2-4 週)

### Week 3-4

- [ ] **提升類型提示覆蓋率** (12-16h)
  - function/ 模組
  - scan/ 模組
  - integration/ 模組

- [ ] **重構高複雜度檢測器** (24-32h)
  - traditional_detector.py
  - param_semantics_analyzer.py
  - smart_ssrf_detector.py
  - 其他複雜檔案

### Week 5-6

- [ ] **添加單元測試** (32-40h)
  - 核心邏輯測試
  - 檢測器測試
  - 工具函數測試
  - 目標覆蓋率: 60%+

## Phase 3: 優化與增強 (4-8 週)

- [ ] **性能分析與優化**
  - 識別瓶頸
  - 優化資料庫查詢
  - 改進並發處理

- [ ] **增加整合測試**
  - 端到端測試
  - API 測試
  - 場景測試

- [ ] **文檔完善**
  - API 文檔
  - 架構文檔
  - 開發指南
  - 部署指南

- [ ] **監控與可觀測性**
  - 指標收集
  - 日誌聚合
  - 告警系統

═══════════════════════════════════════════════════════════════════════════════
📚 相關文檔與資源
═══════════════════════════════════════════════════════════════════════════════

## 已生成的報告

### 程式碼分析報告

```
_out/analysis/
├── analysis_report_20251013_121623.json    # 完整 JSON 數據
└── analysis_report_20251013_121623.txt     # 易讀文字報告
```

### 專案報告

```
_out/
└── PROJECT_REPORT.txt                       # 專案結構報告
```

### 主要文檔

```
根目錄/
├── CODE_ANALYSIS_REPORT_20251013.md         # 程式碼分析報告
├── ANALYSIS_EXECUTION_SUMMARY.md            # 執行摘要
├── COMPREHENSIVE_PROJECT_ANALYSIS.md        # 本報告
├── ARCHITECTURE_REPORT.md                   # 架構報告
├── CORE_MODULE_ANALYSIS.md                  # 核心模組分析
├── DATA_CONTRACT.md                         # 資料契約
├── QUICK_START.md                           # 快速開始指南
└── SCAN_ENGINE_IMPROVEMENT_REPORT.md        # 掃描引擎改進報告
```

### 工具文檔

```
tools/
├── README.md                                # 工具使用說明
└── non_cp950_filtered_report.txt           # 編碼檢查報告
```

## 開發指南

### 環境設置

```bash
# Python 環境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安裝依賴
pip install -e .[dev]

# 安裝其他語言依賴
npm install                 # Node.js (scan module)
go mod download            # Go (SSRF function)
cargo build --release      # Rust (deserialize function)
```

### 程式碼品質工具

```bash
# 格式化
black services/ tools/ tests/

# Linting
ruff check --fix services/ tools/ tests/

# 型別檢查
mypy services/ tools/

# 測試
pytest -v
```

### Docker 服務

```bash
# 啟動服務
docker-compose -f docker/docker-compose.yml up -d

# 停止服務
docker-compose -f docker/docker-compose.yml down
```

## 聯絡資訊

如有任何問題或需要進一步分析，請：

1. 查看相關文檔
2. 執行分析工具獲取最新數據
3. 參考生成的報告進行決策

═══════════════════════════════════════════════════════════════════════════════
✨ 報告結束
═══════════════════════════════════════════════════════════════════════════════

**報告生成時間**: 2025-10-13 12:20:00
**分析版本**: v2.0
**總頁數**: 本報告共約 1000+ 行
**分析工時**: 約 2 小時

**AIVA 專案團隊** | 致力於構建世界級的 AI 驅動安全測試平台
