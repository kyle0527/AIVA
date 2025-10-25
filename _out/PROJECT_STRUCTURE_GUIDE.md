================================================================================
AIVA 專案程式碼樹狀架構圖 - 詳細中文說明文件
================================================================================
更新日期: 2025年10月25日 23:17
專案路徑: C:\D\fold7\AIVA-git
文檔版本: v4.0 (大規模清理後)

📊 項目概況
────────────────────────────────────────────────────────────────────────────────

## 🎯 AIVA 系統簡介

**AIVA (AI-Driven Vulnerability Assessment)** 是一個由 AI 驅動的自動化漏洞評估系統，
專為 HackerOne 等漏洞賞金平台設計。系統採用多語言架構 (Python、Rust、Go、TypeScript)，
結合深度學習和強化學習技術，實現智能化的安全測試。

### 核心特色
- 🧠 **BioNeuron Master**: 生物啟發式 AI 決策引擎
- 🎯 **AI Commander**: 智能攻擊策略規劃器
- 🔍 **SecretDetector**: 57 規則密鑰檢測 + API 驗證
- 🤖 **ModelTrainer**: DQN/PPO 強化學習訓練
- ⚡ **多語言整合**: Python (86%) + Rust (5%) + Go (5%) + TypeScript (3%)

## 📈 最新統計 (2025-10-25)

### 代碼統計
- **總檔案數**: 509 個程式碼檔案
- **總程式碼行數**: 133,999 行
- **主要語言**: Python (86.4%), Go (4.7%), Rust (4.5%), TypeScript (3.3%)

### 語言分布詳情
```
Python:     458 檔案, 115,806 行 (86.4%) - 核心邏輯、AI 引擎、Worker
Rust:       13 檔案, 6,017 行 (4.5%)   - 高性能掃描、密鑰檢測
Go:         22 檔案, 6,254 行 (4.7%)   - 掃描器、並發處理
TypeScript: 12 檔案, 4,460 行 (3.3%)   - Web 前端、API 介面
JavaScript: 1 檔案, 578 行 (0.4%)     - Dashboard
SQL:        2 檔案, 524 行 (0.4%)     - 資料庫 Schema
HTML:       1 檔案, 360 行 (0.3%)     - Web UI
```

### 文檔整理成果 (2025-10-25)
✅ **已刪除約 100 個過時文件**:
- 根目錄: 6 個過時文件 (COMPREHENSIVE_PROGRESS_REPORT.md 等)
- docs/: 5 個重複文件
- reports/: ~40 個過時報告
- _out/: ~50 個臨時文件

✅ **文檔結構優化**:
- 統一索引: docs/DOCUMENTATION_INDEX.md
- 報告分類: reports/ 子目錄組織
- 歸檔管理: _archive/ 歷史文檔

## 🏗️ 專案架構說明

### 1. 📁 根目錄結構

#### 核心啟動文件
```
aiva_launcher.py          # AIVA 系統主啟動器，協調所有模組啟動
verify_p0_fixes.py        # P0 優先級缺陷驗證腳本
aiva_package_validator.py # 套件完整性驗證工具
```

#### 配置文件
```
pyproject.toml            # Python 專案配置 (Black, isort, pytest)
pyrightconfig.json        # Pyright 型別檢查配置
mypy.ini                  # MyPy 型別檢查配置
ruff.toml                 # Ruff Linter 配置
requirements.txt          # Python 依賴項列表
.env.example              # 環境變數範本
```

#### 重要文檔
```
README.md                 # 專案主文檔，快速開始指南
DEVELOPER_GUIDE.md        # 開發者指南，架構與開發規範
QUICK_REFERENCE.md        # 快速參考手冊
REPOSITORY_STRUCTURE.md   # 倉庫結構詳細說明
FILE_CLEANUP_PLAN.md      # 文件清理計劃與執行記錄
```

### 2. 📚 docs/ - 文檔目錄

#### 📖 核心技術文檔
```
AIVA_AI_TECHNICAL_DOCUMENTATION.md  # AI 系統架構、BioNeuron、經驗學習
ARCHITECTURE_MULTILANG.md           # 多語言架構設計與整合策略
SCHEMAS_DIRECTORIES_EXPLANATION.md  # Schema 組織架構說明
```

#### 🆕 功能指南 (最新)
```
API_VERIFICATION_GUIDE.md    # 密鑰 API 驗證功能 (2025-10-25 新增)
                             # - TruffleHog 模式
                             # - 10 個服務驗證 (GitHub, Slack, Stripe...)
                             # - 緩存機制、異步驗證

SECRET_DETECTOR_RULES.md     # 57 個密鑰檢測規則目錄 (2025-10-25 新增)
                             # - AWS, Azure, GitHub, GitLab
                             # - 完整規則說明與範例

RL_ALGORITHM_COMPARISON.md   # 強化學習算法對比 (2025-10-24 新增)
                             # - Q-learning vs DQN vs PPO
                             # - 性能分析與選擇建議

DIAGRAM_FILE_MANAGEMENT.md   # Mermaid 圖表管理規範
```

#### 📏 最佳實踐
```
CROSS_LANGUAGE_BEST_PRACTICES.md  # 跨語言開發規範
                                  # - Python ↔ Rust ↔ Go ↔ TypeScript
                                  # - Schema 共享、錯誤處理

IMPORT_PATH_BEST_PRACTICES.md    # Python 導入路徑規範
                                  # - 絕對導入 vs 相對導入
                                  # - 套件組織最佳實踐
```

#### 📁 子目錄
```
ARCHITECTURE/   # 架構設計文檔
DEVELOPMENT/    # 開發流程與工具
guides/         # 詳細使用指南
plans/          # 開發計劃與路線圖
assessments/    # 系統評估報告
reports/        # (已移至根目錄 reports/)
```

### 3. 📊 reports/ - 報告目錄 (已大幅精簡)

#### 當前有效報告
```
import_path_check_report.md        # 導入路徑檢查報告

connectivity/                      # 連接測試報告
├── aiva_connectivity_report_*.md # 系統連接性測試結果
└── SYSTEM_CONNECTIVITY_REPORT.json

security/                          # 安全評估報告
├── AIVA_Enterprise_Security_*.md # 企業安全評估
└── juice_shop_attack_report_*.md # Juice Shop 攻擊測試

ANALYSIS_REPORTS/                  # 架構分析
├── ARCHITECTURE_ANALYSIS_*.md    # 架構分析建議
└── INTEGRATION_ANALYSIS.md       # 整合分析

IMPLEMENTATION_REPORTS/            # 實現完成報告
├── DELIVERY_SUMMARY.md           # 交付摘要
├── FINAL_COMPLETION_REPORT.md    # 最終完成報告
└── FIX_SUMMARY.md                # 修復摘要

MIGRATION_REPORTS/                 # 遷移報告
├── GO_MIGRATION_REPORT.md        # Go 語言遷移
└── MODULE_IMPORT_FIX_REPORT.md   # 模組導入修復

PROGRESS_REPORTS/                  # 進度報告
├── PROGRESS_DASHBOARD.md         # 進度儀表板
└── ROADMAP_NEXT_10_WEEKS.md      # 未來 10 週路線圖
```

#### ❌ 已刪除的過時報告 (~40 個)
```
已清理：
- 所有 COMPLETE/VERIFICATION/PROGRESS 舊報告
- ASYNC_FILE_OPERATIONS_* (異步操作完成報告)
- WORKER_STATISTICS_* (Worker 統計報告)
- SYSTEM_REPAIR_* (系統修復報告)
- DEBUG/FUNCTIONALITY/TODO 分析報告
- SCHEMAS_ENUMS_* (Schema 枚舉擴展報告)
- 項目完成總結_*.md (中文重複報告)
```

### 4. 🔧 services/ - 核心服務

#### core/ - 核心服務 (Python)
```
aiva_core/
├── __init__.py                 # 核心模組初始化
├── ai_commander.py             # 🧠 AI 指揮官 (策略決策)
│                               # - 計劃生成、風險評估
│                               # - 經驗學習與檢索 (RAG)
│                               # - JSON 經驗資料庫 (待升級 PostgreSQL)
│
├── attack_plan_executor.py     # ⚔️ 攻擊計劃執行器
├── bio_neuron_master.py        # 🧬 生物神經元主控
│                               # - NLU 自然語言理解
│                               # - 降級策略 (關鍵字匹配)
│                               # - 中文指令識別
│
├── experience_manager.py       # 📚 經驗管理器
│                               # - 經驗評分 (成功 40% + 時間 30% + 信心 30%)
│                               # - 時間衰減函式 (7/30/90 天)
│                               # - 經驗去重邏輯
│
├── message_handler.py          # 📨 訊息處理器 (RabbitMQ)
├── rag_engine.py               # 🔍 檢索增強生成引擎
├── task_queue.py               # 📋 任務佇列管理
└── worker_manager.py           # 👷 Worker 管理器
```

#### models/ - AI 模型 (Python)
```
aiva_models/
├── __init__.py
├── base_trainer.py             # 基礎訓練器抽象類
├── model_trainer.py            # 🎓 模型訓練器 (已擴展)
│                               # - 監督學習 (SVM, Random Forest)
│                               # - Q-learning 強化學習
│                               # - DQN (Deep Q-Network) 🆕
│                               # - PPO (Proximal Policy Optimization) 🆕
│                               # - ReplayBuffer 和 RolloutBuffer
│
├── rl_trainer.py               # 強化學習專用訓練器
└── supervised_trainer.py       # 監督學習訓練器
```

#### scan/ - 掃描服務 (多語言)

##### info_gatherer_rust/ - Rust 掃描器 ⭐ 核心
```
src/
├── main.rs                     # 主程式 (RabbitMQ 整合)
├── scanner.rs                  # 敏感資訊掃描器
│
├── secret_detector.rs          # 🔐 密鑰檢測器 (57 規則)
│                               # - AWS (3), Azure (4), GitHub (3)
│                               # - GitLab (3), Stripe (3), Twilio (1)
│                               # - 資料庫連接字串 (5)
│                               # - 通用令牌 (4)
│                               # - Shannon 熵值檢測
│
├── verifier.rs                 # ✅ API 驗證器 (新增 2025-10-25)
│                               # - GitHub Token 驗證
│                               # - Slack Token 驗證
│                               # - Stripe Key 驗證
│                               # - SendGrid, DigitalOcean, Cloudflare
│                               # - Datadog 等 10 個服務
│                               # - 緩存機制 (1 小時 TTL)
│
└── git_history_scanner.rs      # Git 歷史掃描 (基礎功能)

Cargo.toml                      # Rust 依賴
├── regex = "1.10"              # 正則引擎
├── reqwest = "0.11"            # HTTP 客戶端 (API 驗證)
├── tokio = "1.35"              # 異步運行時
├── serde_json = "1.0"          # JSON 序列化
└── git2 = "0.18"               # Git 操作
```

##### scanner_go/ - Go 掃描器
```
cmd/scanner/
└── main.go                     # Go 掃描器主程式

internal/
├── detector/                   # 檢測器
├── scanner/                    # 掃描邏輯
└── utils/                      # 工具函數
```

### 5. 🎭 features/ - 功能模組 (Workers)

```
workers_idor/                   # IDOR 漏洞檢測 Worker
├── __init__.py
├── idor_worker.py              # ✅ P1 已修復
│                               # - 5 種認證方式支持
│                               # - 多帳戶測試
│                               # - 統計資料收集完整

workers_ssrf/                   # SSRF 漏洞檢測 Worker
├── __init__.py
└── ssrf_worker.py              # ✅ P1 已修復
                                # - 統計資料收集完整

workers_sqli/                   # SQL 注入檢測 Worker
workers_xss/                    # XSS 跨站腳本攻擊 Worker
workers_api/                    # API 安全測試 Worker
workers_payment/                # 支付邏輯繞過測試 Worker
```

### 6. 📦 schemas/ - 資料結構定義

```
schemas/
├── __init__.py
├── common/                     # 通用 Schema
│   ├── enums.py                # 枚舉定義
│   └── shared.py               # 共享資料結構
│
├── core/                       # 核心 Schema
│   ├── ai_commander.py         # AI Commander 資料結構
│   ├── experience.py           # 經驗資料結構
│   └── task.py                 # 任務資料結構
│
├── features/                   # 功能 Schema
│   ├── idor.py                 # IDOR Worker Schema
│   ├── ssrf.py                 # SSRF Worker Schema
│   └── ...
│
├── integration/                # 整合 Schema
│   ├── rabbitmq.py             # RabbitMQ 訊息格式
│   └── api.py                  # API 介面格式
│
└── scan/                       # 掃描 Schema
    ├── findings.py             # 掃描發現資料結構
    └── targets.py              # 掃描目標定義
```

### 7. 🧪 testing/ - 測試目錄

```
core/
├── test_ai_commander.py        # AI Commander 測試
└── test_bio_neuron.py          # BioNeuron 測試

features/
├── test_idor_worker.py         # IDOR Worker 測試
└── test_ssrf_worker.py         # SSRF Worker 測試

integration/
└── comprehensive_test.py       # 整合測試

scan/
└── juice_shop_real_attack_test.py  # 真實攻擊測試

p0_fixes_validation_test.py     # ✅ P0 修復驗證測試
```

### 8. 🔌 api/ - API 服務 (FastAPI)

```
routers/
├── admin.py                    # 管理介面路由
├── auth.py                     # 認證路由
└── security.py                 # 安全路由

main.py                         # FastAPI 主應用
start_api.py                    # API 啟動腳本
test_api.py                     # API 測試
```

### 9. 🌐 web/ - Web 前端

```
js/
└── aiva-dashboard.js           # AIVA 控制台 (JavaScript)

index.html                      # Web UI 首頁
```

### 10. 🛠️ tools/ - 開發工具

```
common/
├── analysis/                   # 分析工具
│   ├── analyze_core_modules.py
│   └── analyze_cross_language_ai.py
│
├── development/                # 開發工具
│   ├── generate_mermaid_diagrams.py
│   └── py2mermaid.py
│
├── quality/                    # 代碼品質工具
│   └── markdown_check.py
│
└── schema/                     # Schema 工具
    └── schema_validator.py

core/                           # 核心工具
├── compare_schemas.py
└── verify_migration_completeness.py

features/                       # 功能工具
└── mermaid_optimizer.py

integration/                    # 整合工具
├── aiva-contracts-tooling/
└── aiva-schemas-plugin/

scan/                           # 掃描工具
└── extract_enhanced.py
```

### 11. 🐳 docker/ - Docker 容器

```
initdb/
├── 001_schema.sql              # 資料庫結構初始化
└── 002_enhanced_schema.sql     # 增強資料庫結構
```

### 12. 📁 其他重要目錄

```
_archive/                       # 歷史歸檔
├── completed_tasks/            # 已完成任務報告
├── historical_analysis/        # 歷史分析文檔
└── deprecated_docs/            # 廢棄文檔

_out/                           # 臨時輸出 (已大幅清理)
├── analysis/                   # 分析結果
├── architecture_diagrams/      # 架構圖
└── statistics/                 # 統計數據

backup/                         # 備份目錄
└── 20251024_173429/            # 時間戳備份

config/                         # 配置
├── api_keys.py                 # API 密鑰管理
└── settings.py                 # 系統設定

data/                           # 資料目錄
├── ai_commander/               # AI Commander 資料
│   ├── experience_db/          # 經驗資料庫 (JSON)
│   └── knowledge/vectors/      # 知識向量
├── database/                   # 資料庫檔案
├── knowledge/                  # 知識庫
├── scenarios/                  # 測試場景
└── training/                   # 訓練資料

logs/                           # 日誌目錄

models/                         # 訓練模型存儲
```

## 🎯 開發進度追蹤

### ✅ 已完成 (8/10 任務)

1. **P1 缺陷修復** ✅
   - IDOR Worker 統計資料收集完整
   - SSRF Worker 統計資料收集完整
   - 5 種認證方式支持

2. **BioNeuron Master 降級策略增強** ✅
   - 關鍵字匹配 (difflib SequenceMatcher)
   - 中文指令識別擴展 (10+ 同義詞)
   - 相似度匹配邏輯 (0.6 閾值)

3. **經驗評分系統優化** ✅
   - 新評分公式 (成功 40% + 時間 30% + 信心 30%)
   - 時間衰減函式 (7/30/90 天)
   - 經驗去重邏輯

4. **AI Commander 決策增強** ✅
   - 詳細提示詞生成
   - 多因素信心度計算
   - 五維度風險分析
   - 代碼新增 ~350 行

5. **ModelTrainer 算法擴展** ✅
   - Q-learning → DQN → PPO
   - ReplayBuffer 和 RolloutBuffer
   - 完整訓練流程
   - 代碼新增 ~1200 行

6. **SecretDetector 規則擴展** ✅
   - 15 → 57 規則 (+42)
   - 覆蓋 AWS, Azure, GitHub, GitLab 等
   - 18 個測試案例 (100% 通過)

7. **API 驗證功能** ✅ (2025-10-25 新增)
   - TruffleHog 模式實現
   - 10 個服務驗證 (GitHub, Slack, Stripe...)
   - 緩存機制、異步驗證
   - 代碼新增 ~700 行

8. **文檔整理** ✅ (2025-10-25 新增)
   - 刪除 ~100 個過時文件
   - 統一索引結構
   - 報告分類整理

### ⏭️ 已跳過 (1/10 任務)

9. **Git 歷史掃描優化** ⏭️
   - 原因：不適用於 HackerOne 場景
   - 基礎功能已保留供未來使用

### ⏸️ 待辦 (1/10 任務)

10. **單元測試補充** ⏸️
    - IDOR Worker 測試
    - AI Commander 測試
    - BioNeuron Master 測試
    - ModelTrainer 測試
    - 目標覆蓋率 >80%

### 📊 任務完成率: 80% (8/10)

## 🔑 關鍵技術棧

### 後端
- **Python 3.10+**: 主要開發語言
  - FastAPI: Web API 框架
  - Pydantic: 資料驗證
  - SQLAlchemy: ORM (計劃中)
  - PyTorch: 深度學習
  - LangChain: LLM 整合

- **Rust**: 高性能掃描
  - tokio: 異步運行時
  - reqwest: HTTP 客戶端
  - regex: 正則引擎
  - serde: 序列化

- **Go**: 並發掃描
  - Goroutines: 並發處理
  - Channels: 通信機制

### 前端
- **TypeScript/JavaScript**: Web UI
- **HTML/CSS**: 介面設計

### 資料庫
- **PostgreSQL**: 關係型資料庫 (計劃中)
- **JSON**: 臨時經驗存儲 (待升級)
- **Vector DB**: 知識向量存儲

### 訊息佇列
- **RabbitMQ**: 任務分發與通信

### AI/ML
- **OpenAI GPT**: LLM 推理
- **PyTorch**: 深度學習框架
- **Q-learning/DQN/PPO**: 強化學習算法

## 📝 開發規範

### 導入路徑規範
```python
# ✅ 正確 - 使用絕對導入
from services.core.aiva_core import ai_commander
from schemas.core.experience import ExperienceSchema

# ❌ 錯誤 - 避免相對導入
from ..core import ai_commander
from ...schemas import experience
```

### 命名規範
```python
# 檔案: snake_case.py
# 類別: PascalCase
# 函數: snake_case()
# 常數: UPPER_SNAKE_CASE
```

### 文檔命名
```
指南: *_GUIDE.md
報告: *_REPORT.md
完成總結: *_COMPLETION.md
分析: *_ANALYSIS.md
索引: INDEX.md 或 README.md
```

## 🔍 快速查找指南

### 按功能查找

**AI 相關**:
- BioNeuron Master: `services/core/aiva_core/bio_neuron_master.py`
- AI Commander: `services/core/aiva_core/ai_commander.py`
- ModelTrainer: `services/models/aiva_models/model_trainer.py`
- 文檔: `docs/AIVA_AI_TECHNICAL_DOCUMENTATION.md`

**安全掃描**:
- SecretDetector: `services/scan/info_gatherer_rust/src/secret_detector.rs`
- API 驗證: `services/scan/info_gatherer_rust/src/verifier.rs`
- 規則文檔: `docs/SECRET_DETECTOR_RULES.md`
- 驗證指南: `docs/API_VERIFICATION_GUIDE.md`

**Workers**:
- IDOR: `features/workers_idor/idor_worker.py`
- SSRF: `features/workers_ssrf/ssrf_worker.py`
- 其他: `features/workers_*/`

**測試**:
- P0 驗證: `testing/p0_fixes_validation_test.py`
- 整合測試: `testing/integration/comprehensive_test.py`
- Worker 測試: `testing/features/test_*_worker.py`

### 按語言查找

**Python**:
- 核心服務: `services/core/`
- AI 模型: `services/models/`
- Workers: `features/`
- 測試: `testing/`

**Rust**:
- 掃描器: `services/scan/info_gatherer_rust/src/`
- 文檔: `docs/SECRET_DETECTOR_RULES.md`, `docs/API_VERIFICATION_GUIDE.md`

**Go**:
- 掃描器: `services/scan/scanner_go/`

**TypeScript**:
- API 定義: `api/` (計劃中)
- Web UI: `web/js/`

## 📞 維護資訊

### 聯絡方式
- **項目**: AIVA - AI-Driven Vulnerability Assessment
- **倉庫**: kyle0527/AIVA
- **分支**: main
- **文檔問題**: 請提交 Issue

### 更新記錄

**v4.0 (2025-10-25)**
- ✅ 大規模文檔清理 (刪除 ~100 個文件)
- ✅ API 驗證功能完成
- ✅ SecretDetector 規則擴展
- ✅ 統一索引結構

**v3.0 (2025-10-24)**
- ✅ ModelTrainer DQN/PPO 擴展
- ✅ RL 算法對比文檔

**v2.0 (2025-10-23)**
- ✅ P0 缺陷修復
- ✅ BioNeuron 降級策略
- ✅ AI Commander 優化

---

**文檔版本**: v4.0  
**最後更新**: 2025-10-25 23:17  
**狀態**: ✅ 生產就緒 (80% 任務完成)
