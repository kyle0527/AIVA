# Services 目錄架構概覽

> **AIVA Services 微服務架構指南**  
> 五大核心模組 + 多語言技術棧

---

## 📁 目錄結構總覽

```
services/
├── core/                          # 核心 AIVA 系統
│   └── aiva_core/                 # AIVA 五大核心模組
│       ├── cognitive_core/        # 1. 認知核心模組
│       ├── internal_exploration/  # 2. 內部探索模組
│       ├── task_planning/         # 3. 任務規劃模組
│       ├── core_capabilities/     # 4. 核心能力模組
│       ├── service_backbone/      # 5. 服務骨幹模組
│       └── ai_executor_interface.py  # AI 執行器統一接口
│
├── features/                      # 外部功能模組 (攻擊/掃描能力)
│   ├── function_sqli/             # SQL 注入檢測
│   ├── function_xss/              # XSS 漏洞檢測
│   ├── function_ssrf/             # SSRF 漏洞檢測
│   ├── function_idor/             # IDOR 訪問控制檢測
│   ├── function_bizlogic/         # 業務邏輯漏洞
│   ├── function_authn_go/         # Go 身份驗證
│   ├── function_crypto/           # 加密相關 (Rust)
│   └── function_postex/           # 後滲透
│
├── scan/                          # 掃描引擎
│   ├── python_engine/             # Python 被動掃描引擎
│   ├── rust_engine/               # Rust 主動掃描器
│   └── typescript_engine/         # TypeScript 分析引擎
│
├── integration/                   # 數據整合層
│   └── data/                      # 數據存儲
│       └── internal_exploration/  # 內部探索數據
│
├── aiva_common/                   # 共用庫
│   ├── protocols/                 # gRPC/Protobuf
│   ├── schemas/                   # 數據模型 (Pydantic)
│   └── ai/                        # AI 接口定義
│
└── data/                          # 數據存儲
    └── vector_db/                 # 向量數據庫
```

---

## 🏗️ 五大核心模組架構

### 1️⃣ **cognitive_core** (認知核心模組)

**路徑**: `services/core/aiva_core/cognitive_core/`

**核心職責**: AI 認知、決策、學習、RAG、神經網路

```
cognitive_core/
├── decision/                  # AI 決策引擎
│   ├── skill_graph.py        # 技能圖譜
│   ├── enhanced_decision_agent.py  # 增強決策代理
│   └── ai_decision_core.py   # AI 決策核心
│
├── learning_system/          # 學習子系統
│   ├── training/             # 訓練管道
│   ├── models/               # AI 模型
│   ├── tracking/             # 模型追蹤
│   ├── analysis/             # 分析引擎
│   └── event_listener.py     # 事件監聽
│
├── rag/                      # RAG 檢索增強
│   └── rag_trigger.py        # RAG 觸發器
│
├── embedded_knowledge/       # 內嵌知識庫
└── neural/                   # 神經網路
```

**設計原則**:
- 統一管理所有 AI 認知功能
- 整合學習系統到認知核心
- RAG 系統與決策引擎緊密協作
- 向量數據庫支援知識檢索

---

### 2️⃣ **internal_exploration** (內部探索模組)

**路徑**: `services/core/aiva_core/internal_exploration/`

**核心職責**: 自我認知、能力分析、內部監控、流程執行

```
internal_exploration/
├── aiva_internal_executor.py      # 內部流程執行器
├── aiva_external_executor.py      # 外部模組執行器
├── unified_executor_controller.py # 統一執行器控制層
├── aiva_internal_classifier.py    # 內部流程分類器
├── aiva_external_classifier.py    # 外部模組分類器
│
├── python_tools/                  # Python AST 工具
│   ├── aiva_flow_analyzer.py     # 流程分析器
│   └── aiva_cli_implementation.py
│
├── go_tools/                      # Go 分析工具
├── rust_tools/                    # Rust 分析工具
├── typescript_tools/              # TypeScript 分析工具
│
├── self_healing/                  # 自我修復
└── system_self_explorer.py        # 系統自我探索器
```

**執行器架構**:
```
┌─────────────────────────────────────────┐
│   AI 決策層 (ai_decision_core.py)      │
│   - 決策 + RAG + 歷史反饋               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  AI 接口層 (ai_executor_interface.py)  │
│  - 提供統一執行入口給 AI 決策層使用     │
│  - execute(), execute_batch()          │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 統一控制層 (unified_executor_controller)│
│ - 能力路由: internal vs external        │
│ - 自動選擇執行器                         │
└──────┬────────────────────┬─────────────┘
       │                    │
┌──────▼─────────┐  ┌───────▼──────────┐
│ 內部執行器      │  │ 外部執行器        │
│ - Python only  │  │ - Multi-language │
│ - 調用模組類方法│  │ - 調用模組類方法  │
└────────────────┘  └──────────────────┘
```

**設計特點**:
- AI 通過統一接口調用能力
- 執行器負責路由到正確的模組
- 無需為模組創建"簡化 API"
- 支援內部和外部能力分類

---

### 3️⃣ **task_planning** (任務規劃模組)

**路徑**: `services/core/aiva_core/task_planning/`

**核心職責**: 規劃器、執行器、指揮官、編排

```
task_planning/
├── commander/                # 指揮層
│   └── attack_coordinator.py # 攻擊協調器
│
├── planner/                  # 規劃層
│   ├── planner.py           # 任務規劃器
│   └── two_phase_scan_orchestrator.py  # 兩階段掃描
│
├── executor/                 # 執行層
│   ├── unified_executor.py  # 統一執行器
│   └── task_executor.py     # 任務執行器
│
└── orchestrator/             # 編排層
    └── capability_orchestrator.py  # 能力編排器
```

**設計層次**:
1. **Commander** - 高層戰略決策
2. **Planner** - 任務分解與排程
3. **Orchestrator** - 能力組合與協調
4. **Executor** - 具體執行與監控

---

### 4️⃣ **core_capabilities** (核心能力模組)

**路徑**: `services/core/aiva_core/core_capabilities/`

**核心職責**: 攻擊鏈、業務邏輯、對話、插件

```
core_capabilities/
├── attack/                  # 攻擊能力
│   ├── attack_chain/       # 攻擊鏈
│   └── attack_executor.py  # 攻擊執行器
│
├── business_logic/          # 業務邏輯
├── dialog/                  # 對話處理
│   └── conversation_handler/
│
├── orchestration/           # 編排能力
└── plugin_manager/          # 插件管理
```

---

### 5️⃣ **service_backbone** (服務骨幹模組)

**路徑**: `services/core/aiva_core/service_backbone/`

**核心職責**: API、協調、消息、存儲、狀態

```
service_backbone/
├── api/                     # REST API
│   └── app.py              # FastAPI 應用
│
├── coordination/            # 協調器
│   └── coordinator.py      # 服務協調
│
├── messaging/               # 消息系統
│   └── message_broker.py   # 消息中介
│
├── storage/                 # 存儲層
│   └── backends.py         # 存儲後端
│
├── state/                   # 狀態管理
│   └── state_manager.py    # 狀態管理器
│
└── adapters/                # 適配器層
```

**基礎設施職責**:
- RESTful API 服務
- 微服務間通信協調
- 異步消息傳遞
- 持久化存儲管理
- 全局狀態跟蹤

---

## 🔧 外部功能模組 (Features)

**路徑**: `services/features/`

### Python 模組

| 模組 | 功能 | 技術特點 |
|------|------|----------|
| function_sqli | SQL 注入檢測 | 多引擎協調、智能檢測 |
| function_xss | XSS 漏洞檢測 | DOM/反射/存儲型檢測 |
| function_ssrf | SSRF 漏洞檢測 | 內網探測、協議繞過 |
| function_idor | IDOR 訪問控制 | 水平/垂直越權檢測 |
| function_bizlogic | 業務邏輯漏洞 | 邏輯流程分析 |
| function_postex | 後滲透 | 權限提升、持久化 |

### 多語言模組

| 模組 | 語言 | 功能 | 優勢 |
|------|------|------|------|
| function_authn_go | Go | 身份驗證 | 高性能並發 |
| function_crypto | Rust | 加密相關 | 內存安全 |

**架構特點**:
- 每個模組獨立封裝
- 統一 CLI 接口
- 通過執行器調用
- 支援多語言混用

---

## 🔍 掃描引擎 (Scan)

**路徑**: `services/scan/`

### 多語言引擎架構

```
scan/
├── python_engine/              # Python 被動掃描
│   ├── passive_analyzer.py    # 被動分析器
│   ├── xxe_detector.py        # XXE 檢測
│   └── deserialization_detector_v2.py  # 反序列化檢測
│
├── rust_engine/                # Rust 主動掃描
│   ├── http_smuggling_detector.rs  # HTTP 走私
│   └── auth_brute_forcer.rs        # 認證爆破
│
├── go_engine/                  # Go 並發掃描
│   ├── param_fuzzer.go        # 參數模糊測試
│   └── ssrf_scanner.go        # SSRF 掃描
│
└── typescript_engine/          # TypeScript 動態分析
    ├── dom_xss_scanner.ts     # DOM XSS
    └── spa_crawler.ts         # SPA 爬蟲
```

### 引擎分工

| 引擎 | 適用場景 | 技術優勢 |
|------|----------|----------|
| **Python** | 複雜邏輯分析、AI 決策 | 生態豐富、開發快速 |
| **Rust** | 高性能掃描、內存敏感 | 零成本抽象、類型安全 |
| **Go** | 網路並發、快速檢測 | Goroutine、Channel |
| **TypeScript** | 前端安全、動態分析 | 瀏覽器環境、AST 分析 |

**設計原則**:
- 語言適配場景
- 統一結果格式
- CLI 驅動執行
- 零依賴衝突

---

## 📦 共享庫 (AIVA Common)

**路徑**: `services/aiva_common/`

**核心職責**: 數據標準、協議、配置、工具

```
aiva_common/
├── protocols/                 # 通信協議
│   ├── grpc/                 # gRPC 定義
│   └── protobuf/             # Protobuf Schema
│
├── schemas/                   # 數據模型
│   ├── scan_result.py        # 掃描結果 (Pydantic)
│   ├── vulnerability.py      # 漏洞模型
│   └── capability.py         # 能力定義
│
├── ai/                        # AI 接口
│   ├── ai_command_center.py  # AI 命令中心
│   └── message_broker.py     # 消息代理
│
├── config/                    # 配置管理
│   └── config_manager.py     # 配置管理器
│
├── security/                  # 安全中間件
│   └── security.py           # 安全工具
│
└── utils/                     # 工具函數
    ├── async_utils/          # 異步工具
    └── observability/        # 可觀測性
```

**設計標準**:
- Pydantic v2 類型安全
- 單一數據源 (SOT) 原則
- 雙軌通信 (MessageBroker + CLI)
- 企業級錯誤處理

---

## 🎯 架構設計原則

### 1. 模組化與解耦
- 五大核心模組職責清晰
- 功能模組獨立封裝
- 通過接口而非實現依賴

### 2. 多語言技術棧
- Python: 智能分析與決策
- Rust: 高性能與安全
- Go: 網路並發
- TypeScript: 動態前端

### 3. 統一執行模型
- AI 通過統一接口調用能力
- 執行器負責路由與協調
- CLI 驅動的模組執行

### 4. 事件驅動架構
- 使用 asyncio.Future 
- 消息代理異步通信
- 事件監聽與自動觸發

### 5. 數據標準化
- Pydantic Schema 定義
- CVSS v3.1 風險評級
- SARIF v2.1.0 結果格式

---

## 📊 架構統計

### 模組分布

| 層級 | 數量 | 說明 |
|------|------|------|
| 核心模組 | 5 | cognitive, exploration, planning, capabilities, backbone |
| 外部功能 | 17+ | Python(6+) + Go(1) + Rust(1) |
| 掃描引擎 | 4 | Python + Rust + Go + TypeScript |
| 共享庫 | 1 | aiva_common |
| **總計** | **27+** | 完整微服務架構 |

### 能力分類

| 類型 | 描述 | 數量範圍 |
|------|------|----------|
| 內部流程 | aiva_core 內部能力 | 286 flows |
| 外部流程 | features/scan 模組能力 | 210+ flows |
| **總計** | **全系統能力** | **496+ flows** |

---

## 🔗 相關文檔

- [SERVICES_DETAILED_ARCHITECTURE.md](SERVICES_DETAILED_ARCHITECTURE.md) - 詳細架構說明
- [core/AIVA_CORE_ARCHITECTURE_GUIDE.md](core/AIVA_CORE_ARCHITECTURE_GUIDE.md) - 核心模組指南
- [../../../services/README.md](../../../services/README.md) - Services 總覽

---

**維護**: AIVA 架構團隊  
**參考**: services/ARCHITECTURE_ANALYSIS_2026.md (已歸檔)
