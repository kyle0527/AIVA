# Services 目錄架構分析報告

**分析日期**: 2026-02-01  
**架構版本**: 五大模組 (已整合 external_learning → cognitive_core/learning_system)  
**最新修復**: internal_loop_connector v11.1 (force_refresh 功能修復)

---

## 📁 目錄結構總覽

```
services/
├── core/                          # 核心 AIVA 系統
│   └── aiva_core/                 # AIVA 五大核心模組
│       ├── cognitive_core/        # ✅ 1. 認知核心模組
│       ├── internal_exploration/  # ✅ 2. 內部探索模組
│       ├── task_planning/         # ✅ 3. 任務規劃模組
│       ├── core_capabilities/     # ✅ 4. 核心能力模組
│       ├── service_backbone/      # ✅ 5. 服務骨幹模組
│       ├── ai_executor_interface.py  # AI 執行器統一接口
│       └── config/                # 配置管理
│
├── features/                      # 外部功能模組 (攻擊/掃描能力)
│   ├── function_sqli/             # SQL 注入檢測
│   ├── function_xss/              # XSS 漏洞檢測
│   ├── function_ssrf/             # SSRF 漏洞檢測
│   ├── function_idor/             # IDOR 訪問控制檢測
│   ├── function_bizlogic/         # 業務邏輯漏洞
│   ├── function_authn_go/         # Go 身份驗證 (Go)
│   ├── function_crypto/           # 加密相關 (Rust)
│   └── function_postex/           # 後滲透
│
├── scan/                          # 掃描引擎
│   ├── python_engine/             # Python 被動掃描引擎
│   ├── rust_scanner/              # Rust 主動掃描器
│   └── typescript_engine/         # TypeScript 分析引擎
│
├── integration/                   # 數據整合層
│   └── data/
│       └── internal_exploration/  # 內部探索數據
│           └── latest_classification.json  # 286 內部流程
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

**子模組**:
```
cognitive_core/
├── decision/                  # AI 決策
│   ├── skill_graph.py        # 技能圖譜
│   ├── enhanced_decision_agent.py  # 增強決策代理
│   └── ai_decision_core.py   # AI 決策核心 (新增)
├── learning_system/          # 學習子系統 (整合自 external_learning)
│   ├── training/             # 訓練管道
│   ├── models/               # AI 模型
│   ├── tracking/             # 模型追蹤
│   ├── analysis/             # 分析引擎
│   └── event_listener.py     # 事件監聽
├── rag/                      # RAG 檢索增強
│   └── rag_trigger.py        # RAG 觸發器
├── embedded_knowledge/       # 內嵌知識庫
└── neural_network/           # 神經網路
```

**功能**: AI 認知、決策、學習、RAG、神經網路

**整合說明**: 
- ✅ `external_learning` 已於 2026-01-03 整合到 `cognitive_core/learning_system`
- ✅ 五大模組架構已確立

---

### 2️⃣ **internal_exploration** (內部探索模組)

**路徑**: `services/core/aiva_core/internal_exploration/`

**核心組件**:
```
internal_exploration/
├── aiva_internal_executor.py      # 內部流程執行器 (286 flows)
├── aiva_external_executor.py      # 外部模組執行器 (210 flows)
├── unified_executor_controller.py # 統一執行器控制層
├── aiva_internal_classifier.py    # 內部流程分類器
├── aiva_external_classifier.py    # 外部模組分類器
├── python_tools/                  # Python AST 工具
│   ├── aiva_flow_analyzer.py     # 流程分析器
│   └── aiva_cli_implementation.py
├── self_healing/                  # 自我修復
└── system_self_explorer.py        # 系統自我探索器
```

**功能**: 自我認知、能力分析、內部監控、流程執行

**數據源**:
- 內部: `latest_classification.json` (286 flows)
- 外部: `classification_data.json` (210 flows)

---

### 3️⃣ **task_planning** (任務規劃模組)

**路徑**: `services/core/aiva_core/task_planning/`

```
task_planning/
├── planner.py                # 任務規劃器
├── unified_executor.py       # 統一執行器
├── capability_orchestrator.py  # 能力編排器
└── task_commander.py         # 任務指揮官
```

**功能**: 規劃器、執行器、指揮官、編排

---

### 4️⃣ **core_capabilities** (核心能力模組)

**路徑**: `services/core/aiva_core/core_capabilities/`

```
core_capabilities/
├── attack_chain/            # 攻擊鏈
├── business_logic/          # 業務邏輯
├── conversation_handler/    # 對話處理
└── plugin_manager/          # 插件管理
```

**功能**: 攻擊鏈、業務邏輯、對話、插件

---

### 5️⃣ **service_backbone** (服務骨幹模組)

**路徑**: `services/core/aiva_core/service_backbone/`

```
service_backbone/
├── api/                     # REST API
│   └── app.py              # FastAPI 應用
├── coordinator/             # 協調器
├── message_broker/          # 消息中介
├── storage/                 # 存儲層
│   └── backends.py         # 存儲後端
└── state_manager/           # 狀態管理
```

**功能**: API、協調、消息、存儲、狀態

---

## 🔧 外部功能模組 (Features)

**路徑**: `services/features/`

### Python 模組

| 模組 | 功能 | Flows | 狀態 |
|------|------|-------|------|
| function_sqli | SQL 注入檢測 | 32 | ✅ |
| function_xss | XSS 漏洞檢測 | 109 | ✅ |
| function_ssrf | SSRF 漏洞檢測 | 35 | ✅ |
| function_idor | IDOR 訪問控制 | 19 | ✅ |
| function_bizlogic | 業務邏輯漏洞 | 8 | ✅ |
| function_postex | 後滲透 | - | ✅ |

### 多語言模組

| 模組 | 語言 | 功能 | Flows |
|------|------|------|-------|
| function_authn_go | Go | 身份驗證 | 4 |
| function_crypto | Rust | 加密相關 | - |

**總計**: 203 Python + 4 Go = 207 外部流程

---

## 🔍 掃描引擎 (Scan)

**路徑**: `services/scan/`

### 三大引擎

```
scan/
├── python_engine/              # Python 被動掃描
│   ├── passive_analyzer.py    # 被動分析器
│   └── passive_scanner.py     # 被動掃描器
│
├── rust_scanner/               # Rust 主動掃描
│   └── rust_core/             # Rust 核心 (Cargo 項目)
│
└── typescript_engine/          # TypeScript 分析
    ├── analysis_output/       # 分析輸出
    └── src/                   # TypeScript 源碼
```

---

## 📊 統計數據

### 模組分布

| 層級 | 模組數 | 說明 |
|------|--------|------|
| 核心模組 | 5 | cognitive_core, internal_exploration, task_planning, core_capabilities, service_backbone |
| 外部功能 | 8 | Python: 6, Go: 1, Rust: 1 |
| 掃描引擎 | 3 | Python, Rust, TypeScript |
| **總計** | **16** | |

### 流程數量

| 類型 | 數量 | 數據源 |
|------|------|--------|
| 內部流程 | 286 | latest_classification.json |
| 外部流程 | 210 | classification_data.json |
| **總計** | **496** | |

### 語言分布 (外部模組)

| 語言 | Flows | 百分比 |
|------|-------|--------|
| Python | 203 | 96.7% |
| Go | 4 | 1.9% |
| TypeScript | 3 | 1.4% |
| **總計** | **210** | **100%** |

---

## 🔗 執行器架構

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
│ (286 flows)    │  │ (210 flows)      │
│ - Python only  │  │ - Python/Go/TS   │
│ - 調用模組類方法│  │ - 調用模組類方法  │
└────────────────┘  └──────────────────┘
```

**架構說明**:
- AI 通過 ai_executor_interface 調用能力
- Executor 負責路由到正確的模組並調用其類方法
- 無需為模組創建"簡化 API"或 wrapper 函數

---

## 📋 命名規範

### 內部流程
- **格式**: 數字 ID (`1`, `2`, `3`, ..., `286`)
- **數據源**: `latest_classification.json`
- **執行器**: `aiva_internal_executor.py`

### 外部流程
- **格式**: 語言前綴 + 編號
  - Python: `aivapy1`, `aivapy2`, ...
  - Go: `aivago1`, `aivago2`, ...
  - TypeScript: `aivats1`, `aivats2`, ...
- **數據源**: `classification_data.json`
- **執行器**: `aiva_external_executor.py`

---

## 🔄 模組整合歷史

### 2026-01-03: external_learning → cognitive_core/learning_system

**原因**: 學習功能屬於認知核心，統一管理 AI 訓練和模型

**影響**:
- ✅ 六大模組 → 五大模組（架構整合完成）
- ✅ 代碼遷移完成
- ✅ 文檔已全面更新 (2026-02-01)

**遷移路徑**:
```
external_learning/
├── training/          → cognitive_core/learning_system/training/
├── models/            → cognitive_core/learning_system/models/
├── tracking/          → cognitive_core/learning_system/tracking/
└── analysis/          → cognitive_core/learning_system/analysis/
```

---

## 🎯 優先事項

### 立即處理
1. ✅ 統一命名規範已建立
2. ✅ 執行器整合完成
3. ✅ 文檔全面更新完成 (五大模組架構)

### 待完善
1. 外部模組 AST 分析結果生成
2. 能力映射完整性檢查
3. 參數配置功能添加

---

## 📄 相關文檔

- [UNIFIED_NAMING_CONVENTION.md](../UNIFIED_NAMING_CONVENTION.md) - 統一命名規範
- [AI_EXECUTOR_INTEGRATION_COMPLETE.md](../AI_EXECUTOR_INTEGRATION_COMPLETE.md) - 執行器整合報告
- [services/README.md](README.md) - Services 總覽

---

**最後更新**: 2026-02-01  
**維護者**: AIVA 開發團隊  
**更新內容**: internal_loop_connector 修復 + 文檔驗證完成
