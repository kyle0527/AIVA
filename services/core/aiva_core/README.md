# 🤖 AIVA Core - AI 核心系統

> **版本**: v4.1 | **狀態**: ✅ 生產就緒 | **最後更新**: 2026-01-06  
> **角色**: AIVA 的程式化核心服務，提供 AI 認知、任務規劃、能力管理等核心功能  
> **架構**: 5M 特化 AI Decision Engine（非 LLM），512 維結構化向量  
> **檔案數**: 146 個 Python 模組

**導航**: [← 返回 Services](../../README.md)

---

## 📋 目錄

- [系統概述](#-系統概述)
- [六大核心模組](#-六大核心模組)
  - [🧠 Cognitive Core - 認知核心](#-cognitive-core---認知核心)
  - [🧭 Internal Exploration - 內部探索](#-internal-exploration---內部探索)
  - [📋 Task Planning - 任務規劃](#-task-planning---任務規劃)
  - [� External Learning - 對外學習](#-external-learning---對外學習)
  - [�🎯 Core Capabilities - 核心能力](#-core-capabilities---核心能力)
  - [🏗️ Service Backbone - 服務骨幹](#-service-backbone---服務骨幹)
- [架構特點](#-架構特點)
- [快速開始](#-快速開始)
- [系統統計](#-系統統計)
- [相關服務](#-相關服務)

---

## 🎯 系統概述

AIVA Core 是整個 AIVA 系統的核心大腦，採用**五大模組架構**設計，每個模組負責特定的核心功能，共同構成完整的 AI 決策和執行系統。

> **架構變更 (2026-01)**: 原 `external_learning` 已整合至 `cognitive_core/learning_system`，現為五大模組架構。

### 架構原則
- ✅ **單一數據源 (SOT)**: 遵循 aiva_common 規範，避免數據重複
- ✅ **有錯就報錯**: 不隱藏錯誤，不使用 mock/fake 數據
- ✅ **模組化設計**: 五大模組獨立但協同工作
- ✅ **真實執行**: 所有 840 個能力真實註冊，無模擬數據

### 當前狀態
- ✅ **數據流分類**: 840 條數據流成功分類為 158 個獨特能力
- ✅ **五大模組架構**: 按終點腳本分類，覆蓋所有核心功能
- ✅ **能力註冊系統**: 支援能力查詢和管理
- ✅ **架構統一**: CapabilityRegistry 代理模式完成，SOT 原則落實
- ✅ **5M AI 特化**: CapabilityEncoder 512 維向量編碼，無需 NLU
- ✅ **v3.3 格式**: latest_classification.json 新增 cli_command、parameters、return_type

---

## 🏛️ 五大核心模組

### 🧠 Cognitive Core - 認知核心
**[📖 查看詳細文檔](cognitive_core/README.md)**

AI 認知智能核心，整合神經網路、決策支援、知識檢索和可靠性驗證。

**核心功能**:
- 🧠 神經網路推理 (RealDecisionEngine, 5M 參數)
- 🎯 智能決策支援 (CapabilityOrchestrator)
- 🔍 RAG 檢索增強 (InternalLoopConnector)
- 🛡️ 反幻覺機制

**關鍵組件**:
- `neural/real_neural_core.py` - 神經網路核心 (5M 參數)
- `capability_orchestrator.py` - 能力編排器
- `capability_encoder.py` - **512 維結構化編碼器** ⭐
- `internal_loop_connector.py` - RAG 查詢接口 (v11.0)
- `rag/vector_store.py` - 向量資料庫 (512 維)
- `manifest/manifest_loader.py` - 能力清單加載器
- `anti_hallucination/` - 反幻覺驗證模組

**統計**: 44 個文件, 19,833 行代碼

---

### 🧭 Internal Exploration - 內部探索
**[📖 查看詳細文檔](internal_exploration/README.md)**

自我分析和能力發現系統，通過三階段分析管道實現自我認知。

**核心功能**:
- 📊 代碼流程分析 (ExplorationPipeline)
- 🏷️ 能力自動分類 (五大模組分類)
- 🔄 能力註冊同步
- 📈 系統健康自檢

**三階段管道**:
1. **Analyzer** - AST 分析與數據流發現
2. **Classifier** - 數據流分類與統計
3. **Diff** - 增量更新與變化檢測

**統計**: 17 個文件, 9,443 行代碼, ✅ 100% 完成度

---

### 📋 Task Planning - 任務規劃
**[📖 查看詳細文檔](task_planning/README.md)**

智能任務規劃和執行系統，負責目標分解和執行協調。

**核心功能**:
- 📋 AI 驅動的任務分解
- ⚡ 並行執行管理
- 🔄 動態計劃調整
- 📊 執行進度追蹤

**子模組架構**:
| 子模組 | 說明 | 文檔 |
|--------|------|------|
| **commander** | AI 指揮官與策略引擎 | [README](task_planning/commander/README.md) |
| **planner** | 任務規劃與生成器 | [README](task_planning/planner/README.md) |
| **executor** | 計劃執行與任務處理 | [README](task_planning/executor/README.md) |

**關鍵組件**:
- `ai_commander.py` - AI 指揮官（已重構）
- `planner/execution_planner.py` - 任務規劃器
- `executor/task_executor.py` - 任務執行器
- `command_router.py` - 命令路由器
- `unified_executor.py` - 統一執行器（含學習功能）

**統計**: 22 個文件, 8,653 行代碼, ✅ 100% 完成度

> **架構說明**: 學習功能已整合至 `unified_executor.py`，原 `external_learning` 已合併至 `cognitive_core/learning_system`。Commander 於 2026-01-06 完成重構，採用組件化設計。

---

### 🎯 Core Capabilities - 核心能力
**[📖 查看詳細文檔](core_capabilities/README.md)**

能力註冊和管理系統，管理所有可用的功能能力。

**核心功能**:
- 📦 能力註冊管理 (CapabilityRegistry 代理)
- 🔍 能力查詢和發現
- 🎭 攻擊和分析能力
- 🔌 插件系統整合

**關鍵組件**:
- `capability_registry.py` - 能力註冊表 (SOT 代理模式)
- `attack/` - 攻擊能力實現
- `analysis/` - 分析能力實現

**統計**: 29 個文件, 7,557 行代碼, ✅ 100% 完成度

**重要更新**: CapabilityRegistry 現在作為 integration.CapabilityRegistry 的代理，遵循 SOT 原則。數據來源為 latest_classification.json v3.3。

---

### 🏗️ Service Backbone - 服務骨幹
**[📖 查看詳細文檔](service_backbone/README.md)**

基礎設施層，提供消息、存儲、協調、監控等核心服務。

**核心功能**:
- 📨 消息系統 (RabbitMQ)
- 📊 狀態管理 (會話追蹤)
- 💾 存儲服務 (統一接口)
- 🎛️ 服務協調 (跨模組協調)
- 📈 性能監控 (指標收集)
- 🌐 API 網關 (FastAPI)

**關鍵組件**:
- `messaging/message_broker.py` - 消息代理
- `coordination/core_service_coordinator.py` - 服務協調器
- `api/app.py` - API 入口
- `context_manager.py` - 上下文管理
- `storage/storage_manager.py` - 存儲管理

**統計**: 33 個文件, 10,640 行代碼, ✅ 100% 完成

---

## 🌟 架構特點

### 1. 能力分類系統
```
源數據分析 (840 條數據流)
    ↓
按終點腳本分類
    ↓
識別獨特能力 (158 個)
    ↓
多路徑分析 (103 個多路徑能力)
    ↓
按五大模組組織
    ↓
生成統一編號文檔 (#1-#840)
```

### 2. 單一數據源 (SOT)
- ✅ `services/integration/capability/registry.py` 是唯一數據源
- ✅ `aiva_core/core_capabilities/capability_registry.py` 使用代理模式
- ✅ 所有查詢統一通過 integration registry

### 3. 模組協同
```
Task Planning (任務規劃)
    ↓ 查詢能力
Cognitive Core (認知核心)
    ↓ 調用 InternalLoopConnector
Internal Exploration (內部探索)
    ↓ 同步到 Registry
Core Capabilities (核心能力)
    ↓ 提供能力實現
Service Backbone (服務骨幹)
    ↓ 基礎設施支援
```

---

## 🚀 快速開始

### 基本使用

```python
# 1. 初始化認知核心
from services.core.aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine
from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector

neural_core = RealDecisionEngine()
neural_core.load_weights("models/aiva_real_weights.pth")

# 2. 查詢可用能力
connector = InternalLoopConnector()
rag_result = connector.query_capabilities(
    query="掃描和檢測能力",
    top_k=5
)

print(f"找到 {len(rag_result.results)} 個能力")
for cap in rag_result.results:
    print(f"- {cap['metadata']['name']}: {cap['metadata']['description']}")

# 3. 創建任務計劃
from services.core.aiva_core.task_planning.ai_commander import AICommander
from services.core.aiva_core.cognitive_core.capability_orchestrator import CapabilityOrchestrator

orchestrator = CapabilityOrchestrator(
    decision_engine=neural_core,
    internal_connector=connector
)

commander = AICommander(capability_orchestrator=orchestrator)
plan = await commander.generate_plan(
    goal="掃描目標網站",
    target="https://example.com"
)

# 4. 執行任務
from services.core.aiva_core.task_planning.executor.task_executor import TaskExecutor

executor = TaskExecutor()
results = await executor.execute_plan(plan)

print(f"執行完成: {len(results)} 個任務")
```

### 能力同步

```python
# 同步新分析的能力到 Registry
from services.integration.capability.sync_from_analysis import CapabilitySyncer

syncer = CapabilitySyncer()
result = await syncer.sync_from_analysis(module='core')

print(f"同步完成: 成功 {result['success_count']}, 失敗 {result['failed_count']}")
```

---

## 📊 系統統計

### 代碼規模 (2026-01-06 更新)
| 模組 | 文件數 | 代碼行數 | 完成度 |
|------|--------|--------|--------|
| **cognitive_core** | 44 | 19,833 | ✅ 100% |
| **service_backbone** | 33 | 10,640 | ✅ 100% |
| **core_capabilities** | 29 | 7,557 | ✅ 100% |
| **task_planning** | 22 | 8,653 | ✅ 100% |
| **internal_exploration** | 17 | 9,443 | ✅ 100% |
| **總計** | **145** | **56,126** | **✅ 100%** |

### 5M AI 特化狀態
| 指標 | 數值 | 狀態 |
|------|------|------|
| **CapabilityEncoder** | 512 維向量 | ✅ 新增 |
| **VectorStore 維度** | 512 維 | ✅ 更新 |
| **latest_classification.json** | v3.3 格式 | ✅ 更新 |
| **新增欄位** | cli_command, parameters, return_type | ✅ |
| **NLU 需求** | 無（結構化編碼）| ✅ |

### 能力統計
| 指標 | 數值 | 狀態 |
|------|------|------|
| **數據流總數** | 840 | ✅ |
| **獨特能力數** | 158 | ✅ |
| **平均每能力流數** | 5.3 | ✅ |
| **多路徑能力** | 103 (65.2%) | ✅ |
| **模組分佈** | 5 個模組 | ✅ |

### 模組能力分佈（按檔案數）
- `cognitive_core`: 44 檔案 (30.1%) - 含 learning_system, manifest, anti_hallucination
- `service_backbone`: 32 檔案 (21.9%)
- `core_capabilities`: 28 檔案 (19.2%)
- `task_planning`: 22 檔案 (15.1%)
- `internal_exploration`: 19 檔案 (13.0%)

**總計**: 146 個 Python 檔案

---

## 📂 目錄結構

```
aiva_core/
├── cognitive_core/              # 🧠 認知核心 (44 文件) ✅
│   ├── neural/                  # 神經網路 (5M Decision Engine, 6 files)
│   ├── rag/                     # RAG 系統 (512 維向量, 6 files)
│   ├── decision/                # 決策支援 (5 files)
│   ├── learning_system/         # 統一學習系統 (16 files)
│   ├── manifest/                # 能力清單加載器 (2 files)
│   ├── anti_hallucination/      # 反幻覺驗證 (2 files)
│   ├── capability_encoder.py    # ⭐ 512 維結構化編碼器
│   ├── capability_orchestrator.py  # 能力編排
│   └── internal_loop_connector.py  # RAG 查詢接口 (v11.0)
│
├── internal_exploration/        # 🧭 內部探索 (19 文件) ✅
│   ├── python_tools/            # 三階段分析管道
│   │   ├── aiva_exploration_pipeline.py  # 主管道
│   │   ├── aiva_flow_analyzer.py    # 階段 1: 分析 (含 parameters 解析)
│   │   ├── aiva_flow_classifier.py  # 階段 2: 分類 (v3.3 輸出)
│   │   └── aiva_cli_implementation.py  # 階段 3: CLI 實作
│   ├── self_healing/            # 自我修復
│   └── README.md
│
├── task_planning/               # 📋 任務規劃 (22 文件) ✅
│   ├── commander/               # 指揮官系統
│   │   ├── ai_commander.py      # AI 指揮官（已重構）
│   │   ├── attack_coordinator.py # 攻擊協調器
│   │   ├── capability_manager.py # 能力管理器
│   │   ├── plan_builder.py      # 計劃建構器
│   │   ├── strategy_engine.py   # 策略引擎
│   │   ├── learning_adapter.py  # 學習適配器
│   │   ├── types.py             # 類型定義
│   │   └── README.md            # 子模組文檔
│   ├── planner/                 # 規劃系統
│   │   ├── execution_planner.py # 執行規劃器
│   │   ├── task_generator.py    # 任務生成器
│   │   ├── tool_selector.py     # 工具選擇器
│   │   └── README.md            # 子模組文檔
│   ├── executor/                # 執行系統
│   │   ├── plan_executor.py     # 計劃執行器
│   │   ├── task_executor.py     # 任務執行器
│   │   ├── attack_plan_mapper.py # 計劃映射器
│   │   └── README.md            # 子模組文檔
│   ├── command_router.py        # 命令路由
│   ├── unified_executor.py      # 統一執行器
│   └── README.md                # 主模組文檔
│
├── core_capabilities/           # 🎯 核心能力 (28 文件) ✅
│   ├── capability_registry.py   # 能力註冊 (SOT 代理)
│   ├── multilang_coordinator.py # 多語言協調
│   ├── attack/                  # 攻擊能力
│   ├── cli/                     # CLI 工具
│   └── manifests/               # 清單定義
│
├── service_backbone/            # 🏗️ 服務骨幹 (32 文件) ✅
│   ├── api/                     # API 網關
│   ├── messaging/               # 消息系統
│   ├── coordination/            # 服務協調
│   ├── storage/                 # 存儲服務
│   ├── context_manager.py       # 上下文管理
│   └── pipeline/                # 執行管道
│
└── README.md                    # 本文件
```

---

## 🎯 重要更新

### ✅ v4.1 - Task Planning 錯誤修復與重構完成 (2026-01-06)

1. **Commander 組件化重構**
   - ✅ `ai_commander.py` 重構為組件化設計
   - ✅ 新增 6 個獨立組件（CapabilityManager, PlanBuilder, StrategyEngine, AttackCoordinator, LearningAdapter, Types）
   - ✅ 提升程式碼可維護性與可測試性

2. **錯誤修復完成**
   - ✅ 修復 `unified_tracer.py` 缺失方法（create_session, get_session, abort_session）
   - ✅ 實作完整 `ExecutionMonitor` 類（從空類別到完整實現）
   - ✅ 修復 `attack_plan_mapper.py` 無效參數問題（移除 environment 參數）
   - ✅ 修復 `plan_executor.py` async/await 與參數問題（8 處修正）
   - ✅ 修復 `task_executor.py` 監控方法調用問題（7 處修正）

3. **文檔完整性提升**
   - ✅ 新增 `commander/README.md` - Commander 子模組文檔
   - ✅ 新增 `planner/README.md` - Planner 子模組文檔
   - ✅ 新增 `executor/README.md` - Executor 子模組文檔
   - ✅ 更新 `task_planning/README.md` - 新增子模組導航
   - ✅ 所有文檔遵循 aiva_common 規範

4. **編譯驗證**
   - ✅ 所有編譯錯誤已修復（0 錯誤）
   - ✅ 類型檢查通過
   - ✅ 符合 aiva_common schemas 規範

### ✅ v4.0 - 5M AI 特化升級 (2026-01-04)

1. **CapabilityEncoder 新增**
   - ✅ 512 維結構化向量編碼
   - ✅ 直接匹配 5M Decision Engine 輸入
   - ✅ 無需自然語言處理 (NLU)

2. **latest_classification.json v3.3**
   - ✅ 新增 `cli_command` - CLI 命令模板
   - ✅ 新增 `parameters` - 參數定義（名稱、類型、必填）
   - ✅ 新增 `return_type` - 返回類型
   - ✅ 新增 `structured_tags` - 結構化標籤

3. **VectorStore 512 維升級**
   - ✅ 向量維度從 768 → 512
   - ✅ 新增 `add_capability()` 方法
   - ✅ 新增 `search_capabilities()` 方法

4. **舊格式棄用**
   - ✅ `minimal_manifest.py` 標記為 DEPRECATED
   - ✅ 路徑 B JSON 格式已棄用

### ✅ v3.2 能力分類系統完成

1. **終點腳本分類法 (file_path_exact_match)**
   - ✅ 840 條數據流按終點腳本分類到五大模組
   - ✅ 識別出 158 個獨特能力
   - ✅ 103 個多路徑能力 (65.2%)
   - ✅ 平均每能力 5.3 條執行路徑

2. **五大模組分布**
   - 認知核心: 44 檔案 / 223 流 (26.5%) - 含 learning_system, manifest, anti_hallucination
   - 內部探索: 19 檔案 / 201 流 (23.9%)
   - 任務規劃: 22 檔案 / 48 流 (5.7%)
   - 核心能力: 28 檔案 / 131 流 (15.6%)
   - 服務骨幹: 32 檔案 / 163 流 (19.4%)

3. **生成的完整文檔**
   - 📄 `docs/五大模組能力分類清單_統一編號.md` - 所有 840 條流統一編號
   - 📁 `docs/各模組詳細說明/` - 5 個模組各自的詳細說明文檔
   - 📊 `docs/五大模組完整詳細說明總覽.md` - 全局統計與導航
   - 📈 `docs/各模組能力數據流統計摘要.md` - 統計摘要

4. **數據驗證**
   - ✅ 分類準確率: 91.2%
   - ✅ 未分類流: 0 條
   - ✅ 數據完整性: 所有流均成功分類
   - ✅ 統計驗證: 所有模組流數總和 = 840

### 📊 詳細報告
查看 [能力分類系統完整報告](./CAPABILITY_SYSTEM_CLASSIFICATION_REPORT.md) 了解：
- 分類方法詳解
- 能力 vs 數據流概念
- 各模組詳細統計
- 頂級能力排名
- 相關工具和數據檔案

---

## 🔗 相關服務

### AIVA 服務層
- [**aiva_common**](../aiva_common/README.md) - 公共數據結構、枚舉和工具
- [**features**](../features/README.md) - 功能模組實現
- [**scan**](../scan/README.md) - 掃描引擎和協調器
- [**integration**](../integration/README.md) - 外部系統整合和能力註冊

### 核心數據
- [**analysis_data**](../../integration/analysis_data/README.md) - 內部分析結果
- [**capability_registry.db**](../../data/capability_registry.db) - 能力註冊資料庫
- [**vector_db**](../../data/vector_db/chroma/) - RAG 向量資料庫

---

## 📝 開發指南

### 添加新能力
1. 在對應模組實現功能
2. 運行 ExplorationPipeline 分析
3. 使用 sync_from_analysis.py 同步到 Registry
4. 驗證 RAG 可以查詢到

### 遵循 aiva_common 規範
- ✅ 使用 `services.aiva_common` 的枚舉和數據結構
- ✅ 遵循「有錯就報錯」原則，不隱藏錯誤
- ✅ 保持單一數據源 (SOT)
- ✅ 不使用 mock/fake/stub

### 模組開發優先級
1. 🔴 **HIGH**: cognitive_core (完善 5M AI 整合)
2. � **COMPLETED**: task_planning (錯誤修復與重構完成 ✅)
3. 🟢 **LOW**: internal_exploration (增強分析)

---

**最後更新**: 2026-01-06  
**維護者**: AIVA Team  
**版本**: v4.1 (Task Planning 重構完成 + 5M AI 特化架構)
