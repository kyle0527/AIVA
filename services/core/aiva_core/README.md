# 🤖 AIVA Core - AI 核心系統

> **版本**: v3.0 | **狀態**: ✅ 生產就緒 | **最後更新**: 2025-12-16  
> **角色**: AIVA 的程式化核心服務，提供 AI 認知、任務規劃、能力管理等核心功能

**導航**: [← 返回 Services](../../README.md)

---

## 📋 目錄

- [系統概述](#-系統概述)
- [六大核心模組](#-六大核心模組)
  - [🧠 Cognitive Core - 認知核心](#-cognitive-core---認知核心)
  - [🧭 Internal Exploration - 內部探索](#-internal-exploration---內部探索)
  - [📋 Task Planning - 任務規劃](#-task-planning---任務規劃)
  - [🌍 External Learning - 對外學習](#-external-learning---對外學習)
  - [🎯 Core Capabilities - 核心能力](#-core-capabilities---核心能力)
  - [🏗️ Service Backbone - 服務骨幹](#-service-backbone---服務骨幹)
- [架構特點](#-架構特點)
- [快速開始](#-快速開始)
- [系統統計](#-系統統計)
- [相關服務](#-相關服務)

---

## 🎯 系統概述

AIVA Core 是整個 AIVA 系統的核心大腦，採用**六大模組架構**設計，每個模組負責特定的核心功能，共同構成完整的 AI 決策和執行系統。

### 架構原則
- ✅ **單一數據源 (SOT)**: 遵循 aiva_common 規範，避免數據重複
- ✅ **有錯就報錯**: 不隱藏錯誤，不使用 mock/fake 數據
- ✅ **模組化設計**: 六大模組獨立但協同工作
- ✅ **真實執行**: 所有 840 個能力真實註冊，無模擬數據

### 當前狀態
- ✅ **數據流分類**: 840 條數據流成功分類為 158 個獨特能力
- ✅ **六大模組架構**: 按終點腳本分類，覆蓋所有核心功能
- ✅ **能力註冊系統**: 支援能力查詢和管理
- ✅ **架構統一**: CapabilityRegistry 代理模式完成，SOT 原則落實

---

## 🏛️ 六大核心模組

### 🧠 Cognitive Core - 認知核心
**[📖 查看詳細文檔](cognitive_core/README.md)**

AI 認知智能核心，整合神經網路、決策支援、知識檢索和可靠性驗證。

**核心功能**:
- 🧠 神經網路推理 (RealDecisionEngine, 5M 參數)
- 🎯 智能決策支援 (CapabilityOrchestrator)
- 🔍 RAG 檢索增強 (InternalLoopConnector)
- 🛡️ 反幻覺機制

**關鍵組件**:
- `neural/real_neural_core.py` - 神經網路核心
- `capability_orchestrator.py` - 能力編排器
- `internal_loop_connector.py` - RAG 查詢接口
- `rag/vector_store.py` - 向量資料庫

**統計**: 23 個文件, 10,362 行代碼

---

### 🧭 Internal Exploration - 內部探索
**[📖 查看詳細文檔](internal_exploration/README.md)**

自我分析和能力發現系統，通過三階段分析管道實現自我認知。

**核心功能**:
- 📊 代碼流程分析 (ExplorationPipeline)
- 🏷️ 能力自動分類 (六大模組分類)
- 🔄 能力註冊同步
- 📈 系統健康自檢

**三階段管道**:
1. **Analyzer** - AST 分析與數據流發現
2. **Classifier** - 數據流分類與統計
3. **Diff** - 增量更新與變化檢測

**統計**: 7 個文件, 3,918 行代碼, ✅ 100% 完成度

---

### 📋 Task Planning - 任務規劃
**[📖 查看詳細文檔](task_planning/README.md)**

智能任務規劃和執行系統，負責目標分解和執行協調。

**核心功能**:
- 📋 AI 驅動的任務分解
- ⚡ 並行執行管理
- 🔄 動態計劃調整
- 📊 執行進度追蹤

**關鍵組件**:
- `ai_commander.py` - AI 指揮官
- `planner/execution_planner.py` - 任務規劃器
- `executor/task_executor.py` - 任務執行器
- `command_router.py` - 命令路由器

**統計**: 17 個文件, 6,525 行代碼, ✅ 85% 完成度

---

### 🌍 External Learning - 對外學習
**[📖 查看詳細文檔](external_learning/README.md)**

> ⚠️ **架構更新 (2025-12-17)**: 訓練系統已簡化  
> - ❌ TrainingOrchestrator 已廢棄（包含 40+ 錯誤）  
> - ✅ 學習功能已整合至 UnifiedAttackExecutor（task_planning 模組）  
> - ✅ 代碼量 -47%，數據利用率 +10x，學習覆蓋 100%

持續學習和知識整合系統，從攻擊執行中自動學習並優化。

**核心功能**:
- 🎓 **自動學習和優化** - 透過 UnifiedExecutor 自動觸發
  * 靶場與實戰統一執行
  * 自動收集攻擊經驗
  * 累積到閾值（100 樣本）自動訓練
- 📚 **經驗知識管理** - ExperienceManager
  * 經驗回放緩衝區
  * 採樣和批量訓練
  * 長期記憶管理
- 🔄 **持續學習機制**
  * 監督學習（成功案例）
  * 強化學習（獎勵信號）
  * 權重自動持久化

**關鍵組件**:
- `learning/model_trainer.py` - 模型訓練器
- `learning/experience_manager.py` - 經驗管理器
- ❌ `training/training_orchestrator.py` - 訓練編排器（已廢棄）

**整合點**:
- Task Planning 透過 `unified_executor.py` 統一執行和學習
- Cognitive Core 提供 RAG 檢索相似攻擊案例
- Core Capabilities 執行實際攻擊並回傳結果

**統計**: 17 個文件, 7,398 行代碼, ✅ 架構已簡化

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
- `capability_registry.py` - 能力註冊表 (代理模式)
- `attack/` - 攻擊能力實現
- `analysis/` - 分析能力實現

**統計**: 22 個文件, 8,580 行代碼, ⚠️ 65% 完成度

**重要更新**: CapabilityRegistry 現在作為 integration.CapabilityRegistry 的代理，遵循 SOT 原則。

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
- `monitoring/optimized_core.py` - 性能監控

**統計**: 33 個文件, 10,613 行代碼, ✅ 完成

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
按六大模組組織
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

### 代碼規模
| 模組 | 文件數 | 代碼行數 | 占比 | 完成度 |
|------|--------|----------|------|--------|
| **service_backbone** | 33 | 10,613 | 22.4% | ✅ 100% |
| **cognitive_core** | 23 | 10,362 | 21.9% | ✅ 95% |
| **core_capabilities** | 22 | 8,580 | 18.1% | ⚠️ 65% |
| **external_learning** | 17 | 7,398 | 15.6% | ⚠️ 60% |
| **task_planning** | 17 | 6,525 | 13.8% | ✅ 85% |
| **internal_exploration** | 7 | 3,918 | 8.3% | ✅ 100% |
| **總計** | **119** | **47,396** | **100%** | **✅ 82%** |

### 能力統計
| 指標 | 數值 | 狀態 |
|------|------|------|
| **數據流總數** | 840 | ✅ |
| **獨特能力數** | 158 | ✅ |
| **平均每能力流數** | 5.3 | ✅ |
| **多路徑能力** | 103 (65.2%) | ✅ |
| **模組分佈** | 6 個模組 | ✅ |

### 模組能力分佈（按能力數）
- `service_backbone`: 37 個能力 / 163 條流 (19.4%)
- `internal_exploration`: 19 個能力 / 201 條流 (23.9%)
- `cognitive_core`: 29 個能力 / 189 條流 (22.5%)
- `task_planning`: 33 個能力 / 143 條流 (17.0%)
- `external_learning`: 24 個能力 / 99 條流 (11.8%)
- `core_capabilities`: 16 個能力 / 45 條流 (5.4%)

**總計**: 158 個能力，840 條數據流

---

## 📂 目錄結構

```
aiva_core/
├── cognitive_core/              # 🧠 認知核心 (23 文件, 10K+ 行)
│   ├── neural/                  # 神經網路
│   ├── rag/                     # RAG 系統
│   ├── capability_orchestrator.py  # 能力編排
│   └── internal_loop_connector.py  # RAG 查詢接口
│
├── internal_exploration/        # 🧭 內部探索 (7 文件, 4K 行) ✅
│   ├── python_tools/            # 三階段分析管道
│   │   ├── aiva_exploration_pipeline.py  # 主管道
│   │   ├── aiva_flow_analyzer.py    # 階段 1: 分析
│   │   ├── aiva_flow_classifier.py  # 階段 2: 分類
│   │   └── aiva_cli_implementation.py  # 階段 3: 實作
│   └── README.md
│
├── task_planning/               # 📋 任務規劃 (17 文件, 6.5K 行)
│   ├── ai_commander.py          # AI 指揮官
│   ├── planner/                 # 規劃器
│   ├── executor/                # 執行器
│   └── command_router.py        # 命令路由
│
├── external_learning/           # 🌍 對外學習 (17 文件, 7.4K 行)
│   ├── ai_model/                # 模型訓練
│   ├── learning/                # 經驗管理
│   └── analysis/                # 數據處理
│
├── core_capabilities/           # 🎯 核心能力 (22 文件, 8.6K 行)
│   ├── capability_registry.py   # 能力註冊 (代理模式)
│   ├── attack/                  # 攻擊能力
│   └── analysis/                # 分析能力
│
├── service_backbone/            # 🏗️ 服務骨幹 (33 文件, 10.6K 行) ✅
│   ├── api/                     # API 網關
│   ├── messaging/               # 消息系統
│   ├── coordination/            # 服務協調
│   ├── monitoring/              # 性能監控
│   └── storage/                 # 存儲服務
│
└── README.md                    # 本文件
```

---

## 🎯 重要更新 (2026-01-02)

### ✅ v3.2 能力分類系統完成

1. **終點腳本分類法 (file_path_exact_match)**
   - ✅ 840 條數據流按終點腳本分類到六大模組
   - ✅ 識別出 158 個獨特能力
   - ✅ 103 個多路徑能力 (65.2%)
   - ✅ 平均每能力 5.3 條執行路徑

2. **六大模組分布**
   - 認知核心: 29 能力 / 189 流 (22.5%)
   - 內部探索: 19 能力 / 201 流 (23.9%)
   - 任務規劃: 33 能力 / 143 流 (17.0%)
   - 外部學習: 24 能力 / 99 流 (11.8%)
   - 核心能力: 16 能力 / 45 流 (5.4%)
   - 服務骨幹: 37 能力 / 163 流 (19.4%)

3. **生成的完整文檔**
   - 📄 `docs/六大模組能力分類清單_統一編號.md` (392 KB) - 所有 840 條流統一編號
   - 📁 `docs/各模組詳細說明/` - 6 個模組各自的詳細說明文檔
   - 📊 `docs/六大模組完整詳細說明總覽.md` - 全局統計與導航
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
1. 🔴 **HIGH**: cognitive_core (完善 RAG 查詢)
2. 🟡 **MEDIUM**: task_planning (優化執行器)
3. 🟢 **LOW**: external_learning (增強訓練)

---

**最後更新**: 2025-12-16  
**維護者**: AIVA Team  
**版本**: v3.0 (六大模組架構)
