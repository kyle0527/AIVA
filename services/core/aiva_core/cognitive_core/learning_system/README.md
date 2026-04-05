# 📚 Learning System - 統一學習系統

> **路徑**: `cognitive_core/learning_system/`  
> **狀態**: ✅ 正常 | **最後更新**: 2026-04-05  
> **子模組**: 4 個 | **總 Python 文件數**: 18  
> **父模組**: [Cognitive Core](../README.md)

## 概述

**Learning System** 是 AIVA 的經驗學習系統，負責從執行結果和用戶回饋中學習，並優化 AI 決策策略。整合分析引擎、學習系統、執行追蹤、訓練編排和知識管理五大子系統，實現持續自我優化。

> **說明**: 此模組整合自原 `external_learning`，現為 cognitive_core 的子模組。

**核心職責**：
- 📊 **結果分析** - 分析執行結果，提取學習信號
- 🧠 **策略優化** - 基於學習結果優化決策模型
- 🎯 **經驗管理** - 管理歷史經驗，支持決策推理
- 📝 **執行追蹤** - 追蹤跨模組執行狀態，收集性能數據
- 📚 **知識管理** - 模組知識庫與三路比對評估

---

## 子模組結構

| 子模組 | 功能 | 文件數 | 文檔 |
|--------|------|--------|------|
| [analysis/](analysis/README.md) | 動態策略調整與 AST 軌跡比較 | 2 | [README](analysis/README.md) |
| [knowledge/](knowledge/README.md) | 模組知識管理與三路比對 | 1 | [README](knowledge/README.md) |
| [learning/](learning/README.md) | 模型訓練、強化學習、生物神經訓練 | 7 | [README](learning/README.md) |
| [tracing/](tracing/README.md) | 執行追蹤與軌跡記錄 | 3 | [README](tracing/README.md) |
| [training/](training/README.md) | 場景管理與訓練編排 | 2 | [README](training/README.md) |

### 根目錄組件

- `event_listener.py` - 學習事件監聽器，監聽 TASK_COMPLETED 事件觸發學習
- `experience_manager.py` - 經驗管理器，基於強化學習的經驗重放機制

---

## 主要類別

| 類別 | 文件 | 說明 |
|------|------|------|
| `ExternalLearningListener` | event_listener.py | 外部學習監聽器，連接執行系統和學習系統 |
| `ExperienceManager` | experience_manager.py | 經驗管理器，管理攻擊執行經驗的記錄和採樣 |
| `ModuleKnowledgeManager` | knowledge/module_knowledge_manager.py | 模組知識管理器，三路比對評估 |
| `StrategyAdjuster` | analysis/dynamic_strategy_adjustment.py | 動態策略調整器 |
| `ModelTrainer` | learning/model_trainer.py | 模型訓練器 |
| `TraceRecorder` | tracing/trace_recorder.py | 軌跡記錄器 |

---

## 依賴關係

**外部依賴**：
- `aio_pika` - 異步消息隊列
- `pydantic` - 數據驗證

**內部依賴**：
- `aiva_common.schemas.dual_loop` - 雙閉環數據模型
- `aiva_common.enums` - 枚舉定義
- `service_backbone.messaging` - 消息代理
- `cognitive_core.external_loop_connector` - 外部閉環連接器

---

**導航**: [← 返回 Cognitive Core](../README.md)

---

## 📑 詳細目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心職責](#核心職責)
  - [設計理念](#設計理念)
- [🏗️ 架構設計](#-架構設計)
  - [學習流程](#學習流程)
- [🔧 核心組件](#-核心組件)
  - [1. 📊 Analysis (分析引擎)](#1--analysis-分析引擎)
  - [2. 🧠 Learning (學習系統)](#2--learning-學習系統)
  - [3. 🎯 Training (訓練編排)](#3--training-訓練編排)
  - [4. 🎧 Event Listener (事件監聽)](#4--event-listener-事件監聽)
  - [5. 📝 Tracing (執行追蹤)](#5--tracing-執行追蹤)
- [📖 使用範例](#-使用範例)
  - [完整的學習流程](#完整的學習流程)
  - [事件驅動學習流程](#事件驅動學習流程)
- [🛠️ 開發指南](#-開發指南)
  - [🔨 aiva_common 修復規範](#-aiva_common-修復規範)
  - [添加新的調整策略](#添加新的調整策略)
  - [實現自定義訓練器](#實現自定義訓練器)
- [📊 性能指標](#-性能指標)
  - [策略調整](#策略調整)
  - [模型訓練](#模型訓練)
  - [執行追蹤](#執行追蹤)
- [🔗 相關模組](#-相關模組)

---

**導航**: [← 返回 Cognitive Core](../README.md)

> **說明**: 此模組整合自原 `external_learning`，現為 cognitive_core 的子模組

---

## 🎯 模組概述

**Learning System** 是 AIVA 的經驗學習系統，負責從執行結果和用戶回饋中學習，並優化AI決策策略。主要學習功能已整合至 `task_planning/unified_executor.py`。

### 核心職責

- 📊 **結果分析** - 分析Features模組執行結果，提取學習信號
- 🧠 **策略優化** - 基於學習結果優化攻擊策略和決策模型
- 🎯 **經驗管理** - 管理歷史經驗和知識庫，支持決策推理
- 🎧 **事件監聽** - 監聽系統事件，觸發自適應學習流程
- 📝 **執行追蹤** - 追蹤跨模組執行狀態，收集性能數據
- 🔄 **反饋循環** - 建立學習-優化-驗證的閉環系統
- ✅ 所有攻擊執行都自動收集經驗並持續學習（靶場 = 實戰）
- ✅ 數據利用率提升 10 倍（從月收集 500 樣本 → 5000 樣本）

> **注意**: 此模組原名 `external_learning`，現已整合至 `cognitive_core/learning_system`。所有導入路徑請使用新路徑。

### 設計理念
- **持續學習** - 從每次執行中學習並優化
- **自適應調整** - 根據環境變化動態調整策略
- **知識積累** - 將經驗轉化為知識並復用
- **性能提升** - 通過訓練不斷提升檢測能力

---

## 🏗️ 架構設計

```
cognitive_core/learning_system/
├── 📁 analysis/                  # 分析引擎
│   ├── dynamic_strategy_adjustment.py  # ✅ 動態策略調整器
│   └── ast_trace_comparator.py         # AST 軌跡比較器
│
├── 📁 learning/                  # 學習系統
│   ├── model_trainer.py          # ✅ 模型訓練器
│   ├── rl_models.py              # 強化學習模型
│   ├── rl_trainers.py            # 強化學習訓練器
│   └── scalable_bio_trainer.py   # 可擴展生物神經訓練器
│
├── 📁 tracing/                   # 執行追蹤
│   ├── execution_tracer.py       # 執行追蹤器
│   ├── trace_recorder.py         # 軌跡記錄器
│   └── unified_tracer.py         # 統一追蹤器
│
├── event_listener.py             # ✅ 學習事件監聽器
└── README.md                     # 本文檔
```

> **已移除組件 (2025-12-18)**:
> - ❌ `risk_assessment_engine.py` - 改用 RAG 動態查詢
> - ❌ `train_classifier.py` - 訓練工具非運行時能力
> - ❌ `training/` 目錄 - 整合至 `task_planning/unified_executor.py`

### 學習流程
```
┌─────────────────────────────────────────────────────────┐
│          Learning System (學習系統)                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │       Execution Tracing (執行追蹤)                │  │
│  │    記錄每次攻擊的執行軌跡和結果                    │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │      Strategy Adjustment (策略調整)               │  │
│  │  基於結果和上下文動態調整測試策略                  │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │     Experience Collection (經驗收集)              │  │
│  │   將執行結果轉化為訓練經驗                         │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │       Model Training (模型訓練)                   │  │
│  │  使用收集的經驗訓練強化學習模型                    │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │      Model Deployment (模型部署)                  │  │
│  │    將訓練好的模型部署到生產環境                    │  │
│  └──────────────────────────────────────────────────┘  │
│                       │                                │
└───────────────────────┼────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
     ┌────▼────┐                 ┌───▼────┐
     │Cognitive│                 │  Task  │
     │  Core   │                 │Planning│
     └─────────┘                 └────────┘
```

---

## 🔧 核心組件

⚠️ **更新 (2025-12-18)**: 
- ❌ `risk_assessment_engine.py` 已移除，應改用 RAG 動態查詢風險資訊
- ❌ `train_classifier.py` 已移除，為訓練工具非運行時能力

### 1. 📊 Analysis (分析引擎)

#### `dynamic_strategy_adjustment.py` - 動態策略調整器
**功能**: 基於執行結果和上下文動態調整測試策略
```python
from cognitive_core.learning_system.analysis import StrategyAdjuster

# 初始化策略調整器
adjuster = StrategyAdjuster()

# 調整策略
adjusted_plan = adjuster.adjust(
    plan=base_plan,
    context={
        "scan_id": "scan_001",
        "waf_detected": True,
        "waf_type": "Cloudflare",
        "fingerprints": {"framework": "Django", "database": "PostgreSQL"},
        "findings_count": 3,
        "completed_tasks": 10,
        "total_tasks": 50
    }
)

print(f"調整後的計畫: {adjusted_plan}")

# 從結果中學習
adjuster.learn_from_result({
    "scan_id": "scan_001",
    "module": "sqli",
    "success": True,
    "payload": "' OR '1'='1",
    "waf_bypassed": True,
    "technique": "union_injection"
})
```

**調整策略**:
- ✅ WAF 適應調整 - 檢測到 WAF 時調整 Payload 編碼
- ✅ 歷史成功率調整 - 根據過往成功率調整優先級
- ✅ 技術棧適應 - 根據目標技術棧選擇合適的攻擊向量
- ✅ 發現數量調整 - 已發現漏洞時調整測試深度
- ✅ 進度感知調整 - 根據執行進度動態調整策略

#### ❌ `risk_assessment_engine.py` - 風險評估引擎 (已移除)
**狀態**: 已於 2025-12-18 移除  
**理由**: 硬編碼規則，應改用 RAG 動態查詢最新風險資訊  
**替代方案**: 使用 `BioNeuronRAGAgent` 查詢風險評估資訊

```python
# ❌ 已廢棄
# from external_learning.analysis import RiskAssessmentEngine

# ✅ 使用 RAG 動態查詢
from aiva_core import BioNeuronRAGAgent

rag_agent = BioNeuronRAGAgent()
risk_info = await rag_agent.query(
    f"評估 SQL Injection 對 {target} 的攻擊風險，考慮 WAF 和 HTTPS 因素"
)
```

#### `ast_trace_comparator.py` - AST 軌跡比較器
**功能**: 比較不同執行軌跡的 AST 差異
```python
from cognitive_core.learning_system.analysis import ASTTraceComparator

comparator = ASTTraceComparator()

# 比較兩次執行
similarity = comparator.compare(trace1, trace2)
print(f"相似度: {similarity}")
```

---

### 2. 🧠 Learning (學習系統)

#### `model_trainer.py` - 模型訓練器
**功能**: 訓練和優化強化學習模型
```python
from cognitive_core.learning_system.learning import ModelTrainer
from aiva_common.schemas import ExperienceSample

# 初始化訓練器
trainer = ModelTrainer(
    model_dir="./models",
    storage_backend=storage
)

# 準備訓練數據
experiences = [
    ExperienceSample(
        state={"target": "...", "fingerprints": {...}},
        action="sql_injection",
        reward=1.0,
        next_state={"vulnerability_found": True},
        done=True
    ),
    # ... 更多經驗
]

# 訓練模型
result = await trainer.train_from_experiences(
    experiences=experiences,
    model_type="dqn",  # 或 "ppo"
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
)

print(f"訓練損失: {result.metrics['loss']}")
print(f"平均獎勵: {result.metrics['avg_reward']}")

# 評估模型
eval_result = await trainer.evaluate_model(
    model_id=result.model_id,
    test_scenarios=test_scenarios
)

print(f"測試準確率: {eval_result['accuracy']}")
```

**支援的模型類型**:
- `dqn` - Deep Q-Network (深度 Q 網路)
- `ppo` - Proximal Policy Optimization (近端策略優化)
- `supervised` - 監督學習分類器

**訓練流程**:
1. **經驗收集** - 從執行結果收集訓練樣本
2. **數據預處理** - 特徵提取和標準化
3. **模型訓練** - 使用強化學習算法訓練
4. **性能評估** - 在測試場景上評估性能
5. **模型部署** - 將最佳模型部署到生產環境

#### `rl_models.py` - 強化學習模型
**功能**: 定義 DQN、PPO 等強化學習模型架構
```python
from cognitive_core.learning_system.learning import DQNModel, PPOModel

# 創建 DQN 模型
dqn_model = DQNModel(
    state_dim=128,
    action_dim=10,
    hidden_dim=256
)

# 創建 PPO 模型
ppo_model = PPOModel(
    state_dim=128,
    action_dim=10,
    hidden_dim=256
)
```

#### `rl_trainers.py` - 強化學習訓練器
**功能**: 實現 DQN、PPO 訓練算法
```python
from cognitive_core.learning_system.learning import DQNTrainer, PPOTrainer

# DQN 訓練器
dqn_trainer = DQNTrainer(
    model=dqn_model,
    learning_rate=0.001,
    gamma=0.99
)

# PPO 訓練器
ppo_trainer = PPOTrainer(
    model=ppo_model,
    learning_rate=0.0003,
    clip_epsilon=0.2
)
```

---

### 3. 🎯 Training (訓練編排) - ❌ 已廢棄

> ⚠️ **架構變更 (2025-12-17)**:  
> `TrainingOrchestrator` 和 `ScenarioManager` 已移除。  
> 訓練功能已整合到 `UnifiedAttackExecutor`（位於 `task_planning` 模組）。

**為什麼移除**:
- TrainingOrchestrator 包含 40+ 錯誤
- 與 AI Commander 存在雙重執行路徑（代碼重複 1500 行）
- 靶場訓練與實戰執行分離，導致數據利用率僅 5%

**新架構優勢**:
- ✅ 統一執行路徑（-47% 代碼）
- ✅ 靶場 = 實戰（數據利用率 10x）
- ✅ 自動學習（100% 覆蓋）
- ✅ 可配置學習閾值

**遷移指南**: 請使用 `task_planning.unified_executor.UnifiedAttackExecutor`  
**詳細說明**: [架構簡化報告](../_ARCHITECTURE_SIMPLIFICATION_REPORT_2025-12-17.md)

---

### 4. 🎧 Event Listener (事件監聽)

#### `event_listener.py` - 學習事件監聽器
**功能**: 監聽 TASK_COMPLETED 事件並觸發學習流程
```python
from cognitive_core.learning_system import LearningEventListener
from aiva_common.enums import Topic

# 初始化事件監聽器
listener = LearningEventListener()

# 啟動監聽
await listener.start_listening()

# 監聽器會自動處理以下流程：
# 1. 監聽 TASK_COMPLETED 事件
# 2. 提取執行數據和結果
# 3. 觸發 ExternalLoopConnector 處理
# 4. 啟動 AST vs Trace 偏差分析
# 5. 判斷是否需要模型重訓練
```

**事件處理流程**:
```
任務完成事件 (TASK_COMPLETED)
    ↓
ExternalLearningListener.handle_task_completed()
    ↓
提取執行軌跡和 AST 計劃
    ↓
ExternalLoopConnector.process_execution_result()
    ↓
ASTTraceComparator.compare() - 偏差分析
    ↓
如果偏差 > 閾值
    ↓
ModelTrainer.retrain() - 重新訓練模型
```

**特性**:
- ✅ **自動監聽** - 自動訂閱 TASK_COMPLETED 主題
- ✅ **異常處理** - 完整的錯誤處理和重試機制
- ✅ **日誌記錄** - 詳細的事件處理日誌
- ✅ **閉環觸發** - 自動觸發外部學習閉環
- ✅ **效能監控** - 事件處理性能統計

---

### 5. 📝 Tracing (執行追蹤)

#### `execution_tracer.py` - 執行追蹤器
**功能**: 追蹤攻擊執行的完整軌跡
```python
from cognitive_core.learning_system.tracing import ExecutionTracer

tracer = ExecutionTracer()

# 開始追蹤
trace_id = tracer.start_trace(task_id="task_001")

# 記錄步驟
tracer.record_step(
    trace_id=trace_id,
    step="send_payload",
    data={"payload": "' OR '1'='1", "response_code": 200}
)

# 結束追蹤
tracer.end_trace(trace_id, success=True)

# 獲取軌跡
trace = tracer.get_trace(trace_id)
```

#### `trace_recorder.py` - 軌跡記錄器
**功能**: 持久化執行軌跡到存儲
```python
from cognitive_core.learning_system.tracing import TraceRecorder

recorder = TraceRecorder(storage_backend=storage)

# 保存軌跡
await recorder.save_trace(trace)

# 查詢軌跡
traces = await recorder.query_traces(
    filters={"success": True, "vulnerability_type": "sql_injection"},
    limit=100
)
```

---

## 📖 使用範例

### 完整的學習流程（推薦使用 UnifiedExecutor）
```python
# ⭐ 推薦：使用 UnifiedExecutor（自動學習）
from task_planning.unified_executor import UnifiedAttackExecutor
from cognitive_core.learning_system import ExperienceManager
from cognitive_core.learning_system.learning import ModelTrainer

# 1. 初始化組件（學習功能已內建）
experience_mgr = ExperienceManager()
model_trainer = ModelTrainer(model_dir="./models")

executor = UnifiedAttackExecutor(
    plan_executor=plan_executor,
    experience_manager=experience_mgr,
    model_trainer=model_trainer,
    rag_engine=rag_engine,
    auto_learn=True,  # 啟用自動學習
    learn_threshold=100  # 累積 100 樣本後自動訓練
)

# 2. 執行攻擊（自動收集經驗並訓練）
result = await executor.execute_with_learning(
    plan=attack_plan,
    context=task_context
)

# 3. 查看學習統計
stats = executor.get_learning_stats()
print(f"已收集樣本: {stats['samples_collected']}")
print(f"訓練次數: {stats['training_runs']}")
print(f"最新模型性能: {stats['latest_model_metrics']}")

# 4. 純執行模式（不學習）
result = await executor.execute_without_learning(
    plan=attack_plan,
    context=task_context
)
```

### 傳統方式（手動管理學習）
```python
from cognitive_core.learning_system import (
    StrategyAdjuster,
    ExperienceManager,
    ExecutionTracer
)
from cognitive_core.learning_system.learning import ModelTrainer

# 1. 初始化組件
adjuster = StrategyAdjuster()
trainer = ModelTrainer(model_dir="./models")
experience_mgr = ExperienceManager()
tracer = ExecutionTracer()

# 2. 執行測試並追蹤
trace_id = tracer.start_trace(task_id="task_001")

# 執行攻擊...
result = await execute_attack(target, payload)

tracer.record_step(trace_id, "attack", {"payload": payload, "result": result})
tracer.end_trace(trace_id, success=result["success"])

# 3. 手動收集經驗
experience_mgr.record_experience({
    "objective": "SQL 注入測試",
    "actions": [payload],
    "result": result,
    "success": result["success"]
})

# 4. 手動觸發訓練（累積到閾值）
if len(experience_mgr.buffer) >= 100:
    samples = experience_mgr.sample_batch(batch_size=64)
    await trainer.train(
        samples=samples,
        config={"learning_rate": 0.001},
        mode="supervised"
    )
```

### 事件驅動學習流程
```python
from cognitive_core.learning_system import LearningEventListener
from task_planning.unified_executor import UnifiedAttackExecutor

# 1. 初始化 UnifiedExecutor（內建自動學習）
executor = UnifiedAttackExecutor(
    auto_learn=True,
    learn_threshold=100
)

# 2. 啟動事件監聽器（可選，用於高級分析）
listener = ExternalLearningListener()
await listener.start_listening()
print("事件監聽器已啟動，監聽任務完成事件")

# 事件監聽器會自動處理：
# - 監聽 TASK_COMPLETED 事件
# - 觸發 AST vs Trace 偏差分析
# - 提供額外的診斷信息

# 3. 執行攻擊（自動學習）
result = await executor.execute_with_learning(
    plan=attack_plan,
    context=task_context
)

print(f"執行完成:")
print(f"  成功: {result.success}")
print(f"  已收集樣本: {executor.get_learning_stats()['samples_collected']}")

# 4. 停止監聽器
await listener.stop_listening()
print("事件監聽器已停止")
```

---

## 🛠️ 開發指南

### 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範。

**完整規範**: [aiva_common 開發指南](../../../aiva_common/README.md#-開發指南)

#### 學習模組特別注意

```python
# ✅ 正確：使用標準定義
from aiva_common import (
    FindingPayload, Severity, Confidence,
    CVSSv3Metrics, VulnerabilityType
)

# ❌ 禁止：自創訓練結果格式
class TrainingResult(BaseModel): pass  # 應該擴展標準 Schema

# ✅ 合理的學習專屬枚舉
class TrainingPhase(str, Enum):
    """訓練階段 (training 專用)"""
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
```

**External Learning 原則**:
- 漏洞數據使用 `FindingPayload`
- 評分使用 `CVSSv3Metrics`
- 訓練結果可擴展但不重複定義基礎類型

📖 **完整文檔**: [修復規範詳解](../../../aiva_common/README.md#-開發規範與最佳實踐)

---

### 添加新的調整策略

```python
# external_learning/analysis/dynamic_strategy_adjustment.py
class StrategyAdjuster:
    def _adjust_for_custom_condition(self, plan, context):
        """自定義調整邏輯"""
        if context.get("custom_condition"):
            # 修改計畫
            plan["custom_tasks"] = [...]
        return plan
    
    def adjust(self, plan, context):
        # ... 現有邏輯
        plan = self._adjust_for_custom_condition(plan, context)
        return plan
```

### 實現自定義訓練器

```python
# external_learning/learning/custom_trainer.py
from .rl_trainers import BaseTrainer

class CustomTrainer(BaseTrainer):
    async def train(self, experiences):
        """實現自定義訓練邏輯"""
        # 預處理數據
        processed_data = self._preprocess(experiences)
        
        # 訓練模型
        model = self._train_model(processed_data)
        
        # 評估性能
        metrics = self._evaluate(model)
        
        return {"model": model, "metrics": metrics}
```

### 添加新的測試場景

> ⚠️ **已廢棄**: `ScenarioManager` 已移除，訓練場景現由 `UnifiedAttackExecutor` 統一管理。
> 詳見 [task_planning/unified_executor.py](../../task_planning/unified_executor.py)

```python
# ✅ 使用 UnifiedExecutor 的自動學習機制
from task_planning.unified_executor import UnifiedAttackExecutor

executor = UnifiedAttackExecutor(
    auto_learn=True,  # 啟用自動學習
    learn_threshold=100  # 訓練閾值
)

# 執行會自動收集經驗並觸發訓練
        # ... 更多 Payload
    ]
)
```

---

## 📊 性能指標

### 策略調整
- **調整速度**: < 100ms
- **學習樣本容量**: 10,000+ 樣本
- **策略優化率**: 30%+ 性能提升
- **WAF 繞過率**: 70%+

### 模型訓練
- **訓練速度**: 1000 樣本/秒
- **模型收斂**: 100-500 次迭代
- **準確率**: 85%+ (測試集)
- **模型大小**: < 100MB

### 執行追蹤
- **追蹤開銷**: < 5% CPU
- **存儲效率**: 壓縮率 60%+
- **查詢速度**: < 100ms
- **並發追蹤**: 1000+ 並發

---

## 🔗 相關模組

- **cognitive_core** - 提供 RAG Engine 和神經網路
- **task_planning** - 接收調整後的策略並執行
- **core_capabilities** - 提供執行結果用於學習
- **service_backbone** - 提供存儲和狀態管理

---

**最後更新**: 2026-04-05  
**維護者**: AIVA Development Team  
**授權**: MIT License
