# 📚 External Learning - 對外學習模組

## 📑 目錄

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
  - [自動化訓練編排](#自動化訓練編排)
- [🛠️ 開發指南](#-開發指南)
  - [🔨 aiva_common 修復規範](#-aiva_common-修復規範)
  - [添加新的調整策略](#添加新的調整策略)
  - [實現自定義訓練器](#實現自定義訓練器)
  - [添加新的測試場景](#添加新的測試場景)
- [📊 性能指標](#-性能指標)
  - [策略調整](#策略調整)
  - [模型訓練](#模型訓練)
  - [執行追蹤](#執行追蹤)
- [🔗 相關模組](#-相關模組)

---

**導航**: [← 返回 AIVA Core](../README.md)

> **版本**: 3.0.0-alpha  
> **狀態**: 生產就緒，測試通過  
> **🧪 測試狀態**: 階段 8 測試 100% 通過 (4/4 組件，包含 ExperienceManager 警告)  
> **角色**: AIVA 的「學習大腦」- 從執行結果中學習並持續優化  
> **最後更新**: 2025-12-10

---

## 🎯 模組概述

---

## 🎯 模組概述

**External Learning** 是 AIVA 六大模組架構中的持續學習層，負責從攻擊執行結果中學習經驗、優化策略、訓練模型，實現系統能力的持續提升。整合了動態策略調整、模型訓練、場景管理、執行追蹤等核心能力。

### 核心職責
1. **策略調整** - 基於執行結果動態調整測試策略
2. **模型訓練** - 訓練和優化強化學習模型
3. **場景管理** - 管理 OWASP 和自定義測試場景
4. **執行追蹤** - 追蹤和記錄攻擊執行軌跡
5. **風險評估** - 評估攻擊風險和成功率
6. **訓練編排** - 自動化訓練流程編排

### 設計理念
- **持續學習** - 從每次執行中學習並優化
- **自適應調整** - 根據環境變化動態調整策略
- **知識積累** - 將經驗轉化為知識並復用
- **性能提升** - 通過訓練不斷提升檢測能力

---

## 🏗️ 架構設計

```
external_learning/
├── 📁 analysis/                  # 分析引擎 (3 檔案)
│   ├── dynamic_strategy_adjustment.py  # ✅ 動態策略調整器
│   ├── ast_trace_comparator.py         # AST 軌跡比較器
│   └── risk_assessment_engine.py       # 風險評估引擎
│
├── 📁 learning/                  # 學習系統 (5 檔案)
│   ├── model_trainer.py          # ✅ 模型訓練器
│   ├── rl_models.py              # 強化學習模型
│   ├── rl_trainers.py            # 強化學習訓練器
│   └── scalable_bio_trainer.py   # 可擴展生物神經訓練器
│
├── 📁 training/                  # 訓練編排 (3 檔案)
│   ├── training_orchestrator.py  # ✅ 訓練編排器
│   ├── scenario_manager.py       # 場景管理器
│   └── __init__.py
│
├── 📁 tracing/                   # 執行追蹤 (3 檔案)
│   ├── execution_tracer.py       # 執行追蹤器
│   ├── trace_recorder.py         # 軌跡記錄器
│   └── unified_tracer.py         # 統一追蹤器
│
├── 📁 ai_model/                  # AI 模型 (1 檔案)
│   └── train_classifier.py       # 分類器訓練
│
├── event_listener.py             # ✅ 外部學習事件監聽器
└── README.md                     # 本文檔

總計: 17 個 Python 檔案
```

### 學習流程
```
┌─────────────────────────────────────────────────────────┐
│         External Learning (對外學習)                     │
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

### 1. 📊 Analysis (分析引擎)

#### `dynamic_strategy_adjustment.py` - 動態策略調整器
**功能**: 基於執行結果和上下文動態調整測試策略
```python
from external_learning.analysis import StrategyAdjuster

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

#### `risk_assessment_engine.py` - 風險評估引擎
**功能**: 評估攻擊風險和成功率
```python
from external_learning.analysis import RiskAssessmentEngine

engine = RiskAssessmentEngine()

# 評估風險
risk_score = engine.assess_risk(
    target="https://example.com",
    attack_type="sql_injection",
    context={"waf_detected": True, "https": True}
)

print(f"風險評分: {risk_score}")  # 0.0-1.0
```

#### `ast_trace_comparator.py` - AST 軌跡比較器
**功能**: 比較不同執行軌跡的 AST 差異
```python
from external_learning.analysis import ASTTraceComparator

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
from external_learning.learning import ModelTrainer
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
from external_learning.learning import DQNModel, PPOModel

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
from external_learning.learning import DQNTrainer, PPOTrainer

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

### 3. 🎯 Training (訓練編排)

#### `training_orchestrator.py` - 訓練編排器
**功能**: 自動化訓練流程的完整編排
```python
from external_learning.training import TrainingOrchestrator

# 初始化訓練編排器
orchestrator = TrainingOrchestrator(
    scenario_manager=scenario_manager,
    rag_engine=rag_engine,
    plan_executor=plan_executor,
    model_trainer=model_trainer,
    data_directory="./data"
)

# 執行完整訓練流程
result = await orchestrator.run_training_cycle(
    scenario_type="owasp_top10",
    num_iterations=100,
    model_type="dqn"
)

print(f"訓練完成: {result['model_id']}")
print(f"最終性能: {result['final_performance']}")

# 持續訓練
await orchestrator.continuous_training(
    check_interval_hours=24,
    min_new_experiences=1000
)
```

**訓練週期**:
1. **場景加載** - 從 ScenarioManager 加載測試場景
2. **計畫生成** - 使用 RAG 增強計畫生成
3. **計畫執行** - 執行攻擊計畫並收集結果
4. **經驗收集** - 將結果轉化為訓練經驗
5. **模型訓練** - 使用經驗訓練模型
6. **性能評估** - 評估模型性能
7. **迭代優化** - 重複流程直到收斂

#### `scenario_manager.py` - 場景管理器
**功能**: 管理 OWASP 和自定義測試場景
```python
from external_learning.training import ScenarioManager

manager = ScenarioManager(data_dir="./data/scenarios")

# 加載 OWASP 場景
owasp_scenarios = manager.load_owasp_scenarios()
print(f"加載 {len(owasp_scenarios)} 個 OWASP 場景")

# 創建自定義場景
manager.create_scenario(
    name="custom_sqli_test",
    target_url="http://testphp.vulnweb.com",
    vulnerabilities=["sql_injection"],
    difficulty="medium"
)

# 獲取場景
scenario = manager.get_scenario("custom_sqli_test")
```

---

### 4. 🎧 Event Listener (事件監聽)

#### `event_listener.py` - 外部學習事件監聽器
**功能**: 監聽 TASK_COMPLETED 事件並觸發學習流程
```python
from external_learning import ExternalLearningListener
from aiva_common.enums import Topic

# 初始化事件監聽器
listener = ExternalLearningListener()

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
from external_learning.tracing import ExecutionTracer

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
from external_learning.tracing import TraceRecorder

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

### 完整的學習流程
```python
from external_learning import (
    StrategyAdjuster,
    ModelTrainer,
    TrainingOrchestrator,
    ExecutionTracer
)

# 1. 初始化組件
adjuster = StrategyAdjuster()
trainer = ModelTrainer(model_dir="./models")
orchestrator = TrainingOrchestrator()
tracer = ExecutionTracer()

# 2. 執行測試並追蹤
trace_id = tracer.start_trace(task_id="task_001")

# 執行攻擊...
result = await execute_attack(target, payload)

tracer.record_step(trace_id, "attack", {"payload": payload, "result": result})
tracer.end_trace(trace_id, success=result["success"])

# 3. 調整策略
adjusted_plan = adjuster.adjust(
    plan=next_plan,
    context={
        "waf_detected": result.get("waf_detected"),
        "findings_count": result.get("vulnerabilities_found")
    }
)

# 4. 學習經驗
adjuster.learn_from_result({
    "scan_id": "scan_001",
    "module": "sqli",
    "success": result["success"],
    "payload": payload
})

# 5. 啟動事件監聽器 (自動化學習)
from external_learning import ExternalLearningListener

listener = ExternalLearningListener()
await listener.start_listening()
print("外部學習事件監聽器已啟動，將自動處理任務完成事件")

# 6. 定期訓練模型 (手動觸發)
if should_train():
    experiences = collect_experiences()
    training_result = await trainer.train_from_experiences(
        experiences=experiences,
        model_type="dqn"
    )
    print(f"模型訓練完成: {training_result.model_id}")
```

### 事件驅動學習流程
```python
from external_learning import ExternalLearningListener
from external_learning.training import TrainingOrchestrator

# 1. 啟動事件監聽器
listener = ExternalLearningListener()
await listener.start_listening()
print("事件監聽器已啟動，監聽任務完成事件")

# 2. 事件監聽器會自動處理：
# - 監聽 TASK_COMPLETED 事件
# - 觸發 AST vs Trace 偏差分析
# - 自動決定是否需要重訓練

# 3. 手動訓練編排 (可選)
orchestrator = TrainingOrchestrator()

# 運行訓練週期
result = await orchestrator.run_training_cycle(
    scenario_type="owasp_top10",
    num_iterations=100,
    model_type="dqn"
)

print(f"訓練完成:")
print(f"  模型 ID: {result['model_id']}")
print(f"  最終準確率: {result['final_accuracy']}")
print(f"  訓練時間: {result['training_time']} 秒")

# 4. 停止監聽器
await listener.stop_listening()
print("事件監聽器已停止")
```

### 自動化訓練編排
```python
from external_learning.training import TrainingOrchestrator

orchestrator = TrainingOrchestrator()

# 運行訓練週期
result = await orchestrator.run_training_cycle(
    scenario_type="owasp_top10",
    num_iterations=100,
    model_type="dqn"
)

print(f"訓練完成:")
print(f"  模型 ID: {result['model_id']}")
print(f"  最終準確率: {result['final_accuracy']}")
print(f"  訓練時間: {result['training_time']} 秒")
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

```python
from external_learning.training import ScenarioManager

manager = ScenarioManager()

# 創建新場景
manager.create_scenario(
    name="advanced_xss_test",
    target_url="http://example.com",
    vulnerabilities=["xss"],
    difficulty="hard",
    description="高級 XSS 測試場景",
    payloads=[
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
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

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team  
**授權**: MIT License
