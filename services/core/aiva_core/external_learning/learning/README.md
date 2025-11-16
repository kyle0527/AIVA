# Learning - 學習引擎

**導航**: [← 返回 External Learning](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [model_trainer.py](#model_trainerpy-892-行-)
  - [learning_engine.py](#learning_enginepy-645-行-)
  - [reinforcement_learning.py](#reinforcement_learningpy-312-行)
  - [transfer_learning.py](#transfer_learningpy-172-行)
- [🔄 學習流程](#-學習流程)
- [📊 訓練監控](#-訓練監控)
- [📚 相關模組](#-相關模組)

---

## 📋 概述

**定位**: 機器學習核心引擎  
**狀態**: ✅ 已實現  
**文件數**: 4 個 Python 文件 (2,021 行)

## 📂 文件結構

```
learning/
├── model_trainer.py (892 行) ⭐⭐ - 模型訓練器
├── learning_engine.py (645 行) ⭐ - 學習引擎
├── reinforcement_learning.py (312 行) - 強化學習
├── transfer_learning.py (172 行) - 遷移學習
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### model_trainer.py (892 行) ⭐⭐

**職責**: 統一模型訓練接口

**支持的訓練模式**:
- 監督學習 (Supervised Learning)
- 無監督學習 (Unsupervised Learning)
- 半監督學習 (Semi-supervised Learning)
- 在線學習 (Online Learning)

**使用範例**:
```python
from aiva_core.external_learning.learning import ModelTrainer

trainer = ModelTrainer(
    model_type="random_forest",
    task="classification"
)

# 訓練模型
trainer.train(
    X_train=training_data,
    y_train=labels,
    validation_split=0.2,
    epochs=100,
    early_stopping=True
)

# 評估模型
metrics = trainer.evaluate(X_test, y_test)
# {"accuracy": 0.95, "f1_score": 0.93, "precision": 0.94, "recall": 0.92}

# 保存模型
trainer.save("trained_model.pkl")
```

**訓練回調**:
```python
# 自定義訓練回調
class CustomCallback:
    def on_epoch_end(self, epoch, metrics):
        print(f"Epoch {epoch}: {metrics}")

trainer.train(..., callbacks=[CustomCallback()])
```

---

### learning_engine.py (645 行) ⭐

**職責**: 學習流程編排和管理

**功能**:
- 自動超參數調優
- 模型選擇
- 交叉驗證
- 特徵工程

**使用範例**:
```python
from aiva_core.external_learning.learning import LearningEngine

engine = LearningEngine()

# 自動訓練 (自動選擇最佳模型)
best_model = engine.auto_train(
    data=training_data,
    target="label",
    task="classification",
    metric="f1_score"
)

# 超參數調優
engine.tune_hyperparameters(
    model="random_forest",
    param_grid={
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30]
    },
    cv=5  # 5-fold 交叉驗證
)
```

---

### reinforcement_learning.py (312 行)

**職責**: 強化學習算法實現

**支持算法**:
- Q-Learning
- Deep Q-Network (DQN)
- Policy Gradient

**使用範例**:
```python
from aiva_core.external_learning.learning import RLAgent

# 創建 RL 智能體
agent = RLAgent(
    algorithm="dqn",
    state_dim=10,
    action_dim=4
)

# 訓練
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

---

### transfer_learning.py (172 行)

**職責**: 遷移學習和模型微調

**使用範例**:
```python
from aiva_core.external_learning.learning import TransferLearner

# 加載預訓練模型
learner = TransferLearner.from_pretrained("bert-base")

# 微調
learner.fine_tune(
    train_data=new_data,
    epochs=5,
    freeze_layers=True  # 凍結底層
)
```

## 🔄 學習流程

```
數據收集
  ↓
數據預處理 (analysis/data_preprocessor.py)
  ↓
特徵提取 (analysis/feature_analyzer.py)
  ↓
模型訓練 (model_trainer.py)
  ↓
模型評估
  ↓
超參數調優 (learning_engine.py)
  ↓
模型部署
```

## 📊 訓練監控

```python
from aiva_core.external_learning.learning import TrainingMonitor

monitor = TrainingMonitor()

# 實時監控訓練
trainer.train(..., monitor=monitor)

# 查看訓練曲線
monitor.plot_metrics()

# 獲取訓練歷史
history = monitor.get_history()
# {
#   "epoch": [1, 2, 3, ...],
#   "loss": [0.5, 0.3, 0.2, ...],
#   "accuracy": [0.8, 0.9, 0.95, ...]
# }
```

## 📚 相關模組

- [training](../training/README.md) - 訓練編排
- [tracing](../tracing/README.md) - 訓練追蹤
- [analysis](../analysis/README.md) - 數據分析

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準類型
from aiva_common import TaskStatus, ModuleName

# ❌ 禁止：自定義學習狀態
class TrainingStatus(str, Enum): pass
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: External Learning 團隊
