# Learning 機器學習模組

> **路徑**: `cognitive_core/learning_system/learning`  
> **狀態**: ✅ 正常 | **文件數**: 7 | **最後更新**: 2026-01-07

## 概述

提供模型訓練、強化學習和持續學習功能。包含 DQN/PPO 深度強化學習算法、線上學習器和 ScalableBioNet 專用訓練器。

## 核心組件

### model_trainer.py

- `ModelTrainer` - 模型訓練器
  - 監督學習訓練
  - 強化學習訓練
  - 模型評估和保存
  - 支援 DQN 和 PPO 訓練器

### online_learner.py

- `OnlineLearner` - 線上學習器
  - 即時權重更新（單樣本更新）
  - 小學習率防止災難性遺忘 (lr=0.0001)
  - 梯度裁剪避免梯度爆炸
  - 適用於靶場環境的探索式學習

### scalable_bio_trainer.py

- `ScalableBioTrainingConfig` - 訓練配置數據類
- `ScalableBioTrainer` - ScalableBioNet 專用訓練器
  - Early stopping 支援
  - 批次訓練
  - 訓練歷史記錄

### rl_trainers.py

- `DQNTrainer` - DQN (Deep Q-Network) 訓練器
  - ε-greedy 策略
  - 經驗回放 (ReplayBuffer)
  - Double DQN 實現
  - 目標網絡軟更新
- `PPOTrainer` - PPO (Proximal Policy Optimization) 訓練器
  - Actor-Critic 架構
  - GAE 優勢估計
  - Clipped 目標函數

### rl_models.py

- `DQNNetwork(nn.Module)` - DQN 神經網絡模型
  - 可配置隱藏層和激活函數
  - ε-greedy 動作選擇
- `ActorCritic(nn.Module)` - Actor-Critic 網絡（用於 PPO）
- `ReplayBuffer` - 經驗回放緩衝區（DQN 用）
- `RolloutBuffer` - 軌跡緩衝區（PPO 用）

### continuous_learning.py

- `ContinuousLearningEngine` - 持續學習引擎
  - 整合線上學習和批次學習
  - 靶場環境：即時線上學習
  - 生產環境：選擇性批次學習
  - 智能觸發批次訓練

### __init__.py

- 導出：`ModelTrainer`, `ScalableBioTrainer`, `ScalableBioTrainingConfig`

## 依賴關係

- 內部依賴：
  - `experience_manager.ExperienceManager`
  - `aiva_common.schemas` (ExperienceSample, ModelTrainingConfig)
  - `aiva_common.error_handling`
- 外部依賴：`torch`, `numpy`

## 使用範例

```python
from cognitive_core.learning_system.learning import (
    ModelTrainer, OnlineLearner, ContinuousLearningEngine
)
from cognitive_core.learning_system.learning.rl_trainers import DQNTrainer

# 模型訓練
trainer = ModelTrainer(model_dir="./models")
result = await trainer.train(samples, config, mode="supervised")

# DQN 訓練
dqn = DQNTrainer(state_dim=100, action_dim=10)
loss = dqn.train_step(state, action, reward, next_state, done)

# 線上學習
online_learner = OnlineLearner(model, learning_rate=0.0001)
result = online_learner.update_from_experience(state_tensor, target_tensor)

# 持續學習
engine = ContinuousLearningEngine(
    online_learner=online_learner,
    batch_train_threshold=100
)
result = await engine.process_sandbox_experience(state, action, reward, next_state)
```
