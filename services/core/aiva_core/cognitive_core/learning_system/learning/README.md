# Learning 機器學習模組

> **路徑**: `cognitive_core/learning_system/learning`  
> **狀態**: ✅ 正常 | **Python 文件數**: 7 | **最後更新**: 2026-04-05

## 概述

提供模型訓練、強化學習和持續學習功能。包含 DQN/PPO 深度強化學習算法、線上學習器和 ScalableBioNet 專用訓練器。

## 📄 檔案詳細資訊 (Files Details)

### `continuous_learning.py`
**說明**: Continuous Learning Engine - 持續學習引擎

**類別 (Classes)**:
- `ContinuousLearningEngine` - 持續學習引擎
**函式 (Functions)**:
- `create_continuous_learning_engine()` - 創建持續學習引擎的便捷函數

### `model_trainer.py`
**說明**: Model Trainer - 強化學習模型訓練器

**類別 (Classes)**:
- `ModelTrainer` - 模型訓練器

### `online_learner.py`
**說明**: Online Learner - 線上學習器

**類別 (Classes)**:
- `OnlineLearner` - 線上學習器
**函式 (Functions)**:
- `create_online_learner()` - 創建線上學習器的便捷函數

### `rl_models.py`
**說明**: 強化學習神經網絡模型

**類別 (Classes)**:
- `DQNNetwork` - Deep Q-Network 模型
- `ActorCritic` - Actor-Critic 網絡 (用於 PPO)
- `ReplayBuffer` - Experience Replay Buffer
- `RolloutBuffer` - Rollout Buffer (用於 PPO)

### `rl_trainers.py`
**說明**: 強化學習訓練器

**類別 (Classes)**:
- `DQNTrainer` - DQN (Deep Q-Network) 訓練器
- `PPOTrainer` - PPO (Proximal Policy Optimization) 訓練器

### `scalable_bio_trainer.py`
**說明**: ScalableBio Trainer - ScalableBioNet 專用訓練器

**類別 (Classes)**:
- `ScalableBioTrainingConfig` - ScalableBioNet 訓練配置
- `ScalableBioTrainer` - ScalableBioNet 專用訓練器

