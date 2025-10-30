"""
Model Trainer - 強化學習模型訓練器

負責模型的訓練、評估、微調和部署，支持監督學習和強化學習
"""



from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from services.aiva_common.schemas import (
    ExperienceSample,
    ModelTrainingConfig,
    ModelTrainingResult,
    ScenarioTestResult,
    StandardScenario,
)


logger = logging.getLogger(__name__)

# 動態導入深度學習模組（可選）
try:
    import torch
    from .rl_trainers import DQNTrainer, PPOTrainer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. DQN/PPO training will be disabled. "
        "Install with: pip install torch"
    )


class ModelTrainer:
    """模型訓練器

    實現監督學習和強化學習訓練流程
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        storage_backend: Any | None = None,
    ) -> None:
        """初始化訓練器

        Args:
            model_dir: 模型存儲目錄
            storage_backend: 儲存後端
        """
        self.model_dir = model_dir or Path("./models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage_backend
        self.current_model: Any | None = None
        self.model_version = "v0.0.0"
        self.training_history: list[dict[str, Any]] = []
        
        # 深度強化學習訓練器（延遲初始化）
        self.dqn_trainer: DQNTrainer | None = None
        self.ppo_trainer: PPOTrainer | None = None

        logger.info(f"ModelTrainer initialized with model_dir={self.model_dir}")

    async def train_supervised(
        self,
        samples: list[ExperienceSample],
        config: ModelTrainingConfig,
        validation_split: float = 0.2,
    ) -> ModelTrainingResult:
        """監督學習訓練

        Args:
            samples: 訓練樣本
            config: 訓練配置
            validation_split: 驗證集比例

        Returns:
            訓練結果
        """
        logger.info(
            f"Starting supervised training with {len(samples)} samples "
            f"(validation_split={validation_split})"
        )

        training_id = f"training_{uuid4().hex[:12]}"
        started_at = datetime.now(UTC)

        # 1. 數據預處理
        X_train, y_train, X_val, y_val = self._prepare_supervised_data(
            samples, validation_split
        )

        logger.info(f"Data prepared: train={len(X_train)}, validation={len(X_val)}")

        # 2. 訓練模型
        training_metrics = await self._train_model_supervised(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
        )

        # 3. 評估模型
        eval_metrics = await self._evaluate_model(X_val, y_val)

        # 4. 保存模型
        new_version = self._increment_version(self.model_version)
        model_path = self.model_dir / f"model_{new_version}.pkl"
        await self._save_model(model_path)

        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()

        result = ModelTrainingResult(
            training_id=training_id,
            config=config,
            model_version=new_version,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            training_loss=training_metrics["loss"],
            validation_loss=eval_metrics["loss"],
            accuracy=eval_metrics.get("accuracy"),
            precision=eval_metrics.get("precision"),
            recall=eval_metrics.get("recall"),
            f1_score=eval_metrics.get("f1_score"),
            training_duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            metrics=eval_metrics,
            model_path=str(model_path),
        )

        # 持久化訓練結果
        if self.storage:
            await self._persist_training_result(result)

        logger.info(
            f"Training completed: version={new_version}, "
            f"accuracy={eval_metrics.get('accuracy', 0):.2%}, "
            f"duration={duration:.1f}s"
        )

        self.model_version = new_version
        return result

    async def train_reinforcement(
        self,
        samples: list[ExperienceSample],
        config: ModelTrainingConfig,
    ) -> ModelTrainingResult:
        """強化學習訓練

        Args:
            samples: 訓練樣本
            config: 訓練配置

        Returns:
            訓練結果
        """
        logger.info(
            f"Starting reinforcement learning training with {len(samples)} samples"
        )

        training_id = f"training_rl_{uuid4().hex[:12]}"
        started_at = datetime.now(UTC)

        # 1. 準備強化學習數據
        episodes = self._prepare_rl_data(samples)

        logger.info(f"Prepared {len(episodes)} episodes for RL training")

        # 2. 訓練強化學習模型
        rl_metrics = await self._train_model_rl(episodes, config)

        # 3. 保存模型
        new_version = self._increment_version(self.model_version)
        model_path = self.model_dir / f"model_rl_{new_version}.pkl"
        await self._save_model(model_path)

        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()

        result = ModelTrainingResult(
            training_id=training_id,
            config=config,
            model_version=new_version,
            training_samples=len(samples),
            validation_samples=0,
            training_loss=rl_metrics.get("loss", 0.0),
            validation_loss=0.0,
            average_reward=rl_metrics.get("average_reward"),
            training_duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            metrics=rl_metrics,
            model_path=str(model_path),
        )

        if self.storage:
            await self._persist_training_result(result)

        logger.info(
            f"RL training completed: version={new_version}, "
            f"avg_reward={rl_metrics.get('average_reward', 0):.3f}"
        )

        self.model_version = new_version
        return result

    async def train_dqn(
        self,
        samples: list[ExperienceSample],
        config: ModelTrainingConfig,
        state_dim: int = 12,
        action_dim: int = 5,
    ) -> ModelTrainingResult:
        """DQN 深度強化學習訓練
        
        Args:
            samples: 訓練樣本
            config: 訓練配置
            state_dim: 狀態維度
            action_dim: 動作維度
        
        Returns:
            訓練結果
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Please install with: pip install torch"
            )
        
        logger.info(
            f"Starting DQN training with {len(samples)} samples "
            f"(state_dim={state_dim}, action_dim={action_dim})"
        )
        
        training_id = f"training_dqn_{uuid4().hex[:12]}"
        started_at = datetime.now(UTC)
        
        # 1. 初始化 DQN 訓練器
        if self.dqn_trainer is None:
            self.dqn_trainer = DQNTrainer(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate if hasattr(config, "learning_rate") else 0.001,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                buffer_capacity=10000,
                batch_size=config.batch_size if hasattr(config, "batch_size") else 64,
            )
        
        # 2. 準備訓練數據
        episodes = self._prepare_rl_data(samples)
        
        logger.info(f"Prepared {len(episodes)} episodes for DQN training")
        
        # 3. 訓練 DQN
        total_rewards = []
        losses = []
        
        for episode_idx, episode in enumerate(episodes):
            states = episode["states"]
            actions = episode["actions"]
            rewards = episode["rewards"]
            
            episode_reward = 0.0
            
            for t in range(len(states) - 1):
                state = states[t]
                action = int(actions[t])
                reward = float(rewards[t])
                next_state = states[t + 1]
                done = (t == len(states) - 2)
                
                # 訓練一步
                loss = self.dqn_trainer.train_step(
                    state, action, reward, next_state, done
                )
                
                if loss is not None:
                    losses.append(loss)
                
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            if (episode_idx + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                avg_loss = np.mean(losses[-100:]) if losses else 0.0
                logger.debug(
                    f"DQN Episode {episode_idx + 1}/{len(episodes)}: "
                    f"avg_reward={avg_reward:.2f}, avg_loss={avg_loss:.4f}"
                )
        
        # 4. 保存模型
        new_version = self._increment_version(self.model_version)
        model_path = self.model_dir / f"model_dqn_{new_version}.pt"
        self.dqn_trainer.save(str(model_path))
        
        # 5. 收集指標
        dqn_metrics = self.dqn_trainer.get_metrics()
        
        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()
        
        result = ModelTrainingResult(
            training_id=training_id,
            config=config,
            model_version=new_version,
            training_samples=len(samples),
            validation_samples=0,
            training_loss=dqn_metrics.get("avg_loss", 0.0),
            validation_loss=0.0,
            average_reward=float(np.mean(total_rewards)) if total_rewards else 0.0,
            training_duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            metrics={
                **dqn_metrics,
                "total_episodes": len(episodes),
                "avg_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
                "max_episode_reward": float(np.max(total_rewards)) if total_rewards else 0.0,
                "min_episode_reward": float(np.min(total_rewards)) if total_rewards else 0.0,
                "algorithm": "DQN",
            },
            model_path=str(model_path),
        )
        
        if self.storage:
            await self._persist_training_result(result)
        
        logger.info(
            f"DQN training completed: version={new_version}, "
            f"avg_reward={result.average_reward:.3f}, "
            f"episodes={len(episodes)}, duration={duration:.1f}s"
        )
        
        self.model_version = new_version
        self.current_model = self.dqn_trainer
        return result
    
    async def train_ppo(
        self,
        samples: list[ExperienceSample],
        config: ModelTrainingConfig,
        state_dim: int = 12,
        action_dim: int = 5,
        rollout_steps: int = 2048,
    ) -> ModelTrainingResult:
        """PPO 深度強化學習訓練
        
        Args:
            samples: 訓練樣本
            config: 訓練配置
            state_dim: 狀態維度
            action_dim: 動作維度
            rollout_steps: Rollout 步數
        
        Returns:
            訓練結果
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Please install with: pip install torch"
            )
        
        logger.info(
            f"Starting PPO training with {len(samples)} samples "
            f"(state_dim={state_dim}, action_dim={action_dim})"
        )
        
        training_id = f"training_ppo_{uuid4().hex[:12]}"
        started_at = datetime.now(UTC)
        
        # 1. 初始化 PPO 訓練器
        if self.ppo_trainer is None:
            self.ppo_trainer = PPOTrainer(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=config.learning_rate if hasattr(config, "learning_rate") else 3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                ppo_epochs=4,
                mini_batch_size=config.batch_size if hasattr(config, "batch_size") else 64,
            )
        
        # 2. 準備訓練數據
        episodes = self._prepare_rl_data(samples)
        
        logger.info(f"Prepared {len(episodes)} episodes for PPO training")
        
        # 3. 訓練 PPO
        total_rewards = []
        update_metrics_history: list[dict[str, float]] = []
        
        for episode_idx, episode in enumerate(episodes):
            states = episode["states"]
            actions = episode["actions"]
            rewards = episode["rewards"]
            
            episode_reward = 0.0
            
            # 收集 rollout
            for t in range(len(states)):
                state = states[t]
                action, log_prob, value = self.ppo_trainer.select_action(state)
                reward = float(rewards[t])
                done = (t == len(states) - 1)
                
                self.ppo_trainer.store_transition(
                    state, action, reward, log_prob, value, done
                )
                
                episode_reward += reward
                
                # 達到 rollout steps 或 episode 結束時更新
                if len(self.ppo_trainer.rollout_buffer) >= rollout_steps or done:
                    update_metrics = self.ppo_trainer.update()
                    if update_metrics:
                        update_metrics_history.append(update_metrics)
            
            total_rewards.append(episode_reward)
            
            if (episode_idx + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                recent_metrics = update_metrics_history[-10:] if update_metrics_history else []
                avg_policy_loss = (
                    np.mean([m["policy_loss"] for m in recent_metrics])
                    if recent_metrics else 0.0
                )
                logger.debug(
                    f"PPO Episode {episode_idx + 1}/{len(episodes)}: "
                    f"avg_reward={avg_reward:.2f}, "
                    f"avg_policy_loss={avg_policy_loss:.4f}"
                )
        
        # 4. 保存模型
        new_version = self._increment_version(self.model_version)
        model_path = self.model_dir / f"model_ppo_{new_version}.pt"
        self.ppo_trainer.save(str(model_path))
        
        # 5. 收集指標
        ppo_metrics = self.ppo_trainer.get_metrics()
        
        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()
        
        result = ModelTrainingResult(
            training_id=training_id,
            config=config,
            model_version=new_version,
            training_samples=len(samples),
            validation_samples=0,
            training_loss=ppo_metrics.get("avg_policy_loss", 0.0),
            validation_loss=ppo_metrics.get("avg_value_loss", 0.0),
            average_reward=float(np.mean(total_rewards)) if total_rewards else 0.0,
            training_duration_seconds=duration,
            started_at=started_at,
            completed_at=completed_at,
            metrics={
                **ppo_metrics,
                "total_episodes": len(episodes),
                "total_updates": len(update_metrics_history),
                "avg_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
                "max_episode_reward": float(np.max(total_rewards)) if total_rewards else 0.0,
                "min_episode_reward": float(np.min(total_rewards)) if total_rewards else 0.0,
                "algorithm": "PPO",
            },
            model_path=str(model_path),
        )
        
        if self.storage:
            await self._persist_training_result(result)
        
        logger.info(
            f"PPO training completed: version={new_version}, "
            f"avg_reward={result.average_reward:.3f}, "
            f"episodes={len(episodes)}, updates={len(update_metrics_history)}, "
            f"duration={duration:.1f}s"
        )
        
        self.model_version = new_version
        self.current_model = self.ppo_trainer
        return result

    def _prepare_supervised_data(
        self, samples: list[ExperienceSample], validation_split: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """準備監督學習數據

        Args:
            samples: 經驗樣本
            validation_split: 驗證集比例

        Returns:
            (X_train, y_train, X_val, y_val)
        """
        # 提取特徵和標籤
        X = []
        y = []

        for sample in samples:
            # 構建輸入特徵向量
            features = self._extract_features(sample)
            X.append(features)

            # 構建標籤（成功/失敗）
            label = 1 if sample.label == "success" else 0
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # 劃分訓練集和驗證集
        split_idx = int(len(X) * (1 - validation_split))

        # 隨機打亂
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_val, y_val

    def _extract_features(self, sample: ExperienceSample) -> np.ndarray:
        """從樣本中提取特徵向量

        Args:
            sample: 經驗樣本

        Returns:
            特徵向量
        """
        features: list[float] = []

        # 1. 攻擊類型特徵（one-hot encoding）
        attack_types = ["SQLI", "XSS", "SSRF", "IDOR", "other"]
        attack_type = sample.context.get("attack_type", "other")
        attack_one_hot = [1.0 if t == attack_type else 0.0 for t in attack_types]
        features.extend(attack_one_hot)

        # 2. 計畫複雜度特徵
        features.append(float(len(sample.plan.steps)))  # 步驟數量
        features.append(float(len(sample.plan.dependencies)))  # 依賴關係數量

        # 3. 執行指標特徵
        metrics = sample.metrics
        features.extend(
            [
                metrics.completion_rate,
                metrics.success_rate,
                metrics.sequence_accuracy,
                metrics.total_execution_time / 100.0,  # 標準化
            ]
        )

        # 4. 目標特徵（簡化）
        target_info = sample.context.get("target_info", {})
        features.append(1.0 if target_info.get("waf_detected") else 0.0)
        features.append(len(target_info.get("parameters", [])) / 10.0)  # 標準化

        return np.array(features, dtype=np.float32)

    def _prepare_rl_data(self, samples: list[ExperienceSample]) -> list[dict[str, Any]]:
        """準備強化學習數據（episode 格式）

        Args:
            samples: 經驗樣本

        Returns:
            Episodes 列表
        """
        episodes = []

        for sample in samples:
            # 構建 episode
            states = []
            actions = []
            rewards = []

            for i, trace in enumerate(sample.trace):
                # State: 當前狀態特徵
                state = self._build_state_vector(sample, i)
                states.append(state)

                # Action: 工具選擇和參數
                action = self._encode_action(trace)
                actions.append(action)

                # Reward: 基於執行結果
                reward = self._calculate_step_reward(trace, i, sample)
                rewards.append(reward)

            episode = {
                "states": np.array(states),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "total_reward": sum(rewards),
                "sample_id": sample.sample_id,
            }

            episodes.append(episode)

        return episodes

    def _build_state_vector(
        self, sample: ExperienceSample, step_index: int
    ) -> np.ndarray:
        """構建狀態向量

        Args:
            sample: 經驗樣本
            step_index: 當前步驟索引

        Returns:
            狀態向量
        """
        # 簡化版狀態向量
        state = []

        # 上下文特徵
        attack_type = sample.context.get("attack_type", "")
        state.extend(self._encode_attack_type(attack_type))

        # 當前進度
        progress = step_index / max(len(sample.plan.steps), 1)
        state.append(progress)

        # 已完成步驟數
        completed = sum(1 for t in sample.trace[:step_index] if t.status == "success")
        state.append(completed / max(len(sample.plan.steps), 1))

        return np.array(state, dtype=np.float32)

    def _encode_attack_type(self, attack_type: str) -> list[float]:
        """編碼攻擊類型

        Args:
            attack_type: 攻擊類型

        Returns:
            One-hot 向量
        """
        types = ["SQLI", "XSS", "SSRF", "IDOR", "other"]
        return [1.0 if t == attack_type else 0.0 for t in types]

    def _encode_action(self, trace: Any) -> int:
        """編碼動作

        Args:
            trace: 追蹤記錄

        Returns:
            動作編碼
        """
        # 簡化版：根據工具類型編碼
        tool_map = {
            "function_sqli": 0,
            "function_xss": 1,
            "function_ssrf": 2,
            "function_idor": 3,
        }
        return tool_map.get(trace.tool_name, 4)

    def _calculate_step_reward(
        self, trace: Any, step_index: int, sample: ExperienceSample
    ) -> float:
        """計算步驟獎勵

        Args:
            trace: 追蹤記錄
            step_index: 步驟索引
            sample: 樣本

        Returns:
            獎勵值
        """
        reward = 0.0

        # 基礎獎勵：成功執行
        if trace.status == "success":
            reward += 1.0
        elif trace.status == "failed":
            reward -= 0.5

        # 順序獎勵：按預期順序執行
        if step_index < len(sample.plan.steps):
            expected_step = sample.plan.steps[step_index]
            if expected_step.step_id == trace.step_id:
                reward += 0.5

        # 發現獎勵：發現漏洞
        if trace.output_data.get("findings"):
            reward += 2.0

        # 效率懲罰：執行時間過長
        if trace.execution_time_seconds > 30:
            reward -= 0.2

        return reward

    async def _train_model_supervised(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelTrainingConfig,
    ) -> dict[str, Any]:
        """執行監督學習訓練 (實際實現)

        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_val: 驗證特徵
            y_val: 驗證標籤
            config: 訓練配置

        Returns:
            訓練指標
        """
        logger.info("Training supervised model with scikit-learn")

        try:
            # 根據配置選擇模型
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.metrics import accuracy_score
            
            model_type = config.model_type if hasattr(config, 'model_type') else "random_forest"
            
            # 初始化模型
            if model_type == "random_forest":
                self.current_model = RandomForestClassifier(
                    n_estimators=config.epochs * 10,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                self.current_model = GradientBoostingClassifier(
                    n_estimators=config.epochs * 10,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            elif model_type == "neural_network":
                self.current_model = MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    max_iter=config.epochs,
                    learning_rate_init=0.001,
                    random_state=42
                )
            else:
                self.current_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            
            # 訓練模型
            logger.info(f"Training {model_type} model...")
            self.current_model.fit(X_train, y_train)
            
            # 計算訓練集指標
            y_train_pred = self.current_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # 特徵重要性
            feature_importance = None
            if hasattr(self.current_model, 'feature_importances_'):
                feature_importance = self.current_model.feature_importances_.tolist()
            
            logger.info(f"✅ Model trained (accuracy: {train_accuracy:.4f})")
            
            return {
                "model_type": model_type,
                "train_accuracy": float(train_accuracy),
                "n_samples": len(X_train),
                "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1,
                "feature_importance": feature_importance,
                "final_epoch": config.epochs,
            }
        
        except Exception as e:
            logger.error(f"Supervised training failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "final_epoch": 0,
            }

    async def _train_model_rl(
        self, episodes: list[dict[str, Any]], config: ModelTrainingConfig
    ) -> dict[str, Any]:
        """執行強化學習訓練 (實際實現)

        Args:
            episodes: 訓練 episodes
            config: 訓練配置

        Returns:
            訓練指標
        """
        logger.info("Training RL model with Q-learning approach")

        try:
            # 提取獎勵數據
            total_rewards = [ep.get("total_reward", 0.0) for ep in episodes]
            
            if not total_rewards:
                logger.warning("No episodes with rewards found")
                return {
                    "average_reward": 0.0,
                    "episodes": 0,
                    "error": "No valid episodes"
                }
            
            # 簡單的 Q-learning 實現（可擴展為 DQN、PPO 等）
            # 這裡使用基於經驗的價值估計
            
            # 計算統計數據
            average_reward = float(np.mean(total_rewards))
            max_reward = float(np.max(total_rewards))
            min_reward = float(np.min(total_rewards))
            std_reward = float(np.std(total_rewards))
            
            # 計算改進率（比較前後半段）
            mid_point = len(total_rewards) // 2
            if mid_point > 0:
                early_avg = np.mean(total_rewards[:mid_point])
                late_avg = np.mean(total_rewards[mid_point:])
                improvement_rate = ((late_avg - early_avg) / abs(early_avg)) if early_avg != 0 else 0.0
            else:
                improvement_rate = 0.0
            
            # 構建簡單的 Q-table（狀態-動作價值表）
            q_table = {}
            for episode in episodes:
                states = episode.get("states", [])
                actions = episode.get("actions", [])
                rewards = episode.get("rewards", [])
                
                # 更新 Q 值（簡化版）
                for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
                    state_key = str(state)  # 簡化的狀態表示
                    if state_key not in q_table:
                        q_table[state_key] = {}
                    
                    # Q-learning 更新
                    learning_rate = 0.1
                    discount_factor = 0.95
                    
                    current_q = q_table[state_key].get(action, 0.0)
                    
                    # 獲取下一狀態的最大 Q 值
                    next_max_q = 0.0
                    if i + 1 < len(states):
                        next_state_key = str(states[i + 1])
                        if next_state_key in q_table and q_table[next_state_key]:
                            next_max_q = max(q_table[next_state_key].values())
                    
                    # Q-learning 更新公式
                    new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
                    q_table[state_key][action] = new_q
            
            # 保存 Q-table 作為模型
            self.current_model = {
                "type": "q_learning",
                "q_table": q_table,
                "episodes_trained": len(episodes),
                "config": {
                    "learning_rate": 0.1,
                    "discount_factor": 0.95
                }
            }
            
            logger.info(
                f"✅ RL model trained: {len(episodes)} episodes, "
                f"avg_reward={average_reward:.2f}, "
                f"improvement={improvement_rate:.2%}"
            )
            
            return {
                "average_reward": average_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
                "std_reward": std_reward,
                "improvement_rate": float(improvement_rate),
                "episodes": len(episodes),
                "q_table_size": len(q_table),
            }
        
        except Exception as e:
            logger.error(f"RL training failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "episodes": len(episodes),
                "average_reward": 0.0,
            }

    async def _evaluate_model(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, Any]:
        """評估模型 (實際實現)

        Args:
            X_val: 驗證特徵
            y_val: 驗證標籤

        Returns:
            評估指標
        """
        if self.current_model is None:
            logger.error("Cannot evaluate: no trained model available")
            return {
                "error": "No trained model",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }

        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            # 檢查模型類型
            if isinstance(self.current_model, dict):
                # 強化學習模型（Q-learning）
                logger.warning("RL model evaluation not fully implemented, returning placeholder")
                return {
                    "model_type": "rl",
                    "q_table_size": len(self.current_model.get("q_table", {})),
                    "episodes_trained": self.current_model.get("episodes_trained", 0),
                }
            
            # 監督學習模型（scikit-learn）
            y_pred = self.current_model.predict(X_val)

            # 計算評估指標
            accuracy = accuracy_score(y_val, y_pred)
            
            # 處理多類別問題（使用 weighted average）
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            # 計算混淆矩陣
            cm = confusion_matrix(y_val, y_pred)
            
            logger.info(
                f"✅ Model evaluation: accuracy={accuracy:.2%}, "
                f"precision={precision:.2%}, recall={recall:.2%}, f1={f1:.2%}"
            )

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "test_samples": len(y_val),
            }
        
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }

    async def _save_model(self, path: Path) -> None:
        """保存模型 (實際實現)

        Args:
            path: 保存路徑
        """
        if self.current_model is None:
            logger.warning("No model to save")
            return

        try:
            import pickle
            
            # 確保目錄存在
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 構建保存數據
            model_data = {
                "model": self.current_model,
                "version": self.model_version,
                "training_history": self.training_history,
                "metadata": {
                    "saved_at": datetime.now(UTC).isoformat(),
                    "model_type": "q_learning" if isinstance(self.current_model, dict) else "supervised",
                }
            }
            
            # 保存模型
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved successfully to {path}")
            logger.info(f"   Model version: {self.model_version}")
            logger.info(f"   Training history entries: {len(self.training_history)}")
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)

    async def load_model(self, version: str) -> bool:
        """加載模型 (實際實現)

        Args:
            version: 模型版本

        Returns:
            是否成功
        """
        model_path = self.model_dir / f"model_{version}.pkl"

        if not model_path.exists():
            logger.error(f"Model {version} not found at {model_path}")
            return False

        try:
            import pickle
            
            # 載入模型數據
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 恢復模型狀態
            self.current_model = model_data.get("model")
            self.model_version = model_data.get("version", version)
            self.training_history = model_data.get("training_history", [])
            
            # 驗證模型完整性
            if self.current_model is None:
                logger.error(f"Loaded model data is invalid")
                return False
            
            metadata = model_data.get("metadata", {})
            logger.info(f"✅ Model loaded successfully from {model_path}")
            logger.info(f"   Model version: {self.model_version}")
            logger.info(f"   Model type: {metadata.get('model_type', 'unknown')}")
            logger.info(f"   Saved at: {metadata.get('saved_at', 'unknown')}")
            logger.info(f"   Training history entries: {len(self.training_history)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False

    async def test_on_scenario(
        self, scenario: StandardScenario, model_version: str | None = None
    ) -> ScenarioTestResult:
        """在標準場景上測試模型

        Args:
            scenario: 標準場景
            model_version: 模型版本（None 使用當前版本）

        Returns:
            測試結果
        """
        if model_version:
            await self.load_model(model_version)

        logger.info(
            f"Testing model {self.model_version} on scenario {scenario.scenario_id}"
        )

        # TODO: 實現實際的場景測試邏輯
        # 這裡需要調用 PlanExecutor 執行生成的計畫

        # 模擬測試結果
        test_id = f"test_{uuid4().hex[:12]}"

        result = ScenarioTestResult(
            test_id=test_id,
            scenario_id=scenario.scenario_id,
            model_version=self.model_version,
            generated_plan=scenario.expected_plan,  # 模擬：使用預期計畫
            execution_result=None,  # type: ignore  # 需要實際執行
            score=75.0,  # 模擬分數
            comparison={},
            passed=True,
            tested_at=datetime.now(UTC),
        )

        return result

    def _increment_version(self, current_version: str) -> str:
        """遞增版本號

        Args:
            current_version: 當前版本

        Returns:
            新版本號
        """
        parts = current_version.lstrip("v").split(".")
        major, minor, patch = map(int, parts)

        # 遞增 patch 版本
        patch += 1

        return f"v{major}.{minor}.{patch}"

    async def _persist_training_result(self, result: ModelTrainingResult) -> None:
        """持久化訓練結果

        Args:
            result: 訓練結果
        """
        if not self.storage:
            return

        try:
            if hasattr(self.storage, "save_training_result"):
                await self.storage.save_training_result(result.model_dump())
                logger.debug(f"Persisted training result {result.training_id}")
        except Exception as e:
            logger.error(f"Failed to persist training result: {e}")
