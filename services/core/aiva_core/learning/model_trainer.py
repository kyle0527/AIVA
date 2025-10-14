"""
Model Trainer - 強化學習模型訓練器

負責模型的訓練、評估、微調和部署，支持監督學習和強化學習
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiva_common.schemas import (
    ExperienceSample,
    ModelTrainingConfig,
    ModelTrainingResult,
    ScenarioTestResult,
    StandardScenario,
)
import numpy as np

logger = logging.getLogger(__name__)


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
        features = []

        # 1. 攻擊類型特徵（one-hot encoding）
        attack_types = ["SQLI", "XSS", "SSRF", "IDOR", "other"]
        attack_type = sample.context.get("attack_type", "other")
        attack_one_hot = [1 if t == attack_type else 0 for t in attack_types]
        features.extend(attack_one_hot)

        # 2. 計畫複雜度特徵
        features.append(len(sample.plan.steps))  # 步驟數量
        features.append(len(sample.plan.dependencies))  # 依賴關係數量

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
        features.append(1 if target_info.get("waf_detected") else 0)
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
        """執行監督學習訓練

        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_val: 驗證特徵
            y_val: 驗證標籤
            config: 訓練配置

        Returns:
            訓練指標
        """
        # TODO: 實現實際的模型訓練邏輯
        # 這裡需要整合實際的機器學習框架（如 PyTorch, TensorFlow, sklearn）

        logger.info("Training supervised model (placeholder implementation)")

        # 模擬訓練過程
        epochs = config.epochs
        for epoch in range(epochs):
            # 模擬訓練迭代
            logger.debug(f"Epoch {epoch+1}/{epochs}")

        return {
            "loss": 0.15,  # 模擬損失
            "final_epoch": epochs,
        }

    async def _train_model_rl(
        self, episodes: list[dict[str, Any]], config: ModelTrainingConfig
    ) -> dict[str, Any]:
        """執行強化學習訓練

        Args:
            episodes: 訓練 episodes
            config: 訓練配置

        Returns:
            訓練指標
        """
        # TODO: 實現實際的強化學習訓練
        # 可以整合 Stable Baselines3, RLlib 等框架

        logger.info("Training RL model (placeholder implementation)")

        total_rewards = [ep["total_reward"] for ep in episodes]
        average_reward = np.mean(total_rewards) if total_rewards else 0.0

        return {
            "average_reward": float(average_reward),
            "max_reward": float(max(total_rewards)) if total_rewards else 0.0,
            "min_reward": float(min(total_rewards)) if total_rewards else 0.0,
            "episodes": len(episodes),
            "loss": 0.1,
        }

    async def _evaluate_model(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, Any]:
        """評估模型

        Args:
            X_val: 驗證特徵
            y_val: 驗證標籤

        Returns:
            評估指標
        """
        # TODO: 實現實際的模型評估
        logger.info("Evaluating model (placeholder implementation)")

        return {
            "loss": 0.12,
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
        }

    async def _save_model(self, path: Path) -> None:
        """保存模型

        Args:
            path: 保存路徑
        """
        # TODO: 實現實際的模型保存邏輯
        logger.info(f"Saving model to {path} (placeholder)")

        # 創建占位文件
        path.write_text(f"Model version: {self.model_version}\n")

    async def load_model(self, version: str) -> bool:
        """加載模型

        Args:
            version: 模型版本

        Returns:
            是否成功
        """
        model_path = self.model_dir / f"model_{version}.pkl"

        if not model_path.exists():
            logger.error(f"Model {version} not found at {model_path}")
            return False

        # TODO: 實現實際的模型加載邏輯
        logger.info(f"Loading model {version} (placeholder)")

        self.model_version = version
        return True

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
