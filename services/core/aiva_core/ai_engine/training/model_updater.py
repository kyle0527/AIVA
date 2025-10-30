"""
Model Updater - 模型更新器

負責定期從經驗庫提取樣本並更新模型
"""



import logging
from pathlib import Path
import pickle
from typing import Any

from .data_loader import ExperienceDataLoader
from ...learning.scalable_bio_trainer import ScalableBioTrainer, ScalableBioTrainingConfig

logger = logging.getLogger(__name__)


class ModelUpdater:
    """模型更新器

    協調數據加載、訓練和模型保存的完整流程
    """

    def __init__(
        self,
        model: Any,
        experience_repository: Any,
        model_save_path: str = "./models",
    ) -> None:
        """初始化更新器

        Args:
            model: 要更新的模型
            experience_repository: 經驗資料庫
            model_save_path: 模型保存路徑
        """
        self.model = model
        self.repository = experience_repository
        self.data_loader = ExperienceDataLoader(experience_repository)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelUpdater initialized, save path: {model_save_path}")

    def update_from_recent_experiences(
        self,
        min_score: float = 0.6,
        max_samples: int = 1000,
        training_config: ScalableBioTrainingConfig | None = None,
    ) -> dict[str, Any]:
        """從最近的經驗更新模型

        Args:
            min_score: 最低分數閾值
            max_samples: 最大樣本數
            training_config: 訓練配置

        Returns:
            更新結果
        """
        logger.info(
            f"Starting model update from recent experiences "
            f"(min_score={min_score}, max_samples={max_samples})"
        )

        # 1. 加載訓練數據
        X, y = self.data_loader.load_training_batch(
            min_score=min_score, batch_size=max_samples
        )

        if len(X) == 0:
            logger.warning("No training data available")
            return {"status": "failed", "reason": "no_data"}

        # 2. 分割訓練/驗證集
        X_train, y_train, X_val, y_val = self.data_loader.create_validation_split(X, y)

        # 3. 訓練模型
        trainer = ScalableBioTrainer(self.model, config=training_config)
        training_results = trainer.train(X_train, y_train, X_val, y_val)

        # 4. 保存模型
        model_path = self._save_model()

        # 5. 記錄訓練歷史到資料庫
        training_history = self.repository.save_training_history(
            model_name="ScalableBioNet",
            model_version="1.0",
            dataset_id="recent_experiences",
            training_config=training_config.__dict__ if training_config else {},
            status="completed",
            metadata={
                "samples_used": len(X),
                "min_score_threshold": min_score,
            },
        )

        self.repository.update_training_history(
            training_id=training_history.training_id,
            status="completed",
            final_loss=training_results["final_loss"],
            final_accuracy=training_results["final_accuracy"],
            training_metrics=training_results,
            model_path=str(model_path),
        )

        logger.info(
            f"Model update completed: "
            f"loss={training_results['final_loss']:.4f}, "
            f"acc={training_results['final_accuracy']:.4f}"
        )

        return {
            "status": "success",
            "training_results": training_results,
            "model_path": str(model_path),
            "training_id": training_history.training_id,
            "samples_used": len(X),
        }

    def update_from_dataset(
        self, dataset_id: str, training_config: ScalableBioTrainingConfig | None = None
    ) -> dict[str, Any]:
        """從特定資料集更新模型

        Args:
            dataset_id: 資料集 ID
            training_config: 訓練配置

        Returns:
            更新結果
        """
        logger.info(f"Starting model update from dataset {dataset_id}")

        # 1. 加載資料集
        X, y = self.data_loader.load_dataset_samples(dataset_id)

        if len(X) == 0:
            logger.warning(f"No samples in dataset {dataset_id}")
            return {"status": "failed", "reason": "no_data"}

        # 2. 分割訓練/驗證集
        X_train, y_train, X_val, y_val = self.data_loader.create_validation_split(X, y)

        # 3. 訓練模型
        trainer = ScalableBioTrainer(self.model, config=training_config)
        training_results = trainer.train(X_train, y_train, X_val, y_val)

        # 4. 保存模型
        model_path = self._save_model()

        # 5. 記錄訓練歷史
        training_history = self.repository.save_training_history(
            model_name="ScalableBioNet",
            model_version="1.0",
            dataset_id=dataset_id,
            training_config=training_config.__dict__ if training_config else {},
            status="completed",
        )

        self.repository.update_training_history(
            training_id=training_history.training_id,
            status="completed",
            final_loss=training_results["final_loss"],
            final_accuracy=training_results["final_accuracy"],
            training_metrics=training_results,
            model_path=str(model_path),
        )

        logger.info(f"Model update from dataset {dataset_id} completed")

        return {
            "status": "success",
            "training_results": training_results,
            "model_path": str(model_path),
            "training_id": training_history.training_id,
            "dataset_id": dataset_id,
            "samples_used": len(X),
        }

    def _save_model(self) -> Path:
        """保存模型

        Returns:
            模型保存路徑
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scalable_bionet_{timestamp}.pkl"
        filepath = self.model_save_path / filename

        # 保存模型權重
        model_state = {
            "fc1": self.model.fc1,
            "fc2": self.model.fc2,
            "spiking1_weights": self.model.spiking1.weights,
            "metadata": {
                "total_params": self.model.total_params,
                "input_size": self.model.fc1.shape[0],
                "num_tools": self.model.fc2.shape[1],
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_state, f)

        logger.info(f"Model saved to {filepath}")
        return filepath

    def load_model(self, model_path: str) -> None:
        """加載模型

        Args:
            model_path: 模型路徑
        """
        with open(model_path, "rb") as f:
            model_state = pickle.load(f)

        self.model.fc1 = model_state["fc1"]
        self.model.fc2 = model_state["fc2"]
        self.model.spiking1.weights = model_state["spiking1_weights"]

        logger.info(f"Model loaded from {model_path}")

    def schedule_periodic_update(
        self, interval_hours: int = 24, min_new_experiences: int = 100
    ) -> None:
        """排程定期更新 (TODO: 需要與排程系統整合)

        Args:
            interval_hours: 更新間隔（小時）
            min_new_experiences: 最少新經驗數
        """
        logger.info(
            f"Scheduling periodic updates every {interval_hours} hours "
            f"(min {min_new_experiences} new experiences)"
        )
        # TODO: 實現與 Celery 或其他排程系統的整合
