"""ScalableBio Trainer - ScalableBioNet 專用訓練器

整合 ai_engine/training/trainer.py 的 ScalableBioNet 特定功能到統一的訓練框架中
"""

from dataclasses import dataclass
import logging
from typing import Any

# 使用可選依賴處理策略處理 numpy
try:
    import numpy as np
except ImportError:
    # 為類型標註提供 Mock numpy
    print("Warning: numpy not available, using mock types")
    
    class MockNdarray:
        """Mock numpy.ndarray for type annotations"""
        pass
    
    class MockNumpy:
        ndarray = MockNdarray
        
        @staticmethod
        def mean(arr):
            return 0.0
            
        @staticmethod
        def sum(arr):
            return 0
            
        @staticmethod
        def abs(arr):
            return arr
    
    np = MockNumpy()

logger = logging.getLogger(__name__)


@dataclass
class ScalableBioTrainingConfig:
    """ScalableBioNet 訓練配置"""

    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_delta: float = 0.001


class ScalableBioTrainer:
    """ScalableBioNet 專用訓練器

    繼承統一的 ModelTrainer 功能，並添加 ScalableBioNet 特定的訓練邏輯
    """

    def __init__(
        self, model: Any, config: ScalableBioTrainingConfig | None = None
    ) -> None:
        """初始化 ScalableBioNet 訓練器

        Args:
            model: ScalableBioNet 模型實例
            config: 訓練配置
        """
        self.model = model
        self.config = config or ScalableBioTrainingConfig()
        self.training_history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        logger.info("ScalableBioTrainer initialized")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """訓練 ScalableBioNet 模型

        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_val: 驗證特徵
            y_val: 驗證標籤

        Returns:
            訓練結果
        """
        logger.info(
            f"Starting ScalableBioNet training: {len(X_train)} samples, "
            f"{self.config.epochs} epochs"
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # 訓練一個 epoch
            train_loss, train_acc = self._train_epoch(X_train, y_train)

            # 記錄訓練指標
            self.training_history["loss"].append(train_loss)
            self.training_history["accuracy"].append(train_acc)

            # 驗證
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self._validate(X_val, y_val)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_accuracy"].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"acc={train_acc:.4f}, val_acc={val_acc:.4f}"
                )

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"loss={train_loss:.4f}, acc={train_acc:.4f}"
                )

        return {
            "final_loss": self.training_history["loss"][-1],
            "final_accuracy": self.training_history["accuracy"][-1],
            "best_val_loss": best_val_loss if X_val is not None else None,
            "epochs_trained": len(self.training_history["loss"]),
            "history": self.training_history,
        }

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """訓練一個 epoch

        Args:
            X: 輸入特徵
            y: 目標標籤

        Returns:
            (loss, accuracy)
        """
        total_loss = 0.0
        correct_predictions = 0
        total_samples = len(X)

        # 分批處理
        batch_size = self.config.batch_size
        for i in range(0, total_samples, batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            # 前向傳播
            outputs = self.model.forward(batch_X)
            loss = self._compute_loss(outputs, batch_y)

            # 反向傳播
            self.model.backward(batch_X, batch_y, self.config.learning_rate)

            total_loss += loss
            correct_predictions += self._count_correct_predictions(outputs, batch_y)

        avg_loss = total_loss / (total_samples / batch_size)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def _validate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """驗證模型

        Args:
            X: 驗證特徵
            y: 驗證標籤

        Returns:
            (loss, accuracy)
        """
        outputs = self.model.forward(X)
        loss = self._compute_loss(outputs, y)
        correct_predictions = self._count_correct_predictions(outputs, y)
        accuracy = correct_predictions / len(X)

        return loss, accuracy

    def _compute_loss(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """計算損失

        Args:
            outputs: 模型輸出
            targets: 目標值

        Returns:
            損失值
        """
        # 使用均方誤差
        return float(np.mean((outputs - targets) ** 2))

    def _count_correct_predictions(
        self, outputs: np.ndarray, targets: np.ndarray
    ) -> int:
        """計算正確預測數量

        Args:
            outputs: 模型輸出
            targets: 目標值

        Returns:
            正確預測數量
        """
        # 對於回歸任務，使用閾值判斷
        threshold = 0.1
        predictions = np.abs(outputs - targets) < threshold
        return int(np.sum(predictions))

    def get_training_history(self) -> dict[str, list[float]]:
        """獲取訓練歷史

        Returns:
            訓練歷史
        """
        return self.training_history.copy()

    def save_model(self, path: str) -> None:
        """保存模型

        Args:
            path: 保存路徑
        """
        if hasattr(self.model, "save_weights"):
            self.model.save_weights(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """載入模型

        Args:
            path: 模型路徑
        """
        if hasattr(self.model, "load_weights"):
            self.model.load_weights(path)
        logger.info(f"Model loaded from {path}")
