"""
Model Trainer - 模型訓練器

執行模型訓練和微調
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """訓練配置"""

    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_delta: float = 0.001


class ModelTrainer:
    """模型訓練器

    對 ScalableBioNet 進行訓練和微調
    """

    def __init__(self, model: Any, config: TrainingConfig | None = None) -> None:
        """初始化訓練器

        Args:
            model: 要訓練的模型 (ScalableBioNet)
            config: 訓練配置
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.training_history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        logger.info("ModelTrainer initialized")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """訓練模型

        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_val: 驗證特徵
            y_val: 驗證標籤

        Returns:
            訓練結果
        """
        logger.info(
            f"Starting training: {len(X_train)} samples, "
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
                    f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"loss={train_loss:.4f}, acc={train_acc:.4f}"
                )

        # 返回訓練結果
        final_metrics = {
            "final_loss": self.training_history["loss"][-1],
            "final_accuracy": self.training_history["accuracy"][-1],
            "best_val_loss": best_val_loss if X_val is not None else None,
            "epochs_trained": len(self.training_history["loss"]),
            "history": self.training_history,
        }

        logger.info(
            f"Training completed: "
            f"final_loss={final_metrics['final_loss']:.4f}, "
            f"final_acc={final_metrics['final_accuracy']:.4f}"
        )
        return final_metrics

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """訓練一個 epoch

        Args:
            X: 特徵
            y: 標籤

        Returns:
            (平均損失, 準確度)
        """
        n_samples = len(X)
        batch_size = self.config.batch_size

        total_loss = 0.0
        correct = 0

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            # Forward pass
            predictions = self._forward_batch(batch_X)

            # 計算損失 (Mean Squared Error)
            loss = np.mean((predictions - batch_y) ** 2)
            total_loss += loss

            # 計算準確度 (簡化版：預測值與真實值差距小於 0.1 視為正確)
            correct += np.sum(np.abs(predictions - batch_y) < 0.1)

            # Backward pass (簡化版：梯度下降)
            self._backward_batch(batch_X, batch_y, predictions)

        avg_loss = total_loss / (n_samples // batch_size)
        accuracy = correct / n_samples

        return avg_loss, accuracy

    def _validate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """驗證模型

        Args:
            X: 驗證特徵
            y: 驗證標籤

        Returns:
            (驗證損失, 驗證準確度)
        """
        predictions = self._forward_batch(X)
        loss = np.mean((predictions - y) ** 2)
        accuracy = np.sum(np.abs(predictions - y) < 0.1) / len(y)
        return loss, accuracy

    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        """批次前向傳播

        Args:
            X: 輸入批次

        Returns:
            預測值
        """
        predictions = []
        for x in X:
            # 使用模型的 forward 方法
            pred = self.model.forward(x)
            # 取最大機率作為預測值 (簡化)
            predictions.append(np.max(pred))
        return np.array(predictions)

    def _backward_batch(
        self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray
    ) -> None:
        """批次反向傳播 (簡化版)

        Args:
            X: 輸入批次
            y: 標籤
            predictions: 預測值
        """
        # 簡化的梯度下降更新
        # 實際實現需要完整的反向傳播算法
        lr = self.config.learning_rate
        error = predictions - y

        # 更新第一層權重 (簡化)
        for i in range(min(len(X), 10)):  # 只更新部分樣本以節省計算
            if i < len(error):
                gradient = error[i] * X[i][: self.model.fc1.shape[1]]
                self.model.fc1 -= lr * np.outer(
                    gradient, np.ones(self.model.fc1.shape[1])
                )

    def get_training_summary(self) -> dict[str, Any]:
        """獲取訓練摘要

        Returns:
            訓練摘要字典
        """
        return {
            "config": {
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
            },
            "history": self.training_history,
            "final_metrics": {
                "loss": (
                    self.training_history["loss"][-1]
                    if self.training_history["loss"]
                    else None
                ),
                "accuracy": (
                    self.training_history["accuracy"][-1]
                    if self.training_history["accuracy"]
                    else None
                ),
            },
        }
