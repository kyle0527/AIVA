"""Experience Data Loader - 經驗數據加載器

從經驗資料庫提取並準備訓練樣本
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExperienceDataLoader:
    """經驗數據加載器

    從經驗庫提取並準備訓練數據
    """

    def __init__(self, experience_repository: Any) -> None:
        """初始化數據加載器

        Args:
            experience_repository: 經驗資料庫存取層
        """
        self.repository = experience_repository
        logger.info("ExperienceDataLoader initialized")

    def load_training_batch(
        self,
        attack_type: str | None = None,
        min_score: float = 0.5,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """加載訓練批次

        Args:
            attack_type: 攻擊類型過濾
            min_score: 最低分數閾值
            batch_size: 批次大小

        Returns:
            (輸入特徵, 標籤) 元組
        """
        # 從資料庫查詢經驗
        experiences = self.repository.query_experiences(
            attack_type=attack_type, min_score=min_score, limit=batch_size
        )

        if not experiences:
            logger.warning("No experiences found for training")
            return np.array([]), np.array([])

        # 準備訓練數據
        X, y = self._prepare_training_data(experiences)

        logger.info(
            f"Loaded training batch: {len(experiences)} samples "
            f"(attack_type={attack_type})"
        )
        return X, y

    def _prepare_training_data(
        self, experiences: list[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """準備訓練數據

        Args:
            experiences: 經驗記錄列表

        Returns:
            (特徵矩陣, 標籤向量)
        """
        features = []
        labels = []

        for exp in experiences:
            # 提取特徵
            feature_vector = self._extract_features(exp)
            features.append(feature_vector)

            # 提取標籤 (使用 overall_score 作為標籤)
            labels.append(exp.overall_score)

        X = np.array(features)
        y = np.array(labels)

        return X, y

    def _extract_features(self, experience: Any) -> np.ndarray:
        """從經驗記錄提取特徵向量

        Args:
            experience: 經驗記錄

        Returns:
            特徵向量
        """
        # 簡化版本：從 AST 和 metrics 提取關鍵特徵
        features = []

        # AST 特徵
        ast_graph = experience.ast_graph
        features.append(len(ast_graph.get("nodes", [])))  # 節點數
        features.append(len(ast_graph.get("edges", [])))  # 邊數

        # Metrics 特徵
        metrics = experience.metrics_detail or {}
        features.append(metrics.get("completion_rate", 0.0))
        features.append(metrics.get("sequence_match_rate", 0.0))
        features.append(
            metrics.get("success_steps", 0) / max(metrics.get("expected_steps", 1), 1)
        )
        features.append(
            1.0
            - (metrics.get("error_count", 0) / max(metrics.get("expected_steps", 1), 1))
        )

        # 攻擊類型 one-hot encoding (簡化)
        attack_type_map = {"sqli": 0, "xss": 1, "ssrf": 2, "idor": 3}
        attack_type_idx = attack_type_map.get(experience.attack_type, -1)
        for i in range(len(attack_type_map)):
            features.append(1.0 if i == attack_type_idx else 0.0)

        # 填充到固定長度 (例如 1024)
        target_size = 1024
        current_size = len(features)
        if current_size < target_size:
            features.extend([0.0] * (target_size - current_size))
        else:
            features = features[:target_size]

        return np.array(features, dtype=np.float32)

    def load_dataset_samples(
        self, dataset_id: str, shuffle: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """從特定資料集加載樣本

        Args:
            dataset_id: 資料集 ID
            shuffle: 是否隨機打亂

        Returns:
            (特徵矩陣, 標籤向量)
        """
        samples = self.repository.get_dataset_samples(dataset_id)

        if not samples:
            logger.warning(f"No samples found in dataset {dataset_id}")
            return np.array([]), np.array([])

        # 提取經驗記錄
        experiences = [exp_record for _, exp_record in samples]

        X, y = self._prepare_training_data(experiences)

        if shuffle:
            # 隨機打亂
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

        logger.info(f"Loaded {len(X)} samples from dataset {dataset_id}")
        return X, y

    def create_validation_split(
        self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """創建驗證集分割

        Args:
            X: 特徵矩陣
            y: 標籤向量
            val_ratio: 驗證集比例

        Returns:
            (X_train, y_train, X_val, y_val)
        """
        n_samples = len(X)
        n_val = int(n_samples * val_ratio)

        # 隨機打亂
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        logger.info(
            f"Created train/val split: "
            f"{len(X_train)} train, {len(X_val)} validation"
        )
        return X_train, y_train, X_val, y_val
