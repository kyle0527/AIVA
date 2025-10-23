"""
經驗資料庫存取層 (Experience Repository)

提供統一的介面來存取和管理經驗記錄
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import Session, sessionmaker

from .experience_models import (
    Base,
    DatasetSample,
    ExperienceRecord,
    ModelTrainingHistory,
    TrainingDataset,
)

logger = logging.getLogger(__name__)


class ExperienceRepository:
    """經驗資料庫存取層

    提供對經驗記錄的增刪查改操作
    """

    def __init__(self, database_url: str) -> None:
        """初始化資料庫連接

        Args:
            database_url: 資料庫連接字串
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # 創建資料表
        Base.metadata.create_all(self.engine)

        logger.info(f"ExperienceRepository initialized with DB: {database_url}")

    def get_session(self) -> Session:
        """獲取資料庫會話"""
        return self.SessionLocal()

    def save_experience(
        self,
        plan_id: str,
        attack_type: str,
        ast_graph: dict[str, Any],
        execution_trace: dict[str, Any],
        metrics: dict[str, Any],
        feedback: dict[str, Any],
        target_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperienceRecord:
        """保存經驗記錄

        Args:
            plan_id: 執行計畫 ID
            attack_type: 攻擊類型
            ast_graph: AST 圖結構
            execution_trace: 執行軌跡
            metrics: 比較指標
            feedback: 回饋數據
            target_info: 目標資訊
            metadata: 元數據

        Returns:
            儲存的經驗記錄
        """
        session = self.get_session()
        try:
            experience_id = f"exp_{uuid4().hex[:8]}"

            record = ExperienceRecord(
                experience_id=experience_id,
                plan_id=plan_id,
                attack_type=attack_type,
                target_info=target_info,
                ast_graph=ast_graph,
                trace_session_id=execution_trace.get("trace_session_id", ""),
                execution_trace=execution_trace,
                completion_rate=metrics.get("completion_rate"),
                sequence_match_rate=metrics.get("sequence_match_rate"),
                overall_score=metrics.get("overall_score"),
                metrics_detail=metrics,
                reward=feedback.get("reward"),
                feedback_data=feedback,
                execution_success=metrics.get("success_steps"),
                execution_failed=metrics.get("failed_steps"),
                error_count=metrics.get("error_count"),
                extra_metadata=metadata,
            )

            session.add(record)
            session.commit()
            session.refresh(record)

            logger.info(
                f"Saved experience {experience_id} "
                f"(score: {record.overall_score:.2f})"
            )
            return record

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save experience: {e}")
            raise
        finally:
            session.close()

    def get_experience(self, experience_id: str) -> ExperienceRecord | None:
        """獲取單一經驗記錄

        Args:
            experience_id: 經驗 ID

        Returns:
            經驗記錄或 None
        """
        session = self.get_session()
        try:
            return (
                session.query(ExperienceRecord)
                .filter(ExperienceRecord.experience_id == experience_id)
                .first()
            )
        finally:
            session.close()

    def query_experiences(
        self,
        attack_type: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ExperienceRecord]:
        """查詢經驗記錄

        Args:
            attack_type: 攻擊類型過濾
            min_score: 最低分數
            max_score: 最高分數
            limit: 返回數量限制
            offset: 偏移量

        Returns:
            經驗記錄列表
        """
        session = self.get_session()
        try:
            query = session.query(ExperienceRecord)

            if attack_type:
                query = query.filter(ExperienceRecord.attack_type == attack_type)

            if min_score is not None:
                query = query.filter(ExperienceRecord.overall_score >= min_score)

            if max_score is not None:
                query = query.filter(ExperienceRecord.overall_score <= max_score)

            # 按分數降序排列
            query = query.order_by(desc(ExperienceRecord.overall_score))

            return query.limit(limit).offset(offset).all()

        finally:
            session.close()

    def get_top_experiences(
        self, attack_type: str | None = None, top_k: int = 10
    ) -> list[ExperienceRecord]:
        """獲取評分最高的經驗記錄

        Args:
            attack_type: 攻擊類型過濾
            top_k: 返回數量

        Returns:
            頂級經驗記錄列表
        """
        return self.query_experiences(
            attack_type=attack_type, min_score=0.7, limit=top_k
        )

    def create_training_dataset(
        self,
        name: str,
        description: str | None = None,
        attack_types: list[str] | None = None,
        min_score: float = 0.5,
        max_samples: int = 1000,
    ) -> TrainingDataset:
        """創建訓練資料集

        Args:
            name: 資料集名稱
            description: 描述
            attack_types: 包含的攻擊類型
            min_score: 最低分數閾值
            max_samples: 最大樣本數

        Returns:
            訓練資料集
        """
        session = self.get_session()
        try:
            dataset_id = f"dataset_{uuid4().hex[:8]}"

            dataset = TrainingDataset(
                dataset_id=dataset_id,
                name=name,
                description=description,
                attack_types=attack_types,
                min_score_threshold=min_score,
                max_samples=max_samples,
            )

            session.add(dataset)
            session.commit()
            session.refresh(dataset)

            logger.info(f"Created training dataset '{name}' ({dataset_id})")
            return dataset

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create training dataset: {e}")
            raise
        finally:
            session.close()

    def add_samples_to_dataset(self, dataset_id: str, experience_ids: list[str]) -> int:
        """將經驗樣本加入資料集

        Args:
            dataset_id: 資料集 ID
            experience_ids: 經驗 ID 列表

        Returns:
            成功添加的樣本數
        """
        session = self.get_session()
        try:
            count = 0
            for exp_id in experience_ids:
                sample = DatasetSample(dataset_id=dataset_id, experience_id=exp_id)
                session.add(sample)
                count += 1

            session.commit()
            logger.info(f"Added {count} samples to dataset {dataset_id}")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add samples to dataset: {e}")
            raise
        finally:
            session.close()

    def get_dataset_samples(
        self, dataset_id: str
    ) -> list[tuple[DatasetSample, ExperienceRecord]]:
        """獲取資料集的所有樣本

        Args:
            dataset_id: 資料集 ID

        Returns:
            (樣本, 經驗記錄) 元組列表
        """
        session = self.get_session()
        try:
            results = (
                session.query(DatasetSample, ExperienceRecord)
                .join(
                    ExperienceRecord,
                    DatasetSample.experience_id == ExperienceRecord.experience_id,
                )
                .filter(DatasetSample.dataset_id == dataset_id)
                .all()
            )
            return results
        finally:
            session.close()

    def save_training_history(
        self,
        model_name: str,
        model_version: str,
        dataset_id: str,
        training_config: dict[str, Any],
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> ModelTrainingHistory:
        """保存訓練歷史

        Args:
            model_name: 模型名稱
            model_version: 模型版本
            dataset_id: 資料集 ID
            training_config: 訓練配置
            status: 狀態
            metadata: 元數據

        Returns:
            訓練歷史記錄
        """
        session = self.get_session()
        try:
            training_id = f"train_{uuid4().hex[:8]}"

            history = ModelTrainingHistory(
                training_id=training_id,
                model_name=model_name,
                model_version=model_version,
                dataset_id=dataset_id,
                training_config=training_config,
                completed_at=datetime.now(),
                status=status,
                extra_metadata=metadata,
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.info(
                f"Created training history {training_id} "
                f"for model {model_name} v{model_version}"
            )
            return history

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save training history: {e}")
            raise
        finally:
            session.close()

    def update_training_history(
        self,
        training_id: str,
        status: str | None = None,
        final_loss: float | None = None,
        final_accuracy: float | None = None,
        training_metrics: dict[str, Any] | None = None,
        model_path: str | None = None,
    ) -> None:
        """更新訓練歷史

        Args:
            training_id: 訓練 ID
            status: 新狀態
            final_loss: 最終損失
            final_accuracy: 最終準確度
            training_metrics: 訓練指標
            model_path: 模型路徑
        """
        session = self.get_session()
        try:
            history = (
                session.query(ModelTrainingHistory)
                .filter(ModelTrainingHistory.training_id == training_id)
                .first()
            )

            if not history:
                logger.warning(f"Training history {training_id} not found")
                return

            if status:
                history.status = status
            if final_loss is not None:
                history.final_loss = final_loss
            if final_accuracy is not None:
                history.final_accuracy = final_accuracy
            if training_metrics:
                history.training_metrics = training_metrics
            if model_path:
                history.model_path = model_path

            if status in ("completed", "failed"):
                history.completed_at = datetime.now()

            session.commit()
            logger.info(f"Updated training history {training_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update training history: {e}")
            raise
        finally:
            session.close()

    def get_statistics(self) -> dict[str, Any]:
        """獲取經驗庫統計資訊

        Returns:
            統計字典
        """
        session = self.get_session()
        try:
            total_experiences = session.query(func.count(ExperienceRecord.id)).scalar()

            avg_score = session.query(func.avg(ExperienceRecord.overall_score)).scalar()

            attack_type_counts = (
                session.query(
                    ExperienceRecord.attack_type,
                    func.count(ExperienceRecord.id),
                )
                .group_by(ExperienceRecord.attack_type)
                .all()
            )

            return {
                "total_experiences": total_experiences,
                "average_score": float(avg_score) if avg_score else 0.0,
                "attack_type_distribution": dict(attack_type_counts),
            }

        finally:
            session.close()
