"""
Experience Manager - 經驗樣本管理器

負責收集、存儲和管理執行經驗樣本，用於機器學習訓練

符合標準：
- 數據標註遵循監督學習最佳實踐
- 支持批量訓練和在線學習
- 與強化學習框架（如 Stable Baselines3）兼容
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any
from uuid import uuid4

from aiva_common.schemas import (
    AttackPlan,
    ExperienceSample,
    PlanExecutionMetrics,
    PlanExecutionResult,
)

logger = logging.getLogger(__name__)


class ExperienceManager:
    """經驗樣本管理器

    收集、標註和存儲執行經驗，用於模型訓練
    """

    def __init__(self, storage_backend: Any | None = None) -> None:
        """初始化經驗管理器

        Args:
            storage_backend: 儲存後端
        """
        self.storage = storage_backend
        logger.info("ExperienceManager initialized")

    async def create_experience_sample(
        self,
        result: PlanExecutionResult,
        auto_label: bool = True,
    ) -> ExperienceSample:
        """創建經驗樣本

        Args:
            result: 執行結果
            auto_label: 是否自動標註

        Returns:
            經驗樣本
        """
        sample_id = f"exp_{uuid4().hex[:12]}"

        # 提取場景上下文
        context = self._extract_context(result.plan)

        # 自動標記
        label = self._auto_label(result.metrics) if auto_label else "unlabeled"

        # 計算樣本質量分數
        quality_score = self._calculate_quality_score(result)

        sample = ExperienceSample(
            sample_id=sample_id,
            plan_id=result.plan_id,
            session_id=result.session_id,
            context=context,
            plan=result.plan,
            trace=result.trace,
            metrics=result.metrics,
            result=result,
            label=label,
            quality_score=quality_score,
            created_at=datetime.now(UTC),
        )

        # 持久化樣本
        if self.storage:
            await self._persist_sample(sample)

        logger.info(
            f"Created experience sample {sample_id} "
            f"(label={label}, quality={quality_score:.2f})"
        )

        return sample

    def _extract_context(self, plan: AttackPlan) -> dict[str, Any]:
        """提取場景上下文

        Args:
            plan: 攻擊計畫

        Returns:
            場景上下文字典
        """
        return {
            "attack_type": plan.attack_type.value,
            "target_info": plan.target_info,
            "num_steps": len(plan.steps),
            "mitre_techniques": plan.mitre_techniques,
            "mitre_tactics": plan.mitre_tactics,
            "created_by": plan.created_by,
            **plan.context,
        }

    def _auto_label(self, metrics: PlanExecutionMetrics) -> str:
        """自動標註樣本

        Args:
            metrics: 執行指標

        Returns:
            標籤: "success", "failure", "partial_success"
        """
        if metrics.goal_achieved and metrics.success_rate >= 0.9:
            return "success"
        elif metrics.completion_rate >= 0.5 and metrics.success_rate >= 0.6:
            return "partial_success"
        else:
            return "failure"

    def _calculate_quality_score(self, result: PlanExecutionResult) -> float:
        """計算樣本質量分數

        Args:
            result: 執行結果

        Returns:
            質量分數 (0.0-1.0)
        """
        # 基於多個因素計算質量分數
        factors = []

        # 1. 執行完整度
        completion_factor = result.metrics.completion_rate
        factors.append(completion_factor * 0.3)

        # 2. 數據完整性
        trace_completeness = min(
            (
                sum(1 for t in result.trace if t.input_data and t.output_data)
                / len(result.trace)
                if result.trace
                else 0
            ),
            1.0,
        )
        factors.append(trace_completeness * 0.3)

        # 3. 結果明確性
        result_clarity = 1.0 if result.metrics.goal_achieved is not None else 0.5
        factors.append(result_clarity * 0.2)

        # 4. 異常處理
        anomaly_factor = 1.0 - min(len(result.anomalies) / 10.0, 1.0)
        factors.append(anomaly_factor * 0.2)

        return sum(factors)

    async def _persist_sample(self, sample: ExperienceSample) -> None:
        """持久化經驗樣本

        Args:
            sample: 經驗樣本
        """
        try:
            if self.storage and hasattr(self.storage, "save_experience_sample"):
                # 直接傳遞 ExperienceSample 對象，由後端處理轉換
                await self.storage.save_experience_sample(sample)
                logger.debug(f"Persisted sample {sample.sample_id} to storage backend")
            else:
                logger.warning(
                    "Storage backend not configured or does not support save_experience_sample"
                )
        except Exception as e:
            logger.error(f"Failed to persist sample {sample.sample_id}: {e}")

    async def get_samples_for_training(
        self,
        label_filter: list[str] | None = None,
        min_quality: float = 0.5,
        limit: int = 1000,
    ) -> list[ExperienceSample]:
        """獲取用於訓練的樣本

        Args:
            label_filter: 標籤過濾器
            min_quality: 最小質量分數
            limit: 最大數量

        Returns:
            樣本列表
        """
        if not self.storage:
            logger.warning("No storage backend configured")
            return []

        try:
            if hasattr(self.storage, "get_experience_samples"):
                samples_data = await self.storage.get_experience_samples(
                    label_filter=label_filter,
                    min_quality=min_quality,
                    limit=limit,
                )
                return [ExperienceSample(**data) for data in samples_data]
            else:
                logger.warning(
                    "Storage backend does not support get_experience_samples"
                )
                return []
        except Exception as e:
            logger.error(f"Failed to get training samples: {e}")
            return []

    async def annotate_sample(
        self,
        sample_id: str,
        annotations: dict[str, Any],
        annotated_by: str | None = None,
    ) -> bool:
        """人工標註樣本

        Args:
            sample_id: 樣本 ID
            annotations: 標註內容
            annotated_by: 標註者

        Returns:
            是否成功
        """
        if not self.storage:
            logger.warning("No storage backend configured")
            return False

        try:
            annotation_data = {
                "sample_id": sample_id,
                "annotations": annotations,
                "annotated_by": annotated_by,
                "annotated_at": datetime.now(UTC).isoformat(),
            }

            if hasattr(self.storage, "update_sample_annotations"):
                await self.storage.update_sample_annotations(annotation_data)
                logger.info(f"Annotated sample {sample_id}")
                return True
            else:
                logger.warning(
                    "Storage backend does not support update_sample_annotations"
                )
                return False
        except Exception as e:
            logger.error(f"Failed to annotate sample {sample_id}: {e}")
            return False

    async def get_statistics(self) -> dict[str, Any]:
        """獲取經驗庫統計資訊

        Returns:
            統計資訊
        """
        if not self.storage:
            logger.warning("No storage backend configured")
            return {}

        try:
            if hasattr(self.storage, "get_experience_statistics"):
                return await self.storage.get_experience_statistics()
            else:
                logger.warning(
                    "Storage backend does not support get_experience_statistics"
                )
                return {}
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
