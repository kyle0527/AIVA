"""
Training Orchestrator - 訓練編排器

整合 RAG、場景管理、模型訓練，實現完整的自動化訓練流程
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from aiva_common.schemas import (
    AttackPlan,
    ExperienceSample,
)
from aiva_core.execution.plan_executor import PlanExecutor
from aiva_core.learning.experience_manager import ExperienceManager
from aiva_core.learning.model_trainer import ModelTrainer
from aiva_core.rag import RAGEngine
from aiva_core.training.scenario_manager import (
    Scenario,
    ScenarioManager,
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """訓練編排器

    協調整個訓練流程：
    1. 從場景管理器加載場景
    2. 使用 RAG 增強計畫生成
    3. 執行攻擊計畫
    4. 收集經驗並添加到知識庫
    5. 訓練模型
    6. 評估性能並迭代
    """

    def __init__(
        self,
        scenario_manager: ScenarioManager,
        rag_engine: RAGEngine,
        plan_executor: PlanExecutor,
        experience_manager: ExperienceManager,
        model_trainer: ModelTrainer,
        data_directory: Path | None = None,
    ) -> None:
        """初始化訓練編排器

        Args:
            scenario_manager: 場景管理器
            rag_engine: RAG 引擎
            plan_executor: 計畫執行器
            experience_manager: 經驗管理器
            model_trainer: 模型訓練器
            data_directory: 數據目錄
        """
        self.scenario_manager = scenario_manager
        self.rag_engine = rag_engine
        self.plan_executor = plan_executor
        self.experience_manager = experience_manager
        self.model_trainer = model_trainer

        self.data_directory = data_directory or Path("./data/training")
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # 訓練狀態
        self.training_sessions: list[dict[str, Any]] = []
        self.current_session: dict[str, Any] | None = None

        logger.info("TrainingOrchestrator initialized")

    async def run_training_episode(
        self,
        scenario_id: str,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """運行單個訓練回合

        Args:
            scenario_id: 場景 ID
            use_rag: 是否使用 RAG 增強

        Returns:
            訓練結果字典
        """
        # 1. 加載場景
        scenario = self.scenario_manager.get_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        logger.info(
            f"Starting training episode for scenario: {scenario.name} "
            f"(RAG: {use_rag})"
        )

        # 2. 生成或增強攻擊計畫
        if use_rag:
            # 使用 RAG 獲取上下文
            rag_context = self.rag_engine.enhance_attack_plan(
                target=scenario.target,
                objective=scenario.objective,
            )

            # TODO: 這裡應該調用 AI 模型生成計畫
            # 現在使用場景中的預定義計畫
            attack_plan = scenario.attack_plan

            logger.info(
                f"Enhanced plan with RAG: "
                f"{len(rag_context['similar_techniques'])} techniques, "
                f"{len(rag_context['successful_experiences'])} experiences"
            )
        else:
            attack_plan = scenario.attack_plan

        # 3. 執行計畫
        execution_result = await self.plan_executor.execute_plan(plan=attack_plan)

        # 4. 收集經驗
        if execution_result.trace_records:
            # 從執行結果提取經驗樣本
            samples = await self._extract_experience_samples(
                scenario=scenario,
                attack_plan=attack_plan,
                execution_result=execution_result,
            )

            # 添加到經驗管理器
            for sample in samples:
                self.experience_manager.add_sample(sample)

                # 從經驗中學習（添加到 RAG 知識庫）
                if use_rag:
                    self.rag_engine.learn_from_experience(sample)

            logger.info(f"Collected {len(samples)} experience samples")

        # 5. 計算回合結果
        episode_result = {
            "scenario_id": scenario_id,
            "timestamp": datetime.now().isoformat(),
            "use_rag": use_rag,
            "success": execution_result.success,
            "steps_executed": len(execution_result.trace_records),
            "metrics": {
                "completion_rate": execution_result.metrics.get("completion_rate", 0.0),
                "success_rate": execution_result.metrics.get("success_rate", 0.0),
                "total_reward": execution_result.metrics.get("total_reward", 0.0),
            },
            "samples_collected": (
                len(execution_result.trace_records)
                if execution_result.trace_records
                else 0
            ),
        }

        logger.info(
            f"Episode completed: success={episode_result['success']}, "
            f"steps={episode_result['steps_executed']}"
        )

        return episode_result

    async def _extract_experience_samples(
        self,
        scenario: Scenario,
        attack_plan: AttackPlan,
        execution_result: dict[str, Any],
    ) -> list[ExperienceSample]:
        """從執行結果提取經驗樣本

        Args:
            scenario: 場景
            attack_plan: 攻擊計畫
            execution_result: 執行結果

        Returns:
            經驗樣本列表
        """
        samples: list[ExperienceSample] = []

        # 這裡需要使用 PlanComparator 來計算獎勵
        # TODO: 實現完整的經驗提取邏輯

        return samples

    async def run_training_batch(
        self,
        scenario_ids: list[str] | None = None,
        episodes_per_scenario: int = 10,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """運行批量訓練

        Args:
            scenario_ids: 場景 ID 列表（None 表示全部）
            episodes_per_scenario: 每個場景的回合數
            use_rag: 是否使用 RAG 增強

        Returns:
            批量訓練結果
        """
        # 獲取場景列表
        if scenario_ids is None:
            scenarios = self.scenario_manager.list_scenarios()
            scenario_ids = [s["id"] for s in scenarios]

        logger.info(
            f"Starting batch training: "
            f"{len(scenario_ids)} scenarios, "
            f"{episodes_per_scenario} episodes each"
        )

        # 訓練會話
        session = {
            "session_id": f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "scenario_ids": scenario_ids,
            "episodes_per_scenario": episodes_per_scenario,
            "use_rag": use_rag,
            "episodes": [],
        }

        self.current_session = session

        # 運行所有回合
        for scenario_id in scenario_ids:
            logger.info(f"\nTraining on scenario: {scenario_id}")

            for episode_num in range(episodes_per_scenario):
                logger.info(f"  Episode {episode_num + 1}/{episodes_per_scenario}")

                try:
                    episode_result = await self.run_training_episode(
                        scenario_id=scenario_id,
                        use_rag=use_rag,
                    )

                    session["episodes"].append(episode_result)

                except Exception as e:
                    logger.error(f"Episode failed: {e}", exc_info=True)
                    session["episodes"].append(
                        {
                            "scenario_id": scenario_id,
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                            "success": False,
                        }
                    )

        # 完成會話
        session["end_time"] = datetime.now().isoformat()
        session["total_episodes"] = len(session["episodes"])
        session["successful_episodes"] = sum(
            1 for ep in session["episodes"] if ep.get("success", False)
        )

        self.training_sessions.append(session)
        self.current_session = None

        logger.info(
            f"\nBatch training completed: "
            f"{session['successful_episodes']}/{session['total_episodes']} "
            f"successful"
        )

        return session

    async def train_model(
        self,
        min_samples: int = 100,
        model_type: str = "supervised",
    ) -> dict[str, Any]:
        """訓練模型

        Args:
            min_samples: 最小樣本數量
            model_type: 模型類型（supervised 或 reinforcement）

        Returns:
            訓練結果
        """
        # 獲取高質量樣本
        samples = self.experience_manager.get_high_quality_samples(
            min_quality=0.6,
            limit=10000,
        )

        if len(samples) < min_samples:
            logger.warning(
                f"Insufficient samples for training: " f"{len(samples)} < {min_samples}"
            )
            return {
                "success": False,
                "error": "Insufficient samples",
                "samples_available": len(samples),
                "samples_required": min_samples,
            }

        logger.info(f"Training model with {len(samples)} samples")

        # 訓練模型
        if model_type == "supervised":
            result = await self.model_trainer.train_supervised(samples)
        elif model_type == "reinforcement":
            result = await self.model_trainer.train_reinforcement(samples)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return result

    def get_training_statistics(self) -> dict[str, Any]:
        """獲取訓練統計信息

        Returns:
            統計信息字典
        """
        total_episodes = sum(
            len(session["episodes"]) for session in self.training_sessions
        )

        successful_episodes = sum(
            sum(1 for ep in session["episodes"] if ep.get("success", False))
            for session in self.training_sessions
        )

        return {
            "total_sessions": len(self.training_sessions),
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": (
                successful_episodes / total_episodes if total_episodes > 0 else 0.0
            ),
            "experience_samples": self.experience_manager.get_statistics(),
            "knowledge_base": self.rag_engine.get_statistics(),
        }

    def save_session(self, session_id: str | None = None) -> None:
        """保存訓練會話

        Args:
            session_id: 會話 ID（None 表示保存所有）
        """
        if session_id is None:
            # 保存所有會話
            for session in self.training_sessions:
                self._save_single_session(session)
        else:
            # 保存指定會話
            session = next(
                (s for s in self.training_sessions if s["session_id"] == session_id),
                None,
            )
            if session:
                self._save_single_session(session)

    def _save_single_session(self, session: dict[str, Any]) -> None:
        """保存單個會話

        Args:
            session: 會話數據
        """
        import json

        session_file = self.data_directory / f"{session['session_id']}.json"

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        logger.info(f"Session saved: {session_file}")
