"""
Training Orchestrator - 訓練編排器

整合 RAG、場景管理、模型訓練，實現完整的自動化訓練流程
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

try:
    from ..execution.plan_executor import PlanExecutor
    from ..learning.experience_manager import ExperienceManager
    from ..learning.model_trainer import ModelTrainer
    from ..rag import RAGEngine
    from .scenario_manager import ScenarioManager
except ImportError:
    from services.core.aiva_core.execution.plan_executor import PlanExecutor
    from services.core.aiva_core.learning.experience_manager import ExperienceManager
    from services.core.aiva_core.learning.model_trainer import ModelTrainer
    from services.core.aiva_core.rag import RAGEngine
    from services.core.aiva_core.training.scenario_manager import ScenarioManager

from services.aiva_common.schemas import (
    AttackPlan,
    ExperienceSample,
    StandardScenario,
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
        scenario_manager: ScenarioManager | None = None,
        rag_engine: RAGEngine | None = None,
        plan_executor: PlanExecutor | None = None,
        experience_manager: ExperienceManager | None = None,
        model_trainer: ModelTrainer | None = None,
        data_directory: Path | None = None,
        auto_initialize: bool = True,
    ) -> None:
        """初始化訓練編排器

        Args:
            scenario_manager: 場景管理器（None 時自動創建）
            rag_engine: RAG 引擎（None 時自動創建）
            plan_executor: 計畫執行器（None 時自動創建）
            experience_manager: 經驗管理器（None 時自動創建）
            model_trainer: 模型訓練器（None 時自動創建）
            data_directory: 數據目錄
            auto_initialize: 是否自動初始化缺失的組件
        """
        # 自動初始化組件
        if auto_initialize:
            self.scenario_manager = scenario_manager or self._create_default_scenario_manager()
            self.rag_engine = rag_engine or self._create_default_rag_engine()
            self.plan_executor = plan_executor or self._create_default_plan_executor()
            self.experience_manager = experience_manager or self._create_default_experience_manager()
            self.model_trainer = model_trainer or self._create_default_model_trainer()
        else:
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

        logger.info(
            f"TrainingOrchestrator initialized (auto_initialize={auto_initialize})"
        )
    
    def _create_default_scenario_manager(self) -> ScenarioManager:
        """創建默認場景管理器"""
        logger.debug("Creating default ScenarioManager")
        return ScenarioManager()
    
    def _create_default_rag_engine(self) -> RAGEngine:
        """創建默認 RAG 引擎"""
        from ..rag import KnowledgeBase, VectorStore
        logger.debug("Creating default RAGEngine")
        # 創建簡單的內存知識庫
        vector_store = VectorStore()
        knowledge_base = KnowledgeBase(vector_store=vector_store)
        return RAGEngine(knowledge_base=knowledge_base)
    
    def _create_default_plan_executor(self) -> PlanExecutor:
        """創建默認計劃執行器"""
        logger.debug("Creating default PlanExecutor")
        return PlanExecutor()
    
    def _create_default_experience_manager(self) -> ExperienceManager:
        """創建默認經驗管理器"""
        logger.debug("Creating default ExperienceManager")
        return ExperienceManager()
    
    def _create_default_model_trainer(self) -> ModelTrainer:
        """創建默認模型訓練器"""
        logger.debug("Creating default ModelTrainer")
        return ModelTrainer()

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

            # 使用 AI 模型生成計畫（基於 MITRE ATT&CK 和 RAG 增強）
            attack_plan = await self._generate_ai_attack_plan(
                scenario=scenario,
                rag_context=rag_context
            )

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
        scenario: StandardScenario,
        attack_plan: AttackPlan,
        execution_result: dict[str, Any],
    ) -> list[ExperienceSample]:
        """從執行結果提取經驗樣本

        遵循強化學習最佳實踐：
        1. 狀態-動作-獎勵提取
        2. 經驗重播記憶體格式
        3. 品質評分與置信度計算
        4. 多樣性標籤用於平衡學習

        Args:
            scenario: 場景
            attack_plan: 攻擊計畫
            execution_result: 執行結果

        Returns:
            經驗樣本列表
        """
        samples: list[ExperienceSample] = []
        
        # 檢查執行結果是否包含trace_records
        trace_records = execution_result.get("trace_records", [])
        if not trace_records:
            logger.warning("No trace records found in execution result")
            return samples

        # 使用 PlanComparator 分析執行計畫與預期結果的差異
        plan_comparison = None
        if hasattr(self, 'plan_comparator') and self.plan_comparator:
            try:
                plan_comparison = await self.plan_comparator.compare_plans(
                    expected_plan=scenario.expected_plan,
                    actual_plan=attack_plan,
                    execution_traces=trace_records
                )
            except Exception as e:
                logger.warning(f"Plan comparison failed: {e}")

        # 計算整體成功率和獎勵基準
        overall_success = execution_result.get("overall_success", False)
        total_steps = len(trace_records)
        successful_steps = sum(1 for trace in trace_records if trace.get("success", False))
        success_rate = successful_steps / total_steps if total_steps > 0 else 0.0

        # 為每個執行步驟創建經驗樣本
        for step_index, trace_record in enumerate(trace_records):
            try:
                # 狀態提取 (執行前)
                state_before = {
                    "target_url": trace_record.get("target_url", ""),
                    "vulnerability_type": scenario.vulnerability_type.value,
                    "step_index": step_index,
                    "previous_success_rate": (
                        sum(1 for t in trace_records[:step_index] if t.get("success", False)) / max(1, step_index)
                        if step_index > 0 else 0.0
                    ),
                    "scenario_context": {
                        "difficulty_level": scenario.difficulty_level,
                        "tags": scenario.tags,
                        "expected_outcome": scenario.success_criteria.get("main_indicator", "unknown")
                    }
                }

                # 動作提取
                action_taken = {
                    "tool": trace_record.get("tool_used", "unknown"),
                    "method": trace_record.get("method", "GET"),
                    "parameters": trace_record.get("parameters", {}),
                    "payload": trace_record.get("payload", ""),
                    "headers": trace_record.get("headers", {}),
                    "step_type": trace_record.get("step_type", "unknown")
                }

                # 狀態提取 (執行後)
                state_after = {
                    "response_code": trace_record.get("response_code", 0),
                    "response_time": trace_record.get("response_time", 0),
                    "response_size": trace_record.get("response_size", 0),
                    "success": trace_record.get("success", False),
                    "findings_count": len(trace_record.get("findings", [])),
                    "error_message": trace_record.get("error_message", ""),
                    "final_state": trace_record.get("final_state", {})
                }

                # 獎勵計算 (基於多個因素)
                step_reward = self._calculate_step_reward(
                    trace_record=trace_record,
                    step_index=step_index,
                    total_steps=total_steps,
                    overall_success=overall_success,
                    plan_comparison=plan_comparison
                )

                # 獎勵分解 (用於分析和調試)
                reward_breakdown = {
                    "completion": min(1.0, (step_index + 1) / total_steps),  # 完成度獎勵
                    "success": 1.0 if trace_record.get("success", False) else -0.5,  # 成功/失敗獎勵
                    "sequence": 0.1 if step_index == 0 or trace_records[step_index-1].get("success", False) else -0.1,  # 序列連貫性
                    "goal": 1.0 if overall_success and step_index == total_steps - 1 else 0.0,  # 目標達成獎勵
                    "efficiency": max(0.0, 1.0 - (trace_record.get("response_time", 1000) / 5000.0)),  # 效率獎勵
                    "quality": self._assess_finding_quality(trace_record.get("findings", []))  # 發現品質獎勵
                }

                # 品質評分計算
                quality_score = self._calculate_quality_score(
                    trace_record=trace_record,
                    reward_breakdown=reward_breakdown,
                    step_index=step_index,
                    total_steps=total_steps
                )

                # 學習標籤生成 (用於平衡學習資料集)
                learning_tags = self._generate_learning_tags(
                    trace_record=trace_record,
                    scenario=scenario,
                    step_index=step_index,
                    success_rate=success_rate
                )

                # 創建經驗樣本
                sample = ExperienceSample(
                    sample_id=f"{execution_result.get('session_id', 'unknown')}_{attack_plan.plan_id}_{step_index}",
                    session_id=execution_result.get("session_id", "unknown"),
                    plan_id=attack_plan.plan_id,
                    state_before=state_before,
                    action_taken=action_taken,
                    state_after=state_after,
                    reward=step_reward,
                    reward_breakdown=reward_breakdown,
                    context={
                        "scenario_id": scenario.scenario_id,
                        "target_info": execution_result.get("target_info", {}),
                        "execution_metadata": execution_result.get("metadata", {})
                    },
                    target_info={
                        "base_url": trace_record.get("target_url", ""),
                        "vulnerability_type": scenario.vulnerability_type.value,
                        "difficulty": scenario.difficulty_level
                    },
                    timestamp=datetime.now(UTC),
                    duration_ms=trace_record.get("response_time", 0),
                    quality_score=quality_score,
                    is_positive=step_reward > 0.0,
                    confidence=min(1.0, quality_score + 0.2),  # 置信度基於品質分數
                    learning_tags=learning_tags,
                    difficulty_level=scenario.difficulty_level
                )

                samples.append(sample)
                logger.debug(f"Created experience sample {sample.sample_id} with reward {step_reward:.3f}")

            except Exception as e:
                logger.error(f"Failed to create experience sample for step {step_index}: {e}")
                continue

        logger.info(f"Extracted {len(samples)} experience samples from {total_steps} execution steps")
        return samples

    def _calculate_step_reward(
        self,
        trace_record: dict[str, Any],
        step_index: int,
        total_steps: int,
        overall_success: bool,
        plan_comparison: dict[str, Any] | None = None,
    ) -> float:
        """計算單步獎勵值

        基於強化學習最佳實踐的獎勵函數設計：
        - 即時獎勵：基於當前步驟的直接結果
        - 延遲獎勵：考慮整體目標達成
        - 塑造獎勵：引導學習朝向期望行為

        Args:
            trace_record: 執行記錄
            step_index: 步驟索引
            total_steps: 總步驟數
            overall_success: 整體是否成功
            plan_comparison: 計劃比較結果

        Returns:
            獎勵值 (-2.0 到 2.0)
        """
        base_reward = 0.0

        # 1. 基本成功獎勵
        if trace_record.get("success", False):
            base_reward += 1.0
        else:
            base_reward -= 0.3

        # 2. 回應時間效率獎勵 (快速回應獲得額外獎勵)
        response_time = trace_record.get("response_time", 1000)
        if response_time < 500:  # 快速回應
            base_reward += 0.2
        elif response_time > 10000:  # 過慢回應
            base_reward -= 0.2

        # 3. 發現品質獎勵
        findings = trace_record.get("findings", [])
        if findings:
            # 根據發現的嚴重程度給予獎勵
            severity_weights = {"critical": 0.5, "high": 0.3, "medium": 0.2, "low": 0.1}
            finding_reward = sum(
                severity_weights.get(finding.get("severity", "low").lower(), 0.1)
                for finding in findings
            )
            base_reward += min(0.5, finding_reward)  # 限制最大發現獎勵

        # 4. 進度獎勵 (鼓勵完成更多步驟)
        progress_reward = 0.1 * (step_index + 1) / total_steps
        base_reward += progress_reward

        # 5. 整體成功獎勵 (只有在最後一步且整體成功時給予)
        if step_index == total_steps - 1 and overall_success:
            base_reward += 0.8

        # 6. 計劃偏差懲罰 (如果有計劃比較結果)
        if plan_comparison:
            deviation_score = plan_comparison.get("deviation_score", 0.0)
            if deviation_score > 0.5:  # 高偏差
                base_reward -= 0.2

        # 7. 錯誤懲罰
        if trace_record.get("error_message"):
            base_reward -= 0.3

        # 限制獎勵範圍
        return max(-2.0, min(2.0, base_reward))

    def _assess_finding_quality(self, findings: list[dict[str, Any]]) -> float:
        """評估發現的品質

        Args:
            findings: 發現列表

        Returns:
            品質分數 (0.0 到 1.0)
        """
        if not findings:
            return 0.0

        quality_score = 0.0
        total_weight = 0.0

        for finding in findings:
            # 嚴重程度權重
            severity = finding.get("severity", "low").lower()
            severity_weight = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}.get(severity, 0.2)

            # 置信度權重
            confidence = finding.get("confidence", "medium").lower()
            confidence_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(confidence, 0.5)

            # 證據完整性
            evidence_weight = 1.0 if finding.get("evidence") else 0.5

            finding_quality = severity_weight * confidence_weight * evidence_weight
            quality_score += finding_quality
            total_weight += 1.0

        return min(1.0, quality_score / total_weight if total_weight > 0 else 0.0)

    def _calculate_quality_score(
        self,
        trace_record: dict[str, Any],
        reward_breakdown: dict[str, float],
        step_index: int,
        total_steps: int,
    ) -> float:
        """計算經驗樣本品質分數

        品質分數用於優先選擇高質量樣本進行訓練

        Args:
            trace_record: 執行記錄
            reward_breakdown: 獎勵分解
            step_index: 步驟索引
            total_steps: 總步驟數

        Returns:
            品質分數 (0.0 到 1.0)
        """
        quality_factors = []

        # 1. 獎勵一致性 (正獎勵表示高質量)
        reward_quality = max(0.0, sum(reward_breakdown.values()) / len(reward_breakdown))
        quality_factors.append(min(1.0, reward_quality))

        # 2. 數據完整性 (更完整的數據更有價值)
        required_fields = ["tool_used", "method", "response_code", "response_time"]
        completeness = sum(1 for field in required_fields if trace_record.get(field)) / len(required_fields)
        quality_factors.append(completeness)

        # 3. 響應有效性 (有效HTTP響應)
        response_code = trace_record.get("response_code", 0)
        response_validity = 1.0 if 200 <= response_code < 600 else 0.3
        quality_factors.append(response_validity)

        # 4. 發現品質
        finding_quality = self._assess_finding_quality(trace_record.get("findings", []))
        quality_factors.append(finding_quality)

        # 5. 時序位置價值 (關鍵步驟更有價值)
        if step_index == 0:  # 首步驟
            position_value = 0.9
        elif step_index == total_steps - 1:  # 最後步驟
            position_value = 0.9
        else:  # 中間步驟
            position_value = 0.7
        quality_factors.append(position_value)

        # 加權平均計算最終品質分數
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # 各因子權重
        quality_score = sum(factor * weight for factor, weight in zip(quality_factors, weights))

        return min(1.0, max(0.0, quality_score))

    def _generate_learning_tags(
        self,
        trace_record: dict[str, Any],
        scenario: StandardScenario,
        step_index: int,
        success_rate: float,
    ) -> list[str]:
        """生成學習標籤

        標籤用於平衡學習資料集和提供語義信息

        Args:
            trace_record: 執行記錄
            scenario: 場景
            step_index: 步驟索引
            success_rate: 成功率

        Returns:
            學習標籤列表
        """
        tags = []

        # 1. 漏洞類型標籤
        tags.append(f"vuln_{scenario.vulnerability_type.value}")

        # 2. 工具標籤
        tool_used = trace_record.get("tool_used", "unknown")
        tags.append(f"tool_{tool_used}")

        # 3. 成功/失敗標籤
        if trace_record.get("success", False):
            tags.append("success")
        else:
            tags.append("failure")

        # 4. 響應碼類別標籤
        response_code = trace_record.get("response_code", 0)
        if 200 <= response_code < 300:
            tags.append("http_2xx")
        elif 300 <= response_code < 400:
            tags.append("http_3xx")
        elif 400 <= response_code < 500:
            tags.append("http_4xx")
        elif 500 <= response_code < 600:
            tags.append("http_5xx")

        # 5. 時序標籤
        if step_index == 0:
            tags.append("initial_step")
        elif step_index < 3:
            tags.append("early_step")
        else:
            tags.append("late_step")

        # 6. 難度標籤
        tags.append(f"difficulty_{scenario.difficulty_level}")

        # 7. 成功率標籤
        if success_rate > 0.8:
            tags.append("high_success_rate")
        elif success_rate > 0.5:
            tags.append("medium_success_rate")
        else:
            tags.append("low_success_rate")

        # 8. 發現相關標籤
        findings = trace_record.get("findings", [])
        if findings:
            tags.append("has_findings")
            # 添加嚴重程度標籤
            severities = {finding.get("severity", "low").lower() for finding in findings}
            for severity in severities:
                tags.append(f"severity_{severity}")
        else:
            tags.append("no_findings")

        # 9. 性能標籤
        response_time = trace_record.get("response_time", 1000)
        if response_time < 1000:
            tags.append("fast_response")
        elif response_time > 5000:
            tags.append("slow_response")

        return tags

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
            scenarios = await self.scenario_manager.list_scenarios()
            scenario_ids = [s["id"] for s in scenarios]

        logger.info(
            f"Starting batch training: "
            f"{len(scenario_ids)} scenarios, "
            f"{episodes_per_scenario} episodes each"
        )

        # 訓練會話
        session: dict[str, Any] = {
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
    
    async def _generate_ai_attack_plan(
        self, 
        scenario,
        rag_context: Dict[str, Any]
    ):
        """
        使用 AI 模型生成攻擊計劃
        
        基於 MITRE ATT&CK 框架和 RAG 增強上下文，
        生成針對性的攻擊計劃
        
        Args:
            scenario: 訓練場景
            rag_context: RAG 增強上下文
            
        Returns:
            生成的攻擊計劃
        """
        import time
        
        try:
            # 1. 初始化 AI 引擎（如果尚未初始化）
            if not hasattr(self, '_ai_engine'):
                from ..ai_engine import BioNeuronRAGAgent, AIModelManager
                
                self._ai_engine = BioNeuronRAGAgent(
                    codebase_path=str(Path.cwd())
                )
                
                self._ai_model_manager = AIModelManager()
                
                logger.info("AI 引擎已初始化用於攻擊計劃生成")
            
            # 2. 分析目標和上下文
            target_analysis = await self._analyze_target_context(scenario, rag_context)
            
            # 3. 基於 MITRE ATT&CK 選擇戰術技術
            tactics_and_techniques = await self._select_attack_tactics(
                target_analysis, 
                scenario.objective
            )
            
            # 4. 生成具體的攻擊計劃
            attack_plan = await self._build_attack_plan(
                tactics_and_techniques,
                target_analysis,
                rag_context
            )
            
            logger.info(f"AI 生成攻擊計劃: {len(attack_plan.get('steps', []))} 個步驟")
            
            return attack_plan
            
        except Exception as e:
            logger.error(f"AI 攻擊計劃生成失敗: {e}")
            
            # 降級到預定義計劃
            logger.info("降級使用預定義攻擊計劃")
            return scenario.attack_plan
    
    async def _analyze_target_context(
        self, 
        scenario, 
        rag_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析目標上下文"""
        
        target = scenario.target
        objective = scenario.objective
        
        # 提取目標特徵
        target_features = {
            "target_type": getattr(target, 'type', 'web_application'),
            "target_url": getattr(target, 'url', 'http://localhost:3000'),
            "target_technologies": getattr(target, 'technologies', ['web', 'javascript']),
            "objective": objective,
            "similar_techniques": rag_context.get('similar_techniques', []),
            "successful_experiences": rag_context.get('successful_experiences', []),
        }
        
        logger.debug(f"目標分析完成: {target_features}")
        
        return target_features
    
    async def _select_attack_tactics(
        self, 
        target_analysis: Dict[str, Any], 
        objective: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        基於 MITRE ATT&CK 框架選擇攻擊戰術和技術
        
        根據目標分析結果選擇最適合的 ATT&CK 技術組合
        """
        
        # MITRE ATT&CK 戰術映射（基於我們已實現的漏洞利用能力）
        tactics_mapping = {
            "reconnaissance": {
                "T1595": {  # Active Scanning
                    "name": "主動掃描",
                    "techniques": ["port_scan", "service_discovery", "web_fingerprinting"],
                    "priority": 1
                },
                "T1592": {  # Gather Victim Host Information
                    "name": "收集主機信息",
                    "techniques": ["technology_stack_detection", "endpoint_discovery"],
                    "priority": 2
                }
            },
            "initial_access": {
                "T1190": {  # Exploit Public-Facing Application
                    "name": "利用面向公眾的應用程序",
                    "techniques": ["sql_injection", "xss", "idor"],
                    "priority": 1
                },
                "T1078": {  # Valid Accounts
                    "name": "有效帳戶",
                    "techniques": ["auth_bypass", "weak_credentials"],
                    "priority": 2
                }
            },
            "execution": {
                "T1203": {  # Exploitation for Client Execution
                    "name": "客戶端執行利用",
                    "techniques": ["xss_payload_execution", "csrf_execution"],
                    "priority": 1
                }
            },
            "persistence": {
                "T1078": {  # Valid Accounts
                    "name": "持久化有效賬戶",
                    "techniques": ["jwt_manipulation", "session_persistence"],
                    "priority": 1
                }
            },
            "credential_access": {
                "T1110": {  # Brute Force
                    "name": "暴力破解",
                    "techniques": ["password_brute_force", "jwt_weak_secret"],
                    "priority": 2
                },
                "T1212": {  # Exploitation for Credential Access
                    "name": "憑證訪問利用",
                    "techniques": ["sql_injection_credentials", "idor_user_data"],
                    "priority": 1
                }
            }
        }
        
        # 根據目標類型和目標選擇相關戰術
        selected_tactics = {}
        
        target_type = target_analysis.get("target_type", "web_application")
        
        if target_type == "web_application":
            # Web 應用程序攻擊鏈
            selected_tactics["reconnaissance"] = [
                tactics_mapping["reconnaissance"]["T1595"],
                tactics_mapping["reconnaissance"]["T1592"]
            ]
            
            selected_tactics["initial_access"] = [
                tactics_mapping["initial_access"]["T1190"]
            ]
            
            selected_tactics["credential_access"] = [
                tactics_mapping["credential_access"]["T1212"]
            ]
            
            # 根據歷史成功經驗調整優先級
            successful_techniques = [
                exp.get('technique', '') 
                for exp in target_analysis.get('successful_experiences', [])
            ]
            
            if 'sql_injection' in successful_techniques:
                selected_tactics["initial_access"][0]["priority"] = 0  # 最高優先級
            
            if 'idor' in successful_techniques:
                selected_tactics["credential_access"][0]["priority"] = 0
        
        logger.info(f"選擇的 ATT&CK 戰術: {list(selected_tactics.keys())}")
        
        return selected_tactics
    
    async def _build_attack_plan(
        self,
        tactics_and_techniques: Dict[str, List[Dict[str, Any]]],
        target_analysis: Dict[str, Any],
        rag_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        構建具體的攻擊計劃
        
        將 ATT&CK 戰術技術翻譯成可執行的攻擊步驟
        """
        import time
        
        attack_steps = []
        step_id = 1
        
        # 按照 kill chain 順序執行戰術
        kill_chain_order = [
            "reconnaissance", 
            "initial_access", 
            "execution", 
            "persistence", 
            "credential_access"
        ]
        
        for tactic in kill_chain_order:
            if tactic not in tactics_and_techniques:
                continue
                
            techniques = tactics_and_techniques[tactic]
            
            # 按優先級排序技術
            techniques.sort(key=lambda x: x.get("priority", 999))
            
            for technique in techniques:
                # 將技術轉換為具體步驟
                steps = await self._technique_to_steps(
                    technique, 
                    target_analysis, 
                    step_id
                )
                
                attack_steps.extend(steps)
                step_id += len(steps)
        
        # 構建完整的攻擊計劃
        attack_plan = {
            "plan_id": f"ai_generated_{int(time.time())}",
            "name": f"AI Generated Plan - {target_analysis.get('objective', 'Security Test')}",
            "description": "基於 MITRE ATT&CK 框架的 AI 生成攻擊計劃",
            "target": target_analysis.get("target_url", "http://localhost:3000"),
            "tactics_used": list(tactics_and_techniques.keys()),
            "steps": attack_steps,
            "metadata": {
                "generation_method": "AI + MITRE ATT&CK",
                "rag_enhanced": True,
                "techniques_count": len(attack_steps),
                "generation_timestamp": int(time.time())
            }
        }
        
        logger.info(f"攻擊計劃構建完成: {len(attack_steps)} 個步驟")
        
        return attack_plan
    
    async def _technique_to_steps(
        self,
        technique: Dict[str, Any],
        target_analysis: Dict[str, Any],
        start_step_id: int
    ) -> List[Dict[str, Any]]:
        """
        將 ATT&CK 技術轉換為具體的執行步驟
        """
        
        steps = []
        technique_methods = technique.get("techniques", [])
        target_url = target_analysis.get("target_url", "http://localhost:3000")
        
        for i, method in enumerate(technique_methods):
            step = {
                "step_id": start_step_id + i,
                "name": f"{technique['name']} - {method}",
                "technique_id": method,
                "type": self._map_method_to_type(method),
                "target": {
                    "url": target_url,
                    "method": method
                },
                "payload": self._generate_payload_for_method(method),
                "expected_outcome": self._get_expected_outcome(method),
                "success_criteria": self._get_success_criteria(method),
                "timeout": 30,
                "retry_count": 2
            }
            
            steps.append(step)
        
        return steps
    
    def _map_method_to_type(self, method: str) -> str:
        """將攻擊方法映射到類型"""
        
        method_type_mapping = {
            "port_scan": "reconnaissance",
            "service_discovery": "reconnaissance", 
            "web_fingerprinting": "reconnaissance",
            "technology_stack_detection": "reconnaissance",
            "endpoint_discovery": "reconnaissance",
            "sql_injection": "exploitation",
            "xss": "exploitation",
            "idor": "exploitation",
            "auth_bypass": "exploitation",
            "weak_credentials": "exploitation",
            "jwt_manipulation": "exploitation",
            "csrf_execution": "exploitation",
        }
        
        return method_type_mapping.get(method, "exploitation")
    
    def _generate_payload_for_method(self, method: str) -> Dict[str, Any]:
        """為攻擊方法生成載荷"""
        
        payload_templates = {
            "sql_injection": {
                "payloads": ["' OR 1=1--", "admin'--", "' UNION SELECT null--"],
                "endpoints": ["/rest/user/login", "/rest/products/search"]
            },
            "xss": {
                "payloads": ["<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>"],
                "endpoints": ["/rest/products/search", "/api/Feedbacks"]
            },
            "idor": {
                "payloads": ["user_id_1", "user_id_2", "admin_data"],
                "endpoints": ["/api/Users", "/rest/user/whoami", "/api/Feedbacks"]
            },
            "auth_bypass": {
                "payloads": [("admin", "admin"), ("admin", "password"), ("root", "root")],
                "endpoints": ["/rest/user/login", "/administration"]
            },
            "jwt_manipulation": {
                "payloads": ["none_algorithm", "weak_secret", "algorithm_confusion"],
                "endpoints": ["/rest/user/whoami", "/rest/user/authentication-details"]
            }
        }
        
        return payload_templates.get(method, {"payloads": ["test"], "endpoints": ["/"]})
    
    def _get_expected_outcome(self, method: str) -> str:
        """獲取攻擊方法的預期結果"""
        
        outcomes = {
            "sql_injection": "成功繞過登入驗證或洩露數據庫信息",
            "xss": "成功執行跨站腳本或注入惡意內容", 
            "idor": "成功訪問未授權數據或枚舉用戶信息",
            "auth_bypass": "成功繞過認證機制或訪問管理功能",
            "jwt_manipulation": "成功操縱 JWT token 或提升權限"
        }
        
        return outcomes.get(method, f"成功執行 {method} 攻擊")
    
    def _get_success_criteria(self, method: str) -> List[str]:
        """獲取攻擊方法的成功標準"""
        
        criteria = {
            "sql_injection": [
                "HTTP 200 響應包含認證 token",
                "數據庫錯誤信息洩露",
                "SQL 查詢結果返回"
            ],
            "xss": [
                "惡意腳本成功反射",
                "JavaScript 代碼執行",
                "內容成功注入頁面"
            ],
            "idor": [
                "訪問其他用戶數據",
                "枚舉用戶 ID 成功",
                "未授權數據洩露"
            ],
            "auth_bypass": [
                "成功登入管理帳戶",
                "訪問受保護資源",
                "繞過認證檢查"
            ]
        }
        
        return criteria.get(method, [f"{method} 攻擊成功執行"])
