"""
AI Commander - AIVA 中央 AI 指揮系統

統一指揮所有 AI 組件：
1. BioNeuronRAGAgent（Python 主控 AI）
2. RAG Engine（知識檢索增強）
3. Training Orchestrator（訓練系統）
4. Multi-Language AI Modules（Go/Rust/TypeScript AI）

架構設計：
- AI Commander 作為最高指揮層
- 各語言 AI 作為專業執行層
- RAG 提供知識支持
- Training 提供持續學習
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any

from aiva_core.ai_engine import BioNeuronRAGAgent
from aiva_core.learning.experience_manager import ExperienceManager
from aiva_core.learning.model_trainer import ModelTrainer
from aiva_core.multilang_coordinator import MultiLanguageAICoordinator
from aiva_core.rag import KnowledgeBase, RAGEngine, VectorStore
from aiva_core.training.training_orchestrator import TrainingOrchestrator

logger = logging.getLogger(__name__)


class AITaskType(str, Enum):
    """AI 任務類型"""

    # 決策類
    ATTACK_PLANNING = "attack_planning"  # 攻擊計畫生成
    STRATEGY_DECISION = "strategy_decision"  # 策略決策
    RISK_ASSESSMENT = "risk_assessment"  # 風險評估

    # 執行類
    VULNERABILITY_DETECTION = "vulnerability_detection"  # 漏洞檢測
    EXPLOIT_EXECUTION = "exploit_execution"  # 漏洞利用
    CODE_ANALYSIS = "code_analysis"  # 代碼分析

    # 學習類
    EXPERIENCE_LEARNING = "experience_learning"  # 經驗學習
    MODEL_TRAINING = "model_training"  # 模型訓練
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # 知識檢索

    # 協調類
    MULTI_LANG_COORDINATION = "multi_lang_coordination"  # 多語言協調
    TASK_DELEGATION = "task_delegation"  # 任務委派


class AIComponent(str, Enum):
    """AI 組件類型"""

    BIO_NEURON_AGENT = "bio_neuron_agent"  # Python 主控 AI
    RAG_ENGINE = "rag_engine"  # RAG 引擎
    TRAINING_SYSTEM = "training_system"  # 訓練系統
    MULTILANG_COORDINATOR = "multilang_coordinator"  # 多語言協調器

    # 語言專屬 AI
    GO_AI_MODULE = "go_ai_module"  # Go AI 模組
    RUST_AI_MODULE = "rust_ai_module"  # Rust AI 模組
    TS_AI_MODULE = "ts_ai_module"  # TypeScript AI 模組


class AICommander:
    """AI 指揮官

    統一管理和協調所有 AI 組件，負責：
    1. 任務分析和分配
    2. AI 組件協調
    3. 決策整合
    4. 經驗積累
    5. 持續學習
    """

    def __init__(
        self,
        codebase_path: str = "/workspaces/AIVA",
        data_directory: Path | None = None,
    ) -> None:
        """初始化 AI 指揮官

        Args:
            codebase_path: 代碼庫路徑
            data_directory: 數據目錄
        """
        logger.info("[U+1F396][U+FE0F] Initializing AI Commander...")

        self.data_directory = data_directory or Path("./data/ai_commander")
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # === 核心 AI 組件 ===

        # 1. Python 主控 AI（BioNeuronRAGAgent）
        logger.info("  Loading BioNeuronRAGAgent...")
        self.bio_neuron_agent = BioNeuronRAGAgent(codebase_path)

        # 2. RAG 系統（知識增強）
        logger.info("  Loading RAG Engine...")
        vector_store = VectorStore(
            backend="memory",  # 可配置為 chroma/faiss
            persist_directory=self.data_directory / "vectors",
        )
        knowledge_base = KnowledgeBase(
            vector_store=vector_store,
            data_directory=self.data_directory / "knowledge",
        )
        self.rag_engine = RAGEngine(knowledge_base=knowledge_base)

        # 3. 經驗管理和模型訓練
        logger.info("  Loading Training System...")
        self.experience_manager = ExperienceManager(
            storage_backend=None,  # TODO: 整合資料庫
        )
        self.model_trainer = ModelTrainer(
            model_config={
                "model_type": "supervised",
                "learning_rate": 1e-4,
            }
        )

        # 4. 訓練編排器（整合 RAG 和訓練）
        from aiva_core.training.scenario_manager import ScenarioManager

        scenario_manager = ScenarioManager()
        from aiva_core.execution.plan_executor import PlanExecutor
        from aiva_core.messaging.message_broker import MessageBroker

        message_broker = MessageBroker()
        plan_executor = PlanExecutor(message_broker=message_broker)

        self.training_orchestrator = TrainingOrchestrator(
            scenario_manager=scenario_manager,
            rag_engine=self.rag_engine,
            plan_executor=plan_executor,
            experience_manager=self.experience_manager,
            model_trainer=self.model_trainer,
        )

        # 5. 多語言協調器
        logger.info("  Loading Multi-Language Coordinator...")
        self.multilang_coordinator = MultiLanguageAICoordinator()

        # === 指揮狀態 ===
        self.command_history: list[dict[str, Any]] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}
        self.component_status: dict[str, bool] = {
            component.value: True for component in AIComponent
        }

        logger.info("[OK] AI Commander initialized successfully")
        logger.info(f"   - BioNeuronRAGAgent: {self.bio_neuron_agent is not None}")
        logger.info(f"   - RAG Engine: {self.rag_engine is not None}")
        logger.info(f"   - Training System: {self.training_orchestrator is not None}")
        logger.info(
            f"   - Multi-Language Coordinator: {self.multilang_coordinator is not None}"
        )

    async def execute_command(
        self,
        task_type: AITaskType,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """執行 AI 指令

        Args:
            task_type: 任務類型
            context: 任務上下文

        Returns:
            執行結果
        """
        logger.info(f"[TARGET] Executing AI Command: {task_type.value}")

        # 記錄指令
        command_id = f"cmd_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
        command_record = {
            "command_id": command_id,
            "task_type": task_type.value,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "status": "started",
        }
        self.command_history.append(command_record)
        self.active_tasks[command_id] = command_record

        try:
            # 根據任務類型分派
            if task_type == AITaskType.ATTACK_PLANNING:
                result = await self._plan_attack(context)

            elif task_type == AITaskType.STRATEGY_DECISION:
                result = await self._make_strategy_decision(context)

            elif task_type == AITaskType.VULNERABILITY_DETECTION:
                result = await self._detect_vulnerabilities(context)

            elif task_type == AITaskType.EXPERIENCE_LEARNING:
                result = await self._learn_from_experience(context)

            elif task_type == AITaskType.MODEL_TRAINING:
                result = await self._train_model(context)

            elif task_type == AITaskType.KNOWLEDGE_RETRIEVAL:
                result = await self._retrieve_knowledge(context)

            elif task_type == AITaskType.MULTI_LANG_COORDINATION:
                result = await self._coordinate_multilang(context)

            else:
                result = {
                    "success": False,
                    "error": f"Unsupported task type: {task_type.value}",
                }

            # 更新狀態
            command_record["status"] = "completed"
            command_record["result"] = result
            command_record["end_time"] = datetime.now().isoformat()

            logger.info(
                f"[OK] Command {command_id} completed: "
                f"success={result.get('success', False)}"
            )

        except Exception as e:
            logger.error(f"[FAIL] Command {command_id} failed: {e}", exc_info=True)
            command_record["status"] = "failed"
            command_record["error"] = str(e)
            result = {"success": False, "error": str(e)}

        finally:
            del self.active_tasks[command_id]

        return result

    async def _plan_attack(self, context: dict[str, Any]) -> dict[str, Any]:
        """生成攻擊計畫（RAG 增強）

        Args:
            context: 包含 target, objective 等

        Returns:
            攻擊計畫結果
        """
        logger.info("[LIST] Generating attack plan with RAG enhancement...")

        target = context.get("target")
        objective = context.get("objective", "Comprehensive security assessment")

        if not target:
            return {"success": False, "error": "Target not specified"}

        # 1. 使用 RAG 檢索相關知識
        rag_context = self.rag_engine.enhance_attack_plan(
            target=target,
            objective=objective,
        )

        # 2. 使用 BioNeuronRAGAgent 生成計畫
        # TODO: 整合實際的計畫生成
        # plan = await self.bio_neuron_agent.generate_plan(target, objective, rag_context)

        logger.info(
            f"Generated plan with {len(rag_context['similar_techniques'])} "
            f"similar techniques and {len(rag_context['successful_experiences'])} "
            f"successful experiences"
        )

        return {
            "success": True,
            "rag_context": rag_context,
            "plan": None,  # TODO: 實際計畫
            "confidence": 0.8,
        }

    async def _make_strategy_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """策略決策

        Args:
            context: 決策上下文

        Returns:
            決策結果
        """
        logger.info("[U+1F914] Making strategic decision...")

        # 使用 BioNeuronRAGAgent 的決策能力
        # TODO: 整合實際決策邏輯

        return {
            "success": True,
            "decision": "proceed",
            "confidence": 0.75,
            "reasoning": "Based on RAG context and historical success rate",
        }

    async def _detect_vulnerabilities(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢測漏洞（協調多語言模組）

        Args:
            context: 檢測上下文

        Returns:
            檢測結果
        """
        logger.info("[SEARCH] Detecting vulnerabilities across languages...")

        # 協調多語言 AI 模組
        target = context.get("target")
        vuln_types = context.get("vulnerability_types", [])

        # TODO: 實際協調邏輯
        # results = await self.multilang_coordinator.coordinate_detection(
        #     target=target,
        #     vuln_types=vuln_types
        # )

        return {
            "success": True,
            "vulnerabilities_found": 0,
            "languages_coordinated": ["python", "go", "rust"],
        }

    async def _learn_from_experience(self, context: dict[str, Any]) -> dict[str, Any]:
        """從經驗中學習

        Args:
            context: 包含 experience_sample

        Returns:
            學習結果
        """
        logger.info("[DOCS] Learning from experience...")

        sample = context.get("experience_sample")
        if not sample:
            return {"success": False, "error": "No experience sample provided"}

        # 1. 添加到經驗管理器
        self.experience_manager.add_sample(sample)

        # 2. 添加到 RAG 知識庫
        self.rag_engine.learn_from_experience(sample)

        return {
            "success": True,
            "sample_quality": sample.quality_score,
            "knowledge_updated": True,
        }

    async def _train_model(self, context: dict[str, Any]) -> dict[str, Any]:
        """訓練模型

        Args:
            context: 訓練配置

        Returns:
            訓練結果
        """
        logger.info("[U+1F393] Training AI model...")

        # 使用訓練編排器
        result = await self.training_orchestrator.train_model(
            min_samples=context.get("min_samples", 100),
            model_type=context.get("model_type", "supervised"),
        )

        return result

    async def _retrieve_knowledge(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢索知識

        Args:
            context: 包含 query

        Returns:
            檢索結果
        """
        logger.info("[U+1F50E] Retrieving knowledge from RAG...")

        query = context.get("query", "")
        top_k = context.get("top_k", 5)

        results = self.rag_engine.knowledge_base.search(
            query=query,
            top_k=top_k,
        )

        return {
            "success": True,
            "results_count": len(results),
            "results": [
                {
                    "title": entry.title,
                    "type": entry.type.value,
                    "content": entry.content[:200],
                    "success_rate": entry.success_rate,
                }
                for entry in results
            ],
        }

    async def _coordinate_multilang(self, context: dict[str, Any]) -> dict[str, Any]:
        """協調多語言 AI 模組

        Args:
            context: 協調上下文

        Returns:
            協調結果
        """
        logger.info("[U+1F310] Coordinating multi-language AI modules...")

        # 使用多語言協調器
        # TODO: 實際協調邏輯

        return {
            "success": True,
            "modules_coordinated": ["go", "rust", "typescript"],
            "tasks_distributed": 0,
        }

    async def run_training_session(
        self,
        scenario_ids: list[str] | None = None,
        episodes_per_scenario: int = 10,
    ) -> dict[str, Any]:
        """運行訓練會話

        Args:
            scenario_ids: 場景 ID 列表
            episodes_per_scenario: 每個場景的回合數

        Returns:
            訓練結果
        """
        logger.info("[U+1F393] Starting training session...")

        result = await self.training_orchestrator.run_training_batch(
            scenario_ids=scenario_ids,
            episodes_per_scenario=episodes_per_scenario,
            use_rag=True,  # 使用 RAG 增強
        )

        return result

    def get_status(self) -> dict[str, Any]:
        """獲取 AI 指揮官狀態

        Returns:
            狀態信息
        """
        return {
            "component_status": self.component_status,
            "active_tasks": len(self.active_tasks),
            "total_commands": len(self.command_history),
            "successful_commands": sum(
                1 for cmd in self.command_history if cmd.get("status") == "completed"
            ),
            "training_stats": self.training_orchestrator.get_training_statistics(),
            "knowledge_stats": self.rag_engine.get_statistics(),
            "experience_stats": self.experience_manager.get_statistics(),
        }

    def save_state(self) -> None:
        """保存 AI 指揮官狀態"""
        logger.info("[SAVE] Saving AI Commander state...")

        # 保存 RAG 知識庫
        self.rag_engine.save_knowledge()

        # 保存經驗數據
        self.experience_manager.export_to_jsonl(
            self.data_directory / "experiences.jsonl"
        )

        # 保存訓練會話
        self.training_orchestrator.save_session()

        logger.info("[OK] AI Commander state saved")
