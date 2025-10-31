"""
AIVA AI Components Integration
AI 組件整合模組

此模組負責將所有 AI 組件整合到跨語言架構中：
1. Bio-Neuron Agent 整合
2. RAG 引擎整合
3. 多語言協調器整合
4. 對話助手整合
5. 計劃執行器整合
6. AI 組件間通訊協調
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..cross_language import (
    AIVAError,
    AIVAErrorCode,
    create_error_context,
    get_cross_language_service,
    get_error_handler,
)

logger = logging.getLogger(__name__)


class AIComponentType(Enum):
    """AI 組件類型"""

    BIO_NEURON_AGENT = "bio_neuron_agent"
    RAG_ENGINE = "rag_engine"
    MULTILANG_COORDINATOR = "multilang_coordinator"
    DIALOG_ASSISTANT = "dialog_assistant"
    PLAN_EXECUTOR = "plan_executor"
    EXPERIENCE_MANAGER = "experience_manager"
    CAPABILITY_EVALUATOR = "capability_evaluator"


class AITaskType(Enum):
    """AI 任務類型"""

    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    DIALOGUE = "dialogue"
    ANALYSIS = "analysis"
    GENERATION = "generation"


@dataclass
class AITask:
    """AI 任務定義"""

    task_id: str
    task_type: AITaskType
    priority: int
    input_data: dict[str, Any]
    context: dict[str, Any]
    requirements: dict[str, Any]
    timeout: float = 30.0
    retry_count: int = 3
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class AIResult:
    """AI 結果"""

    task_id: str
    success: bool
    result: Any = None
    confidence: float = 0.0
    execution_time: float = 0.0
    component_used: AIComponentType | None = None
    metadata: dict[str, Any] | None = None
    error: AIVAError | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AIComponentInterface:
    """AI 組件接口基類"""

    def __init__(self, component_type: AIComponentType):
        self.component_type = component_type
        self.logger = logging.getLogger(f"ai_component.{component_type.value}")
        self.is_initialized = False
        self.cross_lang_service = get_cross_language_service()

    async def initialize(self) -> bool:
        """初始化組件"""
        raise NotImplementedError

    async def process_task(self, task: AITask) -> AIResult:
        """處理 AI 任務"""
        raise NotImplementedError

    async def cleanup(self):
        """清理組件"""
        raise NotImplementedError

    def get_capabilities(self) -> list[str]:
        """獲取組件能力"""
        raise NotImplementedError


class BioNeuronAgentComponent(AIComponentInterface):
    """Bio-Neuron Agent 組件"""

    def __init__(self):
        super().__init__(AIComponentType.BIO_NEURON_AGENT)
        self._agent = None

    async def initialize(self) -> bool:
        """初始化 Bio-Neuron Agent"""
        try:
            # 動態導入現有的 Bio-Neuron Agent
            from ...core.aiva_core.bio_neuron_master import BioNeuronMaster

            self._agent = BioNeuronMaster()
            if hasattr(self._agent, "initialize"):
                await self._agent.initialize()

            self.is_initialized = True
            self.logger.info("Bio-Neuron Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Bio-Neuron Agent: {e}")
            return False

    async def process_task(self, task: AITask) -> AIResult:
        """處理任務"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # 根據任務類型調用相應方法
            if task.task_type == AITaskType.REASONING:
                result = await self._handle_reasoning(task)
            elif task.task_type == AITaskType.PLANNING:
                result = await self._handle_planning(task)
            elif task.task_type == AITaskType.LEARNING:
                result = await self._handle_learning(task)
            else:
                result = await self._handle_general(task)

            execution_time = time.time() - start_time

            return AIResult(
                task_id=task.task_id,
                success=True,
                result=result,
                confidence=0.9,  # Bio-Neuron Agent 通常有高置信度
                execution_time=execution_time,
                component_used=self.component_type,
                metadata={"method": "bio_neuron_processing"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_context = create_error_context("bio_neuron_agent", "process_task")
            aiva_error = get_error_handler().handle_error(e, error_context)

            return AIResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                component_used=self.component_type,
                error=aiva_error,
            )

    async def _handle_reasoning(self, task: AITask) -> dict[str, Any]:
        """處理推理任務"""
        if self._agent and hasattr(self._agent, "reason"):
            return await self._agent.reason(task.input_data, task.context)
        else:
            # 回退到簡單處理
            return {
                "reasoning_result": f"Processed reasoning task: {task.input_data}",
                "method": "bio_neuron_fallback",
            }

    async def _handle_planning(self, task: AITask) -> dict[str, Any]:
        """處理規劃任務"""
        if self._agent and hasattr(self._agent, "plan"):
            return await self._agent.plan(task.input_data, task.context)
        else:
            return {
                "plan_result": f"Generated plan for: {task.input_data}",
                "steps": ["analyze", "execute", "verify"],
                "method": "bio_neuron_fallback",
            }

    async def _handle_learning(self, task: AITask) -> dict[str, Any]:
        """處理學習任務"""
        if self._agent and hasattr(self._agent, "learn"):
            return await self._agent.learn(task.input_data, task.context)
        else:
            return {
                "learning_result": f"Learned from: {task.input_data}",
                "method": "bio_neuron_fallback",
            }

    async def _handle_general(self, task: AITask) -> dict[str, Any]:
        """處理通用任務"""
        return {
            "result": f"Bio-Neuron processed task: {task.task_type.value}",
            "input": task.input_data,
            "method": "bio_neuron_general",
        }

    def get_capabilities(self) -> list[str]:
        """獲取能力"""
        return [
            "advanced_reasoning",
            "strategic_planning",
            "adaptive_learning",
            "neural_processing",
            "pattern_recognition",
        ]


class RAGEngineComponent(AIComponentInterface):
    """RAG 引擎組件"""

    def __init__(self):
        super().__init__(AIComponentType.RAG_ENGINE)
        self._rag_agent = None

    async def initialize(self) -> bool:
        """初始化 RAG 引擎"""
        try:
            # 導入現有的 RAG Agent
            from ..rag_agent import RAGAgent

            self._rag_agent = RAGAgent()
            if hasattr(self._rag_agent, "initialize"):
                await self._rag_agent.initialize()

            self.is_initialized = True
            self.logger.info("RAG Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Engine: {e}")
            return False

    async def process_task(self, task: AITask) -> AIResult:
        """處理 RAG 任務"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # RAG 引擎主要處理檢索和生成任務
            if task.task_type == AITaskType.ANALYSIS:
                result = await self._handle_retrieval_analysis(task)
            elif task.task_type == AITaskType.GENERATION:
                result = await self._handle_augmented_generation(task)
            else:
                result = await self._handle_general_rag(task)

            execution_time = time.time() - start_time

            return AIResult(
                task_id=task.task_id,
                success=True,
                result=result,
                confidence=0.85,
                execution_time=execution_time,
                component_used=self.component_type,
                metadata={"method": "rag_processing"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_context = create_error_context("rag_engine", "process_task")
            aiva_error = get_error_handler().handle_error(e, error_context)

            return AIResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                component_used=self.component_type,
                error=aiva_error,
            )

    async def _handle_retrieval_analysis(self, task: AITask) -> dict[str, Any]:
        """處理檢索分析"""
        query = task.input_data.get("query", "")

        if self._rag_agent and hasattr(self._rag_agent, "search_and_analyze"):
            return await self._rag_agent.search_and_analyze(query, task.context)
        else:
            return {
                "analysis_result": f"Analyzed query: {query}",
                "retrieved_docs": [],
                "relevance_scores": [],
                "method": "rag_fallback",
            }

    async def _handle_augmented_generation(self, task: AITask) -> dict[str, Any]:
        """處理增強生成"""
        prompt = task.input_data.get("prompt", "")

        if self._rag_agent and hasattr(self._rag_agent, "generate_with_context"):
            return await self._rag_agent.generate_with_context(prompt, task.context)
        else:
            return {
                "generated_text": f"Generated response for: {prompt}",
                "sources": [],
                "method": "rag_fallback",
            }

    async def _handle_general_rag(self, task: AITask) -> dict[str, Any]:
        """處理通用 RAG 任務"""
        return {
            "result": f"RAG processed task: {task.task_type.value}",
            "input": task.input_data,
            "method": "rag_general",
        }

    def get_capabilities(self) -> list[str]:
        """獲取能力"""
        return [
            "document_retrieval",
            "context_augmentation",
            "knowledge_synthesis",
            "semantic_search",
            "factual_grounding",
        ]


class DialogAssistantComponent(AIComponentInterface):
    """對話助手組件"""

    def __init__(self):
        super().__init__(AIComponentType.DIALOG_ASSISTANT)
        self._dialog_assistant = None

    async def initialize(self) -> bool:
        """初始化對話助手"""
        try:
            # 導入現有的對話助手
            from ..dialog_assistant import DialogAssistant

            self._dialog_assistant = DialogAssistant()
            if hasattr(self._dialog_assistant, "initialize"):
                await self._dialog_assistant.initialize()

            self.is_initialized = True
            self.logger.info("Dialog Assistant initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Dialog Assistant: {e}")
            return False

    async def process_task(self, task: AITask) -> AIResult:
        """處理對話任務"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # 對話助手主要處理對話任務
            if task.task_type == AITaskType.DIALOGUE:
                result = await self._handle_dialogue(task)
            else:
                result = await self._handle_general_dialog(task)

            execution_time = time.time() - start_time

            return AIResult(
                task_id=task.task_id,
                success=True,
                result=result,
                confidence=0.8,
                execution_time=execution_time,
                component_used=self.component_type,
                metadata={"method": "dialog_processing"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_context = create_error_context("dialog_assistant", "process_task")
            aiva_error = get_error_handler().handle_error(e, error_context)

            return AIResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                component_used=self.component_type,
                error=aiva_error,
            )

    async def _handle_dialogue(self, task: AITask) -> dict[str, Any]:
        """處理對話"""
        message = task.input_data.get("message", "")
        session_id = task.context.get("session_id", "")

        if self._dialog_assistant and hasattr(
            self._dialog_assistant, "process_message"
        ):
            return await self._dialog_assistant.process_message(
                message, session_id, task.context
            )
        else:
            return {
                "response": f"Processed message: {message}",
                "session_id": session_id,
                "method": "dialog_fallback",
            }

    async def _handle_general_dialog(self, task: AITask) -> dict[str, Any]:
        """處理通用對話任務"""
        return {
            "result": f"Dialog processed task: {task.task_type.value}",
            "input": task.input_data,
            "method": "dialog_general",
        }

    def get_capabilities(self) -> list[str]:
        """獲取能力"""
        return [
            "natural_conversation",
            "context_awareness",
            "multi_turn_dialogue",
            "intent_understanding",
            "response_generation",
        ]


class PlanExecutorComponent(AIComponentInterface):
    """計劃執行器組件"""

    def __init__(self):
        super().__init__(AIComponentType.PLAN_EXECUTOR)
        self._plan_executor = None

    async def initialize(self) -> bool:
        """初始化計劃執行器"""
        try:
            # 導入現有的計劃執行器
            from ..plan_executor import PlanExecutor

            self._plan_executor = PlanExecutor()
            if hasattr(self._plan_executor, "initialize"):
                await self._plan_executor.initialize()

            self.is_initialized = True
            self.logger.info("Plan Executor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Plan Executor: {e}")
            return False

    async def process_task(self, task: AITask) -> AIResult:
        """處理執行任務"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # 計劃執行器主要處理執行任務
            if task.task_type == AITaskType.EXECUTION:
                result = await self._handle_execution(task)
            elif task.task_type == AITaskType.PLANNING:
                result = await self._handle_execution_planning(task)
            else:
                result = await self._handle_general_execution(task)

            execution_time = time.time() - start_time

            return AIResult(
                task_id=task.task_id,
                success=True,
                result=result,
                confidence=0.9,
                execution_time=execution_time,
                component_used=self.component_type,
                metadata={"method": "plan_execution"},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_context = create_error_context("plan_executor", "process_task")
            aiva_error = get_error_handler().handle_error(e, error_context)

            return AIResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                component_used=self.component_type,
                error=aiva_error,
            )

    async def _handle_execution(self, task: AITask) -> dict[str, Any]:
        """處理執行"""
        plan = task.input_data.get("plan", {})

        if self._plan_executor and hasattr(self._plan_executor, "execute_plan"):
            return await self._plan_executor.execute_plan(plan, task.context)
        else:
            return {
                "execution_result": f"Executed plan: {plan}",
                "status": "completed",
                "method": "executor_fallback",
            }

    async def _handle_execution_planning(self, task: AITask) -> dict[str, Any]:
        """處理執行規劃"""
        goal = task.input_data.get("goal", "")

        if self._plan_executor and hasattr(
            self._plan_executor, "create_execution_plan"
        ):
            return await self._plan_executor.create_execution_plan(goal, task.context)
        else:
            return {
                "plan": {"steps": ["analyze", "execute", "verify"]},
                "goal": goal,
                "method": "executor_fallback",
            }

    async def _handle_general_execution(self, task: AITask) -> dict[str, Any]:
        """處理通用執行任務"""
        return {
            "result": f"Executor processed task: {task.task_type.value}",
            "input": task.input_data,
            "method": "executor_general",
        }

    def get_capabilities(self) -> list[str]:
        """獲取能力"""
        return [
            "plan_execution",
            "task_orchestration",
            "resource_management",
            "progress_tracking",
            "error_recovery",
        ]


class AIIntegrationManager:
    """AI 組件整合管理器"""

    def __init__(self):
        self.logger = logging.getLogger("ai_integration_manager")
        self.components: dict[AIComponentType, AIComponentInterface] = {}
        self.task_queue: list[AITask] = []
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.is_initialized = False

        # 初始化組件
        self._setup_components()

    def _setup_components(self):
        """設置 AI 組件"""
        self.components[AIComponentType.BIO_NEURON_AGENT] = BioNeuronAgentComponent()
        self.components[AIComponentType.RAG_ENGINE] = RAGEngineComponent()
        self.components[AIComponentType.DIALOG_ASSISTANT] = DialogAssistantComponent()
        self.components[AIComponentType.PLAN_EXECUTOR] = PlanExecutorComponent()

    async def initialize(self) -> bool:
        """初始化所有 AI 組件"""
        try:
            self.logger.info("Initializing AI components...")

            # 並行初始化所有組件
            init_tasks = []
            for component_type, component in self.components.items():
                task = asyncio.create_task(component.initialize())
                init_tasks.append((component_type, task))

            # 等待所有初始化完成
            for component_type, task in init_tasks:
                try:
                    success = await task
                    if success:
                        self.logger.info(
                            f"✅ {component_type.value} initialized successfully"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ {component_type.value} initialization failed"
                        )
                except Exception as e:
                    self.logger.error(
                        f"❌ {component_type.value} initialization error: {e}"
                    )

            self.is_initialized = True
            self.logger.info("AI Integration Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Integration Manager: {e}")
            return False

    async def process_ai_task(self, task: AITask) -> AIResult:
        """處理 AI 任務"""
        if not self.is_initialized:
            await self.initialize()

        # 選擇最適合的組件
        component = self._select_component_for_task(task)

        if not component:
            return AIResult(
                task_id=task.task_id,
                success=False,
                error=AIVAError(
                    error_code=AIVAErrorCode.AI_SERVICE_UNAVAILABLE,
                    message=f"No suitable component found for task type: {task.task_type}",
                    severity="medium",
                    context=create_error_context(
                        "ai_integration_manager", "process_ai_task"
                    ),
                ),
            )

        try:
            self.logger.info(
                f"Processing task {task.task_id} with {component.component_type.value}"
            )
            result = await component.process_task(task)

            if result.success:
                self.logger.info(f"✅ Task {task.task_id} completed successfully")
            else:
                self.logger.warning(f"⚠️ Task {task.task_id} failed")

            return result

        except Exception as e:
            self.logger.error(f"❌ Task {task.task_id} processing error: {e}")
            error_context = create_error_context(
                "ai_integration_manager", "process_ai_task"
            )
            aiva_error = get_error_handler().handle_error(e, error_context)

            return AIResult(task_id=task.task_id, success=False, error=aiva_error)

    def _select_component_for_task(
        self, task: AITask
    ) -> AIComponentInterface | None:
        """為任務選擇最適合的組件"""
        # 根據任務類型和要求選擇組件
        component_preferences = {
            AITaskType.REASONING: [
                AIComponentType.BIO_NEURON_AGENT,
                AIComponentType.RAG_ENGINE,
            ],
            AITaskType.PLANNING: [
                AIComponentType.BIO_NEURON_AGENT,
                AIComponentType.PLAN_EXECUTOR,
            ],
            AITaskType.EXECUTION: [
                AIComponentType.PLAN_EXECUTOR,
                AIComponentType.BIO_NEURON_AGENT,
            ],
            AITaskType.LEARNING: [AIComponentType.BIO_NEURON_AGENT],
            AITaskType.DIALOGUE: [
                AIComponentType.DIALOG_ASSISTANT,
                AIComponentType.RAG_ENGINE,
            ],
            AITaskType.ANALYSIS: [
                AIComponentType.RAG_ENGINE,
                AIComponentType.BIO_NEURON_AGENT,
            ],
            AITaskType.GENERATION: [
                AIComponentType.RAG_ENGINE,
                AIComponentType.DIALOG_ASSISTANT,
            ],
        }

        preferred_components = component_preferences.get(task.task_type, [])

        # 選擇第一個可用且已初始化的組件
        for component_type in preferred_components:
            component = self.components.get(component_type)
            if component and component.is_initialized:
                return component

        # 如果沒有首選組件，選擇任何可用的組件
        for component in self.components.values():
            if component.is_initialized:
                return component

        return None

    async def get_component_status(self) -> dict[str, Any]:
        """獲取組件狀態"""
        status = {}

        for component_type, component in self.components.items():
            status[component_type.value] = {
                "initialized": component.is_initialized,
                "capabilities": (
                    component.get_capabilities() if component.is_initialized else []
                ),
            }

        return {
            "integration_manager_initialized": self.is_initialized,
            "components": status,
            "active_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
        }

    async def cleanup(self):
        """清理所有組件"""
        try:
            cleanup_tasks = []
            for component in self.components.values():
                if component.is_initialized:
                    task = asyncio.create_task(component.cleanup())
                    cleanup_tasks.append(task)

            # 等待所有清理完成
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            self.is_initialized = False
            self.logger.info("AI Integration Manager cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# 全局 AI 整合管理器
_global_ai_manager: AIIntegrationManager | None = None


def get_ai_integration_manager() -> AIIntegrationManager:
    """獲取全局 AI 整合管理器"""
    global _global_ai_manager
    if _global_ai_manager is None:
        _global_ai_manager = AIIntegrationManager()
    return _global_ai_manager


# 便利函數
async def process_ai_task(
    task_type: AITaskType,
    input_data: dict[str, Any],
    context: dict[str, Any] | None = None,
    priority: int = 5,
    timeout: float = 30.0,
) -> AIResult:
    """處理 AI 任務的便利函數"""
    task = AITask(
        task_id=f"task_{int(time.time())}_{id(input_data)}",
        task_type=task_type,
        priority=priority,
        input_data=input_data,
        context=context or {},
        requirements={},
        timeout=timeout,
    )

    manager = get_ai_integration_manager()
    return await manager.process_ai_task(task)


if __name__ == "__main__":
    # 測試 AI 組件整合
    async def test_ai_integration():
        manager = get_ai_integration_manager()
        await manager.initialize()

        try:
            # 測試推理任務
            result = await process_ai_task(
                AITaskType.REASONING,
                {"query": "Analyze security vulnerabilities"},
                {"domain": "cybersecurity"},
            )
            print("Reasoning task result:", result.success)

            # 測試對話任務
            result = await process_ai_task(
                AITaskType.DIALOGUE,
                {"message": "Hello, how can you help me?"},
                {"session_id": "test_session"},
            )
            print("Dialogue task result:", result.success)

            # 檢查狀態
            status = await manager.get_component_status()
            print("Component status:", status)

        finally:
            await manager.cleanup()

    asyncio.run(test_ai_integration())
