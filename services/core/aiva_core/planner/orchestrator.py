"""Attack Orchestrator - 攻擊編排器

統籌整個攻擊計畫的執行，協調 AST 解析、任務轉換和工具選擇
"""

from dataclasses import dataclass, field
import logging
from typing import Any
from uuid import uuid4

# aiva_common 統一錯誤處理
from aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context,
)

from .ast_parser import ASTParser, AttackFlowGraph
from .task_converter import ExecutableTask, TaskConverter, TaskSequence, TaskStatus
from .tool_selector import ToolDecision, ToolSelector

logger = logging.getLogger(__name__)
MODULE_NAME = "planner_orchestrator"


@dataclass
class ExecutionPlan:
    """執行計畫

    包含完整的任務序列和工具選擇決策
    """

    plan_id: str
    graph: AttackFlowGraph
    task_sequence: TaskSequence
    tool_decisions: dict[str, ToolDecision] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_decision_for_task(self, task_id: str) -> ToolDecision | None:
        """獲取任務的工具選擇決策"""
        return self.tool_decisions.get(task_id)


class AttackOrchestrator:
    """攻擊編排器

    將 AST 攻擊流程圖轉換為完整的執行計畫
    """

    def __init__(self) -> None:
        """初始化編排器"""
        self.ast_parser = ASTParser()
        self.task_converter = TaskConverter()
        self.tool_selector = ToolSelector()
        logger.info("AttackOrchestrator initialized")

    def create_execution_plan(
        self, ast_input: dict[str, Any] | AttackFlowGraph
    ) -> ExecutionPlan:
        """創建執行計畫

        Args:
            ast_input: AST 輸入 (可以是字典或 AttackFlowGraph)

        Returns:
            完整的執行計畫
        """
        plan_id = f"plan_{uuid4().hex[:8]}"
        logger.info(f"Creating execution plan {plan_id}")

        # 1. 解析 AST
        if isinstance(ast_input, dict):
            graph = self.ast_parser.parse_dict(ast_input)
        elif isinstance(ast_input, AttackFlowGraph):
            graph = ast_input
        else:
            raise AIVAError(
                f"Unsupported AST input type: {type(ast_input)}",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=create_error_context(module=MODULE_NAME, function="create_execution_plan")
            )

        # 2. 轉換為任務序列
        task_sequence = self.task_converter.convert(graph)

        # 3. 為每個任務選擇工具
        tool_decisions: dict[str, ToolDecision] = {}
        for task in task_sequence.tasks:
            decision = self.tool_selector.select_tool(task)
            tool_decisions[task.task_id] = decision

        # 4. 創建執行計畫
        plan = ExecutionPlan(
            plan_id=plan_id,
            graph=graph,
            task_sequence=task_sequence,
            tool_decisions=tool_decisions,
            metadata={
                "source_graph_id": graph.graph_id,
                "total_tasks": len(task_sequence.tasks),
                "attack_type": graph.metadata.get("attack_type", "unknown"),
            },
        )

        logger.info(
            f"Execution plan {plan_id} created with {len(task_sequence.tasks)} tasks"
        )
        return plan

    def get_next_executable_tasks(
        self, plan: ExecutionPlan
    ) -> list[tuple[ExecutableTask, ToolDecision]]:
        """獲取下一批可執行的任務及其工具決策

        Args:
            plan: 執行計畫

        Returns:
            (任務, 工具決策) 元組列表
        """
        runnable_tasks = plan.task_sequence.get_runnable_tasks()

        result = []
        for task in runnable_tasks:
            decision = plan.get_decision_for_task(task.task_id)
            if decision:
                result.append((task, decision))

        logger.debug(f"Found {len(result)} executable tasks")
        return result

    def update_task_status(
        self,
        plan: ExecutionPlan,
        task_id: str,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """更新任務狀態

        Args:
            plan: 執行計畫
            task_id: 任務 ID
            status: 新狀態
            result: 執行結果
            error: 錯誤訊息
        """
        task = plan.task_sequence.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found in plan {plan.plan_id}")
            return

        task.status = status
        task.result = result
        task.error = error

        logger.info(f"Task {task_id} status updated to {status.value}")

    def is_plan_complete(self, plan: ExecutionPlan) -> bool:
        """檢查執行計畫是否已完成

        Args:
            plan: 執行計畫

        Returns:
            是否完成
        """
        for task in plan.task_sequence.tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
        return True

    def get_plan_summary(self, plan: ExecutionPlan) -> dict[str, Any]:
        """獲取執行計畫摘要

        Args:
            plan: 執行計畫

        Returns:
            摘要字典
        """
        tasks = plan.task_sequence.tasks
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for t in tasks if t.status == status)

        return {
            "plan_id": plan.plan_id,
            "total_tasks": len(tasks),
            "status_counts": status_counts,
            "is_complete": self.is_plan_complete(plan),
            "attack_type": plan.metadata.get("attack_type", "unknown"),
            "target": plan.graph.metadata.get("target", "unknown"),
        }
