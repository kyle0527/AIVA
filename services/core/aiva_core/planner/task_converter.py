"""Task Converter - 任務轉換器

將 AST 節點轉換為可執行的任務序列

Compliance Note (遵循 aiva_common 設計原則):
- TaskStatus 已從本地定義移除,改用 aiva_common.enums.common.TaskStatus (4-layer priority)
- TaskPriority 保留為模組特定 enum (AI 規劃器專用優先級)
- 修正日期: 2025-10-25
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

from services.aiva_common.enums.common import TaskStatus

from .ast_parser import AttackFlowGraph, AttackFlowNode, NodeType

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """任務優先級 (AI 規劃器專用)

    Note: 此為模組特定 enum,用於 AI 規劃器的任務優先級排程。
    與通用的 TaskStatus 不同,TaskPriority 是 AI 引擎內部使用的排程策略。
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutableTask:
    """可執行任務

    代表一個可以被任務執行器執行的具體任務
    """

    task_id: str
    task_type: str  # 例如: "scan", "analyze", "exploit"
    action: str  # 具體動作
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # 依賴的任務 ID
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    source_node_id: str | None = None  # 來源 AST 節點 ID
    metadata: dict[str, Any] = field(default_factory=dict)

    # 執行結果
    result: dict[str, Any] | None = None
    error: str | None = None

    def __repr__(self) -> str:
        return f"Task({self.task_id}:{self.task_type}:{self.status.value})"


@dataclass
class TaskSequence:
    """任務序列

    代表一組有序的可執行任務
    """

    sequence_id: str
    tasks: list[ExecutableTask] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: ExecutableTask) -> None:
        """添加任務"""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> ExecutableTask | None:
        """根據 ID 獲取任務"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_pending_tasks(self) -> list[ExecutableTask]:
        """獲取所有待執行的任務"""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_runnable_tasks(self) -> list[ExecutableTask]:
        """獲取所有可以立即執行的任務（依賴已滿足）"""
        runnable = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            # 檢查依賴是否都已完成
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if not dep_task or dep_task.status != TaskStatus.SUCCESS:
                    dependencies_met = False
                    break

            if dependencies_met:
                runnable.append(task)

        return runnable


class TaskConverter:
    """任務轉換器

    將 AttackFlowGraph 轉換為 TaskSequence
    """

    def __init__(self) -> None:
        """初始化任務轉換器"""
        logger.info("TaskConverter initialized")

    def convert(self, graph: AttackFlowGraph) -> TaskSequence:
        """將攻擊流程圖轉換為任務序列

        Args:
            graph: 攻擊流程圖

        Returns:
            任務序列
        """
        sequence = TaskSequence(
            sequence_id=f"seq_{graph.graph_id}_{uuid4().hex[:8]}",
            metadata={"source_graph_id": graph.graph_id, **graph.metadata},
        )

        # 使用拓撲排序確定任務執行順序
        sorted_nodes = self._topological_sort(graph)

        # 為每個節點創建對應的任務
        node_to_task_id: dict[str, str] = {}
        for node in sorted_nodes:
            # 跳過 START 和 END 節點
            if node.node_type in (NodeType.START, NodeType.END):
                continue

            task = self._create_task_from_node(node)

            # 確定任務依賴
            dependencies = []
            for edge in graph.edges:
                if edge.to_node == node.node_id:
                    source_node_id = edge.from_node
                    if source_node_id in node_to_task_id:
                        dependencies.append(node_to_task_id[source_node_id])

            task.dependencies = dependencies
            node_to_task_id[node.node_id] = task.task_id

            sequence.add_task(task)

        logger.info(
            f"Converted graph '{graph.graph_id}' to sequence "
            f"with {len(sequence.tasks)} tasks"
        )
        return sequence

    def _topological_sort(self, graph: AttackFlowGraph) -> list[AttackFlowNode]:
        """拓撲排序

        Args:
            graph: 攻擊流程圖

        Returns:
            排序後的節點列表
        """
        # 簡單的 BFS 拓撲排序
        visited = set()
        result = []

        start_node = graph.get_start_node()
        if not start_node:
            return list(graph.nodes.values())

        queue = [start_node]
        visited.add(start_node.node_id)

        while queue:
            current = queue.pop(0)
            result.append(current)

            next_nodes = graph.get_next_nodes(current.node_id)
            for node in next_nodes:
                if node.node_id not in visited:
                    visited.add(node.node_id)
                    queue.append(node)

        # 添加未連接的節點
        for node in graph.nodes.values():
            if node.node_id not in visited:
                result.append(node)

        return result

    def _create_task_from_node(self, node: AttackFlowNode) -> ExecutableTask:
        """從 AST 節點創建可執行任務

        Args:
            node: AST 節點

        Returns:
            可執行任務
        """
        task_id = f"task_{uuid4().hex[:8]}"

        # 根據節點類型設置任務類型
        task_type_map = {
            NodeType.SCAN: "scan",
            NodeType.ANALYZE: "analyze",
            NodeType.EXPLOIT: "exploit",
            NodeType.VALIDATE: "validate",
            NodeType.BRANCH: "decision",
        }
        task_type = task_type_map.get(node.node_type, "generic")

        # 設置優先級
        priority_map = {
            NodeType.SCAN: TaskPriority.HIGH,
            NodeType.ANALYZE: TaskPriority.NORMAL,
            NodeType.EXPLOIT: TaskPriority.CRITICAL,
            NodeType.VALIDATE: TaskPriority.HIGH,
        }
        priority = priority_map.get(node.node_type, TaskPriority.NORMAL)

        task = ExecutableTask(
            task_id=task_id,
            task_type=task_type,
            action=node.action,
            parameters=node.parameters.copy(),
            priority=priority,
            source_node_id=node.node_id,
            metadata={
                "node_type": node.node_type.value,
                "condition": node.condition,
                **node.metadata,
            },
        )

        return task
