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
from services.aiva_common.error_handling import (
    AIVAError,
    ErrorSeverity,
    ErrorType,
    create_error_context,
)

from .ast_parser import AttackFlowGraph, AttackFlowNode, NodeType

logger = logging.getLogger(__name__)
MODULE_NAME = "aiva_core.planner.task_converter"


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
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
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

    def convert(self, graph: AttackFlowGraph, variables: dict[str, Any] | None = None) -> TaskSequence:
        """將攻擊流程圖轉換為任務序列

        Args:
            graph: 攻擊流程圖
            variables: 用於插值的變數字典

        Returns:
            任務序列
        """
        vars_dict = variables or {}
        sequence = TaskSequence(
            sequence_id=f"seq_{graph.graph_id}_{uuid4().hex[:8]}",
            metadata={"source_graph_id": graph.graph_id, "variables": vars_dict, **graph.metadata},
        )

        # 使用增強的拓撲排序確定任務執行順序
        sorted_nodes = self._topological_sort(graph)

        # 為每個節點創建對應的任務
        node_to_task_id: dict[str, str] = {}
        for node in sorted_nodes:
            # 跳過 START 和 END 節點
            if node.node_type in (NodeType.START, NodeType.END):
                continue

            task = self._create_task_from_node(node, vars_dict)

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
            f"with {len(sequence.tasks)} tasks (variables: {len(vars_dict)})"
        )
        return sequence

    def _topological_sort(self, graph: AttackFlowGraph) -> list[AttackFlowNode]:
        """智能拓撲排序 (整合自 aiva_core_v1)

        基於 Kahn 算法的拓撲排序，支持循環依賴檢測和最優執行順序

        Args:
            graph: 攻擊流程圖

        Returns:
            排序後的節點列表

        Raises:
            RuntimeError: 當檢測到循環依賴或無法滿足的依賴關係
        """
        nodes = list(graph.nodes.values())
        
        # 構建依賴映射表 {節點ID: 依賴節點ID集合}
        dependency_map = {}
        for node in nodes:
            dependencies = set()
            # 通過邊關係構建依賴
            for edge in graph.edges:
                if edge.to_node == node.node_id:
                    dependencies.add(edge.from_node)
            dependency_map[node.node_id] = dependencies

        # Kahn 算法進行拓撲排序
        result = []
        
        while dependency_map:
            # 找到沒有依賴的節點
            acyclic_nodes = [node_id for node_id, deps in dependency_map.items() if not deps]
            
            if not acyclic_nodes:
                # 檢測到循環依賴
                remaining_nodes = list(dependency_map.keys())
                logger.error(f"Circular dependency detected among nodes: {remaining_nodes}")
                raise AIVAError(
                    f"Flow has cycles or unmet dependencies. "
                    f"Remaining nodes with dependencies: {remaining_nodes}",
                    error_type=ErrorType.VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    context=create_error_context(
                        module=MODULE_NAME,
                        function="_topological_sort",
                        remaining_nodes=remaining_nodes
                    )
                )
            
            # 按優先級排序無依賴節點 (確保執行順序最優)
            acyclic_nodes.sort(key=lambda node_id: self._get_node_priority(
                next(n for n in nodes if n.node_id == node_id)
            ), reverse=True)
            
            # 處理當前批次的無依賴節點
            for node_id in acyclic_nodes:
                node_obj = next(n for n in nodes if n.node_id == node_id)
                result.append(node_obj)
                del dependency_map[node_id]
            
            # 從剩餘節點的依賴中移除已處理的節點
            for deps in dependency_map.values():
                deps.difference_update(set(acyclic_nodes))

        logger.info(f"Topological sort completed: {len(result)} nodes ordered")
        return result

    def _get_node_priority(self, node: AttackFlowNode) -> int:
        """獲取節點優先級權重 (用於拓撲排序時的順序優化)

        Args:
            node: 攻擊流程節點

        Returns:
            優先級權重 (數值越大優先級越高)
        """
        priority_weights = {
            NodeType.START: 100,      # 開始節點最高優先級
            NodeType.SCAN: 80,        # 掃描節點高優先級
            NodeType.ANALYZE: 60,     # 分析節點中優先級  
            NodeType.VALIDATE: 70,    # 驗證節點中高優先級
            NodeType.EXPLOIT: 50,     # 攻擊節點中優先級
            NodeType.BRANCH: 40,      # 分支節點低優先級
            NodeType.END: 10,         # 結束節點最低優先級
        }
        
        return priority_weights.get(node.node_type, 30)  # 默認低優先級

    def _create_task_from_node(self, node: AttackFlowNode, variables: dict[str, Any] | None = None) -> ExecutableTask:
        """從 AST 節點創建可執行任務 (支持變數插值)

        Args:
            node: AST 節點
            variables: 用於插值的變數字典

        Returns:
            可執行任務
        """
        vars_dict = variables or {}
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

        # 處理參數中的變數插值 (整合自 aiva_core_v1)
        interpolated_params = self._interpolate_variables(node.parameters, vars_dict)

        task = ExecutableTask(
            task_id=task_id,
            task_type=task_type,
            action=node.action,
            parameters=interpolated_params,
            priority=priority,
            source_node_id=node.node_id,
            metadata={
                "node_type": node.node_type.value,
                "condition": node.condition,
                "original_params": node.parameters,  # 保留原始參數
                **node.metadata,
            },
        )

        return task

    def _interpolate_variables(self, params: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
        """變數插值處理 (整合自 aiva_core_v1)

        支持 ${variable} 語法的變數替換

        Args:
            params: 原始參數字典
            variables: 變數值字典

        Returns:
            插值後的參數字典
        """
        interpolated = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # 提取變數名稱
                var_name = value[2:-1]  # 移除 ${ 和 }
                
                # 支持嵌套變數路徑 (例如: ${result.target_url})
                if "." in var_name:
                    interpolated[key] = self._resolve_nested_variable(var_name, variables)
                else:
                    interpolated[key] = variables.get(var_name, value)  # 找不到變數時保持原值
                
                logger.debug(f"Interpolated variable '{var_name}': {value} -> {interpolated[key]}")
            else:
                interpolated[key] = value
        
        return interpolated

    def _resolve_nested_variable(self, var_path: str, variables: dict[str, Any]) -> Any:
        """解析嵌套變數路徑

        Args:
            var_path: 變數路徑 (例如: "result.target_url") 
            variables: 變數值字典

        Returns:
            解析後的值，如果找不到則返回原始路徑
        """
        try:
            parts = var_path.split(".")
            value = variables
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return f"${{{var_path}}}"  # 返回原始格式
            
            return value
        except (KeyError, TypeError):
            return f"${{{var_path}}}"  # 解析失敗時返回原始格式
