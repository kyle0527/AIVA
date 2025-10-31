"""AST Parser - 攻擊流程圖解析器

負責解析 AI 引擎生成的攻擊流程 AST (Abstract Syntax Tree)，
將其轉換為結構化的圖形表示。
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """攻擊流程節點類型"""

    START = "start"  # 開始節點
    SCAN = "scan"  # 掃描/探測
    ANALYZE = "analyze"  # 分析
    EXPLOIT = "exploit"  # 漏洞利用
    VALIDATE = "validate"  # 驗證
    BRANCH = "branch"  # 條件分支
    END = "end"  # 結束節點


@dataclass
class AttackFlowNode:
    """攻擊流程節點

    代表攻擊流程中的單一步驟
    """

    node_id: str
    node_type: NodeType
    action: str  # 具體動作描述
    parameters: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None  # 對於 BRANCH 節點的條件
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Node({self.node_id}:{self.node_type.value}:{self.action})"


@dataclass
class AttackFlowEdge:
    """攻擊流程邊

    代表節點之間的轉移關係
    """

    from_node: str
    to_node: str
    condition: str | None = None  # 條件邊的條件
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        return f"{self.from_node} -> {self.to_node}{cond}"


@dataclass
class AttackFlowGraph:
    """攻擊流程圖

    使用節點和有向邊表示完整的攻擊流程
    """

    graph_id: str
    nodes: dict[str, AttackFlowNode] = field(default_factory=dict)
    edges: list[AttackFlowEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: AttackFlowNode) -> None:
        """添加節點"""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: AttackFlowEdge) -> None:
        """添加邊"""
        if edge.from_node not in self.nodes:
            raise ValueError(f"Source node {edge.from_node} not found")
        if edge.to_node not in self.nodes:
            raise ValueError(f"Target node {edge.to_node} not found")
        self.edges.append(edge)

    def get_start_node(self) -> AttackFlowNode | None:
        """獲取開始節點"""
        for node in self.nodes.values():
            if node.node_type == NodeType.START:
                return node
        return None

    def get_next_nodes(self, current_node_id: str) -> list[AttackFlowNode]:
        """獲取指定節點的下一步節點"""
        next_nodes = []
        for edge in self.edges:
            if edge.from_node == current_node_id:
                next_nodes.append(self.nodes[edge.to_node])
        return next_nodes

    def validate(self) -> tuple[bool, list[str]]:
        """驗證圖的完整性

        Returns:
            (是否有效, 錯誤訊息列表)
        """
        errors = []

        # 檢查是否有開始節點
        if not self.get_start_node():
            errors.append("Missing START node")

        # 檢查是否有結束節點
        has_end = any(n.node_type == NodeType.END for n in self.nodes.values())
        if not has_end:
            errors.append("Missing END node")

        # 檢查是否有孤立節點
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.from_node)
            connected_nodes.add(edge.to_node)

        for node_id in self.nodes:
            if node_id not in connected_nodes:
                errors.append(f"Isolated node: {node_id}")

        return len(errors) == 0, errors


class ASTParser:
    """AST 解析器

    將各種格式的攻擊流程描述轉換為 AttackFlowGraph
    """

    def __init__(self) -> None:
        """初始化解析器"""
        logger.info("ASTParser initialized")

    def parse_dict(self, ast_dict: dict[str, Any]) -> AttackFlowGraph:
        """從字典格式解析 AST

        Args:
            ast_dict: AST 字典格式，包含 nodes 和 edges

        Returns:
            AttackFlowGraph 實例
        """
        graph_id = ast_dict.get("graph_id", "unknown")
        graph = AttackFlowGraph(
            graph_id=graph_id, metadata=ast_dict.get("metadata", {})
        )

        # 解析節點
        for node_dict in ast_dict.get("nodes", []):
            node = AttackFlowNode(
                node_id=node_dict["node_id"],
                node_type=NodeType(node_dict["node_type"]),
                action=node_dict["action"],
                parameters=node_dict.get("parameters", {}),
                condition=node_dict.get("condition"),
                metadata=node_dict.get("metadata", {}),
            )
            graph.add_node(node)

        # 解析邊
        for edge_dict in ast_dict.get("edges", []):
            edge = AttackFlowEdge(
                from_node=edge_dict["from_node"],
                to_node=edge_dict["to_node"],
                condition=edge_dict.get("condition"),
                metadata=edge_dict.get("metadata", {}),
            )
            graph.add_edge(edge)

        # 驗證圖
        is_valid, errors = graph.validate()
        if not is_valid:
            logger.warning(f"AST validation errors: {errors}")

        logger.info(
            f"Parsed AST graph '{graph_id}' with {len(graph.nodes)} nodes "
            f"and {len(graph.edges)} edges"
        )
        return graph

    def parse_text(self, ast_text: str) -> AttackFlowGraph:
        """從文本格式解析 AST

        支持簡單的文本格式，例如：
        START -> SCAN(url=target) -> ANALYZE -> EXPLOIT -> VALIDATE -> END

        Args:
            ast_text: AST 文本描述

        Returns:
            AttackFlowGraph 實例
        """
        # TODO: 實現文本解析邏輯
        logger.warning("Text parsing not fully implemented yet")
        graph = AttackFlowGraph(graph_id="text_parsed")
        return graph

    def create_example_sqli_flow(self) -> AttackFlowGraph:
        """創建一個 SQL 注入攻擊流程的範例

        Returns:
            SQL 注入攻擊流程圖
        """
        graph = AttackFlowGraph(
            graph_id="sqli_example",
            metadata={"attack_type": "sqli", "target": "example.com"},
        )

        # 創建節點
        nodes = [
            AttackFlowNode("n0", NodeType.START, "開始攻擊"),
            AttackFlowNode(
                "n1",
                NodeType.SCAN,
                "掃描目標",
                parameters={"url": "http://example.com", "depth": 2},
            ),
            AttackFlowNode(
                "n2",
                NodeType.ANALYZE,
                "分析參數",
                parameters={"focus": "sql_injection"},
            ),
            AttackFlowNode(
                "n3",
                NodeType.EXPLOIT,
                "嘗試 SQL 注入",
                parameters={"payload_type": "union_based"},
            ),
            AttackFlowNode(
                "n4",
                NodeType.VALIDATE,
                "驗證漏洞",
                parameters={"expected": "database_error"},
            ),
            AttackFlowNode("n5", NodeType.END, "結束攻擊"),
        ]

        for node in nodes:
            graph.add_node(node)

        # 創建邊
        edges = [
            AttackFlowEdge("n0", "n1"),
            AttackFlowEdge("n1", "n2"),
            AttackFlowEdge("n2", "n3"),
            AttackFlowEdge("n3", "n4"),
            AttackFlowEdge("n4", "n5"),
        ]

        for edge in edges:
            graph.add_edge(edge)

        return graph
