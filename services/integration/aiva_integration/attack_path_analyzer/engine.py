"""
Attack Path Analyzer - 攻擊路徑分析引擎

使用 NetworkX 圖庫建立資產與漏洞的關聯圖,
計算從外部攻擊者到核心資產的攻擊路徑。

Compliance Note (遵循 aiva_common 設計原則):
- NodeType, EdgeType 已移除,改為從 aiva_common.enums.security import (4-layer priority 原則)
- AttackPathNodeType → 節點類型枚舉
- AttackPathEdgeType → 邊類型枚舉
- 修正日期: 2025-10-25
- 遷移記錄: 2025-11-16 從 Neo4j 遷移至 NetworkX (降低外部依賴)
"""

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from services.aiva_common.enums import Severity
from services.aiva_common.enums.security import (
    AttackPathNodeType as NodeType,
    AttackPathEdgeType as EdgeType,
)
from services.aiva_common.schemas import Asset, FindingPayload

logger = logging.getLogger(__name__)


@dataclass
class AttackPath:
    """攻擊路徑"""

    path_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_risk_score: float
    length: int
    description: str


class AttackPathEngine:
    """攻擊路徑分析引擎 (使用 NetworkX)"""

    def __init__(
        self,
        graph_file: str | Path | None = None,
    ):
        """
        初始化引擎

        Args:
            graph_file: 圖持久化檔案路徑 (可選,用於載入既有圖)
        """
        self.graph = nx.DiGraph()  # 有向圖
        self.graph_file = Path(graph_file) if graph_file else None
        
        # 如果提供了檔案且存在,則載入
        if self.graph_file and self.graph_file.exists():
            self.load_graph()
            logger.info(f"Loaded graph from {self.graph_file}")
        else:
            self.initialize_graph()
            logger.info("Initialized new NetworkX graph")

    def close(self):
        """儲存圖 (如果有指定檔案)"""
        if self.graph_file:
            self.save_graph()

    def save_graph(self, file_path: str | Path | None = None) -> None:
        """
        儲存圖到檔案

        Args:
            file_path: 儲存路徑 (可選,預設使用初始化時的路徑)
        """
        target_file = Path(file_path) if file_path else self.graph_file
        if not target_file:
            logger.warning("No file path specified for saving graph")
            return
        
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info(f"Graph saved to {target_file}")

    def load_graph(self, file_path: str | Path | None = None) -> None:
        """
        從檔案載入圖

        Args:
            file_path: 載入路徑 (可選,預設使用初始化時的路徑)
        """
        source_file = Path(file_path) if file_path else self.graph_file
        if not source_file or not source_file.exists():
            logger.warning(f"Graph file not found: {source_file}")
            return
        
        with open(source_file, "rb") as f:
            self.graph = pickle.load(f)
        logger.info(f"Graph loaded from {source_file}")

    def initialize_graph(self):
        """初始化圖結構（建立攻擊者節點）"""
        # 建立攻擊者節點（外部攻擊者）
        self.graph.add_node(
            "external_attacker",
            node_type="Attacker",
            name="External Attacker",
            description="外部攻擊者",
        )
        logger.info("Graph structure initialized with external attacker")

    def add_asset(self, asset: Asset) -> None:
        """
        新增資產節點

        Args:
            asset: 資產物件
        """
        asset_id = asset.asset_id
        
        # 新增資產節點
        self.graph.add_node(
            asset_id,
            node_type="Asset",
            value=asset.value,
            type=asset.type,
            is_public=True,  # 假設從外部掃描發現的都是公開資產
        )

        # 如果是公開資產,連接到外部攻擊者
        if self.graph.nodes[asset_id].get("is_public"):
            self.graph.add_edge(
                "external_attacker",
                asset_id,
                edge_type="CAN_ACCESS",
                risk=1.0,
            )
        
        logger.debug(f"Added asset: {asset_id}")

    def add_finding(self, finding: FindingPayload) -> None:
        """
        新增漏洞發現,建立資產與漏洞的關聯

        Args:
            finding: Finding 物件
        """
        finding_id = finding.finding_id
        risk_score = self._calculate_risk_score(finding)
        
        # 建立漏洞節點
        self.graph.add_node(
            finding_id,
            node_type="Vulnerability",
            name=finding.vulnerability.name.value,
            severity=finding.vulnerability.severity.value,
            confidence=finding.vulnerability.confidence.value,
            cwe=finding.vulnerability.cwe,
            risk_score=risk_score,
        )

        # 建立或取得資產節點
        target_url = str(finding.target.url)
        asset_id = f"asset_{finding_id}"
        
        if not self.graph.has_node(asset_id):
            self.graph.add_node(
                asset_id,
                node_type="Asset",
                value=target_url,
                type="discovered",
            )
        
        # 建立資產與漏洞的關聯
        self.graph.add_edge(
            asset_id,
            finding_id,
            edge_type="HAS_VULNERABILITY",
            risk=risk_score,
        )

        # 根據漏洞類型建立攻擊路徑
        self._create_attack_edges(finding)
        
        logger.debug(f"Added finding: {finding_id}")

    def _create_attack_edges(self, finding: FindingPayload) -> None:
        """
        根據漏洞類型建立攻擊邊

        Args:
            finding: Finding 物件
        """
        finding_id = finding.finding_id
        vuln_name = finding.vulnerability.name.value
        risk_score = self._calculate_risk_score(finding)

        # SSRF -> 內部網路
        if vuln_name == "SSRF":
            internal_id = "internal_network"
            if not self.graph.has_node(internal_id):
                self.graph.add_node(
                    internal_id,
                    node_type="InternalNetwork",
                    name="Internal Network",
                )
            self.graph.add_edge(
                finding_id,
                internal_id,
                edge_type="LEADS_TO",
                risk=risk_score,
            )

        # SQLi -> 資料庫
        elif vuln_name == "SQLI":
            db_id = "database"
            if not self.graph.has_node(db_id):
                self.graph.add_node(
                    db_id,
                    node_type="Database",
                    name="Application Database",
                )
            self.graph.add_edge(
                finding_id,
                db_id,
                edge_type="LEADS_TO",
                risk=risk_score,
            )

        # IDOR/BOLA -> API 端點
        elif vuln_name in ["IDOR", "BOLA"]:
            api_id = f"api_{finding_id}"
            self.graph.add_node(
                api_id,
                node_type="APIEndpoint",
                value=str(finding.target.url),
            )
            self.graph.add_edge(
                finding_id,
                api_id,
                edge_type="GRANTS_ACCESS",
                risk=risk_score,
            )

        # XSS -> 憑證洩漏
        elif vuln_name == "XSS":
            cred_id = f"cred_{finding_id}"
            self.graph.add_node(
                cred_id,
                node_type="Credential",
                type="Session Cookie",
            )
            self.graph.add_edge(
                finding_id,
                cred_id,
                edge_type="EXPOSES",
                risk=risk_score,
            )

    def find_attack_paths(
        self,
        target_node_type: str = "Database",
        max_length: int = 10,
        min_risk_score: float = 0.5,
        limit: int = 10,
    ) -> list[AttackPath]:
        """
        尋找從外部攻擊者到目標節點的攻擊路徑

        Args:
            target_node_type: 目標節點類型
            max_length: 最大路徑長度
            min_risk_score: 最小風險分數
            limit: 回傳路徑數量限制

        Returns:
            攻擊路徑列表
        """
        # 找出所有符合類型的目標節點
        target_nodes = [
            node
            for node, data in self.graph.nodes(data=True)
            if data.get("node_type") == target_node_type
        ]

        if not target_nodes:
            logger.warning(f"No target nodes found with type: {target_node_type}")
            return []

        paths = []
        attacker_id = "external_attacker"

        # 對每個目標節點找路徑
        for target_node in target_nodes:
            try:
                # 使用 NetworkX 的所有簡單路徑演算法
                all_paths = nx.all_simple_paths(
                    self.graph,
                    source=attacker_id,
                    target=target_node,
                    cutoff=max_length,
                )

                for path_nodes in all_paths:
                    # 計算路徑總風險
                    total_risk = 0.0
                    edges = []
                    
                    for i in range(len(path_nodes) - 1):
                        edge_data = self.graph[path_nodes[i]][path_nodes[i + 1]]
                        risk = edge_data.get("risk", 1.0)
                        total_risk += risk
                        edges.append({
                            "source": path_nodes[i],
                            "target": path_nodes[i + 1],
                            **edge_data,
                        })

                    # 過濾風險分數
                    if total_risk < min_risk_score:
                        continue

                    # 收集節點資料
                    nodes = [
                        {"id": node_id, **self.graph.nodes[node_id]}
                        for node_id in path_nodes
                    ]

                    # 生成描述
                    description = self._generate_path_description(nodes, edges)

                    paths.append(
                        AttackPath(
                            path_id=f"path_{len(paths)}",
                            nodes=nodes,
                            edges=edges,
                            total_risk_score=total_risk,
                            length=len(path_nodes) - 1,
                            description=description,
                        )
                    )

                    # 達到限制就停止
                    if len(paths) >= limit:
                        break

            except nx.NetworkXNoPath:
                continue

            if len(paths) >= limit:
                break

        # 依風險分數排序
        paths.sort(key=lambda p: (-p.total_risk_score, p.length))
        paths = paths[:limit]

        logger.info(f"Found {len(paths)} attack paths to {target_node_type}")
        return paths

    def find_critical_nodes(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        尋找圖中的關鍵節點（高中心性節點）

        Args:
            limit: 回傳數量限制

        Returns:
            關鍵節點列表
        """
        # 過濾出資產和漏洞節點
        relevant_nodes = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") in ["Asset", "Vulnerability"]
        ]

        if not relevant_nodes:
            return []

        # 計算度中心性 (degree centrality)
        nodes_with_degree = []
        for node_id in relevant_nodes:
            degree = self.graph.degree(node_id)
            node_data = dict(self.graph.nodes[node_id])
            node_data["id"] = node_id
            node_data["degree"] = degree
            nodes_with_degree.append(node_data)

        # 排序並限制數量
        nodes_with_degree.sort(key=lambda x: x["degree"], reverse=True)
        return nodes_with_degree[:limit]

    def get_vulnerability_statistics(self) -> dict[str, Any]:
        """取得漏洞統計資訊"""
        stats: dict[str, Any] = {"total": 0, "by_severity": {}}

        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "Vulnerability":
                severity = data.get("severity", "UNKNOWN")
                stats["by_severity"][severity] = (
                    stats["by_severity"].get(severity, 0) + 1
                )
                stats["total"] += 1

        return stats

    def _calculate_risk_score(self, finding: FindingPayload) -> float:
        """
        計算風險分數

        Args:
            finding: Finding 物件

        Returns:
            風險分數 (0.0 - 10.0)
        """
        severity_scores = {
            Severity.CRITICAL.value: 10.0,
            Severity.HIGH.value: 7.5,
            Severity.MEDIUM.value: 5.0,
            Severity.LOW.value: 2.5,
            Severity.INFORMATIONAL.value: 1.0,
        }

        confidence_multiplier = {
            "CERTAIN": 1.0,
            "FIRM": 0.8,
            "TENTATIVE": 0.5,
        }

        base_score = severity_scores.get(finding.vulnerability.severity.value, 5.0)
        multiplier = confidence_multiplier.get(
            finding.vulnerability.confidence.value, 0.8
        )

        return base_score * multiplier

    def _generate_path_description(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> str:
        """生成攻擊路徑描述"""
        steps = []
        for i, node in enumerate(nodes):
            node_type = node.get("node_type", "Unknown")
            node_name = node.get("name", node.get("value", node.get("id", "Unknown")))

            if i == 0:
                steps.append(f"起點: {node_name} ({node_type})")
            elif i == len(nodes) - 1:
                steps.append(f"目標: {node_name} ({node_type})")
            else:
                edge = edges[i - 1] if i - 1 < len(edges) else {}
                edge_type = edge.get("edge_type", "UNKNOWN")
                steps.append(f"→ [{edge_type}] → {node_name} ({node_type})")

        return " ".join(steps)

    def clear_graph(self) -> None:
        """清空圖（危險操作,僅用於測試）"""
        self.graph.clear()
        self.initialize_graph()
        logger.warning("Graph cleared!")


# 使用範例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 從配置檔案取得標準化路徑
    from services.integration.aiva_integration.config import ATTACK_GRAPH_FILE

    # 建立引擎 (使用標準化路徑)
    engine = AttackPathEngine(
        graph_file=ATTACK_GRAPH_FILE,
    )

    try:
        # 尋找攻擊路徑
        paths = engine.find_attack_paths(target_node_type="Database")

        # 輸出結果
        for path in paths:
            print(f"\n攻擊路徑 {path.path_id}:")
            print(f"  風險分數: {path.total_risk_score:.2f}")
            print(f"  路徑長度: {path.length}")
            print(f"  描述: {path.description}")

        # 尋找關鍵節點
        critical_nodes = engine.find_critical_nodes()
        print("\n關鍵節點:")
        for node in critical_nodes:
            print(f"  - {node.get('name', node.get('id'))}: Degree={node['degree']}")

        # 漏洞統計
        stats = engine.get_vulnerability_statistics()
        print(f"\n漏洞統計: 總計 {stats['total']} 個")
        for severity, count in stats["by_severity"].items():
            print(f"  {severity}: {count}")

    finally:
        engine.close()  # 自動儲存圖
