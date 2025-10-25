"""
Attack Path Analyzer - 攻擊路徑分析引擎

使用 Neo4j 圖資料庫建立資產與漏洞的關聯圖,
計算從外部攻擊者到核心資產的攻擊路徑。

Compliance Note (遵循 aiva_common 設計原則):
- NodeType, EdgeType 已移除,改為從 aiva_common.enums.security import (4-layer priority 原則)
- AttackPathNodeType → 節點類型枚舉
- AttackPathEdgeType → 邊類型枚舉
- 修正日期: 2025-10-25
"""

from dataclasses import dataclass
import logging
from typing import Any

from neo4j import GraphDatabase

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
    """攻擊路徑分析引擎"""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
    ):
        """
        初始化引擎

        Args:
            neo4j_uri: Neo4j 連線 URI
            neo4j_user: Neo4j 使用者名稱
            neo4j_password: Neo4j 密碼
        """
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )
        logger.info(f"Connected to Neo4j at {neo4j_uri}")

    def close(self):
        """關閉連線"""
        self.driver.close()

    def initialize_graph(self):
        """初始化圖結構（建立索引和約束）"""
        with self.driver.session() as session:
            # 建立唯一性約束
            session.run(
                "CREATE CONSTRAINT asset_id IF NOT EXISTS "
                "FOR (a:Asset) REQUIRE a.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT vuln_id IF NOT EXISTS "
                "FOR (v:Vulnerability) REQUIRE v.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT cred_id IF NOT EXISTS "
                "FOR (c:Credential) REQUIRE c.id IS UNIQUE"
            )

            # 建立索引以加速查詢
            session.run(
                "CREATE INDEX asset_type IF NOT EXISTS " "FOR (a:Asset) ON (a.type)"
            )
            session.run(
                "CREATE INDEX vuln_severity IF NOT EXISTS "
                "FOR (v:Vulnerability) ON (v.severity)"
            )

            # 建立攻擊者節點（外部攻擊者）
            session.run(
                """
                MERGE (attacker:Attacker {id: 'external_attacker'})
                SET attacker.name = 'External Attacker',
                    attacker.description = '外部攻擊者'
                """
            )

            logger.info("Graph structure initialized")

    def add_asset(self, asset: Asset) -> None:
        """
        新增資產節點

        Args:
            asset: 資產物件
        """
        with self.driver.session() as session:
            session.run(
                """
                MERGE (a:Asset {id: $asset_id})
                SET a.value = $value,
                    a.type = $type,
                    a.is_public = $is_public
                """,
                asset_id=asset.asset_id,
                value=asset.value,
                type=asset.type,
                is_public=True,  # 假設從外部掃描發現的都是公開資產
            )

            # 如果是公開資產,連接到外部攻擊者
            session.run(
                """
                MATCH (attacker:Attacker {id: 'external_attacker'})
                MATCH (asset:Asset {id: $asset_id})
                WHERE asset.is_public = true
                MERGE (attacker)-[:CAN_ACCESS]->(asset)
                """,
                asset_id=asset.asset_id,
            )

    def add_finding(self, finding: FindingPayload) -> None:
        """
        新增漏洞發現,建立資產與漏洞的關聯

        Args:
            finding: Finding 物件
        """
        with self.driver.session() as session:
            # 建立漏洞節點
            session.run(
                """
                MERGE (v:Vulnerability {id: $finding_id})
                SET v.name = $vuln_name,
                    v.severity = $severity,
                    v.confidence = $confidence,
                    v.cwe = $cwe,
                    v.risk_score = $risk_score
                """,
                finding_id=finding.finding_id,
                vuln_name=finding.vulnerability.name.value,
                severity=finding.vulnerability.severity.value,
                confidence=finding.vulnerability.confidence.value,
                cwe=finding.vulnerability.cwe,
                risk_score=self._calculate_risk_score(finding),
            )

            # 建立資產與漏洞的關聯
            # 先嘗試找到資產（透過 URL 匹配）
            target_url = str(finding.target.url)
            session.run(
                """
                MATCH (v:Vulnerability {id: $finding_id})
                MERGE (a:Asset {value: $target_url})
                ON CREATE SET a.id = $asset_id, a.type = 'discovered'
                MERGE (a)-[:HAS_VULNERABILITY]->(v)
                """,
                finding_id=finding.finding_id,
                target_url=target_url,
                asset_id=f"asset_{finding.finding_id}",
            )

            # 根據漏洞類型建立攻擊路徑
            self._create_attack_edges(session, finding)

    def _create_attack_edges(self, session: Any, finding: FindingPayload) -> None:
        """
        根據漏洞類型建立攻擊邊

        Args:
            session: Neo4j session
            finding: Finding 物件
        """
        vuln_name = finding.vulnerability.name.value

        # SSRF -> 內部網路
        if vuln_name == "SSRF":
            session.run(
                """
                MATCH (v:Vulnerability {id: $finding_id})
                MERGE (internal:InternalNetwork {id: 'internal_network'})
                ON CREATE SET internal.name = 'Internal Network'
                MERGE (v)-[:LEADS_TO {risk: $risk}]->(internal)
                """,
                finding_id=finding.finding_id,
                risk=self._calculate_risk_score(finding),
            )

        # SQLi -> 資料庫
        elif vuln_name == "SQLI":
            session.run(
                """
                MATCH (v:Vulnerability {id: $finding_id})
                MERGE (db:Database {id: 'database'})
                ON CREATE SET db.name = 'Application Database'
                MERGE (v)-[:LEADS_TO {risk: $risk}]->(db)
                """,
                finding_id=finding.finding_id,
                risk=self._calculate_risk_score(finding),
            )

        # IDOR/BOLA -> API 端點
        elif vuln_name in ["IDOR", "BOLA"]:
            session.run(
                """
                MATCH (v:Vulnerability {id: $finding_id})
                MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v)
                MERGE (api:APIEndpoint {id: $api_id})
                ON CREATE SET api.value = $url
                MERGE (v)-[:GRANTS_ACCESS {risk: $risk}]->(api)
                """,
                finding_id=finding.finding_id,
                api_id=f"api_{finding.finding_id}",
                url=str(finding.target.url),
                risk=self._calculate_risk_score(finding),
            )

        # XSS -> 憑證洩漏
        elif vuln_name == "XSS":
            session.run(
                """
                MATCH (v:Vulnerability {id: $finding_id})
                MERGE (cred:Credential {id: $cred_id})
                ON CREATE SET cred.type = 'Session Cookie'
                MERGE (v)-[:EXPOSES {risk: $risk}]->(cred)
                """,
                finding_id=finding.finding_id,
                cred_id=f"cred_{finding.finding_id}",
                risk=self._calculate_risk_score(finding),
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
        with self.driver.session() as session:
            query_str = f"""
                MATCH path = (attacker:Attacker {{id: 'external_attacker'}})
                             -[*1..{max_length}]->(target:{target_node_type})
                WITH path,
                     reduce(risk = 0.0, r in relationships(path) |
                            risk + coalesce(r.risk, 1.0)) as total_risk
                WHERE total_risk >= $min_risk_score
                RETURN path, total_risk, length(path) as path_length
                ORDER BY total_risk DESC, path_length ASC
                LIMIT {limit}
                """
            result = session.run(query_str, min_risk_score=min_risk_score)  # type: ignore[arg-type]

            paths = []
            for record in result:
                path_data = record["path"]
                nodes = [dict(node) for node in path_data.nodes]
                edges = [dict(rel) for rel in path_data.relationships]

                # 生成描述
                description = self._generate_path_description(nodes, edges)

                paths.append(
                    AttackPath(
                        path_id=f"path_{len(paths)}",
                        nodes=nodes,
                        edges=edges,
                        total_risk_score=record["total_risk"],
                        length=record["path_length"],
                        description=description,
                    )
                )

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
        with self.driver.session() as session:
            query_str = """
                MATCH (n)
                WHERE n:Asset OR n:Vulnerability
                WITH n,
                     size((n)--()) as degree
                RETURN n, degree
                ORDER BY degree DESC
                LIMIT $limit
                """
            result = session.run(query_str, limit=limit)

            nodes = []
            for record in result:
                node = dict(record["n"])
                node["degree"] = record["degree"]
                nodes.append(node)

            return nodes

    def get_vulnerability_statistics(self) -> dict[str, Any]:
        """取得漏洞統計資訊"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (v:Vulnerability)
                RETURN
                    v.severity as severity,
                    count(*) as count
                ORDER BY count DESC
                """
            )

            stats: dict[str, Any] = {"total": 0, "by_severity": {}}
            for record in result:
                severity = record["severity"]
                count = record["count"]
                stats["by_severity"][severity] = count
                stats["total"] += count

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
            node_labels = node.get("labels", [])
            node_type = node_labels[0] if node_labels else "Unknown"
            node_name = node.get("name", node.get("value", node.get("id", "Unknown")))

            if i == 0:
                steps.append(f"起點: {node_name} ({node_type})")
            elif i == len(nodes) - 1:
                steps.append(f"目標: {node_name} ({node_type})")
            else:
                edge = edges[i - 1] if i - 1 < len(edges) else {}
                edge_type = edge.get("type", "UNKNOWN")
                steps.append(f"→ [{edge_type}] → {node_name} ({node_type})")

        return " ".join(steps)

    def clear_graph(self) -> None:
        """清空圖（危險操作,僅用於測試）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Graph cleared!")


# 使用範例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 建立引擎
    engine = AttackPathEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    try:
        # 初始化圖
        engine.initialize_graph()

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
        engine.close()
