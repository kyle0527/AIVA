"""
AIVA Skill Graph Analyzer - 技能圖分析器

基於圖論和語義分析實現的技能圖管理系統，提供技能關係分析、
能力評估和知識圖譜構建功能。參考現代圖分析架構設計。

核心功能:
- 技能節點和關係管理
- 技能路徑分析和推薦
- 能力依賴圖構建
- 學習路徑優化
- 技能匹配和評估

架構設計:
- Graph Theory: 基於圖論的技能關係建模
- Strategy Pattern: 多種分析策略
- Observer Pattern: 技能變化監控
- Command Pattern: 技能操作封裝

技術棧:
- NetworkX: 圖結構處理
- NumPy: 數值計算
- Pydantic v2: 數據模型驗證
- Asyncio: 異步處理

符合標準:
- AIVA Common 設計規範
- 圖分析最佳實踐
- 現代化 Python 架構
- 可插拔設計模式
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import networkx as nx
    import numpy as np

    _has_networkx = True
except ImportError:
    _has_networkx = False
    logging.warning("NetworkX not available, using fallback implementation")


from .interfaces import ISkillGraphAnalyzer

# ============================================================================
# Configuration and Enums (配置和枚舉)
# ============================================================================


class SkillLevel(str, Enum):
    """技能水平枚舉"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class RelationType(str, Enum):
    """關係類型枚舉"""

    PREREQUISITE = "prerequisite"  # 前置條件
    DEPENDENCY = "dependency"  # 依賴關係
    COMPLEMENT = "complement"  # 互補關係
    ALTERNATIVE = "alternative"  # 替代關係
    ENHANCEMENT = "enhancement"  # 增強關係


class AnalysisStrategy(str, Enum):
    """分析策略枚舉"""

    SHORTEST_PATH = "shortest_path"
    WEIGHTED_PATH = "weighted_path"
    CENTRALITY_BASED = "centrality_based"
    CLUSTERING_BASED = "clustering_based"


@dataclass
class SkillNode:
    """技能節點數據類"""

    skill_id: str
    name: str
    description: str = ""
    category: str = "general"
    level: SkillLevel = SkillLevel.BEGINNER
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRelation:
    """技能關係數據類"""

    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillPath:
    """技能路徑數據類"""

    skill_ids: list[str]
    total_weight: float
    path_length: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """分析配置"""

    strategy: AnalysisStrategy = AnalysisStrategy.WEIGHTED_PATH
    max_path_length: int = 10
    min_confidence: float = 0.3
    weight_decay: float = 0.9
    clustering_threshold: float = 0.7
    centrality_weight: float = 0.5


# ============================================================================
# Graph Management Components (圖管理組件)
# ============================================================================


class SkillGraph:
    """技能圖管理器"""

    def __init__(self):
        self.nodes: dict[str, SkillNode] = {}
        self.relations: list[SkillRelation] = []
        self.graph = None
        self._initialize_graph()

    def _initialize_graph(self):
        """初始化圖結構"""
        if _has_networkx:
            self.graph = nx.DiGraph()
        else:
            # 降級實現 - 使用鄰接表
            self.graph = {
                "nodes": {},
                "edges": defaultdict(list),
                "reverse_edges": defaultdict(list),
            }

    def add_skill(self, skill: SkillNode) -> bool:
        """添加技能節點"""
        try:
            self.nodes[skill.skill_id] = skill

            if _has_networkx:
                self.graph.add_node(
                    skill.skill_id,
                    name=skill.name,
                    level=skill.level.value,
                    weight=skill.weight,
                    category=skill.category,
                    **skill.metadata,
                )
            else:
                self.graph["nodes"][skill.skill_id] = skill

            return True
        except Exception as e:
            logging.error(f"Failed to add skill {skill.skill_id}: {e}")
            return False

    def add_relation(self, relation: SkillRelation) -> bool:
        """添加技能關係"""
        try:
            # 檢查節點是否存在
            if (
                relation.source_id not in self.nodes
                or relation.target_id not in self.nodes
            ):
                return False

            self.relations.append(relation)

            if _has_networkx:
                self.graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relation_type=relation.relation_type.value,
                    strength=relation.strength,
                    **relation.metadata,
                )
            else:
                edge_data = {
                    "target": relation.target_id,
                    "relation_type": relation.relation_type.value,
                    "strength": relation.strength,
                    **relation.metadata,
                }
                self.graph["edges"][relation.source_id].append(edge_data)

                # 反向索引
                reverse_edge_data = {
                    "source": relation.source_id,
                    "relation_type": relation.relation_type.value,
                    "strength": relation.strength,
                    **relation.metadata,
                }
                self.graph["reverse_edges"][relation.target_id].append(
                    reverse_edge_data
                )

            return True
        except Exception as e:
            logging.error(
                f"Failed to add relation {relation.source_id}->{relation.target_id}: {e}"
            )
            return False

    def get_neighbors(
        self, skill_id: str, relation_types: list[RelationType] | None = None
    ) -> list[str]:
        """獲取鄰居節點"""
        if skill_id not in self.nodes:
            return []

        neighbors = []

        if _has_networkx:
            for neighbor in self.graph.neighbors(skill_id):
                edge_data = self.graph[skill_id][neighbor]
                if not relation_types or edge_data.get("relation_type") in [
                    rt.value for rt in relation_types
                ]:
                    neighbors.append(neighbor)
        else:
            for edge in self.graph["edges"][skill_id]:
                if not relation_types or edge["relation_type"] in [
                    rt.value for rt in relation_types
                ]:
                    neighbors.append(edge["target"])

        return neighbors

    def get_prerequisites(self, skill_id: str) -> list[str]:
        """獲取前置技能"""
        prerequisites = []

        if _has_networkx:
            for predecessor in self.graph.predecessors(skill_id):
                edge_data = self.graph[predecessor][skill_id]
                if edge_data.get("relation_type") == RelationType.PREREQUISITE.value:
                    prerequisites.append(predecessor)
        else:
            for edge in self.graph["reverse_edges"][skill_id]:
                if edge["relation_type"] == RelationType.PREREQUISITE.value:
                    prerequisites.append(edge["source"])

        return prerequisites


class PathFinder:
    """路徑查找器 - 策略模式實現"""

    def __init__(self, skill_graph: SkillGraph, config: AnalysisConfig):
        self.skill_graph = skill_graph
        self.config = config

    def find_learning_path(
        self, start_skill: str, target_skill: str
    ) -> SkillPath | None:
        """查找學習路徑"""
        if self.config.strategy == AnalysisStrategy.SHORTEST_PATH:
            return self._find_shortest_path(start_skill, target_skill)
        elif self.config.strategy == AnalysisStrategy.WEIGHTED_PATH:
            return self._find_weighted_path(start_skill, target_skill)
        else:
            return self._find_centrality_based_path(start_skill, target_skill)

    def _find_shortest_path(self, start: str, target: str) -> SkillPath | None:
        """查找最短路徑"""
        try:
            if _has_networkx:
                path = nx.shortest_path(self.skill_graph.graph, start, target)
                weight = len(path) - 1
            else:
                path = self._bfs_shortest_path(start, target)
                weight = len(path) - 1 if path else float("inf")

            if path:
                return SkillPath(
                    skill_ids=path,
                    total_weight=weight,
                    path_length=len(path),
                    confidence=max(0.1, 1.0 - weight * 0.1),
                    metadata={"strategy": "shortest_path"},
                )
        except Exception as e:
            logging.error(f"Shortest path finding failed: {e}")

        return None

    def _find_weighted_path(self, start: str, target: str) -> SkillPath | None:
        """查找加權最短路徑"""
        try:
            if _has_networkx:
                path = nx.dijkstra_path(
                    self.skill_graph.graph, start, target, weight="strength"
                )
                weight = nx.dijkstra_path_length(
                    self.skill_graph.graph, start, target, weight="strength"
                )
            else:
                path, weight = self._dijkstra_path(start, target)

            if path:
                confidence = max(0.1, 1.0 / (1.0 + weight))
                return SkillPath(
                    skill_ids=path,
                    total_weight=weight,
                    path_length=len(path),
                    confidence=confidence,
                    metadata={"strategy": "weighted_path"},
                )
        except Exception as e:
            logging.error(f"Weighted path finding failed: {e}")

        return None

    def _find_centrality_based_path(
        self, start: str, target: str
    ) -> SkillPath | None:
        """基於中心性的路徑查找"""
        try:
            # 計算節點中心性
            centrality = self._calculate_centrality()

            # 使用中心性指導路徑查找
            path = self._centrality_guided_search(start, target, centrality)

            if path:
                weight = len(path) - 1
                confidence = max(0.1, centrality.get(target, 0.1))

                return SkillPath(
                    skill_ids=path,
                    total_weight=weight,
                    path_length=len(path),
                    confidence=confidence,
                    metadata={"strategy": "centrality_based", "centrality": centrality},
                )
        except Exception as e:
            logging.error(f"Centrality-based path finding failed: {e}")

        return None

    def _bfs_shortest_path(self, start: str, target: str) -> list[str] | None:
        """BFS 最短路徑 (降級實現)"""
        if start == target:
            return [start]

        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if current in visited:
                continue

            visited.add(current)

            for edge in self.skill_graph.graph["edges"][current]:
                neighbor = edge["target"]

                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _dijkstra_path(
        self, start: str, target: str
    ) -> tuple[list[str] | None, float]:
        """Dijkstra 算法 (降級實現)"""
        distances = {start: 0}
        previous = {}
        unvisited = set(self.skill_graph.nodes.keys())

        while unvisited:
            current = min(unvisited, key=lambda x: distances.get(x, float("inf")))

            if distances.get(current, float("inf")) == float("inf"):
                break

            if current == target:
                # 重建路徑
                path = []
                while current is not None:
                    path.insert(0, current)
                    current = previous.get(current)
                return path, distances[target]

            unvisited.remove(current)

            for edge in self.skill_graph.graph["edges"][current]:
                neighbor = edge["target"]
                weight = 1.0 / edge.get("strength", 1.0)  # 反轉強度作為距離

                distance = distances[current] + weight

                if distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = distance
                    previous[neighbor] = current

        return None, float("inf")

    def _calculate_centrality(self) -> dict[str, float]:
        """計算節點中心性"""
        centrality = {}

        if _has_networkx:
            try:
                # 使用多種中心性測量的組合
                degree_centrality = nx.degree_centrality(self.skill_graph.graph)
                betweenness_centrality = nx.betweenness_centrality(
                    self.skill_graph.graph
                )

                for node in self.skill_graph.nodes:
                    centrality[node] = (
                        degree_centrality.get(node, 0) * 0.6
                        + betweenness_centrality.get(node, 0) * 0.4
                    )
            except Exception:
                # 降級到度中心性
                for node in self.skill_graph.nodes:
                    degree = len(self.skill_graph.graph["edges"][node])
                    centrality[node] = degree / max(1, len(self.skill_graph.nodes) - 1)
        else:
            # 簡單度中心性計算
            for node in self.skill_graph.nodes:
                degree = len(self.skill_graph.graph["edges"][node])
                centrality[node] = degree / max(1, len(self.skill_graph.nodes) - 1)

        return centrality

    def _centrality_guided_search(
        self, start: str, target: str, centrality: dict[str, float]
    ) -> list[str] | None:
        """基於中心性的引導搜索"""
        visited = set()
        queue = deque([(start, [start], 0)])

        while queue:
            current, path, depth = queue.popleft()

            if current == target:
                return path

            if current in visited or depth >= self.config.max_path_length:
                continue

            visited.add(current)

            # 根據中心性排序鄰居
            neighbors = []
            for edge in self.skill_graph.graph["edges"][current]:
                neighbor = edge["target"]
                if neighbor not in visited:
                    neighbor_centrality = centrality.get(neighbor, 0)
                    neighbors.append((neighbor, neighbor_centrality))

            # 按中心性排序 (降序)
            neighbors.sort(key=lambda x: x[1], reverse=True)

            for neighbor, _ in neighbors:
                queue.append((neighbor, path + [neighbor], depth + 1))

        return None


class SkillClustering:
    """技能聚類分析器"""

    def __init__(self, skill_graph: SkillGraph, config: AnalysisConfig):
        self.skill_graph = skill_graph
        self.config = config

    def find_skill_clusters(self) -> dict[str, list[str]]:
        """查找技能聚類"""
        try:
            if _has_networkx:
                return self._networkx_clustering()
            else:
                return self._simple_clustering()
        except Exception as e:
            logging.error(f"Skill clustering failed: {e}")
            return {}

    def _networkx_clustering(self) -> dict[str, list[str]]:
        """使用 NetworkX 的聚類算法"""
        try:
            # 轉換為無向圖進行社區檢測
            undirected_graph = self.skill_graph.graph.to_undirected()

            # 使用社區檢測算法
            import networkx.algorithms.community as nx_comm

            communities = nx_comm.greedy_modularity_communities(undirected_graph)

            clusters = {}
            for i, community in enumerate(communities):
                cluster_name = f"cluster_{i}"
                clusters[cluster_name] = list(community)

            return clusters
        except Exception as e:
            logging.warning(f"NetworkX clustering failed: {e}, using fallback")
            return self._simple_clustering()

    def _simple_clustering(self) -> dict[str, list[str]]:
        """簡單聚類算法 (降級實現)"""
        clusters = {}
        visited = set()
        cluster_id = 0

        for skill_id in self.skill_graph.nodes:
            if skill_id not in visited:
                cluster = self._expand_cluster(skill_id, visited)
                if cluster:
                    clusters[f"cluster_{cluster_id}"] = cluster
                    cluster_id += 1

        return clusters

    def _expand_cluster(self, seed_skill: str, visited: set) -> list[str]:
        """擴展聚類"""
        cluster = []
        queue = deque([seed_skill])

        while queue:
            current = queue.popleft()

            if current in visited:
                continue

            visited.add(current)
            cluster.append(current)

            # 添加相似的鄰居
            for edge in self.skill_graph.graph["edges"][current]:
                neighbor = edge["target"]
                if (
                    neighbor not in visited
                    and edge.get("strength", 0) >= self.config.clustering_threshold
                ):
                    queue.append(neighbor)

        return cluster


# ============================================================================
# Main Skill Graph Analyzer (主要技能圖分析器)
# ============================================================================


class AIVASkillGraphAnalyzer(ISkillGraphAnalyzer):
    """
    AIVA Skill Graph Analyzer - 技能圖分析器

    基於圖論和現代分析架構實現，提供全面的技能關係分析和路徑推薦能力。
    採用策略模式支持多種分析算法。
    """

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.skill_graph = SkillGraph()
        self.path_finder = PathFinder(self.skill_graph, self.config)
        self.clustering = SkillClustering(self.skill_graph, self.config)
        self.is_initialized = False

        # 日誌設置
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """初始化分析器"""
        try:
            self.is_initialized = True
            self.logger.info("Skill Graph Analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Analyzer initialization failed: {e}")
            return False

    async def add_skill(self, skill: SkillNode) -> bool:
        """添加技能節點"""
        return self.skill_graph.add_skill(skill)

    async def add_skill_relation(self, relation: SkillRelation) -> bool:
        """添加技能關係"""
        return self.skill_graph.add_relation(relation)

    async def analyze_skill_path(
        self,
        start_skill: str,
        target_skill: str,
        strategy: AnalysisStrategy | None = None,
    ) -> SkillPath | None:
        """分析技能學習路徑"""
        try:
            if not self.is_initialized:
                await self.initialize()

            # 更新分析策略
            if strategy:
                original_strategy = self.config.strategy
                self.config.strategy = strategy
                self.path_finder.config.strategy = strategy

            # 查找路徑
            path = self.path_finder.find_learning_path(start_skill, target_skill)

            # 恢復原始策略
            if strategy:
                self.config.strategy = original_strategy
                self.path_finder.config.strategy = original_strategy

            return path

        except Exception as e:
            self.logger.error(f"Skill path analysis failed: {e}")
            return None

    async def get_skill_prerequisites(self, skill_id: str) -> list[str]:
        """獲取技能前置條件"""
        try:
            return self.skill_graph.get_prerequisites(skill_id)
        except Exception as e:
            self.logger.error(f"Failed to get prerequisites for {skill_id}: {e}")
            return []

    async def get_related_skills(
        self,
        skill_id: str,
        relation_types: list[RelationType] | None = None,
        max_distance: int = 2,
    ) -> list[dict[str, Any]]:
        """獲取相關技能"""
        try:
            related_skills = []
            visited = set()
            queue = deque([(skill_id, 0)])

            while queue:
                current_skill, distance = queue.popleft()

                if current_skill in visited or distance > max_distance:
                    continue

                visited.add(current_skill)

                if distance > 0:  # 不包含自己
                    skill_info = {
                        "skill_id": current_skill,
                        "skill": self.skill_graph.nodes.get(current_skill),
                        "distance": distance,
                        "relation_path": [],  # 簡化實現
                    }
                    related_skills.append(skill_info)

                # 獲取鄰居
                neighbors = self.skill_graph.get_neighbors(
                    current_skill, relation_types
                )
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

            return related_skills[:20]  # 限制返回數量

        except Exception as e:
            self.logger.error(f"Failed to get related skills for {skill_id}: {e}")
            return []

    async def analyze_skill_clusters(self) -> dict[str, list[str]]:
        """分析技能聚類"""
        try:
            return self.clustering.find_skill_clusters()
        except Exception as e:
            self.logger.error(f"Skill clustering analysis failed: {e}")
            return {}

    async def calculate_skill_importance(self, skill_id: str) -> float:
        """計算技能重要性"""
        try:
            if skill_id not in self.skill_graph.nodes:
                return 0.0

            # 基於多個因素計算重要性

            # 1. 度中心性 (連接數)
            in_degree = len(self.skill_graph.graph["reverse_edges"][skill_id])
            out_degree = len(self.skill_graph.graph["edges"][skill_id])
            degree_score = (in_degree + out_degree) / max(
                1, len(self.skill_graph.nodes)
            )

            # 2. 前置技能數量 (作為其他技能前置的數量)
            prerequisite_count = sum(
                1
                for edge in self.skill_graph.graph["edges"][skill_id]
                if edge.get("relation_type") == RelationType.PREREQUISITE.value
            )
            prerequisite_score = prerequisite_count / max(
                1, len(self.skill_graph.nodes)
            )

            # 3. 技能本身的權重
            skill_weight = self.skill_graph.nodes[skill_id].weight
            weight_score = min(1.0, skill_weight / 10.0)  # 標準化到 [0,1]

            # 綜合評分
            importance = (
                degree_score * 0.4 + prerequisite_score * 0.4 + weight_score * 0.2
            )

            return min(1.0, importance)

        except Exception as e:
            self.logger.error(f"Failed to calculate importance for {skill_id}: {e}")
            return 0.0

    async def recommend_learning_sequence(
        self, target_skills: list[str], current_skills: list[str] | None = None
    ) -> list[SkillPath]:
        """推薦學習序列"""
        try:
            current_skills = current_skills or []
            learning_paths = []

            # 為每個目標技能找到最佳路徑
            for target in target_skills:
                best_path = None
                best_score = -1

                # 從當前技能中找最佳起點
                if current_skills:
                    for current in current_skills:
                        path = await self.analyze_skill_path(current, target)
                        if path and path.confidence > best_score:
                            best_path = path
                            best_score = path.confidence
                else:
                    # 如果沒有當前技能，從所有可能的起點找
                    prerequisites = await self.get_skill_prerequisites(target)
                    if prerequisites:
                        for prereq in prerequisites:
                            path = await self.analyze_skill_path(prereq, target)
                            if path and path.confidence > best_score:
                                best_path = path
                                best_score = path.confidence

                if best_path:
                    learning_paths.append(best_path)

            # 按置信度排序
            learning_paths.sort(key=lambda x: x.confidence, reverse=True)

            return learning_paths

        except Exception as e:
            self.logger.error(f"Learning sequence recommendation failed: {e}")
            return []

    async def get_graph_statistics(self) -> dict[str, Any]:
        """獲取圖統計信息"""
        try:
            stats = {
                "total_skills": len(self.skill_graph.nodes),
                "total_relations": len(self.skill_graph.relations),
                "skill_categories": {},
                "relation_types": {},
                "average_connections": 0,
                "graph_density": 0,
                "has_networkx": _has_networkx,
                "is_initialized": self.is_initialized,
            }

            # 技能分類統計
            for skill in self.skill_graph.nodes.values():
                category = skill.category
                stats["skill_categories"][category] = (
                    stats["skill_categories"].get(category, 0) + 1
                )

            # 關係類型統計
            for relation in self.skill_graph.relations:
                rel_type = relation.relation_type.value
                stats["relation_types"][rel_type] = (
                    stats["relation_types"].get(rel_type, 0) + 1
                )

            # 平均連接數
            if stats["total_skills"] > 0:
                total_connections = sum(
                    len(self.skill_graph.graph["edges"][skill_id])
                    for skill_id in self.skill_graph.nodes
                )
                stats["average_connections"] = total_connections / stats["total_skills"]

                # 圖密度
                max_possible_edges = stats["total_skills"] * (stats["total_skills"] - 1)
                if max_possible_edges > 0:
                    stats["graph_density"] = (
                        stats["total_relations"] / max_possible_edges
                    )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get graph statistics: {e}")
            return {}

    async def export_graph(self, format: str = "json") -> str | None:
        """導出圖數據"""
        try:
            if format.lower() == "json":
                graph_data = {
                    "nodes": [
                        {
                            "skill_id": skill.skill_id,
                            "name": skill.name,
                            "description": skill.description,
                            "category": skill.category,
                            "level": skill.level.value,
                            "weight": skill.weight,
                            "metadata": skill.metadata,
                        }
                        for skill in self.skill_graph.nodes.values()
                    ],
                    "relations": [
                        {
                            "source_id": rel.source_id,
                            "target_id": rel.target_id,
                            "relation_type": rel.relation_type.value,
                            "strength": rel.strength,
                            "metadata": rel.metadata,
                        }
                        for rel in self.skill_graph.relations
                    ],
                }
                return json.dumps(graph_data, ensure_ascii=False, indent=2)

            return None

        except Exception as e:
            self.logger.error(f"Graph export failed: {e}")
            return None

    # ============================================================================
    # ISkillGraphAnalyzer Interface Implementation (介面方法實現)
    # ============================================================================

    async def build_capability_graph(
        self, capabilities: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        建構能力圖 (實現 ISkillGraphAnalyzer 介面)

        Args:
            capabilities: 能力列表

        Returns:
            能力圖結構 (節點和邊的信息)
        """
        try:
            # 清空現有圖
            self.skill_graph = SkillGraph()

            # 添加能力節點
            for capability in capabilities:
                skill_node = SkillNode(
                    skill_id=capability.get(
                        "id", f"skill_{len(self.skill_graph.nodes)}"
                    ),
                    name=capability.get("name", "Unknown Skill"),
                    description=capability.get("description", ""),
                    category=capability.get("category", "general"),
                    level=SkillLevel(capability.get("level", "beginner")),
                    weight=capability.get("weight", 1.0),
                    metadata=capability.get("metadata", {}),
                )
                await self.add_skill(skill_node)

            # 根據能力關係添加邊
            for capability in capabilities:
                skill_id = capability.get("id")
                if not skill_id:
                    continue

                # 處理前置條件
                prerequisites = capability.get("prerequisites", [])
                for prereq in prerequisites:
                    relation = SkillRelation(
                        source_id=prereq,
                        target_id=skill_id,
                        relation_type=RelationType.PREREQUISITE,
                        strength=capability.get("prereq_strength", 0.8),
                        metadata={"auto_generated": True},
                    )
                    await self.add_skill_relation(relation)

                # 處理依賴關係
                dependencies = capability.get("dependencies", [])
                for dep in dependencies:
                    relation = SkillRelation(
                        source_id=skill_id,
                        target_id=dep,
                        relation_type=RelationType.DEPENDENCY,
                        strength=capability.get("dep_strength", 0.7),
                        metadata={"auto_generated": True},
                    )
                    await self.add_skill_relation(relation)

            # 重新初始化路徑查找器和聚類
            self.path_finder = PathFinder(self.skill_graph, self.config)
            self.clustering = SkillClustering(self.skill_graph, self.config)

            # 構建返回的圖結構
            graph_structure = {
                "nodes": [
                    {
                        "id": skill.skill_id,
                        "name": skill.name,
                        "level": skill.level.value,
                        "category": skill.category,
                        "weight": skill.weight,
                        "metadata": skill.metadata,
                    }
                    for skill in self.skill_graph.nodes.values()
                ],
                "edges": [
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "type": rel.relation_type.value,
                        "strength": rel.strength,
                        "metadata": rel.metadata,
                    }
                    for rel in self.skill_graph.relations
                ],
                "statistics": await self.get_graph_statistics(),
                "clusters": await self.analyze_skill_clusters(),
            }

            self.logger.info(
                f"Built capability graph with {len(capabilities)} capabilities"
            )
            return graph_structure

        except Exception as e:
            self.logger.error(f"Failed to build capability graph: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}

    async def get_capability_recommendations(
        self, current_capabilities: list[str], target_scenario: str
    ) -> list[dict[str, Any]]:
        """
        獲取能力建議 (實現 ISkillGraphAnalyzer 介面)

        Args:
            current_capabilities: 當前能力列表
            target_scenario: 目標場景

        Returns:
            建議的能力和優先級
        """
        try:
            recommendations = []

            # 分析當前能力的重要性
            current_importance = {}
            for capability in current_capabilities:
                if capability in self.skill_graph.nodes:
                    importance = await self.calculate_skill_importance(capability)
                    current_importance[capability] = importance

            # 根據目標場景推薦相關技能
            scenario_keywords = target_scenario.lower().split()

            for skill_id, skill in self.skill_graph.nodes.items():
                if skill_id in current_capabilities:
                    continue  # 跳過已有技能

                # 計算與目標場景的相關性
                relevance_score = 0.0
                skill_text = (
                    f"{skill.name} {skill.description} {skill.category}".lower()
                )

                for keyword in scenario_keywords:
                    if keyword in skill_text:
                        relevance_score += 0.2

                # 計算技能重要性
                importance = await self.calculate_skill_importance(skill_id)

                # 檢查是否有當前技能作為前置條件
                prerequisites = await self.get_skill_prerequisites(skill_id)
                prerequisite_satisfaction = 0.0
                if prerequisites:
                    satisfied_prereqs = sum(
                        1 for prereq in prerequisites if prereq in current_capabilities
                    )
                    prerequisite_satisfaction = satisfied_prereqs / len(prerequisites)

                # 計算學習路徑的復雜度
                path_complexity = 1.0
                if current_capabilities:
                    # 找到最短路徑
                    best_path = None
                    best_confidence = 0.0

                    for current_skill in current_capabilities:
                        path = await self.analyze_skill_path(current_skill, skill_id)
                        if path and path.confidence > best_confidence:
                            best_path = path
                            best_confidence = path.confidence

                    if best_path:
                        path_complexity = 1.0 / max(0.1, best_path.confidence)

                # 綜合評分
                final_score = (
                    relevance_score * 0.3
                    + importance * 0.25
                    + prerequisite_satisfaction * 0.25
                    + (1.0 / path_complexity) * 0.2
                )

                if final_score > 0.1:  # 過濾低分推薦
                    recommendation = {
                        "skill_id": skill_id,
                        "skill_name": skill.name,
                        "description": skill.description,
                        "category": skill.category,
                        "level": skill.level.value,
                        "recommendation_score": final_score,
                        "relevance_to_scenario": relevance_score,
                        "skill_importance": importance,
                        "prerequisite_satisfaction": prerequisite_satisfaction,
                        "learning_complexity": path_complexity,
                        "prerequisites": prerequisites,
                        "estimated_learning_path": (
                            best_path.skill_ids
                            if "best_path" in locals() and best_path
                            else []
                        ),
                        "metadata": {
                            "target_scenario": target_scenario,
                            "current_capabilities_count": len(current_capabilities),
                        },
                    }
                    recommendations.append(recommendation)

            # 按推薦分數排序
            recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)

            # 限制返回數量並添加優先級
            top_recommendations = recommendations[:10]
            for i, rec in enumerate(top_recommendations):
                rec["priority"] = i + 1
                rec["priority_level"] = (
                    "high" if i < 3 else "medium" if i < 7 else "low"
                )

            self.logger.info(
                f"Generated {len(top_recommendations)} capability recommendations for scenario: {target_scenario}"
            )
            return top_recommendations

        except Exception as e:
            self.logger.error(f"Failed to get capability recommendations: {e}")
            return []


# ============================================================================
# Factory Functions (工廠函數)
# ============================================================================


def create_skill_graph_analyzer(
    config: AnalysisConfig | None = None,
    strategy: AnalysisStrategy = AnalysisStrategy.WEIGHTED_PATH,
) -> AIVASkillGraphAnalyzer:
    """創建技能圖分析器實例 - 工廠函數"""
    if config is None:
        config = AnalysisConfig(strategy=strategy)

    return AIVASkillGraphAnalyzer(config)


def create_analysis_config(**kwargs) -> AnalysisConfig:
    """創建分析配置 - 工廠函數"""
    return AnalysisConfig(**kwargs)


def create_skill_node(skill_id: str, name: str, **kwargs) -> SkillNode:
    """創建技能節點 - 工廠函數"""
    return SkillNode(skill_id=skill_id, name=name, **kwargs)


def create_skill_relation(
    source_id: str, target_id: str, relation_type: RelationType, **kwargs
) -> SkillRelation:
    """創建技能關係 - 工廠函數"""
    return SkillRelation(
        source_id=source_id, target_id=target_id, relation_type=relation_type, **kwargs
    )


# ============================================================================
# Module Exports (模組導出)
# ============================================================================

__all__ = [
    "AIVASkillGraphAnalyzer",
    "SkillNode",
    "SkillRelation",
    "SkillPath",
    "SkillLevel",
    "RelationType",
    "AnalysisStrategy",
    "AnalysisConfig",
    "create_skill_graph_analyzer",
    "create_analysis_config",
    "create_skill_node",
    "create_skill_relation",
]


# ============================================================================
# Usage Example (使用示例)
# ============================================================================

if __name__ == "__main__":

    async def main():
        """Skill Graph Analyzer 使用示例"""
        # 創建分析器
        analyzer = create_skill_graph_analyzer()

        # 初始化
        await analyzer.initialize()

        # 創建技能節點
        python_basics = create_skill_node(
            "python_basics",
            "Python 基礎",
            category="programming",
            level=SkillLevel.BEGINNER,
        )

        data_structures = create_skill_node(
            "data_structures",
            "數據結構",
            category="computer_science",
            level=SkillLevel.INTERMEDIATE,
        )

        machine_learning = create_skill_node(
            "machine_learning", "機器學習", category="ai", level=SkillLevel.ADVANCED
        )

        # 添加技能
        await analyzer.add_skill(python_basics)
        await analyzer.add_skill(data_structures)
        await analyzer.add_skill(machine_learning)

        # 創建關係
        prereq_relation = create_skill_relation(
            "python_basics", "data_structures", RelationType.PREREQUISITE, strength=0.9
        )

        prereq_relation2 = create_skill_relation(
            "data_structures",
            "machine_learning",
            RelationType.PREREQUISITE,
            strength=0.8,
        )

        # 添加關係
        await analyzer.add_skill_relation(prereq_relation)
        await analyzer.add_skill_relation(prereq_relation2)

        # 分析學習路徑
        path = await analyzer.analyze_skill_path("python_basics", "machine_learning")
        if path:
            print(f"Learning Path: {' -> '.join(path.skill_ids)}")
            print(f"Confidence: {path.confidence}")

        # 獲取統計信息
        stats = await analyzer.get_graph_statistics()
        print(f"Graph Statistics: {stats}")

        # 導出圖數據
        graph_json = await analyzer.export_graph("json")
        if graph_json:
            print("Graph exported successfully")

    # 運行示例
    asyncio.run(main())
