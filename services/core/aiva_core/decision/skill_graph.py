"""
AIVA 技能圖 (Skill Graph) 模組
實現能力關係映射和決策支援
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import networkx as nx

from services.aiva_common.enums import (
    ModuleName,
    Severity,
    TaskStatus,
    ProgrammingLanguage
)
from services.aiva_common.schemas import (
    CapabilityInfo,
    CapabilityScorecard
)
from services.aiva_common.utils.logging import get_logger
from services.integration.capability import CapabilityRegistry

logger = get_logger(__name__)


@dataclass
class SkillNode:
    """技能節點"""
    id: str
    name: str
    language: ProgrammingLanguage
    topic: str
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    avg_latency: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SkillEdge:
    """技能邊關係"""
    source: str
    target: str
    relationship_type: str  # "prerequisite", "alternative", "complement", "sequence"
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillPath:
    """技能執行路徑"""
    nodes: List[str]
    edges: List[SkillEdge]
    total_weight: float
    estimated_time: float
    success_probability: float
    description: str = ""


class SkillGraphBuilder:
    """技能圖構建器"""
    
    def __init__(self, capability_registry: CapabilityRegistry):
        self.capability_registry = capability_registry
        self.graph = nx.DiGraph()
        self.skill_nodes: Dict[str, SkillNode] = {}
        self.skill_edges: List[SkillEdge] = []
        
    async def build_graph(self) -> None:
        """構建技能圖"""
        logger.info("開始構建技能圖...")
        
        # 獲取所有能力
        capabilities = await self.capability_registry.list_capabilities(limit=None)
        
        # 創建節點
        await self._create_skill_nodes(capabilities)
        
        # 分析關係
        await self._analyze_relationships(capabilities)
        
        # 構建 NetworkX 圖
        await self._build_networkx_graph()
        
        logger.info(f"技能圖構建完成: {len(self.skill_nodes)} 節點, {len(self.skill_edges)} 邊")
    
    async def _create_skill_nodes(self, capabilities: List[CapabilityInfo]) -> None:
        """創建技能節點"""
        for cap in capabilities:
            # 獲取性能數據
            scorecard = None
            try:
                scorecard = await self.capability_registry.get_capability_scorecard(cap.id)
            except Exception as e:
                logger.warning(f"無法獲取 {cap.id} 的評分卡: {e}")
            
            node = SkillNode(
                id=cap.id,
                name=cap.name,
                language=cap.language,
                topic=cap.topic,
                tags=cap.tags or [],
                prerequisites=cap.prerequisites or [],
                dependencies=cap.dependencies or [],
                success_rate=scorecard.success_rate_7d if scorecard else 0.0,
                avg_latency=scorecard.avg_latency_ms if scorecard else 0.0,
                last_used=scorecard.last_used_at if scorecard else None,
                usage_count=scorecard.usage_count_7d if scorecard else 0,
                metadata={
                    "description": cap.description,
                    "entrypoint": cap.entrypoint,
                    "inputs": [inp.model_dump() for inp in cap.inputs] if cap.inputs else [],
                    "outputs": [out.model_dump() for out in cap.outputs] if cap.outputs else [],
                    "status": cap.status.value
                }
            )
            
            self.skill_nodes[cap.id] = node
    
    async def _analyze_relationships(self, capabilities: List[CapabilityInfo]) -> None:
        """分析技能間關係"""
        # 1. 前置條件關係
        await self._analyze_prerequisite_relationships()
        
        # 2. 標籤相似性關係
        await self._analyze_tag_similarity_relationships()
        
        # 3. 語言生態關係
        await self._analyze_language_ecosystem_relationships()
        
        # 4. 主題關聯關係
        await self._analyze_topic_relationships()
        
        # 5. 輸入輸出關係
        await self._analyze_io_relationships()
    
    async def _analyze_prerequisite_relationships(self) -> None:
        """分析前置條件關係"""
        for node_id, node in self.skill_nodes.items():
            for prereq in node.prerequisites:
                # 尋找匹配的前置條件能力
                for other_id, other_node in self.skill_nodes.items():
                    if other_id != node_id and prereq.lower() in other_node.name.lower():
                        edge = SkillEdge(
                            source=other_id,
                            target=node_id,
                            relationship_type="prerequisite",
                            weight=0.9,
                            confidence=0.8,
                            metadata={"reason": f"前置條件: {prereq}"}
                        )
                        self.skill_edges.append(edge)
    
    async def _analyze_tag_similarity_relationships(self) -> None:
        """分析標籤相似性關係"""
        nodes_list = list(self.skill_nodes.values())
        
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                if node1.id == node2.id:
                    continue
                
                # 計算標籤交集
                common_tags = set(node1.tags) & set(node2.tags)
                if common_tags:
                    similarity = len(common_tags) / max(len(node1.tags), len(node2.tags), 1)
                    
                    if similarity >= 0.3:  # 相似度閾值
                        edge = SkillEdge(
                            source=node1.id,
                            target=node2.id,
                            relationship_type="alternative",
                            weight=similarity,
                            confidence=0.6,
                            metadata={
                                "reason": f"共同標籤: {', '.join(common_tags)}",
                                "similarity": similarity
                            }
                        )
                        self.skill_edges.append(edge)
    
    async def _analyze_language_ecosystem_relationships(self) -> None:
        """分析語言生態關係"""
        # 按語言分組
        language_groups = defaultdict(list)
        for node in self.skill_nodes.values():
            language_groups[node.language].append(node)
        
        # 在同語言內建立關係
        for language, nodes in language_groups.items():
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    # 同語言、同主題的能力建立補充關係
                    if node1.topic == node2.topic:
                        edge = SkillEdge(
                            source=node1.id,
                            target=node2.id,
                            relationship_type="complement",
                            weight=0.5,
                            confidence=0.4,
                            metadata={
                                "reason": f"同語言同主題: {language.value}/{node1.topic}"
                            }
                        )
                        self.skill_edges.append(edge)
    
    async def _analyze_topic_relationships(self) -> None:
        """分析主題關聯關係"""
        # 定義主題關聯規則
        topic_relationships = {
            "scanning": ["vulnerability", "security", "web"],
            "vulnerability": ["scanning", "exploit", "security"],
            "web": ["scanning", "http", "api"],
            "api": ["web", "rest", "graphql"],
            "database": ["sql", "injection", "security"],
            "network": ["scanning", "port", "service"]
        }
        
        for source_topic, related_topics in topic_relationships.items():
            source_nodes = [n for n in self.skill_nodes.values() if source_topic in n.topic.lower()]
            
            for related_topic in related_topics:
                target_nodes = [n for n in self.skill_nodes.values() if related_topic in n.topic.lower()]
                
                for source_node in source_nodes:
                    for target_node in target_nodes:
                        if source_node.id != target_node.id:
                            edge = SkillEdge(
                                source=source_node.id,
                                target=target_node.id,
                                relationship_type="sequence",
                                weight=0.4,
                                confidence=0.3,
                                metadata={
                                    "reason": f"主題關聯: {source_topic} -> {related_topic}"
                                }
                            )
                            self.skill_edges.append(edge)
    
    async def _analyze_io_relationships(self) -> None:
        """分析輸入輸出關係"""
        for node1 in self.skill_nodes.values():
            for node2 in self.skill_nodes.values():
                if node1.id == node2.id:
                    continue
                
                # 檢查 node1 的輸出是否匹配 node2 的輸入
                node1_outputs = node1.metadata.get("outputs", [])
                node2_inputs = node2.metadata.get("inputs", [])
                
                for output in node1_outputs:
                    for input_param in node2_inputs:
                        if self._is_compatible_io(output, input_param):
                            edge = SkillEdge(
                                source=node1.id,
                                target=node2.id,
                                relationship_type="sequence",
                                weight=0.7,
                                confidence=0.6,
                                metadata={
                                    "reason": f"I/O 匹配: {output.get('name')} -> {input_param.get('name')}"
                                }
                            )
                            self.skill_edges.append(edge)
                            break
    
    def _is_compatible_io(self, output: Dict[str, Any], input_param: Dict[str, Any]) -> bool:
        """檢查輸入輸出兼容性"""
        output_type = output.get("type", "").lower()
        input_type = input_param.get("type", "").lower()
        
        # 類型兼容性規則
        compatibility_rules = {
            "url": ["string", "str", "url"],
            "string": ["str", "text", "url"],
            "json": ["dict", "object", "json"],
            "list": ["array", "list"],
            "report": ["json", "dict", "object"]
        }
        
        if output_type in compatibility_rules:
            return input_type in compatibility_rules[output_type]
        
        return output_type == input_type
    
    async def _build_networkx_graph(self) -> None:
        """構建 NetworkX 圖"""
        # 添加節點
        for node_id, node in self.skill_nodes.items():
            self.graph.add_node(node_id, **node.__dict__)
        
        # 添加邊
        for edge in self.skill_edges:
            self.graph.add_edge(
                edge.source,
                edge.target,
                relationship_type=edge.relationship_type,
                weight=edge.weight,
                confidence=edge.confidence,
                **edge.metadata
            )


class SkillGraphAnalyzer:
    """技能圖分析器"""
    
    def __init__(self, skill_graph: nx.DiGraph, skill_nodes: Dict[str, SkillNode]):
        self.graph = skill_graph
        self.skill_nodes = skill_nodes
    
    def find_optimal_path(
        self, 
        start_capability: str, 
        goal: str,
        max_path_length: int = 5
    ) -> List[SkillPath]:
        """尋找最佳執行路徑"""
        paths = []
        
        try:
            # 尋找相關的目標能力
            target_nodes = self._find_goal_capabilities(goal)
            
            if not target_nodes:
                logger.warning(f"找不到與目標 '{goal}' 相關的能力")
                return paths
            
            for target in target_nodes:
                try:
                    # 使用 NetworkX 尋找最短路徑
                    if nx.has_path(self.graph, start_capability, target):
                        shortest_path = nx.shortest_path(
                            self.graph, 
                            start_capability, 
                            target,
                            weight='weight'
                        )
                        
                        if len(shortest_path) <= max_path_length:
                            path = self._create_skill_path(shortest_path)
                            paths.append(path)
                
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning(f"路徑搜尋錯誤: {e}")
                    continue
            
            # 按成功機率排序
            paths.sort(key=lambda p: p.success_probability, reverse=True)
            
        except Exception as e:
            logger.error(f"尋找最佳路徑時發生錯誤: {e}")
        
        return paths[:3]  # 返回前3個最佳路徑
    
    def _find_goal_capabilities(self, goal: str) -> List[str]:
        """尋找與目標相關的能力"""
        candidates = []
        goal_lower = goal.lower()
        
        for node_id, node in self.skill_nodes.items():
            # 檢查名稱匹配
            if goal_lower in node.name.lower():
                candidates.append((node_id, 1.0))
                continue
            
            # 檢查標籤匹配
            for tag in node.tags:
                if goal_lower in tag.lower():
                    candidates.append((node_id, 0.8))
                    break
            
            # 檢查主題匹配
            if goal_lower in node.topic.lower():
                candidates.append((node_id, 0.6))
        
        # 按匹配度排序並返回
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in candidates[:5]]
    
    def _create_skill_path(self, node_path: List[str]) -> SkillPath:
        """創建技能路徑物件"""
        edges = []
        total_weight = 0.0
        estimated_time = 0.0
        success_probability = 1.0
        
        # 構建邊列表
        for i in range(len(node_path) - 1):
            source, target = node_path[i], node_path[i + 1]
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                edge = SkillEdge(
                    source=source,
                    target=target,
                    relationship_type=edge_data.get('relationship_type', 'unknown'),
                    weight=edge_data.get('weight', 1.0),
                    confidence=edge_data.get('confidence', 1.0),
                    metadata=edge_data
                )
                edges.append(edge)
                total_weight += edge.weight
        
        # 計算預估時間和成功機率
        for node_id in node_path:
            node = self.skill_nodes[node_id]
            estimated_time += node.avg_latency / 1000.0  # 轉換為秒
            success_probability *= max(node.success_rate, 0.1)  # 避免為0
        
        # 生成路徑描述
        description = " -> ".join([self.skill_nodes[node_id].name for node_id in node_path])
        
        return SkillPath(
            nodes=node_path,
            edges=edges,
            total_weight=total_weight,
            estimated_time=estimated_time,
            success_probability=success_probability,
            description=description
        )
    
    def get_capability_recommendations(
        self, 
        capability_id: str, 
        limit: int = 5
    ) -> List[Tuple[str, float, str]]:
        """獲取能力推薦"""
        recommendations = []
        
        try:
            if capability_id not in self.graph:
                return recommendations
            
            # 獲取直接鄰居
            neighbors = list(self.graph.successors(capability_id))
            
            for neighbor in neighbors:
                edge_data = self.graph[capability_id][neighbor]
                score = edge_data.get('weight', 0.0) * edge_data.get('confidence', 1.0)
                reason = edge_data.get('reason', '相關能力')
                
                recommendations.append((neighbor, score, reason))
            
            # 獲取間接相關能力
            try:
                # 使用 PageRank 算法找到重要的相關節點
                pagerank_scores = nx.pagerank(self.graph, personalization={capability_id: 1.0})
                
                for node_id, score in pagerank_scores.items():
                    if node_id != capability_id and node_id not in neighbors:
                        if score > 0.01:  # 分數閾值
                            recommendations.append((node_id, score, "間接相關"))
            
            except Exception as e:
                logger.warning(f"PageRank 計算失敗: {e}")
            
            # 排序並限制數量
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
        
        except Exception as e:
            logger.error(f"獲取推薦時發生錯誤: {e}")
            return recommendations
    
    def analyze_capability_centrality(self) -> Dict[str, Dict[str, float]]:
        """分析能力中心性"""
        centrality_metrics = {}
        
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(self.graph)
            
            # 介數中心性
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # 特徵向量中心性
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                eigenvector_centrality = {}
            
            # PageRank
            pagerank = nx.pagerank(self.graph)
            
            for node_id in self.graph.nodes():
                centrality_metrics[node_id] = {
                    "degree": degree_centrality.get(node_id, 0.0),
                    "betweenness": betweenness_centrality.get(node_id, 0.0),
                    "eigenvector": eigenvector_centrality.get(node_id, 0.0),
                    "pagerank": pagerank.get(node_id, 0.0)
                }
        
        except Exception as e:
            logger.error(f"中心性分析失敗: {e}")
        
        return centrality_metrics


class AIVASkillGraph:
    """
    AIVA 技能圖主類
    
    功能:
    - 構建能力關係圖
    - 路徑規劃和推薦
    - 能力分析和評估
    """
    
    def __init__(self, capability_registry: Optional[CapabilityRegistry] = None):
        self.capability_registry = capability_registry or CapabilityRegistry()
        self.builder = SkillGraphBuilder(self.capability_registry)
        self.analyzer: Optional[SkillGraphAnalyzer] = None
        self.last_build_time: Optional[datetime] = None
        self._build_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """初始化技能圖"""
        async with self._build_lock:
            await self.builder.build_graph()
            
            self.analyzer = SkillGraphAnalyzer(
                self.builder.graph,
                self.builder.skill_nodes
            )
            
            self.last_build_time = datetime.utcnow()
            
        logger.info("AIVA 技能圖初始化完成")
    
    async def rebuild_if_needed(self, max_age_hours: int = 24) -> None:
        """根據需要重建技能圖"""
        if not self.last_build_time:
            await self.initialize()
            return
        
        age = datetime.utcnow() - self.last_build_time
        if age.total_seconds() > max_age_hours * 3600:
            logger.info("技能圖需要重建...")
            await self.initialize()
    
    async def find_execution_path(
        self, 
        start_capability: str, 
        goal: str,
        max_path_length: int = 5
    ) -> List[SkillPath]:
        """尋找執行路徑"""
        await self.rebuild_if_needed()
        
        if not self.analyzer:
            raise RuntimeError("技能圖未初始化")
        
        return self.analyzer.find_optimal_path(start_capability, goal, max_path_length)
    
    async def get_recommendations(
        self, 
        capability_id: str, 
        limit: int = 5
    ) -> List[Tuple[str, float, str]]:
        """獲取能力推薦"""
        await self.rebuild_if_needed()
        
        if not self.analyzer:
            raise RuntimeError("技能圖未初始化")
        
        return self.analyzer.get_capability_recommendations(capability_id, limit)
    
    async def analyze_centrality(self) -> Dict[str, Dict[str, float]]:
        """分析能力中心性"""
        await self.rebuild_if_needed()
        
        if not self.analyzer:
            raise RuntimeError("技能圖未初始化")
        
        return self.analyzer.analyze_capability_centrality()
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """獲取圖統計信息"""
        if not self.builder.graph:
            return {}
        
        graph = self.builder.graph
        
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph),
            "components": nx.number_weakly_connected_components(graph),
            "average_clustering": nx.average_clustering(graph.to_undirected()),
            "last_build_time": self.last_build_time.isoformat() if self.last_build_time else None
        }


# 創建全域技能圖實例
skill_graph = AIVASkillGraph()