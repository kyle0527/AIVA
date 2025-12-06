"""Capability Classifier - 能力分類器

基於終點腳本名稱對照，分類和比較不同的系統能力
支援整個程式的分析，明確說明各類能力的功能

作者: AIVA Team
版本: 1.0  
日期: 2025-12-06
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class CapabilityCategory(Enum):
    """能力分類枚舉"""
    # 核心 AI 能力
    AI_CORE = "ai_core"                    # AI 核心系統
    LEARNING = "learning"                  # 機器學習與訓練
    OPTIMIZATION = "optimization"          # 自我優化與改進
    
    # 分析與處理能力  
    ANALYSIS = "analysis"                  # 能力分析與評估
    PROCESSING = "processing"              # 數據處理與計算
    EXPLORATION = "exploration"            # 內部探索與發現
    
    # 存儲與管理能力
    STORAGE = "storage"                    # 數據存儲與管理
    REGISTRY = "registry"                  # 能力註冊與管理
    MANAGEMENT = "management"              # 系統管理與控制
    
    # 接口與服務能力
    SERVICE = "service"                    # 服務提供與接口
    ORCHESTRATION = "orchestration"       # 流程編排與協調
    COMMUNICATION = "communication"       # 通信與交互
    
    # 安全與攻擊能力
    SECURITY = "security"                  # 安全防護與檢測
    ATTACK = "attack"                      # 攻擊與滲透測試
    EXPLOITATION = "exploitation"          # 漏洞利用與測試
    
    # 工具與輔助能力
    TOOLS = "tools"                        # 工具與輔助功能
    UTILITIES = "utilities"                # 實用工具與幫助
    UNKNOWN = "unknown"                    # 未分類或未知


@dataclass
class CapabilityInfo:
    """能力信息"""
    name: str                              # 能力名稱 (終點腳本名)
    category: CapabilityCategory           # 分類類別
    description: str                       # 功能描述
    frequency: int                         # 使用頻率
    implementation_paths: List[str]        # 實現路徑
    complexity_score: float                # 複雜度評分
    risk_level: str                        # 風險等級 (low/medium/high)
    dependencies: Set[str]                 # 依賴的其他能力
    
    def __hash__(self):
        return hash(self.name)


class CapabilityClassifier:
    """能力分類器
    
    核心功能：
    1. 基於終點腳本名稱識別和分類能力
    2. 提供明確的功能說明和分類體系  
    3. 支援能力比較和相似性分析
    4. 自動識別新能力並建議分類
    5. 支援整個程式的全面分析
    """
    
    def __init__(self):
        """初始化能力分類器"""
        self.capabilities: Dict[str, CapabilityInfo] = {}
        self.category_mapping = self._build_category_mapping()
        self.description_templates = self._build_description_templates()
        
        logger.info("初始化能力分類器")
        
    def _build_category_mapping(self) -> Dict[str, CapabilityCategory]:
        """建立終點名稱到分類的映射"""
        return {
            # AI 核心能力
            "models": CapabilityCategory.AI_CORE,
            "core": CapabilityCategory.AI_CORE,
            "neural": CapabilityCategory.AI_CORE,
            "network": CapabilityCategory.AI_CORE,
            "ai": CapabilityCategory.AI_CORE,
            
            # 學習與訓練
            "trainer": CapabilityCategory.LEARNING,
            "trainers": CapabilityCategory.LEARNING,
            "training": CapabilityCategory.LEARNING,
            "rl": CapabilityCategory.LEARNING,             # 強化學習
            "reinforcement": CapabilityCategory.LEARNING,
            "learn": CapabilityCategory.LEARNING,
            "learner": CapabilityCategory.LEARNING,
            
            # 自我優化
            "optimizer": CapabilityCategory.OPTIMIZATION,
            "optimization": CapabilityCategory.OPTIMIZATION,
            "improve": CapabilityCategory.OPTIMIZATION,
            "enhancement": CapabilityCategory.OPTIMIZATION,
            "tuning": CapabilityCategory.OPTIMIZATION,
            "auto": CapabilityCategory.OPTIMIZATION,
            
            # 分析與評估
            "analyzer": CapabilityCategory.ANALYSIS,
            "analysis": CapabilityCategory.ANALYSIS,
            "evaluator": CapabilityCategory.ANALYSIS,
            "assessment": CapabilityCategory.ANALYSIS,
            "inspector": CapabilityCategory.ANALYSIS,
            "detector": CapabilityCategory.ANALYSIS,
            "classifier": CapabilityCategory.ANALYSIS,      # 分類器屬於分析
            
            # 數據處理
            "processor": CapabilityCategory.PROCESSING,
            "processing": CapabilityCategory.PROCESSING,
            "parser": CapabilityCategory.PROCESSING,
            "transformer": CapabilityCategory.PROCESSING,
            "converter": CapabilityCategory.PROCESSING,
            "filter": CapabilityCategory.PROCESSING,
            "mapper": CapabilityCategory.PROCESSING,        # 數據映射處理
            "adapter": CapabilityCategory.PROCESSING,       # 數據適配處理
            "generator": CapabilityCategory.PROCESSING,     # 數據生成處理
            
            # 內部探索
            "explorer": CapabilityCategory.EXPLORATION,
            "exploration": CapabilityCategory.EXPLORATION,
            "scanner": CapabilityCategory.EXPLORATION,
            "discovery": CapabilityCategory.EXPLORATION,
            "probe": CapabilityCategory.EXPLORATION,
            
            # 存儲管理
            "store": CapabilityCategory.STORAGE,
            "storage": CapabilityCategory.STORAGE,
            "database": CapabilityCategory.STORAGE,
            "cache": CapabilityCategory.STORAGE,
            "repository": CapabilityCategory.STORAGE,
            "vault": CapabilityCategory.STORAGE,
            
            # 能力註冊
            "registry": CapabilityCategory.REGISTRY,
            "register": CapabilityCategory.REGISTRY,
            "catalog": CapabilityCategory.REGISTRY,
            "index": CapabilityCategory.REGISTRY,
            
            # 系統管理
            "manager": CapabilityCategory.MANAGEMENT,
            "management": CapabilityCategory.MANAGEMENT,
            "controller": CapabilityCategory.MANAGEMENT,
            "supervisor": CapabilityCategory.MANAGEMENT,
            "monitor": CapabilityCategory.MANAGEMENT,
            "planner": CapabilityCategory.MANAGEMENT,       # 規劃管理
            "commander": CapabilityCategory.MANAGEMENT,     # 命令管理
            
            # 服務接口
            "service": CapabilityCategory.SERVICE,
            "server": CapabilityCategory.SERVICE,
            "api": CapabilityCategory.SERVICE,
            "interface": CapabilityCategory.SERVICE,
            "endpoint": CapabilityCategory.SERVICE,
            "app": CapabilityCategory.SERVICE,              # 應用服務
            
            # 流程編排
            "orchestrator": CapabilityCategory.ORCHESTRATION,
            "orchestration": CapabilityCategory.ORCHESTRATION,
            "coordinator": CapabilityCategory.ORCHESTRATION,
            "scheduler": CapabilityCategory.ORCHESTRATION,
            "dispatcher": CapabilityCategory.ORCHESTRATION,
            "engine": CapabilityCategory.ORCHESTRATION,     # 執行引擎
            
            # 通信交互
            "communicator": CapabilityCategory.COMMUNICATION,
            "messenger": CapabilityCategory.COMMUNICATION,
            "broker": CapabilityCategory.COMMUNICATION,
            "gateway": CapabilityCategory.COMMUNICATION,
            
            # 安全防護
            "security": CapabilityCategory.SECURITY,
            "guard": CapabilityCategory.SECURITY,
            "protection": CapabilityCategory.SECURITY,
            "shield": CapabilityCategory.SECURITY,
            "validator": CapabilityCategory.SECURITY,
            
            # 攻擊測試
            "attack": CapabilityCategory.ATTACK,
            "attacker": CapabilityCategory.ATTACK,
            "penetration": CapabilityCategory.ATTACK,
            "exploit": CapabilityCategory.EXPLOITATION,
            "vulnerability": CapabilityCategory.EXPLOITATION,
            
            # 工具輔助
            "tool": CapabilityCategory.TOOLS,
            "helper": CapabilityCategory.UTILITIES,
            "utility": CapabilityCategory.UTILITIES,
            "utils": CapabilityCategory.UTILITIES,
            "visualizer": CapabilityCategory.UTILITIES,     # 可視化工具
            "matrix": CapabilityCategory.UTILITIES,         # 矩陣工具
            "system": CapabilityCategory.UTILITIES,         # 系統工具
        }
        
    def _build_description_templates(self) -> Dict[CapabilityCategory, str]:
        """建立分類描述模板"""
        return {
            CapabilityCategory.AI_CORE: "AI 核心系統 - 提供基礎人工智能功能和神經網絡處理能力",
            CapabilityCategory.LEARNING: "機器學習 - 實現模型訓練、強化學習和自適應算法",
            CapabilityCategory.OPTIMIZATION: "自我優化 - 系統性能優化、參數調整和效能提升",
            CapabilityCategory.ANALYSIS: "能力分析 - 系統分析、性能評估和智能檢測功能",
            CapabilityCategory.PROCESSING: "數據處理 - 數據解析、轉換和格式化處理",
            CapabilityCategory.EXPLORATION: "內部探索 - 系統探索、發現和自我檢查機制",
            CapabilityCategory.STORAGE: "數據存儲 - 數據持久化、存儲管理和緩存機制",
            CapabilityCategory.REGISTRY: "能力註冊 - 功能註冊、索引管理和能力目錄維護",
            CapabilityCategory.MANAGEMENT: "系統管理 - 資源管理、流程控制和系統監控",
            CapabilityCategory.SERVICE: "服務接口 - API 服務、接口提供和外部交互",
            CapabilityCategory.ORCHESTRATION: "流程編排 - 任務協調、流程管理和系統編排",
            CapabilityCategory.COMMUNICATION: "通信交互 - 系統間通信、消息傳遞和協議處理",
            CapabilityCategory.SECURITY: "安全防護 - 安全檢查、保護機制和威脅防禦",
            CapabilityCategory.ATTACK: "攻擊測試 - 安全測試、滲透測試和攻擊模擬",
            CapabilityCategory.EXPLOITATION: "漏洞利用 - 漏洞測試、安全評估和弱點分析",
            CapabilityCategory.TOOLS: "輔助工具 - 系統工具、實用程序和輔助功能",
            CapabilityCategory.UTILITIES: "實用工具 - 通用工具、幫助程序和便利功能",
            CapabilityCategory.UNKNOWN: "未分類功能 - 尚未明確分類的系統能力"
        }
        
    def classify_capability(self, endpoint_name: str, path_info: Dict[str, Any] = None) -> CapabilityCategory:
        """基於終點腳本名稱分類能力"""
        endpoint_lower = endpoint_name.lower()
        
        # 精確匹配
        if endpoint_lower in self.category_mapping:
            return self.category_mapping[endpoint_lower]
            
        # 模糊匹配 (包含關鍵詞)
        for keyword, category in self.category_mapping.items():
            if keyword in endpoint_lower or endpoint_lower in keyword:
                return category
                
        # 基於路徑信息推斷 (如果提供)
        if path_info:
            nodes = path_info.get('nodes', [])
            for node in nodes:
                node_lower = node.lower()
                if node_lower in self.category_mapping:
                    # 返回路徑中的優先級較高的分類
                    return self.category_mapping[node_lower]
                    
        return CapabilityCategory.UNKNOWN
        
    def generate_capability_description(self, name: str, category: CapabilityCategory, 
                                      paths: List[str] = None) -> str:
        """生成能力功能描述"""
        base_description = self.description_templates.get(category, "未知功能")
        
        # 基於終點名稱添加具體描述
        specific_descriptions = {
            "models": "神經網絡模型管理 - 管理和維護 AI 模型的核心組件",
            "trainer": "模型訓練器 - 負責 AI 模型的訓練和參數優化過程",
            "rl": "強化學習引擎 - 實現智能體的強化學習算法和策略優化",
            "orchestrator": "系統編排器 - 協調各個系統組件的執行流程和任務分配", 
            "core": "核心處理器 - 提供系統的核心處理邏輯和基礎功能",
            "analyzer": "智能分析器 - 執行深度分析和智能檢測任務",
            "store": "數據存儲中心 - 管理系統數據的持久化和檢索機制",
            "manager": "資源管理器 - 負責系統資源的分配和管理操作",
            "service": "服務提供者 - 向外部系統提供 API 服務和功能接口",
            "registry": "能力註冊中心 - 維護系統功能的註冊和發現機制",
        }
        
        if name.lower() in specific_descriptions:
            return specific_descriptions[name.lower()]
            
        # 構建基於路徑的描述
        if paths and len(paths) > 0:
            common_nodes = self._extract_common_nodes(paths)
            if common_nodes:
                return f"{base_description} (通過 {' → '.join(common_nodes)} 實現)"
                
        return base_description
        
    def _extract_common_nodes(self, paths: List[str]) -> List[str]:
        """從多個路徑中提取常見節點"""
        if not paths:
            return []
            
        # 分割路徑並統計節點頻率
        all_nodes = []
        for path in paths:
            nodes = [node.strip() for node in path.split('→')]
            all_nodes.extend(nodes)
            
        node_counts = Counter(all_nodes)
        
        # 返回出現頻率 >= 50% 的節點
        threshold = len(paths) * 0.5
        common_nodes = [node for node, count in node_counts.items() if count >= threshold]
        
        return common_nodes[:3]  # 最多返回3個常見節點
        
    def calculate_complexity_score(self, paths: List[str]) -> float:
        """計算能力複雜度評分"""
        if not paths:
            return 0.0
            
        # 基於路徑長度和變異性
        path_lengths = []
        for path in paths:
            nodes = [node.strip() for node in path.split('→')]
            path_lengths.append(len(nodes))
            
        avg_length = sum(path_lengths) / len(path_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in path_lengths) / len(path_lengths)
        
        # 複雜度 = 平均路徑長度 + 路徑變異性
        complexity = (avg_length / 10.0) + (length_variance / 5.0)  # 歸一化
        
        return min(complexity, 1.0)  # 限制在 0-1 範圍
        
    def assess_risk_level(self, category: CapabilityCategory, paths: List[str]) -> str:
        """評估能力風險等級"""
        # 高風險分類
        high_risk_categories = {
            CapabilityCategory.ATTACK,
            CapabilityCategory.EXPLOITATION, 
            CapabilityCategory.AI_CORE
        }
        
        # 中風險分類
        medium_risk_categories = {
            CapabilityCategory.LEARNING,
            CapabilityCategory.OPTIMIZATION,
            CapabilityCategory.ORCHESTRATION
        }
        
        if category in high_risk_categories:
            return "high"
        elif category in medium_risk_categories:
            return "medium"
        else:
            # 基於路徑複雜度進一步評估
            complexity = self.calculate_complexity_score(paths)
            if complexity > 0.7:
                return "medium"
            else:
                return "low"
                
    def extract_dependencies(self, paths: List[str]) -> Set[str]:
        """提取能力依賴"""
        dependencies = set()
        
        for path in paths:
            nodes = [node.strip() for node in path.split('→')]
            # 除了終點，其他都是依賴
            dependencies.update(nodes[:-1])
            
        return dependencies
        
    def add_capability(self, name: str, paths: List[str], frequency: int = 1) -> CapabilityInfo:
        """添加能力到分類系統"""
        category = self.classify_capability(name)
        description = self.generate_capability_description(name, category, paths)
        complexity = self.calculate_complexity_score(paths)
        risk_level = self.assess_risk_level(category, paths)
        dependencies = self.extract_dependencies(paths)
        
        capability = CapabilityInfo(
            name=name,
            category=category,
            description=description,
            frequency=frequency,
            implementation_paths=paths.copy(),
            complexity_score=complexity,
            risk_level=risk_level,
            dependencies=dependencies
        )
        
        self.capabilities[name] = capability
        logger.info(f"添加能力: {name} ({category.value})")
        
        return capability
        
    def load_capabilities_from_flow_analysis(self, flow_analysis_dir: Path) -> None:
        """從流程分析結果載入能力"""
        try:
            from .path_difference_analyzer import PathDifferenceAnalyzer
        except ImportError:
            from path_difference_analyzer import PathDifferenceAnalyzer
        
        # 使用路徑分析器獲取數據
        analyzer = PathDifferenceAnalyzer(flow_analysis_dir)
        
        # 為每個終點創建能力
        for endpoint, paths in analyzer.endpoint_groups.items():
            # 收集該終點的所有路徑模式
            path_patterns = []
            for path in paths:
                pattern = " → ".join(path.nodes)
                path_patterns.append(pattern)
                
            # 去重並計算頻率
            unique_patterns = list(set(path_patterns))
            frequency = len(paths)
            
            # 添加能力
            self.add_capability(endpoint, unique_patterns, frequency)
            
        logger.info(f"從流程分析結果載入 {len(self.capabilities)} 個能力")
        
    def get_capabilities_by_category(self) -> Dict[CapabilityCategory, List[CapabilityInfo]]:
        """按分類獲取能力"""
        categorized = defaultdict(list)
        
        for capability in self.capabilities.values():
            categorized[capability.category].append(capability)
            
        # 每個分類內按頻率排序
        for category in categorized:
            categorized[category].sort(key=lambda x: x.frequency, reverse=True)
            
        return dict(categorized)
        
    def find_similar_capabilities(self, capability_name: str, 
                                threshold: float = 0.5) -> List[Tuple[str, float]]:
        """查找相似能力"""
        if capability_name not in self.capabilities:
            return []
            
        target_capability = self.capabilities[capability_name]
        similar_capabilities = []
        
        for name, capability in self.capabilities.items():
            if name == capability_name:
                continue
                
            # 計算相似度
            similarity = self._calculate_similarity(target_capability, capability)
            
            if similarity >= threshold:
                similar_capabilities.append((name, similarity))
                
        # 按相似度排序
        similar_capabilities.sort(key=lambda x: x[1], reverse=True)
        
        return similar_capabilities
        
    def _calculate_similarity(self, cap1: CapabilityInfo, cap2: CapabilityInfo) -> float:
        """計算兩個能力的相似度"""
        similarities = []
        
        # 1. 分類相似度
        category_similarity = 1.0 if cap1.category == cap2.category else 0.3
        similarities.append(category_similarity * 0.4)  # 權重 40%
        
        # 2. 依賴相似度 (Jaccard 係數)
        if cap1.dependencies or cap2.dependencies:
            intersection = len(cap1.dependencies & cap2.dependencies)
            union = len(cap1.dependencies | cap2.dependencies)
            dependency_similarity = intersection / union if union > 0 else 0
            similarities.append(dependency_similarity * 0.3)  # 權重 30%
            
        # 3. 複雜度相似度
        complexity_diff = abs(cap1.complexity_score - cap2.complexity_score)
        complexity_similarity = 1.0 - complexity_diff
        similarities.append(complexity_similarity * 0.2)  # 權重 20%
        
        # 4. 風險等級相似度
        risk_levels = ["low", "medium", "high"]
        risk_similarity = 1.0 if cap1.risk_level == cap2.risk_level else 0.5
        similarities.append(risk_similarity * 0.1)  # 權重 10%
        
        return sum(similarities)
        
    def generate_capability_comparison_report(self, capability_names: List[str]) -> str:
        """生成能力比較報告"""
        if not capability_names:
            return "沒有指定要比較的能力"
            
        # 檢查能力是否存在
        existing_capabilities = [name for name in capability_names if name in self.capabilities]
        
        if not existing_capabilities:
            return "指定的能力都不存在於系統中"
            
        report_lines = [
            "=== 能力比較分析報告 ===\n",
            f"比較能力數量: {len(existing_capabilities)}",
            f"比較時間: 2025-12-06\n",
            "詳細比較:"
        ]
        
        for name in existing_capabilities:
            capability = self.capabilities[name]
            report_lines.extend([
                f"\n【{name}】",
                f"分類: {capability.category.value}",
                f"描述: {capability.description}",
                f"使用頻率: {capability.frequency}",
                f"複雜度: {capability.complexity_score:.2f}",
                f"風險等級: {capability.risk_level}",
                f"路徑數量: {len(capability.implementation_paths)}",
                f"依賴數量: {len(capability.dependencies)}",
                f"主要依賴: {', '.join(list(capability.dependencies)[:3])}" if capability.dependencies else "無依賴"
            ])
            
        # 添加相似性分析
        if len(existing_capabilities) > 1:
            report_lines.extend(["\n=== 相似性分析 ==="])
            for i, name1 in enumerate(existing_capabilities):
                for name2 in existing_capabilities[i+1:]:
                    similarity = self._calculate_similarity(
                        self.capabilities[name1], 
                        self.capabilities[name2]
                    )
                    report_lines.append(f"{name1} ↔ {name2}: {similarity:.2f}")
                    
        return "\n".join(report_lines)
        
    def export_classification_results(self, output_dir: Path) -> None:
        """導出分類結果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 按分類組織數據
        categorized = self.get_capabilities_by_category()
        
        # 生成 JSON 報告
        json_report = {
            "classification_timestamp": "2025-12-06",
            "total_capabilities": len(self.capabilities),
            "categories": {},
            "capability_details": {}
        }
        
        for category, capabilities in categorized.items():
            json_report["categories"][category.value] = {
                "description": self.description_templates[category],
                "count": len(capabilities),
                "capabilities": [cap.name for cap in capabilities]
            }
            
        for name, capability in self.capabilities.items():
            json_report["capability_details"][name] = {
                "category": capability.category.value,
                "description": capability.description,
                "frequency": capability.frequency,
                "complexity": capability.complexity_score,
                "risk_level": capability.risk_level,
                "paths_count": len(capability.implementation_paths),
                "dependencies": list(capability.dependencies)
            }
            
        # 保存 JSON 報告
        json_file = output_dir / "capability_classification.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"分類結果已導出至: {json_file}")
        
        # 生成分類摘要報告
        summary_file = output_dir / "capability_classification_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== AIVA 能力分類摘要報告 ===\n\n")
            f.write(f"總能力數量: {len(self.capabilities)}\n")
            f.write(f"分類數量: {len(categorized)}\n\n")
            
            for category, capabilities in sorted(categorized.items(), key=lambda x: -len(x[1])):
                f.write(f"【{category.value.upper()}】{self.description_templates[category]}\n")
                f.write(f"能力數量: {len(capabilities)}\n")
                
                # 列出前5個最常用的能力
                top_capabilities = capabilities[:5]
                for cap in top_capabilities:
                    f.write(f"  • {cap.name} (頻率: {cap.frequency}, 複雜度: {cap.complexity_score:.2f})\n")
                    
                if len(capabilities) > 5:
                    f.write(f"  ... +{len(capabilities)-5} 個其他能力\n")
                    
                f.write("\n")
                
        logger.info(f"摘要報告已導出至: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="能力分類器")
    parser.add_argument("--flow-dir", help="流程分析結果目錄")
    parser.add_argument("--compare", nargs="+", help="比較指定能力")
    parser.add_argument("--output", default="./capability_classification_results", help="輸出目錄")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    # 初始化分類器
    classifier = CapabilityClassifier()
    
    if args.flow_dir:
        # 從流程分析結果載入能力
        classifier.load_capabilities_from_flow_analysis(Path(args.flow_dir))
        
    if args.compare:
        # 生成比較報告
        report = classifier.generate_capability_comparison_report(args.compare)
        print(report)
    else:
        # 導出完整分類結果
        classifier.export_classification_results(Path(args.output))
        print(f"分類完成，結果已保存至: {args.output}")