"""Path Difference Analyzer - 路徑差異分析器

基於流程分析結果，分析到達同一終點的不同路徑的效率、風險、成本
支援分析整個程式的所有流程，不限於 AI 核心模組

作者: AIVA Team  
版本: 1.0
日期: 2025-12-06
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FlowPath:
    """單一流程路徑"""
    id: str                    # 如 "data_flow_chain_100"
    nodes: List[str]          # 節點序列 ["bio", "trainer", "rl", "models"]
    entry_point: str          # 起點
    endpoint: str             # 終點
    frequency: int            # 使用頻率
    length: int               # 路徑長度
    source_file: str          # 來源文件


@dataclass
class PathMetrics:
    """路徑評估指標"""
    efficiency_score: float   # 效率評分 (0-1)
    risk_score: float         # 風險評分 (0-1)
    cost_score: float         # 成本評分 (0-1)
    popularity_score: float   # 受歡迎度 (0-1)
    overall_score: float      # 綜合評分 (0-1)


@dataclass
class PathAnalysisReport:
    """路徑分析報告"""
    endpoint: str
    total_paths: int
    unique_paths: int
    paths: List[FlowPath]
    path_metrics: Dict[str, PathMetrics]
    recommended_path: Optional[str]
    analysis_summary: str


class PathDifferenceAnalyzer:
    """路徑差異分析器
    
    核心功能：
    1. 解析任意目錄的流程分析結果
    2. 按終點分組路徑
    3. 評估路徑差異（效率、風險、成本）
    4. 推薦最佳路徑
    5. 支援全程式分析（不限於 AI 核心）
    """
    
    def __init__(self, flow_analysis_dir: Path):
        """初始化路徑差異分析器
        
        Args:
            flow_analysis_dir: 流程分析結果目錄 
                             (可以是 AI 核心或整個程式的分析結果)
        """
        self.flow_analysis_dir = Path(flow_analysis_dir)
        self.flows: List[FlowPath] = []
        self.endpoint_groups: Dict[str, List[FlowPath]] = defaultdict(list)
        
        # 高風險節點 (根據分析統計)
        self.high_risk_nodes = {"rl", "trainer", "neural", "network"}
        
        # 效率權重配置
        self.weights = {
            "efficiency": 0.35,    # 效率權重
            "risk": 0.25,          # 風險權重  
            "cost": 0.20,          # 成本權重
            "popularity": 0.20     # 受歡迎度權重
        }
        
        logger.info(f"初始化路徑分析器: {flow_analysis_dir}")
        self._load_flows()
        
    def _load_flows(self) -> None:
        """載入流程分析結果"""
        md_files = list(self.flow_analysis_dir.glob("data_flow_chain_*.md"))
        
        if not md_files:
            logger.warning(f"未找到流程圖文件: {self.flow_analysis_dir}")
            return
            
        logger.info(f"載入 {len(md_files)} 個流程圖文件")
        
        for md_file in md_files:
            try:
                flow = self._parse_flow_file(md_file)
                if flow:
                    self.flows.append(flow)
                    self.endpoint_groups[flow.endpoint].append(flow)
            except Exception as e:
                logger.error(f"解析文件失敗 {md_file}: {e}")
        
        logger.info(f"成功載入 {len(self.flows)} 個流程路徑")
        logger.info(f"發現 {len(self.endpoint_groups)} 個不同終點")
        
    def _parse_flow_file(self, md_file: Path) -> Optional[FlowPath]:
        """解析單個流程圖文件"""
        filename = md_file.stem
        
        # 提取節點序列
        # 格式: data_flow_chain_100_0_scalable_bio_trainer_neural_network_rl_models_flow
        parts = filename.split('_')
        
        # 過濾數字和無意義詞
        skip_words = {'data', 'flow', 'chain', 'scalable'}
        scripts = []
        
        for part in parts:
            if part.isdigit() or part in skip_words:
                continue
            scripts.append(part)
        
        if len(scripts) < 2:
            return None
            
        return FlowPath(
            id=filename,
            nodes=scripts,
            entry_point=scripts[0],
            endpoint=scripts[-1],
            frequency=1,  # 每個文件代表1次使用
            length=len(scripts),
            source_file=str(md_file)
        )
        
    def get_multi_path_endpoints(self) -> Dict[str, int]:
        """獲取有多條路徑的終點"""
        multi_path_endpoints = {}
        
        for endpoint, paths in self.endpoint_groups.items():
            unique_paths = self._get_unique_path_patterns(paths)
            if len(unique_paths) > 1:
                multi_path_endpoints[endpoint] = len(unique_paths)
                
        return multi_path_endpoints
        
    def _get_unique_path_patterns(self, paths: List[FlowPath]) -> List[str]:
        """獲取唯一路徑模式"""
        patterns = set()
        for path in paths:
            pattern = " → ".join(path.nodes)
            patterns.add(pattern)
        return list(patterns)
        
    def analyze_path_differences(self, endpoint: str) -> PathAnalysisReport:
        """分析指定終點的路徑差異
        
        Args:
            endpoint: 終點名稱
            
        Returns:
            PathAnalysisReport: 完整的路徑分析報告
        """
        if endpoint not in self.endpoint_groups:
            raise ValueError(f"終點 '{endpoint}' 不存在")
            
        paths = self.endpoint_groups[endpoint]
        unique_patterns = self._get_unique_path_patterns(paths)
        
        logger.info(f"分析終點 '{endpoint}': {len(paths)} 個路徑, {len(unique_patterns)} 種模式")
        
        # 計算每種路徑模式的指標
        path_metrics = {}
        pattern_paths = self._group_paths_by_pattern(paths)
        
        for pattern, pattern_flows in pattern_paths.items():
            metrics = self._calculate_path_metrics(pattern_flows, paths)
            path_metrics[pattern] = metrics
            
        # 推薦最佳路徑
        recommended_path = self._select_best_path(path_metrics)
        
        # 生成分析摘要
        summary = self._generate_analysis_summary(endpoint, pattern_paths, path_metrics)
        
        return PathAnalysisReport(
            endpoint=endpoint,
            total_paths=len(paths),
            unique_paths=len(unique_patterns),
            paths=paths,
            path_metrics=path_metrics,
            recommended_path=recommended_path,
            analysis_summary=summary
        )
        
    def _group_paths_by_pattern(self, paths: List[FlowPath]) -> Dict[str, List[FlowPath]]:
        """按路徑模式分組"""
        pattern_groups = defaultdict(list)
        
        for path in paths:
            pattern = " → ".join(path.nodes)
            pattern_groups[pattern].append(path)
            
        return dict(pattern_groups)
        
    def _calculate_path_metrics(self, pattern_flows: List[FlowPath], all_flows: List[FlowPath]) -> PathMetrics:
        """計算路徑指標"""
        # 計算頻率 (該模式的使用次數)
        frequency = len(pattern_flows)
        max_frequency = max(len(flows) for flows in self._group_paths_by_pattern(all_flows).values())
        
        # 計算平均路徑長度
        avg_length = sum(flow.length for flow in pattern_flows) / len(pattern_flows)
        max_length = max(flow.length for flow in all_flows)
        min_length = min(flow.length for flow in all_flows)
        
        # 1. 效率評分 (路徑越短越好)
        if max_length == min_length:
            efficiency_score = 1.0
        else:
            efficiency_score = 1.0 - (avg_length - min_length) / (max_length - min_length)
            
        # 2. 風險評分 (依賴高風險節點越少越好)
        risk_nodes = set()
        for flow in pattern_flows:
            risk_nodes.update(node for node in flow.nodes if node in self.high_risk_nodes)
        
        # 風險評分: 0 = 無風險, 1 = 最高風險
        max_possible_risks = len(self.high_risk_nodes)
        risk_score = len(risk_nodes) / max_possible_risks if max_possible_risks > 0 else 0
        
        # 3. 成本評分 (路徑複雜度)
        # 基於路徑長度和節點複雜度
        complexity_score = avg_length / 10.0  # 假設最大合理長度為10
        cost_score = min(complexity_score, 1.0)
        
        # 4. 受歡迎度評分 (使用頻率)
        popularity_score = frequency / max_frequency if max_frequency > 0 else 0
        
        # 5. 綜合評分
        overall_score = (
            efficiency_score * self.weights["efficiency"] +
            (1 - risk_score) * self.weights["risk"] +  # 風險越低越好
            (1 - cost_score) * self.weights["cost"] +  # 成本越低越好  
            popularity_score * self.weights["popularity"]
        )
        
        return PathMetrics(
            efficiency_score=round(efficiency_score, 3),
            risk_score=round(risk_score, 3),
            cost_score=round(cost_score, 3),
            popularity_score=round(popularity_score, 3),
            overall_score=round(overall_score, 3)
        )
        
    def _select_best_path(self, path_metrics: Dict[str, PathMetrics]) -> Optional[str]:
        """選擇最佳路徑"""
        if not path_metrics:
            return None
            
        best_path = max(path_metrics.items(), key=lambda x: x[1].overall_score)
        return best_path[0]
        
    def _generate_analysis_summary(self, 
                                 endpoint: str, 
                                 pattern_paths: Dict[str, List[FlowPath]], 
                                 path_metrics: Dict[str, PathMetrics]) -> str:
        """生成分析摘要"""
        total_flows = sum(len(flows) for flows in pattern_paths.values())
        
        summary_lines = [
            f"終點 '{endpoint}' 路徑分析摘要:",
            f"- 總路徑數: {total_flows}",
            f"- 唯一模式: {len(pattern_paths)}",
            "",
            "路徑評估結果:"
        ]
        
        # 按綜合評分排序
        sorted_paths = sorted(path_metrics.items(), key=lambda x: x[1].overall_score, reverse=True)
        
        for i, (pattern, metrics) in enumerate(sorted_paths[:5], 1):
            frequency = len(pattern_paths[pattern])
            summary_lines.extend([
                f"{i}. 路徑: {pattern}",
                f"   使用次數: {frequency}, 綜合評分: {metrics.overall_score}",
                f"   效率: {metrics.efficiency_score}, 風險: {metrics.risk_score}, "
                f"成本: {metrics.cost_score}, 受歡迎度: {metrics.popularity_score}",
                ""
            ])
            
        return "\n".join(summary_lines)
        
    def analyze_all_multi_path_endpoints(self) -> Dict[str, PathAnalysisReport]:
        """分析所有有多條路徑的終點"""
        multi_path_endpoints = self.get_multi_path_endpoints()
        reports = {}
        
        logger.info(f"分析 {len(multi_path_endpoints)} 個多路徑終點")
        
        for endpoint in multi_path_endpoints.keys():
            try:
                report = self.analyze_path_differences(endpoint)
                reports[endpoint] = report
                logger.info(f"完成分析: {endpoint} ({report.unique_paths} 種路徑)")
            except Exception as e:
                logger.error(f"分析終點 {endpoint} 失敗: {e}")
                
        return reports
        
    def export_analysis_results(self, output_dir: Path) -> None:
        """導出分析結果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 導出多路徑終點分析
        multi_path_reports = self.analyze_all_multi_path_endpoints()
        
        # 生成 JSON 報告
        json_report = {
            "analysis_timestamp": "2025-12-06",
            "flow_analysis_source": str(self.flow_analysis_dir),
            "total_flows": len(self.flows),
            "total_endpoints": len(self.endpoint_groups),
            "multi_path_endpoints": len(multi_path_reports),
            "endpoint_analysis": {}
        }
        
        for endpoint, report in multi_path_reports.items():
            json_report["endpoint_analysis"][endpoint] = {
                "total_paths": report.total_paths,
                "unique_paths": report.unique_paths,
                "recommended_path": report.recommended_path,
                "path_metrics": {
                    pattern: {
                        "efficiency": metrics.efficiency_score,
                        "risk": metrics.risk_score,
                        "cost": metrics.cost_score,
                        "popularity": metrics.popularity_score,
                        "overall": metrics.overall_score
                    }
                    for pattern, metrics in report.path_metrics.items()
                }
            }
        
        # 保存 JSON 報告
        json_file = output_dir / "path_analysis_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"分析結果已導出至: {json_file}")
        
        # 生成詳細報告
        detailed_report_file = output_dir / "path_analysis_detailed.txt"
        with open(detailed_report_file, 'w', encoding='utf-8') as f:
            f.write("=== AIVA 路徑差異分析詳細報告 ===\n\n")
            
            for endpoint, report in multi_path_reports.items():
                f.write(f"{report.analysis_summary}\n")
                f.write("="*80 + "\n\n")
                
        logger.info(f"詳細報告已導出至: {detailed_report_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="路徑差異分析器")
    parser.add_argument("flow_dir", help="流程分析結果目錄")
    parser.add_argument("--endpoint", help="指定終點進行分析")
    parser.add_argument("--output", default="./path_analysis_results", help="輸出目錄")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # 初始化分析器
    analyzer = PathDifferenceAnalyzer(Path(args.flow_dir))
    
    if args.endpoint:
        # 分析指定終點
        report = analyzer.analyze_path_differences(args.endpoint)
        print(report.analysis_summary)
    else:
        # 分析所有多路徑終點
        analyzer.export_analysis_results(Path(args.output))
        print(f"分析完成，結果已保存至: {args.output}")