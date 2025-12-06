"""Capability CLI - 統一能力分析命令行介面

整合路徑分析、能力分類、指令構建等功能的統一CLI工具
支援分析整個程式，提供完整的能力管理和執行體驗

作者: AIVA Team
版本: 1.0
日期: 2025-12-06
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# 導入內部模組
try:
    # 嘗試相對導入 (作為模組使用)
    from .path_difference_analyzer import PathDifferenceAnalyzer
    from .capability_classifier import CapabilityClassifier
    from .capability_command_builder import CapabilityCommandBuilder
except ImportError:
    # 絕對導入 (直接執行)
    from path_difference_analyzer import PathDifferenceAnalyzer
    from capability_classifier import CapabilityClassifier
    from capability_command_builder import CapabilityCommandBuilder


class CapabilityCLI:
    """統一能力分析 CLI
    
    整合功能：
    1. 流程分析 (list-flows) - 列出和分析數據流路徑
    2. 路徑比較 (compare-paths) - 比較到達同終點的不同路徑
    3. 能力分類 (classify) - 分類和管理系統能力
    4. 指令構建 (build-commands) - 構建可執行的 CLI 指令序列
    5. 執行能力 (execute-capability) - 執行特定能力
    6. 驗證路徑 (validate-paths) - 驗證路徑的可執行性
    """
    
    def __init__(self, project_root: Path = None):
        """初始化 CLI
        
        Args:
            project_root: 項目根目錄，默認為當前目錄
        """
        self.project_root = Path(project_root or Path.cwd())
        
        # 初始化各個組件
        self.path_analyzer: Optional[PathDifferenceAnalyzer] = None
        self.classifier: Optional[CapabilityClassifier] = None
        self.command_builder: Optional[CapabilityCommandBuilder] = None
        
        # 設置日誌
        self.logger = logging.getLogger(__name__)
        
    def _ensure_flow_analysis_dir(self, flow_dir: str = None) -> Path:
        """確保流程分析目錄存在"""
        if flow_dir:
            flow_analysis_dir = Path(flow_dir)
        else:
            # 默認查找流程分析結果
            possible_dirs = [
                self.project_root / "flow_analysis_ai_core",
                self.project_root / "flow_analysis_results",
                self.project_root / "analysis_results"
            ]
            
            flow_analysis_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists() and list(dir_path.glob("data_flow_chain_*.md")):
                    flow_analysis_dir = dir_path
                    break
                    
            if not flow_analysis_dir:
                raise ValueError(f"未找到流程分析結果，請指定 --flow-dir 或運行流程分析")
                
        if not flow_analysis_dir.exists():
            raise ValueError(f"流程分析目錄不存在: {flow_analysis_dir}")
            
        return flow_analysis_dir
        
    def _init_path_analyzer(self, flow_dir: str = None) -> None:
        """初始化路徑分析器"""
        if not self.path_analyzer:
            flow_analysis_dir = self._ensure_flow_analysis_dir(flow_dir)
            self.path_analyzer = PathDifferenceAnalyzer(flow_analysis_dir)
            
    def _init_classifier(self, flow_dir: str = None) -> None:
        """初始化能力分類器"""
        if not self.classifier:
            self.classifier = CapabilityClassifier()
            
            # 如果有流程分析結果，自動載入
            try:
                flow_analysis_dir = self._ensure_flow_analysis_dir(flow_dir)
                self.classifier.load_capabilities_from_flow_analysis(flow_analysis_dir)
            except ValueError:
                self.logger.warning("未找到流程分析結果，分類器將以空狀態運行")
                
    def _init_command_builder(self) -> None:
        """初始化指令構建器"""
        if not self.command_builder:
            self.command_builder = CapabilityCommandBuilder(self.project_root)
            
    def cmd_list_flows(self, args) -> None:
        """列出數據流路徑"""
        print("=== 數據流路徑列表 ===\n")
        
        self._init_path_analyzer(args.flow_dir)
        
        if args.endpoint:
            # 顯示特定終點的路徑
            if args.endpoint not in self.path_analyzer.endpoint_groups:
                print(f"❌ 終點 '{args.endpoint}' 不存在")
                return
                
            paths = self.path_analyzer.endpoint_groups[args.endpoint]
            patterns = self.path_analyzer._get_unique_path_patterns(paths)
            
            print(f"終點: {args.endpoint}")
            print(f"路徑數量: {len(paths)}")
            print(f"唯一模式: {len(patterns)}\n")
            
            for i, pattern in enumerate(patterns[:10], 1):
                count = sum(1 for p in paths if " → ".join(p.nodes) == pattern)
                print(f"{i:2d}. [{count:2d}次] {pattern}")
                
            if len(patterns) > 10:
                print(f"     ... +{len(patterns)-10} 種其他模式")
                
        else:
            # 顯示所有終點概覽
            print(f"總流程數: {len(self.path_analyzer.flows)}")
            print(f"終點數量: {len(self.path_analyzer.endpoint_groups)}\n")
            
            print("終點列表:")
            for endpoint, paths in sorted(self.path_analyzer.endpoint_groups.items(), 
                                        key=lambda x: -len(x[1]))[:20]:
                unique_patterns = len(set(" → ".join(p.nodes) for p in paths))
                print(f"  • {endpoint:15} {len(paths):3d} 次使用, {unique_patterns:2d} 種模式")
                
    def cmd_compare_paths(self, args) -> None:
        """比較路徑差異"""
        print("=== 路徑差異比較分析 ===\n")
        
        self._init_path_analyzer(args.flow_dir)
        
        if args.endpoint:
            # 分析指定終點
            try:
                report = self.path_analyzer.analyze_path_differences(args.endpoint)
                print(report.analysis_summary)
                
                if args.detailed:
                    print("\n=== 詳細路徑分析 ===")
                    for pattern, metrics in report.path_metrics.items():
                        print(f"\n路徑: {pattern}")
                        print(f"  效率評分: {metrics.efficiency_score:.3f}")
                        print(f"  風險評分: {metrics.risk_score:.3f}")
                        print(f"  成本評分: {metrics.cost_score:.3f}")
                        print(f"  受歡迎度: {metrics.popularity_score:.3f}")
                        print(f"  綜合評分: {metrics.overall_score:.3f}")
                        
            except ValueError as e:
                print(f"❌ {e}")
                
        else:
            # 分析所有多路徑終點
            multi_path_endpoints = self.path_analyzer.get_multi_path_endpoints()
            
            if not multi_path_endpoints:
                print("✅ 沒有發現多路徑終點")
                return
                
            print(f"發現 {len(multi_path_endpoints)} 個多路徑終點：\n")
            
            for endpoint, path_count in sorted(multi_path_endpoints.items(), 
                                             key=lambda x: -x[1])[:10]:
                print(f"• {endpoint:15} {path_count} 種路徑")
                
                if args.detailed:
                    try:
                        report = self.path_analyzer.analyze_path_differences(endpoint)
                        if report.recommended_path:
                            print(f"  推薦路徑: {report.recommended_path}")
                    except:
                        pass
                        
    def cmd_classify(self, args) -> None:
        """能力分類與管理"""
        print("=== 系統能力分類分析 ===\n")
        
        self._init_classifier(args.flow_dir)
        
        if args.capability:
            # 顯示特定能力詳情
            if args.capability not in self.classifier.capabilities:
                print(f"❌ 能力 '{args.capability}' 不存在")
                
                # 建議相似能力
                all_names = list(self.classifier.capabilities.keys())
                similar = [name for name in all_names if args.capability.lower() in name.lower()]
                if similar:
                    print(f"💡 您是否要查找: {', '.join(similar[:5])}")
                return
                
            capability = self.classifier.capabilities[args.capability]
            print(f"能力名稱: {capability.name}")
            print(f"分類: {capability.category.value}")
            print(f"功能描述: {capability.description}")
            print(f"使用頻率: {capability.frequency}")
            print(f"複雜度評分: {capability.complexity_score:.2f}")
            print(f"風險等級: {capability.risk_level}")
            print(f"實現路徑數: {len(capability.implementation_paths)}")
            print(f"依賴數量: {len(capability.dependencies)}")
            
            if capability.dependencies:
                print(f"主要依賴: {', '.join(list(capability.dependencies)[:5])}")
                
            if capability.implementation_paths:
                print("\n實現路徑:")
                for i, path in enumerate(capability.implementation_paths[:3], 1):
                    print(f"  {i}. {path}")
                if len(capability.implementation_paths) > 3:
                    print(f"     ... +{len(capability.implementation_paths)-3} 種")
                    
            # 查找相似能力
            similar = self.classifier.find_similar_capabilities(args.capability, 0.5)
            if similar:
                print(f"\n相似能力:")
                for name, similarity in similar[:5]:
                    print(f"  • {name} (相似度: {similarity:.2f})")
                    
        elif args.compare:
            # 比較多個能力
            report = self.classifier.generate_capability_comparison_report(args.compare)
            print(report)
            
        else:
            # 顯示分類概覽
            categorized = self.classifier.get_capabilities_by_category()
            
            print(f"總能力數: {len(self.classifier.capabilities)}")
            print(f"分類數: {len(categorized)}\n")
            
            for category, capabilities in sorted(categorized.items(), 
                                               key=lambda x: -len(x[1])):
                desc = self.classifier.description_templates[category]
                print(f"【{category.value.upper()}】{desc}")
                print(f"  能力數量: {len(capabilities)}")
                
                # 顯示前3個最常用的能力
                top_caps = capabilities[:3]
                for cap in top_caps:
                    print(f"  • {cap.name} (頻率: {cap.frequency})")
                    
                if len(capabilities) > 3:
                    print(f"    ... +{len(capabilities)-3} 個其他")
                print()
                
    def cmd_build_commands(self, args) -> None:
        """構建 CLI 指令序列"""
        print("=== CLI 指令構建 ===\n")
        
        self._init_command_builder()
        
        if args.capability and args.nodes:
            # 構建特定能力的指令序列
            print(f"正在為能力 '{args.capability}' 構建指令序列...")
            print(f"節點序列: {' → '.join(args.nodes)}")
            
            sequence = self.command_builder.build_command_sequence(
                args.capability, args.nodes
            )
            
            print(f"\n序列 ID: {sequence.sequence_id}")
            print(f"驗證狀態: {sequence.validation_status}")
            print(f"總步驟數: {len(sequence.steps)}")
            print(f"預估時間: {sequence.total_estimated_time:.1f} 秒")
            
            print(f"\n驗證結果:")
            for msg in sequence.validation_messages:
                print(f"  {msg}")
                
            if args.detailed and sequence.steps:
                print(f"\n指令步驟:")
                for i, step in enumerate(sequence.steps, 1):
                    print(f"  {i}. {step.script_name}")
                    print(f"     指令: {step.command}")
                    print(f"     可執行: {'✅' if step.is_executable else '❌'}")
                    print()
                    
        else:
            # 顯示已構建的序列概覽
            sequences = list(self.command_builder.command_sequences.values())
            
            if not sequences:
                print("尚未構建任何指令序列")
                print("使用 --capability 和 --nodes 參數構建新序列")
                return
                
            print(f"已構建序列數: {len(sequences)}\n")
            
            for sequence in sequences[-10:]:  # 顯示最近10個
                status_icon = {
                    "valid": "✅", 
                    "warning": "⚠️", 
                    "invalid": "❌"
                }[sequence.validation_status]
                
                print(f"{status_icon} {sequence.sequence_id}")
                print(f"   能力: {sequence.capability_name}")
                print(f"   步驟: {len(sequence.steps)}, 時間: {sequence.total_estimated_time:.1f}s")
                print()
                
    def cmd_execute_capability(self, args) -> None:
        """執行系統能力"""
        print("=== 能力執行 ===\n")
        
        if args.sequence_id:
            # 執行指定的序列 ID
            self._init_command_builder()
            
            if args.sequence_id not in self.command_builder.command_sequences:
                print(f"❌ 序列 '{args.sequence_id}' 不存在")
                return
                
            print(f"執行序列: {args.sequence_id}")
            print(f"模式: {'乾運行' if args.dry_run else '實際執行'}")
            
            result = self.command_builder.execute_command_sequence(
                args.sequence_id, args.dry_run
            )
            
            print(f"\n執行結果: {'✅ 成功' if result['success'] else '❌ 失敗'}")
            print(f"執行步驟: {result['steps_executed']}/{result['total_steps']}")
            print(f"失敗步驟: {result['steps_failed']}")
            
            if args.detailed:
                print(f"\n詳細日誌:")
                for log in result["execution_log"]:
                    status = "✅" if log["success"] else "❌"
                    print(f"  {status} {log['script_name']}")
                    if log["output"]:
                        print(f"     輸出: {log['output'][:100]}...")
                    if log["error"]:
                        print(f"     錯誤: {log['error']}")
                        
        elif args.capability and args.nodes:
            # 直接執行能力
            print(f"直接執行能力: {args.capability}")
            print(f"路徑: {' → '.join(args.nodes)}")
            
            self._init_command_builder()
            
            # 先構建序列
            sequence = self.command_builder.build_command_sequence(
                args.capability, args.nodes
            )
            
            if sequence.validation_status == "invalid":
                print("❌ 路徑驗證失敗，無法執行")
                for msg in sequence.validation_messages:
                    if "❌" in msg:
                        print(f"  {msg}")
                return
                
            # 執行序列
            result = self.command_builder.execute_command_sequence(
                sequence.sequence_id, args.dry_run
            )
            
            print(f"\n執行完成: {'✅ 成功' if result['success'] else '❌ 失敗'}")
            
        else:
            print("請指定要執行的序列 ID 或能力+路徑")
            
    def cmd_validate_paths(self, args) -> None:
        """驗證路徑可執行性"""
        print("=== 路徑可執行性驗證 ===\n")
        
        self._init_command_builder()
        
        if args.nodes:
            # 驗證指定路徑
            print(f"驗證路徑: {' → '.join(args.nodes)}")
            
            is_valid, messages = self.command_builder.validate_path_executability(args.nodes)
            
            print(f"驗證結果: {'✅ 通過' if is_valid else '❌ 失敗'}\n")
            
            for msg in messages:
                print(f"  {msg}")
                
        else:
            # 驗證所有已知路徑
            self._init_path_analyzer(args.flow_dir)
            
            print("驗證所有發現的路徑...")
            
            valid_count = 0
            invalid_count = 0
            
            # 取樣驗證 (避免過多輸出)
            sample_endpoints = list(self.path_analyzer.endpoint_groups.keys())[:10]
            
            for endpoint in sample_endpoints:
                paths = self.path_analyzer.endpoint_groups[endpoint]
                if paths:
                    first_path = paths[0]
                    is_valid, _ = self.command_builder.validate_path_executability(first_path.nodes)
                    
                    if is_valid:
                        valid_count += 1
                        status = "✅"
                    else:
                        invalid_count += 1
                        status = "❌"
                        
                    print(f"  {status} {endpoint}: {' → '.join(first_path.nodes)}")
                    
            print(f"\n驗證總結:")
            print(f"  有效路徑: {valid_count}")
            print(f"  無效路徑: {invalid_count}")
            print(f"  有效率: {valid_count/(valid_count+invalid_count)*100:.1f}%")
            
    def run(self, argv: List[str] = None) -> int:
        """運行 CLI"""
        parser = self._build_parser()
        
        if argv is None:
            argv = sys.argv[1:]
            
        if not argv:
            parser.print_help()
            return 0
            
        args = parser.parse_args(argv)
        
        # 設置日誌級別
        if args.verbose:
            logging.basicConfig(level=logging.INFO)
        elif args.quiet:
            logging.basicConfig(level=logging.ERROR)
        else:
            logging.basicConfig(level=logging.WARNING)
            
        try:
            # 調用對應的命令處理函數
            handler = getattr(self, f"cmd_{args.command.replace('-', '_')}")
            handler(args)
            return 0
            
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"❌ 執行失敗: {e}")
            return 1
            
    def _build_parser(self) -> argparse.ArgumentParser:
        """構建命令行解析器"""
        parser = argparse.ArgumentParser(
            description="AIVA 能力分析統一命令行工具",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用範例:
  capability_cli list-flows --endpoint models
  capability_cli compare-paths --endpoint orchestrator --detailed  
  capability_cli classify --capability analyzer
  capability_cli build-commands --capability "自我優化" --nodes bio trainer rl models
  capability_cli execute-capability --sequence-id "自我優化_0" --dry-run
  capability_cli validate-paths --nodes analyzer core models
            """
        )
        
        # 全局參數
        parser.add_argument("--project-root", type=Path, help="項目根目錄")
        parser.add_argument("--flow-dir", help="流程分析結果目錄")
        parser.add_argument("--verbose", "-v", action="store_true", help="詳細輸出")
        parser.add_argument("--quiet", "-q", action="store_true", help="靜默模式")
        
        # 子命令
        subparsers = parser.add_subparsers(dest="command", help="可用命令")
        
        # list-flows: 列出數據流
        flows_parser = subparsers.add_parser("list-flows", help="列出數據流路徑")
        flows_parser.add_argument("--endpoint", help="指定終點")
        
        # compare-paths: 比較路徑
        compare_parser = subparsers.add_parser("compare-paths", help="比較路徑差異")
        compare_parser.add_argument("--endpoint", help="指定終點進行詳細分析")
        compare_parser.add_argument("--detailed", action="store_true", help="顯示詳細分析")
        
        # classify: 能力分類
        classify_parser = subparsers.add_parser("classify", help="能力分類與管理")
        classify_parser.add_argument("--capability", help="指定能力名稱")
        classify_parser.add_argument("--compare", nargs="+", help="比較多個能力")
        
        # build-commands: 構建指令
        build_parser = subparsers.add_parser("build-commands", help="構建CLI指令序列")
        build_parser.add_argument("--capability", help="能力名稱")
        build_parser.add_argument("--nodes", nargs="+", help="節點序列")
        build_parser.add_argument("--detailed", action="store_true", help="顯示詳細步驟")
        
        # execute-capability: 執行能力
        exec_parser = subparsers.add_parser("execute-capability", help="執行系統能力")
        exec_parser.add_argument("--sequence-id", help="序列 ID")
        exec_parser.add_argument("--capability", help="直接執行的能力名稱")
        exec_parser.add_argument("--nodes", nargs="+", help="節點序列")
        exec_parser.add_argument("--dry-run", action="store_true", help="乾運行模式")
        exec_parser.add_argument("--detailed", action="store_true", help="顯示詳細日誌")
        
        # validate-paths: 驗證路徑
        validate_parser = subparsers.add_parser("validate-paths", help="驗證路徑可執行性")
        validate_parser.add_argument("--nodes", nargs="+", help="要驗證的節點序列")
        
        return parser


def main():
    """主入口函數"""
    cli = CapabilityCLI()
    return cli.run()


if __name__ == "__main__":
    exit(main())