#!/usr/bin/env python3
"""
Self-Healing 分析執行腳本 (修復優化版)
=====================================
統一的命令行界面，用於執行各種代碼分析任務

修復內容:
1. 修正了 PracticalAnalyzer 的 API 調用錯誤
2. 修正了 MissingConnectionAnalyzer 的方法名稱錯誤
3. 優化了輸入/輸出路徑的處理邏輯
4. 增加了對連接建議分析器的支持

使用方式:
    # 完整分析 (推薦)
    python run_analysis.py --target ./services/core --mode full
    
    # 單獨執行特定分析器
    python run_analysis.py --target ./services/core --analyzer missing
    python run_analysis.py --target ./services/core --analyzer practical
    
    # 批量分析
    python run_analysis.py --targets dir1 dir2 --mode quick
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# 導入分析器
try:
    from .core_analyzer import CoreAnalyzer, AnalysisReport
    from .analyze_dataflow_breakpoints import DataFlowBreakpointAnalyzer
    from .analyze_missing_function_connections import MissingConnectionAnalyzer
    from .practical_analyzer import PracticalAnalyzer
    from .analyze_connection_recommendations import ConnectionRecommendationAnalyzer
except ImportError:
    # 本地執行時的路徑處理
    from core_analyzer import CoreAnalyzer, AnalysisReport
    from analyze_dataflow_breakpoints import DataFlowBreakpointAnalyzer
    from analyze_missing_function_connections import MissingConnectionAnalyzer
    from practical_analyzer import PracticalAnalyzer
    from analyze_connection_recommendations import ConnectionRecommendationAnalyzer


class AnalysisRunner:
    """分析執行器 - 封裝各個分析器的調用邏輯"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def print_banner(self):
        """打印橫幅"""
        print("=" * 80)
        print("🔬 AIVA Self-Healing Analysis Runner (Optimized)")
        print("=" * 80)
        print()
    
    def _get_output_dir(self, target: Path) -> Path:
        """獲取輸出目錄，優先使用統一配置路徑"""
        if self.output_dir:
            return self.output_dir
        
        # 優先使用統一路徑配置
        if USE_INTEGRATED_PATHS and SELF_HEALING_DIR:
            ensure_directories()
            return SELF_HEALING_DIR
        
        # 向後兼容：使用舊路徑
        default_dir = target / "analysis_results"
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

    def _get_analysis_json(self, target: Path) -> Optional[Path]:
        """尋找 analysis_results.json"""
        # 1. 檢查 target/analysis_results/analysis_results.json (標準位置)
        standard_path = target / "analysis_results" / "analysis_results.json"
        if standard_path.exists():
            return standard_path
        
        # 2. 檢查 target/analysis_results.json (可能直接在目標下)
        root_path = target / "analysis_results.json"
        if root_path.exists():
            return root_path
            
        return None

    def run_full_analysis(self, target: Path) -> dict:
        """執行完整分析 (使用 CoreAnalyzer)"""
        print(f"\n🚀 執行完整分析: {target}")
        print("-" * 80)
        
        output_dir = self._get_output_dir(target)
        # CoreAnalyzer 會自己處理輸出目錄的創建
        analyzer = CoreAnalyzer(str(target), str(output_dir))
        report = analyzer.full_analysis()
        
        self._print_report_summary(report)
        return report.to_dict()
    
    def run_quick_scan(self, target: Path) -> dict:
        """執行快速掃描"""
        print(f"\n⚡ 執行快速掃描: {target}")
        print("-" * 80)
        
        output_dir = self._get_output_dir(target)
        analyzer = CoreAnalyzer(str(target), str(output_dir))
        report = analyzer.quick_scan()
        
        self._print_report_summary(report)
        return report.to_dict()
    
    def run_dataflow_analysis(self, target: Path) -> dict:
        """單獨執行數據流斷點分析"""
        print(f"\n🔄 執行數據流斷點分析: {target}")
        print("-" * 80)
        
        results_file = self._get_analysis_json(target)
        if not results_file:
            print("❌ 錯誤: 找不到 analysis_results.json。")
            print("   請先運行完整分析或流程分析以生成基礎數據。")
            return {}
        
        # DataFlowBreakpointAnalyzer 需要的是包含 json 的目錄路徑
        analysis_dir = results_file.parent
        output_dir = self._get_output_dir(target)
        
        print(f"   輸入數據: {results_file}")
        
        try:
            analyzer = DataFlowBreakpointAnalyzer(str(analysis_dir))
            # 執行分析
            analyzer.run_full_analysis()
            
            # 重新生成報告到指定位置 (如果 run_full_analysis 沒放對位置)
            report_name = f"dataflow_breakpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            output_path = output_dir / report_name
            analyzer.generate_report(str(output_path))
            
            print(f"✅ 數據流分析完成，報告已生成: {output_path}")
            return {"type": "dataflow", "report_path": str(output_path)}
            
        except Exception as e:
            print(f"❌ 數據流分析失敗: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_missing_connection_analysis(self, target: Path) -> dict:
        """單獨執行缺失連接分析"""
        print(f"\n🔗 執行缺失函數連接分析: {target}")
        print("-" * 80)
        
        results_file = self._get_analysis_json(target)
        if not results_file:
            print("❌ 錯誤: 找不到 analysis_results.json。")
            print("   請先運行完整分析或流程分析以生成基礎數據。")
            return {}
            
        analysis_dir = results_file.parent
        output_dir = self._get_output_dir(target)
        
        print(f"   輸入數據: {results_file}")
        
        try:
            analyzer = MissingConnectionAnalyzer(str(analysis_dir), str(target))
            # 執行完整流程
            analyzer.run_full_analysis()
            
            # 如果需要，我們可以將報告移動或複製到指定的 output_dir，
            # 但 analyzer.run_full_analysis() 默認輸出到 analysis_dir。
            # 這裡我們顯式調用 generate_report 輸出到我們想要的位置。
            
            final_report_name = f"missing_connections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            final_output_path = output_dir / final_report_name
            analyzer.generate_report(str(final_output_path))
            
            print(f"✅ 缺失連接分析完成，報告已生成: {final_output_path}")
            return {"type": "missing_connections", "report_path": str(final_output_path)}
            
        except Exception as e:
            print(f"❌ 缺失連接分析失敗: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_practical_analysis(self, target: Path) -> dict:
        """執行實用分析（智能過濾）"""
        print(f"\n🎯 執行實用分析（智能過濾）: {target}")
        print("-" * 80)
        
        output_dir = self._get_output_dir(target)
        
        # 尋找輸入報告: 優先找 missing_connections_full.md (CoreAnalyzer產出)
        # 其次找 missing_function_connections.md (獨立分析器產出)
        input_report = None
        candidates = [
            output_dir / "missing_connections_full.md",
            target / "analysis_results" / "missing_connections_full.md",
            output_dir / "missing_function_connections.md",
            target / "analysis_results" / "missing_function_connections.md"
        ]
        
        for path in candidates:
            if path.exists():
                input_report = path
                break
        
        if not input_report:
            print("❌ 錯誤: 找不到缺失連接報告 (missing_connections*.md)。")
            print("   請先運行缺失連接分析。")
            return {}
            
        print(f"   輸入報告: {input_report}")
        
        try:
            # 1. 初始化 (無參數)
            analyzer = PracticalAnalyzer()
            
            # 2. 分析報告
            issues = analyzer.analyze_report(str(input_report))
            
            # 3. 生成結果文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = output_dir / f"practical_report_{timestamp}.md"
            checklist_path = output_dir / f"fix_checklist_{timestamp}.md"
            
            analyzer.generate_practical_report(issues, str(report_path))
            analyzer.generate_quick_fix_list(issues, str(checklist_path))
            
            print("\u2705 實用分析完成")
            print(f"   📄 報告: {report_path}")
            print(f"   📋 清單: {checklist_path}")
            
            return {
                "type": "practical", 
                "report_path": str(report_path),
                "issues_count": sum(len(v) for k, v in issues.items() if k != 'NOISE')
            }
            
        except Exception as e:
            print(f"❌ 實用分析失敗: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run_connection_recommendations(self, target: Path) -> dict:
        """執行連接建議分析 (實驗性)"""
        print(f"\n💡 執行連接建議分析 (Experimental): {target}")
        print("-" * 80)
        
        output_dir = self._get_output_dir(target)
        
        try:
            analyzer = ConnectionRecommendationAnalyzer(str(target))
            analyzer.extract_all_functions()
            recommendations = analyzer.analyze_missing_connections()
            
            report_name = f"connection_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            output_path = output_dir / report_name
            
            analyzer.generate_report(recommendations, str(output_path))
            
            print(f"✅ 連接建議分析完成，報告已生成: {output_path}")
            return {"type": "recommendations", "report_path": str(output_path)}
            
        except Exception as e:
            print(f"❌ 連接建議分析失敗: {e}")
            return {}

    def run_batch_analysis(self, targets: List[Path], mode: str = "quick") -> List[dict]:
        """批量分析多個目標"""
        print(f"\n📦 批量分析模式: {len(targets)} 個目標")
        print("=" * 80)
        
        results = []
        for i, target in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}] 處理: {target}")
            
            if mode == "full":
                result = self.run_full_analysis(target)
            else:
                result = self.run_quick_scan(target)
            
            results.append({
                "target": str(target),
                "result": result
            })
        
        return results
    
    def _print_report_summary(self, report: AnalysisReport):
        """打印報告摘要"""
        print("\n📊 分析完成")
        print(f"   時間: {report.timestamp}")
        print(f"   類型: {report.analysis_type}")
        print(f"   文件數: {report.total_files}")
        print(f"   函數數: {report.total_functions}")
        
        if hasattr(report, 'critical_issues'):
            print(f"\n🔴 關鍵問題: {len(report.critical_issues)}")
        if hasattr(report, 'high_issues'):
            print(f"🟡 高優先級: {len(report.high_issues)}")
        if hasattr(report, 'medium_issues'):
            print(f"🟢 中優先級: {len(report.medium_issues)}")
    
    def save_json_report(self, data: dict, filename: str | None = None):
        """保存 JSON 報告"""
        if not self.output_dir:
            print("⚠️ 未指定輸出目錄，跳過 JSON 保存")
            return

        if filename is None:
            filename = f"analysis_run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            # 處理 AnalysisReport 對象轉 dict
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 JSON 報告已保存: {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AIVA Self-Healing 分析執行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 完整分析 (推薦)
  python run_analysis.py --target ./services/core --mode full
  
  # 快速掃描
  python run_analysis.py --target ./services/core --mode quick
  
  # 單獨運行特定分析器 (用於調試或分步執行)
  python run_analysis.py --target ./services/core --analyzer dataflow
  python run_analysis.py --target ./services/core --analyzer missing
  python run_analysis.py --target ./services/core --analyzer practical
  
  # 批量分析
  python run_analysis.py --targets dir1 dir2
  
  # 指定輸出目錄
  python run_analysis.py --target ./services/core --output ./my_reports
        """
    )
    
    # 目標參數
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--target',
        type=Path,
        help='單個目標目錄'
    )
    target_group.add_argument(
        '--targets',
        type=Path,
        nargs='+',
        help='多個目標目錄（批量分析）'
    )
    
    # 分析模式
    parser.add_argument(
        '--mode',
        choices=['full', 'quick', 'critical'],
        default='quick',
        help='分析模式 (預設: quick)'
    )
    
    # 特定分析器
    parser.add_argument(
        '--analyzer',
        choices=['dataflow', 'missing', 'practical', 'recommendation', 'all'],
        help='只執行特定分析器 (忽略 mode 參數)'
    )
    
    # 輸出格式
    parser.add_argument(
        '--format',
        choices=['markdown', 'json', 'both'],
        default='markdown',
        help='輸出格式 (預設: markdown)'
    )
    
    # 輸出目錄
    parser.add_argument(
        '--output',
        type=Path,
        help='輸出目錄 (預設: target/analysis_results)'
    )
    
    args = parser.parse_args()
    
    # 創建執行器
    runner = AnalysisRunner(output_dir=args.output)
    runner.print_banner()
    
    try:
        # 批量分析模式
        if args.targets:
            results = runner.run_batch_analysis(args.targets, mode=args.mode)
            
            if args.format in ['json', 'both']:
                runner.save_json_report({
                    "batch_analysis": True,
                    "results": results
                })
        
        # 單目標分析
        elif args.target:
            target = args.target.resolve()
            
            if not target.exists():
                print(f"❌ 錯誤: 目標路徑不存在: {target}")
                return 1
            
            # 特定分析器模式
            if args.analyzer:
                if args.analyzer == 'dataflow':
                    result = runner.run_dataflow_analysis(target)
                elif args.analyzer == 'missing':
                    result = runner.run_missing_connection_analysis(target)
                elif args.analyzer == 'practical':
                    result = runner.run_practical_analysis(target)
                elif args.analyzer == 'recommendation':
                    result = runner.run_connection_recommendations(target)
                elif args.analyzer == 'all':
                    results = {
                        "dataflow": runner.run_dataflow_analysis(target),
                        "missing": runner.run_missing_connection_analysis(target),
                        "practical": runner.run_practical_analysis(target),
                    }
                    result = results
                else:
                    result = {}
            
            # 完整/快速模式 (使用 CoreAnalyzer)
            else:
                if args.mode == 'full':
                    result = runner.run_full_analysis(target)
                elif args.mode == 'quick':
                    result = runner.run_quick_scan(target)
                elif args.mode == 'critical':
                    # 這裡直接調用 CoreAnalyzer 的方法，因為 AnalysisRunner 沒有包裝 critical only
                    print(f"\n🔴 診斷關鍵問題: {target}")
                    output_dir = runner._get_output_dir(target)
                    analyzer = CoreAnalyzer(str(target), str(output_dir))
                    report = analyzer.diagnose_critical_only()
                    runner._print_report_summary(report)
                    result = report.to_dict()
            
            # 保存 JSON
            if args.format in ['json', 'both'] and result:
                runner.save_json_report(result)
        
        print("\n" + "=" * 80)
        print("✅ 所有任務執行完畢")
        print("=" * 80)
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  分析已中斷")
        return 130
    except Exception as e:
        print(f"\n❌ 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
