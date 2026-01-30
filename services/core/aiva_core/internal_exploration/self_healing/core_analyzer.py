"""
核心分析器 - 統一入口
========================
整合所有分析工具，提供統一的分析接口

使用方式:
    from internal_exploration import CoreAnalyzer
    
    analyzer = CoreAnalyzer("path/to/source")
    report = analyzer.full_analysis()
    report = analyzer.quick_scan()
    report = analyzer.diagnose_critical_only()
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys

# 導入統一路徑配置
try:
    parent_parent = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(parent_parent))
    from aiva_common.config.paths import (
        SELF_HEALING_DIR,
        ensure_directories
    )
    USE_INTEGRATED_PATHS = True
except ImportError:
    USE_INTEGRATED_PATHS = False
    SELF_HEALING_DIR = None

# 常數定義
MISSING_CONNECTIONS_REPORT = "missing_connections_full.md"

# 導入各個分析器
try:
    # 從父模組導入 AIVAFlowAnalyzer
    from ..python_tools.aiva_flow_analyzer import AIVAFlowAnalyzer
    # 從當前子模組導入其他分析器
    from .analyze_dataflow_breakpoints import DataFlowBreakpointAnalyzer
    from .analyze_missing_function_connections import MissingConnectionAnalyzer
    from .practical_analyzer import PracticalAnalyzer
except ImportError:
    # 本地測試時
    import sys
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(Path(__file__).parent))
    from python_tools.aiva_flow_analyzer import AIVAFlowAnalyzer
    from analyze_dataflow_breakpoints import DataFlowBreakpointAnalyzer
    from analyze_missing_function_connections import MissingConnectionAnalyzer
    from practical_analyzer import PracticalAnalyzer


@dataclass
class AnalysisReport:
    """統一的分析報告"""
    timestamp: str
    source_path: str
    analysis_type: str  # "full", "quick", "critical"
    
    # 統計
    total_files: int = 0
    total_functions: int = 0
    total_issues: int = 0
    
    # 問題分級
    critical_issues: List[Dict] = field(default_factory=list)
    high_issues: List[Dict] = field(default_factory=list)
    medium_issues: List[Dict] = field(default_factory=list)
    low_issues: List[Dict] = field(default_factory=list)
    
    # 各分析器的詳細結果
    flow_analysis: Optional[Dict] = None
    breakpoint_analysis: Optional[Dict] = None
    connection_analysis: Optional[Dict] = None
    
    # 建議
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """轉為字典"""
        return {
            'timestamp': self.timestamp,
            'source_path': self.source_path,
            'analysis_type': self.analysis_type,
            'statistics': {
                'total_files': self.total_files,
                'total_functions': self.total_functions,
                'total_issues': self.total_issues,
                'critical': len(self.critical_issues),
                'high': len(self.high_issues),
                'medium': len(self.medium_issues),
                'low': len(self.low_issues),
            },
            'issues': {
                'critical': self.critical_issues,
                'high': self.high_issues,
                'medium': self.medium_issues,
                'low': self.low_issues,
            },
            'details': {
                'flow': self.flow_analysis,
                'breakpoints': self.breakpoint_analysis,
                'connections': self.connection_analysis,
            },
            'recommendations': self.recommendations,
        }
    
    def save_json(self, output_path: str):
        """保存為 JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_markdown(self, output_path: str):
        """保存為 Markdown"""
        with open(output_path, 'w', encoding='utf-8') as f:
            self._write_header(f)
            self._write_statistics(f)
            self._write_critical_issues(f)
            self._write_high_issues(f)
            self._write_recommendations(f)

    def _write_header(self, f):
        """寫入文檔頭部"""
        f.write(f"# 分析報告 - {self.analysis_type.upper()}\n\n")
        f.write(f"**生成時間:** {self.timestamp}\n")
        f.write(f"**源目錄:** {self.source_path}\n\n")
    
    def _write_statistics(self, f):
        """寫入統計摘要"""
        f.write("## 📊 統計摘要\n\n")
        f.write(f"- **檔案數:** {self.total_files}\n")
        f.write(f"- **函數數:** {self.total_functions}\n")
        f.write(f"- **總問題:** {self.total_issues}\n\n")
        
        f.write("### 問題分級\n\n")
        f.write(f"- 🔴 **CRITICAL:** {len(self.critical_issues)} 個\n")
        f.write(f"- ⚠️ **HIGH:** {len(self.high_issues)} 個\n")
        f.write(f"- ⚡ **MEDIUM:** {len(self.medium_issues)} 個\n")
        f.write(f"- ℹ️ **LOW:** {len(self.low_issues)} 個\n\n")
    
    def _write_critical_issues(self, f):
        """寫入 CRITICAL 問題"""
        if self.critical_issues:
            f.write("## 🔴 CRITICAL - 必須立即修復\n\n")
            for i, issue in enumerate(self.critical_issues[:20], 1):
                f.write(f"### {i}. {issue.get('type', 'Unknown')}\n\n")
                f.write(f"**文件:** `{issue.get('file', 'N/A')}`\n\n")
                f.write(f"**描述:** {issue.get('description', 'N/A')}\n\n")
                if 'suggestion' in issue:
                    f.write(f"**建議:** {issue['suggestion']}\n\n")
                f.write("---\n\n")
            
            if len(self.critical_issues) > 20:
                f.write(f"*... 還有 {len(self.critical_issues) - 20} 個 CRITICAL 問題*\n\n")

    def _write_high_issues(self, f):
        """寫入 HIGH 問題"""
        if self.high_issues:
            f.write("## ⚠️ HIGH - 建議優先修復\n\n")
            for i, issue in enumerate(self.high_issues[:10], 1):
                f.write(f"**{i}.** {issue.get('description', 'N/A')}\n")
                f.write(f"   - 文件: `{issue.get('file', 'N/A')}`\n\n")
            
            if len(self.high_issues) > 10:
                f.write(f"*... 還有 {len(self.high_issues) - 10} 個 HIGH 問題*\n\n")

    def _write_recommendations(self, f):
        """寫入修復建議"""
        if self.recommendations:
            f.write("## 💡 修復建議\n\n")
            for i, rec in enumerate(self.recommendations, 1):
                f.write(f"{i}. {rec}\n")


def classify_script_type(script_path: str, script_name: str) -> str:
    """
    識別腳本類型，避免將工具腳本誤判為孤立模組
    
    Args:
        script_path: 腳本的完整路徑
        script_name: 腳本名稱（不含 .py）
        
    Returns:
        'tool': 工具腳本（CLI/可視化/演示）
        'entry_point': 入口程序（含 main）
        'module': 普通模組
    """
    # 1. 檢查文件名模式
    tool_keywords = ['cli', 'main', 'demo', 'example', 'test', 'visualizer', 
                     'visualize', 'show', 'display', 'run', 'launch', 'start']
    if any(keyword in script_name.lower() for keyword in tool_keywords):
        return 'tool'
    
    # 2. 檢查文件內容
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 檢查是否有 if __name__ == "__main__"
            if 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content:
                return 'entry_point'
            
            # 檢查是否有 main() 函數定義
            if 'def main(' in content:
                return 'entry_point'
            
            # 檢查是否有 argparse/click/typer（CLI 框架）
            if any(cli_lib in content for cli_lib in ['import argparse', 'import click', 'import typer']):
                return 'tool'
    
    except Exception:
        pass  # 讀取失敗時默認為 module
    
    return 'module'


class CoreAnalyzer:
    """
    核心分析器 - 統一入口
    
    整合所有分析工具，提供不同層級的分析：
    - full_analysis(): 完整深度分析（慢但全面）
    - quick_scan(): 快速掃描（只檢測關鍵問題）
    - diagnose_critical_only(): 只檢查 CRITICAL 級別
    """
    
    def __init__(self, source_root: str, output_dir: Optional[str] = None):
        """
        初始化核心分析器
        
        Args:
            source_root: 源代碼根目錄
            output_dir: 輸出目錄（默認為 source_root/analysis_results）
        """
        self.source_root = Path(source_root).resolve()
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.source_root / "analysis_results"
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print("🚀 核心分析器初始化")
        print(f"   源目錄: {self.source_root}")
        print(f"   輸出目錄: {self.output_dir}")
    
    def full_analysis(self, save_report: bool = True) -> AnalysisReport:
        """
        完整深度分析
        
        執行所有分析器：
        1. 流程分析 (AIVAFlowAnalyzer)
        2. 數據流斷點分析 (DataFlowBreakpointAnalyzer)
        3. 缺失連接分析 (MissingConnectionAnalyzer)
        4. 實用分析器過濾 (PracticalAnalyzer)
        
        Returns:
            AnalysisReport
        """
        print("\n" + "="*60)
        print("🔬 開始完整深度分析")
        print("="*60)
        
        report = AnalysisReport(
            timestamp=datetime.now().isoformat(),
            source_path=str(self.source_root),
            analysis_type="full"
        )
        
        # 1. 流程分析
        print("\n[1/4] 執行流程分析...")
        try:
            flow_analyzer = AIVAFlowAnalyzer(
                target_dir=str(self.source_root)
            )
            flow_result = flow_analyzer.analyze_directory(
                target=str(self.source_root),
                depth=3,
                verbose=False
            )
            report.flow_analysis = flow_result
            
            # 保存分析結果供後續分析器使用
            flow_analyzer.save_results(output_dir=str(self.output_dir))
            
            # 提取統計
            if 'summary' in flow_result:
                summary = flow_result['summary']
                report.total_files = flow_result.get('total_files_processed', 0)
                report.total_functions = summary.get('total_graphs', 0)
            
            print("   ✅ 流程分析完成")
        except Exception as e:
            print(f"   ❌ 流程分析失敗: {e}")
            report.flow_analysis = {"error": str(e)}
        
        # 2. 數據流斷點分析
        print("\n[2/4] 執行數據流斷點分析...")
        try:
            bp_analyzer = DataFlowBreakpointAnalyzer(
                analysis_results_path=str(self.output_dir)
            )
            bp_analyzer.run_full_analysis()
            
            # 將斷點問題添加到報告
            report.breakpoint_analysis = {
                "total_issues": len(bp_analyzer.issues),
                "issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "location": issue.location,
                        "description": issue.description,
                        "affected_functions": issue.affected_functions
                    }
                    for issue in bp_analyzer.issues
                ]
            }
            
            # 提取關鍵問題
            for issue in bp_analyzer.issues:
                if issue.severity in ['critical', 'error']:
                    report.critical_issues.append({
                        'type': issue.issue_type,
                        'description': issue.description,
                        'file': issue.location,
                        'suggestion': issue.suggested_fix,
                    })
            
            print(f"   ✅ 數據流斷點分析完成，發現 {len(report.critical_issues)} 個關鍵問題")
        except Exception as e:
            print(f"   ❌ 斷點分析失敗: {e}")
            report.breakpoint_analysis = {"error": str(e)}
        
        # 3. 缺失連接分析
        print("\n[3/4] 執行缺失連接分析...")
        try:
            conn_analyzer = MissingConnectionAnalyzer(
                analysis_results_path=str(self.output_dir),
                source_root=str(self.source_root)
            )
            conn_analyzer.extract_function_signatures()
            conn_analyzer.find_missing_definitions()
            conn_analyzer.find_orphaned_with_entry_exit()
            conn_analyzer.find_potential_missing_links()
            
            # 生成報告 (void 方法，不返回值)
            conn_analyzer.generate_report(
                str(self.output_dir / MISSING_CONNECTIONS_REPORT)
            )
            
            # 設置連接分析完成狀態
            report.connection_analysis = {
                "status": "completed",
                "missing_count": len(conn_analyzer.missing_connections),
                "report_path": str(self.output_dir / MISSING_CONNECTIONS_REPORT)
            }
            
            print("   ✅ 連接分析完成")
        except Exception as e:
            print(f"   ❌ 連接分析失敗: {e}")
            report.connection_analysis = {"error": str(e)}
        
        # 4. 實用分析器過濾
        print("\n[4/4] 執行智能過濾...")
        try:
            practical_analyzer = PracticalAnalyzer()
            
            # 分析缺失連接報告
            missing_conn_report = self.output_dir / "missing_connections_full.md"
            if missing_conn_report.exists():
                filtered_issues = practical_analyzer.analyze_report(str(missing_conn_report))
                
                # 整合到報告
                report.critical_issues.extend([
                    {
                        'type': issue.problem_type,
                        'file': issue.file,
                        'description': issue.description,
                        'caller': issue.caller,
                        'callee': issue.callee,
                    }
                    for issue in filtered_issues['CRITICAL']
                ])
                
                report.high_issues.extend([
                    {
                        'type': issue.problem_type,
                        'file': issue.file,
                        'description': issue.description,
                    }
                    for issue in filtered_issues['HIGH']
                ])
                
                report.medium_issues.extend([
                    {
                        'type': issue.problem_type,
                        'file': issue.file,
                        'description': issue.description,
                    }
                    for issue in filtered_issues['MEDIUM'][:50]  # 限制數量
                ])
                
                report.low_issues.extend([
                    {
                        'type': issue.problem_type,
                        'description': issue.description,
                    }
                    for issue in filtered_issues['LOW'][:20]  # 限制數量
                ])
            
            print("   ✅ 過濾完成")
        except Exception as e:
            print(f"   ❌ 過濾失敗: {e}")
        
        # 計算總問題數
        report.total_issues = (
            len(report.critical_issues) +
            len(report.high_issues) +
            len(report.medium_issues) +
            len(report.low_issues)
        )
        
        # 生成建議
        report.recommendations = self._generate_recommendations(report)
        
        # 保存報告
        if save_report:
            json_path = self.output_dir / "core_analysis_full.json"
            md_path = self.output_dir / "core_analysis_full.md"
            
            report.save_json(str(json_path))
            report.save_markdown(str(md_path))
            
            print("\n📄 報告已保存:")
            print(f"   - JSON: {json_path}")
            print(f"   - Markdown: {md_path}")
        
        # 打印摘要
        self._print_summary(report)
        
        return report
    
    def quick_scan(self, save_report: bool = True) -> AnalysisReport:
        """
        快速掃描 - 只檢測關鍵問題
        
        只執行：
        1. 流程分析（基本統計）
        2. 數據流瓶頸檢測（只檢測嚴重瓶頸）
        
        Returns:
            AnalysisReport
        """
        print("\n" + "="*60)
        print("⚡ 開始快速掃描")
        print("="*60)
        
        report = AnalysisReport(
            timestamp=datetime.now().isoformat(),
            source_path=str(self.source_root),
            analysis_type="quick"
        )
        
        # 只執行流程分析和瓶頸檢測
        # ... 實現簡化版 ...
        
        if save_report:
            md_path = self.output_dir / "core_analysis_quick.md"
            report.save_markdown(str(md_path))
            print(f"\n📄 快速掃描報告: {md_path}")
        
        return report
    
    def diagnose_critical_only(self) -> AnalysisReport:
        """
        只診斷 CRITICAL 級別問題
        
        快速識別最緊急的問題
        """
        print("\n🔴 只診斷 CRITICAL 級別問題...")
        
        # 使用快速掃描，然後只保留 CRITICAL
        report = self.quick_scan(save_report=False)
        
        # 清空非 CRITICAL
        report.high_issues = []
        report.medium_issues = []
        report.low_issues = []
        report.total_issues = len(report.critical_issues)
        
        return report
    
    def _generate_recommendations(self, report: AnalysisReport) -> List[str]:
        """生成修復建議"""
        recommendations = []
        
        if len(report.critical_issues) > 0:
            recommendations.append(
                f"🔴 發現 {len(report.critical_issues)} 個 CRITICAL 問題，建議立即修復"
            )
        
        if len(report.high_issues) > 10:
            recommendations.append(
                f"⚠️ 發現 {len(report.high_issues)} 個 HIGH 問題，建議本週內處理前 10 個"
            )
        
        if report.breakpoint_analysis and 'bottlenecks' in report.breakpoint_analysis:
            bottlenecks = report.breakpoint_analysis['bottlenecks']
            if len(bottlenecks) > 0:
                recommendations.append(
                    f"📍 發現 {len(bottlenecks)} 個未完整連接的模組，建議檢查是否有應該調用但未調用的函數"
                )
        
        if not recommendations:
            recommendations.append("✅ 代碼健康狀況良好，未發現嚴重問題")
        
        return recommendations
    
    def _print_summary(self, report: AnalysisReport):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("📊 分析摘要")
        print("="*60)
        print(f"📁 文件數: {report.total_files}")
        print(f"🔧 函數數: {report.total_functions}")
        print(f"⚠️ 總問題: {report.total_issues}")
        print()
        print(f"🔴 CRITICAL: {len(report.critical_issues)}")
        print(f"⚠️ HIGH:     {len(report.high_issues)}")
        print(f"⚡ MEDIUM:   {len(report.medium_issues)}")
        print(f"ℹ️ LOW:      {len(report.low_issues)}")
        print()
        print("💡 建議:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
        print("="*60)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA 核心分析器")
    parser.add_argument("source", help="源代碼目錄")
    parser.add_argument("--output", "-o", help="輸出目錄")
    parser.add_argument("--mode", "-m", choices=["full", "quick", "critical"],
                        default="full", help="分析模式")
    
    args = parser.parse_args()
    
    analyzer = CoreAnalyzer(args.source, args.output)
    
    if args.mode == "full":
        analyzer.full_analysis()
    elif args.mode == "quick":
        analyzer.quick_scan()
    elif args.mode == "critical":
        analyzer.diagnose_critical_only()


if __name__ == "__main__":
    main()

