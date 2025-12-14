"""
數據流斷點分析器
================
專門用於發現數據流中的斷點、瓶頸和潛在問題

分析類型：
1. 數據流斷點 - 連接中斷的位置
2. 單向死路 - 只輸出沒輸入或只輸入沒輸出
3. 孤島檢測 - 完全孤立的模組群
4. 瓶頸分析 - 過度依賴的節點
5. 循環依賴 - 可能導致問題的循環
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class BreakpointIssue:
    """數據流斷點問題"""
    issue_type: str  # "斷點", "死路", "孤島", "瓶頸", "循環"
    severity: str  # "critical", "warning", "info"
    location: str
    description: str
    affected_functions: List[str] = field(default_factory=list)
    suggested_fix: str = ""


class DataFlowBreakpointAnalyzer:
    """數據流斷點分析器"""
    
    def __init__(self, analysis_results_path: str):
        self.results_path = Path(analysis_results_path)
        self.analysis_data = self._load_analysis_results()
        self.issues: List[BreakpointIssue] = []
        
        # 構建圖結構
        self.graph = defaultdict(set)  # script -> {called_scripts}
        self.reverse_graph = defaultdict(set)  # script -> {caller_scripts}
        self.all_scripts = set()
        
    def _load_analysis_results(self) -> Dict:
        """載入分析結果"""
        json_path = self.results_path / "analysis_results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"找不到分析結果: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def build_flow_graph(self):
        """構建數據流圖"""
        print("🔨 構建數據流圖...")
        
        flow_chains = self.analysis_data.get('flow_chains', [])
        
        for chain in flow_chains:
            for i in range(len(chain) - 1):
                source = chain[i]
                target = chain[i + 1]
                
                self.graph[source].add(target)
                self.reverse_graph[target].add(source)
                self.all_scripts.add(source)
                self.all_scripts.add(target)
        
        print(f"   ✅ 構建完成: {len(self.all_scripts)} 個節點, {sum(len(v) for v in self.graph.values())} 條邊")
    
    def detect_breakpoints(self):
        """檢測數據流斷點"""
        print("\n🔍 檢測數據流斷點...")
        
        unconnected_report = self.analysis_data.get('unconnected_report', {})
        if not isinstance(unconnected_report, dict):
            print("   ⚠️ 未找到未連接函數報告，跳過斷點檢測")
            return
        
        orphaned = unconnected_report.get('orphaned_functions', [])
        unreachable = unconnected_report.get('unreachable_functions', [])
        
        # 按文件分組未連接函數
        orphaned_by_file = defaultdict(list)
        unreachable_by_file = defaultdict(list)
        
        for func in orphaned:
            file_path = func.get('file_path', '')
            orphaned_by_file[file_path].append(func['function_name'])
        
        for func in unreachable:
            file_path = func.get('file_path', '')
            unreachable_by_file[file_path].append(func['function_name'])
        
        # 找出斷點：有孤立函數且有不可達函數的文件
        breakpoint_files = set(orphaned_by_file.keys()) & set(unreachable_by_file.keys())
        
        for file_path in breakpoint_files:
            issue = BreakpointIssue(
                issue_type="數據流斷點",
                severity="warning",
                location=Path(file_path).name,
                description=f"此文件同時包含孤立函數({len(orphaned_by_file[file_path])}個)和不可達函數({len(unreachable_by_file[file_path])}個)",
                affected_functions=orphaned_by_file[file_path] + unreachable_by_file[file_path],
                suggested_fix="檢查函數調用鏈是否完整，可能存在中間斷開"
            )
            self.issues.append(issue)
        
        print(f"   🔴 發現 {len(breakpoint_files)} 個潛在斷點")
    
    def detect_dead_ends(self):
        """檢測死路（單向連接）"""
        print("\n🔍 檢測數據流死路...")
        
        # 只有輸出沒有輸入（源頭除外）
        sources = set(self.graph.keys()) - set(self.reverse_graph.keys())
        
        # 只有輸入沒有輸出（終點除外）
        _sinks = set(self.reverse_graph.keys()) - set(self.graph.keys())
        
        # 找出異常的死路（不在主要流程中）
        for script in self.all_scripts:
            if script not in self.graph and script not in self.reverse_graph:
                continue
            
            # 完全孤立（既不是源也不是匯）
            out_degree = len(self.graph.get(script, []))
            in_degree = len(self.reverse_graph.get(script, []))
            
            if out_degree > 0 and in_degree == 0:
                # 只有輸出沒輸入
                if script not in sources or out_degree > 3:
                    issue = BreakpointIssue(
                        issue_type="單向死路",
                        severity="info",
                        location=Path(script).name,
                        description=f"只有輸出({out_degree}個)沒有輸入，可能是未正確連接的數據源",
                        suggested_fix="檢查是否應該有上游調用者"
                    )
                    self.issues.append(issue)
            
            elif in_degree > 0 and out_degree == 0:
                # 只有輸入沒輸出
                if in_degree > 3:
                    issue = BreakpointIssue(
                        issue_type="單向死路",
                        severity="warning",
                        location=Path(script).name,
                        description=f"只有輸入({in_degree}個)沒有輸出，數據流在此終止",
                        suggested_fix="檢查是否應該有下游處理"
                    )
                    self.issues.append(issue)
        
        print(f"   🟡 發現 {len([i for i in self.issues if i.issue_type == '單向死路'])} 個單向死路")
    
    def detect_isolated_islands(self):
        """檢測孤島（完全孤立的模組群）"""
        print("\n🔍 檢測孤島模組...")
        
        if not self.all_scripts:
            return
        
        # 使用 BFS 找連通分量
        visited = set()
        islands = []
        
        def bfs(start):
            island = set()
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                
                visited.add(node)
                island.add(node)
                
                # 雙向搜索
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
                
                for neighbor in self.reverse_graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            return island
        
        for script in self.all_scripts:
            if script not in visited:
                island = bfs(script)
                if len(island) > 1:  # 至少2個節點才算孤島
                    islands.append(island)
        
        # 找出小型孤島（可能是問題）
        main_island_size = max(len(island) for island in islands) if islands else 0
        
        for island in islands:
            if len(island) < main_island_size * 0.1 and len(island) > 1:
                issue = BreakpointIssue(
                    issue_type="孤立孤島",
                    severity="warning",
                    location=f"{len(island)}個腳本的孤島",
                    description=f"發現{len(island)}個腳本形成孤立群組，與主流程無連接",
                    affected_functions=[Path(s).name for s in list(island)[:5]],
                    suggested_fix="檢查這些模組是否應該與主系統連接"
                )
                self.issues.append(issue)
        
        print(f"   🟠 發現 {len([i for i in self.issues if i.issue_type == '孤立孤島'])} 個孤島")
    
    def detect_bottlenecks(self):
        """檢測未完整連接的模組（定義了很多函數但只有少數被實際連接）"""
        print("\n🔍 檢測未完整連接的模組...")
        
        # 從 analysis_results.json 讀取每個腳本的函數詳情
        function_details = self.analysis_data.get('function_details', {})
        script_functions = function_details.get('script_functions', {})
        
        if not script_functions:
            print("   ⚠️ 未找到腳本函數詳情，跳過未完整連接檢測")
            return
        
        # 分析每個腳本的連接完整度
        for script in self.all_scripts:
            script_name = Path(script).stem
            script_info = script_functions.get(script_name, {})
            
            # 潛在輸出：定義的所有導出函數
            export_functions = script_info.get('export_functions', [])
            potential_outputs = len(export_functions)
            
            # 實際輸出：在 flow_chains 中實際被其他模組調用的次數
            actual_outputs = len(self.graph.get(script, []))
            
            # 潛在輸入：定義的入口函數數量
            entry_points = script_info.get('entry_points', [])
            potential_inputs = len(entry_points) if entry_points else 1
            
            # 實際輸入：在 flow_chains 中實際被調用的次數
            actual_inputs = len(self.reverse_graph.get(script, []))
            
            # 檢測未連接的輸出接口（定義了很多函數但很少被使用）
            if potential_outputs >= 5 and actual_outputs < potential_outputs * 0.4:
                missing_outputs = potential_outputs - actual_outputs
                issue = BreakpointIssue(
                    issue_type="未完整連接的輸出接口",
                    severity="warning",
                    location=Path(script).name,
                    description=f"定義了 {potential_outputs} 個導出函數，但只有 {actual_outputs} 個在數據流中被其他模組調用（缺失 {missing_outputs} 個連接）",
                    affected_functions=export_functions[:10],  # 顯示前10個函數
                    suggested_fix="檢查這些導出函數是否應該被其他模組調用。如果不需要被外部調用，考慮將其標記為內部函數（以 _ 開頭）"
                )
                self.issues.append(issue)
            
            # 檢測未連接的入口接口（有入口但很少有模組調用）
            if potential_inputs >= 2 and actual_inputs == 0 and actual_outputs == 0:
                issue = BreakpointIssue(
                    issue_type="孤立模組",
                    severity="info",
                    location=Path(script).name,
                    description=f"定義了 {potential_inputs} 個入口函數，但未被任何模組調用，也未調用其他模組",
                    affected_functions=entry_points[:10],
                    suggested_fix="檢查此模組是否應該整合到數據流中，或者是否為獨立工具腳本"
                )
                self.issues.append(issue)
        
        incomplete_count = len([i for i in self.issues if i.issue_type in ['未完整連接的輸出接口', '孤立模組']])
        print(f"   🟡 發現 {incomplete_count} 個未完整連接的模組")
    
    def detect_circular_dependencies(self):
        """檢測循環依賴"""
        print("\n🔍 檢測循環依賴...")
        
        def find_cycle(start, path, visited_in_path):
            if start in visited_in_path:
                # 找到循環
                cycle_start = path.index(start)
                return path[cycle_start:]
            
            if start in visited:
                return None
            
            visited.add(start)
            visited_in_path.add(start)
            path.append(start)
            
            for neighbor in self.graph.get(start, []):
                cycle = find_cycle(neighbor, path[:], visited_in_path.copy())
                if cycle:
                    return cycle
            
            return None
        
        visited = set()
        cycles = []
        
        for script in self.all_scripts:
            if script not in visited:
                cycle = find_cycle(script, [], set())
                if cycle and len(cycle) > 1:
                    cycles.append(cycle)
        
        for cycle in cycles:
            issue = BreakpointIssue(
                issue_type="循環依賴",
                severity="warning",
                location=f"{len(cycle)}個腳本的循環",
                description=f"發現循環依賴: {' -> '.join([Path(s).name for s in cycle[:4]])}...",
                affected_functions=[Path(s).name for s in cycle],
                suggested_fix="重構以打破循環依賴，使用依賴注入或介面"
            )
            self.issues.append(issue)
        
        print(f"   🟡 發現 {len(cycles)} 個循環依賴")
    
    def analyze_missing_connections(self):
        """分析缺失的連接（基於命名和功能推測）"""
        print("\n🔍 分析潛在缺失連接...")
        
        unconnected_report = self.analysis_data.get('unconnected_report', {})
        if not isinstance(unconnected_report, dict):
            print("   ⚠️ 未找到未連接函數報告，跳過缺失連接分析")
            return
        
        orphaned = unconnected_report.get('orphaned_functions', [])
        unreachable = unconnected_report.get('unreachable_functions', [])
        
        # 根據函數名推測可能的連接
        for orphaned_func in orphaned:
            func_name = orphaned_func.get('function_name', '')
            file_name = orphaned_func.get('file_name', '')
            
            # 檢查是否有類似名稱的不可達函數
            for unreachable_func in unreachable:
                unreachable_name = unreachable_func.get('function_name', '')
                
                # 簡單的名稱相似度檢查
                if (func_name in unreachable_name or unreachable_name in func_name or
                    func_name.replace('_', '') in unreachable_name.replace('_', '')):
                    
                    issue = BreakpointIssue(
                        issue_type="潛在缺失連接",
                        severity="info",
                        location=file_name,
                        description=f"孤立函數 '{func_name}' 可能與不可達函數 '{unreachable_name}' 相關",
                        affected_functions=[func_name, unreachable_name],
                        suggested_fix="檢查這兩個函數是否應該相互調用"
                    )
                    self.issues.append(issue)
        
        missing_count = len([i for i in self.issues if i.issue_type == '潛在缺失連接'])
        print(f"   💡 發現 {missing_count} 個潛在缺失連接")
    
    def generate_report(self, output_path: str | None = None):
        """生成斷點分析報告"""
        if output_path is None:
            report_path = self.results_path / "dataflow_breakpoint_analysis.md"
        else:
            report_path = Path(output_path)
        
        print("\n📝 生成報告...")
        
        # 按嚴重程度排序
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        sorted_issues = sorted(self.issues, key=lambda x: severity_order.get(x.severity, 3))
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 數據流斷點分析報告\n\n")
            f.write(f"**分析時間:** {Path(self.results_path).name}\n\n")
            
            # 摘要
            f.write("## 📊 問題摘要\n\n")
            f.write(f"- **總問題數:** {len(self.issues)}\n")
            
            issue_types = defaultdict(int)
            for issue in self.issues:
                issue_types[issue.issue_type] += 1
            
            for issue_type, count in issue_types.items():
                f.write(f"- **{issue_type}:** {count} 個\n")
            f.write("\n")
            
            # 嚴重程度統計
            severity_counts = defaultdict(int)
            for issue in self.issues:
                severity_counts[issue.severity] += 1
            
            f.write("## 🚨 嚴重程度分布\n\n")
            f.write(f"- 🔴 **Critical (嚴重):** {severity_counts['critical']} 個\n")
            f.write(f"- 🟡 **Warning (警告):** {severity_counts['warning']} 個\n")
            f.write(f"- 🔵 **Info (信息):** {severity_counts['info']} 個\n\n")
            
            # 詳細問題列表
            f.write("## 📋 詳細問題列表\n\n")
            
            current_type = None
            for issue in sorted_issues:
                if issue.issue_type != current_type:
                    current_type = issue.issue_type
                    f.write(f"### {current_type}\n\n")
                
                severity_icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}.get(issue.severity, '⚪')
                
                f.write(f"#### {severity_icon} {issue.location}\n\n")
                f.write(f"**描述:** {issue.description}\n\n")
                
                if issue.affected_functions:
                    f.write(f"**影響函數/模組:** {', '.join(issue.affected_functions[:10])}\n")
                    if len(issue.affected_functions) > 10:
                        f.write(f"... 還有 {len(issue.affected_functions) - 10} 個\n")
                    f.write("\n")
                
                if issue.suggested_fix:
                    f.write(f"**建議修復:** {issue.suggested_fix}\n\n")
                
                f.write("---\n\n")
            
            # 優化建議
            f.write("## 💡 整體優化建議\n\n")
            
            if severity_counts['critical'] > 0:
                f.write("### 🔴 優先處理（Critical）\n\n")
                f.write("以下問題應該優先解決：\n\n")
                for issue in sorted_issues:
                    if issue.severity == 'critical':
                        f.write(f"1. **{issue.location}:** {issue.description}\n")
                f.write("\n")
            
            if severity_counts['warning'] > 0:
                f.write("### 🟡 建議處理（Warning）\n\n")
                f.write("以下問題建議盡快解決以提高系統穩定性：\n\n")
                warning_issues = [i for i in sorted_issues if i.severity == 'warning']
                for issue in warning_issues[:5]:
                    f.write(f"1. **{issue.location}:** {issue.description}\n")
                if len(warning_issues) > 5:
                    f.write(f"\n... 還有 {len(warning_issues) - 5} 個警告問題\n")
                f.write("\n")
            
            # 健康度評分
            total_scripts = len(self.all_scripts)
            issue_rate = len(self.issues) / total_scripts if total_scripts > 0 else 0
            
            f.write("## 🏥 系統健康度評分\n\n")
            
            if issue_rate < 0.1:
                health_score = "優秀 ✅"
                health_comment = "系統數據流結構良好，問題較少"
            elif issue_rate < 0.3:
                health_score = "良好 🟢"
                health_comment = "系統整體健康，但有改進空間"
            elif issue_rate < 0.5:
                health_score = "一般 🟡"
                health_comment = "存在一些問題，建議優化"
            else:
                health_score = "需改進 🔴"
                health_comment = "存在較多問題，建議重點優化"
            
            f.write(f"**評分:** {health_score}\n\n")
            f.write(f"**評語:** {health_comment}\n\n")
            f.write(f"- 總腳本數: {total_scripts}\n")
            f.write(f"- 問題腳本比例: {issue_rate*100:.1f}%\n")
            f.write(f"- 數據流連接數: {sum(len(v) for v in self.graph.values())}\n\n")
        
        print(f"   ✅ 報告已保存: {report_path}")
        return report_path
    
    def run_full_analysis(self):
        """執行完整分析"""
        print("="*60)
        print("🔬 數據流斷點分析器")
        print("="*60)
        
        self.build_flow_graph()
        self.detect_breakpoints()
        self.detect_dead_ends()
        self.detect_isolated_islands()
        self.detect_bottlenecks()
        self.detect_circular_dependencies()
        self.analyze_missing_connections()
        
        report_path = self.generate_report()
        
        print("\n" + "="*60)
        print(f"✅ 分析完成！共發現 {len(self.issues)} 個問題")
        print("="*60)
        
        return report_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="數據流斷點分析器")
    parser.add_argument("--analysis-dir", "-d", 
                       default="./analysis_with_unconnected",
                       help="分析結果目錄")
    parser.add_argument("--output", "-o",
                       help="輸出報告路徑")
    
    args = parser.parse_args()
    
    try:
        analyzer = DataFlowBreakpointAnalyzer(args.analysis_dir)
        report_path = analyzer.run_full_analysis()
        
        print(f"\n📄 查看報告: {report_path}")
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
