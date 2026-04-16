#!/usr/bin/env python3
"""
實用錯誤分析器 - 分級展示，確保重要問題不遺漏

策略：
1. 不過度過濾，寧可多列
2. 按影響分級（CRITICAL > HIGH > MEDIUM > LOW）
3. 智能分組，方便查看
4. 提供快速定位和修復建議
"""

from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass
class Issue:
    """問題"""
    severity: str
    file: str
    caller: str
    callee: str
    problem_type: str
    description: str
    line_in_report: int


class PracticalAnalyzer:
    """實用分析器"""
    
    # 定義常量
    MARKDOWN_SEPARATOR = "---\n\n"
    CODE_BLOCK_BASH = "```bash\n"
    CODE_BLOCK_END = "```\n\n"
    
    def __init__(self):
        # 擴展的 Python 內建（更全面的過濾）
        self.python_builtins = {
            'len', 'str', 'int', 'float', 'dict', 'list', 'set', 'tuple',
            'print', 'open', 'range', 'enumerate', 'zip', 'map', 'filter',
            'sum', 'min', 'max', 'sorted', 'reversed', 'any', 'all',
            'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
            'type', 'id', 'hex', 'oct', 'bin', 'chr', 'ord',
            'abs', 'bool', 'bytes', 'callable', 'classmethod', 'compile',
            'complex', 'delattr', 'dir', 'divmod', 'eval', 'exec',
            'frozenset', 'globals', 'hash', 'help', 'input', 'iter',
            'locals', 'memoryview', 'next', 'object', 'pow', 'property',
            'repr', 'round', 'slice', 'staticmethod', 'super', 'vars',
        }
        
        # 字符串/列表/字典方法（內建數據類型的方法）
        self.builtin_methods = {
            # 字符串方法
            'lower', 'upper', 'strip', 'lstrip', 'rstrip', 'split', 'rsplit',
            'join', 'replace', 'format', 'format_map', 'startswith', 'endswith',
            'find', 'rfind', 'index', 'rindex', 'count', 'capitalize', 'title',
            'swapcase', 'center', 'ljust', 'rjust', 'zfill', 'encode', 'decode',
            'isalpha', 'isdigit', 'isalnum', 'isspace', 'isupper', 'islower',
            'istitle', 'isdecimal', 'isnumeric', 'isidentifier', 'isprintable',
            # 列表方法（已移除重複：count, index, copy, clear）
            'append', 'extend', 'insert', 'remove', 'pop', 'sort', 'reverse',
            # 字典方法（已移除重複：copy, clear）
            'get', 'keys', 'values', 'items', 'update', 'setdefault', 'popitem', 'fromkeys',
            # 文件方法
            'read', 'write', 'close', 'readline', 'readlines', 'writelines',
            'seek', 'tell', 'flush', 'fileno', 'isatty', 'truncate',
            # pathlib 方法
            'exists', 'is_file', 'is_dir', 'mkdir', 'rmdir', 'unlink', 'rename',
            'glob', 'rglob', 'stat', 'chmod', 'touch', 'resolve', 'absolute',
            'relative_to', 'with_name', 'with_suffix', 'parent', 'parents', 'name',
            'stem', 'suffix', 'suffixes', 'as_posix', 'as_uri', 'joinpath',
        }
        
        # 關鍵文件（這些文件的問題優先級提升）
        self.critical_files = [
            'app.py', 'main.py', '__init__.py',
            'attack_executor.py', 'attack_validator.py', 'attack_planner.py',
            'analysis_engine.py', 'risk_assessment',
            'unified_function_caller.py', 'enhanced_unified_caller.py',
            'core_service_coordinator.py', 'ai_controller.py',
            'capability_orchestrator.py', 'capability_registry.py',
            'session_state_manager.py', 'storage_manager.py',
        ]
        
        # 關鍵功能關鍵詞
        self.critical_keywords = [
            'xss', 'sql', 'idor', 'csrf', 'auth', 'jwt', 'security', 'exploit',
            'init', 'start', 'execute', 'run', 'main', 'core',
            'validate', 'sanitize', 'verify', 'check',
            'train', 'predict', 'analyze', 'detect',
            'save', 'load', 'record', 'persist',
        ]
        
    def analyze_report(self, report_file: str) -> dict[str, list[Issue]]:
        """分析報告並分級"""
        print(f"📖 讀取報告: {report_file}")
        
        with open(report_file, encoding='utf-8') as f:
            content = f.read()
            
        # 提取三種類型的問題
        issues: dict[str, list[Issue]] = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': [],
            'LOW': [],
            'NOISE': []  # 明確的噪音，但保留以防萬一
        }
        
        # 解析定義缺失
        issues_parsed = self._parse_definition_missing(content)
        for issue in issues_parsed:
            severity = self._classify_issue(issue)
            issues[severity].append(issue)
            
        # 解析調用缺失
        issues_parsed = self._parse_call_missing(content)
        for issue in issues_parsed:
            severity = self._classify_issue(issue)
            issues[severity].append(issue)
            
        # 解析潛在連接缺失
        issues_parsed = self._parse_potential_missing(content)
        for issue in issues_parsed:
            severity = self._classify_issue(issue)
            issues[severity].append(issue)
            
        # 去重相似問題
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            issues[severity] = self._deduplicate_issues(issues[severity])
        
        # 統計
        print("\n📊 問題分級統計:")
        print(f"  🔴 CRITICAL: {len(issues['CRITICAL'])} 個")
        print(f"  ⚠️  HIGH:     {len(issues['HIGH'])} 個")
        print(f"  ⚡ MEDIUM:   {len(issues['MEDIUM'])} 個")
        print(f"  ℹ️  LOW:      {len(issues['LOW'])} 個")
        print(f"  🗑️  NOISE:    {len(issues['NOISE'])} 個（已過濾）")
        
        return issues
        
    def _parse_definition_missing(self, content: str) -> list[Issue]:
        """解析定義缺失"""
        issues: list[Issue] = []
        
        # 找到定義缺失部分
        match = re.search(r'### 定義缺失(.*?)(?=###|\Z)', content, re.DOTALL)
        if not match:
            return issues
            
        section = match.group()
        
        # 解析每個問題
        pattern = r'#### \d+\. ([\U0001f534\U0001f7e1\U0001f535]) (\w+) → (\w+)\s+\*\*源文件:\*\* `([^`]+)`'
        
        for match in re.finditer(pattern, section):
            emoji, caller, callee, file = match.groups()
            
            issues.append(Issue(
                severity=emoji,
                file=file,
                caller=caller,
                callee=callee,
                problem_type='定義缺失',
                description=f"{caller} 調用 {callee} 但找不到定義",
                line_in_report=0
            ))
            
        return issues
        
    def _parse_call_missing(self, content: str) -> list[Issue]:
        """解析調用缺失"""
        issues: list[Issue] = []
        
        match = re.search(r'### 調用缺失(.*?)(?=###|\Z)', content, re.DOTALL)
        if not match:
            return issues
            
        section = match.group()
        
        pattern = r'#### \d+\. ([\U0001f534\U0001f7e1\U0001f535]) (\w+)\s+\*\*文件:\*\* `([^`]+)`'
        
        for match in re.finditer(pattern, section):
            emoji, function, file = match.groups()
            
            issues.append(Issue(
                severity=emoji,
                file=file,
                caller='(無調用者)',
                callee=function,
                problem_type='調用缺失',
                description=f"{function} 有接口但從未被調用",
                line_in_report=0
            ))
            
        return issues
        
    def _parse_potential_missing(self, content: str) -> list[Issue]:
        """解析潛在連接缺失"""
        issues: list[Issue] = []
        
        match = re.search(r'### 潛在連接缺失(.*?)(?=###|\Z)', content, re.DOTALL)
        if not match:
            return issues
            
        section = match.group()
        
        pattern = r'#### \d+\. ([\U0001f534\U0001f7e1\U0001f535]) (\w+) → (\w+)\s+\*\*源文件:\*\* `([^`]+)`'
        
        for match in re.finditer(pattern, section):
            emoji, caller, callee, file = match.groups()
            
            # 潛在問題默認降低一級
            issues.append(Issue(
                severity=emoji,
                file=file,
                caller=caller,
                callee=callee,
                problem_type='潛在缺失',
                description=f"{caller} 可能需要調用 {callee}",
                line_in_report=0
            ))
            
        return issues
        
    def _deduplicate_issues(self, issues: list[Issue]) -> list[Issue]:
        """
        智能去重：合併相似問題
        - 相同 caller → callee 只保留 1 個
        - 相同函數名但不同文件，按文件分組
        """
        if not issues:
            return issues
        
        # 按 (caller, callee) 分組
        grouped = defaultdict(list)
        for issue in issues:
            key = (issue.caller, issue.callee)
            grouped[key].append(issue)
        
        deduplicated = []
        for (caller, callee), group in grouped.items():
            if len(group) == 1:
                # 唯一問題，直接保留
                deduplicated.append(group[0])
            else:
                # 多個相同問題，合併為一個並註明出現次數
                first = group[0]
                files = [issue.file for issue in group]
                unique_files = list(set(files))
                
                # 創建合併的問題
                merged = Issue(
                    severity=first.severity,
                    file=f"{unique_files[0]} (+{len(unique_files)-1} 個文件)" if len(unique_files) > 1 else first.file,
                    caller=first.caller,
                    callee=first.callee,
                    problem_type=first.problem_type,
                    description=f"{first.description} [出現 {len(group)} 次]",
                    line_in_report=first.line_in_report
                )
                deduplicated.append(merged)
        
        return deduplicated
    
    def _classify_issue(self, issue: Issue) -> str:
        """對問題進行分級"""
        
        # 1. 明確的內建方法/函數 -> NOISE
        if issue.callee in self.python_builtins or issue.callee in self.builtin_methods:
            return 'NOISE'
            
        # 2. 關鍵文件中的問題 -> 優先級提升
        is_critical_file = any(cf in issue.file.lower() for cf in self.critical_files)
        
        # 3. 關鍵功能相關 -> 優先級提升
        has_critical_keyword = any(
            kw in issue.caller.lower() or kw in issue.callee.lower() 
            for kw in self.critical_keywords
        )
        
        # 4. 調用缺失（有接口但未使用）-> 比較重要
        is_unused_function = issue.problem_type == '調用缺失'
        
        # 5. 潛在缺失 -> 降低優先級
        is_potential = issue.problem_type == '潛在缺失'
        
        # 綜合判斷
        if is_critical_file and has_critical_keyword or is_unused_function and has_critical_keyword:
            return 'CRITICAL'
        elif is_critical_file or (is_unused_function and not is_potential):
            return 'HIGH'
        elif has_critical_keyword or issue.problem_type == '定義缺失':
            return 'MEDIUM'
        elif is_potential:
            return 'LOW'
        else:
            return 'MEDIUM'
            
    def _write_statistics(self, f, issues: dict[str, list[Issue]]) -> None:
        """寫入統計摘要
        
        複雜度: A (2) - 簡單統計
        """
        total = sum(len(v) for k, v in issues.items() if k != 'NOISE')
        f.write("【## 📊 統計摘要\n\n")
        f.write(f"- **總問題數:** {total}\n")
        f.write(f"- **🔴 CRITICAL:** {len(issues['CRITICAL'])} 個\n")
        f.write(f"- **⚠️ HIGH:** {len(issues['HIGH'])} 個\n")
        f.write(f"- **⚡ MEDIUM:** {len(issues['MEDIUM'])} 個\n")
        f.write(f"- **ℹ️ LOW:** {len(issues['LOW'])} 個\n")
        f.write(f"- **🗑️ 已過濾噪音:** {len(issues['NOISE'])} 個\n\n")
    
    def _write_critical_issues(self, f, issues: list[Issue]) -> None:
        """寫入 CRITICAL 問題
        
        複雜度: B (6) - 雙層循環
        """
        if not issues:
            return
        
        f.write("【## 🔴 CRITICAL - 必須立即修復\n\n")
        f.write("這些問題會直接影響核心功能，必須優先處理。\n\n")
        
        by_file = defaultdict(list)
        for issue in issues:
            by_file[issue.file].append(issue)
        
        for file, file_issues in sorted(by_file.items()):
            f.write(f"### 📄 {file} ({len(file_issues)} 個問題)\n\n")
            
            for i, issue in enumerate(file_issues[:20], 1):
                f.write(f"**{i}. {issue.problem_type}:** `{issue.caller}` → `{issue.callee}`\n")
                f.write(f"   - {issue.description}\n\n")
            
            if len(file_issues) > 20:
                f.write(f"   *... 還有 {len(file_issues) - 20} 個問題*\n\n")
        
        f.write("---\n\n")
    
    def _write_high_issues(self, f, issues: list[Issue]) -> None:
        """寫入 HIGH 問題
        
        複雜度: B (7) - 雙層循環 + 條件
        """
        if not issues:
            return
        
        f.write("【## ⚠️ HIGH - 建議優先修復\n\n")
        f.write("這些問題可能影響重要功能，建議本週內處理。\n\n")
        
        by_file = defaultdict(list)
        for issue in issues:
            by_file[issue.file].append(issue)
        
        for file, file_issues in sorted(by_file.items())[:10]:
            f.write(f"### 📄 {file} ({len(file_issues)} 個問題)\n\n")
            
            for i, issue in enumerate(file_issues[:10], 1):
                f.write(f"**{i}. {issue.problem_type}:** `{issue.caller}` → `{issue.callee}`\n")
                f.write(f"   - {issue.description}\n\n")
            
            if len(file_issues) > 10:
                f.write(f"   *... 還有 {len(file_issues) - 10} 個問題*\n\n")
        
        if len(by_file) > 10:
            f.write(f"\n*... 還有 {len(by_file) - 10} 個檔案的 HIGH 問題*\n\n")
        
        f.write("---\n\n")
    
    def _write_medium_low_issues(self, f, medium_issues: list[Issue], low_issues: list[Issue]) -> None:
        """寫入 MEDIUM 和 LOW 問題
        
        複雜度: A (4) - 簡單統計
        """
        if medium_issues:
            f.write("【## ⚡ MEDIUM - 按需修復\n\n")
            f.write("這些問題影響有限，可以根據實際情況決定是否修復。\n\n")
            f.write(f"**總計:** {len(medium_issues)} 個問題\n\n")
            
            by_type: dict[str, int] = defaultdict(int)
            for issue in medium_issues:
                by_type[issue.problem_type] += 1
            
            f.write("**按類型分布:**\n\n")
            for ptype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {ptype}: {count} 個\n")
            
            f.write("\n**示例問題（前5個）:**\n\n")
            for i, issue in enumerate(medium_issues[:5], 1):
                f.write(f"{i}. `{issue.file}`: {issue.caller} → {issue.callee}\n")
            
            f.write("\n---\n\n")
        
        if low_issues:
            f.write("【## ℹ️ LOW - 可忽略\n\n")
            f.write(f"這些是低優先級問題，通常可以忽略。總計: {len(low_issues)} 個\n\n")
            f.write("---\n\n")
    
    def _write_action_plan(self, f, issues: dict[str, list[Issue]]) -> None:
        """寫入行動計劃
        
        複雜度: A (4) - 條件判斷
        """
        f.write("【## 📋 建議行動計劃\n\n")
        
        if issues['CRITICAL']:
            f.write("### 🚨 今天（必須）\n\n")
            f.write(f"修復 {len(issues['CRITICAL'])} 個 CRITICAL 問題\n\n")
            
            critical_files = set(issue.file for issue in issues['CRITICAL'])
            f.write("**需要檢查的檔案:**\n\n")
            for file in sorted(critical_files)[:10]:
                count = sum(1 for i in issues['CRITICAL'] if i.file == file)
                f.write(f"- [ ] `{file}` ({count} 個問題)\n")
            f.write("\n")
        
        if issues['HIGH']:
            f.write("### 📅 本週\n\n")
            f.write(f"處理 {min(20, len(issues['HIGH']))} 個 HIGH 問題（共 {len(issues['HIGH'])} 個）\n\n")
        
        if issues['MEDIUM']:
            f.write("### 📆 兩週內\n\n")
            f.write(f"根據實際需要處理 MEDIUM 問題（共 {len(issues['MEDIUM'])} 個）\n\n")
    
    def _write_quick_start_guide(self, f, issues: dict[str, list[Issue]]) -> None:
        """寫入快速開始指南
        
        複雜度: A (2) - 簡單條件
        """
        f.write("【## 🚀 快速開始\n\n")
        f.write("### 1. 修復 CRITICAL 問題\n\n")
        f.write("```bash\n")
        f.write("# 檢查第一個 CRITICAL 檔案\n")
        if issues['CRITICAL']:
            first_file = issues['CRITICAL'][0].file
            f.write(f"code services/core/aiva_core/{first_file}\n")
        f.write("```\n\n")
        
        f.write("### 2. 運行類型檢查\n\n")
        f.write("```bash\n")
        f.write("# 使用 Pylance/mypy 檢查類型錯誤\n")
        f.write("mypy services/core/aiva_core --ignore-missing-imports\n")
        f.write("```\n\n")
        
        f.write("### 3. 運行測試\n\n")
        f.write("```bash\n")
        f.write("# 確保修復後功能正常\n")
        f.write("pytest services/core/aiva_core/tests\n")
        f.write("```\n\n")
    
    def generate_practical_report(self, issues: dict[str, list[Issue]], output_file: str):
        """生成實用報告
        
        複雜度: A (1) - 主協調函數，單一職責
        原複雜度: D (27) -> 重構後: A (1) [96% 降低]
        """
        print(f"\n📝 生成實用報告: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("【# 🎯 實用錯誤報告 - 分級展示\n\n")
            f.write(f"生成時間: {self._get_timestamp()}\n\n")
            
            self._write_statistics(f, issues)
            self._write_critical_issues(f, issues['CRITICAL'])
            self._write_high_issues(f, issues['HIGH'])
            self._write_medium_low_issues(f, issues['MEDIUM'], issues['LOW'])
            self._write_action_plan(f, issues)
            self._write_quick_start_guide(f, issues)
        
        print("  ✅ 報告已保存")
        
    def generate_quick_fix_list(self, issues: dict[str, list[Issue]], output_file: str):
        """生成快速修復清單"""
        print(f"📝 生成快速修復清單: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# ✅ 快速修復清單\n\n")
            
            # CRITICAL 清單
            if issues['CRITICAL']:
                f.write("## 🔴 CRITICAL 問題清單\n\n")
                for i, issue in enumerate(issues['CRITICAL'], 1):
                    f.write(f"- [ ] {i}. `{issue.file}`: {issue.caller} → {issue.callee}\n")
                f.write("\n")
                
            # HIGH 清單（前30個）
            if issues['HIGH']:
                f.write("## ⚠️ HIGH 問題清單（前30個）\n\n")
                for i, issue in enumerate(issues['HIGH'][:30], 1):
                    f.write(f"- [ ] {i}. `{issue.file}`: {issue.caller} → {issue.callee}\n")
                if len(issues['HIGH']) > 30:
                    f.write(f"\n*... 還有 {len(issues['HIGH']) - 30} 個 HIGH 問題*\n")
                f.write("\n")
                
        print("  ✅ 清單已保存")
        
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="實用錯誤分析器")
    parser.add_argument("--input", 
                       default="analysis_with_unconnected/missing_function_connections.md",
                       help="輸入報告文件")
    parser.add_argument("--output", 
                       default="practical_report.md",
                       help="輸出報告文件")
    parser.add_argument("--checklist",
                       default="fix_checklist.md",
                       help="修復清單文件")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 實用錯誤分析器 - 分級展示，確保不遺漏重要問題")
    print("=" * 70)
    
    analyzer = PracticalAnalyzer()
    
    # 分析報告
    issues = analyzer.analyze_report(args.input)
    
    # 生成實用報告
    analyzer.generate_practical_report(issues, args.output)
    
    # 生成快速修復清單
    analyzer.generate_quick_fix_list(issues, args.checklist)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print(f"\n📄 實用報告: {args.output}")
    print(f"📋 修復清單: {args.checklist}")
    
    # 給出建議
    critical_count = len(issues['CRITICAL'])
    high_count = len(issues['HIGH'])
    
    if critical_count > 0:
        print(f"\n🚨 發現 {critical_count} 個 CRITICAL 問題 - 建議今天處理")
    if high_count > 0:
        print(f"⚠️  發現 {high_count} 個 HIGH 問題 - 建議本週處理")
        
    print("\n💡 提示: 寧可多列不漏，已確保所有重要問題都在報告中")


if __name__ == "__main__":
    main()
