#!/usr/bin/env python3
"""
AIVA 隊列命名一致性驗證工具

此工具掃描所有 worker 代碼，驗證隊列命名是否符合統一標準。

標準:
- 結果隊列: findings.new
- 任務隊列: tasks.{module}.{function}
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

@dataclass
class QueueUsage:
    """隊列使用情況"""
    file_path: str
    line_number: int
    queue_name: str
    queue_type: str  # 'task' or 'result'
    language: str
    context: str

class QueueValidator:
    """隊列命名驗證器"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.queue_usages: List[QueueUsage] = []
        self.expected_result_queue = "findings.new"
        
    def scan_workers(self) -> None:
        """掃描所有 worker 目錄"""
        print("🔍 掃描 AIVA workers...")
        
        # 掃描目錄
        scan_dirs = [
            "services/scan",
            "services/features",
        ]
        
        for scan_dir in scan_dirs:
            dir_path = self.workspace_root / scan_dir
            if dir_path.exists():
                self._scan_directory(dir_path)
                
        print(f"✅ 掃描完成，找到 {len(self.queue_usages)} 個隊列使用")
    
    def _scan_directory(self, directory: Path) -> None:
        """遞歸掃描目錄"""
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                self._scan_file(file_path)
    
    def _scan_file(self, file_path: Path) -> None:
        """掃描單個檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 根據檔案類型選擇掃描模式
            if file_path.suffix == '.py':
                self._scan_python(file_path, content)
            elif file_path.suffix == '.rs':
                self._scan_rust(file_path, content)
            elif file_path.suffix == '.go':
                self._scan_go(file_path, content)
            elif file_path.suffix in ['.ts', '.js']:
                self._scan_typescript(file_path, content)
                
        except Exception as e:
            # 忽略二進制檔案或無法讀取的檔案
            pass
    
    def _scan_python(self, file_path: Path, content: str) -> None:
        """掃描 Python 檔案中的隊列使用"""
        patterns = [
            (r'queue_name\s*=\s*["\']([^"\']+)["\']', 'unknown'),
            (r'result.*queue.*=\s*["\']([^"\']+)["\']', 'result'),
            (r'task.*queue.*=\s*["\']([^"\']+)["\']', 'task'),
            (r'["\']([^"\']*(?:findings|results)[^"\']*)["\']', 'result'),
            (r'["\']([^"\']*tasks\.[^"\']*)["\']', 'task'),
        ]
        
        self._apply_patterns(file_path, content, patterns, 'python')
    
    def _scan_rust(self, file_path: Path, content: str) -> None:
        """掃描 Rust 檔案中的隊列使用"""
        patterns = [
            (r'const\s+\w*QUEUE\s*:\s*&str\s*=\s*"([^"]+)"', 'unknown'),
            (r'const\s+RESULT_QUEUE\s*:\s*&str\s*=\s*"([^"]+)"', 'result'),
            (r'const\s+FINDING_QUEUE\s*:\s*&str\s*=\s*"([^"]+)"', 'result'),
            (r'const\s+TASK_QUEUE\s*:\s*&str\s*=\s*"([^"]+)"', 'task'),
            (r'"([^"]*(?:findings|results)[^"]*)"', 'result'),
            (r'"([^"]*tasks\.[^"]*)"', 'task'),
        ]
        
        self._apply_patterns(file_path, content, patterns, 'rust')
    
    def _scan_go(self, file_path: Path, content: str) -> None:
        """掃描 Go 檔案中的隊列使用"""
        patterns = [
            (r'queueName\s*:=\s*"([^"]+)"', 'unknown'),
            (r'resultQueue\s*:=\s*"([^"]+)"', 'result'),
            (r'taskQueue\s*:=\s*"([^"]+)"', 'task'),
            (r'"([^"]*(?:findings|results)[^"]*)"', 'result'),
            (r'"([^"]*tasks\.[^"]*)"', 'task'),
        ]
        
        self._apply_patterns(file_path, content, patterns, 'go')
    
    def _scan_typescript(self, file_path: Path, content: str) -> None:
        """掃描 TypeScript/JavaScript 檔案中的隊列使用"""
        patterns = [
            (r'const\s+\w*QUEUE\s*=\s*["\']([^"\']+)["\']', 'unknown'),
            (r'const\s+RESULT_QUEUE\s*=\s*["\']([^"\']+)["\']', 'result'),
            (r'const\s+TASK_QUEUE\s*=\s*["\']([^"\']+)["\']', 'task'),
            (r'resultQueue\s*=\s*["\']([^"\']+)["\']', 'result'),
            (r'["\']([^"\']*(?:findings|results)[^"\']*)["\']', 'result'),
            (r'["\']([^"\']*tasks\.[^"\']*)["\']', 'task'),
        ]
        
        self._apply_patterns(file_path, content, patterns, 'typescript')
    
    def _apply_patterns(self, file_path: Path, content: str, patterns: List[Tuple[str, str]], language: str) -> None:
        """應用正則表達式模式"""
        lines = content.split('\n')
        
        for pattern, queue_type in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                queue_name = match.group(1)
                
                # 跳過明顯不是隊列名的匹配
                if self._is_valid_queue_name(queue_name):
                    line_number = content[:match.start()].count('\n') + 1
                    context = lines[line_number - 1].strip() if line_number <= len(lines) else ""
                    
                    self.queue_usages.append(QueueUsage(
                        file_path=str(file_path.relative_to(self.workspace_root)),
                        line_number=line_number,
                        queue_name=queue_name,
                        queue_type=queue_type,
                        language=language,
                        context=context
                    ))
    
    def _is_valid_queue_name(self, queue_name: str) -> bool:
        """檢查是否是有效的隊列名"""
        if not queue_name or len(queue_name) < 3:
            return False
            
        # 跳過明顯不是隊列名的字串
        invalid_patterns = [
            r'^[A-Z_]+$',  # 全大寫常量
            r'^\w+$',      # 單個單詞
            r'^/.*',       # 路徑
            r'^https?://', # URL
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, queue_name):
                return False
        
        return True
    
    def validate_naming(self) -> Dict[str, List[QueueUsage]]:
        """驗證隊列命名一致性"""
        print("\n📋 驗證隊列命名一致性...")
        
        issues = {
            'incorrect_result_queues': [],
            'non_standard_task_queues': [],
            'duplicate_queues': [],
            'compliant_queues': []
        }
        
        # 按隊列名分組
        queue_groups = {}
        for usage in self.queue_usages:
            if usage.queue_name not in queue_groups:
                queue_groups[usage.queue_name] = []
            queue_groups[usage.queue_name].append(usage)
        
        for queue_name, usages in queue_groups.items():
            result_usages = [u for u in usages if u.queue_type == 'result']
            task_usages = [u for u in usages if u.queue_type == 'task']
            
            # 檢查結果隊列
            if result_usages:
                if queue_name != self.expected_result_queue:
                    issues['incorrect_result_queues'].extend(result_usages)
                else:
                    issues['compliant_queues'].extend(result_usages)
            
            # 檢查任務隊列
            if task_usages:
                if not queue_name.startswith('tasks.'):
                    issues['non_standard_task_queues'].extend(task_usages)
                else:
                    issues['compliant_queues'].extend(task_usages)
            
            # 檢查重複使用
            if len(usages) > 1:
                issues['duplicate_queues'].extend(usages)
        
        return issues
    
    def generate_report(self, issues: Dict[str, List[QueueUsage]]) -> str:
        """生成驗證報告"""
        report = []
        report.append("# AIVA 隊列命名一致性報告")
        report.append("=" * 50)
        report.append("")
        
        # 統計信息
        total_usages = len(self.queue_usages)
        compliant_count = len(issues['compliant_queues'])
        issue_count = total_usages - compliant_count
        
        report.append(f"📊 **統計信息**")
        report.append(f"- 總隊列使用數: {total_usages}")
        report.append(f"- 符合標準: {compliant_count}")
        report.append(f"- 需要修復: {issue_count}")
        report.append(f"- 合規率: {compliant_count/total_usages*100:.1f}%" if total_usages > 0 else "- 合規率: N/A")
        report.append("")
        
        # 問題詳情
        if issues['incorrect_result_queues']:
            report.append("❌ **不正確的結果隊列**")
            report.append(f"標準結果隊列應為: `{self.expected_result_queue}`")
            report.append("")
            for usage in issues['incorrect_result_queues']:
                report.append(f"- 📁 `{usage.file_path}:{usage.line_number}`")
                report.append(f"  - 隊列名: `{usage.queue_name}` ❌")
                report.append(f"  - 語言: {usage.language}")
                report.append(f"  - 上下文: `{usage.context}`")
                report.append("")
        
        if issues['non_standard_task_queues']:
            report.append("❌ **不符合標準的任務隊列**")
            report.append("任務隊列應遵循格式: `tasks.{module}.{function}`")
            report.append("")
            for usage in issues['non_standard_task_queues']:
                report.append(f"- 📁 `{usage.file_path}:{usage.line_number}`")
                report.append(f"  - 隊列名: `{usage.queue_name}` ❌")
                report.append(f"  - 語言: {usage.language}")
                report.append(f"  - 上下文: `{usage.context}`")
                report.append("")
        
        # 合規隊列
        if issues['compliant_queues']:
            report.append("✅ **符合標準的隊列**")
            report.append("")
            
            # 按隊列名分組顯示
            compliant_groups = {}
            for usage in issues['compliant_queues']:
                if usage.queue_name not in compliant_groups:
                    compliant_groups[usage.queue_name] = []
                compliant_groups[usage.queue_name].append(usage)
            
            for queue_name, usages in compliant_groups.items():
                report.append(f"**隊列: `{queue_name}`**")
                for usage in usages:
                    report.append(f"- 📁 `{usage.file_path}:{usage.line_number}` ({usage.language})")
                report.append("")
        
        # 修復建議
        report.append("🔧 **修復建議**")
        report.append("")
        
        if issues['incorrect_result_queues']:
            report.append("1. **結果隊列統一化**:")
            unique_files = set(u.file_path for u in issues['incorrect_result_queues'])
            for file_path in unique_files:
                file_usages = [u for u in issues['incorrect_result_queues'] if u.file_path == file_path]
                old_queues = set(u.queue_name for u in file_usages)
                report.append(f"   - `{file_path}`: {', '.join(old_queues)} → `{self.expected_result_queue}`")
            report.append("")
        
        if issues['non_standard_task_queues']:
            report.append("2. **任務隊列標準化**:")
            for usage in issues['non_standard_task_queues']:
                suggested = self._suggest_task_queue_name(usage)
                report.append(f"   - `{usage.file_path}`: `{usage.queue_name}` → `{suggested}`")
            report.append("")
        
        report.append("3. **環境變數支援**:")
        report.append("   - 所有 workers 應支援 `AIVA_RESULT_QUEUE` 和 `AIVA_TASK_QUEUE` 環境變數")
        report.append("   - 預設值應符合命名標準")
        report.append("")
        
        return "\n".join(report)
    
    def _suggest_task_queue_name(self, usage: QueueUsage) -> str:
        """建議任務隊列名稱"""
        # 簡單的建議邏輯
        if 'authn' in usage.file_path:
            return 'tasks.function.authn'
        elif 'ssrf' in usage.file_path:
            return 'tasks.function.ssrf'
        elif 'sast' in usage.file_path:
            return 'tasks.function.sast'
        elif 'cspm' in usage.file_path:
            return 'tasks.function.cspm'
        elif 'scan' in usage.file_path:
            return 'tasks.scan.dynamic'
        else:
            return 'tasks.unknown.function'

def main():
    """主程序"""
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 AIVA 隊列命名一致性驗證工具")
    print("=" * 50)
    print(f"工作區: {workspace_root}")
    print()
    
    validator = QueueValidator(workspace_root)
    validator.scan_workers()
    issues = validator.validate_naming()
    report = validator.generate_report(issues)
    
    # 輸出報告
    print(report)
    
    # 保存報告到檔案
    report_path = os.path.join(workspace_root, "reports", "queue_naming_validation.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 完整報告已保存至: {report_path}")
    
    # 返回退出代碼
    total_issues = sum(len(issues[key]) for key in issues if key != 'compliant_queues')
    if total_issues > 0:
        print(f"\n❌ 發現 {total_issues} 個需要修復的問題")
        return 1
    else:
        print("\n✅ 所有隊列命名都符合標準！")
        return 0

if __name__ == "__main__":
    exit(main())