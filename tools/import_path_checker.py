#!/usr/bin/env python3
"""
AIVA Import Path Checker Tool
自動檢測和修復 AIVA 項目中的 import 路徑問題

Usage:
    python tools/import_path_checker.py --check        # 僅檢查問題
    python tools/import_path_checker.py --fix          # 檢查並自動修復
    python tools/import_path_checker.py --report       # 生成詳細報告
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

class ImportPathChecker:
    """Import 路徑檢查器"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.issues_found = []
        self.fixes_applied = []
        
        # 定義錯誤的 import 模式
        self.error_patterns = [
            (r'from aiva_core\.', 'from services.core.aiva_core.'),
            (r'from aiva_common\.', 'from services.aiva_common.'),
            (r'from aiva_scan\.', 'from services.scan.aiva_scan.'),
            (r'from aiva_integration\.', 'from services.integration.aiva_integration.'),
            (r'import aiva_core\b', 'import services.core.aiva_core'),
            (r'import aiva_common\b', 'import services.aiva_common'),
        ]
        
        # 排除的目錄和檔案
        self.exclude_paths = {
            '.git', '.venv', '__pycache__', 'node_modules', 
            '.pytest_cache', 'dist', 'build'
        }
        
        # 排除的檔案模式
        self.exclude_files = {
            '*.pyc', '*.pyo', '*.pyd', '__pycache__'
        }

    def find_python_files(self) -> List[Path]:
        """尋找所有 Python 檔案"""
        python_files = []
        
        for path in self.root_path.rglob('*.py'):
            # 檢查是否在排除路徑中
            if any(exclude in path.parts for exclude in self.exclude_paths):
                continue
            python_files.append(path)
            
        return python_files

    def check_file(self, file_path: Path) -> List[Dict]:
        """檢查單個檔案的 import 問題"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                
            for line_num, line in enumerate(lines, 1):
                for error_pattern, _ in self.error_patterns:
                    if re.search(error_pattern, line):
                        issues.append({
                            'file': file_path,
                            'line_number': line_num,
                            'line_content': line.strip(),
                            'pattern': error_pattern,
                            'type': 'import_path_error'
                        })
                        
        except Exception as e:
            issues.append({
                'file': file_path,
                'line_number': 0,
                'line_content': '',
                'pattern': 'file_read_error',
                'type': 'file_error',
                'error': str(e)
            })
            
        return issues

    def fix_file(self, file_path: Path) -> bool:
        """修復單個檔案的 import 問題"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            fixes_count = 0
            
            for error_pattern, correct_pattern in self.error_patterns:
                new_content = re.sub(error_pattern, correct_pattern, content)
                if new_content != content:
                    fixes_count += re.subn(error_pattern, correct_pattern, content)[1]
                    content = new_content
                    
            if content != original_content:
                # 備份原檔案
                backup_path = file_path.with_suffix(f'.py.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # 寫入修復後的內容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                self.fixes_applied.append({
                    'file': file_path,
                    'fixes_count': fixes_count,
                    'backup_path': backup_path
                })
                return True
                
        except Exception as e:
            print(f"錯誤: 無法修復檔案 {file_path}: {e}")
            return False
            
        return False

    def run_check(self) -> Dict:
        """執行檢查"""
        print("🔍 搜尋 Python 檔案...")
        python_files = self.find_python_files()
        print(f"找到 {len(python_files)} 個 Python 檔案")
        
        print("\n🔎 檢查 import 路徑問題...")
        total_issues = 0
        
        for file_path in python_files:
            issues = self.check_file(file_path)
            if issues:
                self.issues_found.extend(issues)
                total_issues += len(issues)
                
        return {
            'total_files': len(python_files),
            'files_with_issues': len(set(issue['file'] for issue in self.issues_found)),
            'total_issues': total_issues,
            'issues': self.issues_found
        }

    def run_fix(self) -> Dict:
        """執行修復"""
        check_result = self.run_check()
        
        if not self.issues_found:
            print("✅ 沒有發現需要修復的問題")
            return check_result
            
        print(f"\n🔧 修復 {len(set(issue['file'] for issue in self.issues_found))} 個檔案...")
        
        files_to_fix = set(issue['file'] for issue in self.issues_found 
                          if issue['type'] == 'import_path_error')
        
        for file_path in files_to_fix:
            if self.fix_file(file_path):
                print(f"✅ 已修復: {file_path.relative_to(self.root_path)}")
            else:
                print(f"❌ 修復失敗: {file_path.relative_to(self.root_path)}")
                
        return {
            **check_result,
            'fixes_applied': len(self.fixes_applied),
            'fix_details': self.fixes_applied
        }

    def generate_report(self) -> str:
        """生成詳細報告"""
        check_result = self.run_check()
        
        report = f"""
# AIVA Import Path Checker 報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 摘要
- 檢查檔案總數: {check_result['total_files']}
- 有問題的檔案數: {check_result['files_with_issues']}
- 問題總數: {check_result['total_issues']}

## 詳細問題列表
"""
        
        if not self.issues_found:
            report += "✅ 沒有發現任何 import 路徑問題\n"
        else:
            current_file = None
            for issue in self.issues_found:
                if issue['file'] != current_file:
                    current_file = issue['file']
                    report += f"\n### {current_file.relative_to(self.root_path)}\n"
                    
                if issue['type'] == 'import_path_error':
                    report += f"- Line {issue['line_number']}: `{issue['line_content']}`\n"
                    report += f"  Pattern: `{issue['pattern']}`\n"
                elif issue['type'] == 'file_error':
                    report += f"- 檔案讀取錯誤: {issue['error']}\n"
                    
        report += f"""
## 建議修復命令
```bash
python tools/import_path_checker.py --fix
```

## 預防措施
1. 在 pre-commit hook 中加入此檢查
2. 在 CI/CD pipeline 中加入自動檢查
3. 定期執行完整掃描
"""
        
        return report

def main():
    parser = argparse.ArgumentParser(description='AIVA Import Path Checker')
    parser.add_argument('--check', action='store_true', help='僅檢查問題')
    parser.add_argument('--fix', action='store_true', help='檢查並自動修復')
    parser.add_argument('--report', action='store_true', help='生成詳細報告')
    parser.add_argument('--root', type=str, default='.', help='項目根目錄')
    
    args = parser.parse_args()
    
    root_path = Path(args.root).resolve()
    checker = ImportPathChecker(root_path)
    
    if args.report:
        report = checker.generate_report()
        report_file = root_path / 'reports' / 'import_path_check_report.md'
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"報告已生成: {report_file}")
        print(report)
        
    elif args.fix:
        result = checker.run_fix()
        print(f"\n📊 修復完成:")
        print(f"- 檢查了 {result['total_files']} 個檔案")
        print(f"- 修復了 {result['fixes_applied']} 個檔案")
        print(f"- 解決了 {result['total_issues']} 個問題")
        
    elif args.check:
        result = checker.run_check()
        print(f"\n📊 檢查完成:")
        print(f"- 檢查了 {result['total_files']} 個檔案")
        print(f"- 發現 {result['files_with_issues']} 個有問題的檔案")
        print(f"- 總計 {result['total_issues']} 個問題")
        
        if result['total_issues'] > 0:
            print("\n問題檔案:")
            for issue in result['issues']:
                if issue['type'] == 'import_path_error':
                    print(f"  {issue['file'].relative_to(root_path)}:{issue['line_number']}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()