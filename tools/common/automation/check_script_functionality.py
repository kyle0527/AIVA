#!/usr/bin/env python3
"""
檢查所有腳本檔案是否具備基本功能
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ScriptChecker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "no_functionality": [],  # 完全沒功能
            "minimal_functionality": [],  # 只有基本架構
            "partial_functionality": [],  # 部分功能
            "full_functionality": []  # 完整功能
        }
        
    def check_python_file(self, file_path: Path) -> Tuple[str, str, Dict]:
        """檢查 Python 檔案功能性"""
        try:
            content = file_path.read_text(encoding='utf-8')
        except:
            try:
                content = file_path.read_text(encoding='utf-8-sig')
            except Exception as e:
                return "error", f"無法讀取: {e}", {}
        
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        
        info = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'has_imports': bool(re.search(r'^(import|from)\s+\w+', content, re.MULTILINE)),
            'has_functions': bool(re.search(r'^def\s+\w+', content, re.MULTILINE)),
            'has_classes': bool(re.search(r'^class\s+\w+', content, re.MULTILINE)),
            'has_main': bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content)),
            'has_docstring': bool(re.search(r'^"""[\s\S]*?"""|^\'\'\'[\s\S]*?\'\'\'', content, re.MULTILINE)),
            'has_pass_only': content.strip() == 'pass' or content.strip() == '"""TODO"""',
            'has_todo': 'TODO' in content.upper() or 'FIXME' in content.upper(),
        }
        
        # 判斷功能級別
        if info['code_lines'] == 0 or info['has_pass_only']:
            return "no_functionality", "空檔案或只有 pass", info
        elif info['code_lines'] < 10 and not (info['has_functions'] or info['has_classes']):
            return "no_functionality", "少於10行且無函數/類別", info
        elif info['has_todo'] and info['code_lines'] < 20:
            return "minimal_functionality", "含 TODO 且程式碼少於20行", info
        elif not info['has_functions'] and not info['has_classes']:
            return "minimal_functionality", "無函數或類別定義", info
        elif info['code_lines'] < 30:
            return "minimal_functionality", "程式碼少於30行", info
        elif info['code_lines'] < 100 and not info['has_main']:
            return "partial_functionality", "程式碼少於100行且無主程式", info
        else:
            return "full_functionality", "具備完整功能", info
    
    def check_powershell_file(self, file_path: Path) -> Tuple[str, str, Dict]:
        """檢查 PowerShell 檔案功能性"""
        try:
            content = file_path.read_text(encoding='utf-8')
        except:
            try:
                content = file_path.read_text(encoding='utf-8-sig')
            except Exception as e:
                return "error", f"無法讀取: {e}", {}
        
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        
        info = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'has_functions': bool(re.search(r'^function\s+\w+', content, re.MULTILINE | re.IGNORECASE)),
            'has_params': bool(re.search(r'param\s*\(', content, re.IGNORECASE)),
            'has_cmdlets': bool(re.search(r'(Get-|Set-|New-|Remove-|Test-|Write-)\w+', content)),
            'has_todo': 'TODO' in content.upper() or 'FIXME' in content.upper(),
        }
        
        # 判斷功能級別
        if info['code_lines'] == 0:
            return "no_functionality", "空檔案", info
        elif info['code_lines'] < 10 and not info['has_cmdlets']:
            return "no_functionality", "少於10行且無 cmdlet", info
        elif info['has_todo'] and info['code_lines'] < 20:
            return "minimal_functionality", "含 TODO 且程式碼少於20行", info
        elif not info['has_functions'] and not info['has_cmdlets']:
            return "minimal_functionality", "無函數或 cmdlet", info
        elif info['code_lines'] < 30:
            return "minimal_functionality", "程式碼少於30行", info
        elif info['code_lines'] < 100 and not info['has_params']:
            return "partial_functionality", "程式碼少於100行且無參數定義", info
        else:
            return "full_functionality", "具備完整功能", info
    
    def scan_directory(self):
        """掃描專案目錄"""
        exclude_dirs = {'.venv', '__pycache__', 'node_modules', '.git', '_archive', 
                       '_cleanup_backup', '_out', 'docs', 'logs', 'reports'}
        
        for file_path in self.project_root.rglob('*'):
            # 排除目錄
            if any(ex in file_path.parts for ex in exclude_dirs):
                continue
            
            if file_path.suffix == '.py':
                level, reason, info = self.check_python_file(file_path)
                rel_path = file_path.relative_to(self.project_root)
                self.results[level].append({
                    'file': str(rel_path),
                    'type': 'Python',
                    'reason': reason,
                    'info': info
                })
            elif file_path.suffix == '.ps1':
                level, reason, info = self.check_powershell_file(file_path)
                rel_path = file_path.relative_to(self.project_root)
                self.results[level].append({
                    'file': str(rel_path),
                    'type': 'PowerShell',
                    'reason': reason,
                    'info': info
                })
    
    def generate_report(self, output_file: str):
        """生成報告"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("AIVA 專案腳本功能性檢查報告")
        report_lines.append(f"檢查時間: {__import__('datetime').datetime.now()}")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # 統計
        total = sum(len(v) for v in self.results.values())
        report_lines.append("## 📊 統計摘要")
        report_lines.append(f"總腳本數: {total}")
        report_lines.append(f"  ✅ 完整功能: {len(self.results['full_functionality'])} ({len(self.results['full_functionality'])/total*100:.1f}%)")
        report_lines.append(f"  ⚠️  部分功能: {len(self.results['partial_functionality'])} ({len(self.results['partial_functionality'])/total*100:.1f}%)")
        report_lines.append(f"  🔶 基本架構: {len(self.results['minimal_functionality'])} ({len(self.results['minimal_functionality'])/total*100:.1f}%)")
        report_lines.append(f"  ❌ 無功能: {len(self.results['no_functionality'])} ({len(self.results['no_functionality'])/total*100:.1f}%)")
        report_lines.append("")
        
        # 詳細列表
        if self.results['no_functionality']:
            report_lines.append("=" * 100)
            report_lines.append("## ❌ 無基本功能的腳本 (需要實作)")
            report_lines.append("=" * 100)
            for item in sorted(self.results['no_functionality'], key=lambda x: x['file']):
                report_lines.append(f"\n📁 {item['file']}")
                report_lines.append(f"   類型: {item['type']}")
                report_lines.append(f"   原因: {item['reason']}")
                report_lines.append(f"   總行數: {item['info'].get('total_lines', 0)}")
                report_lines.append(f"   程式碼行數: {item['info'].get('code_lines', 0)}")
            report_lines.append("")
        
        if self.results['minimal_functionality']:
            report_lines.append("=" * 100)
            report_lines.append("## 🔶 僅有基本架構的腳本 (需要補充)")
            report_lines.append("=" * 100)
            for item in sorted(self.results['minimal_functionality'], key=lambda x: x['file']):
                report_lines.append(f"\n📁 {item['file']}")
                report_lines.append(f"   類型: {item['type']}")
                report_lines.append(f"   原因: {item['reason']}")
                report_lines.append(f"   程式碼行數: {item['info'].get('code_lines', 0)}")
                if item['info'].get('has_todo'):
                    report_lines.append(f"   ⚠️  含 TODO 標記")
            report_lines.append("")
        
        if self.results['partial_functionality']:
            report_lines.append("=" * 100)
            report_lines.append("## ⚠️  部分功能的腳本 (可以改進)")
            report_lines.append("=" * 100)
            for item in sorted(self.results['partial_functionality'], key=lambda x: x['file']):
                report_lines.append(f"\n📁 {item['file']}")
                report_lines.append(f"   類型: {item['type']}")
                report_lines.append(f"   程式碼行數: {item['info'].get('code_lines', 0)}")
            report_lines.append("")
        
        # 寫入檔案
        report_text = '\n'.join(report_lines)
        Path(output_file).write_text(report_text, encoding='utf-8')
        
        # 同時輸出 JSON
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(report_text)
        print(f"\n報告已儲存至: {output_file}")
        print(f"JSON 數據已儲存至: {json_file}")

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent.parent
    output_file = project_root / '_out' / 'script_functionality_report.txt'
    
    checker = ScriptChecker(project_root)
    print("開始掃描專案...")
    checker.scan_directory()
    print("生成報告...")
    checker.generate_report(output_file)
    print("✅ 完成!")
