#!/usr/bin/env python3
"""
AIVA Schema 版本一致性檢查工具
防止意外混用手動維護版本和自動生成版本的 Schema

使用方式:
    python schema_version_checker.py
    python schema_version_checker.py --fix  # 自動修復不一致的導入
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class SchemaVersionChecker:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues = []
        
        # 定義正確和錯誤的導入模式
        self.correct_patterns = [
            r"from\s+services\.aiva_common\.schemas\.base\s+import",
            r"from\s+services\.aiva_common\.schemas\.findings\s+import", 
            r"from\s+services\.aiva_common\.enums\s+import",
        ]
        
        self.problematic_patterns = [
            r"from\s+services\.aiva_common\.schemas\.generated\.",
            r"from\s+aiva_common\.schemas\.generated\.",
            r"import\s+.*generated\.base_types",
        ]
    
    def scan_files(self) -> List[Path]:
        """掃描所有 Python 檔案"""
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.root_dir.glob(pattern))
        
        # 排除不需要檢查的目錄
        excluded_dirs = {"__pycache__", ".git", "venv", "_archive", "node_modules"}
        
        filtered_files = []
        for file_path in python_files:
            if not any(excluded in file_path.parts for excluded in excluded_dirs):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def check_file(self, file_path: Path) -> List[Dict]:
        """檢查單個檔案的 Schema 導入"""
        file_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            return [{"type": "read_error", "file": file_path, "error": str(e)}]
        
        # 檢查是否有問題的導入
        for line_num, line in enumerate(lines, 1):
            for pattern in self.problematic_patterns:
                if re.search(pattern, line):
                    file_issues.append({
                        "type": "problematic_import",
                        "file": file_path,
                        "line": line_num,
                        "content": line.strip(),
                        "pattern": pattern
                    })
        
        # 檢查是否混用了不同版本
        has_manual = any(re.search(pattern, content) for pattern in self.correct_patterns)
        has_generated = any(re.search(pattern, content) for pattern in self.problematic_patterns)
        
        if has_manual and has_generated:
            file_issues.append({
                "type": "mixed_versions",
                "file": file_path,
                "description": "同一檔案中混用了手動維護和自動生成的 Schema"
            })
        
        return file_issues
    
    def generate_fixes(self, issues: List[Dict]) -> Dict[Path, List[str]]:
        """為發現的問題生成修復建議"""
        fixes = {}
        
        for issue in issues:
            if issue["type"] == "problematic_import":
                file_path = issue["file"]
                line_content = issue["content"]
                
                # 生成修復建議
                fixed_line = line_content
                
                # 修復常見的錯誤導入
                replacements = {
                    r"from\s+services\.aiva_common\.schemas\.generated\.base_types\s+import": 
                        "from services.aiva_common.schemas.base import",
                    r"from\s+aiva_common\.schemas\.generated\.base_types\s+import":
                        "from services.aiva_common.schemas.base import",
                    r"from\s+services\.aiva_common\.schemas\.generated\.": 
                        "from services.aiva_common.schemas.",
                }
                
                for old_pattern, new_pattern in replacements.items():
                    fixed_line = re.sub(old_pattern, new_pattern, fixed_line)
                
                if file_path not in fixes:
                    fixes[file_path] = []
                
                fixes[file_path].append({
                    "line": issue["line"],
                    "original": line_content,
                    "fixed": fixed_line,
                    "description": f"修復第 {issue['line']} 行的 Schema 導入"
                })
        
        return fixes
    
    def run_check(self) -> bool:
        """執行完整的檢查"""
        print("🔍 AIVA Schema 版本一致性檢查")
        print("=" * 50)
        
        files = self.scan_files()
        print(f"📁 掃描 {len(files)} 個 Python 檔案...")
        
        all_issues = []
        problem_files = 0
        
        for file_path in files:
            issues = self.check_file(file_path)
            if issues:
                all_issues.extend(issues)
                problem_files += 1
        
        # 統計結果
        print(f"\n📊 檢查結果:")
        print(f"   總檔案數: {len(files)}")
        print(f"   問題檔案數: {problem_files}")
        print(f"   發現問題數: {len(all_issues)}")
        
        if not all_issues:
            print("\n✅ 恭喜！沒有發現 Schema 版本不一致問題")
            return True
        else:
            print(f"\n⚠️ 發現 {len(all_issues)} 個問題:")
            
            for issue in all_issues:
                if issue["type"] == "problematic_import":
                    print(f"   📍 {issue['file']}:{issue['line']}")
                    print(f"      問題: {issue['content']}")
                elif issue["type"] == "mixed_versions":
                    print(f"   📍 {issue['file']}")
                    print(f"      問題: {issue['description']}")
                elif issue["type"] == "read_error":
                    print(f"   📍 {issue['file']}")
                    print(f"      錯誤: {issue['error']}")
            
            # 生成修復建議
            fixes = self.generate_fixes(all_issues)
            if fixes:
                print(f"\n💡 修復建議:")
                for file_path, file_fixes in fixes.items():
                    print(f"   📁 {file_path}:")
                    for fix in file_fixes:
                        print(f"      行 {fix['line']}: {fix['original']}")
                        print(f"      改為: {fix['fixed']}")
                
                print(f"\n🛠️ 使用 --fix 參數自動修復這些問題")
            
            return False
    
    def apply_fixes(self) -> bool:
        """自動修復發現的問題"""
        print("🔧 自動修復 Schema 導入問題...")
        
        files = self.scan_files()
        all_issues = []
        
        for file_path in files:
            issues = self.check_file(file_path)
            all_issues.extend(issues)
        
        fixes = self.generate_fixes(all_issues)
        
        if not fixes:
            print("✅ 沒有可自動修復的問題")
            return True
        
        fixed_files = 0
        for file_path, file_fixes in fixes.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 按行號倒序排列，避免修改時行號偏移
                file_fixes.sort(key=lambda x: x["line"], reverse=True)
                
                for fix in file_fixes:
                    line_idx = fix["line"] - 1  # 轉換為 0 索引
                    if line_idx < len(lines):
                        lines[line_idx] = fix["fixed"] + "\n"
                        print(f"   ✅ {file_path}:{fix['line']} 已修復")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                fixed_files += 1
                
            except Exception as e:
                print(f"   ❌ 修復 {file_path} 失敗: {e}")
        
        print(f"\n🎉 修復完成！共修復 {fixed_files} 個檔案")
        return True

def main():
    """主程式入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AIVA Schema 版本一致性檢查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python schema_version_checker.py              # 檢查版本一致性
    python schema_version_checker.py --fix        # 自動修復問題
    python schema_version_checker.py --dir /path  # 指定檢查目錄
        """
    )
    
    parser.add_argument("--fix", action="store_true", 
                       help="自動修復發現的問題")
    parser.add_argument("--dir", default=".", 
                       help="指定要檢查的根目錄 (預設: 當前目錄)")
    
    args = parser.parse_args()
    
    checker = SchemaVersionChecker(args.dir)
    
    if args.fix:
        success = checker.apply_fixes()
        # 修復後再次檢查
        if success:
            print("\n🔍 修復後重新檢查...")
            checker.run_check()
    else:
        success = checker.run_check()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()