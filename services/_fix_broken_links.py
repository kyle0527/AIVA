#!/usr/bin/env python3
"""
修復 services 目錄中所有損壞的連結
基於 SERVICES_STRUCTURE_ANALYSIS_REPORT.md 的分析結果
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class LinkFixer:
    def __init__(self, services_root: str):
        self.services_root = Path(services_root)
        self.fixes_applied = 0
        self.files_modified = 0
        
    def fix_absolute_paths(self, content: str, file_path: Path) -> Tuple[str, int]:
        """修復絕對路徑（Downloads 資料夾）到相對路徑"""
        fixes = 0
        
        # Pattern: C:\Users\User\Downloads\新增資料夾 (6)\...
        patterns = [
            (r'C:\\Users\\User\\Downloads\\新增資料夾 \(6\)\\([^\)]+\.md)', 
             r'../../../docs/reports/\1'),
            (r'\.\./\.\./\.\./Users/User/Downloads/新增資料夾%20\(6\)/([^)]+\.md)', 
             r'../../../docs/reports/\1'),
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                print(f"  → 修復 {len(matches)} 個絕對路徑")
                content = re.sub(pattern, replacement, content)
                fixes += len(matches)
        
        return content, fixes
    
    def fix_missing_docs_readme(self, content: str) -> Tuple[str, int]:
        """修復不存在的 ../../docs/README.md"""
        fixes = 0
        
        # 替換為實際存在的文檔
        patterns = [
            (r'\[.*?文檔中心.*?\]\(\.\./\.\./docs/README\.md\)', 
             '[📖 文檔中心](../../docs/guides/services/)'),
            (r'\(\.\./\.\./docs/README\.md\)', 
             '(../../docs/)'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes += 1
        
        return content, fixes
    
    def fix_out_directory_links(self, content: str) -> Tuple[str, int]:
        """修復 _out 目錄連結（已移動）"""
        fixes = 0
        
        # _out 目錄的文件已移動到 docs/reports
        patterns = [
            (r'\(\.\./\.\./(_out/[^)]+\.md)\)', r'(../../docs/reports/\1)'),
            (r'\[([^\]]+)\]\(\.\./_out/([^)]+\.md)\)', r'[\1](../docs/reports/\2)'),
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, replacement, content)
                fixes += len(matches)
        
        return content, fixes
    
    def fix_missing_module_files(self, content: str, module_name: str) -> Tuple[str, int]:
        """修復不存在的模組文件連結"""
        fixes = 0
        
        # 常見的不存在文件
        missing_files = {
            'API_REFERENCE.md': 'README.md',
            'ARCHITECTURE.md': 'README.md',
            'DEVELOPMENT_GUIDE.md': 'README.md',
        }
        
        for missing, replacement in missing_files.items():
            pattern = f'({missing})'
            if re.search(pattern, content):
                content = content.replace(missing, replacement)
                fixes += 1
        
        return content, fixes
    
    def remove_broken_links(self, content: str) -> Tuple[str, int]:
        """移除指向不存在檔案的連結（保留文字）"""
        fixes = 0
        
        # 找出所有 Markdown 連結
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        def check_and_fix(match):
            nonlocal fixes
            text = match.group(1)
            url = match.group(2)
            
            # 跳過外部連結和錨點
            if url.startswith(('http://', 'https://', '#')):
                return match.group(0)
            
            # 檢查檔案是否存在
            # 這裡簡化處理 - 實際應該解析相對路徑
            # 暫時保留所有連結，只標記
            return match.group(0)
        
        # 實際檢查會很複雜，先跳過
        return content, fixes
    
    def fix_file(self, file_path: Path) -> bool:
        """修復單一檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            total_fixes = 0
            
            # 套用所有修復
            content, f1 = self.fix_absolute_paths(content, file_path)
            content, f2 = self.fix_missing_docs_readme(content)
            content, f3 = self.fix_out_directory_links(content)
            
            # 根據模組特定修復
            module_name = self._get_module_name(file_path)
            content, f4 = self.fix_missing_module_files(content, module_name)
            
            total_fixes = f1 + f2 + f3 + f4
            
            if content != original:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ {file_path.relative_to(self.services_root)}: 修復 {total_fixes} 處")
                self.fixes_applied += total_fixes
                self.files_modified += 1
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ 錯誤處理 {file_path}: {e}")
            return False
    
    def _get_module_name(self, file_path: Path) -> str:
        """取得模組名稱"""
        parts = file_path.relative_to(self.services_root).parts
        return parts[0] if parts else ""
    
    def fix_all_markdown_files(self):
        """修復所有 MD 檔案"""
        print(f"🔍 掃描 {self.services_root}")
        
        md_files = list(self.services_root.glob('**/*.md'))
        # 排除 node_modules, target 等
        md_files = [f for f in md_files if 'node_modules' not in str(f) and 'target' not in str(f)]
        
        print(f"📝 找到 {len(md_files)} 個 MD 檔案\n")
        
        for md_file in md_files:
            self.fix_file(md_file)
        
        print(f"\n✨ 完成！")
        print(f"   修改檔案: {self.files_modified}")
        print(f"   修復連結: {self.fixes_applied}")

def main():
    services_root = Path(r"C:\D\fold7\AIVA-git\services")
    
    fixer = LinkFixer(str(services_root))
    fixer.fix_all_markdown_files()
    
    print("\n📊 生成修復報告...")
    
    report = f"""# 連結修復報告

## 執行摘要

- **修改檔案**: {fixer.files_modified}
- **修復連結**: {fixer.fixes_applied}
- **執行時間**: {Path(__file__).stat().st_mtime}

## 修復類型

1. **絕對路徑修復**: 將 `C:\\Users\\User\\Downloads\\新增資料夾 (6)\\` 替換為相對路徑
2. **不存在文檔修復**: 修正 `../../docs/README.md` 等不存在的連結
3. **_out 目錄修復**: 將 `_out` 目錄連結更新為 `docs/reports`
4. **模組文件修復**: 修正不存在的模組文件連結

## 後續行動

- [ ] 驗證所有連結指向正確
- [ ] 建立缺失的文檔檔案
- [ ] 更新導航結構
"""
    
    report_path = services_root / "LINK_FIX_REPORT_PHASE2.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 報告已儲存: {report_path}")

if __name__ == '__main__':
    main()
