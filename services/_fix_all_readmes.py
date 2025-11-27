#!/usr/bin/env python3
"""
系統性修正所有 services README
從最底層（Level 4）開始往上修正
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

class ReadmeHierarchyFixer:
    def __init__(self, services_root: str):
        self.services_root = Path(services_root)
        self.readmes_by_level = defaultdict(list)
        self.fixes_log = []
        
    def scan_all_readmes(self):
        """掃描所有 README 並按階層分類"""
        exclude_dirs = {'node_modules', 'target', '__pycache__', '.git', '.pytest_cache', 'dist', 'build'}
        
        for readme in self.services_root.rglob('README.md'):
            # 排除特定目錄
            if any(ex in readme.parts for ex in exclude_dirs):
                continue
            
            # 計算階層深度
            rel_path = readme.relative_to(self.services_root)
            level = len(rel_path.parts) - 1  # README.md 本身不算
            
            self.readmes_by_level[level].append(readme)
        
        # 排序
        for level in self.readmes_by_level:
            self.readmes_by_level[level].sort()
        
        print(f"📊 掃描完成:")
        for level in sorted(self.readmes_by_level.keys()):
            count = len(self.readmes_by_level[level])
            print(f"   Level {level}: {count} 個 README")
    
    def check_and_fix_links(self, readme_path: Path) -> Tuple[bool, List[str]]:
        """檢查並修復單個 README 的連結"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_applied = []
            
            # 1. 修復絕對路徑
            absolute_path_pattern = r'C:\\Users\\User\\Downloads\\[^)"]+'
            if re.search(absolute_path_pattern, content):
                # 替換為相對路徑到 docs/reports
                content = re.sub(
                    r'C:\\Users\\User\\Downloads\\新增資料夾 \(6\)\\([^)"]+)',
                    r'../../../../docs/reports/\1',
                    content
                )
                fixes_applied.append("修復絕對路徑")
            
            # 2. 修復不存在的 docs/README.md
            if '../../docs/README.md' in content or '../../../docs/README.md' in content:
                content = re.sub(r'\[([^\]]*文檔中心[^\]]*)\]\([.\/]*docs/README\.md\)', 
                               r'[\1](../../docs/)', content)
                fixes_applied.append("修復 docs/README.md 連結")
            
            # 3. 修復 _out 目錄
            if '_out/' in content:
                content = re.sub(r'\([.\/]*_out/([^)]+)\)', r'(../docs/reports/\1)', content)
                fixes_applied.append("修復 _out 目錄連結")
            
            # 4. 移除不存在的檔案連結（docs/ 下的特定文件）
            broken_links = [
                r'\[.*?\]\(docs/README_HIGH_VALUE\.md\)',
                r'\[.*?\]\(docs/README_SECURITY_CORE\.md\)',
                r'\[.*?\]\(docs/README_BUSINESS_LOGIC\.md\)',
                r'\[.*?\]\(docs/README_INFRASTRUCTURE\.md\)',
                r'\[.*?\]\(docs/README_LANGUAGES\.md\)',
                r'\[.*?\]\(docs/README\.md\)',
                r'\[.*?\]\(docs/security/README\.md\)',
                r'\[.*?\]\(docs/development/README\.md\)',
                r'\[.*?\]\(docs/python/README\.md\)',
                r'\[.*?\]\(docs/go/README\.md\)',
                r'\[.*?\]\(docs/rust/README\.md\)',
            ]
            
            for pattern in broken_links:
                if re.search(pattern, content):
                    # 只移除連結，保留文字
                    content = re.sub(pattern, lambda m: re.search(r'\[([^\]]+)\]', m.group(0)).group(1), content)
                    fixes_applied.append(f"移除損壞連結: {pattern[:30]}...")
            
            # 5. 修復導航連結（確保正確指向上層）
            # 計算相對路徑
            rel_path = readme_path.relative_to(self.services_root)
            level = len(rel_path.parts) - 1
            
            if level > 1:  # 第二層以上才需要返回連結
                parent_path = rel_path.parent
                # 簡單處理：確保有返回上層的導航
                # 這部分可以根據實際需要細化
            
            # 6. 更新最後更新時間
            if '最後更新' in content or '🔄' in content:
                content = re.sub(
                    r'(最後更新|🔄[^:]*):.*?\d{4}年\d{1,2}月\d{1,2}日',
                    r'\1: 2025年11月27日',
                    content
                )
                fixes_applied.append("更新時間戳")
            
            # 如果有修改，寫回檔案
            if content != original_content:
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixes_applied
            
            return False, []
            
        except Exception as e:
            print(f"❌ 錯誤處理 {readme_path}: {e}")
            return False, [f"錯誤: {e}"]
    
    def fix_level(self, level: int):
        """修復特定階層的所有 README"""
        readmes = self.readmes_by_level.get(level, [])
        if not readmes:
            print(f"⏭️  Level {level}: 沒有 README")
            return
        
        print(f"\n🔧 處理 Level {level}: {len(readmes)} 個 README")
        
        fixed_count = 0
        for readme in readmes:
            rel_path = readme.relative_to(self.services_root)
            modified, fixes = self.check_and_fix_links(readme)
            
            if modified:
                fixed_count += 1
                print(f"  ✅ {rel_path}")
                for fix in fixes:
                    print(f"     - {fix}")
                self.fixes_log.append({
                    'path': str(rel_path),
                    'level': level,
                    'fixes': fixes
                })
        
        if fixed_count == 0:
            print(f"  ✓ Level {level} 所有 README 無需修復")
        else:
            print(f"  ✨ 完成 Level {level}: 修復 {fixed_count}/{len(readmes)} 個檔案")
    
    def fix_all_from_bottom_up(self):
        """從最底層開始往上修復"""
        print("=" * 60)
        print("🚀 開始由下而上修復所有 README")
        print("=" * 60)
        
        # 從最高 level 開始（最深層）
        max_level = max(self.readmes_by_level.keys()) if self.readmes_by_level else 0
        
        for level in range(max_level, -1, -1):
            self.fix_level(level)
        
        print("\n" + "=" * 60)
        print(f"✨ 完成！總共修復 {len(self.fixes_log)} 個檔案")
        print("=" * 60)
    
    def generate_report(self):
        """生成修復報告"""
        report = f"""# README 階層修復報告

**執行時間**: 2025年11月27日
**修復策略**: 由下而上（Level {max(self.readmes_by_level.keys())} → Level 0）

## 執行摘要

- **總 README 數**: {sum(len(v) for v in self.readmes_by_level.values())}
- **修復檔案數**: {len(self.fixes_log)}

## 階層分佈

"""
        for level in sorted(self.readmes_by_level.keys()):
            count = len(self.readmes_by_level[level])
            fixed = len([f for f in self.fixes_log if f['level'] == level])
            report += f"- **Level {level}**: {count} 個 README，修復 {fixed} 個\n"
        
        report += "\n## 修復詳情\n\n"
        
        for log in self.fixes_log:
            report += f"### {log['path']}\n"
            report += f"**階層**: Level {log['level']}\n\n"
            report += "**修復項目**:\n"
            for fix in log['fixes']:
                report += f"- {fix}\n"
            report += "\n"
        
        report += """## 修復類型統計

"""
        fix_types = defaultdict(int)
        for log in self.fixes_log:
            for fix in log['fixes']:
                fix_type = fix.split(':')[0] if ':' in fix else fix
                fix_types[fix_type] += 1
        
        for fix_type, count in sorted(fix_types.items(), key=lambda x: -x[1]):
            report += f"- **{fix_type}**: {count} 次\n"
        
        return report

def main():
    services_root = r"C:\D\fold7\AIVA-git\services"
    
    fixer = ReadmeHierarchyFixer(services_root)
    
    # 步驟 1: 掃描
    fixer.scan_all_readmes()
    
    # 步驟 2: 由下而上修復
    fixer.fix_all_from_bottom_up()
    
    # 步驟 3: 生成報告
    report = fixer.generate_report()
    report_path = Path(services_root) / "README_HIERARCHY_FIX_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 報告已儲存: {report_path}")

if __name__ == '__main__':
    main()
