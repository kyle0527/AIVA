#!/usr/bin/env python3


"""修正移動文件後的內部連結"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
project_root = Path(__file__).resolve().parent.parent.parent.parent

# 文件移動映射
MOVED_FILES = {
    'services/scan/engines/rust_engine/USAGE_GUIDE.md': 'docs/guides/services/rust_engine_USAGE_GUIDE.md',
    'services/scan\\engines\\rust_engine\\USAGE_GUIDE.md': 'docs/guides/services/rust_engine_USAGE_GUIDE.md',
    'services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md': 'docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md',
    'services/scan\\engines\\typescript_engine\\DEPENDENCIES_GUIDE.md': 'docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md',
    'services/core/aiva_core/USAGE_GUIDE.md': 'docs/guides/services/aiva_core_USAGE_GUIDE.md',
    'services/core\\aiva_core\\USAGE_GUIDE.md': 'docs/guides/services/aiva_core_USAGE_GUIDE.md',
    'services/features/DEVELOPMENT_STANDARDS.md': 'docs/development/services_DEVELOPMENT_STANDARDS.md',
    'services/features\\DEVELOPMENT_STANDARDS.md': 'docs/development/services_DEVELOPMENT_STANDARDS.md',
}

def calculate_relative_path(from_file: Path, to_file: Path) -> str:
    """計算相對路徑"""
    try:
        rel_path = os.path.relpath(to_file, from_file.parent)
        # 將反斜線轉換為正斜線（Markdown 標準）
        return rel_path.replace('\\', '/')
    except ValueError:
        # 如果在不同磁碟，返回絕對路徑
        return str(to_file)

def fix_links_in_file(file_path: Path, root_dir: Path) -> Tuple[int, List[str]]:
    """修正單個文件中的連結"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 無法讀取 {file_path}: {e}")
        return 0, []
    
    original_content = content
    fixed_count = 0
    changes = []
    
    # 查找所有 Markdown 連結
    # 模式: [text](path) 或 直接路徑引用
    patterns = [
        r'\[([^\]]+)\]\(([^\)]+)\)',  # [text](link)
        r'`([^`]*(?:USAGE_GUIDE|DEPENDENCIES_GUIDE|DEVELOPMENT_STANDARDS)\.md[^`]*)`',  # `path`
        r'(?:路徑|Path|位置)[:：]\s*`([^`]+)`',  # 路徑: `path`
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, content))
        for match in reversed(matches):  # 從後往前處理，避免位置偏移
            if pattern == patterns[0]:  # Markdown 連結
                link_text = match.group(1)
                link_path = match.group(2)
                original_link = match.group(0)
            else:  # 其他模式
                link_path = match.group(1) if len(match.groups()) == 1 else match.group(2)
                original_link = match.group(0)
                link_text = None
            
            # 檢查是否是我們需要更新的路徑
            needs_update = False
            new_target = None
            
            for old_path, new_path in MOVED_FILES.items():
                # 標準化路徑進行比較
                normalized_link = link_path.replace('\\', '/').strip()
                normalized_old = old_path.replace('\\', '/')
                
                if normalized_old in normalized_link or link_path.endswith(Path(old_path).name):
                    needs_update = True
                    new_target = new_path
                    break
            
            if needs_update and new_target:
                # 計算新的相對路徑
                new_target_abs = root_dir / new_target
                current_file_abs = file_path
                
                new_relative_path = calculate_relative_path(current_file_abs, new_target_abs)
                
                # 構建新的連結
                if link_text:
                    new_link = f'[{link_text}]({new_relative_path})'
                else:
                    # 對於路徑引用，替換為新路徑
                    new_link = original_link.replace(link_path, new_target)
                
                # 替換
                start, end = match.span()
                content = content[:start] + new_link + content[end:]
                fixed_count += 1
                
                changes.append(f"  {original_link} → {new_link}")
    
    # 如果有變更，寫回文件
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return fixed_count, changes
        except Exception as e:
            print(f"⚠️ 無法寫入 {file_path}: {e}")
            return 0, []
    
    return 0, []

def main():
    root_dir = project_root
    
    print("="*80)
    print("🔗 開始修正移動文件的內部連結")
    print("="*80)
    
    # 掃描所有 MD 文件（排除 node_modules 和移動的文件本身）
    md_files = []
    for md_file in root_dir.rglob("*.md"):
        if 'node_modules' in str(md_file):
            continue
        if any(moved in str(md_file) for moved in ['docs/guides/services/', 'docs/development/services_']):
            continue
        md_files.append(md_file)
    
    print(f"\n📁 找到 {len(md_files)} 個需要檢查的 MD 文件")
    print(f"🎯 需要修正的文件映射:")
    for old, new in MOVED_FILES.items():
        if '\\' not in old:  # 只顯示正斜線版本
            print(f"  • {old} → {new}")
    
    print("\n" + "-"*80)
    print("開始處理...")
    print("-"*80)
    
    total_fixed = 0
    files_with_changes = []
    
    for md_file in md_files:
        rel_path = md_file.relative_to(root_dir)
        fixed_count, changes = fix_links_in_file(md_file, root_dir)
        
        if fixed_count > 0:
            total_fixed += fixed_count
            files_with_changes.append((rel_path, fixed_count, changes))
            print(f"\n✅ {rel_path}")
            print(f"   修正 {fixed_count} 個連結:")
            for change in changes[:3]:  # 只顯示前3個變更
                print(change)
            if len(changes) > 3:
                print(f"   ... 還有 {len(changes) - 3} 個變更")
    
    print("\n" + "="*80)
    print("📊 修正摘要")
    print("="*80)
    print(f"✅ 檢查文件數: {len(md_files)}")
    print(f"🔧 修改文件數: {len(files_with_changes)}")
    print(f"🔗 修正連結數: {total_fixed}")
    
    if files_with_changes:
        print(f"\n📝 已修改的文件:")
        for file_path, count, _ in files_with_changes:
            print(f"  • {file_path} ({count} 個連結)")
    
    # 生成報告
    report = []
    report.append("# 🔗 內部連結修正報告\n\n")
    report.append("---\n")
    report.append("**執行時間**: 2025年11月27日\n\n")
    report.append("## 📑 目錄\n\n")
    report.append("- [修正摘要](#修正摘要)\n")
    report.append("- [文件移動映射](#文件移動映射)\n")
    report.append("- [修正詳情](#修正詳情)\n\n")
    report.append("---\n\n")
    report.append("## 📊 修正摘要\n\n")
    report.append(f"- **檢查文件數**: {len(md_files)}\n")
    report.append(f"- **修改文件數**: {len(files_with_changes)}\n")
    report.append(f"- **修正連結數**: {total_fixed}\n\n")
    
    report.append("## 📂 文件移動映射\n\n")
    report.append("以下文件已移動到新位置:\n\n")
    for old, new in MOVED_FILES.items():
        if '\\' not in old:
            report.append(f"- `{old}` → `{new}`\n")
    report.append("\n")
    
    report.append("## 🔧 修正詳情\n\n")
    if files_with_changes:
        for file_path, count, changes in files_with_changes:
            report.append(f"### {file_path}\n\n")
            report.append(f"修正 {count} 個連結:\n\n")
            for change in changes:
                report.append(f"-{change}\n")
            report.append("\n")
    else:
        report.append("✅ 沒有發現需要修正的連結\n\n")
    
    report.append("---\n\n")
    report.append("*報告生成時間: 2025年11月27日*\n")
    
    report_file = root_dir / "LINK_FIX_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"\n📄 報告已保存: {report_file}")
    print("\n✨ 完成！")

if __name__ == "__main__":
    main()
