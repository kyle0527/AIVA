#!/usr/bin/env python3
"""全面檢查專案中所有 MD 文件的目錄狀態"""
import re
from pathlib import Path
from collections import defaultdict

def has_toc(content):
    """檢查是否有目錄"""
    lines = content.split('\n')
    
    # 檢查各種目錄格式
    toc_patterns = [
        r'##\s*目錄',
        r'##\s*📑\s*目錄',
        r'##\s*Table of Contents',
        r'##\s*Contents',
        r'##\s*索引',
    ]
    
    for line in lines[:50]:  # 檢查前50行
        for pattern in toc_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
    
    return False

def analyze_md_structure(content):
    """分析 MD 文件結構"""
    lines = content.split('\n')
    
    h1_count = 0
    h2_count = 0
    h3_count = 0
    
    for line in lines:
        if line.startswith('# '):
            h1_count += 1
        elif line.startswith('## '):
            h2_count += 1
        elif line.startswith('### '):
            h3_count += 1
    
    return {
        'h1': h1_count,
        'h2': h2_count,
        'h3': h3_count,
        'total_headers': h1_count + h2_count + h3_count
    }

def main():
    print("=== 全面檢查專案中的 MD 文件 ===\n")
    
    # 收集所有 MD 文件（排除 .git）
    all_md_files = []
    for md_file in Path('.').rglob('*.md'):
        if '.git' not in str(md_file):
            all_md_files.append(md_file)
    
    print(f"找到 {len(all_md_files)} 個 MD 文件\n")
    
    # 分類統計
    by_location = defaultdict(list)
    with_toc = []
    without_toc = []
    node_modules_files = []
    
    print("正在分析...\n")
    
    for md_file in all_md_files:
        try:
            content = md_file.read_text(encoding='utf-8', errors='ignore')
            
            # 分類
            if 'node_modules' in str(md_file):
                node_modules_files.append(md_file)
                continue
            
            # 檢查目錄
            has_toc_flag = has_toc(content)
            structure = analyze_md_structure(content)
            
            file_info = {
                'path': str(md_file),
                'size': len(content),
                'lines': len(content.splitlines()),
                'has_toc': has_toc_flag,
                'structure': structure
            }
            
            # 按位置分類
            if str(md_file).startswith('reports'):
                by_location['reports'].append(file_info)
            elif str(md_file).startswith('docs'):
                by_location['docs'].append(file_info)
            elif str(md_file).startswith('guides'):
                by_location['guides'].append(file_info)
            elif str(md_file).startswith('services'):
                by_location['services'].append(file_info)
            else:
                by_location['root'].append(file_info)
            
            # 目錄狀態
            if has_toc_flag:
                with_toc.append(file_info)
            else:
                without_toc.append(file_info)
                
        except Exception as e:
            print(f"⚠️  無法讀取 {md_file}: {e}")
    
    # ===== 報告 =====
    
    print("=" * 60)
    print("📊 統計報告")
    print("=" * 60)
    print()
    
    # 總體統計
    user_files = len(all_md_files) - len(node_modules_files)
    print(f"總 MD 文件: {len(all_md_files)}")
    print(f"  - 用戶文件: {user_files}")
    print(f"  - node_modules: {len(node_modules_files)}")
    print()
    
    print(f"目錄狀態:")
    print(f"  ✅ 有目錄: {len(with_toc)} 個 ({len(with_toc)/user_files*100:.1f}%)")
    print(f"  ❌ 無目錄: {len(without_toc)} 個 ({len(without_toc)/user_files*100:.1f}%)")
    print()
    
    # 按位置統計
    print("按位置分類:")
    for location, files in sorted(by_location.items()):
        toc_count = sum(1 for f in files if f['has_toc'])
        print(f"  {location:15s}: {len(files):3d} 個 (有目錄: {toc_count:3d}, 無: {len(files)-toc_count:3d})")
    print()
    
    # 詳細列出沒有目錄的文件
    print("=" * 60)
    print(f"❌ 沒有目錄的文件 ({len(without_toc)} 個)")
    print("=" * 60)
    print()
    
    # 按位置分組顯示
    without_toc_by_loc = defaultdict(list)
    for f in without_toc:
        path = f['path']
        if path.startswith('reports'):
            without_toc_by_loc['reports'].append(f)
        elif path.startswith('docs'):
            without_toc_by_loc['docs'].append(f)
        elif path.startswith('guides'):
            without_toc_by_loc['guides'].append(f)
        elif path.startswith('services'):
            without_toc_by_loc['services'].append(f)
        else:
            without_toc_by_loc['root'].append(f)
    
    for location, files in sorted(without_toc_by_loc.items()):
        if files:
            print(f"\n### {location.upper()} ({len(files)} 個)")
            print("-" * 60)
            for f in sorted(files, key=lambda x: x['path']):
                headers = f['structure']['total_headers']
                print(f"  {f['path']}")
                print(f"    大小: {f['size']} bytes, 行數: {f['lines']}, 標題數: {headers}")
    
    # 儲存詳細報告
    output_lines = []
    output_lines.append("# 專案 MD 文件完整檢查報告")
    output_lines.append("")
    output_lines.append(f"生成時間: 2025-11-27")
    output_lines.append("")
    output_lines.append("## 📑 目錄")
    output_lines.append("")
    output_lines.append("- [總體統計](#總體統計)")
    output_lines.append("- [按位置分類](#按位置分類)")
    output_lines.append("- [沒有目錄的文件](#沒有目錄的文件)")
    output_lines.append("- [有目錄的文件](#有目錄的文件)")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    
    output_lines.append("## 總體統計")
    output_lines.append("")
    output_lines.append(f"- **總 MD 文件**: {len(all_md_files)} 個")
    output_lines.append(f"  - 用戶文件: {user_files} 個")
    output_lines.append(f"  - node_modules: {len(node_modules_files)} 個")
    output_lines.append("")
    output_lines.append(f"- **目錄狀態**:")
    output_lines.append(f"  - ✅ 有目錄: {len(with_toc)} 個 ({len(with_toc)/user_files*100:.1f}%)")
    output_lines.append(f"  - ❌ 無目錄: {len(without_toc)} 個 ({len(without_toc)/user_files*100:.1f}%)")
    output_lines.append("")
    
    output_lines.append("## 按位置分類")
    output_lines.append("")
    output_lines.append("| 位置 | 總數 | 有目錄 | 無目錄 | 完成率 |")
    output_lines.append("|------|------|--------|--------|--------|")
    for location, files in sorted(by_location.items()):
        toc_count = sum(1 for f in files if f['has_toc'])
        no_toc = len(files) - toc_count
        rate = toc_count / len(files) * 100 if files else 0
        output_lines.append(f"| {location} | {len(files)} | {toc_count} | {no_toc} | {rate:.1f}% |")
    output_lines.append("")
    
    output_lines.append("## 沒有目錄的文件")
    output_lines.append("")
    for location, files in sorted(without_toc_by_loc.items()):
        if files:
            output_lines.append(f"### {location}")
            output_lines.append("")
            for f in sorted(files, key=lambda x: x['path']):
                headers = f['structure']['total_headers']
                output_lines.append(f"- `{f['path']}`")
                output_lines.append(f"  - 大小: {f['size']} bytes")
                output_lines.append(f"  - 行數: {f['lines']}")
                output_lines.append(f"  - 標題數: {headers} (H1:{f['structure']['h1']}, H2:{f['structure']['h2']}, H3:{f['structure']['h3']})")
                output_lines.append("")
    
    output_lines.append("## 有目錄的文件")
    output_lines.append("")
    with_toc_by_loc = defaultdict(list)
    for f in with_toc:
        path = f['path']
        if path.startswith('reports'):
            with_toc_by_loc['reports'].append(f)
        elif path.startswith('docs'):
            with_toc_by_loc['docs'].append(f)
        elif path.startswith('guides'):
            with_toc_by_loc['guides'].append(f)
        elif path.startswith('services'):
            with_toc_by_loc['services'].append(f)
        else:
            with_toc_by_loc['root'].append(f)
    
    for location, files in sorted(with_toc_by_loc.items()):
        if files:
            output_lines.append(f"### {location} ({len(files)} 個)")
            output_lines.append("")
            for f in sorted(files, key=lambda x: x['path']):
                output_lines.append(f"- `{f['path']}` ({f['size']} bytes, {f['lines']} 行)")
            output_lines.append("")
    
    # 儲存報告
    report_path = Path('MD_FILES_COMPLETE_CHECK_REPORT.md')
    report_path.write_text('\n'.join(output_lines), encoding='utf-8')
    
    print()
    print("=" * 60)
    print(f"✅ 詳細報告已儲存: {report_path}")
    print("=" * 60)
    print()
    
    # 總結
    print("🎯 快速總結:")
    print(f"  - 總共 {len(all_md_files)} 個 MD 文件")
    print(f"  - 用戶文件 {user_files} 個")
    print(f"  - 有目錄 {len(with_toc)} 個 ({len(with_toc)/user_files*100:.1f}%)")
    print(f"  - 需要添加目錄 {len(without_toc)} 個")
    print()

if __name__ == '__main__':
    main()
