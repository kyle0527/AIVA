#!/usr/bin/env python3
"""詳細分析被略過的 101 個檔案，判斷是否可以刪除或整併"""
import re
from pathlib import Path
from collections import defaultdict

def analyze_file_content(file_path):
    """分析檔案內容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本統計
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # 標題統計
        h1 = re.findall(r'^#\s+(.+)$', content, re.MULTILINE)
        h2 = re.findall(r'^#{2}\s+(.+)$', content, re.MULTILINE)
        h3 = re.findall(r'^#{3}\s+(.+)$', content, re.MULTILINE)
        
        # 判斷檔案類型
        is_node_modules = 'node_modules' in str(file_path)
        is_readme = file_path.name.upper() == 'README.MD'
        is_changelog = 'CHANGELOG' in file_path.name.upper() or 'HISTORY' in file_path.name.upper()
        
        # 內容分析
        has_license = 'MIT' in content or 'LICENSE' in content or 'Apache' in content
        has_npm_info = 'npm install' in content or 'package.json' in content
        has_github_link = 'github.com' in content.lower()
        has_code_examples = '```' in content
        
        return {
            'path': str(file_path.relative_to('.')).replace('\\', '/'),
            'size': len(content),
            'lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'h1_count': len(h1),
            'h2_count': len(h2),
            'h3_count': len(h3),
            'total_headings': len(h2) + len(h3),
            'is_node_modules': is_node_modules,
            'is_readme': is_readme,
            'is_changelog': is_changelog,
            'has_license': has_license,
            'has_npm_info': has_npm_info,
            'has_github_link': has_github_link,
            'has_code_examples': has_code_examples,
            'first_h1': h1[0] if h1 else None,
            'content_preview': '\n'.join(non_empty_lines[:10])
        }
    except Exception as e:
        return {'error': str(e), 'path': str(file_path)}

def main():
    # 掃描 services/ 所有 MD 檔案
    services_path = Path('services')
    md_files = list(services_path.rglob('*.md'))
    
    print(f"掃描 services/ 目錄，找到 {len(md_files)} 個 MD 檔案\n")
    
    # 篩選出標題太少的檔案 (total_headings < 2)
    skipped_files = []
    
    for file_path in md_files:
        info = analyze_file_content(file_path)
        if 'error' not in info and info['total_headings'] < 2:
            skipped_files.append(info)
    
    print(f"找到 {len(skipped_files)} 個標題太少的檔案\n")
    print("=" * 80)
    print("詳細分析")
    print("=" * 80)
    print()
    
    # 按類型分組
    by_location = defaultdict(list)
    node_modules_files = []
    other_files = []
    
    for info in skipped_files:
        if info['is_node_modules']:
            node_modules_files.append(info)
            # 進一步按套件分組
            parts = info['path'].split('/')
            if 'node_modules' in parts:
                idx = parts.index('node_modules')
                if idx + 1 < len(parts):
                    package_name = parts[idx + 1]
                    by_location[package_name].append(info)
        else:
            other_files.append(info)
    
    print(f"📦 Node_modules 相關檔案: {len(node_modules_files)}")
    print(f"📄 其他檔案: {len(other_files)}")
    print()
    
    # 分析 node_modules 檔案
    if node_modules_files:
        print("=" * 80)
        print("Node_modules 檔案分析")
        print("=" * 80)
        print()
        
        print(f"涉及 {len(by_location)} 個套件:")
        for package, files in sorted(by_location.items()):
            print(f"\n📦 {package} ({len(files)} 個檔案)")
            for f in files[:3]:
                print(f"  - {f['path'].split('/')[-1]}")
                print(f"    大小: {f['size']} bytes, 行數: {f['non_empty_lines']}")
                if f['first_h1']:
                    print(f"    標題: {f['first_h1']}")
            if len(files) > 3:
                print(f"  ... 還有 {len(files) - 3} 個")
        
        print()
        print("💡 建議:")
        print("  ✅ 可以安全刪除 - 這些是第三方套件的 README")
        print("  ✅ 不影響專案功能 - node_modules 由 package.json 管理")
        print("  ✅ 重新安裝時會自動恢復")
        print()
    
    # 分析非 node_modules 檔案
    if other_files:
        print("=" * 80)
        print("其他檔案分析")
        print("=" * 80)
        print()
        
        for f in other_files:
            print(f"📄 {f['path']}")
            print(f"   大小: {f['size']} bytes ({f['non_empty_lines']} 非空行)")
            print(f"   標題: H1={f['h1_count']}, H2={f['h2_count']}, H3={f['h3_count']}")
            if f['first_h1']:
                print(f"   主標題: {f['first_h1']}")
            
            # 內容特徵
            features = []
            if f['has_license']:
                features.append("包含授權資訊")
            if f['has_npm_info']:
                features.append("包含 npm 資訊")
            if f['has_github_link']:
                features.append("包含 GitHub 連結")
            if f['has_code_examples']:
                features.append("包含程式碼範例")
            
            if features:
                print(f"   特徵: {', '.join(features)}")
            
            print(f"\n   內容預覽:")
            for line in f['content_preview'].split('\n')[:5]:
                if line.strip():
                    print(f"     {line[:70]}")
            print()
        
        print("💡 建議:")
        print("  ⚠️  需要逐一審查 - 可能包含專案相關資訊")
        print("  📋 建議保留或整併到上層文件")
        print()
    
    # 總結建議
    print("=" * 80)
    print("處理建議總結")
    print("=" * 80)
    print()
    
    print(f"✅ 可以直接刪除的檔案: {len(node_modules_files)} 個")
    print(f"   位置: services/scan/engines/*/node_modules/**/*.md")
    print(f"   原因: 第三方套件文檔，由 package.json 管理")
    print()
    
    if other_files:
        print(f"⚠️  需要審查的檔案: {len(other_files)} 個")
        for f in other_files:
            print(f"   - {f['path']}")
        print()
    
    # 生成刪除命令
    print("=" * 80)
    print("建議執行的刪除命令")
    print("=" * 80)
    print()
    
    if node_modules_files:
        print("# 刪除所有 node_modules 內的 MD 檔案")
        print("# PowerShell 命令:")
        print('Get-ChildItem -Path "services/scan/engines/*/node_modules" -Filter "*.md" -Recurse | Remove-Item -Force')
        print()
        print("# 或使用 Python 腳本安全刪除:")
        print("python _delete_node_modules_md.py")
        print()

if __name__ == '__main__':
    main()
