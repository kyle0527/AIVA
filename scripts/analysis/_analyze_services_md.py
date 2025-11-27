#!/usr/bin/env python3
"""分析 services/ 目錄 MD 檔案的詳細狀況"""
import re
from pathlib import Path
from collections import defaultdict

def analyze_file(file_path):
    """分析單個檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查是否有 H1
        has_h1 = bool(re.search(r'^#\s+.+$', content, re.MULTILINE))
        
        # 檢查是否有目錄
        has_toc_emoji = bool(re.search(r'## [📑📋] 目錄', content))
        has_toc_text = bool(re.search(r'## (Table of Contents|目錄|Contents)', content, re.IGNORECASE))
        has_toc = has_toc_emoji or has_toc_text
        
        # 統計標題數量
        h1_count = len(re.findall(r'^#\s+.+$', content, re.MULTILINE))
        h2_count = len(re.findall(r'^#{2}\s+.+$', content, re.MULTILINE))
        h3_count = len(re.findall(r'^#{3}\s+.+$', content, re.MULTILINE))
        h4_count = len(re.findall(r'^#{4}\s+.+$', content, re.MULTILINE))
        h5_count = len(re.findall(r'^#{5}\s+.+$', content, re.MULTILINE))
        h6_count = len(re.findall(r'^#{6}\s+.+$', content, re.MULTILINE))
        
        total_headings = h2_count + h3_count + h4_count + h5_count + h6_count
        
        # 檔案大小
        file_size = len(content)
        line_count = content.count('\n')
        
        return {
            'has_h1': has_h1,
            'has_toc': has_toc,
            'has_toc_emoji': has_toc_emoji,
            'h1_count': h1_count,
            'h2_count': h2_count,
            'h3_count': h3_count,
            'h4_count': h4_count,
            'h5_count': h5_count,
            'h6_count': h6_count,
            'total_headings': total_headings,
            'file_size': file_size,
            'line_count': line_count,
            'is_empty': file_size < 50,
            'content_preview': content[:200] if content else ""
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    services_path = Path('services')
    md_files = list(services_path.rglob('*.md'))
    
    print(f"找到 {len(md_files)} 個 MD 檔案\n")
    
    # 分類統計
    stats = defaultdict(list)
    
    for file_path in md_files:
        info = analyze_file(file_path)
        rel_path = str(file_path.relative_to('.')).replace('\\', '/')
        
        if 'error' in info:
            stats['error'].append((rel_path, info['error']))
        elif info['is_empty']:
            stats['empty'].append((rel_path, info['file_size']))
        elif info['has_toc_emoji']:
            stats['has_toc_emoji'].append(rel_path)
        elif info['has_toc']:
            stats['has_toc_text'].append((rel_path, info['content_preview']))
        elif info['total_headings'] < 2:
            stats['too_few_headings'].append((rel_path, info['total_headings'], info['h1_count']))
        elif not info['has_h1'] and info['total_headings'] >= 2:
            stats['no_h1_but_has_headings'].append((rel_path, info['total_headings']))
        elif info['has_h1'] and info['total_headings'] >= 2:
            stats['needs_toc'].append((rel_path, info['h1_count'], info['total_headings']))
    
    # 輸出分析結果
    print("=" * 70)
    print("分析結果")
    print("=" * 70)
    print()
    
    print(f"📊 總覽:")
    print(f"  總檔案數: {len(md_files)}")
    print(f"  有表情目錄 (📑/📋): {len(stats['has_toc_emoji'])}")
    print(f"  有文字目錄: {len(stats['has_toc_text'])}")
    print(f"  空檔案/太小: {len(stats['empty'])}")
    print(f"  標題太少 (<2): {len(stats['too_few_headings'])}")
    print(f"  無H1但有標題: {len(stats['no_h1_but_has_headings'])}")
    print(f"  需要加目錄: {len(stats['needs_toc'])}")
    print(f"  錯誤: {len(stats['error'])}")
    print()
    
    # 詳細列出各類別
    if stats['has_toc_text']:
        print(f"\n【有文字目錄但無表情符號】({len(stats['has_toc_text'])} 個)")
        print("可能需要統一格式:")
        for path, preview in stats['has_toc_text'][:10]:
            print(f"  - {path}")
            # 找出目錄標題
            toc_match = re.search(r'##\s+(.*目錄.*|Table of Contents|Contents)', preview, re.IGNORECASE)
            if toc_match:
                print(f"    格式: ## {toc_match.group(1)}")
        if len(stats['has_toc_text']) > 10:
            print(f"  ... 還有 {len(stats['has_toc_text']) - 10} 個")
    
    if stats['too_few_headings']:
        print(f"\n【標題太少 (<2 個二級以上標題)】({len(stats['too_few_headings'])} 個)")
        print("這些檔案內容簡單，不需要目錄:")
        for path, heading_count, h1_count in stats['too_few_headings'][:5]:
            print(f"  - {path} (H1:{h1_count}, 總標題:{heading_count})")
        if len(stats['too_few_headings']) > 5:
            print(f"  ... 還有 {len(stats['too_few_headings']) - 5} 個")
    
    if stats['no_h1_but_has_headings']:
        print(f"\n【無H1但有多個標題】({len(stats['no_h1_but_has_headings'])} 個)")
        print("這些檔案可以添加目錄:")
        for path, heading_count in stats['no_h1_but_has_headings'][:10]:
            print(f"  - {path} (標題數: {heading_count})")
        if len(stats['no_h1_but_has_headings']) > 10:
            print(f"  ... 還有 {len(stats['no_h1_but_has_headings']) - 10} 個")
    
    if stats['needs_toc']:
        print(f"\n【需要添加目錄】({len(stats['needs_toc'])} 個)")
        print("有H1且有足夠標題，應該添加目錄:")
        for path, h1_count, heading_count in stats['needs_toc'][:10]:
            print(f"  - {path} (H1:{h1_count}, 標題:{heading_count})")
        if len(stats['needs_toc']) > 10:
            print(f"  ... 還有 {len(stats['needs_toc']) - 10} 個")
    
    if stats['empty']:
        print(f"\n【空檔案/太小】({len(stats['empty'])} 個)")
        for path, size in stats['empty'][:5]:
            print(f"  - {path} ({size} bytes)")
        if len(stats['empty']) > 5:
            print(f"  ... 還有 {len(stats['empty']) - 5} 個")
    
    if stats['error']:
        print(f"\n【錯誤】({len(stats['error'])} 個)")
        for path, error in stats['error'][:5]:
            print(f"  - {path}: {error}")
    
    print()
    print("=" * 70)
    print("結論:")
    print("=" * 70)
    actual_need_toc = len(stats['needs_toc']) + len(stats['no_h1_but_has_headings'])
    already_has = len(stats['has_toc_emoji']) + len(stats['has_toc_text'])
    print(f"  ✅ 已有目錄: {already_has} 個")
    print(f"  🔧 需要處理: {actual_need_toc} 個")
    print(f"  ⚠️  格式需統一: {len(stats['has_toc_text'])} 個 (有目錄但格式不統一)")
    print(f"  ℹ️  不需處理: {len(stats['too_few_headings']) + len(stats['empty'])} 個 (內容太少)")
    print()

if __name__ == '__main__':
    main()
