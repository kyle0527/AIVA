#!/usr/bin/env python3
"""
在專案樹中標注沒有基本功能的腳本
"""

import json
from pathlib import Path

def main():
    # 使用相對路徑，從項目根目錄計算
    project_root = Path(__file__).parent.parent.parent
    result_file = project_root / '_out' / 'script_functionality_report.json'
    tree_file = project_root / '_out' / 'tree_ultimate_chinese_20251019_082355.txt'
    output_file = project_root / '_out' / 'tree_with_functionality_marks.txt'
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 建立檔案標記字典
    file_marks = {}
    
    for item in results['no_functionality']:
        file_marks[item['file']] = '❌ 無功能'
    
    for item in results['minimal_functionality']:
        file_marks[item['file']] = '🔶 基本架構'
    
    for item in results['partial_functionality']:
        file_marks[item['file']] = '⚠️  部分功能'
    
    # 讀取樹狀圖
    tree_content = tree_file.read_text(encoding='utf-8')
    lines = tree_content.split('\n')
    
    # 標注檔案
    marked_lines = []
    for line in lines:
        marked_line = line
        # 檢查每個檔案路徑
        for file_path, mark in file_marks.items():
            # 處理路徑中的反斜線
            file_name = Path(file_path).name
            if file_name in line and file_name.endswith(('.py', '.ps1')):
                # 確認是檔案行(包含 # 註解)
                if '#' in line:
                    # 在註解前插入標記
                    marked_line = line.replace(' #', f' {mark} #')
                    break
        
        marked_lines.append(marked_line)
    
    # 寫入結果
    output_content = '\n'.join(marked_lines)
    
    # 添加統計摘要
    summary = f"""
{'='*100}
腳本功能性標記說明
{'='*100}

❌ 無功能        : {len(results['no_functionality'])} 個檔案 - 需要完整實作
🔶 基本架構      : {len(results['minimal_functionality'])} 個檔案 - 需要補充功能
⚠️  部分功能     : {len(results['partial_functionality'])} 個檔案 - 可以改進
✅ 完整功能      : {len(results['full_functionality'])} 個檔案 - 正常運作

總計: {sum(len(v) for v in results.values())} 個腳本檔案

{'='*100}

"""
    
    final_content = summary + output_content
    output_file.write_text(final_content, encoding='utf-8')
    
    print(f"✅ 已生成標記版本: {output_file}")
    print(f"\n統計:")
    print(f"  ❌ 無功能: {len(results['no_functionality'])}")
    print(f"  🔶 基本架構: {len(results['minimal_functionality'])}")
    print(f"  ⚠️  部分功能: {len(results['partial_functionality'])}")
    print(f"  ✅ 完整功能: {len(results['full_functionality'])}")

if __name__ == '__main__':
    main()
