#!/usr/bin/env python3
"""驗證提取的文件數量和完整性"""
import json
from pathlib import Path

def main():
    print("=== 驗證提取結果 ===\n")
    
    # 1. 檢查實際文件數
    node_modules = Path('services/scan/engines/typescript_engine/node_modules')
    actual_files = list(node_modules.rglob('*.md'))
    print(f"node_modules 中實際的 .md 文件數: {len(actual_files)}")
    print()
    
    # 2. 檢查提取的數據
    with open('_node_modules_md_content.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_total = (
        len(data['core']) + 
        len(data['dev_tools']) + 
        len(data['dependencies']) + 
        len(data['licenses']) + 
        len(data['changelogs']) + 
        len(data['other'])
    )
    
    print("提取到 JSON 的文件數:")
    print(f"  核心套件: {len(data['core'])}")
    print(f"  開發工具: {len(data['dev_tools'])}")
    print(f"  傳遞依賴: {len(data['dependencies'])}")
    print(f"  授權文件: {len(data['licenses'])}")
    print(f"  變更記錄: {len(data['changelogs'])}")
    print(f"  其他文檔: {len(data['other'])}")
    print(f"  總計: {extracted_total}")
    print()
    
    # 3. 比對
    if len(actual_files) == extracted_total:
        print(f"✅ 完全匹配！已提取全部 {extracted_total} 個文件")
    else:
        print(f"⚠️  數量不符: 實際 {len(actual_files)} vs 提取 {extracted_total}")
        print(f"   差異: {abs(len(actual_files) - extracted_total)} 個文件")
    
    print()
    print("=== 檢查生成的指南 ===\n")
    
    guide_file = Path('services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md')
    if guide_file.exists():
        content = guide_file.read_text(encoding='utf-8')
        print(f"DEPENDENCIES_GUIDE.md 大小: {len(content)} bytes")
        print(f"行數: {len(content.splitlines())}")
        
        # 檢查是否提到所有分類
        checks = {
            '核心套件': '核心運行時依賴' in content,
            '開發工具': '開發工具依賴' in content,
            '傳遞依賴': '主要傳遞依賴' in content,
            '快速參考': '快速參考' in content,
            '完整清單': '完整套件清單' in content,
        }
        
        print("\n內容檢查:")
        for name, exists in checks.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {name}")
    else:
        print("❌ DEPENDENCIES_GUIDE.md 不存在！")

if __name__ == '__main__':
    main()
