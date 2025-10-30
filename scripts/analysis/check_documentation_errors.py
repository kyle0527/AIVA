#!/usr/bin/env python3
"""
AIVA 文檔錯誤資訊檢查器
檢查並報告文檔中是否還有舊的或錯誤的掃描器數量資訊
"""

import os
import re
from pathlib import Path

def check_documentation_errors():
    """檢查文檔中的錯誤資訊"""
    
    print("🔍 AIVA 文檔錯誤資訊檢查")
    print("=" * 60)
    
    # 需要檢查的錯誤模式 (排除報告文件中的歷史記錄)
    error_patterns = [
        (r"Python\s*\(5\s*掃描器\)", "括號內的錯誤數量"),
        (r"Go\s*\(4\s*掃描器\)", "括號內的錯誤數量"), 
        (r"Rust\s*\(1\s*掃描器\)", "括號內的錯誤數量"),
        (r"(?<!原聲稱.*|超出.*|聲稱.*|原來.*|之前.*|舊的.*)10\s*個掃描器(?!.*聲稱|.*原來)", "當前錯誤的掃描器總數")
    ]
    
    # 需要檢查的文件路徑
    check_paths = [
        "README.md",
        "AIVA_COMPREHENSIVE_GUIDE.md", 
        "*.md"  # 其他 markdown 文件
    ]
    
    errors_found = []
    
    # 掃描所有 markdown 文件
    for md_file in Path(".").rglob("*.md"):
        # 跳過報告文件和輸出目錄
        if "_out" in str(md_file) or "reports" in str(md_file):
            continue
            
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, description in error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        errors_found.append({
                            "file": str(md_file),
                            "line": line_num,
                            "content": line.strip(),
                            "pattern": pattern,
                            "description": description
                        })
                        
        except Exception as e:
            print(f"⚠️ 無法讀取文件 {md_file}: {e}")
    
    # 報告結果
    if errors_found:
        print(f"❌ 發現 {len(errors_found)} 個錯誤資訊需要修正:")
        print()
        
        for error in errors_found:
            print(f"📁 文件: {error['file']}")
            print(f"📍 行數: {error['line']}")
            print(f"📝 內容: {error['content']}")
            print(f"🎯 問題: {error['description']}")
            print(f"🔍 模式: {error['pattern']}")
            print("-" * 50)
    else:
        print("✅ 未發現錯誤的掃描器數量資訊")
        print("✅ 所有文檔已更新至正確資訊")
    
    print(f"\n📊 檢查統計:")
    print(f"   檢查文件數: {len(list(Path('.').rglob('*.md')))}")
    print(f"   發現錯誤數: {len(errors_found)}")
    print(f"   檢查模式數: {len(error_patterns)}")
    
    return len(errors_found) == 0

def suggest_corrections():
    """提供修正建議"""
    print(f"\n💡 正確的資訊應為:")
    print(f"   總掃描器數: 19 個")
    print(f"   Python 功能掃描器: 15 個")
    print(f"   AI 智能檢測器: 4 個")
    print(f"   Go/Rust 掃描器: 潛在支援 (需配置)")

if __name__ == "__main__":
    is_clean = check_documentation_errors()
    suggest_corrections()
    
    if is_clean:
        print(f"\n🎉 文檔檢查完成 - 無錯誤資訊")
    else:
        print(f"\n⚠️ 發現錯誤資訊，請手動修正")