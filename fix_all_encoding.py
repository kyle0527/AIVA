#!/usr/bin/env python3
"""
全面修復 sensitive_info_detector.py 的所有編碼問題
"""

import re

# 定義所有需要修復的亂碼 → 正確中文的映射
REPLACEMENTS = {
    # Class docstring - 第一輪
    "Detect內容中的info瘣拚": "檢測敏感資訊洩漏的功能模組",
    "Detect批捆銝剔info瘣拚": "檢測敏感資訊洩漏的功能模組",
    "銝餉:": "主要功能:",
    "蝑霅霅": "等認證資訊",
    "犖霅info": "個人識別資訊",
    "脫霅": "雲端憑證",
    "豢摨恍摮泵銝": "資料庫連線字串",
    "折頝臬頂result曹縑": "內部路徑與系統配置資訊",
    "隤文璉": "與錯誤訊息",
    "閮駁中的info": "註釋中的資訊",
    "閮駁銝剔info": "註釋中的資訊",
    "雿輻蝭:": "使用範例:",
    
    # __init__ docstring
    "Initialize detector": "初始化檢測器",
    "雿摨雿甇斤亦寥撠◤蕪": "最小嚴重性等級過濾閾值",
    
    # detect_in_html
    "批捆": "內容",
    "隞Ⅳ憛": "程式碼區塊",
    "銝剔info": "中的資訊",
    "蕪雿摨衣寥": "過濾最小嚴重性等級",
    
    # detect_in_headers
    "銝剔": "中的",
    
    # _detect_in_text
    "蝚佗": "字符作為上下文",
    "銝": "中",
}

def fix_file_encoding():
    """修復檔案編碼"""
    file_path = r'C:\D\fold7\AIVA-git\services\features\function_info_leak\sensitive_info_detector.py'
    
    print("開始修復編碼問題...")
    print(f"檔案: {file_path}\n")
    
    # 讀取檔案
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"原始檔案大小: {len(content)} 字符")
    
    # 執行所有替換
    replacement_count = 0
    for old_text, new_text in REPLACEMENTS.items():
        if old_text in content:
            count = content.count(old_text)
            content = content.replace(old_text, new_text)
            replacement_count += count
            print(f"✅ 替換 '{old_text[:20]}...' → '{new_text[:20]}...' ({count} 次)")
    
    print(f"\n總共完成 {replacement_count} 處替換")
    
    # 備份並寫入
    backup_path = file_path + '.bak2'
    with open(backup_path, 'w', encoding='utf-8') as f:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as orig:
            f.write(orig.read())
    print(f"✅ 已備份至: {backup_path}")
    
    # 寫回檔案
    with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(content)
    
    print(f"✅ 修復完成!新檔案大小: {len(content)} 字符")
    
    # 驗證修復結果
    print("\n驗證修復結果...")
    remaining_issues = []
    for old_text in REPLACEMENTS.keys():
        if old_text in content:
            remaining_issues.append(old_text)
    
    if remaining_issues:
        print(f"⚠️  仍有 {len(remaining_issues)} 個問題未完全修復:")
        for issue in remaining_issues:
            print(f"   - {issue}")
    else:
        print("✅ 所有已知編碼問題已修復!")

if __name__ == '__main__':
    fix_file_encoding()
