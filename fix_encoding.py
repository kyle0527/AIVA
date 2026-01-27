#!/usr/bin/env python3
"""
檔案編碼修復腳本
目的:將 sensitive_info_detector.py 中的 Big5/GBK 亂碼還原為正確的 UTF-8 中文
"""

import sys
import os

def detect_and_fix_encoding(file_path):
    """檢測並修復檔案編碼問題"""
    
    # 讀取二進位內容
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    print(f"檔案大小: {len(raw_data)} bytes")
    print(f"檔案路徑: {file_path}\n")
    
    # 嘗試不同編碼
    encodings_to_try = [
        ('big5', 'Big5 (繁體中文)'),
        ('gbk', 'GBK (簡體中文)'),
        ('gb2312', 'GB2312'),
        ('cp950', 'CP950 (Windows Big5)'),
        ('utf-8', 'UTF-8 (當前)'),
    ]
    
    results = {}
    
    for enc_name, enc_desc in encodings_to_try:
        try:
            content = raw_data.decode(enc_name, errors='ignore')
            lines = content.split('\n')
            
            # 檢查Line 228, 230, 252附近的內容
            sample_lines = {}
            for line_num in [227, 229, 251]:  # 0-based index
                if line_num < len(lines):
                    sample_lines[line_num + 1] = lines[line_num].strip()
            
            results[enc_name] = {
                'desc': enc_desc,
                'total_lines': len(lines),
                'samples': sample_lines,
                'success': True
            }
            
            print(f"✅ {enc_desc} 解碼成功!")
            print(f"   總行數: {len(lines)}")
            for ln, content in sample_lines.items():
                # 只顯示前80個字元
                display = content[:80] + ('...' if len(content) > 80 else '')
                print(f"   Line {ln}: {display}")
            print()
            
        except Exception as e:
            results[enc_name] = {
                'desc': enc_desc,
                'success': False,
                'error': str(e)
            }
            print(f"❌ {enc_desc} 解碼失敗: {e}\n")
    
    return results

def fix_file_encoding(file_path, source_encoding='big5'):
    """修復檔案編碼"""
    
    print(f"\n{'='*60}")
    print(f"開始修復檔案編碼...")
    print(f"來源編碼: {source_encoding}")
    print(f"目標編碼: UTF-8")
    print(f"{'='*60}\n")
    
    # 備份原始檔案
    backup_path = file_path + '.bak'
    
    try:
        # 讀取原始檔案 (使用source_encoding)
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # 先備份
        with open(backup_path, 'wb') as f:
            f.write(raw_data)
        print(f"✅ 已備份原始檔案至: {backup_path}")
        
        # 解碼為文字
        try:
            content = raw_data.decode(source_encoding)
            print(f"✅ 成功使用 {source_encoding} 解碼檔案")
        except Exception as e:
            print(f"❌ 使用 {source_encoding} 解碼失敗: {e}")
            # 嘗試使用 errors='replace' 強制解碼
            content = raw_data.decode(source_encoding, errors='replace')
            print(f"⚠️  使用 replace 模式強制解碼")
        
        # 編碼為UTF-8並寫回
        with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
            f.write(content)
        
        print(f"✅ 已將修復後的內容寫回檔案 (UTF-8 編碼)")
        print(f"\n檔案修復完成!")
        print(f"如需還原,請使用備份檔案: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修復過程出錯: {e}")
        return False

if __name__ == '__main__':
    file_path = r'C:\D\fold7\AIVA-git\services\features\function_info_leak\sensitive_info_detector.py'
    
    print("="*60)
    print("檔案編碼修復工具")
    print("="*60)
    print()
    
    # 步驟1: 檢測編碼
    print("步驟 1: 檢測可能的編碼...\n")
    results = detect_and_fix_encoding(file_path)
    
    # 步驟2: 詢問用戶要使用哪種編碼修復
    print("\n" + "="*60)
    print("根據上述結果,哪種編碼看起來最正確?")
    print("="*60)
    
    # 自動判斷:找出包含最多中文字符的編碼
    best_encoding = None
    max_chinese_chars = 0
    
    for enc_name, result in results.items():
        if result['success']:
            # 統計中文字符數量 (簡單判斷:看是否包含常見中文)
            chinese_count = 0
            for line_content in result['samples'].values():
                chinese_count += sum(1 for c in line_content if '\u4e00' <= c <= '\u9fff')
            
            if chinese_count > max_chinese_chars:
                max_chinese_chars = chinese_count
                best_encoding = enc_name
    
    if best_encoding:
        print(f"\n推薦使用編碼: {best_encoding} (檢測到 {max_chinese_chars} 個中文字符)")
        
        # 步驟3: 執行修復
        user_input = input(f"\n是否使用 {best_encoding} 修復檔案? (y/n): ").strip().lower()
        
        if user_input == 'y':
            success = fix_file_encoding(file_path, source_encoding=best_encoding)
            if success:
                print("\n✅ 修復成功!請檢查檔案內容是否正確。")
            else:
                print("\n❌ 修復失敗,請檢查錯誤訊息。")
        else:
            print("\n取消修復操作。")
    else:
        print("\n❌ 無法自動判斷最佳編碼,請手動選擇。")
