"""測試動態模組提取功能"""
import json
from pathlib import Path

print("="*70)
print("測試外部模組動態提取功能")
print("="*70)

test_files = [
    {
        'name': 'function_sqli (Python)',
        'path': 'module_analysis/function_sqli/analysis_results.json',
        'expected': ('function_sqli', 'features')
    },
    {
        'name': 'TypeScript Engine',
        'path': 'services/scan/typescript_engine/analysis_output/analysis_results.json',
        'expected': ('typescript_engine', 'scan')
    },
    {
        'name': 'function_crypto (Rust)',
        'path': 'services/integration/data/internal_exploration/analysis_results/rust/analysis_results.json',
        'expected': ('function_crypto', 'features')
    },
    {
        'name': 'function_authn_go (Go)',
        'path': 'services/features/function_authn_go/analysis_output/analysis_results.json',
        'expected': ('function_authn_go', 'features')
    }
]

for test in test_files:
    print(f"\n[測試] {test['name']}")
    print(f"  路徑: {test['path']}")
    
    try:
        data = json.load(open(test['path'], encoding='utf-8'))
        
        # 策略1: 從檔案路徑本身提取
        file_path = test['path']
        print(f"  檔案路徑分析: {file_path}")
        
        import re
        
        # 檢測 function_*
        function_match = re.search(r'function_([a-z_0-9]+)', file_path)
        if function_match:
            module_name = f"function_{function_match.group(1)}"
            print(f"  [OK] 提取模組: {module_name} (features)")
            continue
        
        # 檢測 scan/*
        if '/scan/' in file_path or '\\scan\\' in file_path:
            scan_patterns = [
                (r'typescript', 'typescript_engine'),
                (r'rust', 'rust_engine'),
                (r'scan_engine', 'scan_engine')
            ]
            found = False
            for pattern, name in scan_patterns:
                if re.search(pattern, file_path):
                    print(f"  [OK] 提取模組: {name} (scan)")
                    found = True
                    break
            if found:
                continue
        
        print(f"  [警告] 無法從檔案路徑提取，嘗試從數據內容...")
        
        # 策略2: 從數據中的 target 欄位
        target = data.get('target', '')
        if target:
            print(f"  target 欄位: {target}")
            function_match = re.search(r'function_([a-z_0-9]+)', target)
            if function_match:
                module_name = f"function_{function_match.group(1)}"
                print(f"  [OK] 從 target 提取: {module_name} (features)")
                continue
        
        print(f"  [失敗] 無法提取模組名稱")
        
    except Exception as e:
        print(f"  [錯誤] {e}")

print("\n" + "="*70)
print("結論：應使用 分析檔案的路徑 來提取模組名稱")
print("="*70)
print("""
最佳策略：
1. 從分類器的 input_dir 參數中提取模組名稱
   範例: module_analysis/function_sqli → function_sqli
         services/scan/typescript_engine → typescript_engine
         
2. 將模組名稱存儲在分類器初始化時

3. 在分類流程時使用這個模組名稱

這樣就不需要從每個 flow 的路徑中提取！
""")
