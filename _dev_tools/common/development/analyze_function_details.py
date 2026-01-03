import json

def analyze_function_details():
    print("=== Function Details 深度分析 ===")
    
    # 檢查V3的原始數據
    print("1. V3原始分析數據:")
    with open('aiva_core_analysis_v3/analysis_results.json', 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    
    v3_fd = v3_data.get('function_details', {})
    print(f"   Type: {type(v3_fd)}")
    print(f"   Keys: {list(v3_fd.keys())}")
    
    if isinstance(v3_fd, dict):
        function_map = v3_fd.get('function_map', {})
        script_functions = v3_fd.get('script_functions', {})
        
        print(f"   function_map: {len(function_map)} 個函數")
        print(f"   script_functions: {len(script_functions)} 個腳本")
        
        # 檢查function_map的結構
        if function_map:
            first_func_name = list(function_map.keys())[0]
            first_func_info = function_map[first_func_name]
            print(f"   function_map範例 - {first_func_name}:")
            for key, value in first_func_info.items():
                print(f"     {key}: {value}")
        
        # 檢查script_functions的結構
        if script_functions:
            first_script_name = list(script_functions.keys())[0]
            first_script_info = script_functions[first_script_name]
            print(f"   script_functions範例 - {first_script_name}:")
            for key, value in first_script_info.items():
                if key == 'functions' and isinstance(value, dict):
                    print(f"     functions: {len(value)} 個函數")
                    func_names = list(value.keys())[:3]
                    print(f"     函數名稱(前3個): {func_names}")
                else:
                    print(f"     {key}: {value}")
    
    # 檢查分類器如何處理function_details
    print("\n2. 分類器對function_details的處理:")
    
    # 檢查V6分類結果
    print("   V6分類結果:")
    with open('aiva_core_classification_v6_endpoint/classification_data.json', 'r', encoding='utf-8') as f:
        v6_data = json.load(f)
    
    v6_fd = v6_data.get('function_details', {})
    print(f"     Type: {type(v6_fd)}")
    print(f"     Content: {v6_fd}")
    
    # 檢查V8分類結果
    print("   V8分類結果:")
    with open('classification_data_v8_endpoint.json/classification_data.json', 'r', encoding='utf-8') as f:
        v8_data = json.load(f)
    
    v8_fd = v8_data.get('function_details', {})
    print(f"     Type: {type(v8_fd)}")
    print(f"     Content: {v8_fd}")
    
    # 檢查分類器載入邏輯
    print("\n3. 分類器載入邏輯檢查:")
    
    # 模擬分類器的載入過程
    print("   模擬載入V3數據:")
    function_details = v3_data.get('function_details', {})
    function_map = function_details.get('function_map', {})
    script_functions = function_details.get('script_functions', {})
    
    print(f"     載入了 {len(function_map)} 個函數詳細信息")
    print(f"     載入了 {len(script_functions)} 個腳本的函數映射")
    
    # 檢查分類器是否將這些信息保存到輸出
    print("\n4. 問題診斷:")
    if len(function_map) > 0 and len(v6_fd) == 0:
        print("   ❌ 問題確認：分類器載入了function_details但沒有保存到輸出")
        print("   📝 原因：分類器的save_results方法可能沒有包含function_details")
    
    # 檢查實際的函數內容範例
    print("\n5. Function_details實際內容範例:")
    if function_map:
        print("   函數映射範例:")
        for i, (func_name, func_info) in enumerate(list(function_map.items())[:2]):
            print(f"     {i+1}. {func_name}:")
            print(f"        所屬腳本: {func_info.get('script_name', 'N/A')}")
            print(f"        文件路徑: {func_info.get('file_path', 'N/A')}")
            print(f"        行號: {func_info.get('line_number', 'N/A')}")
            print(f"        調用函數: {func_info.get('called_functions', [])}")
            print(f"        是否異步: {func_info.get('is_async', False)}")
            print(f"        是否入口點: {func_info.get('is_entry_point', False)}")

if __name__ == '__main__':
    analyze_function_details()