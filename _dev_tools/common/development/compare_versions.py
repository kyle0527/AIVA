import json
import os

def compare_three_versions():
    print("=== 三版本數據比對分析 ===")
    
    # 路徑定義
    v3_path = "aiva_core_analysis_v3/analysis_results.json"
    v6_path = "aiva_core_classification_v6_endpoint/classification_data.json"
    v8_path = "classification_data_v8_endpoint.json/classification_data.json"
    
    # 檢查文件存在性
    print("1. 文件存在性檢查:")
    files = [v3_path, v6_path, v8_path]
    file_exists = {}
    for file_path in files:
        exists = os.path.exists(file_path)
        file_exists[file_path] = exists
        print(f"   {file_path}: {'✅' if exists else '❌'}")
    
    if not all(file_exists.values()):
        print("❌ 部分文件不存在，無法進行完整比對")
        return
    
    # 載入數據
    print("\n2. 載入數據:")
    try:
        with open(v3_path, 'r', encoding='utf-8') as f:
            v3_data = json.load(f)
        print("   V3 (analysis_v3): ✅")
        
        with open(v6_path, 'r', encoding='utf-8') as f:
            v6_data = json.load(f)
        print("   V6 (classification_v6_endpoint): ✅")
        
        with open(v8_path, 'r', encoding='utf-8') as f:
            v8_data = json.load(f)
        print("   V8 (classification_v8_endpoint): ✅")
        
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return
    
    # 基本統計比對
    print("\n3. 基本統計比對:")
    print("版本".ljust(25), "文件數".ljust(10), "Flow數".ljust(10), "函數數".ljust(10), "腳本數".ljust(10))
    print("-" * 65)
    
    # V3 統計
    v3_files = v3_data.get('total_files_processed', 0)
    v3_flows = len(v3_data.get('flow_chains', []))
    v3_funcs = len(v3_data.get('function_details', {}).get('function_map', {}))
    v3_scripts = len(v3_data.get('function_details', {}).get('script_functions', {}))
    print(f"V3 (原始分析)".ljust(25), str(v3_files).ljust(10), str(v3_flows).ljust(10), str(v3_funcs).ljust(10), str(v3_scripts).ljust(10))
    
    # V6 統計
    v6_flows = len(v6_data.get('flows', []))
    v6_funcs = len(v6_data.get('function_details', {}).get('function_map', {}))
    v6_scripts = len(v6_data.get('function_details', {}).get('script_functions', {}))
    print(f"V6 (分類結果)".ljust(25), "N/A".ljust(10), str(v6_flows).ljust(10), str(v6_funcs).ljust(10), str(v6_scripts).ljust(10))
    
    # V8 統計  
    v8_flows = len(v8_data.get('flows', []))
    v8_funcs = len(v8_data.get('function_details', {}).get('function_map', {}))
    v8_scripts = len(v8_data.get('function_details', {}).get('script_functions', {}))
    print(f"V8 (分類結果)".ljust(25), "N/A".ljust(10), str(v8_flows).ljust(10), str(v8_funcs).ljust(10), str(v8_scripts).ljust(10))
    
    # 數據來源檢查
    print("\n4. 數據來源和版本檢查:")
    print(f"   V3 目標路徑: {v3_data.get('target', 'N/A')}")
    if 'metadata' in v6_data:
        print(f"   V6 輸入來源: {v6_data['metadata'].get('input_source', 'N/A')}")
    if 'metadata' in v8_data:
        print(f"   V8 輸入來源: {v8_data['metadata'].get('input_source', 'N/A')}")
    
    # Flow內容一致性檢查
    print("\n5. Flow內容一致性檢查:")
    
    # 比對V6和V8的flow數量是否一致（它們應該基於相同的分析結果）
    if v6_flows == v8_flows:
        print(f"   ✅ V6和V8 flow數量一致: {v6_flows}")
    else:
        print(f"   ⚠️  V6和V8 flow數量不一致: V6={v6_flows}, V8={v8_flows}")
    
    # 檢查函數映射是否一致
    if v6_funcs == v8_funcs and v6_scripts == v8_scripts:
        print(f"   ✅ V6和V8函數映射一致: {v6_funcs}函數, {v6_scripts}腳本")
    else:
        print(f"   ⚠️  V6和V8函數映射不一致:")
        print(f"      V6: {v6_funcs}函數, {v6_scripts}腳本")
        print(f"      V8: {v8_funcs}函數, {v8_scripts}腳本")
    
    # 檢查V3與V6/V8的差異
    print(f"\n6. V3與分類結果差異:")
    flow_diff_v6 = abs(v3_flows - v6_flows)
    flow_diff_v8 = abs(v3_flows - v8_flows)
    print(f"   V3 vs V6 flow差異: {flow_diff_v6} ({v3_flows} vs {v6_flows})")
    print(f"   V3 vs V8 flow差異: {flow_diff_v8} ({v3_flows} vs {v8_flows})")
    
    if flow_diff_v6 == 0 and flow_diff_v8 == 0:
        print("   ✅ 所有版本flow數量完全一致")
    elif max(flow_diff_v6, flow_diff_v8) <= 50:
        print("   ✅ Flow數量差異在可接受範圍內")
    else:
        print("   ⚠️  Flow數量差異較大，需要進一步檢查")
    
    # 模組分類結果比對
    print("\n7. 模組分類結果比對:")
    
    def get_module_distribution(data):
        module_counts = {}
        for flow in data.get('flows', []):
            module = flow.get('primary_module', 'unknown')
            module_counts[module] = module_counts.get(module, 0) + 1
        return module_counts
    
    v6_modules = get_module_distribution(v6_data)
    v8_modules = get_module_distribution(v8_data)
    
    print("   V6 (多數決 → 終點):      V8 (終點分類):")
    all_modules = set(v6_modules.keys()) | set(v8_modules.keys())
    for module in sorted(all_modules):
        v6_count = v6_modules.get(module, 0)
        v8_count = v8_modules.get(module, 0)
        diff = v8_count - v6_count
        print(f"   {module}: {v6_count} → {v8_count} ({diff:+d})")
    
    # 結論
    print("\n8. 結論:")
    print("   - 數據完整性: ✅ 所有版本數據載入成功")
    print(f"   - Flow一致性: {'✅' if v6_flows == v8_flows else '⚠️'}")
    print(f"   - 函數映射: {'✅' if v6_funcs == v8_funcs else '⚠️'}")
    print(f"   - 版本追踪: {'✅' if flow_diff_v8 <= 50 else '⚠️'}")

if __name__ == '__main__':
    compare_three_versions()