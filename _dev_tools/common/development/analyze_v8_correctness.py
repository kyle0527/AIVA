import json
import os

def analyze_v8_correctness():
    print("=== V8 數據流正確性分析 ===")
    
    # 載入V8分類結果
    v8_path = "classification_data_v8_endpoint.json/classification_data.json"
    
    if not os.path.exists(v8_path):
        print("❌ V8分類結果文件不存在")
        return
    
    with open(v8_path, 'r', encoding='utf-8') as f:
        v8_data = json.load(f)
    
    # 載入V3原始分析數據（V8的輸入源）
    v3_path = "aiva_core_analysis_v3/analysis_results.json"
    with open(v3_path, 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    
    print("1. 基本數據一致性檢查:")
    v8_flows = v8_data.get('flows', [])
    v3_flows = v3_data.get('flow_chains', [])
    
    print(f"   V3原始flow chains: {len(v3_flows)}")
    print(f"   V8分類flows: {len(v8_flows)}")
    print(f"   數量一致: {'✅' if len(v3_flows) == len(v8_flows) else '❌'}")
    
    # 檢查函數和腳本關聯
    print("\n2. 函數和腳本關聯檢查:")
    v3_function_details = v3_data.get('function_details', {})
    function_map = v3_function_details.get('function_map', {})
    script_functions = v3_function_details.get('script_functions', {})
    
    print(f"   V3原始函數映射: {len(function_map)} 個函數")
    print(f"   V3原始腳本映射: {len(script_functions)} 個腳本")
    
    # 檢查V8中的腳本是否都在V3的腳本映射中
    v8_scripts = set()
    for flow in v8_flows:
        for script in flow.get('path', []):
            v8_scripts.add(script)
    
    v3_scripts = set(script_functions.keys())
    
    print(f"   V8中的腳本: {len(v8_scripts)} 個")
    print(f"   V3中的腳本: {len(v3_scripts)} 個")
    
    missing_scripts = v8_scripts - v3_scripts
    extra_scripts = v3_scripts - v8_scripts
    
    print(f"   V8中但V3沒有的腳本: {len(missing_scripts)}")
    if missing_scripts and len(missing_scripts) <= 5:
        for script in list(missing_scripts)[:5]:
            print(f"     - {script}")
    
    print(f"   V3中但V8沒用到的腳本: {len(extra_scripts)}")
    
    # 檢查數據流路徑的正確性
    print("\n3. 數據流路徑正確性檢查:")
    
    # 比對前5個flow的詳細信息
    print("   前5個flow路徑對比:")
    for i in range(min(5, len(v3_flows), len(v8_flows))):
        v3_chain = v3_flows[i]
        v8_flow = v8_flows[i]
        
        # 從V3的路徑提取腳本名
        v3_scripts = [path.split('\\')[-1].replace('.py', '') for path in v3_chain]
        v8_scripts = v8_flow.get('path', [])
        
        print(f"   Flow {i+1}:")
        print(f"     V3: {' → '.join(v3_scripts)}")
        print(f"     V8: {' → '.join(v8_scripts)}")
        print(f"     一致: {'✅' if v3_scripts == v8_scripts else '❌'}")
        
        if v3_scripts != v8_scripts:
            print(f"     差異詳細:")
            print(f"       V3原始路徑: {v3_chain}")
            print(f"       V8處理結果: {v8_scripts}")
    
    # 檢查分類邏輯的正確性
    print("\n4. 分類邏輯正確性檢查:")
    
    # 統計模組分佈
    module_distribution = {}
    endpoint_modules = {}
    majority_modules = {}
    
    for flow in v8_flows:
        primary = flow.get('primary_module', 'unknown')
        endpoint = flow.get('endpoint_module', 'unknown')
        majority = flow.get('majority_module', 'unknown')
        
        module_distribution[primary] = module_distribution.get(primary, 0) + 1
        endpoint_modules[endpoint] = endpoint_modules.get(endpoint, 0) + 1
        majority_modules[majority] = majority_modules.get(majority, 0) + 1
    
    print("   模組分佈檢查:")
    print(f"     Primary modules: {len(module_distribution)} 種")
    print(f"     Endpoint modules: {len(endpoint_modules)} 種") 
    print(f"     Majority modules: {len(majority_modules)} 種")
    
    print("   Primary vs Endpoint 一致性:")
    primary_endpoint_match = 0
    for flow in v8_flows:
        if flow.get('primary_module') == flow.get('endpoint_module'):
            primary_endpoint_match += 1
    
    print(f"     一致的flows: {primary_endpoint_match}/{len(v8_flows)} ({primary_endpoint_match/len(v8_flows)*100:.1f}%)")
    
    # 檢查終點分類法的實現
    print("\n5. 終點分類法實現檢查:")
    
    # 檢查是否每個flow都有正確的終點模組
    flows_with_endpoint = 0
    flows_with_valid_path = 0
    
    for flow in v8_flows:
        path = flow.get('path', [])
        endpoint_module = flow.get('endpoint_module')
        primary_module = flow.get('primary_module')
        
        if endpoint_module:
            flows_with_endpoint += 1
        
        if path:
            flows_with_valid_path += 1
            
        # 檢查primary_module是否確實來自最後一個腳本
        if path and endpoint_module == primary_module:
            # 這是正確的終點分類實現
            pass
    
    print(f"   有endpoint_module的flows: {flows_with_endpoint}/{len(v8_flows)}")
    print(f"   有有效path的flows: {flows_with_valid_path}/{len(v8_flows)}")
    
    # 檢查異常情況
    print("\n6. 異常情況檢查:")
    
    anomalies = []
    for i, flow in enumerate(v8_flows):
        flow_id = flow.get('id', i+1)
        path = flow.get('path', [])
        primary = flow.get('primary_module')
        endpoint = flow.get('endpoint_module')
        
        if not path:
            anomalies.append(f"Flow {flow_id}: 空路徑")
        elif not primary:
            anomalies.append(f"Flow {flow_id}: 缺少primary_module")
        elif not endpoint:
            anomalies.append(f"Flow {flow_id}: 缺少endpoint_module")
        elif primary != endpoint:
            anomalies.append(f"Flow {flow_id}: primary與endpoint不一致")
    
    if anomalies:
        print(f"   發現 {len(anomalies)} 個異常:")
        for anomaly in anomalies[:10]:  # 只顯示前10個
            print(f"     - {anomaly}")
    else:
        print("   ✅ 沒有發現異常")
    
    # 總結
    print("\n7. 總結:")
    data_consistency = len(v3_flows) == len(v8_flows)
    script_consistency = len(missing_scripts) == 0
    classification_valid = flows_with_endpoint == len(v8_flows)
    no_anomalies = len(anomalies) == 0
    
    print(f"   數據一致性: {'✅' if data_consistency else '❌'}")
    print(f"   腳本關聯: {'✅' if script_consistency else '⚠️'}")
    print(f"   分類完整性: {'✅' if classification_valid else '❌'}")
    print(f"   無異常: {'✅' if no_anomalies else '⚠️'}")
    
    overall_status = all([data_consistency, script_consistency, classification_valid, no_anomalies])
    print(f"   整體評估: {'✅ 正確' if overall_status else '⚠️ 需要檢查'}")

if __name__ == '__main__':
    analyze_v8_correctness()