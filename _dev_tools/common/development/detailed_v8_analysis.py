import json

def detailed_v8_analysis():
    print("=== V8 詳細分析補充 ===")
    
    # 載入數據
    with open('classification_data_v8_endpoint.json/classification_data.json', 'r', encoding='utf-8') as f:
        v8_data = json.load(f)
    
    with open('aiva_core_analysis_v3/analysis_results.json', 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    
    # 1. 檢查未使用的腳本
    print("1. 未使用腳本分析:")
    v8_scripts = set()
    for flow in v8_data.get('flows', []):
        for script in flow.get('path', []):
            v8_scripts.add(script)
    
    v3_scripts = set(v3_data.get('function_details', {}).get('script_functions', {}).keys())
    unused_scripts = v3_scripts - v8_scripts
    
    print(f"   未使用的腳本 ({len(unused_scripts)}):")
    for script in sorted(unused_scripts):
        # 檢查這個腳本的函數數量
        script_info = v3_data['function_details']['script_functions'].get(script, {})
        func_count = len(script_info.get('functions', {}))
        print(f"     - {script} ({func_count} 個函數)")
    
    # 2. 模組分類詳細分析
    print("\n2. 模組分類詳細分析:")
    
    module_flows = {}
    for flow in v8_data.get('flows', []):
        primary = flow.get('primary_module', 'unknown')
        majority = flow.get('majority_module', 'unknown')
        
        if primary not in module_flows:
            module_flows[primary] = {'flows': [], 'majority_external': 0}
        
        module_flows[primary]['flows'].append(flow)
        if majority == 'external_learning':
            module_flows[primary]['majority_external'] += 1
    
    print("   各模組詳細統計:")
    for module, data in sorted(module_flows.items(), key=lambda x: len(x[1]['flows']), reverse=True):
        flow_count = len(data['flows'])
        external_count = data['majority_external']
        external_pct = external_count / flow_count * 100 if flow_count > 0 else 0
        
        print(f"   {module}:")
        print(f"     總flows: {flow_count}")
        print(f"     多數決為external_learning: {external_count} ({external_pct:.1f}%)")
        
        # 檢查典型的flow例子
        typical_flows = data['flows'][:3]
        for i, flow in enumerate(typical_flows):
            path_str = ' → '.join(flow.get('path', []))
            print(f"     例子{i+1}: {path_str}")
    
    # 3. 終點分類法效果驗證
    print("\n3. 終點分類法效果驗證:")
    
    # 統計多數決與終點分類的差異
    classification_changes = {
        'external_to_other': 0,
        'other_to_external': 0,
        'same': 0,
        'other_changes': 0
    }
    
    for flow in v8_data.get('flows', []):
        primary = flow.get('primary_module')
        majority = flow.get('majority_module')
        
        if primary == majority:
            classification_changes['same'] += 1
        elif majority == 'external_learning' and primary != 'external_learning':
            classification_changes['external_to_other'] += 1
        elif primary == 'external_learning' and majority != 'external_learning':
            classification_changes['other_to_external'] += 1
        else:
            classification_changes['other_changes'] += 1
    
    total = len(v8_data.get('flows', []))
    print(f"   分類變化統計:")
    print(f"     保持不變: {classification_changes['same']} ({classification_changes['same']/total*100:.1f}%)")
    print(f"     external_learning → 其他模組: {classification_changes['external_to_other']} ({classification_changes['external_to_other']/total*100:.1f}%)")
    print(f"     其他模組 → external_learning: {classification_changes['other_to_external']} ({classification_changes['other_to_external']/total*100:.1f}%)")
    print(f"     其他變化: {classification_changes['other_changes']} ({classification_changes['other_changes']/total*100:.1f}%)")
    
    # 4. 檢查分類關鍵詞匹配的有效性
    print("\n4. 分類關鍵詞匹配驗證:")
    
    # 統計各模組的腳本關鍵詞分佈
    module_scripts = {}
    for flow in v8_data.get('flows', []):
        primary = flow.get('primary_module')
        path = flow.get('path', [])
        
        if primary not in module_scripts:
            module_scripts[primary] = set()
        
        for script in path:
            module_scripts[primary].add(script)
    
    print("   各模組包含的腳本類型:")
    for module, scripts in sorted(module_scripts.items()):
        print(f"   {module} ({len(scripts)} 種腳本):")
        # 顯示前10個最常見的腳本
        script_list = sorted(list(scripts))[:10]
        print(f"     {', '.join(script_list)}")
    
    # 5. 數據流長度分析
    print("\n5. 數據流長度分析:")
    
    length_distribution = {}
    for flow in v8_data.get('flows', []):
        length = flow.get('length', len(flow.get('path', [])))
        length_distribution[length] = length_distribution.get(length, 0) + 1
    
    print("   Flow長度分佈:")
    for length in sorted(length_distribution.keys()):
        count = length_distribution[length]
        pct = count / total * 100
        print(f"     {length}步: {count} flows ({pct:.1f}%)")
    
    # 6. 總結評估
    print("\n6. 總體正確性評估:")
    
    issues = []
    
    # 檢查是否有異常的分類
    if classification_changes['other_to_external'] > 0:
        issues.append(f"有{classification_changes['other_to_external']}個flows從其他模組變為external_learning")
    
    # 檢查是否有過短或過長的flows
    short_flows = sum(1 for flow in v8_data.get('flows', []) if len(flow.get('path', [])) < 2)
    if short_flows > 0:
        issues.append(f"有{short_flows}個flows長度小於2")
    
    long_flows = sum(1 for flow in v8_data.get('flows', []) if len(flow.get('path', [])) > 5)
    if long_flows > 0:
        issues.append(f"有{long_flows}個flows長度大於5")
    
    if issues:
        print("   發現的問題:")
        for issue in issues:
            print(f"     ⚠️  {issue}")
    else:
        print("   ✅ 沒有發現重大問題")
    
    print(f"   整體評估: {'✅ V8數據流分類正確且合理' if len(issues) == 0 else '⚠️ V8基本正確但有小問題'}")

if __name__ == '__main__':
    detailed_v8_analysis()