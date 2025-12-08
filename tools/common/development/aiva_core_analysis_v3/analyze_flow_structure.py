import json

def analyze_real_flow_structure():
    with open('analysis_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 真實Flow結構分析 ===")
    
    # 檢查combined_flows vs flow_chains
    flow_chains = data.get('flow_chains', [])
    combined_flows = data.get('combined_flows', [])
    
    print(f"Flow Chains: {len(flow_chains)}")
    print(f"Combined Flows: {len(combined_flows)}")
    
    print("\n=== Flow Chains 前5個例子 ===")
    for i, chain in enumerate(flow_chains[:5]):
        print(f"{i+1}. {chain}")
    
    print("\n=== Combined Flows 前5個例子 ===")
    for i, flow in enumerate(combined_flows[:5]):
        print(f"{i+1}. {flow}")
    
    # 檢查函數映射結構
    print("\n=== 函數映射結構 ===")
    fd = data.get('function_details', {})
    function_map = fd.get('function_map', {})
    script_functions = fd.get('script_functions', {})
    
    print(f"Function Map 範例 (前3個):")
    for i, (func_name, func_info) in enumerate(list(function_map.items())[:3]):
        print(f"  {func_name}: {func_info}")
    
    print(f"\nScript Functions 範例 (前2個):")
    for i, (script_name, script_info) in enumerate(list(script_functions.items())[:2]):
        print(f"  {script_name}: {list(script_info.get('functions', {}).keys())[:5]}...")

if __name__ == '__main__':
    analyze_real_flow_structure()