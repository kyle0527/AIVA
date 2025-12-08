import json

def analyze_real_flow_structure():
    with open('analysis_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== 真實Flow結構分析 ===")
    
    # 檢查combined_flows vs flow_chains
    flow_chains = data.get('flow_chains', [])
    combined_flows = data.get('combined_flows', {})
    
    print(f"Flow Chains: {len(flow_chains)}")
    print(f"Combined Flows Type: {type(combined_flows)}")
    
    print("\n=== Flow Chains 前5個例子 ===")
    for i, chain in enumerate(flow_chains[:5]):
        scripts = [path.split('\\')[-1].replace('.py', '') for path in chain]
        print(f"{i+1}. {' → '.join(scripts)}")
        print(f"   Full: {chain}")
    
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
        functions = list(script_info.get('functions', {}).keys())
        print(f"  {script_name}: {functions[:5]}...")
        
    # 檢查真實的函數到函數調用
    print("\n=== 檢查流程是否包含函數信息 ===")
    print("Flow chains 是否只是腳本路徑:", isinstance(flow_chains[0][0], str))
    
    # 檢查是否有實際的函數調用流
    print(f"第一個flow chain第一個元素類型: {type(flow_chains[0][0])}")

if __name__ == '__main__':
    analyze_real_flow_structure()