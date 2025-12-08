import json

def analyze_version_discrepancies():
    print("=== 版本差異深度分析 ===")
    
    # 載入V6和V8數據進行詳細比較
    with open('aiva_core_classification_v6_endpoint/classification_data.json', 'r', encoding='utf-8') as f:
        v6_data = json.load(f)
    
    with open('classification_data_v8_endpoint.json/classification_data.json', 'r', encoding='utf-8') as f:
        v8_data = json.load(f)
    
    print(f"V6 flows: {len(v6_data.get('flows', []))}")
    print(f"V8 flows: {len(v8_data.get('flows', []))}")
    
    # 檢查V6和V8數據來源
    print("\n=== 檢查數據來源 ===")
    
    # V6的metadata
    v6_meta = v6_data.get('metadata', {})
    print("V6 Metadata Keys:", list(v6_meta.keys()))
    
    # V8的metadata 
    v8_meta = v8_data.get('metadata', {})
    print("V8 Metadata Keys:", list(v8_meta.keys()))
    
    # 檢查function_details是否真的為空
    print("\n=== 函數詳情檢查 ===")
    v6_fd = v6_data.get('function_details', {})
    v8_fd = v8_data.get('function_details', {})
    
    print(f"V6 function_details type: {type(v6_fd)}")
    print(f"V6 function_details keys: {list(v6_fd.keys()) if isinstance(v6_fd, dict) else 'N/A'}")
    
    print(f"V8 function_details type: {type(v8_fd)}")
    print(f"V8 function_details keys: {list(v8_fd.keys()) if isinstance(v8_fd, dict) else 'N/A'}")
    
    # 檢查V6和V8是否使用相同的輸入
    print("\n=== 輸入來源推測 ===")
    
    # 檢查第一個flow的詳細信息
    if v6_data.get('flows') and v8_data.get('flows'):
        v6_first = v6_data['flows'][0]
        v8_first = v8_data['flows'][0]
        
        print("V6第一個flow:")
        print(f"  ID: {v6_first.get('id')}")
        print(f"  Path: {v6_first.get('path', [])}")
        print(f"  Start: {v6_first.get('start')} → End: {v6_first.get('end')}")
        
        print("V8第一個flow:")
        print(f"  ID: {v8_first.get('id')}")
        print(f"  Path: {v8_first.get('path', [])}")
        print(f"  Start: {v8_first.get('start')} → End: {v8_first.get('end')}")
    
    # 檢查總共有多少不同的flow
    print("\n=== Flow內容對比 ===")
    v6_flows_set = set()
    v8_flows_set = set()
    
    for flow in v6_data.get('flows', []):
        path_str = ' → '.join(flow.get('path', []))
        v6_flows_set.add(path_str)
    
    for flow in v8_data.get('flows', []):
        path_str = ' → '.join(flow.get('path', []))
        v8_flows_set.add(path_str)
    
    print(f"V6 唯一flow paths: {len(v6_flows_set)}")
    print(f"V8 唯一flow paths: {len(v8_flows_set)}")
    
    # 找出差異的flows
    v6_only = v6_flows_set - v8_flows_set
    v8_only = v8_flows_set - v6_flows_set
    common = v6_flows_set & v8_flows_set
    
    print(f"共同flows: {len(common)}")
    print(f"僅V6有的flows: {len(v6_only)}")
    print(f"僅V8有的flows: {len(v8_only)}")
    
    if v6_only:
        print("\n僅V6有的flows (前5個):")
        for i, flow in enumerate(list(v6_only)[:5]):
            print(f"  {i+1}. {flow}")
    
    if v8_only:
        print("\n僅V8有的flows (前5個):")
        for i, flow in enumerate(list(v8_only)[:5]):
            print(f"  {i+1}. {flow}")

if __name__ == '__main__':
    analyze_version_discrepancies()