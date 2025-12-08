import json

def analyze_v3_structure():
    with open('analysis_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== V3 完整數據結構分析 ===")
    print(f"Keys: {list(data.keys())}")
    print(f"Target: {data.get('target')}")
    print(f"Files processed: {data.get('total_files_processed')}")
    print(f"Flow chains: {len(data.get('flow_chains', []))}")
    
    # 檢查function_details結構
    fd = data.get('function_details', {})
    print(f"\nFunction Details Type: {type(fd)}")
    if isinstance(fd, dict):
        print(f"Function Details Keys: {list(fd.keys())}")
        for key, value in fd.items():
            print(f"  {key}: {type(value)} - {len(value) if isinstance(value, (list, dict)) else value}")
    else:
        print(f"Function Details Content: {fd}")
    
    # 檢查第一個flow chain的詳細結構
    if data.get('flow_chains'):
        print(f"\n第一個Flow Chain: {data['flow_chains'][0]}")
        
    # 檢查summary
    if 'summary' in data:
        summary = data['summary']
        print(f"\nSummary Keys: {list(summary.keys())}")
        
    # 檢查combined_flows
    if 'combined_flows' in data:
        print(f"Combined Flows: {len(data['combined_flows'])}")

if __name__ == '__main__':
    analyze_v3_structure()