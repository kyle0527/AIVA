#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析外部模組的流程結構
回答問題：
1. 每個流程是否代表一種攻擊方式？
2. 是否有相同起終點但路徑不同的流程？
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_module_flows(module_path: Path):
    """分析單個模組的流程"""
    analysis_file = module_path / "analysis_results.json"
    
    if not analysis_file.exists():
        return None
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flow_chains = data.get('flow_chains', [])
    
    # flow_chains 是 list of lists (路徑列表)
    # 每個 flow_chain 是一個路徑: [file1, file2, file3, ...]
    
    # 分析相同起終點的流程
    endpoint_groups = defaultdict(list)
    
    for idx, flow_chain in enumerate(flow_chains):
        if isinstance(flow_chain, list) and len(flow_chain) > 0:
            start = flow_chain[0] if len(flow_chain) > 0 else ''
            end = flow_chain[-1] if len(flow_chain) > 0 else ''
            flow_id = idx + 1
            length = len(flow_chain)
            path = flow_chain
            
            # 提取文件名
            start_file = Path(start).name if start else 'unknown'
            end_file = Path(end).name if end else 'unknown'
            
            key = f"{start_file} → {end_file}"
            endpoint_groups[key].append({
                'id': flow_id,
                'length': length,
                'path': [Path(p).name for p in path] if path else []
            })
    
    # 找出相同起終點的流程組
    duplicates = {k: v for k, v in endpoint_groups.items() if len(v) > 1}
    
    return {
        'module': module_path.name,
        'total_flows': len(flow_chains),
        'unique_endpoints': len(endpoint_groups),
        'duplicate_endpoints': len(duplicates),
        'duplicates': duplicates,
        'sample_flows': flow_chains[:3] if flow_chains else []
    }

def main():
    base_path = Path("module_analysis")
    
    print("=" * 80)
    print("外部模組流程分析")
    print("=" * 80)
    
    all_results = []
    
    # 分析所有模組
    for module_dir in sorted(base_path.iterdir()):
        if module_dir.is_dir():
            result = analyze_module_flows(module_dir)
            if result:
                all_results.append(result)
    
    # 輸出結果
    print(f"\n分析了 {len(all_results)} 個模組\n")
    
    for result in all_results:
        print(f"📦 {result['module']}")
        print(f"   總流程數: {result['total_flows']}")
        print(f"   唯一起終點組合: {result['unique_endpoints']}")
        print(f"   相同起終點但路徑不同的組: {result['duplicate_endpoints']}")
        
        # 顯示相同起終點的範例
        if result['duplicates']:
            print(f"   \n   範例（相同起終點，不同路徑）:")
            for endpoint, flows in list(result['duplicates'].items())[:2]:
                print(f"   • {endpoint}")
                for flow in flows[:2]:
                    path_str = " → ".join(flow['path'][:3])
                    if len(flow['path']) > 3:
                        path_str += "..."
                    print(f"     - Flow {flow['id']}: 長度{flow['length']}, {path_str}")
        
        # 顯示流程範例
        if result['sample_flows']:
            print(f"   \n   流程範例:")
            for flow in result['sample_flows'][:1]:
                if isinstance(flow, dict):
                    start = Path(flow.get('start', '')).name
                    end = Path(flow.get('end', '')).name
                    print(f"   • Flow {flow.get('id')}: {start} → {end} (長度: {flow.get('length')})")
        
        print()
    
    # 總結
    print("=" * 80)
    print("總結")
    print("=" * 80)
    
    total_flows = sum(r['total_flows'] for r in all_results)
    total_duplicates = sum(r['duplicate_endpoints'] for r in all_results)
    
    print(f"\n✅ 總流程數: {total_flows}")
    print(f"✅ 相同起終點但路徑不同的組: {total_duplicates}")
    
    print("\n回答問題:")
    print("1. 每個流程是否代表一種攻擊方式？")
    print("   → 不完全是。一個流程代表一條代碼執行路徑（從一個函數到另一個函數）")
    print("   → 一個攻擊方式可能包含多個流程（多條路徑組合）")
    print("   → 例如：XSS檢測可能有109個不同的流程（不同的檢測路徑）")
    
    print("\n2. 是否有相同起終點但路徑不同的流程？")
    if total_duplicates > 0:
        print(f"   → 是！發現 {total_duplicates} 組相同起終點但路徑不同的流程")
        print("   → 這表示有多種方式到達同一個目標（不同的實現路徑）")
    else:
        print("   → 否。每個起終點組合都是唯一的路徑")

if __name__ == "__main__":
    main()
