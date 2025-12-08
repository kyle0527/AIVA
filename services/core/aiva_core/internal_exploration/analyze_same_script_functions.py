#!/usr/bin/env python3
"""
分析數據流中同一腳本內函數調用的情況
檢查是否存在三個以上函數在同一個腳本內的情況
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

def analyze_same_script_functions(analysis_file):
    """分析同一腳本內函數調用情況"""
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flow_chains = data.get('flow_chains', [])
    function_details = data.get('function_details', {})
    function_map = function_details.get('function_map', {})
    script_functions = function_details.get('script_functions', {})
    
    print(f"=== 數據流中同一腳本函數調用分析 ===")
    print(f"總數據流數量: {len(flow_chains)}")
    print(f"總函數數量: {len(function_map)}")
    print(f"總腳本數量: {len(script_functions)}")
    print()
    
    # 分析每條數據流
    same_script_cases = []
    script_function_stats = Counter()
    
    for flow_idx, flow in enumerate(flow_chains):
        if len(flow) < 2:
            continue
            
        # 對每條數據流，檢查是否有重複的腳本
        script_files = [Path(script_path).stem for script_path in flow]
        script_counts = Counter(script_files)
        
        # 找出出現多次的腳本
        repeated_scripts = {script: count for script, count in script_counts.items() if count > 1}
        
        if repeated_scripts:
            same_script_cases.append({
                'flow_index': flow_idx + 1,
                'flow_length': len(flow),
                'flow_path': [Path(p).name for p in flow],
                'repeated_scripts': repeated_scripts,
                'full_flow': flow
            })
    
    # 分析每個腳本包含的函數數量
    for script_name, script_info in script_functions.items():
        functions = script_info.get('functions', {})
        function_count = len(functions)
        script_function_stats[function_count] += 1
        
        if function_count >= 10:  # 顯示包含10個以上函數的腳本
            print(f"📁 {script_name}: {function_count} 個函數")
            func_names = list(functions.keys())[:5]  # 顯示前5個函數名
            if len(functions) > 5:
                func_names.append(f"...+{len(functions)-5}個")
            print(f"   函數: {', '.join(func_names)}")
            print()
    
    print(f"=== 腳本函數數量統計 ===")
    for count in sorted(script_function_stats.keys()):
        print(f"{count} 個函數的腳本: {script_function_stats[count]} 個")
    
    print(f"\n=== 同一腳本內重複調用分析 ===")
    if same_script_cases:
        print(f"發現 {len(same_script_cases)} 條數據流存在同一腳本重複調用:")
        for case in same_script_cases:
            print(f"\nFlow {case['flow_index']} (長度: {case['flow_length']}):")
            print(f"  路徑: {' → '.join(case['flow_path'])}")
            print(f"  重複腳本: {case['repeated_scripts']}")
            
            # 分析具體的函數調用
            for script_name, count in case['repeated_scripts'].items():
                if script_name in script_functions:
                    functions = script_functions[script_name].get('functions', {})
                    print(f"  📝 {script_name} (重複 {count} 次, 包含 {len(functions)} 個函數):")
                    
                    # 檢查這些函數是否可能在同一條數據流中被多次調用
                    if len(functions) >= 3:
                        print(f"     ⚠️ 該腳本包含 {len(functions)} 個函數，可能存在多函數調用情況")
                        func_list = list(functions.keys())[:10]
                        print(f"     函數列表: {', '.join(func_list)}")
                        if len(functions) > 10:
                            print(f"     (還有 {len(functions)-10} 個函數...)")
    else:
        print("✅ 未發現同一腳本在單一數據流中重複調用的情況")
    
    # 理論分析：單個腳本內可能的函數調用情況
    print(f"\n=== 理論分析：單腳本多函數調用可能性 ===")
    high_function_scripts = []
    for script_name, script_info in script_functions.items():
        functions = script_info.get('functions', {})
        function_count = len(functions)
        
        if function_count >= 3:
            high_function_scripts.append({
                'script': script_name,
                'function_count': function_count,
                'functions': list(functions.keys())
            })
    
    high_function_scripts.sort(key=lambda x: x['function_count'], reverse=True)
    
    print(f"包含3個以上函數的腳本數量: {len(high_function_scripts)}")
    print(f"這些腳本理論上可能在內部進行多函數調用:")
    
    for script in high_function_scripts[:10]:  # 顯示前10個
        print(f"  📁 {script['script']}: {script['function_count']} 個函數")
        entry_points = []
        regular_funcs = []
        
        # 分析函數類型
        script_info = script_functions.get(script['script'], {})
        functions_detail = script_info.get('functions', {})
        
        for func_name, func_info in functions_detail.items():
            if func_info.get('is_entry_point', False):
                entry_points.append(func_name)
            else:
                regular_funcs.append(func_name)
        
        if entry_points:
            print(f"     入口點: {', '.join(entry_points[:3])}{' ...' if len(entry_points) > 3 else ''}")
        if regular_funcs:
            print(f"     內部函數: {', '.join(regular_funcs[:3])}{' ...' if len(regular_funcs) > 3 else ''}")
    
    return {
        'total_flows': len(flow_chains),
        'same_script_cases': len(same_script_cases),
        'high_function_scripts': len(high_function_scripts),
        'max_functions_per_script': max(len(info.get('functions', {})) for info in script_functions.values()) if script_functions else 0
    }

def main():
    if len(sys.argv) != 2:
        print("使用方法: python script.py <analysis_results.json>")
        sys.exit(1)
    
    analysis_file = sys.argv[1]
    if not Path(analysis_file).exists():
        print(f"錯誤: 找不到文件 {analysis_file}")
        sys.exit(1)
    
    results = analyze_same_script_functions(analysis_file)
    
    print(f"\n=== 分析總結 ===")
    print(f"總數據流: {results['total_flows']}")
    print(f"同腳本重複調用的流: {results['same_script_cases']}")
    print(f"包含3+函數的腳本: {results['high_function_scripts']}")
    print(f"單腳本最大函數數量: {results['max_functions_per_script']}")
    
    if results['same_script_cases'] > 0:
        print(f"\n⚠️  發現 {results['same_script_cases']} 條數據流存在同一腳本重複調用情況")
        print(f"這表明可能存在三個以上函數在同一個腳本內被調用的情況")
    else:
        print(f"\n✅ 當前數據流分析顯示沒有同一腳本的重複調用")
        print(f"但這並不排除單個腳本內部函數間的調用關係")
    
    if results['max_functions_per_script'] >= 10:
        print(f"\n💡 發現最大單腳本包含 {results['max_functions_per_script']} 個函數")
        print(f"這些大型腳本內部很可能存在多函數調用關係")

if __name__ == "__main__":
    main()