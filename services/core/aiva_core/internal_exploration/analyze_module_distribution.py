#!/usr/bin/env python3
"""
分析internal_exploration和core_capabilities模組的具體情況
"""
import json
import sys
from collections import Counter, defaultdict

def analyze_module_distribution(classification_file):
    """分析模組分布的詳細情況"""
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flows = data.get('flows', [])
    
    # 統計各種分布
    all_modules = Counter()
    primary_modules = Counter() 
    module_combinations = defaultdict(int)
    flow_analysis = []
    
    print(f"=== 模組分布分析 ===")
    print(f"總數據流數量: {len(flows)}")
    print()
    
    for flow in flows:
        modules = flow.get('modules', [])
        primary_module = flow.get('primary_module', 'unknown')
        classifications = flow.get('classifications', [])
        
        # 統計所有出現的模組
        for module in modules:
            all_modules[module] += 1
        
        # 統計主要模組
        primary_modules[primary_module] += 1
        
        # 統計模組組合
        unique_modules = list(set(modules))
        combo_key = ' + '.join(sorted(unique_modules))
        module_combinations[combo_key] += 1
        
        # 分析包含特定模組的流
        if 'core_capabilities' in modules or 'internal_exploration' in modules:
            flow_analysis.append({
                'id': flow.get('id'),
                'modules': modules,
                'primary_module': primary_module,
                'scripts': [c.get('script') for c in classifications],
                'length': flow.get('length')
            })
    
    print("=== 所有模組出現次數統計 ===")
    for module, count in sorted(all_modules.items(), key=lambda x: x[1], reverse=True):
        print(f"{module}: {count} 次")
    
    print(f"\n=== 主要模組統計 (primary_module) ===")
    for module, count in sorted(primary_modules.items(), key=lambda x: x[1], reverse=True):
        print(f"{module}: {count} 個流")
    
    print(f"\n=== 包含core_capabilities或internal_exploration的流 ===")
    print(f"發現 {len(flow_analysis)} 個相關流:")
    
    for i, flow in enumerate(flow_analysis[:10], 1):  # 顯示前10個
        print(f"\nFlow {flow['id']} (長度: {flow['length']}):")
        print(f"  模組序列: {' → '.join(flow['modules'])}")
        print(f"  主要模組: {flow['primary_module']}")
        print(f"  腳本序列: {' → '.join(flow['scripts'])}")
        
        # 分析為什麼不是主要模組
        module_counts = Counter(flow['modules'])
        print(f"  模組計數: {dict(module_counts)}")
        
        max_count = max(module_counts.values())
        dominant_modules = [k for k, v in module_counts.items() if v == max_count]
        
        if len(dominant_modules) > 1:
            print(f"  ⚠️ 平手情況: {dominant_modules} 都出現 {max_count} 次")
        elif 'core_capabilities' in module_counts and module_counts['core_capabilities'] < max_count:
            print(f"  📊 core_capabilities ({module_counts['core_capabilities']}次) < {dominant_modules[0]} ({max_count}次)")
        elif 'internal_exploration' in module_counts and module_counts['internal_exploration'] < max_count:
            print(f"  📊 internal_exploration ({module_counts['internal_exploration']}次) < {dominant_modules[0]} ({max_count}次)")
    
    print(f"\n=== 模組組合統計 (前15名) ===")
    for combo, count in sorted(module_combinations.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{combo}: {count} 個流")
    
    # 特別分析
    print(f"\n=== 特別分析 ===")
    
    core_cap_flows = [f for f in flow_analysis if 'core_capabilities' in f['modules']]
    internal_exp_flows = [f for f in flow_analysis if 'internal_exploration' in f['modules']]
    
    print(f"包含 core_capabilities 的流: {len(core_cap_flows)} 個")
    print(f"包含 internal_exploration 的流: {len(internal_exp_flows)} 個")
    
    if core_cap_flows:
        scripts_in_core_cap = set()
        for flow in core_cap_flows:
            for i, module in enumerate(flow['modules']):
                if module == 'core_capabilities':
                    scripts_in_core_cap.add(flow['scripts'][i])
        
        print(f"\ncore_capabilities 模組中的腳本:")
        for script in sorted(scripts_in_core_cap):
            print(f"  - {script}")
    
    if internal_exp_flows:
        scripts_in_internal_exp = set()
        for flow in internal_exp_flows:
            for i, module in enumerate(flow['modules']):
                if module == 'internal_exploration':
                    scripts_in_internal_exp.add(flow['scripts'][i])
        
        print(f"\ninternal_exploration 模組中的腳本:")
        for script in sorted(scripts_in_internal_exp):
            print(f"  - {script}")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python script.py <classification_data.json>")
        sys.exit(1)
    
    classification_file = sys.argv[1]
    analyze_module_distribution(classification_file)

if __name__ == "__main__":
    main()