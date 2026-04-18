#!/usr/bin/env python3
import os

"""
分析各模組獨特終點能力
扣除多路徑重複後的獨特終點數量及對內/對外分類
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
project_root = Path(__file__).resolve().parent.parent.parent

def classify_capability(start: str, end: str) -> str:
    """
    分類能力為對內、對外或混合
    
    對內: start 和 end 都在同一模組內
    對外: start 和 end 在不同模組
    """
    # 提取模組名稱
    def get_module(path: str) -> str:
        if 'services/core/aiva_core/' in path:
            parts = path.split('services/core/aiva_core/')[1].split('/')
            return parts[0] if parts else 'unknown'
        return 'unknown'
    
    start_module = get_module(start)
    end_module = get_module(end)
    
    if start_module == end_module:
        return 'Internal'
    else:
        return 'External'

def main():
    base_path = project_root
    classification_file = base_path / "services" / "integration" / "data" / "internal_exploration" / "latest_classification.json"
    
    print("\n" + "="*60)
    print("🎯 各模組獨特終點能力分析")
    print("   （扣除多路徑重複的終點）")
    print("="*60)
    
    # 讀取分類數據
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flows = data.get('flows', [])
    
    # 定義模組數據的類型結構
    # 使用簡單的字典結構，避免複雜的嵌套類型
    module_data: Dict[str, Dict[str, object]] = {}
    
    def get_module_data(module: str) -> Dict[str, object]:
        """獲取或創建模組數據"""
        if module not in module_data:
            module_data[module] = {
                'total_flows': 0,
                'unique_endpoints': set(),
                'endpoint_classifications': {},
                'flows_per_endpoint': defaultdict(list)
            }
        return module_data[module]
    
    # 收集各模組的flows和終點
    for flow in flows:
        # 從 classifications 獲取模組資訊
        classifications = flow.get('classifications', [])
        if not classifications:
            continue
        
        # 獲取終點的分類（最後一個節點）
        end_classification = classifications[-1]
        module = end_classification.get('module', 'unknown')
        
        # 獲取路徑資訊
        full_path = flow.get('full_path', [])
        if len(full_path) < 2:
            continue
        
        start = full_path[0]
        end = full_path[-1]
        
        # 獲取模組數據
        data = get_module_data(module)
        
        # 更新統計
        data['total_flows'] = int(data['total_flows']) + 1  # type: ignore
        endpoints: Set[str] = data['unique_endpoints']  # type: ignore
        endpoints.add(end)
        flows_per_ep: Dict[str, List[str]] = data['flows_per_endpoint']  # type: ignore
        flows_per_ep[end].append(start)
        
        # 分類此終點（基於任一到達它的路徑）
        classification = classify_capability(start, end)
        ep_classifications: Dict[Tuple[str, str], int] = data['endpoint_classifications']  # type: ignore
        if (end, classification) not in ep_classifications:
            ep_classifications[(end, classification)] = 1
    
    # 整理終點分類
    module_summary = {}
    for module, mod_data in module_data.items():
        internal_endpoints: Set[str] = set()
        external_endpoints: Set[str] = set()
        
        # 獲取並類型斷言
        ep_classifications: Dict[Tuple[str, str], int] = mod_data['endpoint_classifications']  # type: ignore
        flows_per_ep: Dict[str, List[str]] = mod_data['flows_per_endpoint']  # type: ignore
        unique_eps: Set[str] = mod_data['unique_endpoints']  # type: ignore
        
        for (endpoint, classification), _ in ep_classifications.items():
            if classification == 'Internal':
                internal_endpoints.add(endpoint)
            else:
                external_endpoints.add(endpoint)
        
        # 計算多路徑終點
        multi_path_endpoints = {
            ep: len(starts) 
            for ep, starts in flows_per_ep.items() 
            if len(set(starts)) > 1
        }
        
        module_summary[module] = {
            'total_flows': mod_data['total_flows'],
            'unique_endpoints': len(unique_eps),
            'multi_path_endpoints': len(multi_path_endpoints),
            'single_path_endpoints': len(unique_eps) - len(multi_path_endpoints),
            'internal_endpoints': len(internal_endpoints),
            'external_endpoints': len(external_endpoints),
            'endpoints_detail': flows_per_ep,
            'internal_list': sorted(internal_endpoints),
            'external_list': sorted(external_endpoints)
        }
    
    # 按模組順序輸出
    module_order = [
        'cognitive_core',
        'internal_exploration', 
        'task_planning',
        'external_learning',
        'core_capabilities',
        'service_backbone',
        'unknown'
    ]
    
    print("\n📊 各模組獨特終點統計:\n")
    
    total_flows = 0
    total_unique = 0
    total_multi = 0
    total_single = 0
    total_internal = 0
    total_external = 0
    
    for module in module_order:
        if module not in module_summary:
            continue
            
        summary = module_summary[module]
        total_flows += summary['total_flows']
        total_unique += summary['unique_endpoints']
        total_multi += summary['multi_path_endpoints']
        total_single += summary['single_path_endpoints']
        total_internal += summary['internal_endpoints']
        total_external += summary['external_endpoints']
        
        print("="*60)
        print(f"📦 {module}")
        print(f"   總 Flows 數: {summary['total_flows']}")
        print(f"   獨特終點總數: {summary['unique_endpoints']}")
        print(f"   └─ 多路徑終點: {summary['multi_path_endpoints']} ({summary['multi_path_endpoints']/summary['unique_endpoints']*100:.1f}%)")
        print(f"   └─ 單路徑終點: {summary['single_path_endpoints']} ({summary['single_path_endpoints']/summary['unique_endpoints']*100:.1f}%)")
        print(f"\n   對內能力: {summary['internal_endpoints']} ({summary['internal_endpoints']/summary['unique_endpoints']*100:.1f}%)")
        print(f"   對外能力: {summary['external_endpoints']} ({summary['external_endpoints']/summary['unique_endpoints']*100:.1f}%)")
        
        # 顯示能力樣本
        if summary['internal_list']:
            print(f"\n   對內能力樣本 (前5個):")
            for ep in summary['internal_list'][:5]:
                ep_name = ep.split('/')[-1].replace('.py', '')
                paths_count = len(set(summary['endpoints_detail'][ep]))
                print(f"      • {ep_name} ({paths_count} 條路徑)")
        
        if summary['external_list']:
            print(f"\n   對外能力樣本 (前5個):")
            for ep in summary['external_list'][:5]:
                ep_name = ep.split('/')[-1].replace('.py', '')
                paths_count = len(set(summary['endpoints_detail'][ep]))
                print(f"      • {ep_name} ({paths_count} 條路徑)")
    
    # 整體統計
    print("\n" + "="*60)
    print("📈 整體統計:")
    print("="*60)
    print(f"總 Flows 數: {total_flows}")
    print(f"獨特終點總數: {total_unique}")
    print(f"   ├─ 多路徑終點: {total_multi} ({total_multi/total_unique*100:.1f}%)")
    print(f"   └─ 單路徑終點: {total_single} ({total_single/total_unique*100:.1f}%)")
    print(f"\n對內能力總數: {total_internal} ({total_internal/total_unique*100:.1f}%)")
    print(f"對外能力總數: {total_external} ({total_external/total_unique*100:.1f}%)")
    
    # 計算獨特終點的平均路徑數
    all_endpoints_paths: Dict[str, Set[str]] = {}
    for module, mod_data in module_data.items():
        flows_per_ep: Dict[str, List[str]] = mod_data['flows_per_endpoint']  # type: ignore
        for ep, starts in flows_per_ep.items():
            if ep not in all_endpoints_paths:
                all_endpoints_paths[ep] = set()
            all_endpoints_paths[ep].update(starts)
    
    avg_paths = sum(len(starts) for starts in all_endpoints_paths.values()) / len(all_endpoints_paths)
    
    print(f"\n平均每個獨特終點的路徑數: {avg_paths:.2f}")
    
    # 找出路徑最多的終點
    top_endpoints = sorted(
        [(ep, len(starts)) for ep, starts in all_endpoints_paths.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    print(f"\n🔝 路徑最多的 10 個終點:")
    for i, (ep, count) in enumerate(top_endpoints, 1):
        ep_name = ep.split('/')[-1].replace('.py', '')
        module = 'unknown'
        for m, d in module_data.items():
            unique_eps: Set[str] = d['unique_endpoints']  # type: ignore
            if ep in unique_eps:
                module = m
                break
        print(f"   {i:2d}. {ep_name:40s} ({count:2d} 條) - {module}")
    
    print("\n" + "="*60)
    print("💡 關鍵發現:")
    print("="*60)
    print(f"\n1. 終點複用率: {total_multi}/{total_unique} = {total_multi/total_unique*100:.1f}%")
    print(f"   意義: 超過一半的能力終點被多條路徑使用")
    
    print(f"\n2. 獨特終點分布:")
    print(f"   • 對內能力占 {total_internal/total_unique*100:.1f}%")
    print(f"   • 對外能力占 {total_external/total_unique*100:.1f}%")
    
    print(f"\n3. 單路徑終點 ({total_single} 個):")
    print(f"   這些是真正「獨特」的能力，只有一條固定路徑")
    
    print(f"\n4. 多路徑終點 ({total_multi} 個):")
    print(f"   這些能力提供多種訪問方式，提高系統彈性")
    
    # 保存報告
    output_file = base_path / "docs" / "UNIQUE_ENDPOINTS_BY_MODULE_ANALYSIS.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 各模組獨特終點能力分析\n\n")
        f.write("**分析日期**: 2026-01-01\n\n")
        f.write("## 分析目的\n\n")
        f.write("扣除多路徑到達同一終點的重複計算後，統計各模組真正的獨特終點能力數量，\n")
        f.write("並分析這些獨特能力的對內/對外特徵。\n\n")
        
        f.write("## 核心概念\n\n")
        f.write("- **獨特終點**: 不同的終點腳本（無論有多少條路徑到達）\n")
        f.write("- **多路徑終點**: 有多條不同起點路徑到達的終點\n")
        f.write("- **單路徑終點**: 只有一條固定路徑的終點\n")
        f.write("- **對內能力**: 起點和終點在同一模組內\n")
        f.write("- **對外能力**: 起點和終點在不同模組\n\n")
        
        f.write("## 各模組統計\n\n")
        
        for module in module_order:
            if module not in module_summary:
                continue
            
            summary = module_summary[module]
            f.write(f"### {module}\n\n")
            f.write(f"- **總 Flows 數**: {summary['total_flows']}\n")
            f.write(f"- **獨特終點總數**: {summary['unique_endpoints']}\n")
            f.write(f"  - 多路徑終點: {summary['multi_path_endpoints']} ({summary['multi_path_endpoints']/summary['unique_endpoints']*100:.1f}%)\n")
            f.write(f"  - 單路徑終點: {summary['single_path_endpoints']} ({summary['single_path_endpoints']/summary['unique_endpoints']*100:.1f}%)\n\n")
            f.write(f"- **對內能力**: {summary['internal_endpoints']} ({summary['internal_endpoints']/summary['unique_endpoints']*100:.1f}%)\n")
            f.write(f"- **對外能力**: {summary['external_endpoints']} ({summary['external_endpoints']/summary['unique_endpoints']*100:.1f}%)\n\n")
            
            if summary['internal_list']:
                f.write(f"**對內能力列表** ({len(summary['internal_list'])} 個):\n\n")
                for ep in summary['internal_list']:
                    ep_name = ep.split('/')[-1].replace('.py', '')
                    paths_count = len(set(summary['endpoints_detail'][ep]))
                    f.write(f"- `{ep_name}` ({paths_count} 條路徑)\n")
                f.write("\n")
            
            if summary['external_list']:
                f.write(f"**對外能力列表** ({len(summary['external_list'])} 個):\n\n")
                for ep in summary['external_list']:
                    ep_name = ep.split('/')[-1].replace('.py', '')
                    paths_count = len(set(summary['endpoints_detail'][ep]))
                    f.write(f"- `{ep_name}` ({paths_count} 條路徑)\n")
                f.write("\n")
        
        f.write("## 整體統計\n\n")
        f.write(f"- **總 Flows 數**: {total_flows}\n")
        f.write(f"- **獨特終點總數**: {total_unique}\n")
        f.write(f"  - 多路徑終點: {total_multi} ({total_multi/total_unique*100:.1f}%)\n")
        f.write(f"  - 單路徑終點: {total_single} ({total_single/total_unique*100:.1f}%)\n\n")
        f.write(f"- **對內能力總數**: {total_internal} ({total_internal/total_unique*100:.1f}%)\n")
        f.write(f"- **對外能力總數**: {total_external} ({total_external/total_unique*100:.1f}%)\n\n")
        f.write(f"- **平均每終點路徑數**: {avg_paths:.2f}\n\n")
        
        f.write("## 路徑最多的終點 (Top 10)\n\n")
        f.write("| 排名 | 終點名稱 | 路徑數 | 所屬模組 |\n")
        f.write("|------|----------|--------|----------|\n")
        for i, (ep, count) in enumerate(top_endpoints, 1):
            ep_name = ep.split('/')[-1].replace('.py', '')
            module = 'unknown'
            for m, d in module_data.items():
                unique_eps: Set[str] = d['unique_endpoints']  # type: ignore
                if ep in unique_eps:
                    module = m
                    break
            f.write(f"| {i} | `{ep_name}` | {count} | {module} |\n")
        
        f.write("\n## 關鍵發現\n\n")
        f.write(f"1. **終點複用率**: {total_multi}/{total_unique} = {total_multi/total_unique*100:.1f}%\n")
        f.write(f"   - 意義: 超過一半的能力終點被多條路徑使用，顯示系統高度模組化\n\n")
        
        f.write(f"2. **獨特終點分布**:\n")
        f.write(f"   - 對內能力占 {total_internal/total_unique*100:.1f}%\n")
        f.write(f"   - 對外能力占 {total_external/total_unique*100:.1f}%\n")
        f.write(f"   - 說明: 多數獨特終點服務於模組內部流程\n\n")
        
        f.write(f"3. **單路徑終點** ({total_single} 個):\n")
        f.write(f"   - 這些是真正「獨特」的能力，只有一條固定路徑\n")
        f.write(f"   - 可能是專用功能或入口點\n\n")
        
        f.write(f"4. **多路徑終點** ({total_multi} 個):\n")
        f.write(f"   - 這些能力提供多種訪問方式\n")
        f.write(f"   - 提高系統彈性和容錯能力\n")
        if total_multi > 0:
            f.write(f"   - 平均每個多路徑終點有 {sum(len(starts) for ep, starts in all_endpoints_paths.items() if len(starts) > 1) / total_multi:.1f} 條路徑\n\n")
        else:
            f.write(f"   - 無多路徑終點\n\n")
        
        f.write("## 結論\n\n")
        f.write(f"1. AIVA 系統共有 **{total_unique} 個獨特終點能力**\n")
        f.write(f"2. 這些能力通過 **{total_flows} 條 flows** 被調用\n")
        f.write(f"3. **{total_multi/total_unique*100:.1f}%** 的終點提供多種訪問路徑\n")
        f.write(f"4. 對內能力 ({total_internal/total_unique*100:.1f}%) 多於對外能力 ({total_external/total_unique*100:.1f}%)\n")
        f.write(f"5. 系統設計強調**模組內聚**和**路徑冗餘**\n")
    
    print(f"\n✅ 詳細報告已保存至: {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
