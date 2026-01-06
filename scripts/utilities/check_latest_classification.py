#!/usr/bin/env python3
"""
檢查最新分類數據
分析 latest_classification.json 的完整狀態
"""

import json
from pathlib import Path
from collections import Counter

# 數據路徑
data_dir = Path("C:/Users/User/Downloads/data/internal_exploration")
json_file = data_dir / "latest_classification.json"

# 讀取數據
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("\n" + "="*60)
print("📊 最新分類數據檢查報告")
print("="*60)

# 基本信息
metadata = data.get('metadata', {})
print(f"\n生成時間: {metadata.get('generated_at', 'N/A')}")
print(f"總流程數: {metadata.get('total_flows', 'N/A')}")

# 模組分佈
flows = data.get('flows', [])

# 提取每個流程的終點模組 (endpoint module)
def get_endpoint_module(flow):
    """取得流程終點的模組"""
    modules = flow.get('modules', [])
    if modules:
        return modules[-1]  # 最後一個模組就是終點
    return 'unknown'

module_counts = Counter(get_endpoint_module(flow) for flow in flows)

print(f"\n📈 模組分佈:")
for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = count / len(flows) * 100 if flows else 0
    print(f"  {module:25} : {count:4} ({percentage:5.1f}%)")

# 檢查 unknown 模組的詳細信息
unknown_flows = [f for f in flows if get_endpoint_module(f) == 'unknown']
if unknown_flows:
    print(f"\n🔍 Unknown 模組詳細分析 (共 {len(unknown_flows)} 個):")
    
    # 按路徑分組
    path_groups = {}
    for flow in unknown_flows:
        full_paths = flow.get('full_path', [])
        if not full_paths:
            continue
        
        # 取終點路徑
        endpoint_path = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
        
        # 提取主要目錄
        if 'services/core/tools/' in endpoint_path:
            key = 'services/core/tools/'
        elif 'services/core/ui/' in endpoint_path:
            key = 'services/core/ui/'
        elif 'services/core/aiva_core/' in endpoint_path:
            key = 'services/core/aiva_core/'
        else:
            key = 'other'
        
        if key not in path_groups:
            path_groups[key] = []
        path_groups[key].append(flow)
    
    print(f"\n  路徑分組:")
    for path, group_flows in sorted(path_groups.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = len(group_flows) / len(unknown_flows) * 100
        print(f"    {path:35} : {len(group_flows):3} ({percentage:5.1f}%)")
        
        # 顯示前3個範例
        print(f"      範例:")
        for flow in group_flows[:3]:
            full_paths = flow.get('full_path', [])
            endpoint = full_paths[-1] if isinstance(full_paths, list) and full_paths else 'N/A'
            print(f"        Flow {flow['id']:3} : {endpoint}")
        if len(group_flows) > 3:
            print(f"        ... 還有 {len(group_flows) - 3} 個")
        print()

# 檢查是否有重複分類
flow_ids = [f['id'] for f in flows]
if len(flow_ids) != len(set(flow_ids)):
    print("\n⚠️  警告: 發現重複的 Flow ID!")
    duplicates = [id for id in flow_ids if flow_ids.count(id) > 1]
    print(f"  重複的 ID: {set(duplicates)}")

# 檢查分類準確性指標
if 'accuracy' in data:
    print(f"\n✅ 分類準確性: {data['accuracy']}")

# 檢查各模組的終點統計
print(f"\n📍 終點腳本統計 (前15個):")
endpoint_scripts = Counter()
for flow in flows:
    full_paths = flow.get('full_path', [])
    if full_paths:
        # 取終點路徑
        endpoint_path = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
        # 提取腳本名稱
        script = endpoint_path.split('/')[-1] if '/' in endpoint_path else endpoint_path.split('\\')[-1]
        endpoint_scripts[script] += 1

for script, count in endpoint_scripts.most_common(15):
    print(f"  {script:50} : {count:3}")

print("\n" + "="*60)
