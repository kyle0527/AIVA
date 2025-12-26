"""從 840 flows 中選擇 10 個有代表性的 flows 進行驗證"""
import json
from pathlib import Path
from collections import defaultdict

# 讀取 840 flows
flows_file = Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json")
with open(flows_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data['flows']

print("=" * 80)
print("🎯 從 840 Flows 中選擇 10 個驗證樣本")
print("=" * 80)

# 選擇標準：
# 1. 不同路徑長度 (2-5 個節點)
# 2. 不同入口點
# 3. 涵蓋不同模組
# 4. 常見和罕見的組合

selected_flows = []

# 按入口點和路徑長度分組
by_entry_length = defaultdict(list)
for i, flow in enumerate(flows):
    key = (flow['start'], len(flow['path']))
    by_entry_length[key].append((i, flow))

# 選擇策略
selections = [
    ('monitoring', 2),              # 最短路徑
    ('monitoring', 3),              
    ('monitoring', 4),              
    ('monitoring', 5),              # 最長路徑
    ('initial_surface', 2),         # 不同入口點
    ('initial_surface', 3),
    ('session_state_manager', 2),   # 最常見入口點
    ('session_state_manager', 5),   # 長路徑
    ('scalable_bio_trainer', 2),    # AI 訓練入口
    ('scalable_bio_trainer', 4),
]

print("\n📋 選擇的 10 個驗證 Flows:\n")

for idx, (entry, length) in enumerate(selections, 1):
    candidates = by_entry_length.get((entry, length), [])
    if candidates:
        flow_idx, flow = candidates[0]  # 選第一個
        selected_flows.append({
            'test_id': idx,
            'flow_index': flow_idx,
            'flow': flow
        })
        
        path_str = ' → '.join(flow['path'])
        print(f"Flow {idx:2d} (原始索引 {flow_idx:3d}):")
        print(f"  入口點: {flow['start']}")
        print(f"  終點: {flow['end']}")
        print(f"  路徑 ({len(flow['path'])} 節點): {path_str}")
        print(f"  模組: {flow['classifications'][0]['module']}")
        print()

# 保存選擇的 flows 供後續使用
output_file = Path("C:/D/fold7/AIVA-git/services/core/aiva_core/core_capabilities/manifests/test_flows_10.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(selected_flows, f, indent=2, ensure_ascii=False)

print(f"✅ 已保存到: {output_file}")

# 統計涵蓋的腳本
covered_scripts = set()
for sf in selected_flows:
    for script in sf['flow']['path']:
        covered_scripts.add(script)

print(f"\n📊 統計:")
print(f"  選擇的 flows: {len(selected_flows)}")
print(f"  涵蓋的唯一腳本: {len(covered_scripts)}")
print(f"  腳本列表: {', '.join(sorted(covered_scripts)[:10])}...")
