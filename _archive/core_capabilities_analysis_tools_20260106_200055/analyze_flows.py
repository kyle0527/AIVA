"""分析 840 flows 的入口點和路徑差異"""
import json
from pathlib import Path
from collections import defaultdict

# 讀取數據
json_file = Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json")
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data['flows']

# 1. 分析起點分佈
print("=" * 80)
print("📊 Flow 起點分析")
print("=" * 80)

start_points = defaultdict(lambda: {'count': 0, 'module': '', 'type': ''})
for flow in flows:
    start = flow['start']
    start_points[start]['count'] += 1
    if not start_points[start]['module']:
        start_points[start]['module'] = flow['classifications'][0]['module']
        start_points[start]['type'] = flow['classifications'][0]['component_type']

print(f"\n總起點數量: {len(start_points)}\n")
print("Top 20 最常見起點:\n")
sorted_starts = sorted(start_points.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
for start, info in sorted_starts:
    print(f"  {start}: {info['count']} flows [{info['module']}, {info['type']}]")

# 2. 分析多路徑情況
print("\n" + "=" * 80)
print("🔀 同起點同終點但路徑不同的案例")
print("=" * 80)

routes = defaultdict(lambda: {'paths': [], 'start': '', 'end': ''})
for flow in flows:
    key = f"{flow['start']}|{flow['end']}"
    path = " → ".join(flow['path'])
    if not routes[key]['start']:
        routes[key]['start'] = flow['start']
        routes[key]['end'] = flow['end']
    routes[key]['paths'].append(path)

multi_path = {k: v for k, v in routes.items() if len(v['paths']) > 1}
print(f"\n發現 {len(multi_path)} 個多路徑案例（顯示前 10 個）:\n")

for i, (key, info) in enumerate(list(multi_path.items())[:10], 1):
    print(f"{i}. 起點→終點: {info['start']} → {info['end']}")
    print(f"   路徑數: {len(info['paths'])}")
    for j, path in enumerate(info['paths'][:3], 1):
        print(f"     路徑{j}: {path}")
    if len(info['paths']) > 3:
        print(f"     ... 還有 {len(info['paths']) - 3} 條路徑")
    print()

# 3. 六大模組的入口點統計
print("=" * 80)
print("📦 六大模組的入口點統計")
print("=" * 80)

module_starts = defaultdict(lambda: defaultdict(int))
for flow in flows:
    start_module = flow['classifications'][0]['module']
    start_script = flow['start']
    module_starts[start_module][start_script] += 1

modules = ["service_backbone", "cognitive_core", "task_planning", "external_learning", "core_capabilities"]
for module in modules:
    print(f"\n[{module}]")
    module_data = sorted(module_starts[module].items(), key=lambda x: x[1], reverse=True)[:5]
    for script, count in module_data:
        print(f"  - {script}: {count} flows")

# 4. 生成 Manifest 創建建議
print("\n" + "=" * 80)
print("💡 Manifest 創建建議（按優先級排序）")
print("=" * 80)

# 統計每個起點作為入口的次數
entry_candidates = []
for start, info in start_points.items():
    # 優先級計算：flow數量 + AI組件加權
    priority = info['count']
    if info['type'] == 'AI組件':
        priority *= 2  # AI組件優先級翻倍
    
    entry_candidates.append({
        'script': start,
        'module': info['module'],
        'type': info['type'],
        'flows': info['count'],
        'priority': priority
    })

entry_candidates.sort(key=lambda x: x['priority'], reverse=True)

print("\nTop 30 建議創建 Manifest 的入口點：\n")
for i, candidate in enumerate(entry_candidates[:30], 1):
    print(f"{i:2d}. {candidate['script']:30s} | {candidate['module']:20s} | "
          f"{candidate['type']:8s} | {candidate['flows']:3d} flows")

# 5. 按模組分組的建議
print("\n" + "=" * 80)
print("📋 按模組分組的 Manifest 創建優先級")
print("=" * 80)

for module in modules:
    module_candidates = [c for c in entry_candidates if c['module'] == module][:10]
    if module_candidates:
        print(f"\n[{module}] - Top 10:")
        for i, c in enumerate(module_candidates, 1):
            print(f"  {i}. {c['script']} ({c['flows']} flows, {c['type']})")

print("\n✅ 分析完成！建議優先為高 flow 數量的入口點創建 Manifest。")
