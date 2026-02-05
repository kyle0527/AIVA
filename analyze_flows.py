#!/usr/bin/env python3
"""分析 Internal Exploration flows"""
import json
from pathlib import Path
from collections import defaultdict

# 讀取 classification_data.json
data_path = Path("services/integration/data/internal_exploration/classification_data.json")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data['flows']
print(f"總flows: {data['metadata']['total_flows']}")
print(f"實際flows數: {len(flows)}\n")

# 按模組統計
modules = defaultdict(int)
for f in flows:
    modules[f.get('module', 'unknown')] += 1

print("=== 模組分布 ===")
for mod, count in sorted(modules.items(), key=lambda x: -x[1]):
    print(f"  {mod}: {count}")

# 查找攻擊相關flows
print("\n=== 攻擊相關 flows ===")
keywords = ['attack', 'exploit', 'vuln', 'coordinator', 'scan', 'executor', 'xss', 'sqli']
attack_related = []

for f in flows:
    flow_text = str(f).lower()
    matched_kws = [kw for kw in keywords if kw in flow_text]
    if matched_kws:
        attack_related.append((f, matched_kws))

print(f"找到 {len(attack_related)} 個相關flows\n")

# 顯示前10個
for i, (flow, kws) in enumerate(attack_related[:10], 1):
    print(f"\n--- Flow {flow['id']}: {flow.get('ability_name', 'N/A')} ---")
    print(f"匹配關鍵字: {', '.join(kws)}")
    print(f"路徑長度: {flow['length']}")
    print(f"模組: {flow.get('module', 'N/A')}")
    print(f"入口函數: {flow['func_names'][0] if flow.get('func_names') else 'N/A'}")
    print(f"完整路徑: {' -> '.join(flow['path'])}")
    
# AI能力統計
print("\n\n=== AI 能力分類 ===")
ai_types = defaultdict(int)
for f in flows:
    ai_type = f.get('ai_classification', {}).get('component_type', 'unknown')
    ai_types[ai_type] += 1

for ai_type, count in sorted(ai_types.items(), key=lambda x: -x[1]):
    print(f"  {ai_type}: {count}")

# 有入口函數的flows
flows_with_entry = [f for f in flows if f.get('func_names') and len(f['func_names']) > 0]
print(f"\n=== 入口函數統計 ===")
print(f"有明確入口函數的flows: {len(flows_with_entry)} / {len(flows)}")
