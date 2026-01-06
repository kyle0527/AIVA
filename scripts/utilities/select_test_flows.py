#!/usr/bin/env python3
"""選擇各模組代表性 Flow 進行測試"""
import json

with open('flows_by_endpoint_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 每個模組選擇一些代表性的 Flow
modules = {}
for flow in data['flows']:
    mod = flow.get('endpoint_module', 'unknown')
    if mod not in modules:
        modules[mod] = []
    if len(modules[mod]) < 3:
        modules[mod].append({
            'id': flow['id'],
            'path': ' -> '.join(flow['path']),
            'length': flow['length'],
            'capability': flow.get('capability_key', '')
        })

for mod, flows in modules.items():
    print(f'\n=== {mod} ===')
    for f in flows:
        print(f"  ID {f['id']}: {f['path']} (len:{f['length']})")
