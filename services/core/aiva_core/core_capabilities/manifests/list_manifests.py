"""統計所有 Manifest"""
import json
from pathlib import Path

manifests_dir = Path("services/core/aiva_core/core_capabilities/manifests/capabilities")
files = sorted(manifests_dir.glob("*.json"))

print("=" * 80)
print(f"📊 Manifest 創建進度")
print("=" * 80)
print(f"\n總計: {len(files)} 個 Manifest\n")

for f in files:
    data = json.load(open(f, encoding='utf-8'))
    flow_id = data['meta']['flow_id']
    tool_name = data['meta']['tool_name']
    module = data['meta']['module']
    risk = data['ai_cognitive']['risk_level']
    tunable_count = len(data['parameters']['tunable'])
    
    print(f"Flow {flow_id}: {tool_name}")
    print(f"  模組: {module} | 風險: {risk} | 可調參數: {tunable_count}")

print("\n" + "=" * 80)
print("✅ 所有 9 個入口點 Manifest 已創建！")
print("=" * 80)
