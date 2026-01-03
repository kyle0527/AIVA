"""驗證 840 flows 與 102 腳本的關係"""
import json
from pathlib import Path
from collections import defaultdict

# 讀取 840 flows 數據
flows_file = Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json")
with open(flows_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data['flows']

print("=" * 80)
print("🔍 驗證：840 Flows 與 102 腳本的關係")
print("=" * 80)

# 1. 確認總 flows 數
print("\n📊 基本統計:")
print(f"  總 flows 數量: {len(flows)}")

# 2. 提取所有唯一腳本
all_scripts = set()
for flow in flows:
    # 每個 flow 有一個 path，包含多個腳本節點
    path = flow.get('path', [])
    for script in path:
        all_scripts.add(script)

print(f"  唯一腳本數量: {len(all_scripts)}")
print("  (從所有 flow 的 path 中提取並去重)")

# 3. 顯示 flow 結構示例
print("\n📋 Flow 結構示例 (前 3 個):")
for i, flow in enumerate(flows[:3], 1):
    print(f"\n  Flow {i}:")
    print(f"    flow_id: {flow.get('flow_id')}")
    print(f"    start: {flow.get('start')}")
    print(f"    end: {flow.get('end')}")
    print(f"    path: {' → '.join(flow.get('path', []))}")
    print(f"    路徑長度: {len(flow.get('path', []))} 個節點")

# 4. 統計路徑長度分佈
path_lengths = defaultdict(int)
for flow in flows:
    length = len(flow.get('path', []))
    path_lengths[length] += 1

print("\n📊 路徑長度分佈:")
for length in sorted(path_lengths.keys()):
    print(f"  {length} 個節點: {path_lengths[length]} flows")

# 5. 統計每個腳本在多少 flows 中出現
script_usage = defaultdict(int)
for flow in flows:
    for script in flow.get('path', []):
        script_usage[script] += 1

print("\n🔥 最常出現的腳本 (Top 20):")
sorted_scripts = sorted(script_usage.items(), key=lambda x: x[1], reverse=True)
for script, count in sorted_scripts[:20]:
    print(f"  {script}: 出現在 {count} 個 flows 中")

# 6. 確認入口點腳本
start_points = defaultdict(int)
for flow in flows:
    start_points[flow['start']] += 1

print("\n🚪 入口點腳本 (作為 start 的腳本):")
for start, count in sorted(start_points.items(), key=lambda x: x[1], reverse=True):
    print(f"  {start}: 作為 {count} 個 flows 的起點")

# 7. 關鍵結論
print("\n" + "=" * 80)
print("✅ 確認結論:")
print("=" * 80)
print(f"  1. 總共有 {len(flows)} 個 flows（調用鏈組合）")
print(f"  2. 這些 flows 由 {len(all_scripts)} 個唯一腳本組成")
print(f"  3. 每個 flow 包含 {min(path_lengths.keys())} 到 {max(path_lengths.keys())} 個節點")
print(f"  4. 只有 {len(start_points)} 個腳本作為入口點（可接收命令）")
print(f"  5. 其他 {len(all_scripts) - len(start_points)} 個腳本只作為內部節點")
print("\n💡 理解:")
print("  - 102 個腳本 = 從 840 flows 的所有 path 中提取的唯一腳本")
print("  - 6 個入口點 = 可以接收外部命令的腳本")
print("  - Manifest 應該為：")
print("    • 6 個入口點（必須，用於接收命令）")
print("    • 其他關鍵內部節點（可選，用於調用鏈）")
print("=" * 80)

# 8. 列出所有 102 個腳本（前 30 個）
print("\n📜 所有唯一腳本列表 (顯示前 30 個):")
for i, script in enumerate(sorted(all_scripts)[:30], 1):
    usage = script_usage[script]
    is_entry = "🚪入口" if script in start_points else ""
    print(f"  {i:2d}. {script:40s} (用於 {usage:3d} flows) {is_entry}")
if len(all_scripts) > 30:
    print(f"  ... 還有 {len(all_scripts) - 30} 個腳本")
