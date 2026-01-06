"""全面分析 Manifest 創建狀態與 840 flows 對應關係"""
import json
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("🔍 全面分析：Manifest 系統 vs 840 Flows")
print("=" * 80)

# 1. 讀取 840 flows 數據
flows_file = Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json")
with open(flows_file, 'r', encoding='utf-8') as f:
    flows_data = json.load(f)

flows = flows_data['flows']
print(f"\n📊 840 Flows 分析結果:")
print(f"  總 flows: {len(flows)}")

# 統計真實入口點
start_points = defaultdict(lambda: {'count': 0, 'module': '', 'type': ''})
for flow in flows:
    start = flow['start']
    start_points[start]['count'] += 1
    if not start_points[start]['module']:
        start_points[start]['module'] = flow['classifications'][0]['module']
        start_points[start]['type'] = flow['classifications'][0]['component_type']

print(f"  唯一起點: {len(start_points)}")
print(f"\n  起點分佈:")
for start, info in sorted(start_points.items(), key=lambda x: x[1]['count'], reverse=True):
    print(f"    {start}: {info['count']} flows [{info['module']}, {info['type']}]")

# 2. 讀取已創建的 Manifests
manifests_dir = Path("C:/D/fold7/AIVA-git/services/core/aiva_core/core_capabilities/manifests/capabilities")
manifest_files = sorted(manifests_dir.glob("*.json"))

print(f"\n📦 已創建的 Manifests:")
print(f"  總數: {len(manifest_files)}")

manifests = []
for f in manifest_files:
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
        manifests.append({
            'flow_id': data['meta']['flow_id'],
            'name': data['meta']['tool_name'],
            'module': data['meta']['module'],
            'file': f.name
        })
        print(f"  Flow {data['meta']['flow_id']}: {data['meta']['tool_name']}")
        print(f"    - 文件: {f.name}")
        print(f"    - 模組: {data['meta']['module']}")
        print(f"    - 風險: {data['ai_cognitive']['risk_level']}")
        print(f"    - 可調參數: {len(data['parameters']['tunable'])}")

# 3. 對應關係分析
print(f"\n🔗 對應關係分析:")
print(f"\n  840 Flows 的 6 個入口點:")
entry_points = [
    "session_state_manager",
    "scalable_bio_trainer", 
    "logging_formatter",
    "websocket_manager",
    "monitoring",
    "initial_surface"
]

covered = []
not_covered = []

for ep in entry_points:
    flows_count = start_points[ep]['count']
    # 檢查是否有對應的 Manifest
    has_manifest = any(ep in m['name'].lower() or ep in m['file'] for m in manifests)
    
    if has_manifest:
        covered.append(f"✅ {ep} ({flows_count} flows) - 已創建 Manifest")
    else:
        not_covered.append(f"❌ {ep} ({flows_count} flows) - 缺少 Manifest")

print("\n  已覆蓋的入口點:")
for item in covered:
    print(f"    {item}")

if not_covered:
    print("\n  未覆蓋的入口點:")
    for item in not_covered:
        print(f"    {item}")
else:
    print("\n  ✅ 所有 6 個入口點都已創建 Manifest！")

# 4. 額外創建的 Manifests
print(f"\n📋 額外創建的 Manifests（非 840 flows 入口點）:")
extra_manifests = [m for m in manifests if not any(ep in m['name'].lower() for ep in entry_points)]
if extra_manifests:
    for m in extra_manifests:
        print(f"  - Flow {m['flow_id']}: {m['name']} (模組: {m['module']})")
else:
    print("  無")

# 5. 問題診斷
print(f"\n⚠️  潛在問題診斷:")

issues = []

# 檢查 flow_id 連續性
flow_ids = sorted([m['flow_id'] for m in manifests])
expected_ids = list(range(len(manifests)))
if flow_ids != expected_ids:
    issues.append(f"Flow ID 不連續: 實際 {flow_ids}, 預期 {expected_ids}")

# 檢查模組分佈
modules = defaultdict(int)
for m in manifests:
    modules[m['module']] += 1

print(f"\n  模組分佈:")
for module, count in sorted(modules.items(), key=lambda x: x[1], reverse=True):
    print(f"    {module}: {count} 個 Manifests")

# 檢查是否缺少關鍵模組
critical_modules = ["cognitive_core", "task_planning", "core_capabilities", "service_backbone", "external_learning"]
missing_modules = [m for m in critical_modules if modules[m] == 0]
if missing_modules:
    issues.append(f"缺少關鍵模組的 Manifests: {missing_modules}")

if issues:
    for issue in issues:
        print(f"  ⚠️  {issue}")
else:
    print(f"  ✅ 未發現結構性問題")

# 6. 覆蓋率統計
print(f"\n📈 覆蓋率統計:")
total_entry_flows = sum(start_points[ep]['count'] for ep in entry_points)
print(f"  840 flows 中的入口點 flows: {total_entry_flows}/{len(flows)} ({total_entry_flows/len(flows)*100:.1f}%)")
print(f"  已創建 Manifest 的入口點: {len(covered)}/6 ({len(covered)/6*100:.1f}%)")

# 7. 下一步建議
print(f"\n💡 下一步建議:")
if len(covered) == 6:
    print(f"  ✅ Phase 1 完成：所有 6 個入口點 Manifest 已創建")
    print(f"  📋 建議：")
    print(f"    1. 測試所有 9 個 Manifests 的命令生成")
    print(f"    2. 驗證 CommandBuilder 與 ManifestLoader 整合")
    print(f"    3. 開始 Phase 2：創建核心 AI 決策組件 Manifests")
else:
    print(f"  ⚠️  尚未完成：還需創建 {len(not_covered)} 個入口點 Manifest")
    for item in not_covered:
        print(f"    - {item}")

print(f"\n" + "=" * 80)
print(f"✅ 分析完成")
print(f"=" * 80)
