#!/usr/bin/env python3
"""分析外部模組 flows - 真正的攻擊能力"""
import json
from pathlib import Path
from collections import defaultdict

# 讀取 external_classification.json
data_path = Path("services/integration/data/internal_exploration/external_classification.json")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("AIVA 外部模組攻擊能力分析")
print("=" * 80)

# 總體統計
metadata = data['metadata']
print(f"\n📊 總體統計")
print(f"  總flows: {metadata['total_flows']}")
print(f"  總模組: {metadata['total_modules']}")
print(f"  支援語言: {', '.join(metadata['languages'])}")

# 模組分布
print(f"\n📦 模組分布 (按flow數量)")
stats = metadata['statistics']['by_module']
for module, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {module:30s}: {count:3d} flows")

# 攻擊類型分布
print(f"\n🎯 攻擊類型分布")
types = metadata['statistics']['by_type']
for type_name, count in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {type_name:20s}: {count:3d} flows")

# 可操作性統計
print(f"\n✅ 可操作性統計")
op = metadata['operability']
print(f"  可操作flows: {op['total_operable']} ({op['operable_rate']:.1f}%)")
print(f"  不可操作flows: {op['total_non_operable']}")
print(f"\n  不可操作原因:")
for reason, count in op['non_operable_reasons'].items():
    print(f"    {reason}: {count}")

# 核心攻擊模組詳細分析
print("\n" + "=" * 80)
print("🔍 核心攻擊模組詳細分析")
print("=" * 80)

modules_data = data['modules']

# 定義核心攻擊模組
core_modules = [
    'function_xss',
    'function_sqli', 
    'function_ssrf',
    'function_idor',
    'function_web_scanner',
    'function_postex'
]

for module_name in core_modules:
    if module_name not in modules_data:
        continue
    
    module = modules_data[module_name]
    print(f"\n{'='*60}")
    print(f"📌 {module_name.upper()}")
    print(f"{'='*60}")
    print(f"類型: {module['info'].get('type', 'N/A')}")
    print(f"分類: {module['info'].get('category', 'N/A')}")
    print(f"說明: {module['info'].get('name', 'N/A')}")
    print(f"Flow數量: {module['flow_count']}")
    
    # 分析可操作的flows
    flows = module['flows']
    operable = [f for f in flows if f.get('operable') == '✅ 可操作']
    non_operable = [f for f in flows if f.get('operable') != '✅ 可操作']
    
    print(f"\n可操作flows: {len(operable)} / {len(flows)} ({len(operable)/len(flows)*100:.1f}%)")
    
    # 顯示前5個可操作flows
    if operable:
        print(f"\n前5個可操作flows:")
        for i, flow in enumerate(operable[:5], 1):
            func_name = flow['func_names'][0] if flow.get('func_names') else 'N/A'
            path = ' -> '.join(flow['path'][:3]) + ('...' if len(flow['path']) > 3 else '')
            print(f"  {i}. Flow {flow['id']:3d}: {func_name}")
            print(f"     路徑: {path}")
            if flow.get('cli_command'):
                print(f"     CLI: {flow['cli_command']}")

# 生成 P0 實施建議
print("\n" + "=" * 80)
print("🚀 P0 實施建議")
print("=" * 80)

print("""
基於分析結果，P0 應該聚焦以下模組：

1️⃣ function_sqli (115 flows)
   - 最大的攻擊模組
   - 優先級: ⭐⭐⭐
   - 建議: 創建 sqli_commands.jsonl

2️⃣ function_xss (97 flows)
   - 第二大攻擊模組  
   - 優先級: ⭐⭐⭐
   - 建議: 創建 xss_commands.jsonl

3️⃣ function_web_scanner (74 flows)
   - 掃描能力
   - 優先級: ⭐⭐⭐
   - 建議: 創建 scanner_commands.jsonl

4️⃣ function_ssrf (64 flows)
   - SSRF 攻擊
   - 優先級: ⭐⭐
   - 建議: 創建 ssrf_commands.jsonl

5️⃣ function_postex (49 flows)
   - 後滲透
   - 優先級: ⭐⭐
   - 建議: 創建 postex_commands.jsonl

注意事項:
- 只選擇「可操作」的flows (54.67%)
- 排除私有函數 (以 _ 開頭的 226 個)
- 排除需要 Web Context 的函數 (12 個)
- 優先使用有 CLI 命令的 flows
""")

# 統計有 CLI 命令的 flows
flows_with_cli = []
for module_name, module in modules_data.items():
    for flow in module['flows']:
        if flow.get('cli_command') and flow.get('operable') == '✅ 可操作':
            flows_with_cli.append({
                'module': module_name,
                'flow_id': flow['id'],
                'func_name': flow['func_names'][0] if flow.get('func_names') else 'N/A',
                'cli': flow['cli_command']
            })

print(f"\n📋 有 CLI 命令的可操作flows: {len(flows_with_cli)}")
if flows_with_cli:
    print("\n示例:")
    for i, f in enumerate(flows_with_cli[:5], 1):
        print(f"  {i}. {f['module']}: {f['func_name']}")
        print(f"     {f['cli']}")
