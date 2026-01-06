#!/usr/bin/env python3
"""
分析 74 個 unknown flows 的詳細來源
找出為何這些 flows 無法被分類到六大模組
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

flows = data.get('flows', [])

# 六大模組列表
SIX_MODULES = [
    'cognitive_core',
    'internal_exploration', 
    'task_planning',
    'external_learning',
    'core_capabilities',
    'service_backbone'
]

print("\n" + "="*70)
print("🔍 74 個 Unknown Flows 詳細分析報告")
print("="*70)

# 提取 unknown flows
unknown_flows = []
for flow in flows:
    modules = flow.get('modules', [])
    if modules:
        endpoint_module = modules[-1]
        if endpoint_module == 'unknown':
            unknown_flows.append(flow)

print(f"\n📊 總數: {len(unknown_flows)} 個 unknown flows")
print(f"   佔比: {len(unknown_flows)/len(flows)*100:.1f}% (共 {len(flows)} 個 flows)")

# 分析終點路徑
print(f"\n📍 終點路徑分析:")
print("-" * 70)

endpoint_paths = Counter()
endpoint_details = {}

for flow in unknown_flows:
    full_paths = flow.get('full_path', [])
    if full_paths:
        endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
        endpoint_paths[endpoint] += 1
        
        if endpoint not in endpoint_details:
            endpoint_details[endpoint] = []
        endpoint_details[endpoint].append(flow['id'])

# 按數量排序顯示
for path, count in endpoint_paths.most_common():
    print(f"\n  {count:2} 個 flows → {path}")
    
    # 提取目錄結構
    if 'services/core/tools/' in path:
        category = "🔧 工具層 (services/core/tools/)"
    elif 'services/core/ui/' in path:
        category = "🖥️ UI層 (services/core/ui/)"
    elif 'services/core/aiva_core/' in path:
        category = "⚠️ 在 aiva_core 內但未匹配"
    else:
        category = "❓ 其他位置"
    
    print(f"     分類: {category}")
    
    # 顯示前5個 flow ID
    flow_ids = endpoint_details[path][:5]
    print(f"     Flow IDs: {', '.join(map(str, flow_ids))}", end="")
    if len(endpoint_details[path]) > 5:
        print(f" ... (+{len(endpoint_details[path])-5} 個)")
    else:
        print()

# 按目錄分組統計
print(f"\n📂 按目錄結構分組:")
print("-" * 70)

directory_groups = {
    'services/core/tools/': [],
    'services/core/ui/': [],
    'services/core/aiva_core/': [],
    'other': []
}

for flow in unknown_flows:
    full_paths = flow.get('full_path', [])
    if full_paths:
        endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
        
        categorized = False
        for dir_key in ['services/core/tools/', 'services/core/ui/', 'services/core/aiva_core/']:
            if dir_key in endpoint:
                directory_groups[dir_key].append(flow)
                categorized = True
                break
        
        if not categorized:
            directory_groups['other'].append(flow)

for dir_name, group_flows in directory_groups.items():
    if group_flows:
        percentage = len(group_flows) / len(unknown_flows) * 100
        print(f"\n  {dir_name:35} : {len(group_flows):3} ({percentage:5.1f}%)")

# 深入分析 services/core/tools/
tools_flows = directory_groups.get('services/core/tools/', [])
if tools_flows:
    print(f"\n🔧 services/core/tools/ 詳細分析 ({len(tools_flows)} 個):")
    print("-" * 70)
    
    tools_scripts = Counter()
    for flow in tools_flows:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            tools_scripts[script_name] += 1
    
    for script, count in tools_scripts.most_common():
        print(f"  {script:50} : {count:3} 個 flows")

# 深入分析 services/core/ui/
ui_flows = directory_groups.get('services/core/ui/', [])
if ui_flows:
    print(f"\n🖥️ services/core/ui/ 詳細分析 ({len(ui_flows)} 個):")
    print("-" * 70)
    
    ui_scripts = Counter()
    for flow in ui_flows:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            ui_scripts[script_name] += 1
    
    for script, count in ui_scripts.most_common():
        print(f"  {script:50} : {count:3} 個 flows")

# 分析為何無法分類
print(f"\n❓ 為何這些 flows 被標記為 unknown?")
print("-" * 70)
print("""
原因:
  1. 這些腳本不在六大模組的標準路徑下:
     ✅ services/core/aiva_core/cognitive_core/
     ✅ services/core/aiva_core/internal_exploration/
     ✅ services/core/aiva_core/task_planning/
     ✅ services/core/aiva_core/external_learning/
     ✅ services/core/aiva_core/core_capabilities/
     ✅ services/core/aiva_core/service_backbone/
  
  2. 而是在以下位置:
     ❌ services/core/tools/          (工具層，不屬於六大模組)
     ❌ services/core/ui/              (UI層，不屬於六大模組)
     ❌ 其他非標準路徑
  
  3. 分類器算法 (_classify_module_from_path) 只識別:
     - 路徑中包含 '/cognitive_core/' 的文件
     - 路徑中包含 '/internal_exploration/' 的文件
     - 路徑中包含 '/task_planning/' 的文件
     - 路徑中包含 '/external_learning/' 的文件
     - 路徑中包含 '/core_capabilities/' 的文件
     - 路徑中包含 '/service_backbone/' 的文件
     
     這些 flows 的終點不包含上述任何模組路徑，因此被標記為 'unknown'。
""")

# 解決方案建議
print(f"\n💡 解決方案建議:")
print("-" * 70)
print("""
選項 A: 擴展分類器以識別 services/core/ 下的其他目錄
  - 將 tools/ 映射到 service_backbone (基礎設施)
  - 將 ui/ 映射到 core_capabilities (用戶交互)
  - 優點: 所有 flows 都有明確分類
  - 缺點: 可能不符合原始架構設計意圖

選項 B: 保持 unknown 狀態
  - 明確標記這些是「非六大模組」的通用組件
  - 優點: 符合實際架構（tools 和 ui 確實不在六大模組內）
  - 缺點: 8.8% 的 flows 無法通過模組過濾

選項 C: 創建新的分類類別
  - 添加 'common' 或 'shared' 類別
  - 專門用於跨模組的共享組件
  - 優點: 清晰反映架構層次
  - 缺點: 需要修改更多代碼和文檔

推薦: 選項 B (保持 unknown)
  原因: 這些確實不屬於六大模組，保持 unknown 狀態更準確反映實際架構。
""")

print("\n" + "="*70)
