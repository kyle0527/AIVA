#!/usr/bin/env python3
"""
分析各模組內的能力分佈
統計每個模組有多少不同路徑的能力，以及對內/對外能力的分佈
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any


def create_type_summary_entry() -> Dict[str, Any]:
    """工廠函數：創建類型統計條目"""
    return {'flows': 0, 'capabilities': 0, 'modules': []}

# 數據路徑
data_dir = Path("C:/Users/User/Downloads/data/internal_exploration")
json_file = data_dir / "latest_classification.json"

# 讀取數據
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data.get('flows', [])

# 六大模組的對內/對外分類
MODULE_CLASSIFICATION = {
    'cognitive_core': '對內能力',
    'internal_exploration': '對內能力',
    'task_planning': '混合能力',
    'external_learning': '對外能力',
    'core_capabilities': '對外能力',
    'service_backbone': '對內能力',
    'unknown': '通用組件'
}

print("\n" + "="*90)
print("🎯 各模組內能力分佈詳細分析")
print("="*90)

# 按終點模組分組
module_flows = defaultdict(list)
for flow in flows:
    modules = flow.get('modules', [])
    if modules:
        endpoint_module = modules[-1]
        module_flows[endpoint_module].append(flow)

# 統計各模組的唯一路徑
print("\n📊 各模組能力統計:\n")

all_stats = []

for module in sorted(module_flows.keys()):
    flows_in_module = module_flows[module]
    capability_type = MODULE_CLASSIFICATION.get(module, '未分類')
    
    # 統計唯一的終點腳本（代表不同的能力）
    endpoint_scripts = set()
    for flow in flows_in_module:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            # 提取腳本名稱
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            endpoint_scripts.add(script_name)
    
    # 統計路徑數
    total_flows = len(flows_in_module)
    unique_capabilities = len(endpoint_scripts)
    
    # 計算路徑密度（平均每個能力有多少條路徑）
    path_density = total_flows / unique_capabilities if unique_capabilities > 0 else 0
    
    all_stats.append({
        'module': module,
        'type': capability_type,
        'total_flows': total_flows,
        'unique_capabilities': unique_capabilities,
        'path_density': path_density,
        'endpoint_scripts': endpoint_scripts
    })
    
    print(f"{'='*90}")
    print(f"📦 模組: {module}")
    print(f"   類型: {capability_type}")
    print(f"   總 Flows: {total_flows}")
    print(f"   不同能力數: {unique_capabilities}")
    print(f"   路徑密度: {path_density:.2f} (平均每個能力有多少條路徑)")
    print()

# 按能力類型匯總
print("\n" + "="*90)
print("📈 按能力類型匯總:\n")

type_summary: Dict[str, Dict[str, Any]] = defaultdict(create_type_summary_entry)

for stat in all_stats:
    cap_type = stat['type']
    type_summary[cap_type]['flows'] += stat['total_flows']
    type_summary[cap_type]['capabilities'] += stat['unique_capabilities']
    type_summary[cap_type]['modules'].append(stat['module'])

for cap_type in ['對內能力', '對外能力', '混合能力', '通用組件']:
    if cap_type in type_summary:
        info = type_summary[cap_type]
        flows_count = int(info['flows'])  # type: ignore
        cap_count = int(info['capabilities'])  # type: ignore
        avg_density = flows_count / cap_count if cap_count > 0 else 0
        
        print(f"{'='*90}")
        print(f"🔹 {cap_type}")
        modules_list: List[str] = info['modules']  # type: ignore
        print(f"   包含模組: {', '.join(modules_list)}")
        print(f"   總 Flows: {info['flows']}")
        print(f"   總能力數: {info['capabilities']}")
        print(f"   平均路徑密度: {avg_density:.2f}")
        print()

# 詳細分析：各模組的具體能力列表（Top 10）
print("="*90)
print("🔍 各模組具體能力分析 (Top 10 最多 Flow 的能力):\n")

for stat in sorted(all_stats, key=lambda x: x['total_flows'], reverse=True):
    module = stat['module']
    flows_in_module = module_flows[module]
    
    # 統計每個終點腳本的 flow 數量
    endpoint_count = Counter()
    for flow in flows_in_module:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            endpoint_count[script_name] += 1
    
    print(f"{'='*90}")
    print(f"📦 {module} ({stat['type']})")
    print(f"   共 {stat['unique_capabilities']} 種能力，{stat['total_flows']} 條路徑\n")
    
    for i, (script, count) in enumerate(endpoint_count.most_common(10), 1):
        percentage = count / stat['total_flows'] * 100
        print(f"   {i:2}. {script:50} : {count:3} flows ({percentage:5.1f}%)")
    
    if len(endpoint_count) > 10:
        print(f"   ... 還有 {len(endpoint_count) - 10} 種能力")
    print()

# 生成交叉分析表
print("="*90)
print("📊 能力類型 × 模組 交叉統計表:\n")

print(f"{'模組':<25} | {'類型':<10} | {'Flows':<8} | {'能力數':<8} | {'路徑密度':<10}")
print("-" * 90)

for stat in sorted(all_stats, key=lambda x: x['total_flows'], reverse=True):
    print(f"{stat['module']:<25} | {stat['type']:<10} | {stat['total_flows']:<8} | {stat['unique_capabilities']:<8} | {stat['path_density']:<10.2f}")

# 洞察分析
print("\n" + "="*90)
print("💡 關鍵洞察:\n")

# 找出路徑密度最高和最低的模組
max_density_stat = max(all_stats, key=lambda x: x['path_density'])
min_density_stat = min(all_stats, key=lambda x: x['path_density'])

# 找出能力最多和最少的模組
max_cap_stat = max(all_stats, key=lambda x: x['unique_capabilities'])
min_cap_stat = min(all_stats, key=lambda x: x['unique_capabilities'])

print(f"""
1. 路徑冗餘度分析:
   • 最高路徑密度: {max_density_stat['module']} ({max_density_stat['path_density']:.2f})
     - 每個能力平均有 {max_density_stat['path_density']:.1f} 條不同路徑
     - 說明: 該模組提供了高度靈活的路徑選擇
   
   • 最低路徑密度: {min_density_stat['module']} ({min_density_stat['path_density']:.2f})
     - 每個能力平均有 {min_density_stat['path_density']:.1f} 條路徑
     - 說明: 該模組的路徑較為直接

2. 能力豐富度分析:
   • 能力最豐富: {max_cap_stat['module']} ({max_cap_stat['unique_capabilities']} 種能力)
     - 類型: {max_cap_stat['type']}
     - 體現了該模組的多樣性
   
   • 能力最專注: {min_cap_stat['module']} ({min_cap_stat['unique_capabilities']} 種能力)
     - 類型: {min_cap_stat['type']}
     - 體現了該模組的專業性

3. 對內/對外能力對比:
   • 對內能力: {int(type_summary['對內能力']['capabilities'])} 種能力，{int(type_summary['對內能力']['flows'])} 條路徑
     - 平均密度: {int(type_summary['對內能力']['flows']) / int(type_summary['對內能力']['capabilities']):.2f}
     - 特點: {'路徑選擇豐富' if int(type_summary['對內能力']['flows']) / int(type_summary['對內能力']['capabilities']) > 5 else '路徑較為直接'}
   
   • 對外能力: {int(type_summary['對外能力']['capabilities'])} 種能力，{int(type_summary['對外能力']['flows'])} 條路徑
     - 平均密度: {int(type_summary['對外能力']['flows']) / int(type_summary['對外能力']['capabilities']):.2f}
     - 特點: {'路徑選擇豐富' if int(type_summary['對外能力']['flows']) / int(type_summary['對外能力']['capabilities']) > 5 else '路徑較為直接'}

4. 設計理念:
   • 內部模組通常有更高的路徑密度，提供靈活性和容錯性
   • 外部模組通常更專注，路徑更直接，確保執行效率
   • 混合能力在兩者之間取得平衡
""")

print("="*90)

# 生成 Markdown 報告
report_content = f"""# 各模組能力分佈詳細分析報告

> **生成時間**: 2026-01-01  
> **分析維度**: 模組 × 能力類型 × 路徑密度

## 📊 整體統計

| 模組 | 類型 | Flows | 能力數 | 路徑密度 |
|------|------|-------|--------|----------|
"""

for stat in sorted(all_stats, key=lambda x: x['total_flows'], reverse=True):
    report_content += f"| {stat['module']} | {stat['type']} | {stat['total_flows']} | {stat['unique_capabilities']} | {stat['path_density']:.2f} |\n"

report_content += f"""

## 🔹 對內能力詳細

**包含模組**: {', '.join(list(type_summary['對內能力']['modules']))}

| 指標 | 數值 |
|------|------|
| 總 Flows | {int(type_summary['對內能力']['flows'])} |
| 總能力數 | {int(type_summary['對內能力']['capabilities'])} |
| 平均路徑密度 | {int(type_summary['對內能力']['flows']) / int(type_summary['對內能力']['capabilities']):.2f} |

### 各模組詳細

"""

for module in list(type_summary['對內能力']['modules']):  # type: ignore
    stat = next(s for s in all_stats if s['module'] == module)
    flows_in_module = module_flows[module]
    endpoint_count = Counter()
    for flow in flows_in_module:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            endpoint_count[script_name] += 1
    
    report_content += f"""
#### {module}

- **Flows**: {stat['total_flows']}
- **能力數**: {stat['unique_capabilities']}
- **路徑密度**: {stat['path_density']:.2f}

**Top 5 能力**:

"""
    for i, (script, count) in enumerate(endpoint_count.most_common(5), 1):
        percentage = count / stat['total_flows'] * 100
        report_content += f"{i}. `{script}`: {count} flows ({percentage:.1f}%)\n"

report_content += f"""

## 🔸 對外能力詳細

**包含模組**: {', '.join(list(type_summary['對外能力']['modules']))}

| 指標 | 數值 |
|------|------|
| 總 Flows | {int(type_summary['對外能力']['flows'])} |
| 總能力數 | {int(type_summary['對外能力']['capabilities'])} |
| 平均路徑密度 | {int(type_summary['對外能力']['flows']) / int(type_summary['對外能力']['capabilities']):.2f} |

### 各模組詳細

"""

for module in list(type_summary['對外能力']['modules']):  # type: ignore
    stat = next(s for s in all_stats if s['module'] == module)
    flows_in_module = module_flows[module]
    endpoint_count = Counter()
    for flow in flows_in_module:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            endpoint_count[script_name] += 1
    
    report_content += f"""
#### {module}

- **Flows**: {stat['total_flows']}
- **能力數**: {stat['unique_capabilities']}
- **路徑密度**: {stat['path_density']:.2f}

**Top 5 能力**:

"""
    for i, (script, count) in enumerate(endpoint_count.most_common(5), 1):
        percentage = count / stat['total_flows'] * 100
        report_content += f"{i}. `{script}`: {count} flows ({percentage:.1f}%)\n"

report_content += f"""

## 🔶 混合能力詳細

**包含模組**: {', '.join(list(type_summary['混合能力']['modules']))}

"""

for module in list(type_summary['混合能力']['modules']):  # type: ignore
    stat = next(s for s in all_stats if s['module'] == module)
    flows_in_module = module_flows[module]
    endpoint_count = Counter()
    for flow in flows_in_module:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            endpoint_count[script_name] += 1
    
    report_content += f"""
#### {module}

- **Flows**: {stat['total_flows']}
- **能力數**: {stat['unique_capabilities']}
- **路徑密度**: {stat['path_density']:.2f}

**Top 5 能力**:

"""
    for i, (script, count) in enumerate(endpoint_count.most_common(5), 1):
        percentage = count / stat['total_flows'] * 100
        report_content += f"{i}. `{script}`: {count} flows ({percentage:.1f}%)\n"

report_content += f"""

## ⚪ 通用組件詳細

**包含模組**: {', '.join(list(type_summary['通用組件']['modules']))}

"""

for module in list(type_summary['通用組件']['modules']):  # type: ignore
    stat = next(s for s in all_stats if s['module'] == module)
    flows_in_module = module_flows[module]
    endpoint_count = Counter()
    for flow in flows_in_module:
        full_paths = flow.get('full_path', [])
        if full_paths:
            endpoint = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
            script_name = endpoint.split('/')[-1] if '/' in endpoint else endpoint.split('\\')[-1]
            endpoint_count[script_name] += 1
    
    report_content += f"""
#### {module}

- **Flows**: {stat['total_flows']}
- **能力數**: {stat['unique_capabilities']}
- **路徑密度**: {stat['path_density']:.2f}

**Top 5 能力**:

"""
    for i, (script, count) in enumerate(endpoint_count.most_common(5), 1):
        percentage = count / stat['total_flows'] * 100
        report_content += f"{i}. `{script}`: {count} flows ({percentage:.1f}%)\n"

report_content += f"""

## 💡 關鍵洞察

### 1. 路徑冗餘度

- **最高路徑密度**: {max_density_stat['module']} ({max_density_stat['path_density']:.2f})
  - 每個能力平均有 {max_density_stat['path_density']:.1f} 條不同路徑
  - 提供高度靈活的路徑選擇

- **最低路徑密度**: {min_density_stat['module']} ({min_density_stat['path_density']:.2f})
  - 每個能力平均有 {min_density_stat['path_density']:.1f} 條路徑
  - 路徑較為直接

### 2. 能力豐富度

- **能力最豐富**: {max_cap_stat['module']} ({max_cap_stat['unique_capabilities']} 種能力)
  - 類型: {max_cap_stat['type']}
  - 體現模組的多樣性

- **能力最專注**: {min_cap_stat['module']} ({min_cap_stat['unique_capabilities']} 種能力)
  - 類型: {min_cap_stat['type']}
  - 體現模組的專業性

### 3. 內外能力對比

**對內能力**:
- 能力數: {type_summary['對內能力']['capabilities']}
- Flows: {type_summary['對內能力']['flows']}
- 平均密度: {int(type_summary['對內能力']['flows']) / int(type_summary['對內能力']['capabilities']):.2f}
- 特點: {'路徑選擇豐富，靈活性高' if int(type_summary['對內能力']['flows']) / int(type_summary['對內能力']['capabilities']) > 5 else '路徑較為直接'}

**對外能力**:
- 能力數: {int(type_summary['對外能力']['capabilities'])}
- Flows: {int(type_summary['對外能力']['flows'])}
- 平均密度: {int(type_summary['對外能力']['flows']) / int(type_summary['對外能力']['capabilities']):.2f}
- 特點: {'路徑選擇豐富' if int(type_summary['對外能力']['flows']) / int(type_summary['對外能力']['capabilities']) > 5 else '路徑較為直接，執行高效'}

### 4. 設計理念

1. **內部模組**: 通常有更高的路徑密度，提供靈活性和容錯性
2. **外部模組**: 更專注，路徑更直接，確保執行效率
3. **混合能力**: 在靈活性和效率之間取得平衡
4. **通用組件**: 提供跨模組的共享功能支持

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍
"""

# 保存報告
report_path = Path("C:/D/fold7/AIVA-git/docs/MODULE_CAPABILITIES_DISTRIBUTION_ANALYSIS.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n✅ 詳細報告已保存至: {report_path}")
print("="*90 + "\n")
