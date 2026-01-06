#!/usr/bin/env python3
"""
分析起點終點相同、中間路徑不同的 flows
找出系統中存在的多路徑流程
"""

import json
from pathlib import Path
from collections import defaultdict

# 數據路徑
data_dir = Path("C:/Users/User/Downloads/data/internal_exploration")
json_file = data_dir / "latest_classification.json"

# 讀取數據
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data.get('flows', [])

print("\n" + "="*80)
print("🔍 起點終點相同、中間路徑不同的 Flows 分析")
print("="*80)

# 按 (起點, 終點) 分組
start_end_groups = defaultdict(list)

for flow in flows:
    path = flow.get('path', [])
    if len(path) >= 2:
        start = path[0]
        end = path[-1]
        key = (start, end)
        start_end_groups[key].append(flow)

# 篩選出有多條路徑的組（起點終點相同）
multi_path_groups = {k: v for k, v in start_end_groups.items() if len(v) > 1}

print(f"\n📊 統計結果:")
print("-" * 80)
print(f"總 flows 數: {len(flows)}")
print(f"唯一的 (起點, 終點) 組合數: {len(start_end_groups)}")
print(f"有多條路徑的 (起點, 終點) 組合數: {len(multi_path_groups)}")
print(f"涉及的 flows 總數: {sum(len(v) for v in multi_path_groups.values())}")

# 統計每組的路徑數量
path_count_distribution = defaultdict(int)
for flows_in_group in multi_path_groups.values():
    path_count_distribution[len(flows_in_group)] += 1

print(f"\n📈 路徑數量分佈:")
print("-" * 80)
for count in sorted(path_count_distribution.keys(), reverse=True):
    num_groups = path_count_distribution[count]
    print(f"{count:3} 條路徑: {num_groups:3} 組")

# 顯示前20個最多路徑的組
print(f"\n🏆 路徑最多的組 (Top 20):")
print("-" * 80)

sorted_groups = sorted(multi_path_groups.items(), key=lambda x: len(x[1]), reverse=True)

for i, ((start, end), flows_in_group) in enumerate(sorted_groups[:20], 1):
    print(f"\n{i}. {start} → {end}")
    print(f"   共 {len(flows_in_group)} 條不同路徑")
    
    # 統計路徑長度
    lengths = [len(f['path']) for f in flows_in_group]
    min_len, max_len = min(lengths), max(lengths)
    print(f"   路徑長度範圍: {min_len} - {max_len} 步")
    
    # 顯示前5條路徑範例
    print(f"   路徑範例:")
    for j, flow in enumerate(flows_in_group[:5], 1):
        flow_path = flow.get('path', [])
        path_str = ' → '.join(flow_path)
        print(f"     {j}. Flow {flow['id']:3} ({len(flow_path)} 步): {path_str[:70]}{'...' if len(path_str) > 70 else ''}")
    
    if len(flows_in_group) > 5:
        print(f"     ... 還有 {len(flows_in_group) - 5} 條路徑")

# 分析路徑長度差異
print(f"\n📏 路徑長度差異分析:")
print("-" * 80)

length_diff_groups = []
for (start, end), flows_in_group in multi_path_groups.items():
    lengths = [len(f['path']) for f in flows_in_group]
    if len(set(lengths)) > 1:  # 有不同長度
        min_len, max_len = min(lengths), max(lengths)
        diff = max_len - min_len
        length_diff_groups.append(((start, end), flows_in_group, diff, min_len, max_len))

# 按長度差異排序
length_diff_groups.sort(key=lambda x: x[2], reverse=True)

print(f"有 {len(length_diff_groups)} 組存在不同長度的路徑")
print(f"\n路徑長度差異最大的組 (Top 10):")
print("-" * 80)

for i, ((start, end), flows_in_group, diff, min_len, max_len) in enumerate(length_diff_groups[:10], 1):
    print(f"\n{i}. {start} → {end}")
    print(f"   長度差異: {diff} 步 (最短 {min_len} 步，最長 {max_len} 步)")
    print(f"   共 {len(flows_in_group)} 條路徑")
    
    # 顯示最短和最長路徑
    sorted_by_length = sorted(flows_in_group, key=lambda f: len(f['path']))
    shortest = sorted_by_length[0]
    longest = sorted_by_length[-1]
    
    print(f"   最短路徑 (Flow {shortest['id']}, {len(shortest['path'])} 步):")
    print(f"     {' → '.join(shortest['path'])}")
    print(f"   最長路徑 (Flow {longest['id']}, {len(longest['path'])} 步):")
    longest_path_str = ' → '.join(longest['path'])
    if len(longest_path_str) > 100:
        print(f"     {longest_path_str[:100]}...")
    else:
        print(f"     {longest_path_str}")

# 分析模組終點分佈
print(f"\n🎯 按終點模組分類:")
print("-" * 80)

module_groups = defaultdict(list)
for (start, end), flows_in_group in multi_path_groups.items():
    # 取第一個flow的終點模組
    endpoint_module = flows_in_group[0].get('modules', [])[-1] if flows_in_group[0].get('modules') else 'unknown'
    module_groups[endpoint_module].append(((start, end), len(flows_in_group)))

for module in sorted(module_groups.keys()):
    groups = module_groups[module]
    total_paths = sum(count for _, count in groups)
    print(f"\n{module}:")
    print(f"  {len(groups)} 組多路徑組合，共 {total_paths} 條路徑")
    print(f"  平均每組 {total_paths/len(groups):.1f} 條路徑")

# 典型案例分析
print(f"\n💡 典型案例分析:")
print("-" * 80)

# 找一個中等複雜度的組進行詳細分析
for (start, end), flows_in_group in sorted_groups:
    if 3 <= len(flows_in_group) <= 5 and 3 <= len(flows_in_group[0]['path']) <= 6:
        print(f"\n案例: {start} → {end}")
        print(f"共 {len(flows_in_group)} 條不同路徑\n")
        
        for i, flow in enumerate(flows_in_group, 1):
            path = flow.get('path', [])
            modules = flow.get('modules', [])
            print(f"路徑 {i} (Flow {flow['id']}):")
            print(f"  腳本序列: {' → '.join(path)}")
            print(f"  模組序列: {' → '.join(modules)}")
            print()
        
        print("說明: 相同的起點和終點，系統提供了多種達到目標的方式")
        break

# 總結
print(f"\n{'='*80}")
print("📝 總結:")
print("-" * 80)

total_multi_path_flows = sum(len(v) for v in multi_path_groups.values())
percentage = total_multi_path_flows / len(flows) * 100

print(f"""
1. 多路徑特徵:
   • {len(multi_path_groups)} 組 (起點, 終點) 存在多條路徑
   • 涉及 {total_multi_path_flows} 個 flows ({percentage:.1f}%)
   • 平均每組有 {total_multi_path_flows/len(multi_path_groups):.1f} 條路徑

2. 系統冗餘性:
   • 相同目標可通過不同路徑達成
   • 提供了靈活性和容錯能力
   • 不同路徑可能適用於不同場景

3. 設計意義:
   • 路徑多樣性反映系統的複雜度和靈活性
   • 多條路徑提供了選擇空間
   • 可根據實際情況選擇最優路徑

4. 優化建議:
   • 可以分析哪些路徑最常用
   • 識別冗餘或低效路徑
   • 優化路徑選擇策略
""")

print("="*80)

# 生成詳細報告
report_content = f"""# 多路徑 Flows 分析報告

> **生成時間**: 2026-01-01  
> **分析對象**: 起點終點相同、中間路徑不同的 flows

## 📊 整體統計

| 指標 | 數值 |
|------|------|
| 總 flows 數 | {len(flows)} |
| 唯一的 (起點, 終點) 組合數 | {len(start_end_groups)} |
| 有多條路徑的組合數 | {len(multi_path_groups)} |
| 多路徑 flows 總數 | {total_multi_path_flows} ({percentage:.1f}%) |
| 平均每組路徑數 | {total_multi_path_flows/len(multi_path_groups):.1f} |

## 📈 路徑數量分佈

| 路徑數量 | 組數 |
|---------|------|
"""

for count in sorted(path_count_distribution.keys(), reverse=True):
    num_groups = path_count_distribution[count]
    report_content += f"| {count} 條路徑 | {num_groups} |\n"

report_content += f"""
## 🏆 路徑最多的組合 (Top 10)

"""

for i, ((start, end), flows_in_group) in enumerate(sorted_groups[:10], 1):
    lengths = [len(f['path']) for f in flows_in_group]
    min_len, max_len = min(lengths), max(lengths)
    report_content += f"""
### {i}. {start} → {end}

- **路徑數量**: {len(flows_in_group)} 條
- **路徑長度範圍**: {min_len} - {max_len} 步
- **Flow IDs**: {', '.join(f"flow{f['id']}" for f in flows_in_group[:10])}{'...' if len(flows_in_group) > 10 else ''}

"""

report_content += f"""
## 📏 路徑長度差異最大的組合 (Top 5)

"""

for i, ((start, end), flows_in_group, diff, min_len, max_len) in enumerate(length_diff_groups[:5], 1):
    sorted_by_length = sorted(flows_in_group, key=lambda f: len(f['path']))
    shortest = sorted_by_length[0]
    longest = sorted_by_length[-1]
    
    report_content += f"""
### {i}. {start} → {end}

- **長度差異**: {diff} 步
- **最短路徑**: {min_len} 步 (Flow {shortest['id']})
  ```
  {' → '.join(shortest['path'])}
  ```
- **最長路徑**: {max_len} 步 (Flow {longest['id']})
  ```
  {' → '.join(longest['path'][:10])}{'...' if len(longest['path']) > 10 else ''}
  ```

"""

report_content += f"""
## 💡 設計洞察

### 多路徑的價值

1. **靈活性**: 相同目標可通過不同路徑達成，適應不同場景
2. **容錯性**: 某條路徑失敗時可嘗試其他路徑
3. **優化空間**: 可根據性能、成本等因素選擇最優路徑
4. **功能覆蓋**: 不同路徑可能提供不同的中間功能

### 系統特徵

- **{percentage:.1f}%** 的 flows 屬於多路徑組合
- 平均每組有 **{total_multi_path_flows/len(multi_path_groups):.1f}** 條路徑
- 最多的組有 **{max(len(v) for v in multi_path_groups.values())}** 條不同路徑

### 應用建議

1. **路徑選擇策略**: 開發智能路徑選擇算法
2. **性能優化**: 識別並優化常用路徑
3. **冗餘管理**: 評估是否存在過度冗餘
4. **文檔完善**: 說明不同路徑的適用場景

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍
"""

# 保存報告
report_path = Path("C:/D/fold7/AIVA-git/docs/MULTI_PATH_FLOWS_ANALYSIS.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n✅ 詳細報告已保存至: {report_path}")
print("="*80 + "\n")
