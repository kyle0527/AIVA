#!/usr/bin/env python3
"""
分析各模組中未被 flows 覆蓋的能力
找出哪些腳本文件不在任何 flow 的終點，分析這些「隱藏」能力
"""

import json
from pathlib import Path
from collections import defaultdict
import os

# 數據路徑
data_dir = Path("C:/Users/User/Downloads/data/internal_exploration")
json_file = data_dir / "latest_classification.json"

# 讀取數據
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data.get('flows', [])

# AIVA 核心路徑
aiva_core_root = Path("C:/D/fold7/AIVA-git/services/core/aiva_core")

# 六大模組
modules = [
    'cognitive_core',
    'internal_exploration',
    'task_planning',
    'external_learning',
    'core_capabilities',
    'service_backbone'
]

print("\n" + "="*90)
print("🔍 各模組未被 Flows 覆蓋的能力分析")
print("="*90)

# 收集 flows 中出現的終點腳本
flow_endpoint_scripts = defaultdict(set)

for flow in flows:
    full_paths = flow.get('full_path', [])
    modules_list = flow.get('modules', [])
    
    if full_paths and modules_list:
        endpoint_path = full_paths[-1] if isinstance(full_paths, list) else str(full_paths)
        endpoint_module = modules_list[-1]
        
        # 提取腳本名稱
        script_name = endpoint_path.split('/')[-1] if '/' in endpoint_path else endpoint_path.split('\\')[-1]
        script_name = script_name.replace('.py', '')
        
        flow_endpoint_scripts[endpoint_module].add(script_name)

# 掃描各模組的所有 Python 文件
print("\n📊 各模組能力覆蓋率分析:\n")

all_uncovered = {}

for module in modules:
    module_path = aiva_core_root / module
    
    if not module_path.exists():
        print(f"⚠️  模組路徑不存在: {module_path}")
        continue
    
    # 收集所有 .py 文件
    all_scripts = set()
    for root, dirs, files in os.walk(module_path):
        # 排除 __pycache__ 等目錄
        dirs[:] = [d for d in dirs if not d.startswith('__') and d != '.git']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                script_name = file.replace('.py', '')
                all_scripts.add(script_name)
    
    # 找出未被覆蓋的腳本
    covered_scripts = flow_endpoint_scripts.get(module, set())
    uncovered_scripts = all_scripts - covered_scripts
    
    coverage_rate = len(covered_scripts) / len(all_scripts) * 100 if all_scripts else 0
    
    all_uncovered[module] = {
        'total': len(all_scripts),
        'covered': len(covered_scripts),
        'uncovered': len(uncovered_scripts),
        'coverage_rate': coverage_rate,
        'uncovered_list': sorted(uncovered_scripts)
    }
    
    print(f"{'='*90}")
    print(f"📦 {module}")
    print(f"   總腳本數: {len(all_scripts)}")
    print(f"   被 Flows 覆蓋: {len(covered_scripts)} ({coverage_rate:.1f}%)")
    print(f"   未被覆蓋: {len(uncovered_scripts)} ({100-coverage_rate:.1f}%)")
    print()

# 詳細列出未覆蓋的腳本
print("="*90)
print("🔍 未被覆蓋的能力詳細列表:\n")

for module in modules:
    info = all_uncovered.get(module)
    if not info or not info['uncovered_list']:
        continue
    
    print(f"{'='*90}")
    print(f"📦 {module} - {info['uncovered']} 個未覆蓋腳本\n")
    
    module_path = aiva_core_root / module
    
    # 列出並分析每個未覆蓋的腳本
    for i, script in enumerate(info['uncovered_list'][:20], 1):  # 只顯示前20個
        # 找到腳本文件
        script_file = None
        for root, dirs, files in os.walk(module_path):
            for file in files:
                if file == f"{script}.py":
                    script_file = Path(root) / file
                    break
            if script_file:
                break
        
        if script_file and script_file.exists():
            # 讀取前幾行以獲取描述
            try:
                with open(script_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:50]  # 讀取前50行
                
                # 嘗試提取文檔字符串或註釋
                description = "無描述"
                in_docstring = False
                docstring_lines = []
                
                for line in lines:
                    line_strip = line.strip()
                    
                    # 檢查文檔字符串
                    if '"""' in line_strip or "'''" in line_strip:
                        if not in_docstring:
                            in_docstring = True
                            # 提取同行內容
                            if line_strip.count('"""') == 2 or line_strip.count("'''") == 2:
                                # 單行文檔字符串
                                description = line_strip.strip('"""').strip("'''").strip()
                                break
                            else:
                                docstring_lines.append(line_strip.replace('"""', '').replace("'''", ''))
                        else:
                            # 結束文檔字符串
                            docstring_lines.append(line_strip.replace('"""', '').replace("'''", ''))
                            description = ' '.join(docstring_lines).strip()
                            break
                    elif in_docstring:
                        docstring_lines.append(line_strip)
                    
                    # 檢查類定義
                    if line_strip.startswith('class '):
                        if not description or description == "無描述":
                            description = f"類: {line_strip.replace('class ', '').split('(')[0].split(':')[0].strip()}"
                        break
                    
                    # 檢查函數定義
                    if line_strip.startswith('def ') and 'main' in line_strip:
                        if not description or description == "無描述":
                            description = "主函數入口"
                        break
                
                # 限制描述長度
                if len(description) > 80:
                    description = description[:77] + "..."
                
                # 獲取相對路徑
                rel_path = script_file.relative_to(module_path)
                
                print(f"   {i:2}. {script:40} ({rel_path.parent})")
                print(f"       {description}")
                
            except Exception as e:
                print(f"   {i:2}. {script:40} (無法讀取文件: {e})")
        else:
            print(f"   {i:2}. {script:40} (文件未找到)")
        
        print()
    
    if len(info['uncovered_list']) > 20:
        print(f"   ... 還有 {len(info['uncovered_list']) - 20} 個未列出")
        print()

# 統計匯總
print("="*90)
print("📈 整體統計:\n")

total_scripts = sum(info['total'] for info in all_uncovered.values())
total_covered = sum(info['covered'] for info in all_uncovered.values())
total_uncovered = sum(info['uncovered'] for info in all_uncovered.values())
overall_coverage = total_covered / total_scripts * 100 if total_scripts else 0

print(f"總腳本數: {total_scripts}")
print(f"被 Flows 覆蓋: {total_covered} ({overall_coverage:.1f}%)")
print(f"未被覆蓋: {total_uncovered} ({100-overall_coverage:.1f}%)")
print()

# 按覆蓋率排序
print("按覆蓋率排序:")
sorted_modules = sorted(all_uncovered.items(), key=lambda x: x[1]['coverage_rate'], reverse=True)
for module, info in sorted_modules:
    print(f"  {module:25}: {info['coverage_rate']:5.1f}% ({info['covered']}/{info['total']})")

print("\n" + "="*90)
print("💡 關鍵發現:\n")

# 找出覆蓋率最高和最低的模組
best_module = max(all_uncovered.items(), key=lambda x: x[1]['coverage_rate'])
worst_module = min(all_uncovered.items(), key=lambda x: x[1]['coverage_rate'])

print(f"""
1. 覆蓋率分析:
   • 整體覆蓋率: {overall_coverage:.1f}%
   • 最高覆蓋率: {best_module[0]} ({best_module[1]['coverage_rate']:.1f}%)
   • 最低覆蓋率: {worst_module[0]} ({worst_module[1]['coverage_rate']:.1f}%)

2. 未覆蓋腳本特徵:
   • 可能是內部輔助工具
   • 可能是測試或示例代碼
   • 可能是尚未整合到流程中的新功能
   • 可能是底層基礎設施（不作為終點）

3. 影響評估:
   • {total_uncovered} 個腳本未被 flows 直接調用
   • 這些可能是:
     - 被其他腳本間接調用的模組
     - 獨立的工具腳本
     - 配置或初始化文件
     - 廢棄或待刪除的代碼

4. 建議行動:
   • 審查未覆蓋腳本的必要性
   • 考慮是否需要創建 flows 來調用它們
   • 識別並清理廢棄代碼
   • 文檔化獨立工具的使用方式
""")

print("="*90)

# 生成報告
report_content = f"""# 各模組未被 Flows 覆蓋的能力分析報告

> **生成時間**: 2026-01-01  
> **分析對象**: 六大模組中的所有 Python 腳本

## 📊 整體統計

| 指標 | 數值 |
|------|------|
| 總腳本數 | {total_scripts} |
| 被 Flows 覆蓋 | {total_covered} ({overall_coverage:.1f}%) |
| 未被覆蓋 | {total_uncovered} ({100-overall_coverage:.1f}%) |

## 📈 各模組覆蓋率

| 模組 | 總腳本數 | 已覆蓋 | 未覆蓋 | 覆蓋率 |
|------|----------|--------|--------|--------|
"""

for module, info in sorted_modules:
    report_content += f"| {module} | {info['total']} | {info['covered']} | {info['uncovered']} | {info['coverage_rate']:.1f}% |\n"

report_content += f"""

## 🔍 未被覆蓋的能力詳細

"""

for module in modules:
    info = all_uncovered.get(module)
    if not info or not info['uncovered_list']:
        continue
    
    report_content += f"""
### {module}

**未覆蓋腳本**: {info['uncovered']} 個

"""
    
    for script in info['uncovered_list'][:15]:  # 報告中只列出前15個
        report_content += f"- `{script}.py`\n"
    
    if len(info['uncovered_list']) > 15:
        report_content += f"\n... 還有 {len(info['uncovered_list']) - 15} 個\n"

report_content += f"""

## 💡 分析與建議

### 1. 覆蓋率評估

- **整體覆蓋率**: {overall_coverage:.1f}%
- **最高覆蓋率**: {best_module[0]} ({best_module[1]['coverage_rate']:.1f}%)
- **最低覆蓋率**: {worst_module[0]} ({worst_module[1]['coverage_rate']:.1f}%)

### 2. 未覆蓋腳本性質

未被 flows 覆蓋的 {total_uncovered} 個腳本可能屬於以下類別：

1. **內部模組**: 被其他腳本導入使用，不作為獨立入口
2. **工具腳本**: 獨立執行的工具，不通過 flow 調用
3. **測試代碼**: 測試文件或示例代碼
4. **基礎設施**: 配置、初始化等基礎組件
5. **待整合**: 新開發尚未整合到流程中的功能
6. **廢棄代碼**: 可能需要清理的舊代碼

### 3. 建議行動

**立即行動**:
- ✅ 審查未覆蓋腳本列表
- ✅ 識別哪些是必要的內部模組
- ✅ 確定哪些是廢棄代碼

**短期計劃**:
- 🔧 為需要的獨立工具創建對應 flows
- 📝 文檔化工具腳本的使用方式
- 🗑️ 清理確認廢棄的代碼

**長期規劃**:
- 🎯 提升整體覆蓋率到 80%+
- 📚 建立代碼審查機制
- 🔄 定期更新能力地圖

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍
"""

# 保存報告
report_path = Path("C:/D/fold7/AIVA-git/docs/UNCOVERED_CAPABILITIES_ANALYSIS.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n✅ 詳細報告已保存至: {report_path}")
print("="*90 + "\n")
