"""找出可直接測試的純程式組件數據流"""
import json
from pathlib import Path

# 讀取分類數據
data_file = Path(__file__).parent / "classification_results_final" / "classification_data.json"
with open(data_file, encoding='utf-8') as f:
    data = json.load(f)

# 篩選條件:
# 1. 全部是"程式組件" (不依賴AI/RAG/NLP)
# 2. 長度 2-5 (容易測試)
# 3. 排除未完成的 capability_cli

testable_flows = []

for flow in data['flows']:
    # 檢查是否全部為程式組件
    all_prog = all(c['component_type'] == '程式組件' for c in flow['classifications'])
    
    # 長度合適
    suitable_length = 2 <= flow['length'] <= 5
    
    # 排除未完成模組
    excludes = ['capability_cli', 'path_difference_analyzer']
    no_excludes = not any(exc in flow['start'] or exc in flow['end'] for exc in excludes)
    
    if all_prog and suitable_length and no_excludes:
        testable_flows.append(flow)

print(f"✅ 找到 {len(testable_flows)} 條可測試的純程式組件流\n")

# 顯示前15條
for i, flow in enumerate(testable_flows[:15], 1):
    print(f"\n{'='*60}")
    print(f"Flow {flow['id']}: 長度 {flow['length']}")
    print(f"路徑: {' → '.join(flow['path'])}")
    print(f"\n模組分布:")
    for cls in flow['classifications']:
        print(f"  • {cls['script']}")
        print(f"    模組: {cls['module']}")
        print(f"    描述: {cls['description']}")
    
    # 顯示實際檔案路徑
    print(f"\n實際檔案:")
    for path in flow['full_path']:
        print(f"  📄 {Path(path).name}")

print(f"\n{'='*60}")
print(f"總共找到 {len(testable_flows)} 條,已顯示前15條")
