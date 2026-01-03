#!/usr/bin/env python3
"""
檢查路徑格式問題
"""

import json
from pathlib import Path

# 數據路徑
data_dir = Path("C:/Users/User/Downloads/data/internal_exploration")
json_file = data_dir / "latest_classification.json"

# 讀取數據
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data.get('flows', [])

# 檢查前5個 unknown flows
print("檢查 unknown flows 的路徑格式:\n")

unknown_count = 0
for flow in flows:
    modules = flow.get('modules', [])
    if modules and modules[-1] == 'unknown':
        unknown_count += 1
        if unknown_count <= 5:
            print(f"Flow {flow['id']}:")
            full_paths = flow.get('full_path', [])
            if isinstance(full_paths, list):
                for i, path in enumerate(full_paths):
                    print(f"  Step {i+1}: {path}")
                    # 檢查路徑是否包含反斜杠
                    if '\\' in path:
                        print(f"    ⚠️ 包含 Windows 反斜杠: {path.count('\\\\')} 個")
                        normalized = path.replace('\\', '/')
                        print(f"    標準化後: {normalized}")
                        # 檢查是否包含模組名稱
                        for module in ['cognitive_core', 'internal_exploration', 'task_planning', 
                                      'external_learning', 'core_capabilities', 'service_backbone']:
                            if f'/{module}/' in normalized:
                                print(f"    ✅ 標準化後包含模組: {module}")
            print()

print(f"\n總共 {unknown_count} 個 unknown flows")
