"""提取 840 flows 中的函數與腳本對應關係"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

flows_file = PROJECT_ROOT / "services/integration/data/internal_exploration/latest_classification.json"

with open(flows_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

flows = data['flows']

# 提取所有唯一的腳本名和文件路徑對應關係
script_to_file = {}

for flow in flows:
    for script_name, file_path in zip(flow['path'], flow['full_path']):
        if script_name not in script_to_file:
            # 提取相對路徑和模組信息
            script_to_file[script_name] = {
                'full_path': file_path,
                'filename': Path(file_path).name,
                'relative_path': str(Path(file_path).relative_to(PROJECT_ROOT)) if PROJECT_ROOT in Path(file_path).parents else file_path
            }

# 輸出前 20 個
print("腳本名稱 -> 文件映射（前 20 個）：\n")
for i, (script_name, info) in enumerate(list(script_to_file.items())[:20], 1):
    print(f"{i}. {script_name}")
    print(f"   文件: {info['filename']}")
    print(f"   路徑: {info['relative_path']}")
    print()

print(f"\n總計: {len(script_to_file)} 個唯一腳本")

# 保存完整映射
output_file = PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_to_file_mapping.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(script_to_file, f, indent=2, ensure_ascii=False)

print(f"\n已保存完整映射到: {output_file}")
