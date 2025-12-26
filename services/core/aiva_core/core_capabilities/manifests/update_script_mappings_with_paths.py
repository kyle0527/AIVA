"""生成完整的 script_mappings.json，包含腳本文件路徑信息"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

# 載入腳本到文件的映射
script_file_mapping_path = PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_to_file_mapping.json"
with open(script_file_mapping_path, 'r', encoding='utf-8') as f:
    script_to_file = json.load(f)

# 載入現有的 script_mappings.json
current_mappings_path = PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_mappings.json"
with open(current_mappings_path, 'r', encoding='utf-8') as f:
    current_mappings = json.load(f)

# 更新每個映射，添加腳本文件路徑信息
updated_mappings = {}

for script_name, mapping in current_mappings.items():
    if script_name in script_to_file:
        file_info = script_to_file[script_name]
        
        # 更新映射
        updated_mapping = {
            "script_name": script_name,
            "script_file": file_info['filename'],
            "script_path": file_info['relative_path'].replace('\\', '/'),
            **mapping
        }
        
        updated_mappings[script_name] = updated_mapping
    else:
        # 保留原有映射
        updated_mappings[script_name] = mapping

# 保存更新後的映射
with open(current_mappings_path, 'w', encoding='utf-8') as f:
    json.dump(updated_mappings, f, indent=2, ensure_ascii=False)

print(f"✅ 已更新 script_mappings.json")
print(f"   包含 {len(updated_mappings)} 個腳本映射")
print(f"   每個映射現在包含:")
print(f"   - script_name: 腳本名稱")
print(f"   - script_file: 文件名")
print(f"   - script_path: 相對路徑")
print(f"   - module: Python 模組路徑")
print(f"   - class/function: 類/函數名")
print(f"   - method: 方法名")

# 顯示更新的示例
print("\n示例（monitoring）:")
print(json.dumps(updated_mappings.get('monitoring', {}), indent=2, ensure_ascii=False))
