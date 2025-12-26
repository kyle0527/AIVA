"""驗證重現指南的完整性和可執行性"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

print("=" * 80)
print("🔍 驗證重現指南的可執行性")
print("=" * 80)
print()

# 檢查必需的數據源
print("📁 檢查數據源...")
print()

required_files = {
    "840 Flows 數據源": PROJECT_ROOT / "services/integration/data/internal_exploration/latest_classification.json",
}

optional_files = {
    "腳本映射": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_mappings.json",
    "腳本路徑映射": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_to_file_mapping.json",
    "測試 Flows": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/test_flows_10.json",
    "執行器": PROJECT_ROOT / "aiva_flow_runner.py",
}

tools = {
    "提取腳本映射工具": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/extract_script_file_mapping.py",
    "選擇測試 Flows 工具": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/select_test_flows.py",
    "分析修正工具": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/analyze_and_fix_mappings.py",
    "統計分析工具": PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/analyze_cli_capabilities.py",
}

all_ready = True

# 檢查必需文件
print("1️⃣ 必需數據源:")
for name, path in required_files.items():
    if path.exists():
        print(f"   ✅ {name}")
        print(f"      路徑: {path.relative_to(PROJECT_ROOT)}")
        
        # 檢查內容
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'flows' in data:
                print(f"      包含: {len(data['flows'])} 個 flows")
    else:
        print(f"   ❌ {name} - 不存在")
        print(f"      路徑: {path.relative_to(PROJECT_ROOT)}")
        all_ready = False

print()

# 檢查可選文件（中間產物）
print("2️⃣ 可選文件（可生成）:")
for name, path in optional_files.items():
    if path.exists():
        print(f"   ✅ {name} - 已存在")
        
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"      包含: {len(data)} 項")
            elif isinstance(data, list):
                print(f"      包含: {len(data)} 項")
    else:
        print(f"   ⚠️  {name} - 不存在（可生成）")

print()

# 檢查工具腳本
print("3️⃣ 工具腳本:")
for name, path in tools.items():
    if path.exists():
        print(f"   ✅ {name}")
    else:
        print(f"   ❌ {name} - 不存在")
        all_ready = False

print()
print("=" * 80)

if all_ready:
    print("✅ 所有必需文件和工具都存在")
    print()
    print("🚀 可以開始重現流程:")
    print()
    print("步驟 1: 提取腳本路徑映射")
    print("   cd services\\core\\aiva_core\\core_capabilities\\manifests")
    print("   python extract_script_file_mapping.py")
    print()
    print("步驟 2: 選擇測試 Flows")
    print("   python select_test_flows.py")
    print()
    print("步驟 3: 創建初始映射（如不存在）")
    print("   # 手動創建或使用現有模板")
    print()
    print("步驟 4: 分析並修正映射")
    print("   python analyze_and_fix_mappings.py")
    print()
    print("步驟 5: 測試執行")
    print("   cd C:\\D\\fold7\\AIVA-git")
    print("   python aiva_flow_runner.py --test")
    print()
    print("步驟 6: 統計能力")
    print("   cd services\\core\\aiva_core\\core_capabilities\\manifests")
    print("   python analyze_cli_capabilities.py")
else:
    print("❌ 缺少必需文件，無法完整重現")
    print()
    print("請確保以下文件存在:")
    print("  - services/integration/data/internal_exploration/latest_classification.json")
    print("  - 所有工具腳本")

print()
print("=" * 80)
print("📋 當前狀態檢查")
print("=" * 80)
print()

# 檢查 latest_classification.json 的詳細信息
flows_file = PROJECT_ROOT / "services/integration/data/internal_exploration/latest_classification.json"
if flows_file.exists():
    with open(flows_file, 'r', encoding='utf-8') as f:
        flows_data = json.load(f)
    
    print("📊 840 Flows 數據詳情:")
    print(f"   總 Flows: {len(flows_data['flows'])}")
    
    if 'metadata' in flows_data:
        meta = flows_data['metadata']
        print(f"   生成時間: {meta.get('generated_at', 'N/A')}")
        if 'module_distribution' in meta:
            print(f"   模組分布:")
            for module, count in meta['module_distribution'].items():
                print(f"      - {module}: {count}")
    
    # 統計唯一腳本
    unique_scripts = set()
    for flow in flows_data['flows']:
        unique_scripts.update(flow['path'])
    
    print(f"   唯一腳本數: {len(unique_scripts)}")
    
    # 統計路徑長度
    from collections import Counter
    lengths = Counter([f['length'] for f in flows_data['flows']])
    print(f"   路徑長度分布:")
    for length in sorted(lengths.keys()):
        count = lengths[length]
        print(f"      - {length} 個腳本: {count} 個 flows ({count/len(flows_data['flows'])*100:.1f}%)")

print()
print("=" * 80)

# 檢查現有映射狀態
mappings_file = PROJECT_ROOT / "services/core/aiva_core/core_capabilities/manifests/script_mappings.json"
if mappings_file.exists():
    print("📋 現有映射狀態:")
    with open(mappings_file, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    mock_count = sum(1 for m in mappings.values() if m.get('mock', True))
    real_count = len(mappings) - mock_count
    
    print(f"   總映射數: {len(mappings)}")
    print(f"   真實執行: {real_count} ({real_count/len(mappings)*100:.1f}%)")
    print(f"   Mock: {mock_count} ({mock_count/len(mappings)*100:.1f}%)")
else:
    print("📋 映射文件不存在，需要創建")

print()
print("=" * 80)
print("✅ 驗證完成")
print("=" * 80)
