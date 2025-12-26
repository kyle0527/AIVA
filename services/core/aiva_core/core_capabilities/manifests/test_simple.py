"""Phase 0 Manual Test - Manifest 架構驗證 (獨立版本)

避免複雜導入依賴，直接測試核心功能。

Usage:
    cd C:\\D\\fold7\\AIVA-git
    python services/core/aiva_core/core_capabilities/manifests/test_simple.py
"""

import json
from pathlib import Path
import sys

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_manifest_json_loading():
    """測試 1: 直接加載 JSON 文件"""
    print("=" * 80)
    print("📦 TEST 1: Direct JSON Loading")
    print("=" * 80)
    
    manifest_file = Path(__file__).parent / "capabilities" / "00_internal_loop_connector.json"
    
    if not manifest_file.exists():
        print(f"❌ Manifest file not found: {manifest_file}")
        return None
    
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded JSON")
        print(f"\n📋 Manifest Content:")
        print(f"  Flow ID: {data['meta']['flow_id']}")
        print(f"  Tool Name: {data['meta']['tool_name']}")
        print(f"  Language: {data['meta']['language']}")
        print(f"  Module: {data['meta']['module']}")
        print(f"\n🎯 AI Cognitive:")
        print(f"  Primary Capability: {data['ai_cognitive']['primary_capability']}")
        print(f"  Risk Level: {data['ai_cognitive']['risk_level']}")
        print(f"  Required Tags: {data['ai_cognitive']['tags']['required']}")
        print(f"\n⚙️ Execution:")
        print(f"  Template: {data['execution']['command_template']}")
        print(f"\n📊 Parameters:")
        print(f"  Context Mapping (A-class): {data['parameters']['context_mapping']}")
        print(f"  Tunable Parameters (B-class): {list(data['parameters']['tunable'].keys())}")
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_parameter_mapping(manifest_data):
    """測試 2: 參數映射邏輯 (手動實現)"""
    print("\n" + "=" * 80)
    print("🔨 TEST 2: Parameter Mapping (Manual Implementation)")
    print("=" * 80)
    
    if not manifest_data:
        print("❌ No manifest data to test")
        return
    
    # 測試場景
    scenarios = [
        {"name": "低強度", "intensity": 0.2, "query": "SQL injection"},
        {"name": "中等強度", "intensity": 0.5, "query": "XSS attack"},
        {"name": "高強度", "intensity": 0.9, "query": "model training"}
    ]
    
    template = manifest_data['execution']['command_template']
    context_mapping = manifest_data['parameters']['context_mapping']
    tunable = manifest_data['parameters']['tunable']
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']} (intensity={scenario['intensity']})")
        
        # A類參數 (從 context)
        context_data = {"query": scenario['query']}
        params = {}
        for param_name, context_key in context_mapping.items():
            params[param_name] = context_data[context_key]
        
        # B類參數 (從 intensity 映射)
        for param_name, param_def in tunable.items():
            intensity = scenario['intensity']
            min_val = param_def['min']
            max_val = param_def['max']
            strategy = param_def['mapping_strategy']
            
            if strategy == "linear":
                value = min_val + intensity * (max_val - min_val)
            else:
                value = min_val  # 其他策略暫不實現
            
            params[param_name] = int(round(value))
        
        # 生成命令
        try:
            command = template.format(**params)
            print(f"   Query: '{context_data['query']}'")
            print(f"   Result Count: {params.get('result_count', 'N/A')}")
            print(f"   ✅ Generated Command:")
            print(f"      {command}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def test_json_structure_validation():
    """測試 3: JSON 結構驗證"""
    print("\n" + "=" * 80)
    print("🔍 TEST 3: JSON Structure Validation")
    print("=" * 80)
    
    manifest_file = Path(__file__).parent / "capabilities" / "00_internal_loop_connector.json"
    
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 檢查必需字段
        required_fields = {
            "meta": ["flow_id", "tool_name", "description", "language", "version"],
            "ai_cognitive": ["primary_capability", "risk_level", "tags"],
            "execution": ["binary_path", "command_template", "working_dir"],
            "parameters": ["context_mapping", "tunable"]
        }
        
        errors = []
        for section, fields in required_fields.items():
            if section not in data:
                errors.append(f"Missing section: {section}")
                continue
            
            for field in fields:
                if field not in data[section]:
                    errors.append(f"Missing field: {section}.{field}")
        
        if errors:
            print("❌ Validation Errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ All required fields present")
            print("\n📊 Structure Summary:")
            print(f"   Sections: {list(data.keys())}")
            print(f"   Examples: {len(data.get('examples', []))} scenarios")
            print(f"   Dependencies: {len(data.get('dependencies', []))}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")


def main():
    """主測試流程"""
    print("\n" + "🚀" * 40)
    print("  Phase 0 Manual Test - Simple Validation")
    print("🚀" * 40 + "\n")
    
    print("💡 This is a simplified test that avoids complex imports")
    print("   It directly tests JSON loading and parameter mapping logic\n")
    
    # 1. 加載 JSON
    manifest_data = test_manifest_json_loading()
    
    if not manifest_data:
        print("\n❌ Cannot proceed without manifest data")
        sys.exit(1)
    
    # 2. 測試參數映射
    test_parameter_mapping(manifest_data)
    
    # 3. 驗證結構
    test_json_structure_validation()
    
    print("\n" + "=" * 80)
    print("✅ ALL SIMPLE TESTS COMPLETED")
    print("=" * 80)
    
    print("\n💡 Next Steps:")
    print("   1. Review the generated commands above")
    print("   2. Manually execute one command in terminal:")
    print("      cd C:\\D\\fold7\\AIVA-git")
    print("      python -m services.core.aiva_core.cognitive_core.internal_loop_connector query --text 'SQL injection' --top-k 5")
    print("   3. If successful, the Manifest architecture is validated!")


if __name__ == "__main__":
    main()
