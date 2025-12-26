"""Phase 0 Manual Test - Manifest 架構驗證

測試步驟:
1. 加載 Manifest (ManifestLoader)
2. 構建命令 (CommandBuilder)
3. 預覽參數映射
4. 顯示生成的 CLI 命令

Usage:
    cd C:\D\fold7\AIVA-git
    python services/core/aiva_core/core_capabilities/manifests/test_manifest_system.py
"""

import sys
from pathlib import Path

# 確保可以導入 services
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.core.aiva_core.cognitive_core.manifest import ManifestLoader
from services.core.aiva_core.task_planning.command_builder import CommandBuilder


def test_manifest_loading():
    """測試 1: Manifest 加載"""
    print("=" * 80)
    print("📦 TEST 1: Manifest Loading")
    print("=" * 80)
    
    manifests_dir = Path(__file__).parent / "capabilities"
    loader = ManifestLoader(manifests_dir)
    
    try:
        manifests = loader.load_all()
        print(f"✅ Successfully loaded {len(manifests)} manifests")
        
        # 顯示統計
        stats = loader.get_stats()
        print(f"\n📊 Statistics:")
        print(f"  Total: {stats['total_manifests']}")
        print(f"  Flow ID Range: {stats['flow_id_range']}")
        print(f"  Language: {stats['language_distribution']}")
        print(f"  Module: {stats['module_distribution']}")
        print(f"  Risk: {stats['risk_distribution']}")
        
        # 顯示加載的 Manifest
        for flow_id, manifest in manifests.items():
            print(f"\n  ├─ Flow ID {flow_id}: {manifest.meta.tool_name}")
            print(f"  │  └─ Language: {manifest.meta.language}")
            print(f"  │     Module: {manifest.meta.module}")
            print(f"  │     Command: {manifest.execution.command_template}")
        
        return manifests
        
    except Exception as e:
        print(f"❌ Failed to load manifests: {e}")
        raise


def test_command_building(manifests):
    """測試 2: 命令構建"""
    print("\n" + "=" * 80)
    print("🔨 TEST 2: Command Building")
    print("=" * 80)
    
    builder = CommandBuilder(manifests)
    
    # 測試場景
    scenarios = [
        {
            "name": "低強度查詢 (intensity=0.2)",
            "flow_id": 0,
            "context_data": {"query": "SQL injection detection capabilities"},
            "ai_intensity": 0.2
        },
        {
            "name": "中等強度查詢 (intensity=0.5)",
            "flow_id": 0,
            "context_data": {"query": "XSS attack and detection"},
            "ai_intensity": 0.5
        },
        {
            "name": "高強度查詢 (intensity=0.9)",
            "flow_id": 0,
            "context_data": {"query": "model training and learning"},
            "ai_intensity": 0.9
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"   Context: {scenario['context_data']}")
        print(f"   Intensity: {scenario['ai_intensity']}")
        
        try:
            cmd = builder.build_command(
                flow_id=scenario['flow_id'],
                context_data=scenario['context_data'],
                ai_intensity=scenario['ai_intensity']
            )
            print(f"   ✅ Generated Command:")
            print(f"      {cmd}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def test_parameter_preview(manifests):
    """測試 3: 參數預覽"""
    print("\n" + "=" * 80)
    print("🔍 TEST 3: Parameter Preview")
    print("=" * 80)
    
    builder = CommandBuilder(manifests)
    
    context_data = {"query": "test query"}
    
    try:
        preview = builder.preview_parameters(
            flow_id=0,
            context_data=context_data,
            intensity_samples=[0.0, 0.25, 0.5, 0.75, 1.0]
        )
        
        print(f"\n📝 Manifest: {preview['manifest']['tool_name']}")
        print(f"   Template: {preview['manifest']['command_template']}")
        
        print(f"\n📌 Context Parameters (A-class):")
        for param_name, value in preview['context_params'].items():
            print(f"   {param_name} = {value}")
        
        print(f"\n⚙️ Tunable Parameters (B-class):")
        for param_name, samples in preview['tunable_params'].items():
            print(f"\n   {param_name}:")
            for sample in samples:
                print(f"      intensity={sample['intensity']:.2f} → {sample['value']}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


def main():
    """主測試流程"""
    print("\n" + "🚀" * 40)
    print("  Phase 0 Manual Test - Manifest System Validation")
    print("🚀" * 40 + "\n")
    
    try:
        # 1. 加載 Manifest
        manifests = test_manifest_loading()
        
        if not manifests:
            print("\n❌ No manifests loaded, cannot proceed with tests")
            return
        
        # 2. 測試命令構建
        test_command_building(manifests)
        
        # 3. 測試參數預覽
        test_parameter_preview(manifests)
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        print("\n💡 Next Steps:")
        print("   1. Review the generated commands above")
        print("   2. Pick one command and execute it manually in terminal")
        print("   3. Verify the command works correctly")
        print("   4. If successful, proceed to create more Manifests")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
