#!/usr/bin/env python3
"""
快速測試 CLI 和 AI 訓練整合

此腳本會:
1. 測試 CLI 命令結構
2. 驗證 500 萬參數 BioNeuronCore
3. 運行簡單的訓練場景
4. 顯示訓練統計
"""

import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"項目根目錄: {project_root}")
print(f"Python 路徑: {sys.path[0]}")


def test_cli_structure():
    """測試 CLI 結構"""
    print("="*60)
    print("測試 1: CLI 命令結構")
    print("="*60)
    
    try:
        from services.cli import aiva_cli
        parser = aiva_cli.create_parser()
        
        print("✅ CLI 解析器創建成功")
        
        # 測試命令
        test_commands = [
            ["scan", "start", "https://example.com"],
            ["detect", "sqli", "https://example.com", "--param", "id"],
            ["ai", "status"],
        ]
        
        for cmd in test_commands:
            try:
                _ = parser.parse_args(cmd)
                print(f"✅ 命令解析成功: {' '.join(cmd)}")
            except SystemExit:
                print(f"⚠️ 命令需要額外參數: {' '.join(cmd)}")
                pass
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ CLI 測試失敗: {e}")
        return False


def test_bio_neuron_params():
    """測試 BioNeuronCore 參數量"""
    print("="*60)
    print("測試 2: BioNeuronCore 參數量")
    print("="*60)
    
    try:
        from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
        
        # 創建 500 萬參數模型
        net = ScalableBioNet(
            input_dim=512,
            hidden_dims=[1024, 2048, 1024],
            output_dim=256,
        )
        
        param_count = net.count_params()
        
        print(f"神經網路架構:")
        print(f"  輸入層: 512")
        print(f"  隱藏層 1: 1024")
        print(f"  隱藏層 2: 2048")
        print(f"  隱藏層 3: 1024")
        print(f"  輸出層: 256")
        print(f"\n總參數量: {param_count:,}")
        
        # 驗證參數量
        expected_params = (
            512 * 1024 +      # Layer 1
            1024 * 2048 +     # Layer 2
            2048 * 1024 +     # Layer 3
            1024 * 256        # Layer 4
        )
        
        if param_count == expected_params:
            print(f"✅ 參數量正確 (預期: {expected_params:,})")
        else:
            print(f"⚠️ 參數量不符 (預期: {expected_params:,}, 實際: {param_count:,})")
        
        # 測試前向傳播
        import numpy as np
        test_input = np.random.randn(512)
        output = net.forward(test_input)
        
        print(f"\n前向傳播測試:")
        print(f"  輸入形狀: {test_input.shape}")
        print(f"  輸出形狀: {output.shape}")
        print(f"✅ 前向傳播成功")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ BioNeuron 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_training_components():
    """測試訓練組件"""
    print("="*60)
    print("測試 3: AI 訓練組件")
    print("="*60)
    
    try:
        from scripts.ai_training.integrated_cli_training import (
            AITrainingOrchestrator,
        )
        
        # 創建訓練編排器
        print("創建 AITrainingOrchestrator...")
        orchestrator = AITrainingOrchestrator(
            storage_path=Path("./data/test_ai")
        )
        
        print("✅ 訓練編排器創建成功")
        
        # 初始化
        print("初始化訓練系統...")
        await orchestrator.initialize()
        print("✅ 訓練系統初始化成功")
        
        # 獲取統計
        stats = await orchestrator.get_training_stats()
        print(f"\n訓練系統統計:")
        print(f"  模型參數量: {stats['model_params']:,}")
        print(f"  經驗條數: {stats['experiences_count']}")
        print(f"  知識庫條目: {stats['knowledge_entries']}")
        print(f"  最後更新: {stats['last_update']}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 訓練組件測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_training():
    """測試簡單訓練場景"""
    print("="*60)
    print("測試 4: 簡單訓練場景")
    print("="*60)
    
    try:
        # 直接從當前目錄匯入
        from integrated_cli_training import AITrainingOrchestrator
        
        print("開始訓練 (2 個場景, 1 輪)...")
        orchestrator = AITrainingOrchestrator(
            storage_path=Path("./data/test_ai")
        )
        
        await orchestrator.initialize()
        
        # 運行簡短訓練
        await orchestrator.train_from_simulations(
            num_scenarios=2,
            epochs=1,
        )
        
        # 獲取最終統計
        stats = await orchestrator.get_training_stats()
        print(f"\n訓練後統計:")
        print(f"  經驗條數: {stats['experiences_count']}")
        print(f"  知識庫條目: {stats['knowledge_entries']}")
        
        print("\n✅ 訓練測試完成")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 訓練測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_summary(results: dict[str, bool]):
    """顯示測試總結"""
    print("="*60)
    print("測試總結")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    print(f"\n總計: {passed}/{total} 測試通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！系統已就緒。")
        print("\n下一步:")
        print("  1. 運行完整訓練:")
        print("     python scripts/ai_training/integrated_cli_training.py")
        print("  2. 使用 CLI 命令:")
        print("     python services/cli/aiva_cli.py --help")
    else:
        print("\n⚠️ 部分測試失敗，請檢查錯誤訊息。")


async def main():
    """主測試函數"""
    print("\n" + "="*60)
    print("AIVA CLI 和 AI 訓練整合測試")
    print("="*60 + "\n")
    
    results = {}
    
    # 測試 1: CLI 結構
    results["CLI 命令結構"] = test_cli_structure()
    
    # 測試 2: BioNeuron 參數
    results["BioNeuron 參數量"] = test_bio_neuron_params()
    
    # 測試 3: 訓練組件
    results["AI 訓練組件"] = await test_training_components()
    
    # 測試 4: 簡單訓練
    if results["AI 訓練組件"]:
        results["簡單訓練場景"] = await test_simple_training()
    else:
        print("⏭️ 跳過訓練測試 (組件初始化失敗)\n")
        results["簡單訓練場景"] = False
    
    # 顯示總結
    display_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
