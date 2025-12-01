#!/usr/bin/env python3
"""測試權重載入功能"""
import sys
from pathlib import Path

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))

def test_realaicore_loading():
    """測試 RealAICore 權重載入"""
    print("\n" + "="*80)
    print("測試 RealAICore 權重載入")
    print("="*80)
    
    try:
        from services.core.aiva_core.cognitive_core.neural.real_neural_core import RealAICore
        import torch
        
        # 創建模型
        model = RealAICore(
            input_size=512,
            output_size=100,
            use_5m_model=True,
            weights_path="models/weights/aiva_real_weights.pth"
        )
        
        print("✅ RealAICore 實例化成功")
        print(f"   總參數: {model.total_params:,} ({model.total_params/1_000_000:.2f}M)")
        
        # 載入權重
        model.load_weights()
        print("✅ 權重載入成功")
        
        # 測試前向傳播
        test_input = torch.randn(1, 512)
        output = model(test_input)
        print(f"✅ 前向傳播成功: 輸入 {tuple(test_input.shape)} -> 輸出 {tuple(output.shape)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decision_engine_loading():
    """測試 RealDecisionEngine 權重載入"""
    print("\n" + "="*80)
    print("測試 RealDecisionEngine 權重載入")
    print("="*80)
    
    try:
        from services.core.aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine
        import torch
        
        # 創建決策引擎
        engine = RealDecisionEngine(
            weights_path="models/weights/aiva_real_weights.pth",
            use_5m_model=True
        )
        
        print("✅ RealDecisionEngine 實例化成功")
        
        # 測試決策
        test_input = torch.randn(1, 512)
        decision = engine.decide(test_input)
        print(f"✅ 決策測試成功")
        print(f"   決策輸出形狀: {tuple(decision.shape)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bionet_loading():
    """測試 BioNet 權重載入"""
    print("\n" + "="*80)
    print("測試 RealScalableBioNet 權重載入")
    print("="*80)
    
    try:
        from services.core.aiva_core.cognitive_core.neural.real_bio_net_adapter import create_real_scalable_bionet
        
        # 創建 BioNet
        bionet = create_real_scalable_bionet(
            input_size=512,
            num_tools=20,
            weights_path="weights/models/aiva_scalable_bionet_latest.pth"
        )
        
        print("✅ RealScalableBioNet 實例化成功")
        print(f"   輸入維度: 512")
        print(f"   工具數量: 20")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AIVA 權重載入功能測試")
    print("="*80)
    
    results = []
    
    # 測試 1: RealAICore
    results.append(("RealAICore", test_realaicore_loading()))
    
    # 測試 2: RealDecisionEngine
    results.append(("RealDecisionEngine", test_decision_engine_loading()))
    
    # 測試 3: RealScalableBioNet
    results.append(("RealScalableBioNet", test_bionet_loading()))
    
    # 總結
    print("\n" + "="*80)
    print("測試總結")
    print("="*80)
    
    for name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{name:30s} {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n通過率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    sys.exit(0 if passed == total else 1)
