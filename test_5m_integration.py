#!/usr/bin/env python3
"""
測試5M特化神經網路整合
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# 添加服務路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'services' / 'core'))

try:
    from aiva_core.ai_engine.real_neural_core import RealAICore, RealDecisionEngine
    print("✅ 成功導入5M AI核心模組")
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    
    # 嘗試直接導入
    try:
        sys.path.insert(0, str(current_dir / 'services' / 'core' / 'aiva_core' / 'ai_engine'))
        from real_neural_core import RealAICore, RealDecisionEngine
        print("✅ 直接導入5M AI核心模組成功")
    except ImportError as e2:
        print(f"❌ 直接導入也失敗: {e2}")
        sys.exit(1)

def test_5m_model():
    """測試5M特化神經網路"""
    print("\n🔬 測試5M特化神經網路...")
    
    # 檢查權重檔案
    weights_path = "services/core/aiva_core/ai_engine/aiva_5M_weights.pth"
    if not Path(weights_path).exists():
        print(f"❌ 權重檔案不存在: {weights_path}")
        return False
    
    # 創建5M模型實例
    try:
        ai_core = RealAICore(
            input_size=512,
            output_size=100,
            aux_output_size=531,
            use_5m_model=True,
            weights_path=weights_path
        )
        print("✅ 5M AI核心創建成功")
        
        # 載入權重
        ai_core.load_weights()
        print("✅ 5M模型權重載入成功")
        
        # 測試前向推理
        test_input = torch.randn(1, 512)  # 批次大小1, 512維輸入
        
        with torch.no_grad():
            # 測試主輸出
            main_output = ai_core.forward(test_input)
            print(f"✅ 主輸出形狀: {main_output.shape} (預期: [1, 100])")
            
            # 測試雙輸出
            main_out, aux_out = ai_core.forward_with_aux(test_input)
            print(f"✅ 雙輸出形狀: 主{main_out.shape}, 輔{aux_out.shape}")
            
            # 決策測試
            decision = torch.argmax(main_output, dim=1).item()
            confidence = torch.max(torch.softmax(main_output, dim=1)).item()
            
            print(f"🎯 決策結果: {decision} (信心度: {confidence:.3f})")
            
        return True
        
    except Exception as e:
        print(f"❌ 5M模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decision_engine():
    """測試5M決策引擎"""
    print("\n🧠 測試5M決策引擎...")
    
    try:
        # 創建5M決策引擎
        engine = RealDecisionEngine(
            weights_path="services/core/aiva_core/ai_engine/aiva_5M_weights.pth",
            use_5m_model=True
        )
        print("✅ 5M決策引擎創建成功")
        
        # 載入權重
        engine.ai_core.load_weights()
        print("✅ 決策引擎權重載入成功")
        
        # 測試決策生成
        test_context = {
            "target_info": "test_target",
            "tools_available": ["nmap", "nikto", "sqlmap"],
            "scan_results": {"ports": [80, 443], "services": ["http", "https"]}
        }
        
        decision = engine.generate_decision("選擇最佳攻擊工具", test_context)
        print(f"✅ 決策生成成功: {decision['decision']}")
        print(f"   信心度: {decision['confidence']:.3f}")
        print(f"   推理: {decision['reasoning'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 決策引擎測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility():
    """測試向後兼容性"""
    print("\n🔄 測試向後兼容性...")
    
    try:
        # 測試原始架構
        legacy_core = RealAICore(
            input_size=512,
            output_size=128,
            use_5m_model=False
        )
        print("✅ 原始架構核心創建成功")
        
        # 測試前向推理
        test_input = torch.randn(1, 512)
        with torch.no_grad():
            output = legacy_core.forward(test_input)
            print(f"✅ 原始架構輸出形狀: {output.shape} (預期: [1, 128])")
        
        # 測試5M架構不支援雙輸出
        try:
            legacy_core.forward_with_aux(test_input)
            print("❌ 原始架構不應支援雙輸出")
            return False
        except ValueError:
            print("✅ 原始架構正確拒絕雙輸出")
        
        return True
        
    except Exception as e:
        print(f"❌ 兼容性測試失敗: {e}")
        return False

def main():
    """主測試流程"""
    print("🚀 開始5M特化神經網路整合測試")
    print("=" * 50)
    
    # 檢查CUDA
    if torch.cuda.is_available():
        print(f"🎮 CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 使用CPU運算")
    
    # 執行測試
    tests = [
        ("5M模型核心", test_5m_model),
        ("5M決策引擎", test_decision_engine),
        ("向後兼容性", test_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通過")
        else:
            print(f"❌ {test_name} 失敗")
    
    print(f"\n{'='*50}")
    print(f"🏆 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！5M特化神經網路整合成功!")
        return True
    else:
        print("⚠️  部分測試失敗，需要檢查問題")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)