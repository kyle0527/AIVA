#!/usr/bin/env python3
"""
測試AI組件功能
"""

import sys
from pathlib import Path

# 設置路徑
sys.path.insert(0, str(Path(__file__).parent / "services"))

def test_ai_commander():
    """測試AI指揮官組件"""
    try:
        from core.aiva_core.ai_commander import AICommander
        commander = AICommander()
        
        print("✅ AI指揮官初始化成功")
        
        # 獲取AI指揮官的方法
        methods = [method for method in dir(commander) if not method.startswith('_')]
        print(f"🎯 AI指揮官功能: {methods[:10]}")
        
        return True
    except Exception as e:
        print(f"❌ AI指揮官測試失敗: {e}")
        return False

def test_learning_engine():
    """測試學習引擎組件"""
    try:
        from core.aiva_core.ai_engine.learning_engine import LearningEngine
        engine = LearningEngine()
        
        print("✅ 學習引擎初始化成功")
        
        # 獲取學習引擎的方法
        methods = [method for method in dir(engine) if not method.startswith('_')]
        print(f"🧠 學習引擎功能: {methods[:10]}")
        
        return True
    except Exception as e:
        print(f"❌ 學習引擎測試失敗: {e}")
        return False

def test_bio_neuron():
    """測試BioNeuron組件"""
    try:
        from core.aiva_core.bio_neuron_master import BioNeuronMaster
        bio_neuron = BioNeuronMaster()
        
        print("✅ BioNeuron主控初始化成功")
        
        # 獲取BioNeuron的方法
        methods = [method for method in dir(bio_neuron) if not method.startswith('_')]
        print(f"🧬 BioNeuron功能: {methods[:10]}")
        
        return True
    except Exception as e:
        print(f"❌ BioNeuron測試失敗: {e}")
        return False

def test_smart_detector():
    """測試智能檢測管理器"""
    try:
        from features.smart_detection_manager import SmartDetectionManager
        detector = SmartDetectionManager()
        
        print("✅ 智能檢測管理器初始化成功")
        
        # 獲取智能檢測器的方法
        methods = [method for method in dir(detector) if not method.startswith('_')]
        print(f"🔍 智能檢測功能: {methods[:10]}")
        
        return True
    except Exception as e:
        print(f"❌ 智能檢測器測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🔍 開始測試AI組件功能...")
    print("="*50)
    
    test_results = []
    
    # 測試核心AI組件
    print("\n🧠 測試核心AI組件:")
    test_results.append(("AI指揮官", test_ai_commander()))
    test_results.append(("學習引擎", test_learning_engine()))
    test_results.append(("BioNeuron", test_bio_neuron()))
    
    print("\n🔍 測試智能檢測組件:")
    test_results.append(("智能檢測器", test_smart_detector()))
    
    # 總結
    print("\n" + "="*50)
    print("🎯 AI組件測試總結:")
    
    success_count = 0
    for component_name, success in test_results:
        status = "✅" if success else "❌"
        print(f"   {status} {component_name}")
        if success:
            success_count += 1
    
    print(f"\n📊 成功率: {success_count}/{len(test_results)} ({success_count/len(test_results)*100:.1f}%)")

if __name__ == "__main__":
    main()