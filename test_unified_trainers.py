#!/usr/bin/env python3
"""
測試統一的訓練系統
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'core'))

def test_unified_trainers():
    """測試統一的訓練器"""
    print("🔍 測試統一的訓練系統...")
    
    try:
        # 測試導入
        from aiva_core.learning.model_trainer import ModelTrainer
        from aiva_core.learning.scalable_bio_trainer import ScalableBioTrainer, ScalableBioTrainingConfig
        from aiva_core.ai_engine.training import ModelUpdater
        
        print("✅ 成功導入統一的訓練器")
        
        # 測試 ScalableBioTrainingConfig
        config = ScalableBioTrainingConfig(
            learning_rate=0.002,
            epochs=5,
            batch_size=16
        )
        print(f"✅ ScalableBioTrainingConfig 創建成功: lr={config.learning_rate}")
        
        # 模擬創建 ScalableBioNet 模型
        class MockScalableBioNet:
            def __init__(self):
                import numpy as np
                self.fc1 = np.random.randn(100, 50)
                self.fc2 = np.random.randn(50, 10)
                self.total_params = 100 * 50 + 50 * 10
                
                # 模擬 spiking layer
                class MockSpikingLayer:
                    def __init__(self):
                        self.weights = np.random.randn(50, 50)
                self.spiking1 = MockSpikingLayer()
                
            def forward(self, x):
                import numpy as np
                # 簡單的前向傳播模擬
                return np.random.randn(len(x), 10)
                
            def backward(self, x, y, lr):
                # 簡單的反向傳播模擬
                pass
        
        model = MockScalableBioNet()
        
        # 測試 ScalableBioTrainer
        trainer = ScalableBioTrainer(model, config)
        print("✅ ScalableBioTrainer 創建成功")
        
        # 測試訓練功能
        import numpy as np
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100, 10)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20, 10)
        
        # 設置較少的 epochs 以快速測試
        config.epochs = 2
        trainer.config = config
        
        results = trainer.train(X_train, y_train, X_val, y_val)
        print(f"✅ 訓練測試成功: loss={results['final_loss']:.4f}")
        
        # 測試訓練歷史
        history = trainer.get_training_history()
        print(f"✅ 訓練歷史獲取成功: {len(history['loss'])} epochs")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_integration():
    """測試整合狀況"""
    print("\n🔍 測試模組整合...")
    
    try:
        # 測試從不同模組導入
        from aiva_core.ai_engine.training import ScalableBioTrainer, ModelTrainer
        from aiva_core.learning import ModelTrainer as LearningModelTrainer
        
        print("✅ 模組間整合成功")
        
        # 確認是同一個類
        assert ModelTrainer == LearningModelTrainer
        print("✅ ModelTrainer 統一確認")
        
        return True
        
    except Exception as e:
        print(f"❌ 整合測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("AI 訓練系統統一測試")
    print("=" * 50)
    
    success = True
    success &= test_unified_trainers()
    success &= test_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有測試通過！訓練系統統一成功")
    else:
        print("❌ 測試失敗，需要進一步調整")
    print("=" * 50)