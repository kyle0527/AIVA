#!/usr/bin/env python3
"""測試AIVA真實AI核心"""

import sys
sys.path.append('.')
import numpy as np

# 模擬utilities模組
class MockDeps:
    def get_or_mock(self, name):
        if name == 'numpy':
            import numpy as np
            return np
        elif name == 'torch':
            try:
                import torch
                return torch
            except ImportError:
                return None
        return None

class MockUtilities:
    @property 
    def optional_deps(self):
        return MockDeps()

# 設置模擬模組
sys.modules['utilities'] = MockUtilities()

print('🧪 測試AIVA真實AI核心...')

# 先測試沒有依賴衝突
try:
    from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
    
    # 測試真實AI核心
    print('創建ScalableBioNet...')
    net = ScalableBioNet(input_size=512, num_tools=10)
    
    has_real_torch = getattr(net, 'has_real_torch', False)
    print(f'真實AI狀態: {has_real_torch}')
    print(f'總參數: {net.total_params:,}')
    
    # 測試前向傳播
    print('測試前向傳播...')
    test_input = np.random.random((1, 512))
    output = net.forward(test_input)
    
    print(f'輸出形狀: {output.shape}')
    print(f'輸出和: {np.sum(output):.3f}')
    print(f'最大值: {np.max(output):.3f}')
    
    if has_real_torch:
        print('🎉 使用真實PyTorch神經網路!')
        print(f'設備: {getattr(net, "device", "unknown")}')
        print(f'網路: {type(getattr(net, "network", None))}')
    else:
        print('⚠️ 降級到假AI模式')
    
    print('✅ AIVA真實AI核心測試完成!')

except Exception as e:
    print(f'❌ 測試失敗: {e}')
    import traceback
    traceback.print_exc()