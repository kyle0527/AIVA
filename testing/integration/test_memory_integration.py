#!/usr/bin/env python3
"""
記憶體管理整合驗證測試
測試統一記憶體管理器的功能是否正常運作
"""

import sys
import asyncio
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_memory_manager_integration():
    """測試記憶體管理器整合"""
    print("🔍 測試記憶體管理器整合...")
    
    try:
        # 測試導入
        from services.core.aiva_core.performance import UnifiedMemoryManager
        print("✅ 統一記憶體管理器導入成功")
        
        # 創建實例
        memory_manager = UnifiedMemoryManager(
            max_cache_size=100,
            gc_threshold_mb=256,
            enable_monitoring=False  # 測試時關閉監控
        )
        print("✅ 記憶體管理器創建成功")
        
        # 測試AI快取功能
        memory_manager.cache_prediction("test_input", {"result": "test_output"})
        cached_result = memory_manager.get_cached_prediction("test_input")
        assert cached_result == {"result": "test_output"}
        print("✅ AI預測快取功能正常")
        
        # 測試記憶體優化
        optimize_result = memory_manager.optimize_memory(force_gc=True)
        print(f"✅ 記憶體優化功能正常: {optimize_result}")
        
        # 測試統計功能
        stats = memory_manager.get_comprehensive_stats()
        assert 'cache' in stats
        assert 'memory' in stats
        assert 'pools' in stats
        print("✅ 統計功能正常")
        
        # 測試批次處理
        test_items = [1, 2, 3, 4, 5]
        results = memory_manager.process_batch(test_items, lambda x: x * 2)
        assert results == [2, 4, 6, 8, 10]
        print("✅ 批次處理功能正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 記憶體管理器整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """測試向後兼容性"""
    print("\n🔍 測試向後兼容性...")
    
    try:
        # 測試舊名稱導入
        from services.core.aiva_core.performance import MemoryManager, ComponentPool
        print("✅ 向後兼容性導入成功")
        
        # 測試AI引擎導入
        from services.core.aiva_core.ai_engine import MemoryManager as AIMemoryManager
        print("✅ AI引擎記憶體管理器導入成功")
        
        # 確認是同一個類
        assert MemoryManager == AIMemoryManager
        print("✅ 記憶體管理器統一確認")
        
        return True
        
    except Exception as e:
        print(f"❌ 向後兼容性測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("📋 記憶體管理整合驗證測試開始\n")
    
    results = []
    
    # 執行測試
    results.append(test_memory_manager_integration())
    results.append(test_legacy_compatibility())
    
    # 統計結果
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"\n📊 測試結果總結:")
    print(f"   總測試數: {total_tests}")
    print(f"   通過測試: {passed_tests}")
    print(f"   失敗測試: {total_tests - passed_tests}")
    
    if all(results):
        print("🎉 記憶體管理整合完全成功！")
        print("   ✅ 消除了重複代碼")
        print("   ✅ 保持了向後兼容性") 
        print("   ✅ 整合了所有功能")
        return True
    else:
        print("⚠️ 部分測試失敗，需要進一步調整")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)