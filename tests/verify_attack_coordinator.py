"""驗證 AttackCoordinator 修復

測試初始化參數是否正確工作
"""
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_attack_coordinator_init():
    """測試 AttackCoordinator 初始化參數"""
    from services.core.aiva_core.task_planning.commander.attack_coordinator import AttackCoordinator
    
    print("Testing AttackCoordinator initialization with all parameters...")
    
    # 測試 1: 只傳入必要參數（向後兼容）
    coordinator1 = AttackCoordinator(
        data_directory=Path("./test_data")
    )
    assert coordinator1.data_directory == Path("./test_data")
    assert coordinator1.dispatcher is None
    assert coordinator1.unified_executor is None
    print("✅ Test 1 PASSED: Basic initialization (backward compatible)")
    
    # 測試 2: 傳入所有參數
    coordinator2 = AttackCoordinator(
        data_directory=Path("./test_data"),
        dispatcher="mock_dispatcher",
        unified_executor="mock_executor",
        multilang_coordinator="mock_coordinator",
        internal_loop="mock_loop"
    )
    assert coordinator2.data_directory == Path("./test_data")
    assert coordinator2.dispatcher == "mock_dispatcher"
    assert coordinator2.unified_executor == "mock_executor"
    assert coordinator2.multilang_coordinator == "mock_coordinator"
    assert coordinator2.internal_loop == "mock_loop"
    print("✅ Test 2 PASSED: Full initialization with all dependencies")
    
    # 測試 3: CLI 執行器初始化
    assert coordinator2._cli_executor is not None
    assert "subprocess" in coordinator2._cli_executor
    assert coordinator2._cli_executor["timeout"] == 300
    print("✅ Test 3 PASSED: CLI executor initialized correctly")
    
    print("\n🎉 All AttackCoordinator tests PASSED!")
    return True

if __name__ == "__main__":
    try:
        success = test_attack_coordinator_init()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
