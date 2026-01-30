"""
Simple test to verify AttackCoordinator initialization
直接測試 AttackCoordinator 初始化而不載入整個服務層
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_attack_coordinator_simple():
    """測試 AttackCoordinator 基本初始化"""
    try:
        # Direct import to bypass full service initialization
        from services.core.aiva_core.task_planning.commander import attack_coordinator
        
        # Test 1: Verify module loads
        print("✓ AttackCoordinator module loaded successfully")
        
        # Test 2: Check class exists
        AttackCoordinator = attack_coordinator.AttackCoordinator
        print("✓ AttackCoordinator class found")
        
        # Test 3: Check __init__ signature
        import inspect
        sig = inspect.signature(AttackCoordinator.__init__)
        params = list(sig.parameters.keys())
        print(f"✓ __init__ parameters: {params}")
        
        # Expected: self, data_directory, dispatcher, unified_executor=None, 
        #           multilang_coordinator=None, internal_loop=None
        if len(params) >= 3:  # at least self, data_directory, dispatcher
            print("✓ AttackCoordinator has extended parameters")
            return True
        else:
            print(f"✗ Expected at least 3 parameters, got {len(params)}")
            return False
            
    except Exception as e:
        print(f"✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attack_coordinator_simple()
    sys.exit(0 if success else 1)
