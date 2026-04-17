"""
Ultra simple test - direct file import only
直接導入檔案而不透過任何 __init__.py
"""

import sys
from pathlib import Path

# Add directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_direct_import():
    """直接導入 attack_coordinator.py 檔案"""
    try:
        print("Loading attack_coordinator module...")
        
        # Import the module file directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "attack_coordinator",
            project_root / "services" / "core" / "aiva_core" / "task_planning" / "commander" / "attack_coordinator.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # Execute to load classes
        spec.loader.exec_module(module)
        
        print("Module loaded successfully")
        
        # Check for AttackCoordinator class
        if hasattr(module, 'AttackCoordinator'):
            print("AttackCoordinator class found")
            
            # Check __init__ signature
            import inspect
            sig = inspect.signature(module.AttackCoordinator.__init__)
            params = list(sig.parameters.keys())
            print(f"__init__ parameters: {params}")
            
            expected_params = ['self', 'unified_executor', 'multilang_coordinator', 'internal_loop', 'session_state_manager']
            if set(expected_params).issubset(set(params)):
                print("SUCCESS: All expected parameters present")
                return True
            else:
                print(f"FAIL: Expected {expected_params}, got {params}")
                return False
        else:
            print("FAIL: AttackCoordinator class not found")
            return False
            
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_import()
    sys.exit(0 if success else 1)
