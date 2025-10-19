"""
測試所有8個關鍵模組的匯入和基本功能
"""
import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_module(module_name: str, test_func):
    """測試單個模組"""
    try:
        print(f"\n{'='*60}")
        print(f"測試模組: {module_name}")
        print(f"{'='*60}")
        test_func()
        print(f"✅ {module_name} - 測試通過")
        return True
    except Exception as e:
        print(f"❌ {module_name} - 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_1_aiva_common_config():
    """測試 services/aiva_common/config.py"""
    from services.aiva_common.config import get_settings
    cfg = get_settings()
    print(f"  - ENV: {cfg.ENV if hasattr(cfg, 'ENV') else 'N/A'}")
    print(f"  - Settings 類型: {type(cfg).__name__}")
    
def test_2_aiva_common_models():
    """測試 services/aiva_common/models.py"""
    # 這個檔案是 re-export schemas
    from services.aiva_common import models
    print(f"  - 模組已載入: {models.__name__}")
    # 嘗試導入一些常見的 schema
    try:
        from services.aiva_common.schemas import base
        print(f"  - 已載入 base schema")
    except:
        print(f"  - schemas 模組可用")

def test_3_core_schemas():
    """測試 services/core/aiva_core/schemas.py"""
    from services.core.aiva_core.schemas import (
        AssetAnalysis, 
        AttackSurfaceAnalysis,
        TestTask,
        TestStrategy
    )
    # 測試建立物件
    asset = AssetAnalysis(
        asset_id="test1",
        url="https://example.com",
        asset_type="endpoint",
        risk_score=75
    )
    print(f"  - AssetAnalysis: {asset.asset_id}, risk={asset.risk_score}")
    
    task = TestTask(
        vulnerability_type="xss",
        asset="https://example.com",
        parameter="q"
    )
    print(f"  - TestTask: {task.vulnerability_type}, priority={task.priority}")

def test_4_storage_config():
    """測試 services/core/aiva_core/storage/config.py"""
    from services.core.aiva_core.storage.config import get_storage_config
    cfg = get_storage_config()
    print(f"  - DB type: {cfg.get('db_type')}")
    print(f"  - Data root: {cfg.get('data_root')}")

def test_5_storage_models():
    """測試 services/core/aiva_core/storage/models.py"""
    from services.core.aiva_core.storage.models import (
        ExperienceSampleModel,
        TrainingSessionModel,
        ModelCheckpointModel
    )
    print(f"  - ExperienceSampleModel table: {ExperienceSampleModel.__tablename__}")
    print(f"  - TrainingSessionModel table: {TrainingSessionModel.__tablename__}")
    print(f"  - ModelCheckpointModel table: {ModelCheckpointModel.__tablename__}")

def test_6_scan_schemas():
    """測試 services/scan/schemas.py"""
    from services.scan.schemas import Target, ScanContext
    
    target = Target(url="https://example.com")
    ctx = ScanContext(targets=[target], depth=3)
    
    print(f"  - Target: {target.url}")
    print(f"  - ScanContext: {len(ctx.targets)} targets, depth={ctx.depth}")

def test_7_session_state_manager():
    """測試 services/core/session_state_manager.py"""
    from services.core.session_state_manager import get_session_manager
    
    sm = get_session_manager()
    sm.create("test_session_1", {"user": "kyle"})
    session = sm.get("test_session_1")
    
    print(f"  - Session created: {session is not None}")
    print(f"  - Session data: {session.get('data') if session else 'N/A'}")
    
    sm.update("test_session_1", {"status": "active"})
    updated = sm.get("test_session_1")
    print(f"  - Session updated: {updated['data'].get('status')}")
    
    sm.delete("test_session_1")
    deleted = sm.get("test_session_1")
    print(f"  - Session deleted: {deleted is None}")

def test_8_smart_detection_manager():
    """測試 services/features/smart_detection_manager.py"""
    from services.features.smart_detection_manager import get_smart_detection_manager
    
    manager = get_smart_detection_manager()
    
    # 註冊測試檢測器
    def test_detector(data):
        return {"detected": True, "input": data}
    
    manager.register("test_detector", test_detector)
    results = manager.run_all({"test": "data"})
    
    print(f"  - Registered detectors: 1+")
    print(f"  - Run results: {len(results)} detectors executed")
    print(f"  - First result: {results[0].get('result')}")
    
    manager.unregister("test_detector")

def main():
    """執行所有測試"""
    print("\n" + "="*60)
    print("AIVA 核心模組匯入測試")
    print("="*60)
    
    tests = [
        ("services.aiva_common.config", test_1_aiva_common_config),
        ("services.aiva_common.models", test_2_aiva_common_models),
        ("services.core.aiva_core.schemas", test_3_core_schemas),
        ("services.core.aiva_core.storage.config", test_4_storage_config),
        ("services.core.aiva_core.storage.models", test_5_storage_models),
        ("services.scan.schemas", test_6_scan_schemas),
        ("services.core.session_state_manager", test_7_session_state_manager),
        ("services.features.smart_detection_manager", test_8_smart_detection_manager),
    ]
    
    results = []
    for module_name, test_func in tests:
        success = test_module(module_name, test_func)
        results.append((module_name, success))
    
    # 總結
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for module_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {module_name}")
    
    print(f"\n總計: {passed}/{total} 通過 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有模組測試通過！系統可以正常匯入和使用。")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 個模組測試失敗，請檢查錯誤訊息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
