#!/usr/bin/env python3
"""
測試 AIVA 模組導入修復

此測試驗證模組導入問題的修復:
1. 檢查 schemas.py 中的類是否可以正確導入
2. 檢查 models.py 的向後兼容性（重新導出）
3. 檢查 aiva_common 包的導出
4. 檢查服務模組的導入
"""

import sys
from pathlib import Path

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))


def test_schemas_direct_import():
    """測試從 schemas.py 直接導入"""
    print("🧪 測試 1: 從 schemas.py 直接導入...")
    try:
        from services.aiva_common.schemas import (
            # 核心消息協議
            MessageHeader,
            AivaMessage,
            # 認證和限流
            Authentication,
            RateLimit,
            # 安全標準
            CVSSv3Metrics,
            CVEReference,
            CWEReference,
            CAPECReference,
            # SARIF 格式
            SARIFLocation,
            SARIFResult,
            SARIFRule,
            SARIFTool,
            SARIFRun,
            SARIFReport,
        )
        print("  ✅ 所有核心類成功從 schemas.py 導入")
        
        # 驗證類的存在
        assert MessageHeader is not None
        assert AivaMessage is not None
        assert CVSSv3Metrics is not None
        assert CVEReference is not None
        assert CWEReference is not None
        assert CAPECReference is not None
        assert SARIFLocation is not None
        print("  ✅ 所有類驗證通過")
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        assert False, f"Schema 導入失敗: {e}"


def test_models_backward_compatibility():
    """測試 models.py 的向後兼容性"""
    print("\n🧪 測試 2: models.py 向後兼容性（重新導出）...")
    try:
        from services.aiva_common.models import (
            MessageHeader,
            AivaMessage,
            Authentication,
            RateLimit,
            CVSSv3Metrics,
            CVEReference,
            CWEReference,
            CAPECReference,
            SARIFLocation,
            SARIFResult,
            SARIFRule,
            SARIFTool,
            SARIFRun,
            SARIFReport,
        )
        print("  ✅ 成功從 models.py 導入（通過重新導出）")
        
        # 驗證這些類實際上來自 schemas.py
        from services.aiva_common import schemas
        assert MessageHeader is schemas.MessageHeader
        assert CVSSv3Metrics is schemas.CVSSv3Metrics
        print("  ✅ 確認類來自 schemas.py（非重複定義）")
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        assert False, f"Models backward compatibility 導入失敗: {e}"
    except AssertionError as ae:
        print("  ❌ 類不是來自 schemas.py（可能存在重複定義）")
        assert False, "類不是來自 schemas.py（可能存在重複定義）"


def test_aiva_common_package_exports():
    """測試從 aiva_common 包導入"""
    print("\n🧪 測試 3: 從 aiva_common 包導入...")
    try:
        from services.aiva_common import (
            # 枚舉
            ModuleName,
            Topic,
            Severity,
            # 核心類
            MessageHeader,
            AivaMessage,
            CVSSv3Metrics,
            CVEReference,
            CWEReference,
            CAPECReference,
            SARIFLocation,
            SARIFResult,
            SARIFRule,
            SARIFTool,
            SARIFRun,
            SARIFReport,
        )
        print("  ✅ 成功從 aiva_common 包導入")
        
        # 檢查枚舉
        assert ModuleName is not None
        assert Topic is not None
        print("  ✅ 枚舉和類都可用")
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        assert False, f"Package exports 導入失敗: {e}"


def test_service_module_imports():
    """測試服務模組的導入"""
    print("\n🧪 測試 4: 從服務模組導入...")
    
    results = []
    
    # 測試 scan 模組
    try:
        from services.scan import CVEReference, CVSSv3Metrics, CWEReference
        print("  ✅ services.scan 導入成功")
        results.append(True)
    except ImportError as e:
        print(f"  ❌ services.scan 導入失敗: {e}")
        results.append(False)
    
    # 測試 core 模組
    try:
        from services.core.aiva_core import CVEReference, CVSSv3Metrics, CWEReference
        print("  ✅ services.core.aiva_core 導入成功")
        results.append(True)
    except ImportError as e:
        print(f"  ❌ services.core.aiva_core 導入失敗: {e}")
        results.append(False)
    
    # 測試 features 模組 (修正路徑)
    try:
        from services.features import CVSSv3Metrics
        print("  ✅ services.features 導入成功")
        results.append(True)
    except ImportError as e:
        print(f"  ❌ services.features 導入失敗: {e}")
        results.append(False)
    
    # 檢查所有結果
    if not all(results):
        failed_imports = [f"測試 {i+1}" for i, result in enumerate(results) if not result]
        assert False, f"服務模組導入失敗: {failed_imports}"


def test_no_circular_imports():
    """測試沒有循環導入"""
    print("\n🧪 測試 5: 檢查循環導入...")
    try:
        # 嘗試導入可能產生循環依賴的模組
        import services.aiva_common
        import services.aiva_common.schemas
        import services.aiva_common.models
        import services.aiva_common.enums
        print("  ✅ 沒有檢測到循環導入")
    except ImportError as e:
        print(f"  ❌ 可能存在循環導入: {e}")
        assert False, f"檢測到循環導入: {e}"


def test_class_consistency():
    """測試類的一致性"""
    print("\n🧪 測試 6: 類的一致性檢查...")
    try:
        from services.aiva_common.schemas import MessageHeader as SchemaHeader
        from services.aiva_common.models import MessageHeader as ModelHeader
        from services.aiva_common import MessageHeader as PackageHeader
        
        # 確保它們都是同一個類
        assert SchemaHeader is ModelHeader is PackageHeader
        print("  ✅ MessageHeader 在所有導入位置保持一致")
        
        from services.aiva_common.schemas import CVSSv3Metrics as SchemaCVSS
        from services.aiva_common.models import CVSSv3Metrics as ModelCVSS
        from services.aiva_common import CVSSv3Metrics as PackageCVSS
        
        assert SchemaCVSS is ModelCVSS is PackageCVSS
        print("  ✅ CVSSv3Metrics 在所有導入位置保持一致")
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        assert False, f"Class consistency 導入失敗: {e}"
    except AssertionError:
        print("  ❌ 類在不同位置不一致（可能存在重複定義）")
        assert False, "類在不同位置不一致（可能存在重複定義）"


def main():
    """主測試函數"""
    print("=" * 60)
    print("🚀 AIVA 模組導入修復測試")
    print("=" * 60)
    
    results = []
    
    # 執行所有測試
    results.append(("schemas.py 直接導入", test_schemas_direct_import()))
    results.append(("models.py 向後兼容性", test_models_backward_compatibility()))
    results.append(("aiva_common 包導出", test_aiva_common_package_exports()))
    results.append(("服務模組導入", test_service_module_imports()))
    results.append(("循環導入檢查", test_no_circular_imports()))
    results.append(("類一致性", test_class_consistency()))
    
    # 統計結果
    print("\n" + "=" * 60)
    print("📊 測試結果總結")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"通過: {passed}/{total} ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n✨ 所有測試通過！模組導入問題已修復。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 個測試失敗。")
        print("注意: 如果錯誤是 'No module named pydantic'，")
        print("請先安裝依賴: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
