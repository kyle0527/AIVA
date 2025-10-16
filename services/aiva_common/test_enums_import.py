"""
測試 enums 模組遷移後的導入功能
"""

import sys
from pathlib import Path

# 確保可以導入
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("測試 Enums 模組導入")
print("=" * 70)

# 測試 1: 從 __init__.py 導入所有枚舉
print("\n✓ 測試 1: 從 enums 包導入所有枚舉...")
try:
    from aiva_common.enums import (
        # Common
        Severity, Confidence, TaskStatus, TestStatus,
        ScanStatus, ThreatLevel, RiskLevel, RemediationStatus,
        # Modules
        ModuleName, Topic,
        # Security
        VulnerabilityType, VulnerabilityStatus, Location,
        SensitiveInfoType, IntelSource, IOCType, RemediationType,
        Permission, AccessDecision, PostExTestType, PersistenceType,
        Exploitability, AttackPathNodeType, AttackPathEdgeType,
        # Assets
        BusinessCriticality, Environment, AssetType, AssetStatus,
        DataSensitivity, AssetExposure, ComplianceFramework
    )
    print("✅ 成功導入所有 31 個枚舉")
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    sys.exit(1)

# 測試 2: 驗證 ModuleName 枚舉成員
print("\n✓ 測試 2: 驗證 ModuleName 枚舉...")
try:
    module_names = [m.value for m in ModuleName]
    print(f"✅ ModuleName 有 {len(module_names)} 個成員")
    print(f"   範例: {', '.join(module_names[:5])}...")
except Exception as e:
    print(f"❌ ModuleName 驗證失敗: {e}")

# 測試 3: 驗證 Topic 枚舉成員
print("\n✓ 測試 3: 驗證 Topic 枚舉...")
try:
    topics = [t.value for t in Topic]
    print(f"✅ Topic 有 {len(topics)} 個成員")
    print(f"   範例: {', '.join(topics[:5])}...")
except Exception as e:
    print(f"❌ Topic 驗證失敗: {e}")

# 測試 4: 驗證各個子模組可以單獨導入
print("\n✓ 測試 4: 驗證子模組單獨導入...")
try:
    from aiva_common.enums.common import Severity as Sev1
    from aiva_common.enums.modules import ModuleName as Mod1
    from aiva_common.enums.security import VulnerabilityType as Vuln1
    from aiva_common.enums.assets import AssetType as Asset1
    print("✅ 所有子模組都可以單獨導入")
except ImportError as e:
    print(f"❌ 子模組導入失敗: {e}")

# 測試 5: 檢查 __all__ 列表
print("\n✓ 測試 5: 檢查 __all__ 列表...")
try:
    import aiva_common.enums as enums_module
    all_exports = getattr(enums_module, '__all__', [])
    print(f"✅ __all__ 列表包含 {len(all_exports)} 個項目")
    
    # 驗證所有導出的項目都可訪問
    missing = []
    for name in all_exports:
        if not hasattr(enums_module, name):
            missing.append(name)
    
    if missing:
        print(f"⚠️  以下項目在 __all__ 中但無法訪問: {missing}")
    else:
        print("✅ 所有 __all__ 項目都可以訪問")
except Exception as e:
    print(f"❌ __all__ 驗證失敗: {e}")

# 測試 6: 比較與舊的 enums.py
print("\n✓ 測試 6: 與舊 enums.py 比較...")
try:
    # 計算新模組的枚舉總數
    from aiva_common import enums
    
    enum_classes = [
        name for name in dir(enums)
        if not name.startswith('_') and name[0].isupper()
    ]
    
    print(f"✅ 新 enums 包導出 {len(enum_classes)} 個枚舉類別")
    
    expected = 31
    if len(enum_classes) == expected:
        print(f"✅ 數量符合預期 ({expected} 個)")
    else:
        print(f"⚠️  數量不符: 預期 {expected}, 實際 {len(enum_classes)}")
        
except Exception as e:
    print(f"❌ 比較失敗: {e}")

print("\n" + "=" * 70)
print("測試完成!")
print("=" * 70)
