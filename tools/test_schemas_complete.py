"""
測試所有 schemas 類別導入
"""

import sys
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    print("[SEARCH] 開始測試 schemas 模組導入...")
    print("=" * 60)
    
    # 測試基礎導入
    from aiva_common import schemas
    
    # 計算總類別數
    all_items = [x for x in dir(schemas) if not x.startswith('_')]
    class_count = len([x for x in all_items if x[0].isupper()])
    
    print(f"\n[OK] 模組載入成功")
    print(f"   總項目: {len(all_items)}")
    print(f"   類別數: {class_count}")
    
    # 測試新增的 AI 類別
    print("\n[U+1F4E6] 測試 AI 模組新增類別...")
    from aiva_common.schemas import (
        AITrainingCompletedPayload,
        AIExperienceCreatedEvent,
        AITraceCompletedEvent,
        AIModelUpdatedEvent,
        AIModelDeployCommand,
        RAGResponsePayload,
    )
    print("   [OK] AI 類別 (6/6)")
    
    # 測試新增的 Tasks 類別
    print("\n[U+1F4E6] 測試 Tasks 模組新增類別...")
    from aiva_common.schemas import (
        StandardScenario,
        ScenarioTestResult,
        ExploitPayload,
        TestExecution,
        ExploitResult,
        TestStrategy,
    )
    print("   [OK] Tasks 類別 (6/6)")
    
    # 測試新增的 Assets 類別
    print("\n[U+1F4E6] 測試 Assets 模組新增類別...")
    from aiva_common.schemas import (
        TechnicalFingerprint,
        AssetInventoryItem,
        EASMAsset,
    )
    print("   [OK] Assets 類別 (3/3)")
    
    # 測試新增的 Telemetry 類別
    print("\n[U+1F4E6] 測試 Telemetry 模組新增類別...")
    from aiva_common.schemas import SIEMEvent
    print("   [OK] Telemetry 類別 (1/1)")
    
    # 測試 Enhanced 模組
    print("\n[U+1F4E6] 測試 Enhanced 模組類別...")
    from aiva_common.schemas import (
        EnhancedFindingPayload,
        EnhancedScanScope,
        EnhancedScanRequest,
        EnhancedFunctionTaskTarget,
        EnhancedIOCRecord,
        EnhancedRiskAssessment,
        EnhancedAttackPathNode,
        EnhancedAttackPath,
        EnhancedTaskExecution,
        EnhancedModuleStatus,
        EnhancedVulnerabilityCorrelation,
    )
    print("   [OK] Enhanced 類別 (11/11)")
    
    # 測試 System 模組
    print("\n[U+1F4E6] 測試 System 模組類別...")
    from aiva_common.schemas import (
        SessionState,
        ModelTrainingResult,
        TaskQueue,
        SystemOrchestration,
        WebhookPayload,
    )
    print("   [OK] System 類別 (5/5)")
    
    # 測試 References 模組
    print("\n[U+1F4E6] 測試 References 模組類別...")
    from aiva_common.schemas import (
        CVEReference,
        CWEReference,
        VulnerabilityDiscovery,
    )
    print("   [OK] References 類別 (3/3)")
    
    # 測試原有核心類別
    print("\n[U+1F4E6] 測試原有核心類別...")
    from aiva_common.schemas import (
        MessageHeader,
        FindingPayload,
        ScanStartPayload,
        HeartbeatPayload,
        AttackPlan,
        RiskAssessmentResult,
    )
    print("   [OK] 核心類別正常")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 所有測試通過!")
    print(f"[STATS] 成功導入 {class_count} 個類別")
    print("=" * 60)
    
    # 驗證 __all__ 列表
    if hasattr(schemas, '__all__'):
        all_count = len(schemas.__all__)
        print(f"\n[OK] __all__ 列表: {all_count} 個導出項目")
    
    print(f"\n[OK] 版本信息:")
    print(f"   __version__: {schemas.__version__}")
    print(f"   __schema_version__: {schemas.__schema_version__}")
    
except ImportError as e:
    print(f"\n[FAIL] 導入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"\n[FAIL] 測試失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
