"""
測試範圍管理功能 (v11.0)

測試 InternalLoopConnector 的範圍分類功能，驗證：
1. CapabilityScopeClassifier 的自動分類邏輯
2. 範圍、可見性、訪問級別的正確分配
3. CLI 成熟度檢測
4. 服務依賴檢測

遵循 aiva_common v2.0 規範：
- 使用統一的日誌記錄
- 使用 Pydantic 模型進行數據驗證
- 異步執行測試
"""

import asyncio
import sys
from pathlib import Path

# 添加 services 目錄到路徑
services_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(services_path))

from services.aiva_common.utils.logging import get_logger
from services.aiva_common.schemas.dual_loop import (
    CapabilityScope,
    CapabilityVisibility,
    CapabilityAccessLevel,
    CLIMaturityLevel
)

logger = get_logger(__name__)


def test_scope_classifier():
    """測試範圍分類器"""
    from internal_loop_connector import CapabilityScopeClassifier
    
    classifier = CapabilityScopeClassifier()
    
    logger.info("=" * 80)
    logger.info("🧪 測試範圍分類器 (CapabilityScopeClassifier)")
    logger.info("=" * 80)
    
    # 測試案例
    test_cases = [
        {
            "name": "內部探索工具",
            "file_path": "services/core/aiva_core/internal_exploration/python_tools/analyzer.py",
            "expected_scope": CapabilityScope.CORE,
            "expected_visibility": CapabilityVisibility.INTERNAL
        },
        {
            "name": "任務規劃模組",
            "file_path": "services/core/aiva_core/task_planning/ai_commander.py",
            "expected_scope": CapabilityScope.SERVICE,
            "expected_visibility": CapabilityVisibility.PUBLIC
        },
        {
            "name": "SQL 注入檢測",
            "file_path": "services/features/function_sqli/scanner.py",
            "expected_scope": CapabilityScope.GLOBAL,
            "expected_visibility": CapabilityVisibility.PUBLIC
        },
        {
            "name": "XSS 檢測",
            "file_path": "services/features/function_xss/detector.py",
            "expected_scope": CapabilityScope.GLOBAL,
            "expected_visibility": CapabilityVisibility.PUBLIC
        },
        {
            "name": "掃描引擎",
            "file_path": "services/scan/engines/python_scanner.py",
            "expected_scope": CapabilityScope.GLOBAL,
            "expected_visibility": CapabilityVisibility.PUBLIC
        },
        {
            "name": "整合層",
            "file_path": "services/integration/capability/registry.py",
            "expected_scope": CapabilityScope.GLOBAL,
            "expected_visibility": CapabilityVisibility.SYSTEM
        }
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        logger.info("\n📝 測試案例 %d/%d: %s", i, total_count, case['name'])
        logger.info("   文件路徑: %s", case['file_path'])
        
        # 執行分類
        scope, visibility = classifier.classify_scope(case['file_path'])
        
        # 驗證結果
        scope_match = scope == case['expected_scope']
        visibility_match = visibility == case['expected_visibility']
        
        if scope_match and visibility_match:
            logger.info("   ✅ 分類正確")
            logger.info("      Scope: %s", scope.value)
            logger.info("      Visibility: %s", visibility.value)
            success_count += 1
        else:
            logger.error(f"   ❌ 分類錯誤")
            logger.error(f"      預期 Scope: {case['expected_scope'].value}, 實際: {scope.value}")
            logger.error(f"      預期 Visibility: {case['expected_visibility'].value}, 實際: {visibility.value}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 測試結果: {success_count}/{total_count} 通過")
    logger.info(f"{'=' * 80}")
    
    return success_count == total_count


def test_access_level_classification():
    """測試訪問級別分類"""
    from internal_loop_connector import CapabilityScopeClassifier
    
    classifier = CapabilityScopeClassifier()
    
    logger.info("\n" + "=" * 80)
    logger.info("🧪 測試訪問級別分類")
    logger.info("=" * 80)
    
    test_cases = [
        ("scanning", None, CapabilityAccessLevel.L2_MODULE),
        ("attacking", "sql_injection", CapabilityAccessLevel.L2_MODULE),
        ("analysis", "deviation_analysis", CapabilityAccessLevel.L1_SERVICE),
        ("reporting", "report_generation", CapabilityAccessLevel.L1_SERVICE),
        ("integration", "orchestration", CapabilityAccessLevel.L0_SYSTEM),
        ("utility", "encoding", CapabilityAccessLevel.L3_INTERNAL)
    ]
    
    success_count = 0
    
    for category, sub_category, expected_level in test_cases:
        level = classifier.classify_access_level(category, sub_category)
        
        if level == expected_level:
            logger.info(f"✅ {category:15} → {level.value}")
            success_count += 1
        else:
            logger.error(f"❌ {category:15} → {level.value} (預期: {expected_level.value})")
    
    logger.info(f"\n測試結果: {success_count}/{len(test_cases)} 通過")
    
    return success_count == len(test_cases)


def test_cli_detection():
    """測試 CLI 檢測"""
    from internal_loop_connector import CapabilityScopeClassifier
    
    classifier = CapabilityScopeClassifier()
    
    logger.info("\n" + "=" * 80)
    logger.info("🧪 測試 CLI 檢測")
    logger.info("=" * 80)
    
    test_caps = [
        {
            "name": "sqli_scan",
            "file_path": "services/features/function_sqli/cli.py"
        },
        {
            "name": "xss_detect",
            "file_path": "services/features/function_xss/detector.py"
        },
        {
            "name": "internal_analyze",
            "file_path": "services/core/aiva_core/internal_exploration/analyzer.py"
        }
    ]
    
    for cap in test_caps:
        has_cli, cli_command, maturity = classifier.detect_cli_info(cap)
        
        logger.info(f"\n📝 {cap['name']}")
        logger.info(f"   文件: {cap['file_path']}")
        logger.info(f"   Has CLI: {has_cli}")
        if has_cli:
            logger.info(f"   CLI 命令: {cli_command}")
            logger.info(f"   成熟度: {maturity.value}")


def test_capability_enhancement():
    """測試能力增強（完整流程）"""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 測試能力增強（完整流程）")
    logger.info("=" * 80)
    
    try:
        from internal_loop_connector import InternalLoopConnector
        
        # 創建連接器實例
        connector = InternalLoopConnector(rag_knowledge_base=None, pg_session=None)
        
        logger.info("✅ InternalLoopConnector 初始化成功")
        logger.info(f"   範圍分類器: {connector.scope_classifier.__class__.__name__}")
        
        # 測試增強單個能力
        test_cap = {
            "name": "sql_injection_scan",
            "module": "function_sqli",
            "file_path": "services/features/function_sqli/scanner.py",
            "parameters": [
                {"name": "target_url", "type": "str"},
                {"name": "scan_engine", "type": "ScanEngine"}
            ]
        }
        
        logger.info("\n📝 測試能力增強:")
        logger.info(f"   能力名稱: {test_cap['name']}")
        logger.info(f"   模組: {test_cap['module']}")
        
        # 測試範圍分類
        scope, visibility = connector.scope_classifier.classify_scope(test_cap['file_path'])
        access_level = connector.scope_classifier.classify_access_level("attacking", "sql_injection")
        available_in = connector.scope_classifier.detect_available_in(test_cap['file_path'])
        has_cli, cli_command, cli_maturity = connector.scope_classifier.detect_cli_info(test_cap)
        depends_on = connector.scope_classifier.detect_service_dependencies(test_cap)
        
        logger.info("\n✅ 範圍分類結果:")
        logger.info(f"   Scope: {scope.value}")
        logger.info(f"   Visibility: {visibility.value}")
        logger.info(f"   Access Level: {access_level.value}")
        logger.info(f"   Available In: {available_in}")
        logger.info(f"   Has CLI: {has_cli}")
        if has_cli:
            logger.info(f"   CLI Command: {cli_command}")
            logger.info(f"   CLI Maturity: {cli_maturity.value}")
        logger.info(f"   Depends On: {depends_on}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    logger.info("\n" + "🚀 " * 20)
    logger.info("開始測試範圍管理功能 (v11.0)")
    logger.info("🚀 " * 20)
    
    results = []
    
    # 測試 1: 範圍分類器
    try:
        result = test_scope_classifier()
        results.append(("範圍分類器", result))
    except Exception as e:
        logger.error(f"❌ 範圍分類器測試失敗: {e}")
        results.append(("範圍分類器", False))
    
    # 測試 2: 訪問級別分類
    try:
        result = test_access_level_classification()
        results.append(("訪問級別分類", result))
    except Exception as e:
        logger.error(f"❌ 訪問級別分類測試失敗: {e}")
        results.append(("訪問級別分類", False))
    
    # 測試 3: CLI 檢測
    try:
        test_cli_detection()
        results.append(("CLI 檢測", True))
    except Exception as e:
        logger.error(f"❌ CLI 檢測測試失敗: {e}")
        results.append(("CLI 檢測", False))
    
    # 測試 4: 能力增強
    try:
        result = test_capability_enhancement()
        results.append(("能力增強", result))
    except Exception as e:
        logger.error(f"❌ 能力增強測試失敗: {e}")
        results.append(("能力增強", False))
    
    # 總結
    logger.info("\n" + "=" * 80)
    logger.info("📊 測試總結")
    logger.info("=" * 80)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"   {test_name:20} {status}")
    
    total_pass = sum(1 for _, r in results if r)
    total_tests = len(results)
    
    logger.info(f"\n總計: {total_pass}/{total_tests} 測試通過")
    
    if total_pass == total_tests:
        logger.info("✅ 所有測試通過！")
        return 0
    else:
        logger.error("❌ 部分測試失敗")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
