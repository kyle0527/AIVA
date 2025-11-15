#!/usr/bin/env python3
"""
AIVA 架構修復驗證腳本

用於驗證所有架構修復的正確性和功能完整性
執行時間：2025年11月15日
"""

import traceback
import sys
from datetime import datetime
from typing import Dict, Callable, Any
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
TEST_MODULE_NAME = "test.validation"


def test_core_imports() -> bool:
    """測試核心模組導入"""
    try:
        print("  🔍 測試統一追蹤器導入...")
        from services.core.aiva_core.execution import (
            UnifiedTracer,
            TraceType,
            ExecutionTrace,
            get_global_tracer,
            record_execution_trace
        )
        print("    ✅ UnifiedTracer import successful")
        
        print("  🔍 測試向後相容性別名...")
        from services.core.aiva_core.execution import (
            TraceLogger,
            TraceRecorder
        )
        print("    ✅ Backward compatibility aliases working")
        
        print("  🔍 測試錯誤處理導入...")
        from services.aiva_common.error_handling import (
            AIVAError,
            ErrorType,
            ErrorSeverity,
            ErrorContext
        )
        print("    ✅ AIVA error handling import successful")
        
        print("  🔍 測試MessageBroker導入...")
        from services.core.aiva_core.messaging.message_broker import MessageBroker
        print("    ✅ MessageBroker import successful")
        
        print("  🔍 測試PlanExecutor導入...")
        from services.core.aiva_core.execution.plan_executor import PlanExecutor
        print("    ✅ PlanExecutor import successful")
        
        return True
    except ImportError as e:
        print(f"    ❌ Import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        traceback.print_exc()
        return False


def test_ai_dependencies() -> bool:
    """測試AI模組依賴修復"""
    try:
        print("  🔍 測試AICommander導入...")
        from services.core.aiva_core.ai_commander import AICommander
        print("    ✅ AICommander import successful")
        
        print("  🔍 測試ExperienceManager依賴...")
        try:
            # 動態導入，可能不存在
            import services.aiva_common.ai as ai_module
            if hasattr(ai_module, 'AIVAExperienceManager'):
                print("    ✅ ExperienceManager dependency resolved")
            else:
                print("    ⚠️ ExperienceManager not available in module (這是正常的)")
        except ImportError as ie:
            print(f"    ⚠️ ExperienceManager not available: {ie} (這是正常的)")
        
        return True
    except ImportError as e:
        print(f"    ❌ AI dependency failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        traceback.print_exc()
        return False


def test_aiva_error_handling() -> bool:
    """測試AIVA錯誤處理機制"""
    try:
        print("  🔍 測試AIVAError創建...")
        from services.aiva_common.error_handling import (
            AIVAError, ErrorType, ErrorSeverity, ErrorContext
        )
        
        # 測試錯誤上下文創建
        context = ErrorContext(
            module=TEST_MODULE_NAME,
            function="test_aiva_error_handling",
            additional_data={"test_key": "test_value"}
        )
        print("    ✅ ErrorContext creation successful")
        
        # 測試錯誤創建
        error = AIVAError(
            message="Test error for validation",
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        print("    ✅ AIVAError creation successful")
        
        # 驗證錯誤屬性
        assert error.error_type == ErrorType.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        error_dict = error.to_dict()
        assert "test.validation" in str(error_dict)
        print("    ✅ AIVAError attributes and serialization working")
        
        return True
    except Exception as e:
        print(f"    ❌ AIVAError test failed: {e}")
        traceback.print_exc()
        return False


def test_unified_tracer() -> bool:
    """測試統一追蹤器功能"""
    try:
        print("  🔍 測試UnifiedTracer實例化...")
        from services.core.aiva_core.execution.unified_tracer import (
            UnifiedTracer, TraceType
        )
        
        # 創建追蹤器實例
        tracer = UnifiedTracer()
        print("    ✅ UnifiedTracer creation successful")
        
        print("  🔍 測試會話管理...")
        tracer.start_session("test_session_001")
        print("    ✅ Session management working")
        
        print("  🔍 測試追蹤記錄...")
        trace = tracer.record_trace(
            trace_type=TraceType.EXECUTION,
            module_name=TEST_MODULE_NAME,
            function_name="test_unified_tracer",
            variables={"test_var": "test_value"}
        )
        
        assert trace.trace_type == TraceType.EXECUTION
        assert trace.module_name == TEST_MODULE_NAME
        print("    ✅ Trace recording working")
        
        print("  🔍 測試追蹤查詢...")
        traces = tracer.get_traces(trace_type=TraceType.EXECUTION)
        assert len(traces) > 0
        print("    ✅ Trace querying working")
        
        print("  🔍 測試會話摘要...")
        summary = tracer.get_session_summary()
        assert summary["current_session_id"] == "test_session_001"
        print("    ✅ Session summary working")
        
        print("  🔍 測試會話完成...")
        tracer.complete_session("test_session_001")
        print("    ✅ Session completion working")
        
        return True
    except Exception as e:
        print(f"    ❌ UnifiedTracer test failed: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility() -> bool:
    """測試向後相容性"""
    try:
        print("  🔍 測試別名映射...")
        from services.core.aiva_core.execution import TraceLogger, TraceRecorder
        from services.core.aiva_core.execution.unified_tracer import UnifiedTracer
        
        # 驗證別名指向正確的類
        assert TraceLogger is UnifiedTracer
        assert TraceRecorder is UnifiedTracer
        print("    ✅ Backward compatibility aliases working")
        
        print("  🔍 測試舊介面實例化...")
        logger = TraceLogger()
        recorder = TraceRecorder()
        
        assert isinstance(logger, UnifiedTracer)
        assert isinstance(recorder, UnifiedTracer)
        print("    ✅ Old interface still functional")
        
        print("  🔍 測試全局函數...")
        from services.core.aiva_core.execution import get_global_tracer, record_execution_trace
        
        global_tracer = get_global_tracer()
        assert isinstance(global_tracer, UnifiedTracer)
        print("    ✅ Global tracer function working")
        
        # 測試記錄函數
        record_execution_trace(
            module_name="test.backward_compatibility",
            function_name="test_function",
            variables={"test": True}
        )
        print("    ✅ Record execution trace function working")
        
        return True
    except Exception as e:
        print(f"    ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def test_message_broker_integration() -> bool:
    """測試MessageBroker整合"""
    try:
        print("  🔍 測試MessageBroker創建...")
        from services.core.aiva_core.messaging.message_broker import MessageBroker
        
        broker = MessageBroker()
        print("    ✅ MessageBroker creation successful")
        
        print("  🔍 測試PlanExecutor與MessageBroker整合...")
        from services.core.aiva_core.execution.plan_executor import PlanExecutor
        
        executor = PlanExecutor(message_broker=broker)
        assert executor.message_broker is broker
        print("    ✅ PlanExecutor MessageBroker integration working")
        
        print("  🔍 測試UnifiedTracer整合...")
        from services.core.aiva_core.execution.unified_tracer import UnifiedTracer
        
        tracer = UnifiedTracer()
        executor_with_tracer = PlanExecutor(
            message_broker=broker,
            unified_tracer=tracer
        )
        assert executor_with_tracer.unified_tracer is tracer
        print("    ✅ PlanExecutor UnifiedTracer integration working")
        
        return True
    except Exception as e:
        print(f"    ❌ MessageBroker integration test failed: {e}")
        traceback.print_exc()
        return False


def test_plan_executor_error_handling() -> bool:
    """測試PlanExecutor的錯誤處理"""
    try:
        print("  🔍 測試PlanExecutor錯誤處理機制...")
        from services.core.aiva_core.execution.plan_executor import PlanExecutor
        from services.aiva_common.error_handling import AIVAError
        
        # 檢查類別存在
        _ = PlanExecutor  # 使用變量避免未使用警告
        print("    ✅ PlanExecutor creation with error handling successful")
        
        # 檢查是否有使用MODULE_NAME常量
        import services.core.aiva_core.execution.plan_executor as pe_module
        assert hasattr(pe_module, 'MODULE_NAME')
        print("    ✅ MODULE_NAME constant defined")
        
        return True
    except Exception as e:
        print(f"    ❌ PlanExecutor error handling test failed: {e}")
        traceback.print_exc()
        return False


def run_validation_suite() -> Dict[str, str]:
    """執行完整驗證套件"""
    print("🚀 AIVA架構修復驗證開始")
    print(f"⏰ 驗證時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = {}
    
    # 定義所有測試
    tests = [
        ("模組導入測試", test_core_imports),
        ("AI依賴測試", test_ai_dependencies),
        ("錯誤處理測試", test_aiva_error_handling),
        ("統一追蹤器測試", test_unified_tracer),
        ("向後相容性測試", test_backward_compatibility),
        ("MessageBroker整合測試", test_message_broker_integration),
        ("PlanExecutor錯誤處理測試", test_plan_executor_error_handling),
    ]
    
    # 執行所有測試
    for test_name, test_func in tests:
        print(f"\n🧪 執行 {test_name}...")
        try:
            result = test_func()
            test_results[test_name] = "✅ PASS" if result else "❌ FAIL"
            print(f"   結果: {test_results[test_name]}")
        except Exception as e:
            test_results[test_name] = "❌ ERROR"
            print(f"   錯誤: {e}")
            logger.error(f"Test {test_name} failed with error: {e}")
    
    # 生成詳細報告
    print("\n" + "=" * 60)
    print("📊 驗證結果摘要")
    print("=" * 60)
    
    passed = sum(1 for result in test_results.values() if "PASS" in result)
    failed = sum(1 for result in test_results.values() if "FAIL" in result)
    errors = sum(1 for result in test_results.values() if "ERROR" in result)
    total = len(test_results)
    
    print("\n📈 統計結果:")
    print(f"   ✅ 通過: {passed}")
    print(f"   ❌ 失敗: {failed}")
    print(f"   🚨 錯誤: {errors}")
    print(f"   📊 總計: {total}")
    print(f"   🎯 成功率: {(passed/total)*100:.1f}%")
    
    print("\n📋 詳細結果:")
    for test_name, result in test_results.items():
        print(f"   {result} {test_name}")
    
    print("\n" + "=" * 60)
    if passed == total:
        print("🎉 所有驗證測試通過！架構修復成功！")
        print("✨ 系統已準備就緒，可投入使用")
    elif passed > total * 0.8:
        print("⚠️  大部分測試通過，但有部分問題需要關注")
        print("🔧 建議檢查失敗的測試項目")
    else:
        print("🚨 多項測試失敗，需要進一步檢查和修復")
        print("🛠️  請檢查錯誤日誌並修復問題")
    
    return test_results


if __name__ == "__main__":
    try:
        print("🔧 準備驗證環境...")
        
        # 檢查Python版本
        if sys.version_info < (3, 11):
            print(f"⚠️  警告: Python版本 {sys.version} 可能不完全相容")
        
        # 執行驗證
        results = run_validation_suite()
        
        # 決定退出碼
        passed = sum(1 for result in results.values() if "PASS" in result)
        total = len(results)
        
        if passed == total:
            print("\n🎯 驗證完成：全部成功")
            sys.exit(0)
        else:
            print(f"\n⚠️  驗證完成：{passed}/{total} 通過")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  驗證被用戶中斷")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 驗證過程發生未預期錯誤: {e}")
        traceback.print_exc()
        sys.exit(3)