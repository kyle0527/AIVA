"""測試動態能力調用機制

驗證 TaskExecutor 與 CapabilityRegistry + UnifiedFunctionCaller 的整合

Architecture Fix Note:
- 創建日期: 2025-11-16
- 目的: 測試問題四「規劃器如何實際調用工具」的修復
"""

import asyncio
import logging
import sys

# 設置 UTF-8 輸出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_capability_registry():
    """測試能力註冊表"""
    from services.core.aiva_core.core_capabilities.capability_registry import (
        get_capability_registry,
        initialize_capability_registry,
    )
    
    print("\n" + "="*60)
    print("測試 1: CapabilityRegistry 初始化")
    print("="*60)
    
    # 初始化註冊表
    result = await initialize_capability_registry(force_refresh=False)
    
    print(f"✅ 載入結果:")
    print(f"   - 載入能力數: {result['capabilities_loaded']}")
    print(f"   - 索引模組數: {result['modules_indexed']}")
    print(f"   - 錯誤: {result.get('errors', [])}")
    
    # 獲取註冊表
    registry = get_capability_registry()
    
    # 統計信息
    stats = registry.get_statistics()
    print(f"\n📊 統計信息:")
    print(f"   - 總能力數: {stats['total_capabilities']}")
    print(f"   - 總模組數: {stats['total_modules']}")
    print(f"   - 異步能力數: {stats['async_capabilities']}")
    
    # 列出模組
    modules = registry.list_modules()
    print(f"\n📦 可用模組 ({len(modules)}):")
    for module in modules[:5]:  # 只顯示前 5 個
        caps = registry.list_capabilities(module=module)
        print(f"   - {module}: {len(caps)} 個能力")
    
    # 搜索能力
    search_results = registry.search_capabilities("sql")
    print(f"\n🔍 搜索 'sql': 找到 {len(search_results)} 個結果")
    for cap in search_results[:3]:  # 只顯示前 3 個
        print(f"   - {cap.name} ({cap.module})")
    
    return registry


async def test_unified_function_caller():
    """測試統一功能調用器"""
    from services.core.aiva_core.service_backbone.api.unified_function_caller import (
        UnifiedFunctionCaller,
    )
    
    print("\n" + "="*60)
    print("測試 2: UnifiedFunctionCaller")
    print("="*60)
    
    caller = UnifiedFunctionCaller()
    
    # 測試調用（可能失敗，因為模組可能未運行）
    print("\n🔧 測試模組調用:")
    
    # 測試 Python 模組調用
    try:
        result = await caller.call_function(
            module_name="function_sqli",
            function_name="detect_sqli",
            parameters={"url": "https://example.com/test"}
        )
        
        print(f"   ✅ Python 模組調用:")
        print(f"      - 成功: {result.success}")
        print(f"      - 模組: {result.module_name}")
        print(f"      - 函數: {result.function_name}")
        print(f"      - 執行時間: {result.execution_time:.3f}s")
        if not result.success:
            print(f"      - 錯誤: {result.error}")
    except Exception as e:
        print(f"   ⚠️ Python 模組調用失敗: {e}")
    
    # 測試 Go 模組調用
    try:
        result = await caller.call_function(
            module_name="SSRFDetector",
            function_name="detect_ssrf",
            parameters={"target": "https://example.com"}
        )
        
        print(f"\n   ✅ Go 模組調用:")
        print(f"      - 成功: {result.success}")
        print(f"      - 模組: {result.module_name}")
        print(f"      - 函數: {result.function_name}")
        print(f"      - 執行時間: {result.execution_time:.3f}s")
        if not result.success:
            print(f"      - 錯誤: {result.error}")
    except Exception as e:
        print(f"   ⚠️ Go 模組調用失敗: {e}")
    
    return caller


async def test_task_executor_integration():
    """測試 TaskExecutor 整合"""
    from services.core.aiva_core.task_planning.executor.task_executor import (
        TaskExecutor,
    )
    from services.core.aiva_core.task_planning.planner.task_converter import (
        ExecutableTask,
    )
    from services.core.aiva_core.task_planning.planner.tool_selector import (
        ToolDecision,
    )
    from services.aiva_common.enums import ServiceType
    
    print("\n" + "="*60)
    print("測試 3: TaskExecutor 動態能力調用整合")
    print("="*60)
    
    # 創建 TaskExecutor
    executor = TaskExecutor()
    
    print(f"\n✅ TaskExecutor 初始化:")
    print(f"   - 使用動態調用: {executor.use_dynamic_calling}")
    print(f"   - CapabilityRegistry: {executor.capability_registry is not None}")
    print(f"   - UnifiedFunctionCaller: {executor.function_caller is not None}")
    
    # 創建測試任務
    test_task = ExecutableTask(
        task_id="test_001",
        task_type="sqli",
        parameters={
            "url": "https://example.com/login",
            "parameter": "username",
            "payload": "' OR '1'='1",
        }
    )
    
    tool_decision = ToolDecision(
        service_type=ServiceType.FUNCTION,
        tool_name="detect_sqli",
        reason="SQL injection test",
        confidence=0.9,
    )
    
    print(f"\n🎯 執行測試任務:")
    print(f"   - 任務 ID: {test_task.task_id}")
    print(f"   - 任務類型: {test_task.task_type}")
    print(f"   - 服務類型: {tool_decision.service_type.value}")
    
    # 執行任務
    try:
        result = await executor.execute_task(
            task=test_task,
            tool_decision=tool_decision,
            trace_session_id="test_session_001"
        )
        
        print(f"\n✅ 任務執行結果:")
        print(f"   - 成功: {result.success}")
        print(f"   - 任務 ID: {result.task_id}")
        print(f"   - 輸出: {result.output}")
        if result.error:
            print(f"   - 錯誤: {result.error}")
    except Exception as e:
        print(f"\n❌ 任務執行失敗: {e}")
        import traceback
        traceback.print_exc()


async def test_capability_inference():
    """測試能力推斷"""
    from services.core.aiva_core.task_planning.executor.task_executor import (
        TaskExecutor,
    )
    from services.core.aiva_core.task_planning.planner.task_converter import (
        ExecutableTask,
    )
    
    print("\n" + "="*60)
    print("測試 4: 能力名稱推斷")
    print("="*60)
    
    executor = TaskExecutor()
    
    test_cases = [
        ("sqli", {"vulnerability_type": "sql_injection"}),
        ("xss", {"vulnerability_type": "cross_site_scripting"}),
        ("ssrf", {"target": "https://example.com"}),
        ("idor", {"object_id": "123"}),
        ("generic_test", {}),
    ]
    
    for task_type, params in test_cases:
        task = ExecutableTask(
            task_id=f"test_{task_type}",
            task_type=task_type,
            parameters=params,
        )
        
        inferred_capability = executor._infer_capability_name(task)
        print(f"   - {task_type:20s} → {inferred_capability}")


async def main():
    """主測試函數"""
    print("\n" + "="*60)
    print("🧪 AIVA 動態能力調用機制測試")
    print("問題四修復驗證")
    print("="*60)
    
    try:
        # 測試 1: CapabilityRegistry
        await test_capability_registry()
        
        # 測試 2: UnifiedFunctionCaller
        await test_unified_function_caller()
        
        # 測試 3: TaskExecutor 整合
        await test_task_executor_integration()
        
        # 測試 4: 能力推斷
        await test_capability_inference()
        
        print("\n" + "="*60)
        print("✅ 所有測試完成")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
