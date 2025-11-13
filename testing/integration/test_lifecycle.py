#!/usr/bin/env python3
"""
AIVA 工具生命週期管理器測試
驗證基於 HackingTool 的生命週期管理功能
"""

import asyncio
import tempfile
from pathlib import Path

from services.integration.capability.lifecycle import ToolLifecycleManager
from services.integration.capability.models import CapabilityRecord, CapabilityType, CapabilityStatus
from aiva_common.enums import ProgrammingLanguage


def create_test_capability() -> CapabilityRecord:
    """創建測試用的能力記錄"""
    return CapabilityRecord(
        id="test.lifecycle.python_tool",
        name="測試 Python 工具",
        description="用於測試生命週期管理的 Python 工具",
        version="1.0.0",
        module="test",
        language=ProgrammingLanguage.PYTHON,
        capability_type=CapabilityType.DETECTOR,
        entrypoint="test_tool.py",
        status=CapabilityStatus.HEALTHY,
        dependencies=["requests", "click"],
        inputs=[],
        outputs=[],
        tags=["test", "lifecycle"],
        timeout_seconds=30,
        retry_count=3,
        priority=50,
        prerequisites=[]
    )


async def test_lifecycle_operations():
    """測試生命週期操作"""
    print("🧪 開始測試工具生命週期管理器")
    print("=" * 60)
    
    # 初始化管理器
    lifecycle_manager = ToolLifecycleManager()
    
    # 創建測試能力
    test_capability = create_test_capability()
    
    # 手動註冊到註冊中心 (通常由發現過程完成)
    await lifecycle_manager.registry.register_capability(test_capability)
    print(f"✅ 已註冊測試能力: {test_capability.id}")
    
    # 測試 1: 安裝工具
    print(f"\n📦 測試 1: 安裝工具")
    print("-" * 30)
    
    result = await lifecycle_manager.install_tool(test_capability.id)
    print(f"安裝結果: {'成功' if result.success else '失敗'}")
    if result.success:
        print(f"安裝路徑: {result.installation_path}")
        print(f"安裝版本: {result.installed_version}")
        print(f"安裝時間: {result.installation_time_seconds:.2f} 秒")
        if result.dependencies_installed:
            print(f"已安裝依賴: {', '.join(result.dependencies_installed)}")
    else:
        print(f"錯誤訊息: {result.error_message}")
    
    # 測試 2: 健康檢查
    print(f"\n🩺 測試 2: 健康檢查")
    print("-" * 30)
    
    health_info = await lifecycle_manager.health_check_tool(test_capability.id)
    print(f"健康狀態: {'健康' if health_info['success'] else '異常'}")
    print(f"已安裝: {health_info.get('is_installed', False)}")
    print(f"延遲時間: {health_info.get('latency_ms', 0)} ms")
    if health_info.get('error_message'):
        print(f"錯誤訊息: {health_info['error_message']}")
    
    # 測試 3: 更新工具
    print(f"\n🔄 測試 3: 更新工具")
    print("-" * 30)
    
    update_success = await lifecycle_manager.update_tool(test_capability.id)
    print(f"更新結果: {'成功' if update_success else '失敗'}")
    
    # 測試 4: 查看事件歷史
    print(f"\n📜 測試 4: 事件歷史")
    print("-" * 30)
    
    events = lifecycle_manager.get_lifecycle_events(
        capability_id=test_capability.id,
        limit=10
    )
    
    print(f"事件總數: {len(events)}")
    for i, event in enumerate(events[:5], 1):  # 顯示前5個事件
        print(f"{i}. {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
              f"{event.event_type} - {event.status}")
        if event.error_message:
            print(f"   錯誤: {event.error_message}")
    
    # 測試 5: 批量健康檢查
    print(f"\n🩺 測試 5: 批量健康檢查")
    print("-" * 30)
    
    batch_result = await lifecycle_manager.batch_health_check([test_capability.id])
    print(f"檢查工具數: {batch_result['total_tools']}")
    print(f"健康工具數: {batch_result['healthy_tools']}")
    print(f"健康率: {batch_result['health_rate']:.1%}")
    
    # 測試 6: 卸載工具
    print(f"\n🗑️ 測試 6: 卸載工具")
    print("-" * 30)
    
    uninstall_success = await lifecycle_manager.uninstall_tool(
        test_capability.id, 
        remove_dependencies=True
    )
    print(f"卸載結果: {'成功' if uninstall_success else '失敗'}")
    
    # 最終事件總結
    print(f"\n📊 測試總結")
    print("-" * 30)
    
    final_events = lifecycle_manager.get_lifecycle_events(
        capability_id=test_capability.id
    )
    
    event_counts = {}
    for event in final_events:
        event_type = event.event_type
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print("事件統計:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count} 次")
    
    print(f"\n🎉 測試完成！")


def test_multi_language_support():
    """測試多語言支援"""
    print("\n🌐 測試多語言支援")
    print("=" * 60)
    
    lifecycle_manager = ToolLifecycleManager()
    
    # 測試不同語言的工具
    test_languages = [
        (ProgrammingLanguage.PYTHON, "test_python", "Python 測試工具"),
        (ProgrammingLanguage.GO, "test_go", "Go 測試工具"),
        (ProgrammingLanguage.RUST, "test_rust", "Rust 測試工具"),
        (ProgrammingLanguage.RUBY, "test_ruby", "Ruby 測試工具"),
        (ProgrammingLanguage.JAVASCRIPT, "test_js", "JavaScript 測試工具"),
        (ProgrammingLanguage.PHP, "test_php", "PHP 測試工具"),
    ]
    
    for language, tool_id, tool_name in test_languages:
        print(f"\n測試 {language.value} 工具...")
        
        # 檢查安裝路徑配置
        installation_path = lifecycle_manager.installation_paths.get(language)
        if installation_path:
            print(f"✅ {language.value} 安裝路徑: {installation_path}")
            print(f"   路徑存在: {installation_path.exists()}")
        else:
            print(f"❌ {language.value} 安裝路徑未配置")
    
    print(f"\n🎉 多語言支援測試完成！")


async def main():
    """主測試函數"""
    try:
        await test_lifecycle_operations()
        test_multi_language_support()
    except Exception as e:
        print(f"❌ 測試過程中出現錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())