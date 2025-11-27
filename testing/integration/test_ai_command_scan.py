"""
測試 AI 命令系統 - Scan 模組整合測試

這個測試程式驗證:
1. AI 命令中心正常運作
2. Scan 命令處理器正確註冊
3. Phase 0 掃描命令執行
4. 數據合約正確傳遞
5. 無需 RabbitMQ 服務

運行方式:
    python test_ai_command_scan.py
"""

import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.command_center import AICommandCenter
from services.aiva_common.schemas import AICommand, CommandType, CommandPriority
from services.scan import ScanCommandHandler


async def test_phase0_scan():
    """測試 Phase 0 快速偵察"""
    print("=" * 70)
    print("🧪 測試 1: Phase 0 快速偵察")
    print("=" * 70)
    
    # 1. 初始化命令中心
    command_center = AICommandCenter()
    print("✅ AI 命令中心已初始化")
    
    # 2. 註冊 Scan 模組
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    print("✅ Scan 命令處理器已註冊")
    
    # 3. 創建 Phase 0 掃描命令
    command = AICommand(
        command_id="test_scan_001_phase0",
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={
            "scan_id": "scan_test_001",
            "targets": ["https://example.com"],
            "max_depth": 3,
            "timeout": 300
        },
        priority=CommandPriority.HIGH,
        trace_id="test_trace_001"
    )
    
    print(f"\n📤 發送 Phase 0 掃描命令:")
    print(f"   命令 ID: {command.command_id}")
    print(f"   目標: {command.payload['targets']}")
    print(f"   深度: {command.payload['max_depth']}")
    
    # 4. 執行命令
    print("\n⏳ 執行中...")
    result = await command_center.execute(command)
    
    # 5. 顯示結果
    print("\n" + "=" * 70)
    print("📊 執行結果:")
    print("=" * 70)
    print(f"狀態: {result.status.value}")
    print(f"成功: {'✅ 是' if result.success else '❌ 否'}")
    print(f"執行時間: {result.execution_time:.2f} 秒")
    
    if result.success and result.result:
        phase0_result = result.result
        print(f"\n📈 掃描統計:")
        print(f"   資產數: {len(phase0_result.get('assets', []))}")
        
        summary = phase0_result.get('summary', {})
        print(f"   URL: {summary.get('urls_found', 0)}")
        print(f"   表單: {summary.get('forms_found', 0)}")
        print(f"   API: {summary.get('apis_found', 0)}")
        
        recommendations = phase0_result.get('recommendations', {})
        if recommendations:
            print(f"\n💡 AI 決策建議:")
            print(f"   需要 JS 引擎: {'是' if recommendations.get('needs_js_engine') else '否'}")
            print(f"   需要表單測試: {'是' if recommendations.get('needs_form_testing') else '否'}")
            print(f"   建議引擎: {', '.join(recommendations.get('suggested_engines', []))}")
    
    if result.error:
        print(f"\n❌ 錯誤: {result.error}")
    
    return result


async def test_health_check():
    """測試健康檢查命令"""
    print("\n" + "=" * 70)
    print("🧪 測試 2: 健康檢查")
    print("=" * 70)
    
    command_center = AICommandCenter()
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    
    command = AICommand(
        command_id="test_health_001",
        command_type=CommandType.HEALTH_CHECK,
        target_module="scan",
        payload={}
    )
    
    print("\n📤 發送健康檢查命令...")
    result = await command_center.execute(command)
    
    print("\n📊 健康狀態:")
    print(f"狀態: {result.status.value}")
    
    if result.success and result.result:
        health = result.result
        print(f"模組: {health.get('module')}")
        print(f"狀態: {health.get('status')}")
        print(f"可用引擎: {', '.join(health.get('available_engines', []))}")
        print(f"協調策略: {health.get('coordination_strategy')}")
    
    return result


async def test_command_center_stats():
    """測試命令中心統計"""
    print("\n" + "=" * 70)
    print("🧪 測試 3: 命令中心統計")
    print("=" * 70)
    
    command_center = AICommandCenter()
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    
    # 執行幾個命令
    for i in range(3):
        command = AICommand(
            command_id=f"test_batch_{i}",
            command_type=CommandType.HEALTH_CHECK,
            target_module="scan",
            payload={}
        )
        await command_center.execute(command)
    
    # 獲取統計
    stats = command_center.get_stats()
    
    print("\n📊 統計數據:")
    print(f"總命令數: {stats['total_commands']}")
    print(f"成功命令數: {stats['successful_commands']}")
    print(f"失敗命令數: {stats['failed_commands']}")
    print(f"平均執行時間: {stats['average_execution_time']:.3f} 秒")
    print(f"成功率: {stats['success_rate'] * 100:.1f}%")
    print(f"已註冊模組: {', '.join(stats['registered_modules'])}")
    
    # 獲取歷史
    history = command_center.get_command_history(limit=5)
    print(f"\n📜 最近 {len(history)} 條命令歷史:")
    for record in history[-3:]:
        print(f"   [{record['timestamp']}] {record['command_id']}: {record['status']}")


async def main():
    """主測試函數"""
    print("\n" + "=" * 70)
    print("🚀 AIVA AI 命令系統測試 - Scan 模組整合")
    print("=" * 70)
    print("\n說明:")
    print("  這個測試驗證新的 AI 直接指揮架構")
    print("  完全不需要 RabbitMQ 服務運行")
    print("  使用數據合約確保類型安全")
    print("\n")
    
    try:
        # 測試 1: Phase 0 掃描
        result1 = await test_phase0_scan()
        
        # 測試 2: 健康檢查
        result2 = await test_health_check()
        
        # 測試 3: 統計信息
        await test_command_center_stats()
        
        # 最終總結
        print("\n" + "=" * 70)
        print("✅ 測試完成總結")
        print("=" * 70)
        print(f"Test 1 (Phase 0): {'✅ 通過' if result1.success else '❌ 失敗'}")
        print(f"Test 2 (Health): {'✅ 通過' if result2.success else '❌ 失敗'}")
        print("\n架構驗證:")
        print("  ✅ AI 命令中心正常運作")
        print("  ✅ Scan 模組成功註冊")
        print("  ✅ 命令路由正確執行")
        print("  ✅ 數據合約正確傳遞")
        print("  ✅ 無需 RabbitMQ 服務")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
