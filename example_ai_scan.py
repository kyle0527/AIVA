"""
簡單示例: AI 如何使用新架構下達掃描命令

這個示例展示:
1. 最簡單的使用方式
2. AI 如何決策並下令
3. 完全移除 RabbitMQ 的乾淨架構
"""

import asyncio
from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import AICommand, CommandType
from services.scan import ScanCommandHandler


async def ai_scan_workflow():
    """模擬 AI 的完整掃描工作流程"""
    
    # 1. 初始化 (系統啟動時執行一次)
    command_center = get_command_center()
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    
    print("🤖 AI: 收到用戶請求 - 掃描 example.com")
    print("=" * 60)
    
    # 2. AI 決策: 先執行 Phase 0 快速偵察
    print("\n🧠 AI 決策: 執行 Phase 0 快速偵察")
    
    phase0_command = AICommand(
        command_id="scan_001_phase0",
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={
            "scan_id": "scan_001_example",
            "targets": ["https://example.com"],
            "max_depth": 3,
            "timeout": 300
        },
        trace_id=None,
        session_id=None,
        parent_command_id=None,
        callback_url=None
    )
    
    # 3. 下達命令 (直接調用,無需 RabbitMQ)
    print("📤 AI: 下達 Phase 0 掃描命令...")
    phase0_result = await command_center.execute(phase0_command)
    
    if not phase0_result.success:
        print(f"❌ Phase 0 失敗: {phase0_result.error}")
        return
    
    # 4. AI 分析結果
    print(f"\n✅ Phase 0 完成 ({phase0_result.execution_time:.2f}s)")
    print(f"   發現資產: {len(phase0_result.result.get('assets', []))} 個")
    
    recommendations = phase0_result.result.get('recommendations', {})
    
    # 5. AI 決策: 是否需要 Phase 1
    needs_phase1 = (
        len(phase0_result.result.get('assets', [])) > 5 or
        recommendations.get('needs_js_engine') or
        recommendations.get('needs_form_testing')
    )
    
    if needs_phase1:
        print("\n🧠 AI 決策: 需要 Phase 1 深度掃描")
        print(f"   建議引擎: {', '.join(recommendations.get('suggested_engines', []))}")
        
        # 6. 下達 Phase 1 命令
        phase1_command = AICommand(
            command_id="scan_001_phase1",
            command_type=CommandType.SCAN_PHASE1,
            target_module="scan",
            payload={
                "scan_id": "scan_001",
                "targets": ["https://example.com"],
                "selected_engines": recommendations.get('suggested_engines', ['rust']),
                "max_depth": 5,
                "max_pages": 1000,
                "phase0_result": phase0_result.result
            },
            trace_id=None,
            session_id=None,
            parent_command_id="scan_001_phase0",
            callback_url=None
        )
        
        print("📤 AI: 下達 Phase 1 掃描命令...")
        phase1_result = await command_center.execute(phase1_command)
        
        if phase1_result.success:
            print(f"\n✅ Phase 1 完成 ({phase1_result.execution_time:.2f}s)")
            print(f"   總資產: {len(phase1_result.result.get('assets', []))} 個")
        else:
            print(f"❌ Phase 1 失敗: {phase1_result.error}")
    else:
        print("\n🧠 AI 決策: Phase 0 結果已充足,無需 Phase 1")
    
    # 7. 顯示統計
    stats = command_center.get_stats()
    print("\n📊 本次會話統計:")
    print(f"   執行命令: {stats['total_commands']}")
    print(f"   成功率: {stats['success_rate'] * 100:.1f}%")
    print(f"   平均耗時: {stats['average_execution_time']:.2f}s")


if __name__ == "__main__":
    print("🚀 AI 掃描工作流程示例 (無 RabbitMQ)")
    print("=" * 60)
    asyncio.run(ai_scan_workflow())
