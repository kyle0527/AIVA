#!/usr/bin/env python3
"""
測試 CommandHandler 執行方式
演示：AI 核心如何通過 CommandCenter 調用外部模組（無需 CLI）
"""
import asyncio
import sys
from pathlib import Path

# 添加路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas.commands import AICommand, CommandType, CommandStatus
from services.features.function_xss.command_handler import XSSCommandHandler
from services.features.function_idor.command_handler import IDORCommandHandler
from services.features.function_bizlogic.command_handler import BizLogicCommandHandler

print("="*70)
print("🧪 測試：CommandHandler 直接執行（無需 CLI）")
print("="*70)
print()

async def test_command_handler_direct_call():
    """方式 1：直接調用 CommandHandler（推薦）"""
    print("【方式 1】直接調用 CommandHandler")
    print("-"*70)
    
    # 創建處理器
    xss_handler = XSSCommandHandler()
    
    # 創建命令
    command = AICommand(
        command_id="test_xss_001",
        command_type=CommandType.FEATURE_XSS_TEST,
        target_module="features.xss",
        payload={
            "target_url": "http://localhost:3000/search",
            "param": "q"
        },
        timeout=30
    )
    
    print(f"📝 命令: {command.command_id}")
    print(f"   類型: {command.command_type.value}")
    print(f"   目標: {command.payload['target_url']}")
    print()
    
    # 直接執行
    try:
        result = await xss_handler.handle_command(command)
        
        print(f"✅ 執行完成")
        print(f"   狀態: {result.status}")
        print(f"   耗時: {result.execution_time:.2f}s")
        if result.data:
            findings_count = result.data.get("findings_count", 0)
            print(f"   結果: 發現 {findings_count} 個漏洞")
    except Exception as e:
        print(f"❌ 執行失敗: {e}")
    
    print()


async def test_command_center():
    """方式 2：通過 CommandCenter（AI 統一接口）"""
    print("【方式 2】通過 CommandCenter（AI 核心使用）")
    print("-"*70)
    
    # 獲取命令中心
    command_center = get_command_center()
    
    # 註冊處理器
    command_center.register_module("features.xss", XSSCommandHandler())
    command_center.register_module("features.idor", IDORCommandHandler())
    command_center.register_module("features.bizlogic", BizLogicCommandHandler())
    
    print(f"✅ 已註冊 {len(command_center._handlers)} 個處理器")
    print()
    
    # 測試多個模組
    test_commands = [
        AICommand(
            command_id="ai_cmd_001",
            command_type=CommandType.FEATURE_XSS_TEST,
            target_module="features.xss",
            payload={"target_url": "http://localhost:3000/search", "param": "q"},
            timeout=30
        ),
        AICommand(
            command_id="ai_cmd_002",
            command_type=CommandType.FEATURE_IDOR_TEST,
            target_module="features.idor",
            payload={"target_url": "http://localhost:3000/api/user/1"},
            timeout=30
        ),
        AICommand(
            command_id="ai_cmd_003",
            command_type=CommandType.FEATURE_BIZLOGIC_TEST,
            target_module="features.bizlogic",
            payload={"target_url": "http://localhost:3000/api/basket/1"},
            timeout=30
        ),
    ]
    
    # 並發執行
    print("🚀 並發執行 3 個模組...")
    print()
    
    tasks = [command_center.execute(cmd) for cmd in test_commands]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 顯示結果
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"{i}. ❌ {test_commands[i-1].target_module}: {result}")
        else:
            status_icon = "✅" if result.status == CommandStatus.COMPLETED else "❌"
            print(f"{i}. {status_icon} {test_commands[i-1].target_module}")
            print(f"      狀態: {result.status}")
            print(f"      耗時: {result.execution_time:.2f}s")
    
    print()
    
    # 統計
    stats = command_center.get_stats()
    print(f"📊 統計:")
    print(f"   總命令數: {stats['total_commands']}")
    print(f"   成功: {stats['successful_commands']}")
    print(f"   失敗: {stats['failed_commands']}")
    print()


async def main():
    """主測試"""
    
    # 測試 1：直接調用
    await test_command_handler_direct_call()
    
    # 測試 2：通過 CommandCenter
    await test_command_center()
    
    print("="*70)
    print("🎉 測試完成")
    print()
    print("💡 總結:")
    print("   1️⃣  外部模組通過 CommandHandler 執行")
    print("   2️⃣  不需要 subprocess + CLI")
    print("   3️⃣  AI 核心使用 CommandCenter 統一調度")
    print("   4️⃣  同步/異步都由 AI 核心控制")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
