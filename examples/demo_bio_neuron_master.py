"""
BioNeuron Master - 三種操作模式演示

展示如何通過 UI、AI 自主、對話三種方式控制 AIVA
"""

import asyncio
import logging

from services.core.aiva_core.bio_neuron_master import (
    BioNeuronMasterController,
    OperationMode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_ui_mode(controller: BioNeuronMasterController):
    """演示 UI 模式"""
    print("\n" + "=" * 60)
    print("🖥️  UI 模式演示 - 需要用戶確認")
    print("=" * 60)

    # 切換到 UI 模式
    controller.switch_mode(OperationMode.UI)

    # 模擬 UI 操作
    result = await controller.process_request(
        request={
            "action": "start_scan",
            "params": {
                "target": "http://testphp.vulnweb.com",
                "auto_confirm": True,  # 演示用自動確認
            },
        }
    )

    print(f"\n結果: {result}")


async def demo_ai_mode(controller: BioNeuronMasterController):
    """演示 AI 自主模式"""
    print("\n" + "=" * 60)
    print("🤖 AI 自主模式演示 - 完全自動")
    print("=" * 60)

    # 切換到 AI 模式
    controller.switch_mode(OperationMode.AI)

    # AI 自主決策和執行
    result = await controller.process_request(
        request={
            "objective": "對目標進行全面安全評估",
            "target": {
                "url": "http://testphp.vulnweb.com",
                "type": "web_application",
            },
        }
    )

    print(f"\n結果: {result}")


async def demo_chat_mode(controller: BioNeuronMasterController):
    """演示對話模式"""
    print("\n" + "=" * 60)
    print("💬 對話模式演示 - 自然語言交互")
    print("=" * 60)

    # 切換到對話模式
    controller.switch_mode(OperationMode.CHAT)

    # 模擬多輪對話
    conversations = [
        "你好，我想進行安全掃描",
        "掃描 http://testphp.vulnweb.com",
        "查看目前狀態",
        "開始訓練",
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n--- 對話 {i} ---")
        print(f"用戶: {user_input}")

        result = await controller.process_request(request=user_input)

        print(f"AI: {result.get('message', result)}")


async def demo_hybrid_mode(controller: BioNeuronMasterController):
    """演示混合模式"""
    print("\n" + "=" * 60)
    print("🔀 混合模式演示 - 智能切換")
    print("=" * 60)

    # 切換到混合模式
    controller.switch_mode(OperationMode.HYBRID)

    # 不同風險級別的請求
    requests = [
        ("查看狀態", "低風險 → AI 自動"),
        ("掃描目標", "中風險 → 對話確認"),
        ("刪除數據", "高風險 → UI 確認"),
    ]

    for request, description in requests:
        print(f"\n--- {description} ---")
        print(f"請求: {request}")

        result = await controller.process_request(request=request)

        print(f"處理方式: {result.get('response_type', 'auto')}")
        print(f"結果: {result.get('message', result)}")


async def main():
    """主程序"""
    print("=" * 60)
    print("🧠 BioNeuron Master Controller 演示")
    print("=" * 60)
    print("\n初始化 BioNeuronRAGAgent 主控系統...")

    # 初始化主控器
    controller = BioNeuronMasterController(
        codebase_path="/workspaces/AIVA",
        default_mode=OperationMode.HYBRID,
    )

    # 註冊 UI 回調（模擬）
    def mock_ui_update(data):
        print(f"[UI Update] {data}")

    async def mock_confirmation(action, params):
        print(f"[UI Confirmation] Action: {action}, Params: {params}")
        return {"confirmed": True}

    controller.register_ui_callback("ui_update", mock_ui_update)
    controller.register_ui_callback("request_confirmation", mock_confirmation)

    # 運行各種模式演示
    try:
        # 1. UI 模式
        await demo_ui_mode(controller)
        await asyncio.sleep(1)

        # 2. AI 自主模式
        await demo_ai_mode(controller)
        await asyncio.sleep(1)

        # 3. 對話模式
        await demo_chat_mode(controller)
        await asyncio.sleep(1)

        # 4. 混合模式
        await demo_hybrid_mode(controller)

        # 顯示系統狀態
        print("\n" + "=" * 60)
        print("📊 系統狀態")
        print("=" * 60)
        status = controller._get_system_status()
        print(controller._format_status_message(status))

        # 顯示對話歷史
        print("\n" + "=" * 60)
        print("📜 對話歷史 (最近 5 條)")
        print("=" * 60)
        history = controller.get_conversation_history(limit=5)
        for i, entry in enumerate(history, 1):
            print(
                f"{i}. [{entry['role']}] {entry['timestamp']}: "
                f"{str(entry['content'])[:100]}..."
            )

    except Exception as e:
        logger.error(f"演示過程出錯: {e}", exc_info=True)

    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
