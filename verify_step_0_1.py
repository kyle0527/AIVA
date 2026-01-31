
import sys
import asyncio
import logging
from pathlib import Path
from uuid import uuid4

# 設置路徑 (根據環境)
PROJECT_ROOT = r"C:\D\fold7\AIVA-git"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}\\services\\common")
sys.path.insert(0, f"{PROJECT_ROOT}\\services\\core")

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VERIFICATION_SCRIPT")

async def verify_step_0_1():
    print("\n=== 開始驗證 Step 0 & 1 ===\n")

    try:
        print("[1] 嘗試導入核心組件...")
        from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import AIVACoreServiceCoordinator
        from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent, DecisionContext
        print("✅ 導入成功")
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return

    # 模擬 app.py startup
    print("\n[2] 初始化核心服務 (模擬啟動)...")
    coordinator = AIVACoreServiceCoordinator()
    try:
        await coordinator.start()
        print("✅ Coordinator 啟動成功")
    except Exception as e:
        print(f"❌ Coordinator 啟動失敗: {e}")
        # 即使失敗也繼續嘗試 Decision Agent，看是否獨立可用
    
    print("\n[3] 初始化 AI 決策 Agent...")
    try:
        decision_agent = EnhancedDecisionAgent()
        print("✅ EnhancedDecisionAgent 初始化成功")
    except Exception as e:
        print(f"❌ EnhancedDecisionAgent 初始化失敗: {e}")
        return

    # 模擬 Step 0: 接收請求
    target = "http://localhost:3000"
    scan_id = f"test_scan_{uuid4().hex[:8]}"
    print(f"\n[4] 模擬 Step 0: 接收目標 {target}")
    
    # 模擬 Step 1: AI 分析 (app.py start_scan 邏輯)
    print("[5] 模擬 Step 1: AI 初步分析...")
    context = DecisionContext()
    context.target_info = {
        "type": "web",
        "value": target,
        "id": scan_id,
        "scan_type": "quick",
    }
    context.available_tools = ["nmap", "nikto"]

    try:
        print("    -> 調用 make_enhanced_decision...")
        initial_decision = await decision_agent.make_enhanced_decision(
            context=context,
            use_embedded_knowledge=True
        )
        print(f"✅ AI 決策成功: {initial_decision.action} (信心度: {initial_decision.confidence})")
        print(f"    理由: {initial_decision.reasoning}")
    except Exception as e:
        print(f"❌ Step 1 執行失敗: {e}")

    print("\n=== 驗證結束 ===")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_step_0_1())
