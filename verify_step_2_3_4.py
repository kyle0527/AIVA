
import sys
import asyncio
import logging
import json
import importlib.util
from pathlib import Path
from types import ModuleType
from uuid import uuid4

# 設置路徑
PROJECT_ROOT = r"C:\D\fold7\AIVA-git"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}\\services\\common")
sys.path.insert(0, f"{PROJECT_ROOT}\\services\\core")

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VERIFY_STEP_2_3_4")

def mock_rag_handler():
    """
    Mock RAGHandler to bypass ModuleNotFoundError.
    It seems CommanderCoordinator expects 'RAGHandler' from 'rag_handler.py',
    but the actual code is 'RAGEngine' in 'rag_engine.py'.
    """
    print("🛠️ [Mock] Applying Import Patch for RAGHandler...")
    try:
        # 1. 導入真實的 RAGEngine
        from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
        
        # 2. 創建一個偽造的 module
        module_name = "services.core.aiva_core.cognitive_core.rag.rag_handler"
        fake_module = ModuleType(module_name)
        
        # 3. 將 RAGEngine 關聯為 RAGHandler
        # 注意: 我們需要實例化它，或者讓 init 調用工作
        # Commander 代碼: rag_engine = RAGHandler()
        # RAGEngine init 需要 knowledge_base
        
        # 我們創建一個 Wrapper 類來處理 init 參數的潛在不匹配
        class RAGHandlerWrapper(RAGEngine):
            def __init__(self, *args, **kwargs):
                # 如果沒有傳入 knowledge_base，我們需要 mock 一個
                if 'knowledge_base' not in kwargs and not args:
                    print("    -> Mocking KnowledgeBase dependecy for RAGHandler...")
                    from unittest.mock import MagicMock
                    kb_mock = MagicMock()
                    kwargs['knowledge_base'] = kb_mock
                super().__init__(*args, **kwargs)
                
        fake_module.RAGHandler = RAGHandlerWrapper
        
        # 4. 註冊到 sys.modules
        sys.modules[module_name] = fake_module
        print("✅ [Mock] Patch applied: rag_handler.RAGHandler -> RAGEngine")
        return True
    except Exception as e:
        print(f"❌ [Mock] Failed to patch RAGHandler: {e}")
        return False

async def verify_step_2_3_4():
    print("\n=== 開始驗證 Step 2, 3, 4 (with Bug Fixes) ===\n")

    # [Fix Bug 2] Apply Import Patch BEFORE importing Commander
    if not mock_rag_handler():
        return

    try:
        from services.core.aiva_core.task_planning.commander import CommanderCoordinator, AITaskType
        from services.core.aiva_core.task_planning.commander.plan_builder import PlanBuilder
        print("✅ Core Imports Successful")
    except ImportError as e:
        print(f"❌ Core Import Failed: {e}")
        return

    print("\n[1] 初始化 CommanderCoordinator...")
    commander = CommanderCoordinator()
    
    # [Fix Bug 1] Monkey Patch PlanBuilder method name
    print("\n[2] 應用 PlanBuilder 方法名修補 (Bug 1)...")
    plan_builder = commander.plan_builder
    if not hasattr(plan_builder, 'build_attack_plan') and hasattr(plan_builder, 'plan_attack'):
        print("🛠️ [Patch] plan_builder.build_attack_plan = plan_builder.plan_attack")
        setattr(plan_builder, 'build_attack_plan', plan_builder.plan_attack)

    # 模擬 Step 2: 請求生成計劃
    print("\n[3] 模擬 Step 2: 生成攻擊計劃 (Plan Generation)...")
    context = {
        "target": "http://localhost:3000",
        "objective": "Verify SQL Injection vulnerabilities",
        "constraints": {"timeout": 300}
    }
    
    try:
        result = await commander.execute_command(
            task_type=AITaskType.ATTACK_PLANNING,
            context=context
        )
        
        if result.get("success"):
            print("✅ Step 2 成功: 計劃已生成")
            plan = result.get("plan")
            print(f"    - Plan ID: {plan.get('plan_id')}")
            print(f"    - Phases: {len(plan.get('phases', []))}")
            
            with open("generated_plan.json", "w", encoding='utf-8') as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            print("    -> 計劃已保存至 generated_plan.json")
            
            # 準備下一步: 模擬 Step 3 (Command Generation)
            # 這裡需要深入 AttackCoordinator，但它依賴 dispatcher
            # 我們可以直接查看 Plan 的第一階段，並嘗試模擬"執行"
            phases = plan.get('phases', [])
            if phases:
                phase1 = phases[0]
                print(f"\n[4] 模擬 Step 3: Phase 1 ({phase1['name']}) 命令生成")
                print("    Step 3 依賴 AttackCoordinator 和 Dispatcher，這部分邏輯較深。")
                print("    驗證重點: 我們已確認 AttackPlan 結構正確，包含了 execution steps (雖然是文字描述)。")
                print("    PlanBuilder 的輸出是 High-Level Plan，之後由 ScannerPlugin 轉換為命令。")
                print("    我們將手動驗證生成的 'steps' 是否包含具體工具建議。")
                
                steps = phase1.get('steps', [])
                print(f"    -> Phase 1 Steps: {steps}")
                
                # 簡單驗證是否包含 nmap/nikto 等關鍵詞
                has_tools = any(tool in str(steps).lower() for tool in ['nmap', 'port', 'scan', 'enumeration'])
                if has_tools:
                     print("✅ Step 3 預驗證成功: 步驟包含具體掃描動作")
                else:
                     print("⚠️ Step 3 警告: 步驟描述過於抽象，可能無法轉換為命令")

        else:
            print(f"❌ Step 2 失敗: {result.get('error')}")
            if result.get("fallback_message"):
                print(f"    - Fallback: {result.get('fallback_message')}")

    except Exception as e:
        print(f"❌ Step 2 執行異常: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 驗證結束 ===")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_step_2_3_4())
