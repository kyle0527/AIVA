
import sys
import asyncio
import logging
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock

# 設置路徑
PROJECT_ROOT = r"C:\D\fold7\AIVA-git"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}\\services\\common")
sys.path.insert(0, f"{PROJECT_ROOT}\\services\\core")

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VERIFY_STEP_3")

# Mock RAG Handler again to avoid import errors if they persist
import importlib.util
from types import ModuleType
if "services.core.aiva_core.cognitive_core.rag.rag_handler" not in sys.modules:
    try:
        from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
        fake_mod = ModuleType("services.core.aiva_core.cognitive_core.rag.rag_handler")
        class RAGWrapper(RAGEngine):
             def __init__(self, *args, **kwargs):
                 if 'knowledge_base' not in kwargs and not args:
                     kwargs['knowledge_base'] = MagicMock()
                 super().__init__(*args, **kwargs)
        fake_mod.RAGHandler = RAGWrapper
        sys.modules["services.core.aiva_core.cognitive_core.rag.rag_handler"] = fake_mod
    except ImportError:
        pass # Might be handled appropriately later or not needed for this script

async def verify_step_3():
    print("\n=== 開始驗證 Step 3: CLI 命令生成 ===\n")

    try:
        from services.core.aiva_core.cognitive_core.capability_orchestrator import CapabilityOrchestrator, TaskRequirement, CapabilityPlan
        print("✅ CapabilityOrchestrator 導入成功")
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return

    # 1. 準備 Mock 的 RAG 數據 (模擬已註冊的能力)
    mock_capabilities = [
        {
            "metadata": {
                "capability_id": "security.scan.nmap.basic",
                "capability_name": "Nmap Basic Scan",
                "description": "Perform basic port scanning",
                "aiva_module": "core_instruments",
                "cli_template": "nmap -sV -p- {target}", # 關鍵: CLI 模板
                "health_score": 1.0,
                "availability": 1.0
            },
            "text": "Nmap scanning capability",
            "score": 0.95
        },
        {
             "metadata": {
                "capability_id": "security.scan.nikto.basic",
                "capability_name": "Nikto Web Scan",
                "cli_command": "nikto -h {target}", # 另一種格式
                "health_score": 1.0,
                "availability": 1.0
            },
            "score": 0.9
        }
    ]

    # 2. 初始化 Orchestrator 並注入 Mock Connector
    # 我們 Mock InternalLoopConnector 來返回上面的數據
    mock_connector = MagicMock()
    # query_capabilities 可能是異步或同步，取決於內部實現。Code says: await asyncio.to_thread(self.internal_connector.query_capabilities...)
    # So it calls a sync method in a thread.
    mock_connector_query = MagicMock()
    mock_connector_query.results = mock_capabilities # query_capabilities 返回一個對象，該對象有 results 屬性
    
    # 實際上 query_capabilities 返回的是 RAGResult 對象，我們模擬它
    class MockRAGResult:
        def __init__(self, results):
            self.results = results
    
    mock_connector.query_capabilities.return_value = MockRAGResult(mock_capabilities)

    orchestrator = CapabilityOrchestrator(internal_connector=mock_connector)
    print("✅ Orchestrator 初始化 (Mocked Connector)")

    # 3. 定義任務需求
    requirement = TaskRequirement(
        task_id="test_task_step_3",
        task_type="scan",
        target="127.0.0.1",
        objectives=["port scan", "web scan"]
    )

    # 4. 執行規劃
    print("\n[Action] 執行 orchestrator.plan()...")
    try:
        plan = await orchestrator.plan(requirement)
        
        print(f"✅ 計劃生成成功: {plan.plan_id}")
        print(f"   Selected Capabilities: {len(plan.selected_capabilities)}")
        print(f"   CLI Commands Generated: {len(plan.cli_commands)}")
        
        print("\n--- Generated Commands ---")
        for cmd in plan.cli_commands:
            print(f"   > {cmd}")
        print("--------------------------")
        
        if len(plan.cli_commands) > 0 and "nmap" in plan.cli_commands[0]:
            print("\n✨ 驗證成功: 成功從能力元數據生成 CLI 命令")
        else:
            print("\n❌ 驗證失敗: 未生成預期的 CLI 命令")

    except Exception as e:
        print(f"❌ 執行異常: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 驗證結束 ===")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_step_3())
