import asyncio
import logging
from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector

async def test_sync():
    # Setup basic logging to see the warning
    logging.basicConfig(level=logging.INFO)
    connector = InternalLoopConnector()
    print("Testing sync_capabilities_to_rag(force_refresh=True)...")
    # This should log a warning but not crash
    await connector.sync_capabilities_to_rag(force_refresh=True, target_module="core")
    print("SUCCESS: Sync executed without crash.")

if __name__ == "__main__":
    asyncio.run(test_sync())
