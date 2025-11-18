"""Self-Awareness Update Script - 自我認知更新腳本

定期執行此腳本，將最新的能力分析結果更新到 RAG 知識庫

使用方式:
    python scripts/update_self_awareness.py

或在代碼中:
    from scripts.update_self_awareness import update_self_awareness
    result = await update_self_awareness()
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def update_self_awareness(force_refresh: bool = False) -> dict:
    """更新自我認知知識庫
    
    Args:
        force_refresh: 是否強制刷新（清空舊數據）
        
    Returns:
        更新結果統計
    """
    try:
        # 導入必要的組件
        from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
        from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
        from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
        
        logger.info("=" * 60)
        logger.info("🧠 AIVA Self-Awareness Update Starting...")
        logger.info("=" * 60)
        
        # 初始化組件
        logger.info("\n📦 Initializing components...")
        vector_store = VectorStore()
        kb = KnowledgeBase(vector_store=vector_store)
        connector = InternalLoopConnector(rag_knowledge_base=kb)
        
        # 執行同步
        logger.info("\n🔄 Synchronizing capabilities to RAG...")
        result = await connector.sync_capabilities_to_rag(force_refresh=force_refresh)
        
        # 顯示結果
        logger.info("\n" + "=" * 60)
        logger.info("✅ Self-Awareness Update Completed!")
        logger.info("=" * 60)
        logger.info(f"📊 Statistics:")
        logger.info(f"   - Modules scanned:      {result['modules_scanned']}")
        logger.info(f"   - Capabilities found:   {result['capabilities_found']}")
        logger.info(f"   - Documents added:      {result['documents_added']}")
        logger.info(f"   - Timestamp:            {result['timestamp']}")
        logger.info(f"   - Success:              {result['success']}")
        
        if not result['success']:
            logger.error(f"   - Error:                {result.get('error', 'Unknown')}")
        
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Self-awareness update failed: {e}", exc_info=True)
        return {
            "modules_scanned": 0,
            "capabilities_found": 0,
            "documents_added": 0,
            "success": False,
            "error": str(e)
        }


async def test_self_awareness_query():
    """測試自我認知查詢能力"""
    try:
        from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
        from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
        from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
        
        logger.info("\n" + "=" * 60)
        logger.info("🧪 Testing Self-Awareness Query...")
        logger.info("=" * 60)
        
        # 初始化
        vector_store = VectorStore()
        kb = KnowledgeBase(vector_store=vector_store)
        connector = InternalLoopConnector(rag_knowledge_base=kb)
        
        # 測試查詢
        test_queries = [
            "我有哪些攻擊能力",
            "掃描相關的功能",
            "payload 生成",
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Query: '{query}'")
            results = await connector.query_self_awareness(query, top_k=3)
            
            if results:
                logger.info(f"   Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    cap_name = result.get("metadata", {}).get("capability_name", "Unknown")
                    module = result.get("metadata", {}).get("module", "Unknown")
                    logger.info(f"   {i}. {cap_name} (from {module})")
            else:
                logger.info("   No results found")
        
        logger.info("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Self-awareness query test failed: {e}", exc_info=True)


async def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Self-Awareness Update Script")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh (clear old data)"
    )
    parser.add_argument(
        "--test-query",
        action="store_true",
        help="Run self-awareness query tests after update"
    )
    
    args = parser.parse_args()
    
    # 執行更新
    result = await update_self_awareness(force_refresh=args.force_refresh)
    
    # 可選：測試查詢
    if args.test_query and result["success"]:
        await test_self_awareness_query()
    
    # 返回退出碼
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
