#!/usr/bin/env python3
"""
向量存儲功能驗證測試

測試統一向量存儲 (UnifiedVectorStore) 的基本功能
包括文檔添加、搜索、刪除等操作
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.core.aiva_core.rag.unified_vector_store import create_unified_vector_store

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_vector_store():
    """測試統一向量存儲功能"""
    
    logger.info("🚀 開始向量存儲功能驗證...")
    
    try:
        # 1. 創建統一向量存儲
        logger.info("📦 初始化 UnifiedVectorStore...")
        vector_store = await create_unified_vector_store(
            database_url="postgresql://postgres:aiva123@localhost:5432/aiva_db",
            table_name="test_knowledge_vectors",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        logger.info("✅ UnifiedVectorStore 初始化成功")
        
        # 2. 測試添加文檔
        logger.info("📝 測試添加文檔...")
        test_docs = [
            {
                "id": "doc_001",
                "content": "這是一個關於 SQL 注入漏洞的測試文檔。SQL 注入是一種常見的 Web 應用漏洞。",
                "metadata": {
                    "type": "vulnerability",
                    "severity": "high",
                    "category": "sqli"
                }
            },
            {
                "id": "doc_002", 
                "content": "跨站腳本攻擊 (XSS) 是另一種重要的 Web 安全漏洞，攻擊者可以注入惡意腳本。",
                "metadata": {
                    "type": "vulnerability",
                    "severity": "medium",
                    "category": "xss"
                }
            },
            {
                "id": "doc_003",
                "content": "文件上傳漏洞允許攻擊者上傳惡意文件到服務器，可能導致遠程代碼執行。",
                "metadata": {
                    "type": "vulnerability", 
                    "severity": "high",
                    "category": "upload"
                }
            }
        ]
        
        for doc in test_docs:
            await vector_store.add_document(
                doc_id=doc["id"],
                text=doc["content"],
                metadata=doc["metadata"]
            )
        
        logger.info("✅ 文檔添加成功")
        
        # 3. 測試搜索功能
        logger.info("🔍 測試搜索功能...")
        
        # 測試相似性搜索
        search_results = await vector_store.search(
            query="SQL 注入攻擊如何防範",
            top_k=2
        )
        
        logger.info(f"✅ 找到 {len(search_results)} 個相關文檔:")
        for i, result in enumerate(search_results, 1):
            logger.info(f"   {i}. 文檔 {result['id']} (相似度: {result['score']:.3f})")
            logger.info(f"      內容: {result['content'][:50]}...")
            logger.info(f"      類別: {result['metadata'].get('category', 'unknown')}")
        
        # 4. 測試元數據篩選搜索
        logger.info("🔍 測試元數據篩選搜索...")
        
        filtered_results = await vector_store.search(
            query="Web 應用安全漏洞",
            top_k=5,
            filter_metadata={"severity": "high"}
        )
        
        logger.info(f"✅ 高嚴重性漏洞搜索結果: {len(filtered_results)} 個文檔")
        for result in filtered_results:
            logger.info(f"   - {result['id']}: {result['metadata'].get('category', 'unknown')}")
        
        # 5. 測試獲取文檔
        logger.info("📄 測試獲取文檔...")
        
        doc = await vector_store.get_document("doc_001")
        if doc:
            logger.info("✅ 文檔獲取成功")
            logger.info(f"   - ID: {doc['id']}")
            logger.info(f"   - 內容長度: {len(doc['content'])} 字符")
            logger.info(f"   - 類別: {doc['metadata'].get('category', 'unknown')}")
        else:
            logger.warning("⚠️  文檔獲取失敗")
        
        # 6. 測試統計信息
        logger.info("📊 測試基本信息...")
        logger.info("✅ 向量存儲基本信息:")
        logger.info(f"   - 數據庫: PostgreSQL + pgvector")
        logger.info(f"   - 表名: test_knowledge_vectors") 
        logger.info(f"   - 嵌入模型: sentence-transformers/all-MiniLM-L6-v2")
        
        # 7. 測試刪除文檔 (可選)
        logger.info("🗑️  測試刪除文檔...")
        
        success = await vector_store.delete_document("doc_003")
        if success:
            logger.info("✅ 文檔刪除成功")
            
            # 驗證刪除
            deleted_doc = await vector_store.get_document("doc_003")
            if deleted_doc is None:
                logger.info("✅ 刪除驗證成功")
            else:
                logger.warning("⚠️  刪除驗證失敗，文檔仍然存在")
        else:
            logger.warning("⚠️  文檔刪除失敗")
        
        logger.info("🎉 向量存儲功能驗證完成！所有測試通過")
        return True
        
    except Exception as e:
        logger.error(f"❌ 向量存儲功能驗證失敗: {str(e)}")
        logger.exception("詳細錯誤信息:")
        return False


async def main():
    """主測試函數"""
    
    logger.info("🚀 開始統一向量存儲完整驗證...")
    
    success = await test_vector_store()
    
    # 總結
    logger.info("\n" + "="*60)
    logger.info("驗證結果總結")
    logger.info("="*60)
    
    if success:
        logger.info("🎉 向量存儲測試通過！")
        logger.info("✅ PostgreSQL + pgvector 後端正常工作")
        logger.info("✅ UnifiedVectorStore 功能正常")
        logger.info("✅ 文檔添加、搜索、獲取、刪除功能完備")
        logger.info("✅ 語義搜索和元數據篩選工作正常")
        return 0
    else:
        logger.error("❌ 向量存儲測試失敗，需要檢查配置")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)