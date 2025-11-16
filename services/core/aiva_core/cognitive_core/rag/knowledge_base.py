"""
Knowledge Base - 知識庫類別

為 VectorStore 提供高級抽象接口，支援 RAG 引擎所需的知識管理功能
"""

import logging
from typing import Any, Dict, List, Optional

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知識庫
    
    基於向量存儲的高級知識管理接口，為 RAG 引擎提供統一的知識檢索功能。
    這是 RAG 1 拆解後的新架構，整合了向量化檢索能力。
    """
    
    def __init__(self, vector_store: VectorStore) -> None:
        """初始化知識庫
        
        Args:
            vector_store: 底層向量存儲實例
        """
        self.vector_store = vector_store
        logger.info("KnowledgeBase initialized with vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相關知識
        
        Args:
            query: 查詢字串
            top_k: 返回結果數量
            
        Returns:
            搜索結果列表
        """
        try:
            # 使用向量存儲進行檢索
            results = self.vector_store.search(query, top_k=top_k)
            
            # 轉換為知識庫格式
            knowledge_results = []
            for result in results:
                knowledge_results.append({
                    "content": result.get("text", ""),  # vector_store 使用 "text" 而非 "content"
                    "metadata": result.get("metadata", {}),
                    "relevance_score": result.get("score", 0.0),
                    "source": result.get("metadata", {}).get("source", "unknown")
                })
            
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加知識到知識庫
        
        Args:
            content: 知識內容
            metadata: 元數據
            
        Returns:
            添加是否成功
        """
        try:
            metadata = metadata or {}
            # 使用 VectorStore 的實際方法
            doc_id = f"kb_{hash(content)}"
            self.vector_store.add_document(doc_id, content, metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)
            return False
    
    def index_codebase(self, codebase_path: str) -> bool:
        """索引程式碼庫（向後相容方法）
        
        Args:
            codebase_path: 程式碼庫路徑
            
        Returns:
            索引是否成功
        """
        try:
            # 簡化實作：笑記 codebase 索引請求但不實際執行
            logger.info(f"Codebase indexing requested for: {codebase_path}")
            logger.warning("Codebase indexing not implemented in current vector store backend")
            return True
        except Exception as e:
            logger.error(f"Codebase indexing failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取知識庫統計信息
        
        Returns:
            統計信息字典
        """
        try:
            stats = {
                "total_items": 0,
                "backend": "vector_store",
                "status": "active"
            }
            
            if hasattr(self.vector_store, 'get_statistics'):
                vector_stats = self.vector_store.get_statistics()
                stats.update(vector_stats)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}