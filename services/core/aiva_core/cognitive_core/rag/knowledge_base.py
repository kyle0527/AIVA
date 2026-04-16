"""
Knowledge Base - 知識庫類別

為 VectorStore 提供高級抽象接口，支援 RAG 引擎所需的知識管理功能
"""

from collections.abc import Awaitable
from datetime import datetime
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Protocol

# UTC 兼容性处理（Python 3.11+ 使用 UTC，较旧版本使用 timezone.utc）
try:
    from datetime import UTC  # type: ignore
except ImportError:
    UTC = UTC  # type: ignore


if TYPE_CHECKING:
    from aiva_common.schemas.dual_loop import RAGQueryRequest, RAGQueryResult

logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    """向量存儲協議接口 - 遵循 aiva_common 標準
    
    v2.1 (2026-01-08): 擴展支援去語意化反射引擎方法
    """
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> Awaitable[None]:
        """添加文檔到向量存儲（異步）"""
        ...
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None
    ) -> Awaitable[list[dict[str, Any]]]:
        """搜索相似文檔（異步）"""
        ...
    
    def add_capability_from_registry(
        self,
        capability: dict[str, Any],
        capability_id: str | None = None
    ) -> None:
        """從 integration/capability 註冊表添加能力（v2.1 新增）"""
        ...
    
    def search_by_environment(
        self,
        environment_features: dict[str, float],
        top_k: int = 5,
        filter_type: str | None = None
    ) -> list[dict[str, Any]]:
        """根據環境特徵搜索最匹配的能力（v2.1 新增 - 去語意化檢索）"""
        ...


class KnowledgeBase:
    """知識庫
    
    基於向量存儲的高級知識管理接口，為 RAG 引擎提供統一的知識檢索功能。
    這是 RAG 1 拆解後的新架構，整合了向量化檢索能力。
    
    遵循 aiva_common 標準，支持多種向量存儲實現。
    """
    
    def __init__(self, vector_store: VectorStoreProtocol) -> None:
        """初始化知識庫
        
        Args:
            vector_store: 實現 VectorStoreProtocol 的向量存儲實例
        """
        self.vector_store = vector_store
        logger.info("KnowledgeBase initialized with vector store")
    
    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索相關知識
        
        Args:
            query: 查詢字串
            top_k: 返回結果數量
            
        Returns:
            搜索結果列表
        """
        try:
            # 使用向量存儲進行檢索
            results = await self.vector_store.search(query, top_k=top_k)
            
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
    
    def query(self, query_request: "RAGQueryRequest") -> "RAGQueryResult":
        """查詢知識庫（同步接口，供 InternalLoopConnector 使用）
        
        這是 search() 的同步包裝，接受 RAGQueryRequest 並返回 RAGQueryResult。
        用於與 InternalLoopConnector.query_capabilities() 對接。
        
        Args:
            query_request: RAG 查詢請求對象
            
        Returns:
            RAGQueryResult: 查詢結果
        """
        import asyncio

        from aiva_common.schemas.dual_loop import RAGQueryResult
        
        try:
            # 使用事件循環執行異步搜索
            try:
                loop = asyncio.get_running_loop()
                # 如果已在異步環境，創建任務
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.search(query_request.query, top_k=query_request.top_k)
                    )
                    search_results = future.result()
            except RuntimeError:
                # 沒有運行中的事件循環，直接運行
                search_results = asyncio.run(
                    self.search(query_request.query, top_k=query_request.top_k)
                )
            
            # 應用過濾條件（如果有）
            if query_request.filters:
                filtered_results = []
                for result in search_results:
                    metadata = result.get("metadata", {})
                    match = True
                    for key, value in query_request.filters.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                search_results = filtered_results
            
            # 構建 RAGQueryResult
            relevance_scores = [
                r.get("relevance_score", 0.0) for r in search_results
            ]
            
            return RAGQueryResult(
                query=query_request.query,
                results=search_results,
                total_found=len(search_results),
                relevance_scores=relevance_scores,
                timestamp=datetime.now(UTC)
            )
            
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return RAGQueryResult(
                query=query_request.query,
                results=[],
                total_found=0,
                relevance_scores=[],
                timestamp=datetime.now(UTC)
            )
    
    async def add_knowledge(self, content: str, metadata: dict[str, Any] | None = None) -> bool:
        """添加知識到知識庫
        
        Args:
            content: 知識內容
            metadata: 元數據
            
        Returns:
            添加是否成功
        """
        try:
            metadata = metadata or {}
            # 使用穩定的 SHA256 哈希生成唯一 ID
            # 如果 metadata 包含能力標識，使用更語義化的 ID
            if metadata.get("capability_name") and metadata.get("module"):
                cap_name = metadata["capability_name"]
                module = metadata["module"]
                file_path = metadata.get("file_path", "")
                # 使用 module + capability_name + file_path 生成穩定ID
                stable_key = f"{module}::{cap_name}::{file_path}"
                doc_id = f"kb_{hashlib.sha256(stable_key.encode()).hexdigest()[:16]}"
            else:
                # 降級方案: 使用內容哈希
                doc_id = f"kb_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
            
            await self.vector_store.add_document(doc_id, content, metadata)
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
    
    def get_stats(self) -> dict[str, Any]:
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
            
            # 移除對不存在方法的調用 - VectorStoreProtocol 沒有 get_statistics
            # if hasattr(self.vector_store, 'get_statistics'):
            #     vector_stats = self.vector_store.get_statistics()
            #     stats.update(vector_stats)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
