"""
統一向量存儲管理器

將現有的 VectorStore 接口與 PostgreSQL + pgvector 後端整合
遵循 aiva_common 規範，實現統一的向量存儲管理
"""



import logging
from pathlib import Path
from typing import Any

import numpy as np

from .postgresql_vector_store import PostgreSQLVectorStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class UnifiedVectorStore:
    """
    統一向量存儲管理器
    
    功能：
    1. 保持現有 VectorStore 接口的兼容性
    2. 底層使用 PostgreSQL + pgvector 實現統一存儲
    3. 支持從舊的文件式存儲遷移數據
    4. 遵循 aiva_common 標準配置
    """

    def __init__(
        self,
        database_url: str | None = None,
        table_name: str = "vectors",
        embedding_dimension: int = 384,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        legacy_persist_directory: Path | None = None,
    ):
        """
        初始化統一向量存儲

        Args:
            database_url: PostgreSQL 數據庫連接字符串
            table_name: 向量表名稱
            embedding_dimension: 嵌入維度
            embedding_model: 嵌入模型名稱
            legacy_persist_directory: 舊的文件存儲目錄（用於遷移）
        """
        self.database_url = database_url or "postgresql://postgres:aiva123@postgres:5432/aiva_db"
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.embedding_model_name = embedding_model
        self.legacy_persist_directory = legacy_persist_directory

        # PostgreSQL 向量存儲後端
        self.pg_store = PostgreSQLVectorStore(
            database_url=self.database_url,
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
        )

        # 舊的文件存儲（用於遷移）
        self.legacy_store: VectorStore | None = None
        if legacy_persist_directory and legacy_persist_directory.exists():
            self.legacy_store = VectorStore(
                backend="memory",
                persist_directory=legacy_persist_directory,
                embedding_model=embedding_model,
            )

        # 嵌入模型（延遲加載）
        self._embedding_model: Any | None = None

        logger.info(
            f"UnifiedVectorStore initialized: "
            f"database={self.database_url}, "
            f"table={self.table_name}, "
            f"model={self.embedding_model_name}"
        )

    async def initialize(self) -> None:
        """初始化統一向量存儲"""
        # 初始化 PostgreSQL 後端
        await self.pg_store.initialize()
        
        # 如果有舊數據，執行遷移
        if self.legacy_store:
            await self._migrate_from_legacy()

        logger.info("UnifiedVectorStore initialized successfully")

    async def _migrate_from_legacy(self) -> None:
        """從舊的文件存儲遷移數據"""
        if not self.legacy_store:
            return

        logger.info("開始從舊的文件存儲遷移向量數據...")

        try:
            # 加載舊數據
            self.legacy_store.load()
            
            migrated_count = 0
            for doc_id in self.legacy_store.vectors.keys():
                # 檢查是否已經存在於 PostgreSQL 中
                existing = await self.pg_store.get_document(doc_id)
                if existing:
                    logger.debug(f"文檔 {doc_id} 已存在，跳過遷移")
                    continue

                # 遷移文檔
                embedding = self.legacy_store.vectors[doc_id]
                text = self.legacy_store.documents[doc_id]
                metadata = self.legacy_store.metadata[doc_id]

                await self.pg_store.add_document(
                    doc_id=doc_id,
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                )

                migrated_count += 1
                logger.debug(f"遷移文檔: {doc_id}")

            logger.info(f"✅ 成功遷移 {migrated_count} 個文檔到 PostgreSQL")

        except Exception as e:
            logger.error(f"遷移過程中發生錯誤: {str(e)}")
            raise

    def _get_embedding_model(self) -> Any:
        """獲取嵌入模型（延遲加載）"""
        if self._embedding_model is None:
            try:
                # 動態導入避免編譯時錯誤
                import importlib
                st_module = importlib.import_module("sentence_transformers")
                SentenceTransformer = st_module.SentenceTransformer

                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")

            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                # 使用簡單的嵌入作為後備
                self._embedding_model = self._simple_embedding

        return self._embedding_model

    def _simple_embedding(self, text: str) -> np.ndarray:
        """簡單的嵌入函數（後備方案）"""
        # 使用字符哈希生成固定維度向量
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dimension).astype(np.float32)

        # 歸一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        添加文檔到統一向量存儲
        
        兼容原有 VectorStore 接口
        """
        # 生成嵌入
        model = self._get_embedding_model()

        if callable(model):
            embedding = model(text)
        else:
            embedding = model.encode(text, convert_to_numpy=True)

        # 確保是 numpy 數組
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # 存儲到 PostgreSQL
        await self.pg_store.add_document(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )

        logger.debug(f"Added document {doc_id} to unified vector store")

    async def add_batch(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        批量添加文檔
        
        兼容原有 VectorStore 接口
        """
        if metadatas is None:
            metadatas = [{}] * len(doc_ids)

        for doc_id, text, metadata in zip(doc_ids, texts, metadatas, strict=False):
            await self.add_document(doc_id, text, metadata)

        logger.info(f"Added {len(doc_ids)} documents to unified vector store")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        搜索相似文檔
        
        兼容原有 VectorStore 接口
        """
        # 生成查詢嵌入
        model = self._get_embedding_model()

        if callable(model):
            query_embedding = model(query)
        else:
            query_embedding = model.encode(query, convert_to_numpy=True)

        # 確保是 numpy 數組
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # 使用 PostgreSQL 後端搜索
        results = await self.pg_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # 格式化結果以兼容原有接口
        formatted_results = []
        for result in results:
            formatted_results.append({
                "doc_id": result["doc_id"],
                "text": result["text"],
                "metadata": result["metadata"],
                "score": float(result["similarity_score"]),
            })

        return formatted_results

    async def delete_document(self, doc_id: str) -> bool:
        """
        刪除文檔
        
        兼容原有 VectorStore 接口
        """
        return await self.pg_store.delete_document(doc_id)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """
        獲取文檔
        
        兼容原有 VectorStore 接口
        """
        result = await self.pg_store.get_document(doc_id)
        
        if result:
            return {
                "doc_id": result["doc_id"],
                "text": result["document_text"],
                "metadata": result["metadata"],
                "embedding": np.array(result["embedding"]),
            }

        return None

    async def get_statistics(self) -> dict[str, Any]:
        """
        獲取統計信息
        
        兼容原有 VectorStore 接口
        """
        pg_stats = await self.pg_store.get_statistics()
        
        return {
            "total_documents": pg_stats["total_documents"],
            "backend": "postgresql+pgvector",
            "embedding_model": self.embedding_model_name,
            "database_url": self.database_url,
            "table_name": self.table_name,
            "embedding_dimension": self.embedding_dimension,
            **pg_stats,
        }

    async def close(self) -> None:
        """關閉連接"""
        await self.pg_store.close()


# 工廠函數，方便創建統一向量存儲
async def create_unified_vector_store(
    database_url: str | None = None,
    table_name: str = "knowledge_vectors",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    auto_migrate_from: Path | None = None,
) -> UnifiedVectorStore:
    """
    創建並初始化統一向量存儲
    
    Args:
        database_url: PostgreSQL 連接字符串
        table_name: 向量表名稱
        embedding_model: 嵌入模型名稱
        auto_migrate_from: 自動從該目錄遷移舊數據
    
    Returns:
        初始化完成的統一向量存儲實例
    """
    store = UnifiedVectorStore(
        database_url=database_url,
        table_name=table_name,
        embedding_model=embedding_model,
        legacy_persist_directory=auto_migrate_from,
    )
    
    await store.initialize()
    return store