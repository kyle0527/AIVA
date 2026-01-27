"""統一向量存儲管理器

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
    """統一向量存儲管理器

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
        """初始化統一向量存儲

        Args:
            database_url: PostgreSQL 數據庫連接字符串
            table_name: 向量表名稱
            embedding_dimension: 嵌入維度
            embedding_model: 嵌入模型名稱
            legacy_persist_directory: 舊的文件存儲目錄（用於遷移）
        """
        import os
        
        # 從環境變量獲取或使用無密碼本地開發配置
        self.database_url = (
            database_url or os.getenv(
                "AIVA_DATABASE_URL",
                "postgresql://postgres:change_me_in_production@localhost:5432/aiva_db"  # 開發環境密碼 - 生產環境需更改
            )
        )
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
        """獲取嵌入模型（延遲加載）- AIVA 自研版本"""
        if self._embedding_model is None:
            try:
                # 導入 AIVA 自研 Embedding 層
                import sys
                from pathlib import Path
                
                # 確保可以導入 aiva_embedding
                neural_path = Path(__file__).parent.parent / "neural"
                if str(neural_path) not in sys.path:
                    sys.path.insert(0, str(neural_path))
                
                from aiva_embedding import AIVAEmbedding

                self._embedding_model = AIVAEmbedding(
                    model_name_or_path=self.embedding_model_name
                )
                logger.info(f"✅ AIVA Embedding 已載入: {self.embedding_model_name} (AIVA 架構)")

            except ImportError as e:
                logger.warning(
                    f"Failed to load AIVA Embedding: {e}. "
                    "Using simple embedding fallback."
                )
                # 使用簡單的嵌入作為後備
                self._embedding_model = self._simple_embedding

        return self._embedding_model

    def _simple_embedding(self, text: str) -> np.ndarray:
        """簡單的嵌入函數（後備方案）"""
        # 使用字符哈希生成固定維度向量
        hash_val = hash(text)
        rng = np.random.default_rng(hash_val % (2**32))
        embedding = rng.standard_normal(self.embedding_dimension, dtype=np.float32)

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
        """添加文檔到統一向量存儲

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
        """批量添加文檔

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
        """搜索相似文檔

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
            formatted_results.append(
                {
                    "doc_id": result["doc_id"],
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": float(result["similarity_score"]),
                }
            )

        return formatted_results

    async def delete_document(self, doc_id: str) -> bool:
        """刪除文檔

        兼容原有 VectorStore 接口
        """
        return await self.pg_store.delete_document(doc_id)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """獲取文檔

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
        """獲取統計信息

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

    # ==================== v2.1 去語意化反射引擎方法 ====================

    def add_capability_from_registry(
        self,
        capability: dict[str, Any],
        capability_id: str | None = None,
    ) -> None:
        """從 integration/capability 註冊表添加能力（同步版本）
        
        v2.1 (2026-01-08): 新增方法，支援 CapabilityRecord 格式
        
        Args:
            capability: CapabilityRecord 格式的能力記錄
            capability_id: 能力 ID（如未提供則使用 capability['id']）
        """
        import asyncio
        import json
        
        cap_id = capability_id or capability.get('id', 'unknown')
        
        # 優先使用 rag_trigger 進行編碼
        if 'rag_trigger' in capability and capability['rag_trigger']:
            # 使用 VectorStore 的 _encode_rag_trigger
            from .vector_store import VectorStore
            temp_store = VectorStore(backend="memory")
            embedding = temp_store._encode_rag_trigger(capability['rag_trigger'], 512)
        else:
            # 後備：使用完整記錄進行編碼
            text = json.dumps(capability, ensure_ascii=False)
            model = self._get_embedding_model()
            if callable(model):
                embedding = model(text)
            else:
                embedding = model.encode(text, convert_to_numpy=True)
            
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
        
        # 構建文檔文本
        doc_text = capability.get('name', cap_id)
        
        # 構建元數據
        metadata = {
            'capability_id': cap_id,
            'module': capability.get('module', 'unknown'),
            'capability_type': capability.get('capability_type', 'unknown'),
            'feature_signature': capability.get('feature_signature', []),
            'tags': capability.get('tags', []),
        }
        
        # 異步添加到 PostgreSQL（在事件循環中執行）
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(
                self.pg_store.add_document(
                    doc_id=cap_id,
                    text=doc_text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
        except RuntimeError:
            # 沒有運行中的事件循環，使用同步方式
            asyncio.run(
                self.pg_store.add_document(
                    doc_id=cap_id,
                    text=doc_text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
        
        logger.debug(f"Added capability {cap_id} from registry to unified vector store")

    def search_by_environment(
        self,
        environment_features: dict[str, float],
        top_k: int = 5,
        filter_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """根據環境特徵搜索最匹配的能力（去語意化檢索，同步版本）
        
        v2.1 (2026-01-08): 新增方法
        
        核心特性：
        - 不使用語義理解，純向量相似度
        - 毫秒級檢索速度
        - 確定性結果（相同輸入 → 相同輸出）
        
        Args:
            environment_features: 環境特徵字典（如 {'http_403': 3, 'waf_cloudflare': 1}）
            top_k: 返回前 k 個結果
            filter_type: 能力類型過濾器
            
        Returns:
            匹配能力列表，按相似度排序
        """
        import asyncio
        
        # 將環境特徵編碼為查詢向量
        from .vector_store import VectorStore
        temp_store = VectorStore(backend="memory")
        query_embedding = temp_store._encode_rag_trigger(environment_features, 512)
        
        # 構建元數據過濾器
        filter_metadata = None
        if filter_type:
            filter_metadata = {'capability_type': filter_type}
        
        # 執行異步搜索
        try:
            loop = asyncio.get_running_loop()
            # 如果已在異步環境，創建任務
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.pg_store.search(
                        query_embedding=query_embedding,
                        top_k=top_k,
                        filter_metadata=filter_metadata,
                    )
                )
                results = future.result()
        except RuntimeError:
            # 沒有運行中的事件循環，直接運行
            results = asyncio.run(
                self.pg_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                )
            )
        
        # 格式化結果
        formatted_results = []
        for result in results:
            formatted_results.append({
                'capability_id': result['doc_id'],
                'match_score': float(result['similarity_score']),
                'metadata': result['metadata'],
                'document': result['text'],
            })
        
        logger.info(f"Environment search found {len(formatted_results)} matching capabilities")
        return formatted_results


# 工廠函數，方便創建統一向量存儲
async def create_unified_vector_store(
    database_url: str | None = None,
    table_name: str = "knowledge_vectors",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    auto_migrate_from: Path | None = None,
) -> UnifiedVectorStore:
    """創建並初始化統一向量存儲

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
