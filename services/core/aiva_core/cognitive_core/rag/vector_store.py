"""Vector Store - 向量數據庫

負責向量化和存儲攻擊模式、經驗樣本、知識文檔
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """向量數據庫

    使用嵌入向量存儲和檢索知識

    支持的後端：
    - ChromaDB (推薦)
    - FAISS
    - 內存存儲（開發用）
    """

    def __init__(
        self,
        backend: str = "memory",
        persist_directory: Path | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "aiva_capabilities",
    ) -> None:
        """初始化向量存儲

        Args:
            backend: 後端類型 ("memory", "chroma", "faiss")
            persist_directory: 持久化目錄
            embedding_model: 嵌入模型名稱
            collection_name: ChromaDB collection 名稱
        """
        self.backend = backend
        self.persist_directory = persist_directory or Path("./data/vectors")
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        self.vectors: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, str] = {}

        # 嵌入模型（延遲加載）
        self._embedding_model: Any | None = None
        
        # ChromaDB collection（延遲加載）
        self.collection: Any | None = None

        self._initialize_backend()

        logger.info(
            f"VectorStore initialized with {backend} backend, "
            f"model={embedding_model}"
        )

    def _initialize_backend(self) -> None:
        """初始化後端存儲"""
        if self.backend == "memory":
            logger.info("Using in-memory vector store")

        elif self.backend == "chroma":
            try:
                import chromadb

                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
                # 獲取或創建 collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "AIVA capabilities knowledge base"}
                )
                logger.info(
                    f"ChromaDB initialized at {self.persist_directory}, "
                    f"collection '{self.collection_name}' ready"
                )

            except ImportError:
                logger.warning(
                    "ChromaDB not installed, falling back to memory store. "
                    "Install with: pip install chromadb"
                )
                self.backend = "memory"

        elif self.backend == "faiss":
            try:
                import faiss  # noqa: F401

                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self.index = None  # 延遲初始化
                logger.info("FAISS backend initialized")

            except ImportError:
                logger.warning(
                    "FAISS not installed, falling back to memory store. "
                    "Install with: pip install faiss-cpu"
                )
                self.backend = "memory"

    def _get_embedding_model(self) -> Any:
        """獲取嵌入模型（延遲加載）"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

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
        """簡單的嵌入函數（後備方案）

        Args:
            text: 輸入文本

        Returns:
            嵌入向量
        """
        # 使用字符哈希生成固定維度向量
        hash_val = hash(text)
        dim = 384  # 與 all-MiniLM-L6-v2 維度一致

        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(dim).astype(np.float32)

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
        """添加文檔到向量存儲

        Args:
            doc_id: 文檔 ID
            text: 文檔文本
            metadata: 元數據
        """
        if self.backend == "chroma" and self.collection is not None:
            # 使用 ChromaDB API
            # ChromaDB 自動處理嵌入，不需要手動生成
            await asyncio.to_thread(
                self.collection.add,
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata or {}]
            )
            logger.debug(f"Added document {doc_id} to ChromaDB collection")
        else:
            # Memory backend: 手動生成嵌入
            model = self._get_embedding_model()

            # 檢查是否為 SentenceTransformer 模型
            if hasattr(model, 'encode'):
                # 使用 encode 方法（SentenceTransformer）
                embedding = await asyncio.to_thread(
                    model.encode, text, convert_to_numpy=True
                )
            elif callable(model):
                # 使用簡單嵌入函數
                embedding = await asyncio.to_thread(model, text)
            else:
                raise ValueError(f"Unknown embedding model type: {type(model)}")

            # 存儲到內存
            self.vectors[doc_id] = embedding
            self.documents[doc_id] = text
            self.metadata[doc_id] = metadata or {}
            self.metadata[doc_id] = metadata or {}

            logger.debug(f"Added document {doc_id} to memory vector store")

    def add_batch(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """批量添加文檔

        Args:
            doc_ids: 文檔 ID 列表
            texts: 文檔文本列表
            metadatas: 元數據列表
        """
        if metadatas is None:
            metadatas = [{}] * len(doc_ids)

        if self.backend == "chroma" and self.collection is not None:
            # 使用 ChromaDB 批量 API（更高效）
            self.collection.add(
                ids=doc_ids,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Added {len(doc_ids)} documents to ChromaDB collection")
        else:
            # Memory backend: 逐個添加
            for doc_id, text, metadata in zip(doc_ids, texts, metadatas, strict=False):
                self.add_document(doc_id, text, metadata)

            logger.info(f"Added {len(doc_ids)} documents to memory vector store")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """搜索相似文檔

        Args:
            query: 查詢文本
            top_k: 返回前 k 個結果
            filter_metadata: 元數據過濾條件

        Returns:
            搜索結果列表，每個結果包含 doc_id, text, metadata, score
        """
        if self.backend == "chroma" and self.collection is not None:
            # 使用 ChromaDB API 查詢
            where_filter = filter_metadata if filter_metadata else None
            
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=top_k,
                where=where_filter  # ChromaDB 元數據過濾
            )
            
            # 轉換為統一格式
            formatted_results = []
            if results and results.get('ids') and len(results['ids']) > 0:
                ids = results['ids'][0]
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, doc_id in enumerate(ids):
                    # ChromaDB 返回距離（越小越相似），轉換為相似度分數
                    distance = distances[i] if i < len(distances) else 0.0
                    score = 1.0 / (1.0 + distance)  # 轉換為 0-1 之間的分數
                    
                    formatted_results.append({
                        "doc_id": doc_id,
                        "text": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "score": float(score),
                    })
            
            logger.debug(f"ChromaDB query returned {len(formatted_results)} results")
            return formatted_results
        
        else:
            # Memory backend: 手動計算相似度
            model = self._get_embedding_model()

            # 檢查是否為 SentenceTransformer 模型
            if hasattr(model, 'encode'):
                query_embedding = model.encode(query, convert_to_numpy=True)
            elif callable(model):
                query_embedding = model(query)
            else:
                raise ValueError(f"Unknown embedding model type: {type(model)}")

            # 計算相似度
            similarities = []

            for doc_id, doc_embedding in self.vectors.items():
                # 應用過濾器
                if filter_metadata:
                    doc_meta = self.metadata.get(doc_id, {})
                    if not all(doc_meta.get(k) == v for k, v in filter_metadata.items()):
                        continue

                # 計算餘弦相似度
                similarity = np.dot(query_embedding, doc_embedding)

                similarities.append(
                    {
                        "doc_id": doc_id,
                        "text": self.documents[doc_id],
                        "metadata": self.metadata[doc_id],
                        "score": float(similarity),
                    }
                )

            # 排序並返回 top_k
            similarities.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)  # type: ignore[arg-type]

            logger.debug(f"Memory search returned {len(similarities[:top_k])} results")
            return similarities[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """刪除文檔

        Args:
            doc_id: 文檔 ID

        Returns:
            是否成功刪除
        """
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.documents[doc_id]
            del self.metadata[doc_id]
            logger.debug(f"Deleted document {doc_id}")
            return True

        return False

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """獲取文檔

        Args:
            doc_id: 文檔 ID

        Returns:
            文檔數據，不存在則返回 None
        """
        if doc_id in self.vectors:
            return {
                "doc_id": doc_id,
                "text": self.documents[doc_id],
                "metadata": self.metadata[doc_id],
                "embedding": self.vectors[doc_id],
            }

        return None

    def save(self, path: Path | None = None) -> None:
        """保存向量存儲到磁盤

        Args:
            path: 保存路徑，默認使用 persist_directory
        """
        save_path = path or self.persist_directory
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存向量
        vectors_file = save_path / "vectors.npy"
        np.save(vectors_file, self.vectors)

        # 保存文檔和元數據
        data_file = save_path / "data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )

        logger.info(f"Vector store saved to {save_path}")

    def load(self, path: Path | None = None) -> None:
        """從磁盤加載向量存儲

        Args:
            path: 加載路徑，默認使用 persist_directory
        """
        load_path = path or self.persist_directory

        # 加載向量
        vectors_file = load_path / "vectors.npy"
        if vectors_file.exists():
            self.vectors = np.load(vectors_file, allow_pickle=True).item()

        # 加載文檔和元數據
        data_file = load_path / "data.json"
        if data_file.exists():
            with open(data_file, encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", {})
                self.metadata = data.get("metadata", {})

        logger.info(f"Loaded {len(self.vectors)} documents from {load_path}")

    def count(self) -> int:
        """獲取文檔數量

        Returns:
            文檔總數
        """
        if self.backend == "chroma" and self.collection is not None:
            return self.collection.count()
        else:
            return len(self.vectors)

    def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息

        Returns:
            統計信息字典
        """
        return {
            "total_documents": self.count(),
            "backend": self.backend,
            "embedding_model": self.embedding_model_name,
            "persist_directory": str(self.persist_directory),
            "collection_name": getattr(self, 'collection_name', 'N/A'),
        }
