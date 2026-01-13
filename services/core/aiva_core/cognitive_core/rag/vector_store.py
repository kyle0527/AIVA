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
                import faiss  # noqa: F401  # type: ignore[import-not-found]

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
        
        v2.1 (2026-01-08): 去語意化反射引擎整合
        
        現在支援三種模式：
        1. 去語意化特徵編碼 - 如果 text 包含 rag_trigger 權重表
        2. 結構化能力編碼 - 如果 text 是 JSON 格式的能力記錄
        3. 哈希編碼 - 純文本的後備方案

        Args:
            text: 輸入文本或 JSON 能力記錄

        Returns:
            嵌入向量 (512 維，與 5M 模型匹配)
        """
        dim = 512  # 更新為 512 維（匹配 5M Decision Engine）
        
        # 嘗試解析為 JSON（能力記錄）
        try:
            import json
            flow_data = json.loads(text)
            
            # 模式 1: 去語意化特徵編碼（優先級最高）
            if isinstance(flow_data, dict) and 'rag_trigger' in flow_data:
                logger.debug("Using de-semanticized feature encoding")
                return self._encode_rag_trigger(flow_data.get('rag_trigger', {}), dim)
            
            # 模式 2: 結構化能力編碼
            if isinstance(flow_data, dict) and ('primary_module' in flow_data or 'structured_tags' in flow_data):
                # 使用結構化能力編碼器
                try:
                    from ..capability_encoder import CapabilityEncoder
                    encoder = CapabilityEncoder()
                    return encoder.encode(flow_data)
                except ImportError:
                    logger.warning("CapabilityEncoder not available, falling back to hash embedding")
        except (json.JSONDecodeError, TypeError):
            pass  # 不是 JSON，使用後備方案
        
        # 模式 3: 後備方案 - 基於哈希的確定性嵌入
        # 使用更好的哈希方法以提高區分度
        import hashlib
        
        # 多層哈希以增加向量的區分度
        hash_md5 = hashlib.md5(text.encode()).hexdigest()
        hash_sha1 = hashlib.sha1(text.encode()).hexdigest()
        combined_hash = hash_md5 + hash_sha1
        
        # 將哈希轉為確定性的偽隨機數
        seed = int(combined_hash[:8], 16) % (2**31)
        np.random.seed(seed)
        
        embedding = np.random.randn(dim).astype(np.float32)

        # 歸一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding
    
    def _encode_rag_trigger(self, rag_trigger: dict[str, float], target_dim: int = 512) -> np.ndarray:
        """將 RAG 觸發權重表編碼為固定維度向量
        
        去語意化反射引擎核心：
        - 使用確定性哈希映射特徵到固定槽位
        - 權重值直接映射，不經過語義轉換
        - 毫秒級編碼速度
        
        Args:
            rag_trigger: 特徵名稱 → 權重的字典
            target_dim: 目標維度（預設 512）
            
        Returns:
            固定維度的特徵向量
        """
        import hashlib
        
        # 初始化零向量
        feature_vector = np.zeros(target_dim, dtype=np.float32)
        
        # 將每個特徵映射到固定槽位
        for feature_name, weight in rag_trigger.items():
            # 使用 MD5 哈希確定槽位
            hash_digest = hashlib.md5(feature_name.encode('utf-8')).digest()
            slot_index = int.from_bytes(hash_digest[:4], 'little') % target_dim
            
            # 累加權重（支援多個特徵映射到同一槽位）
            feature_vector[slot_index] += float(weight)
        
        # L2 歸一化（保持向量長度一致）
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
        
        return feature_vector
    
    def add_capability_from_registry(
        self,
        capability: dict[str, Any],
        capability_id: str | None = None,
    ) -> None:
        """從 integration/capability 註冊表添加能力
        
        v2.1 (2026-01-08): 新增方法，支援 CapabilityRecord 格式
        
        Args:
            capability: CapabilityRecord 格式的能力記錄
            capability_id: 能力 ID（如未提供則使用 capability['id']）
        """
        cap_id = capability_id or capability.get('id', 'unknown')
        
        # 優先使用 rag_trigger 進行編碼
        if 'rag_trigger' in capability and capability['rag_trigger']:
            embedding = self._encode_rag_trigger(capability['rag_trigger'], 512)
        else:
            # 後備：使用完整記錄進行編碼
            import json
            embedding = self._simple_embedding(json.dumps(capability, ensure_ascii=False))
        
        # 存儲
        self.vectors[cap_id] = embedding
        self.documents[cap_id] = capability.get('name', cap_id)
        self.metadata[cap_id] = {
            'capability_id': cap_id,
            'module': capability.get('module', 'unknown'),
            'capability_type': capability.get('capability_type', 'unknown'),
            'feature_signature': capability.get('feature_signature', []),
            'tags': capability.get('tags', []),
        }
        
        logger.debug(f"Added capability {cap_id} from registry to vector store")
    
    def search_by_environment(
        self,
        environment_features: dict[str, float],
        top_k: int = 5,
        filter_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """根據環境特徵搜索最匹配的能力（去語意化檢索）
        
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
        # 將環境特徵編碼為查詢向量
        query_embedding = self._encode_rag_trigger(environment_features, 512)
        
        # 計算與所有能力的相似度
        results = []
        for cap_id, cap_embedding in self.vectors.items():
            # 應用類型過濾
            if filter_type:
                cap_type = self.metadata.get(cap_id, {}).get('capability_type', '')
                if cap_type != filter_type:
                    continue
            
            # 計算餘弦相似度
            similarity = float(np.dot(query_embedding, cap_embedding))
            
            results.append({
                'capability_id': cap_id,
                'match_score': similarity,
                'metadata': self.metadata.get(cap_id, {}),
                'document': self.documents.get(cap_id, ''),
            })
        
        # 排序並返回 top-k
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        logger.info(f"Environment search found {len(results[:top_k])} matching capabilities")
        return results[:top_k]
    
    def add_capability(
        self,
        capability_id: str,
        capability: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加能力記錄到向量存儲（5M AI 專用）
        
        v2.0 (2026-01-04): 新增方法
        使用 CapabilityEncoder 直接編碼，避免文本轉換損失
        
        Args:
            capability_id: 能力 ID（通常為 flow_id）
            capability: 能力記錄（來自 latest_classification.json）
            metadata: 額外元數據
        """
        try:
            from ..capability_encoder import CapabilityEncoder
            encoder = CapabilityEncoder()
            embedding = encoder.encode(capability)
        except ImportError:
            logger.warning("CapabilityEncoder not available, using fallback")
            import json
            embedding = self._simple_embedding(json.dumps(capability, ensure_ascii=False))
        
        # 合併元數據
        full_metadata = metadata or {}
        full_metadata.update({
            'capability_id': capability_id,
            'primary_module': capability.get('primary_module', 'unknown'),
            'primary_component_type': capability.get('primary_component_type', 'unknown'),
            'length': capability.get('length', 0),
        })
        
        # 存儲
        self.vectors[capability_id] = embedding
        self.documents[capability_id] = str(capability.get('id', capability_id))
        self.metadata[capability_id] = full_metadata
        
        logger.debug(f"Added capability {capability_id} to vector store")
    
    def add_capabilities_batch(
        self,
        capabilities: list[dict[str, Any]],
    ) -> int:
        """批量添加能力記錄
        
        v2.0 (2026-01-04): 新增方法
        
        Args:
            capabilities: 能力記錄列表
            
        Returns:
            成功添加的數量
        """
        try:
            from ..capability_encoder import CapabilityEncoder
            encoder = CapabilityEncoder()
            use_encoder = True
        except ImportError:
            logger.warning("CapabilityEncoder not available, using fallback")
            use_encoder = False
        
        count = 0
        for cap in capabilities:
            cap_id = f"flow_{cap.get('id', count)}"
            
            if use_encoder:
                embedding = encoder.encode(cap)
            else:
                import json
                embedding = self._simple_embedding(json.dumps(cap, ensure_ascii=False))
            
            self.vectors[cap_id] = embedding
            self.documents[cap_id] = str(cap.get('id', cap_id))
            self.metadata[cap_id] = {
                'capability_id': cap_id,
                'primary_module': cap.get('primary_module', 'unknown'),
                'primary_component_type': cap.get('primary_component_type', 'unknown'),
                'length': cap.get('length', 0),
            }
            count += 1
        
        logger.info(f"Added {count} capabilities to vector store")
        return count
    
    def search_capabilities(
        self,
        query_capability: dict[str, Any],
        top_k: int = 5,
        filter_module: str | None = None,
    ) -> list[dict[str, Any]]:
        """搜索相似能力（5M AI 專用）
        
        v2.0 (2026-01-04): 新增方法
        使用向量相似度而非字串匹配
        
        Args:
            query_capability: 查詢能力記錄
            top_k: 返回前 k 個結果
            filter_module: 模組過濾器
            
        Returns:
            相似能力列表
        """
        try:
            from ..capability_encoder import CapabilityEncoder
            encoder = CapabilityEncoder()
            query_embedding = encoder.encode(query_capability)
        except ImportError:
            import json
            query_embedding = self._simple_embedding(json.dumps(query_capability, ensure_ascii=False))
        
        # 計算相似度
        results = []
        
        for cap_id, embedding in self.vectors.items():
            # 應用模組過濾
            if filter_module:
                cap_module = self.metadata.get(cap_id, {}).get('primary_module', '')
                if cap_module != filter_module:
                    continue
            
            # 計算餘弦相似度
            similarity = float(np.dot(query_embedding, embedding))
            
            results.append({
                'capability_id': cap_id,
                'score': similarity,
                'metadata': self.metadata.get(cap_id, {}),
            })
        
        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]

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

            embedding: np.ndarray
            # 檢查是否為 SentenceTransformer 模型
            if hasattr(model, 'encode'):
                # 使用 encode 方法（SentenceTransformer）
                embedding = await asyncio.to_thread(
                    model.encode, text, convert_to_numpy=True
                )
            elif callable(model):
                # 使用簡單嵌入函數
                raw_embedding = await asyncio.to_thread(model, text)
                embedding = np.asarray(raw_embedding)  # 確保轉換為 numpy array
            else:
                raise ValueError(f"Unknown embedding model type: {type(model)}")

            # 存儲到內存
            self.vectors[doc_id] = np.asarray(embedding)  # 確保是 numpy array
            self.documents[doc_id] = text
            self.metadata[doc_id] = metadata or {}
            self.metadata[doc_id] = metadata or {}

            logger.debug(f"Added document {doc_id} to memory vector store")

    async def add_batch(
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
                await self.add_document(doc_id, text, metadata)  # add_document 是 async 方法

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

            query_embedding: np.ndarray
            # 檢查是否為 SentenceTransformer 模型
            if hasattr(model, 'encode'):
                query_embedding = np.asarray(model.encode(query, convert_to_numpy=True))
            elif callable(model):
                query_embedding = np.asarray(model(query))
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

        # 保存向量（使用 numpy savez 保存字典）
        vectors_file = save_path / "vectors.npz"
        vector_dict = {k: v for k, v in self.vectors.items()}  # 轉換為普通字典
        np.savez(str(vectors_file), **vector_dict)  # type: ignore[arg-type]  # 解包字典保存為 .npz 格式

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

        # 加載向量（從 .npz 格式）
        vectors_file = load_path / "vectors.npz"
        if vectors_file.exists():
            loaded = np.load(vectors_file)
            self.vectors = {key: loaded[key] for key in loaded.files}  # 重建字典

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
