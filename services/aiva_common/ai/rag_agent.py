"""
AIVA RAG Agent - 檢索增強生成代理

基於 LangChain 架構實現的 RAG (Retrieval-Augmented Generation) 系統，
提供知識檢索和智能問答能力。參考業界最佳實踐設計。

核心功能:
- 文檔檢索和向量搜索
- 上下文增強的問答生成
- 多源知識整合
- 語義相似度搜索
- 混合檢索策略

架構設計:
- Abstract Factory Pattern: 可插拔的檢索器和生成器
- Strategy Pattern: 多種檢索策略
- Chain of Responsibility: 查詢處理鏈
- Observer Pattern: 檢索結果監控

技術棧:
- LangChain: RAG 框架核心
- Pydantic v2: 數據模型驗證
- FAISS/Chroma: 向量數據庫
- OpenAI/Hugging Face: 嵌入模型
- Asyncio: 異步處理

符合標準:
- AIVA Common 設計規範
- OpenAPI 3.1.1 兼容
- 現代化 Python 最佳實踐
- 可插拔架構設計
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS, Chroma

    _has_langchain = True
except ImportError:
    _has_langchain = False
    Document = object
    logging.warning("LangChain not available, using fallback implementation")


from ..schemas import RAGQueryPayload, RAGResponsePayload
from .interfaces import IRAGAgent

# ============================================================================
# Configuration and Enums (配置和枚舉)
# ============================================================================


class RetrievalStrategy(str, Enum):
    """檢索策略枚舉"""

    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    SEMANTIC_RERANK = "semantic_rerank"


class EmbeddingProvider(str, Enum):
    """嵌入模型提供商"""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class RAGConfig:
    """RAG 系統配置"""

    # 檢索配置
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    similarity_threshold: float = 0.7

    # 嵌入配置
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # 文檔處理配置
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_doc_length: int = 10000

    # 生成配置
    max_context_length: int = 4000
    response_max_tokens: int = 1000
    temperature: float = 0.7

    # 存儲配置
    vector_store_path: str | None = None
    persist_store: bool = True

    # API 配置
    openai_api_key: str | None = None
    huggingface_api_key: str | None = None


# ============================================================================
# Core RAG Components (核心 RAG 組件)
# ============================================================================


class DocumentProcessor:
    """文檔處理器 - 負責文檔分割和預處理"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_documents(self, documents: list[str]) -> list[Document]:
        """處理文檔列表，返回分割後的文檔塊"""
        processed_docs = []

        for i, doc_content in enumerate(documents):
            # 長度限制
            if len(doc_content) > self.config.max_doc_length:
                doc_content = doc_content[: self.config.max_doc_length]

            # 分割文檔
            chunks = self.text_splitter.split_text(doc_content)

            # 創建 Document 對象
            for j, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": f"doc_{i}",
                        "chunk_id": j,
                        "total_chunks": len(chunks),
                    },
                )
                processed_docs.append(doc)

        return processed_docs


class VectorStoreManager:
    """向量存儲管理器 - 抽象工廠模式"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = self._create_embeddings()
        self.vector_store = None

    def _create_embeddings(self):
        """創建嵌入模型 - 工廠方法"""
        if not _has_langchain:
            return None

        if self.config.embedding_provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                openai_api_key=self.config.openai_api_key,
                model=self.config.embedding_model,
            )
        elif self.config.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        else:
            # 本地嵌入模型實現
            return self._create_local_embeddings()

    def _create_local_embeddings(self):
        """創建本地嵌入模型 - 降級實現"""

        class LocalEmbeddings:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # 簡單的詞向量實現 (生產環境需要更強的模型)
                import hashlib

                embeddings = []
                for text in texts:
                    # 使用哈希生成偽向量 (僅示例)
                    hash_obj = hashlib.md5(text.encode())
                    hash_hex = hash_obj.hexdigest()
                    vector = [
                        float(int(hash_hex[i : i + 2], 16)) / 255.0
                        for i in range(0, min(32, len(hash_hex)), 2)
                    ]
                    # 填充到指定維度
                    while len(vector) < self.config.embedding_dim:
                        vector.extend(vector[: self.config.embedding_dim - len(vector)])
                    embeddings.append(vector[: self.config.embedding_dim])
                return embeddings

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

        return LocalEmbeddings()

    def create_vector_store(self, documents: list[Document]) -> Any:
        """創建向量存儲"""
        if not _has_langchain or not documents:
            return None

        try:
            # 優先使用 FAISS
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

            # 持久化存儲
            if self.config.persist_store and self.config.vector_store_path:
                Path(self.config.vector_store_path).parent.mkdir(
                    parents=True, exist_ok=True
                )
                self.vector_store.save_local(self.config.vector_store_path)

        except Exception as e:
            logging.warning(
                f"FAISS creation failed: {e}, falling back to in-memory store"
            )
            # 降級到內存存儲
            self.vector_store = self._create_memory_store(documents)

        return self.vector_store

    def _create_memory_store(self, documents: list[Document]):
        """創建內存向量存儲 - 降級實現"""

        class MemoryVectorStore:
            def __init__(self, docs, embeddings):
                self.docs = docs
                self.embeddings = embeddings
                self.doc_embeddings = embeddings.embed_documents(
                    [doc.page_content for doc in docs]
                )

            def similarity_search(self, query: str, k: int = 4) -> list[Document]:
                query_embedding = self.embeddings.embed_query(query)

                # 計算餘弦相似度
                similarities = []
                for i, doc_embedding in enumerate(self.doc_embeddings):
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    similarities.append((similarity, i))

                # 排序並返回前 k 個
                similarities.sort(reverse=True)
                return [self.docs[i] for _, i in similarities[:k]]

            def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
                import math

                dot_product = sum(x * y for x, y in zip(a, b, strict=False))
                magnitude_a = math.sqrt(sum(x * x for x in a))
                magnitude_b = math.sqrt(sum(x * x for x in b))
                if magnitude_a == 0 or magnitude_b == 0:
                    return 0
                return dot_product / (magnitude_a * magnitude_b)

        return MemoryVectorStore(documents, self.embeddings)


class HybridRetriever:
    """混合檢索器 - 結合向量檢索和 BM25"""

    def __init__(self, vector_store: Any, documents: list[Document], config: RAGConfig):
        self.vector_store = vector_store
        self.config = config
        self.bm25_retriever = None

        if _has_langchain and documents:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = config.top_k
            except Exception as e:
                logging.warning(f"BM25 retriever creation failed: {e}")

    def retrieve(self, query: str) -> list[Document]:
        """執行混合檢索"""
        if self.config.retrieval_strategy == RetrievalStrategy.VECTOR_ONLY:
            return self._vector_retrieve(query)
        elif self.config.retrieval_strategy == RetrievalStrategy.BM25_ONLY:
            return self._bm25_retrieve(query)
        else:
            return self._hybrid_retrieve(query)

    def _vector_retrieve(self, query: str) -> list[Document]:
        """向量檢索"""
        if not self.vector_store:
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=self.config.top_k)
            return docs
        except Exception as e:
            logging.error(f"Vector retrieval failed: {e}")
            return []

    def _bm25_retrieve(self, query: str) -> list[Document]:
        """BM25 檢索"""
        if not self.bm25_retriever:
            return []

        try:
            docs = self.bm25_retriever.get_relevant_documents(query)
            return docs[: self.config.top_k]
        except Exception as e:
            logging.error(f"BM25 retrieval failed: {e}")
            return []

    def _hybrid_retrieve(self, query: str) -> list[Document]:
        """混合檢索"""
        vector_docs = self._vector_retrieve(query)
        bm25_docs = self._bm25_retrieve(query)

        # 合併並去重
        seen_content = set()
        combined_docs = []

        # 先添加向量檢索結果
        for doc in vector_docs[: self.config.top_k // 2]:
            if doc.page_content not in seen_content:
                combined_docs.append(doc)
                seen_content.add(doc.page_content)

        # 再添加 BM25 結果
        for doc in bm25_docs[: self.config.top_k // 2]:
            if (
                doc.page_content not in seen_content
                and len(combined_docs) < self.config.top_k
            ):
                combined_docs.append(doc)
                seen_content.add(doc.page_content)

        return combined_docs


# ============================================================================
# Main RAG Agent Implementation (主要 RAG 代理實現)
# ============================================================================


class BioNeuronRAGAgent(IRAGAgent):
    """
    AIVA BioNeuron RAG Agent - 檢索增強生成代理

    基於 LangChain 和現代 RAG 架構實現，提供智能文檔檢索和問答能力。
    採用 Abstract Factory 模式支持可插拔組件。
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)
        self.retriever = None
        self.knowledge_base = []
        self.is_initialized = False

        # 日誌設置
        self.logger = logging.getLogger(__name__)

    async def initialize(self, documents: list[str] | None = None) -> bool:
        """初始化 RAG 系統"""
        try:
            if documents:
                await self.add_documents(documents)

            self.is_initialized = True
            self.logger.info("RAG Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"RAG Agent initialization failed: {e}")
            return False

    async def add_documents(self, documents: list[str]) -> bool:
        """添加文檔到知識庫"""
        try:
            # 處理文檔
            processed_docs = self.document_processor.process_documents(documents)
            self.knowledge_base.extend(processed_docs)

            # 創建向量存儲
            vector_store = self.vector_store_manager.create_vector_store(processed_docs)

            # 創建檢索器
            self.retriever = HybridRetriever(vector_store, processed_docs, self.config)

            self.logger.info(f"Added {len(documents)} documents to knowledge base")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False

    async def invoke(self, query: RAGQueryPayload) -> RAGResponsePayload:
        """執行 RAG 查詢 (實現 IRAGAgent 介面)"""
        return await self.query(query)

    async def query(self, payload: RAGQueryPayload) -> RAGResponsePayload:
        """執行 RAG 查詢"""
        try:
            if not self.is_initialized or not self.retriever:
                return RAGResponsePayload(
                    query_id=payload.query_id,
                    results=[],
                    total_results=0,
                    enhanced_context="RAG Agent not initialized",
                    metadata={"error": "not_initialized"},
                )

            # 檢索相關文檔
            relevant_docs = self.retriever.retrieve(payload.query_text)

            if not relevant_docs:
                return RAGResponsePayload(
                    query_id=payload.query_id,
                    results=[],
                    total_results=0,
                    enhanced_context="No relevant documents found",
                    metadata={"status": "no_documents_found"},
                )

            # 生成回應
            response = await self._generate_response(
                payload.query_text, relevant_docs, payload.metadata
            )

            # 提取結果信息
            results = []
            similarity_scores = []

            for i, doc in enumerate(relevant_docs):
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    metadata = getattr(doc, "metadata", {})
                else:
                    content = str(doc)
                    metadata = {}

                similarity_score = 0.8 - (i * 0.1)  # 簡化實現
                similarity_scores.append(similarity_score)

                result = {
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "rank": i + 1,
                }
                results.append(result)

            avg_similarity = (
                sum(similarity_scores) / len(similarity_scores)
                if similarity_scores
                else 0.0
            )

            return RAGResponsePayload(
                query_id=payload.query_id,
                results=results,
                total_results=len(relevant_docs),
                avg_similarity=avg_similarity,
                enhanced_context=response,
                metadata={
                    "retrieval_strategy": self.config.retrieval_strategy.value,
                    "documents_retrieved": len(relevant_docs),
                    "knowledge_base_size": len(self.knowledge_base),
                    "top_k": payload.top_k,
                    "min_similarity": payload.min_similarity,
                },
            )

        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return RAGResponsePayload(
                query_id=payload.query_id,
                results=[],
                total_results=0,
                enhanced_context=f"Query processing failed: {str(e)}",
                metadata={"error": str(e)},
            )

    async def _generate_response(
        self,
        query: str,
        documents: list[Document],
        context: dict[str, Any] | None = None,
    ) -> str:
        """生成基於檢索文檔的回應"""
        try:
            # 構建上下文
            context_text = "\n\n".join([doc.page_content for doc in documents])

            # 限制上下文長度
            if len(context_text) > self.config.max_context_length:
                context_text = context_text[: self.config.max_context_length]

            # 簡化的生成邏輯 (生產環境需要使用 LLM)
            prompt = f"""
            基於以下知識內容回答問題：

            知識內容：
            {context_text}

            問題：{query}

            請基於提供的知識內容給出準確、有用的回答。如果知識內容中沒有相關信息，請明確說明。
            """

            # 這裡應該調用 LLM (如 OpenAI, Hugging Face 等)
            # 為了示例，我們使用簡化的邏輯
            if _has_langchain and self.config.openai_api_key:
                try:
                    llm = OpenAI(
                        openai_api_key=self.config.openai_api_key,
                        temperature=self.config.temperature,
                        max_tokens=self.config.response_max_tokens,
                    )
                    response = llm(prompt)
                    return response
                except Exception as e:
                    self.logger.warning(f"LLM generation failed: {e}, using fallback")

            # 降級實現 - 基於關鍵詞匹配的簡單回應
            return self._generate_fallback_response(query, documents)

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"無法生成回應: {str(e)}"

    def _generate_fallback_response(self, query: str, documents: list[Document]) -> str:
        """降級回應生成 - 基於簡單邏輯"""
        if not documents:
            return "沒有找到相關的知識內容來回答您的問題。"

        # 簡單的關鍵詞匹配和摘要
        query_lower = query.lower()
        relevant_sentences = []

        for doc in documents:
            sentences = doc.page_content.split("。")
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_lower.split()):
                    relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            # 返回最相關的句子組合
            return "根據知識庫，" + "。".join(relevant_sentences[:3]) + "。"
        else:
            # 返回第一個文檔的摘要
            return f"根據相關文檔：{documents[0].page_content[:300]}..."

    async def get_knowledge_base_info(self) -> dict[str, Any]:
        """獲取知識庫信息"""
        return {
            "total_documents": len(self.knowledge_base),
            "configuration": {
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "top_k": self.config.top_k,
                "embedding_provider": self.config.embedding_provider.value,
                "chunk_size": self.config.chunk_size,
            },
            "is_initialized": self.is_initialized,
            "has_langchain": _has_langchain,
        }

    async def clear_knowledge_base(self) -> bool:
        """清空知識庫"""
        try:
            self.knowledge_base = []
            self.retriever = None
            self.vector_store_manager.vector_store = None
            self.logger.info("Knowledge base cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear knowledge base: {e}")
            return False

    # ============================================================================
    # IRAGAgent Interface Implementation (介面方法實現)
    # ============================================================================

    async def update_knowledge_base(
        self,
        knowledge_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        更新知識庫 (實現 IRAGAgent 介面)

        Args:
            knowledge_type: 知識類型
            content: 知識內容
            metadata: 元數據

        Returns:
            是否更新成功
        """
        try:
            # 為內容添加類型標記
            tagged_content = f"[{knowledge_type}] {content}"

            # 添加到知識庫
            success = await self.add_documents([tagged_content])

            if success and metadata:
                # 將元數據存儲在實例變量中
                if not hasattr(self, "knowledge_metadata"):
                    self.knowledge_metadata = {}
                self.knowledge_metadata[knowledge_type] = metadata

            self.logger.info(
                f"Updated knowledge base with {knowledge_type}: {len(content)} chars"
            )
            return success

        except Exception as e:
            self.logger.error(f"Failed to update knowledge base: {e}")
            return False

    async def search_knowledge(
        self, query_text: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        搜索知識 (實現 IRAGAgent 介面)

        Args:
            query_text: 查詢文字
            top_k: 返回數量
            filters: 過濾條件

        Returns:
            匹配的知識項目列表
        """
        try:
            if not self.is_initialized or not self.retriever:
                return []

            # 執行檢索
            relevant_docs = self.retriever.retrieve(query_text)

            # 限制數量
            relevant_docs = relevant_docs[:top_k]

            # 轉換為知識項目格式
            knowledge_items = []
            for i, doc in enumerate(relevant_docs):
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    doc_metadata = getattr(doc, "metadata", {})
                else:
                    content = str(doc)
                    doc_metadata = {}

                # 應用過濾條件
                if filters:
                    should_include = True
                    for filter_key, filter_value in filters.items():
                        if filter_key in doc_metadata:
                            if doc_metadata[filter_key] != filter_value:
                                should_include = False
                                break
                    if not should_include:
                        continue

                # 提取知識類型
                knowledge_type = "general"
                if content.startswith("[") and "]" in content:
                    end_bracket = content.find("]")
                    knowledge_type = content[1:end_bracket]
                    content = content[end_bracket + 1 :].strip()

                item = {
                    "knowledge_id": f"item_{i}_{doc_metadata.get('chunk_id', 0)}",
                    "knowledge_type": knowledge_type,
                    "content": content,
                    "similarity_score": 0.8 - (i * 0.1),  # 簡化實現
                    "metadata": doc_metadata,
                    "rank": i + 1,
                }
                knowledge_items.append(item)

            return knowledge_items

        except Exception as e:
            self.logger.error(f"Knowledge search failed: {e}")
            return []


# ============================================================================
# Factory Functions (工廠函數)
# ============================================================================


def create_rag_agent(
    config: RAGConfig | None = None,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE,
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
) -> BioNeuronRAGAgent:
    """創建 RAG 代理實例 - 工廠函數"""
    if config is None:
        config = RAGConfig(
            embedding_provider=embedding_provider, retrieval_strategy=retrieval_strategy
        )

    return BioNeuronRAGAgent(config)


def create_rag_config(**kwargs) -> RAGConfig:
    """創建 RAG 配置 - 工廠函數"""
    return RAGConfig(**kwargs)


# ============================================================================
# Utilities and Helpers (工具和輔助函數)
# ============================================================================


async def load_documents_from_files(file_paths: list[str]) -> list[str]:
    """從文件加載文檔內容"""
    documents = []

    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                content = path.read_text(encoding="utf-8")
                documents.append(content)
            else:
                logging.warning(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to load file {file_path}: {e}")

    return documents


# ============================================================================
# Module Exports (模組導出)
# ============================================================================

__all__ = [
    "BioNeuronRAGAgent",
    "RAGConfig",
    "RetrievalStrategy",
    "EmbeddingProvider",
    "create_rag_agent",
    "create_rag_config",
    "load_documents_from_files",
]


# ============================================================================
# Usage Example (使用示例)
# ============================================================================

if __name__ == "__main__":

    async def main():
        """RAG Agent 使用示例"""
        # 創建配置
        config = RAGConfig(
            retrieval_strategy=RetrievalStrategy.HYBRID,
            embedding_provider=EmbeddingProvider.HUGGINGFACE,
            top_k=3,
        )

        # 創建 RAG 代理
        rag_agent = create_rag_agent(config)

        # 準備測試文檔
        test_documents = [
            "人工智慧 (AI) 是指機器模擬人類智慧的能力，包括學習、推理和自我修正。",
            "機器學習是 AI 的一個分支，讓電腦能夠從數據中學習而無需明確編程。",
            "深度學習使用神經網絡來模擬人腦處理信息的方式。",
        ]

        # 初始化系統
        await rag_agent.initialize(test_documents)

        # 創建查詢
        from ..schemas import RAGQueryPayload

        query = RAGQueryPayload(
            query_id="test_001",
            query_text="什麼是人工智慧？",
            metadata={"user_id": "test_user"},
        )

        # 執行查詢
        response = await rag_agent.query(query)

        print(f"Query: {query.query_text}")
        print(f"Response: {response.enhanced_context}")
        print(f"Total Results: {response.total_results}")
        print(f"Results: {len(response.results)}")

        # 獲取知識庫信息
        info = await rag_agent.get_knowledge_base_info()
        print(f"Knowledge Base Info: {info}")

    # 運行示例 (如果直接執行此文件)
    asyncio.run(main())
