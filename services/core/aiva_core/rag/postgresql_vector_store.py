"""PostgreSQL + pgvector 向量存儲實現
解決未命名.txt 中提到的資料孤島問題
"""

import asyncio
import json
import logging
from typing import Any

import asyncpg
import numpy as np

from services.aiva_common.error_handling import (
    AIVAError,
    ErrorSeverity,
    ErrorType,
    create_error_context,
)

logger = logging.getLogger(__name__)
MODULE_NAME = "aiva_core.rag.postgresql_vector_store"


class PostgreSQLVectorStore:
    """PostgreSQL + pgvector 向量存儲

    優勢：
    1. 解決併發瓶頸：支援高併發讀寫
    2. 統一存儲：向量、文檔、戰果在同一資料庫
    3. 可擴展：支援水平擴展
    4. AI 增強：可進行複雜的關聯查詢
    """

    def __init__(
        self,
        database_url: str,
        table_name: str = "vectors",
        embedding_dimension: int = 384,
    ):
        self.database_url = database_url
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """初始化資料庫連接和表結構"""
        self.pool = await asyncpg.create_pool(self.database_url)

        if self.pool is None:
            raise AIVAError(
                "Failed to create database connection pool",
                error_type=ErrorType.DATABASE,
                severity=ErrorSeverity.CRITICAL,
                context=create_error_context(
                    module=MODULE_NAME,
                    function="initialize"
                )
            )

        async with self.pool.acquire() as conn:
            # 啟用 pgvector 擴展
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # 創建向量表
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    doc_id TEXT PRIMARY KEY,
                    embedding vector({self.embedding_dimension}),
                    document_text TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """
            )

            # 創建索引以提升查詢性能
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """
            )

            # 創建 GIN 索引用於元數據查詢
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
                ON {self.table_name} USING GIN (metadata)
            """
            )

    async def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加文檔和向量"""
        if self.pool is None:
            await self.initialize()

        # 確保向量維度正確
        if len(embedding) != self.embedding_dimension:
            raise AIVAError(
                f"Embedding dimension {len(embedding)} doesn't match "
                f"expected {self.embedding_dimension}",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=create_error_context(
                    module=MODULE_NAME,
                    function="add_document",
                    embedding_dim=len(embedding),
                    expected_dim=self.embedding_dimension
                )
            )

        if self.pool is None:
            await self.initialize()

        assert self.pool is not None  # 類型檢查
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} 
                (doc_id, embedding, document_text, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (doc_id) DO UPDATE SET
                    embedding = $2,
                    document_text = $3,
                    metadata = $4,
                    updated_at = NOW()
                """,
                doc_id,
                embedding.tolist(),  # pgvector 接受 list
                text,
                json.dumps(metadata or {}),
            )

        logger.debug(f"Added document {doc_id} to PostgreSQL vector store")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """向量相似性搜索"""
        if self.pool is None:
            await self.initialize()

        # 構建查詢
        base_query = f"""
            SELECT 
                doc_id,
                document_text,
                metadata,
                1 - (embedding <=> $1) AS similarity_score
            FROM {self.table_name}
        """

        conditions = ["1 - (embedding <=> $1) >= $2"]
        params = [query_embedding.tolist(), similarity_threshold]
        param_counter = 3

        # 添加元數據過濾
        if filter_metadata:
            for key, value in filter_metadata.items():
                conditions.append(f"metadata->>${param_counter} = ${param_counter + 1}")
                params.extend([key, json.dumps(value)])
                param_counter += 2

        where_clause = " AND ".join(conditions)
        full_query = f"""
            {base_query}
            WHERE {where_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        params.insert(1, top_k)  # top_k 作為第二個參數

        assert self.pool is not None  # 類型檢查
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(full_query, *params)

        results = []
        for row in rows:
            results.append(
                {
                    "doc_id": row["doc_id"],
                    "text": row["document_text"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "similarity_score": float(row["similarity_score"]),
                }
            )

        return results

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """獲取特定文檔"""
        if self.pool is None:
            await self.initialize()

        assert self.pool is not None  # 類型檢查
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT doc_id, document_text, metadata, created_at, updated_at
                FROM {self.table_name}
                WHERE doc_id = $1
                """,
                doc_id,
            )

        if row:
            return {
                "doc_id": row["doc_id"],
                "text": row["document_text"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        return None

    async def delete_document(self, doc_id: str) -> bool:
        """刪除文檔"""
        if self.pool is None:
            await self.initialize()

        assert self.pool is not None  # 類型檢查
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE doc_id = $1", doc_id
            )

        # asyncpg 返回 "DELETE n" 格式
        deleted_count = int(result.split()[-1])
        return deleted_count > 0

    async def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息"""
        if self.pool is None:
            await self.initialize()

        assert self.pool is not None  # 類型檢查
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                f"""
                SELECT 
                    COUNT(*) as total_documents,
                    AVG(length(document_text)) as avg_text_length,
                    MIN(created_at) as oldest_document,
                    MAX(created_at) as newest_document
                FROM {self.table_name}
            """
            )

        return {
            "total_documents": stats["total_documents"],
            "avg_text_length": float(stats["avg_text_length"] or 0),
            "oldest_document": stats["oldest_document"],
            "newest_document": stats["newest_document"],
            "backend": "postgresql+pgvector",
            "embedding_dimension": self.embedding_dimension,
        }

    async def execute_unified_query(self, vulnerability_id: str) -> dict[str, Any]:
        """統一查詢：結合戰果資料和向量知識

        這是新架構的核心優勢！
        可以在同一個查詢中獲取：
        1. 漏洞詳情（從 findings 表）
        2. 相關攻擊技術（從 vectors 表）
        3. 歷史經驗（從 experience_records 表）
        """
        if self.pool is None:
            await self.initialize()

        assert self.pool is not None  # 類型檢查
        async with self.pool.acquire() as conn:
            # 複雜的關聯查詢，這在舊架構中是不可能的
            result = await conn.fetchrow(
                """
                WITH vulnerability_info AS (
                    SELECT 
                        finding_id,
                        vulnerability_name,
                        severity,
                        target_url,
                        raw_data
                    FROM findings 
                    WHERE finding_id = $1
                ),
                related_techniques AS (
                    SELECT 
                        doc_id,
                        document_text,
                        metadata
                    FROM vectors 
                    WHERE metadata->>'type' = 'attack_technique'
                    AND metadata->>'vulnerability_type' = (
                        SELECT vulnerability_name FROM vulnerability_info
                    )
                    LIMIT 5
                ),
                historical_experience AS (
                    SELECT 
                        COUNT(*) as attempt_count,
                        AVG(overall_score) as avg_success_rate
                    FROM experience_records 
                    WHERE attack_type = (
                        SELECT vulnerability_name FROM vulnerability_info
                    )
                )
                SELECT 
                    vi.*,
                    he.attempt_count,
                    he.avg_success_rate,
                    array_agg(rt.document_text) as related_techniques
                FROM vulnerability_info vi
                CROSS JOIN historical_experience he
                LEFT JOIN related_techniques rt ON true
                GROUP BY vi.finding_id, vi.vulnerability_name, vi.severity, 
                         vi.target_url, vi.raw_data, he.attempt_count, he.avg_success_rate
            """,
                vulnerability_id,
            )

        if result:
            return {
                "vulnerability": {
                    "id": result["finding_id"],
                    "name": result["vulnerability_name"],
                    "severity": result["severity"],
                    "target": result["target_url"],
                },
                "ai_insights": {
                    "historical_attempts": result["attempt_count"],
                    "average_success_rate": float(result["avg_success_rate"] or 0),
                    "related_techniques": result["related_techniques"] or [],
                },
            }
        return {}

    async def close(self) -> None:
        """關閉連接池"""
        if self.pool:
            await self.pool.close()


# 使用示例
async def demo_postgresql_vector_store():
    """演示新架構的強大功能"""
    # 初始化
    store = PostgreSQLVectorStore(
        database_url="postgresql://postgres:aiva123@localhost:5432/aiva_db"
    )

    try:
        await store.initialize()

        # 添加攻擊技術知識
        embedding = np.random.randn(384).astype(np.float32)
        await store.add_document(
            doc_id="sqli_union_technique",
            text="UNION based SQL injection allows...",
            embedding=embedding,
            metadata={
                "type": "attack_technique",
                "vulnerability_type": "SQL Injection",
                "success_rate": 0.85,
            },
        )

        # 搜索相關技術
        query_embedding = np.random.randn(384).astype(np.float32)
        results = await store.search(
            query_embedding=query_embedding,
            top_k=3,
            filter_metadata={"type": "attack_technique"},
        )

        print(f"找到 {len(results)} 個相關技術")

        # 統一查詢（新架構的核心優勢）
        unified_result = await store.execute_unified_query("vuln_123")
        print("統一查詢結果:", unified_result)

    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(demo_postgresql_vector_store())
