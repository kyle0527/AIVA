# RAG 檢索增強生成模組

> **路徑**: `cognitive_core/rag/`  
> **狀態**: ✅ 正常 | **Python 文件數**: 7 | **最後更新**: 2026-04-05  
> **父模組**: [Cognitive Core](../README.md)

## 概述

負責向量數據庫管理、知識檢索和增強 AI 決策。將向量檢索與 AI 決策結合，提供上下文增強的攻擊計畫生成。

## 📄 檔案詳細資訊 (Files Details)

### `knowledge_base.py`
**說明**: Knowledge Base - 知識庫類別

**類別 (Classes)**:
- `VectorStoreProtocol` - 向量存儲協議接口 - 遵循 aiva_common 標準
- `KnowledgeBase` - 知識庫

### `postgresql_vector_store.py`
**說明**: PostgreSQL + pgvector 向量存儲實現

**類別 (Classes)**:
- `PostgreSQLVectorStore` - PostgreSQL + pgvector 向量存儲

### `rag_engine.py`
**說明**: RAG Engine - 檢索增強生成引擎

**類別 (Classes)**:
- `KnowledgeType` - 知識類型
- `QueryCache` - 查詢緩存
- `RAGEngine` - RAG 引擎

### `sync_experiences.py`
**說明**: RAG 經驗同步模組


### `unified_vector_store.py`
**說明**: 統一向量存儲管理器

**類別 (Classes)**:
- `UnifiedVectorStore` - 統一向量存儲管理器

### `vector_store.py`
**說明**: Vector Store - 向量數據庫

**類別 (Classes)**:
- `VectorStore` - 向量數據庫

