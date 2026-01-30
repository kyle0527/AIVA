# RAG 檢索增強生成模組

> **路徑**: `cognitive_core/rag/`  
> **狀態**: ✅ 正常 | **文件數**: 8 | **最後更新**: 2026-01-21  
> **父模組**: [Cognitive Core](../README.md)

## 概述

負責向量數據庫管理、知識檢索和增強 AI 決策。將向量檢索與 AI 決策結合，提供上下文增強的攻擊計畫生成。

## 核心組件

### sync_experiences.py ⭐ 新增
- `sync_experiences_to_vector_store()` - 同步執行經驗到向量存儲
- 自動化經驗學習和知識累積
- 與 RAG 系統整合

### knowledge_base.py

- `VectorStoreProtocol` - 向量存儲協議接口（Protocol 類）
- `KnowledgeBase` - 知識庫高級抽象
  - 基於向量存儲的知識管理
  - 支援同步和異步搜索接口
  - 整合 RAGQueryRequest/RAGQueryResult

### rag_engine.py

- `KnowledgeType` - 知識類型枚舉 (VULNERABILITY, ATTACK_TECHNIQUE, BEST_PRACTICE, EXPERIENCE, MITIGATION)
- `RAGEngine` - RAG 引擎
  - 增強攻擊計畫生成
  - 建議下一步驟
  - 結合向量檢索和 AI 生成

### vector_store.py

- `VectorStore` - 向量數據庫
  - 支援後端：memory, ChromaDB, FAISS
  - 延遲載入嵌入模型 (sentence-transformers)
  - 512 維向量（匹配 5M Decision Engine）

### unified_vector_store.py

- `UnifiedVectorStore` - 統一向量存儲管理器
  - 整合 VectorStore 和 PostgreSQL + pgvector
  - 支援從舊文件式存儲遷移數據
  - 遵循 aiva_common 標準配置

### postgresql_vector_store.py

- `PostgreSQLVectorStore` - PostgreSQL + pgvector 向量存儲
  - 解決併發瓶頸
  - 統一存儲向量、文檔、戰果
  - 支援 IVFFlat 索引和 GIN 元數據索引

### __init__.py

- 導出：`KnowledgeBase`, `RAGEngine`, `VectorStore`, `UnifiedVectorStore`

## 依賴關係

- 內部依賴：
  - `aiva_common.schemas.dual_loop`
  - `aiva_common.error_handling`
- 外部依賴：
  - `sentence-transformers`
  - `chromadb` (可選)
  - `faiss-cpu` (可選)
  - `asyncpg` (PostgreSQL)
  - `numpy`

## 使用範例

```python
from cognitive_core.rag import KnowledgeBase, RAGEngine, VectorStore

# 初始化向量存儲
vector_store = VectorStore(backend="chroma", persist_directory="./data/vectors")

# 創建知識庫
kb = KnowledgeBase(vector_store)

# 初始化 RAG 引擎
rag = RAGEngine(knowledge_base=kb)

# 增強攻擊計畫
context = await rag.enhance_attack_plan(
    target=attack_target,
    objective="測試 SQL 注入漏洞"
)

# 搜索相關知識
results = await kb.search("SQL injection techniques", top_k=5)
```
