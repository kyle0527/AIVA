# 🔍 RAG - 檢索增強生成系統

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心功能](#核心功能)
- [📂 檔案列表](#-檔案列表)
- [🔧 核心組件](#-核心組件)
  - [1. `rag_engine.py` - RAG 核心引擎](#1-rag_enginepy---rag-核心引擎)
  - [2. `knowledge_base.py` - 知識庫管理](#2-knowledge_basepy---知識庫管理)
  - [3. `unified_vector_store.py` - 統一向量存儲](#3-unified_vector_storepy---統一向量存儲)
  - [4. `postgresql_vector_store.py` - PostgreSQL 後端](#4-postgresql_vector_storepy---postgresql-後端)
- [🚀 完整使用流程](#-完整使用流程)
  - [初始化 RAG 系統](#初始化-rag-系統)
  - [與 Neural 整合](#與-neural-整合)
- [📊 性能指標](#-性能指標)

---

**導航**: [← 返回 Cognitive Core](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: v2.1.2  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2025-12-20  
> **角色**: 知識檢索和上下文增強

---

## 🎯 模組概述

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [使用範例](#使用範例)

---

## 🎯 模組概述

RAG (Retrieval-Augmented Generation) 子模組實現了 AIVA 的知識檢索和上下文增強能力，支援多種向量存儲後端，整合對內探索和對外學習的知識源。

### 核心功能
- **知識檢索** - 高效的向量相似度搜索
- **上下文增強** - 將檢索結果融合到生成過程
- **多源整合** - 內部探索 + 外部學習知識
- **向量存儲** - 支援內存和 PostgreSQL 後端

---

## 📂 檔案列表

| 檔案 | 行數 | 功能 | 狀態 |
|------|------|------|------|
| `rag_engine.py` | ~800 | RAG 核心引擎 | ✅ |
| `knowledge_base.py` | ~600 | 統一知識庫管理 | ✅ |
| `unified_vector_store.py` | ~500 | 統一向量存儲抽象層 | ✅ |
| `vector_store.py` | ~300 | 向量存儲接口定義 | ✅ |
| `postgresql_vector_store.py` | ~400 | PostgreSQL 向量存儲 | ✅ |
| `demo_rag_integration.py` | ~200 | RAG 整合示範 | 🔧 |
| `__init__.py` | ~50 | 模組入口 | ✅ |

**總計**: 6 個 Python 檔案，約 2850+ 行代碼

---

## 🔧 核心組件

### 1. `rag_engine.py` - RAG 核心引擎

**功能**: 檢索增強生成的核心實現

**主要流程**:
```python
query → 向量化 → 相似度檢索 → 重排序 → 上下文融合 → 增強輸出
```

**使用範例**:
```python
from aiva_core.cognitive_core.rag import RAGEngine

engine = RAGEngine(vector_store=vector_store)

# 檢索增強
result = await engine.retrieve_and_enhance(
    query="如何執行SQL注入測試",
    context={"target": "https://example.com"},
    top_k=5
)

print(result.enhanced_context)  # 增強後的上下文
print(result.sources)            # 知識來源
print(result.relevance_scores)  # 相關性分數
```

**關鍵方法**:
- `retrieve_and_enhance()` - 端到端 RAG
- `retrieve()` - 純檢索
- `rerank()` - 結果重排序
- `merge_contexts()` - 上下文融合

---

### 2. `knowledge_base.py` - 知識庫管理

**功能**: 統一管理對內和對外知識源

**知識來源**:
```
KnowledgeBase
├── Internal Knowledge (對內探索)
│   ├── 系統能力知識
│   ├── 模組結構知識
│   └── AST 分析知識
│
└── External Knowledge (對外學習)
    ├── 執行經驗
    ├── 測試案例
    └── 漏洞模式
```

**使用範例**:
```python
from aiva_core.cognitive_core.rag import KnowledgeBase

kb = KnowledgeBase()

# 添加知識
await kb.add_knowledge(
    content="SQL注入測試方法...",
    source="external_learning",
    metadata={"type": "test_case", "severity": "high"}
)

# 查詢知識
results = await kb.query(
    query="SQL注入",
    filters={"source": "external_learning"},
    top_k=10
)

# 更新知識
await kb.update_knowledge(
    knowledge_id="kb_001",
    content="更新的內容..."
)
```

---

### 3. `unified_vector_store.py` - 統一向量存儲

**功能**: 提供統一的向量存儲抽象層

**支援後端**:
- In-Memory (內存，用於開發)
- PostgreSQL + pgvector (生產環境)
- 可擴展其他後端

**使用範例**:
```python
from aiva_core.cognitive_core.rag import UnifiedVectorStore

# 自動選擇後端
store = UnifiedVectorStore.create(
    backend="postgresql",
    config={"host": "localhost", "database": "aiva"}
)

# 添加向量
await store.add(
    vectors=embeddings,
    metadata=[{"id": "doc_1", "content": "..."}]
)

# 相似度搜索
results = await store.search(
    query_vector=query_embedding,
    top_k=5,
    filters={"type": "capability"}
)
```

---

### 4. `postgresql_vector_store.py` - PostgreSQL 後端

**功能**: 基於 PostgreSQL + pgvector 的持久化向量存儲

**特性**:
- ✅ 持久化存儲
- ✅ HNSW 索引加速
- ✅ 事務支援
- ✅ 並發控制

**使用範例**:
```python
from aiva_core.cognitive_core.rag import PostgreSQLVectorStore

store = PostgreSQLVectorStore(
    connection_string="postgresql://user:pass@localhost/aiva"
)

# 創建索引
await store.create_index(
    index_type="hnsw",
    m=16,
    ef_construction=200
)

# 批次插入
await store.batch_insert(
    vectors=batch_embeddings,
    metadata=batch_metadata
)
```

---

## 🚀 完整使用流程

### 初始化 RAG 系統
```python
from aiva_core.cognitive_core.rag import (
    RAGEngine,
    KnowledgeBase,
    UnifiedVectorStore
)

# 1. 初始化向量存儲
vector_store = UnifiedVectorStore.create(
    backend="postgresql",
    config={
        "host": "localhost",
        "database": "aiva_knowledge"
    }
)

# 2. 初始化知識庫
knowledge_base = KnowledgeBase(vector_store=vector_store)

# 3. 添加知識
await knowledge_base.add_knowledge(
    content="SQL注入是一種常見的Web攻擊...",
    source="external_learning",
    metadata={"category": "vulnerability", "severity": "high"}
)

# 4. 初始化 RAG 引擎
rag_engine = RAGEngine(
    vector_store=vector_store,
    knowledge_base=knowledge_base
)

# 5. 執行 RAG
result = await rag_engine.retrieve_and_enhance(
    query="如何測試SQL注入漏洞",
    context={"target": "https://example.com"},
    top_k=5
)

print(f"找到 {len(result.sources)} 個相關知識")
print(f"增強後的上下文: {result.enhanced_context}")
```

### 與 Neural 整合
```python
from aiva_core.cognitive_core.neural import BioNeuronMaster
from aiva_core.cognitive_core.rag import RAGEngine

# RAG 增強的 AI 決策
master = BioNeuronMaster(
    mode="ai",
    rag_engine=rag_engine
)

# 使用 RAG 知識增強推理
result = await master.process_request_with_rag({
    "task": "執行SQL注入測試",
    "target": "https://example.com"
})
```

---

## 📊 性能指標

| 指標 | 數值 | 備註 |
|------|------|------|
| 檢索速度 | < 100ms | top_k=10 |
| 向量維度 | 768/1536 | 依模型而定 |
| 索引類型 | HNSW | 高效近似搜索 |
| 並發查詢 | 100+ QPS | PostgreSQL 後端 |
| 準確率 | 90%+ | 相關性評估 |

---

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team
