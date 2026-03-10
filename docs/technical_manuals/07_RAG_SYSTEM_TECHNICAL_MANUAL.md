# AIVA RAG 系統技術手冊

**版本**: v2.1（de-semanticized protocol）
**狀態**: Production Ready
**路徑**: `services/core/aiva_core/cognitive_core/rag/`

---

## 1. 模組概述

RAG（Retrieval-Augmented Generation）系統是 AIVA 的長期記憶層。透過向量相似度搜尋，在每次 AI 決策前快速提取相關歷史經驗與知識，使 AI 決策更精準、更有據可循。

---

## 2. 系統架構

```
使用者查詢 / AI 決策請求
      │
      ▼
rag_engine.py（RAG 核心引擎）
      │
      ├── knowledge_base.py（知識管理）
      │         └── 5 種知識類型
      │
      └── vector_store 層（選擇對應 backend）
               │
               ├── vector_store.py        (記憶體/ChromaDB/FAISS)
               ├── unified_vector_store.py (統一介面)
               └── postgresql_vector_store.py (PostgreSQL + pgvector)
```

---

## 3. 核心檔案

| 檔案 | 功能 |
|---|---|
| `rag_engine.py` | RAG 核心引擎，上下文增強 |
| `knowledge_base.py` | 高層知識抽象（v2.1 協定擴展） |
| `vector_store.py` | 512 維向量 DB（記憶體/ChromaDB/FAISS） |
| `unified_vector_store.py` | 統一介面，自動選擇 Backend |
| `postgresql_vector_store.py` | PostgreSQL + pgvector Backend |
| `sync_experiences.py` | 非同步經驗同步（⭐ 核心流程） |
| `experience_sync.py` | 經驗學習整合 |

---

## 4. 知識類型（5 種）

```python
class KnowledgeType(Enum):
    VULNERABILITY     = "vulnerability"     # 漏洞模式
    ATTACK_TECHNIQUE  = "attack_technique"  # 攻擊技術
    BEST_PRACTICE     = "best_practice"     # 最佳實踐
    EXPERIENCE        = "experience"        # 歷史執行經驗
    MITIGATION        = "mitigation"        # 緩解措施
```

---

## 5. 向量儲存技術

### 5.1 向量規格

- **維度**：512 維（與 5M 決策引擎一致）
- **編碼方式**：Feature Hashing（去語意化，確定性）
- **Embedding 模型**：sentence-transformers（lazy loading，可選）

### 5.2 Backend 選擇

| Backend | 場景 | 特性 |
|---|---|---|
| 記憶體（In-memory） | 測試 / 小規模 | 快速，無持久化 |
| ChromaDB | 中等規模 | 本地持久化 |
| FAISS | 大規模搜尋 | Facebook AI 高效能 |
| PostgreSQL + pgvector | 生產環境 | 完整持久化，SQL 查詢 |

### 5.3 PostgreSQL 索引

```sql
-- IVFFlat 索引（近似最近鄰搜尋）
CREATE INDEX ON embeddings USING ivfflat (vector vector_cosine_ops);

-- GIN 索引（metadata 查詢）
CREATE INDEX ON embeddings USING gin (metadata);
```

---

## 6. 知識操作

### 6.1 新增知識

```python
from aiva_common.cognitive_core.rag import rag_engine

await rag_engine.add_knowledge(
    content="SQL Injection in login endpoint via 'id' parameter",
    knowledge_type=KnowledgeType.VULNERABILITY,
    metadata={
        "severity": "HIGH",
        "cve": "CVE-2024-XXXX",
        "target": "example.com"
    }
)
```

### 6.2 查詢相關知識

```python
# 同步查詢
results = rag_engine.search(
    query="SQL injection bypass techniques",
    knowledge_types=[KnowledgeType.ATTACK_TECHNIQUE],
    top_k=5
)

# 非同步查詢
results = await rag_engine.async_search(
    query="WAF bypass for Cloudflare",
    top_k=3
)
```

### 6.3 上下文增強

```python
# RAG 核心功能：在 AI 決策前注入相關知識
enhanced_context = await rag_engine.enhance_context(
    base_prompt="Analyze SQL injection vulnerability at /api/login",
    max_context_length=2048
)
# enhanced_context 包含：base_prompt + 相關歷史經驗 + 攻擊技術知識
```

---

## 7. 經驗同步流程

`sync_experiences.py` 負責將 Integration 模組的執行結果異步寫入 RAG：

```
攻擊執行完成
  │
  ▼
Integration/AI Operation Recorder
  │ 觸發
  ▼
sync_experiences.py
  │
  ├── 提取有效執行片段
  ├── 計算向量嵌入
  └── 寫入 vector_store
            │
            ▼
        下次 AI 決策可使用此經驗
```

---

## 8. v2.1 去語意化協定

**問題**：傳統 RAG 依賴語言模型生成嵌入，導致：
- 相同輸入可能產生不同向量（不確定性）
- 依賴外部 NLU 服務

**v2.1 解決方案**：Feature Hashing 確定性編碼

```python
# 去語意化流程
raw_feature = extract_features(content)  # 提取結構化特徵
vector = feature_hash(raw_feature, dims=512)  # 確定性哈希到 512 維
# 相同 content → 永遠相同 vector
```

**驗證**：12/12 測試全部通過

---

## 9. 與其他模組的整合

| 模組 | 角色 | 整合點 |
|---|---|---|
| `cognitive_core/` | 主要使用方 | `rag_engine.search()`, `enhance_context()` |
| `learning_system/` | 知識提供方 | `sync_experiences.py` |
| `integration/` | 觸發同步 | 執行完成後呼叫 sync |
| `aiva_common/schemas/` | 資料格式 | 知識物件 schema |

---

## 10. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第4-1冊_RAG_P1驗證指南.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第4冊_功能模組操作.md`
- **技術手冊**：`docs/technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md`
