# AIVA RAG 系統技術手冊

**版本**: v2.1（de-semanticized protocol） | **狀態**: ✅ P0 Complete | **路徑**: `services/core/aiva_core/cognitive_core/rag/`

---

## 目錄

1. [模組概述](#1-模組概述)
2. [系統架構](#2-系統架構)
3. [核心檔案](#3-核心檔案)
4. [知識類型（5 種）](#4-知識類型5-種)
5. [向量儲存技術](#5-向量儲存技術)
   - 5.1 [向量規格](#51-向量規格)
   - 5.2 [Backend 選擇](#52-backend-選擇)
   - 5.3 [PostgreSQL 索引](#53-postgresql-索引)
6. [知識操作](#6-知識操作)
7. [經驗同步流程](#7-經驗同步流程)
8. [v2.1 去語意化協定](#8-v21-去語意化協定)
9. [CLIDecisionEngine — RAG 驅動的 CLI 決策](#9-clidecisionengine--rag-驅動的-cli-決策)
10. [完成狀態](#10-完成狀態)
    - 10.1 [已完成功能（P0 階段）](#101-已完成功能p0-階段-)
    - 10.2 [待完成 / 目標功能（P1-P3 階段）](#102-待完成--目標功能p1-p3-階段-)
11. [與其他模組的整合](#11-與其他模組的整合)
12. [搭配閱讀](#12-搭配閱讀)

---

## 1. 模組概述

RAG（Retrieval-Augmented Generation）系統是 AIVA 的長期記憶層。透過向量相似度搜尋，在每次 AI 決策前快速提取相關歷史經驗與知識，使 AI 決策更精準、更有據可循。

**P0 成果**：
- 286 個內部探索數據流整合完成
- 525 個攻擊 flows 載入，287 個可執行（54.7%）
- CLIDecisionEngine、FlowExecutorAdapter 完整實作
- 5/5 整合測試通過

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
               ├── vector_store.py        （記憶體/ChromaDB/FAISS）
               ├── unified_vector_store.py （統一介面）
               └── postgresql_vector_store.py（PostgreSQL + pgvector）
```

---

## 3. 核心檔案

| 檔案 | 功能 |
|---|---|
| `rag_engine.py` | RAG 核心引擎，上下文增強 |
| `knowledge_base.py` | 高層知識抽象（v2.1 協定擴展）|
| `vector_store.py` | 512 維向量 DB（記憶體/ChromaDB/FAISS）|
| `unified_vector_store.py` | 統一介面，自動選擇 Backend |
| `postgresql_vector_store.py` | PostgreSQL + pgvector Backend |
| `sync_experiences.py` | 非同步經驗同步（⭐ 核心流程）|
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
| 記憶體（In-memory）| 測試 / 小規模 | 快速，無持久化 |
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
# enhanced_context = base_prompt + 相關歷史經驗 + 攻擊技術知識
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
  ├── 計算向量嵌入（Feature Hashing）
  └── 寫入 vector_store
            │
            ▼
        下次 AI 決策可使用此經驗
```

---

## 8. v2.1 去語意化協定

**問題**：傳統 RAG 依賴語言模型生成嵌入，導致不確定性與外部依賴。

**v2.1 解決方案**：Feature Hashing 確定性編碼

```python
# 去語意化流程
raw_feature = extract_features(content)  # 提取結構化特徵
vector = feature_hash(raw_feature, dims=512)  # 確定性哈希
# 相同 content → 永遠相同 vector（可重現）
```

**驗證**：12/12 測試全部通過

---

## 9. CLIDecisionEngine — RAG 驅動的 CLI 決策

P0 新增功能，透過 RAG 知識庫決策最佳 CLI 攻擊流程：

```python
# CLIDecisionEngine 整合 525 個攻擊 flows
engine = CLIDecisionEngine()
# 287 個可執行 flows（54.7%）

# 按漏洞類型分布
XSS:  48/97 可執行
SQLi: 68/115 可執行
SSRF: 28/64 可執行

# 與 AttackCoordinator 整合完成
coordinator = AttackCoordinator()
await coordinator.execute_targeted_attack(target_url, vuln_type="sqli")
```

---

## 10. 完成狀態

### 10.1 已完成功能（P0 階段）✅

| 功能 | 說明 |
|---|---|
| RAG 架構設計 | JSONL 格式決策，完整架構 |
| CLIDecisionEngine | 525 flows 載入，287 可執行 |
| FlowExecutorAdapter | 參數轉換，流程執行 |
| AttackCoordinator 整合 | 完整整合 |
| 5/5 整合測試通過 | 基礎功能 + 多能力攻擊 + 完整整合流程 |
| 286 內部數據流整合 | 來自 internal exploration |
| PostgreSQL backend 支援 | pgvector 整合 |
| 去語意化 v2.1 協定 | 12/12 驗證通過 |

### 10.2 待完成 / 目標功能（P1-P3 階段）🎯

| 功能 | 優先級 | 說明 |
|---|---|---|
| **P1: 實際執行驗證** | P1 | 對 testphp.vulnweb.com 等目標實測 |
| **P1: 錯誤收集與優化** | P1 | 根據實測結果修正 flows |
| 未執行 flows 啟用 | P1 | 287→更多可執行 flows（目前 54.7%）|
| XXE 攻擊能力支援 | P2 | 新增 XXE flows 到 CLIDecisionEngine |
| File Upload 攻擊支援 | P2 | 惡意檔案上傳 flows |
| 攻擊鏈組合支援 | P2 | Phase1 → Phase2 → PostEx 自動串接 |
| 決策算法強化 | P2 | 基於目標指紋動態調整 flow 優先級 |
| 向量語意搜尋（選用）| P3 | 補充 Feature Hashing 的語意搜尋能力 |
| 知識庫自動更新 | P3 | 每次執行後自動擴充知識庫 |
| RAG 命中率監控 | P3 | 目標 >70% 命中率，建立 dashboard |

---

## 11. 與其他模組的整合

| 模組 | 角色 | 整合點 |
|---|---|---|
| `cognitive_core/` | 主要使用方 | `rag_engine.search()`, `enhance_context()` |
| `learning_system/` | 知識提供方 | `sync_experiences.py` |
| `integration/` | 觸發同步 | 執行完成後呼叫 sync |
| `aiva_common/schemas/` | 資料格式 | 知識物件 schema |

---

## 12. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第4-1冊_RAG_P1驗證指南.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第4冊_功能模組操作.md`
- **技術手冊**：`docs/technical_manuals/06_COGNITIVE_CORE_TECHNICAL_MANUAL.md`
- **技術手冊**：`docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md`
