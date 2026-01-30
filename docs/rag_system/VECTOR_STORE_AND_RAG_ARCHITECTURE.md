# 向量庫和 RAG 完整架構說明

## 一、目前向量庫現狀

### 1.1 現有數據

**向量庫位置**: `data/vector_db/chroma/`
**後端**: ChromaDB (persistent)
**Collection 名稱**: `aiva_capabilities`
**當前記錄數**: **782 條**

**現有數據類型**:
```json
{
  "type": "capability",
  "source": "internal_exploration",
  "namespace": "self_awareness",
  "module": "core/aiva_core",
  "language": "python",
  "capability_name": "comparator",
  "file_path": "...",
  "parameters_count": 1,
  "is_async": false,
  "sync_timestamp": "2025-11-28T04:07:20..."
}
```

**數據來源**:
1. **內部探測 (internal_exploration)**: 系統自我掃描的能力
2. **能力註冊中心 (capability registry)**: 註冊的模組能力
3. **知識庫文檔**: Markdown 分析報告（待添加）

### 1.2 向量庫結構

```
data/
└── vector_db/
    └── chroma/
        ├── chroma.sqlite3        # ChromaDB 元數據
        ├── data_level0.bin       # 向量數據
        ├── header.bin
        ├── length.bin
        └── link_lists.bin
```

## 二、預計要放入向量庫的數據

### 2.1 整合模組數據 (Integration Module)

**來源**: `services/integration/data/experiences/*.jsonl`

**數據類型**:
- ✅ **執行經驗記錄** (Experience Records)
  - 文件: `xss.jsonl`, `sqli.jsonl`, `ssrf.jsonl`, `phase0.jsonl`
  - 內容: 實際攻擊執行的請求、響應、結果
  - 用途: 學習系統訓練數據

**如何添加到向量庫**:
```python
# 讀取 JSONL 數據
from services.integration.simple_data_manager import get_data_manager

data_manager = get_data_manager()
records = data_manager.load_capability_data(capability="xss", limit=100)

# 添加到向量庫
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore

vector_store = VectorStore(backend="chroma")

for record in records:
    # 構建文本描述
    text = f"""
    Capability: {record['capability']}
    Target: {record['target']}
    Request: {record['request']}
    Response: {record['response']}
    Result: {record['result']}
    """
    
    # 添加到向量庫
    await vector_store.add_document(
        doc_id=f"exp_{record['task_id']}",
        text=text,
        metadata={
            "type": "experience",
            "capability": record['capability'],
            "timestamp": record['timestamp'],
            "success": record['result'].get('success', False),
        }
    )
```

**預計數據量**: 每個能力 1000+ 條記錄

### 2.2 知識庫分析報告 (Knowledge Base)

**來源**: `services/core/aiva_core/cognitive_core/learning_system/knowledge/`

**現有報告**:
- ✅ `XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md`
- ✅ `SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md`
- ✅ `SSRF_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md`

**如何添加**:
```python
from pathlib import Path

knowledge_dir = Path("services/core/aiva_core/cognitive_core/learning_system/knowledge")

for md_file in knowledge_dir.glob("*_ANALYSIS.md"):
    content = md_file.read_text(encoding="utf-8")
    
    await vector_store.add_document(
        doc_id=f"kb_{md_file.stem}",
        text=content,
        metadata={
            "type": "knowledge_base",
            "capability": md_file.stem.split('_')[0].lower(),
            "format": "markdown",
        }
    )
```

**預計數據量**: 20+ 份分析報告

### 2.3 外部模組能力 (External Modules)

**來源**: `外部模組/` 目錄下的工具

**數據類型**:
- ✅ 外部工具說明
- ✅ 使用範例
- ✅ 參數配置

**預計數據量**: 100+ 個外部模組

### 2.4 CVE 和漏洞數據 (未來擴展)

**來源**: 
- 外部 API 抓取的 CVE 數據
- 安全研究論文
- Exploit-DB 數據

**存儲方式**: 定期同步到向量庫

## 三、RAG 雙向搜索架構

### 3.1 對內搜索 (Internal Search)

**搜索範圍**: 本地向量庫
**數據源**:
1. **能力註冊表** (782 條能力記錄) ✅ 已有
2. **執行經驗記錄** (JSONL 數據) 🔄 需添加
3. **知識庫分析報告** (Markdown) 🔄 需添加
4. **外部模組說明** 🔄 需添加

**搜索方式**: 向量相似度搜索
```python
# 內部搜索
from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine

results = await rag_engine.search_internal(
    query="XSS bypass WAF",
    top_k=5
)
```

**返回內容**:
- 相關能力記錄
- 歷史執行經驗
- 知識庫建議

### 3.2 對外搜索 (External Search)

**搜索範圍**: 外部資源
**數據源**:
1. **CVE 數據庫** (NVD API)
2. **Exploit-DB** (Web Scraping)
3. **Google 搜索** (Custom Search API)
4. **GitHub Advisory** (GraphQL API)
5. **安全研究網站** (arXiv, 技術博客)

**搜索方式**: HTTP/HTTPS 請求
```python
# 外部搜索
results = await rag_trigger.search_external(
    query="XSS bypass WAF",
    sources=["cve", "exploit-db", "google", "github"]
)
```

**返回內容**:
- CVE 詳情和修復建議
- 公開的 Exploit 代碼
- 技術文章和討論
- 開源項目的安全公告

### 3.3 混合搜索策略 (Hybrid Search)

**流程**:
```
1. 先對內搜索 (快速)
   ├─ 向量庫查詢
   └─ 計算相似度

2. 判斷是否需要對外搜索
   ├─ 相似度 >= 0.6 → 使用內部結果 ✅
   └─ 相似度 < 0.6 → 觸發外部搜索 🔍

3. 對外搜索 (如果觸發)
   ├─ CVE 數據庫
   ├─ Exploit-DB
   ├─ Google 搜索
   └─ GitHub Advisory

4. 合併結果
   ├─ 內部結果 (高相關度)
   └─ 外部結果 (新知識)

5. 返回給學習系統
```

## 四、RAG 能否找到整合模組的數據？

### 4.1 當前狀況 ❌

**問題**: 整合模組的 JSONL 數據**目前不在向量庫中**

```
services/integration/data/experiences/
├── xss.jsonl          ❌ 不在向量庫
├── sqli.jsonl         ❌ 不在向量庫
├── ssrf.jsonl         ❌ 不在向量庫
└── phase0.jsonl       ❌ 不在向量庫

data/vector_db/chroma/
└── 只有能力註冊表數據 (782 條) ✅
```

### 4.2 解決方案 ✅

**方案 1: 定期同步到向量庫** (推薦)

創建同步腳本，定期將 JSONL 數據添加到向量庫：

```python
# sync_experiences_to_vector_store.py

import asyncio
from services.integration.simple_data_manager import get_data_manager
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore

async def sync_experiences():
    """同步執行經驗到向量庫"""
    data_manager = get_data_manager()
    vector_store = VectorStore(backend="chroma", persist_directory="data/vector_db/chroma")
    
    capabilities = ["xss", "sqli", "ssrf", "phase0", "phase1"]
    
    for capability in capabilities:
        print(f"同步 {capability} 經驗記錄...")
        
        # 讀取最近的記錄
        records = data_manager.load_capability_data(
            capability=capability,
            limit=1000  # 最近 1000 條
        )
        
        for record in records:
            # 構建文本
            text = f"""
            能力類型: {record['capability']}
            目標: {record.get('target', 'N/A')}
            請求方法: {record.get('request', {}).get('method', 'N/A')}
            響應狀態: {record.get('response', {}).get('status_code', 'N/A')}
            執行結果: {'成功' if record.get('result', {}).get('success') else '失敗'}
            錯誤訊息: {record.get('response', {}).get('error_message', 'N/A')}
            發現數量: {len(record.get('result', {}).get('findings', []))}
            """
            
            # 添加到向量庫
            await vector_store.add_document(
                doc_id=f"exp_{capability}_{record['timestamp']}",
                text=text,
                metadata={
                    "type": "experience",
                    "capability": capability,
                    "timestamp": record['timestamp'],
                    "success": record.get('result', {}).get('success', False),
                    "task_id": record.get('task_id', ''),
                }
            )
        
        print(f"✅ {capability}: 同步 {len(records)} 條記錄")

if __name__ == "__main__":
    asyncio.run(sync_experiences())
```

**執行**: 
```bash
python sync_experiences_to_vector_store.py
```

**效果**: 
- JSONL 數據 → 向量庫
- RAG 可以搜索到執行經驗 ✅
- 支持向量相似度搜索 ✅

**方案 2: 實時添加** (可選)

在 `app.py` 保存數據時，同時添加到向量庫：

```python
# app.py

# 保存到整合模組 (JSONL)
data_manager.save_task_data(...)

# 同時添加到向量庫
await vector_store.add_document(
    doc_id=f"exp_{scan_id}",
    text=...,
    metadata=...
)
```

### 4.3 添加後的效果 ✅

```
data/vector_db/chroma/
├── 能力註冊表 (782 條)          ✅ 已有
├── 執行經驗記錄 (5000+ 條)      ✅ 添加後
├── 知識庫分析報告 (20+ 份)      ✅ 添加後
└── 外部模組說明 (100+ 個)       ✅ 未來添加

總計: 6000+ 條向量記錄
```

**RAG 搜索能力**:
- ✅ 找到相似的執行經驗
- ✅ 找到知識庫建議
- ✅ 找到相關能力記錄
- ✅ 對外搜索新知識（如果內部找不到）

## 五、完整實現方案

### 5.1 更新 RAG 觸發器支持雙向搜索

文件: `services/core/aiva_core/cognitive_core/learning_system/rag_trigger.py`

```python
class RAGTrigger:
    """RAG 觸發器 - 支持對內和對外搜索"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        vector_store=None,  # 內部向量庫
        enable_external_search: bool = True,  # 是否啟用外部搜索
        notification_callback=None,
    ):
        self.similarity_threshold = similarity_threshold
        self.vector_store = vector_store  # 對內搜索
        self.enable_external_search = enable_external_search  # 對外搜索
        self.notification_callback = notification_callback
    
    async def search(
        self,
        query: str,
        current_data: dict,
        search_mode: str = "hybrid",  # "internal", "external", "hybrid"
    ) -> list[dict]:
        """執行搜索
        
        Args:
            query: 搜索查詢
            current_data: 當前數據
            search_mode: 搜索模式
                - "internal": 只搜索內部向量庫
                - "external": 只搜索外部資源
                - "hybrid": 先內部後外部（默認）
        """
        results = []
        
        # 1. 內部搜索（向量庫）
        if search_mode in ["internal", "hybrid"]:
            internal_results = await self._search_internal(query, current_data)
            results.extend(internal_results)
            
            # 計算最高相似度
            max_similarity = max(
                (r.get("relevance_score", 0.0) for r in internal_results),
                default=0.0
            )
            
            # 如果內部找到高相關結果，不需要外部搜索
            if search_mode == "hybrid" and max_similarity >= self.similarity_threshold:
                logger.info(f"✅ 內部搜索已找到高相關結果 (similarity={max_similarity:.3f})")
                return results
        
        # 2. 外部搜索（CVE, Google 等）
        if search_mode in ["external", "hybrid"]:
            if self.enable_external_search:
                external_results = await self._search_external(query, current_data)
                results.extend(external_results)
            else:
                logger.info("外部搜索已禁用")
        
        return results
    
    async def _search_internal(self, query: str, current_data: dict) -> list[dict]:
        """對內搜索 - 向量庫"""
        if not self.vector_store:
            return []
        
        try:
            # 向量相似度搜索
            vector_results = await self.vector_store.search(
                query=query,
                top_k=10
            )
            
            return [
                {
                    "type": "internal_knowledge",
                    "source": "vector_store",
                    "content": r.get("content", ""),
                    "relevance_score": r.get("score", 0.0),
                    "metadata": r.get("metadata", {}),
                }
                for r in vector_results
            ]
        
        except Exception as e:
            logger.error(f"Internal search error: {e}")
            return []
    
    async def _search_external(self, query: str, current_data: dict) -> list[dict]:
        """對外搜索 - 外部資源"""
        # (已實現的外部搜索代碼)
        ...
```

### 5.2 使用示例

```python
from services.core.aiva_core.cognitive_core.learning_system.rag_trigger import RAGTrigger
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore

# 初始化
vector_store = VectorStore(backend="chroma", persist_directory="data/vector_db/chroma")

rag_trigger = RAGTrigger(
    similarity_threshold=0.6,
    vector_store=vector_store,  # 內部搜索
    enable_external_search=True,  # 啟用外部搜索
)

# 混合搜索（先內部後外部）
results = await rag_trigger.search(
    query="XSS bypass WAF",
    current_data={...},
    search_mode="hybrid"
)

# 結果包含：
# - 內部向量庫的相關記錄
# - 外部 CVE、Exploit-DB、Google 的搜索結果
```

## 六、總結

### 6.1 現狀
- ✅ 向量庫已有 782 條能力記錄
- ❌ 整合模組的 JSONL 數據不在向量庫中
- ❌ 知識庫 Markdown 報告不在向量庫中

### 6.2 需要做的事
1. ✅ 創建同步腳本將 JSONL 添加到向量庫
2. ✅ 將知識庫 Markdown 添加到向量庫
3. ✅ 更新 RAG 觸發器支持雙向搜索
4. ✅ 實現混合搜索策略（先內後外）

### 6.3 最終效果
- ✅ RAG 可以搜索內部向量庫（6000+ 條記錄）
- ✅ RAG 可以搜索外部資源（CVE, Google 等）
- ✅ 智能判斷何時需要外部搜索
- ✅ 提供完整的知識檢索能力

### 6.4 數據流
```
執行任務 → 保存到整合模組 (JSONL) → 同步到向量庫 → RAG 可搜索

學習系統 → 觸發 RAG → 
    ├─ 先搜索內部向量庫 (快速)
    └─ 如果找不到 → 搜索外部資源 (全面)
```
