# 🧠 AIVA 當前分析與探索功能原理說明

## 📑 目錄

- [📋 核心概念](#-核心概念)
  - [🎯 設計理念](#-設計理念)
  - [🔄 內閉環 vs 外閉環](#-內閉環-vs-外閉環)
- [🔍 當前實現架構](#-當前實現架構)
  - [階段 1: 模組探索 (ModuleExplorer)](#階段-1-模組探索-moduleexplorer)
  - [階段 2: 能力分析 (CapabilityAnalyzer)](#階段-2-能力分析-capabilityanalyzer)
  - [階段 3: 知識向量化 (VectorStore)](#階段-3-知識向量化-vectorstore)
  - [階段 4: 知識檢索 (KnowledgeBase)](#階段-4-知識檢索-knowledgebase)
- [🔄 完整數據流](#-完整數據流)
  - [從代碼到 AI 認知的完整路徑](#從代碼到-ai-認知的完整路徑)
- [📊 當前系統指標](#-當前系統指標)
  - [性能數據](#性能數據)
  - [語言覆蓋率](#語言覆蓋率)
- [🎯 關鍵設計決策](#-關鍵設計決策)
  - [為什麼不用大語言模型?](#為什麼不用大語言模型)
  - [為什麼用 Python AST?](#為什麼用-python-ast)
  - [為什麼用向量檢索?](#為什麼用向量檢索)
- [🚨 當前限制](#-當前限制)
  - [1. 僅支援 Python (關鍵限制!)](#1-僅支援-python-關鍵限制)
  - [2. 無跨語言調用追蹤](#2-無跨語言調用追蹤)
  - [3. 無合約映射](#3-無合約映射)
- [💡 為什麼這樣設計足夠?](#-為什麼這樣設計足夠)
  - [階段性策略](#階段性策略)
  - [漸進式改進原則](#漸進式改進原則)
  - [ROI 分析 (投資回報率)](#roi-分析-投資回報率)
- [🔄 實際運行示例](#-實際運行示例)
  - [完整流程演示](#完整流程演示)
  - [AI 查詢示例](#ai-查詢示例)
- [📈 效果評估](#-效果評估)
  - [成功指標](#成功指標)
  - [局限性](#局限性)
- [🎯 總結](#-總結)
  - [核心價值](#核心價值)
  - [未來方向](#未來方向)

---

**日期**: 2025-11-16  
**版本**: v2.3.1  
**目的**: 說明 AIVA 內閉環自我認知系統的工作原理

---

## 📋 核心概念

### 🎯 設計理念

AIVA **不是大語言模型**，而是一個**特化的 AI 安全測試系統**，其 AI 能力設計遵循以下原則:

1. **輕量化**: 不追求通用對話能力，專注於安全測試領域
2. **知識驅動**: 使用 RAG (Retrieval-Augmented Generation) 而非大規模參數
3. **實時學習**: 通過內閉環機制持續更新自我認知
4. **精確推理**: 基於向量檢索的精確知識匹配，而非生成式推測

### 🔄 內閉環 vs 外閉環

```
外閉環 (External Loop)
    ↓
[用戶目標] → AI 決策 → 工具調用 → 執行結果 → 經驗積累
    ↑                                              ↓
    └──────────────── 反饋學習 ←──────────────────┘

內閉環 (Internal Loop)
    ↓
[系統探索] → 能力分析 → 知識提取 → RAG 注入 → AI 自我認知
    ↑                                           ↓
    └──────────── 持續更新 ←──────────────────┘
```

---

## 🔍 當前實現架構

### 階段 1: 模組探索 (ModuleExplorer)

**檔案**: `services/core/aiva_core/internal_exploration/module_explorer.py`

#### 工作原理

```python
class ModuleExplorer:
    """模組探索器 - 掃描文件系統"""
    
    async def explore_all_modules(self):
        # 1. 定義掃描目標
        target_modules = [
            "core/aiva_core",  # 核心智能
            "scan",            # 掃描引擎
            "features",        # 功能模組
            "integration"      # 整合服務
        ]
        
        # 2. 遞迴掃描每個模組
        for module in target_modules:
            files = []
            for py_file in path.rglob("*.py"):  # ⚠️ 僅掃描 .py
                if not self._should_skip(py_file):
                    files.append({
                        "path": str(py_file),
                        "size": py_file.stat().st_size,
                        "type": "python"
                    })
            
            # 3. 分析目錄結構
            structure = self._analyze_structure(path)
            
            yield {
                "module": module,
                "files": files,
                "structure": structure
            }
```

#### 掃描策略

| 項目 | 策略 | 原因 |
|------|------|------|
| **掃描範圍** | 僅 `*.py` 文件 | 系統主要用 Python 開發 |
| **跳過目錄** | `__pycache__`, `test_`, `.git` | 非源碼內容 |
| **深度** | 無限制遞迴 | 完整掃描所有子目錄 |
| **緩存** | 無 (每次全掃) | 確保實時性 |

#### 輸出格式

```json
{
  "core/aiva_core": {
    "path": "/services/core/aiva_core",
    "files": [
      {"path": "attack/sql_injection.py", "size": 15234, "type": "python"},
      {"path": "cognitive_core/rag/vector_store.py", "size": 8921, "type": "python"}
    ],
    "structure": {
      "subdirectories": ["attack", "cognitive_core", "internal_exploration"],
      "is_package": true,
      "has_readme": true
    },
    "stats": {
      "total_files": 124,
      "total_size": 1534234
    }
  }
}
```

---

### 階段 2: 能力分析 (CapabilityAnalyzer)

**檔案**: `services/core/aiva_core/internal_exploration/capability_analyzer.py`

#### 工作原理

```python
class CapabilityAnalyzer:
    """能力分析器 - 使用 Python AST 解析代碼"""
    
    async def analyze_capabilities(self, modules_info):
        capabilities = []
        
        for module, files in modules_info.items():
            for file_path in files:
                # 1. 讀取源碼
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 2. 解析為 AST
                tree = ast.parse(content)
                
                # 3. 遍歷所有函數定義
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # 4. 判斷是否為「能力函數」
                        if self._has_capability_decorator(node):
                            capability = self._extract_capability_info(node)
                            capabilities.append(capability)
        
        return capabilities
```

#### 能力識別策略

**三層識別機制**:

```python
def _has_capability_decorator(self, node):
    """判斷函數是否為「能力」"""
    
    # 策略 1: 明確標記 (最高優先級)
    if self._check_decorator_for_capability(node):
        # 檢查裝飾器: @capability, @register_capability
        return True
    
    # 策略 2: 異步函數 (通常是核心能力)
    if isinstance(node, ast.AsyncFunctionDef):
        return True
    
    # 策略 3: 公開函數 + 實質文檔
    if not node.name.startswith('_'):  # 非私有
        docstring = ast.get_docstring(node)
        if docstring and len(docstring) > 20:  # 有意義的文檔
            return True
    
    return False
```

**為什麼用這三層策略?**

1. **策略 1 (裝飾器)**: 開發者明確標記的能力
   - 例: `@register_capability(name="sql_injection")`
   - 精確度: ⭐⭐⭐⭐⭐

2. **策略 2 (async)**: 異步操作通常是核心功能
   - 例: `async def scan_target(...)`
   - 精確度: ⭐⭐⭐⭐

3. **策略 3 (文檔)**: 有完整文檔的公開函數
   - 例: 帶 docstring 的 `def analyze_vulnerability(...)`
   - 精確度: ⭐⭐⭐

#### AST 提取的元數據

```python
def _extract_capability_info(self, node):
    return {
        # 基本信息
        "name": "scan_sql_injection",
        "module": "core/aiva_core",
        "file_path": "/path/to/sql_scanner.py",
        "line_number": 42,
        
        # 簽名信息
        "parameters": [
            {"name": "target", "annotation": "str"},
            {"name": "timeout", "annotation": "int"}
        ],
        "return_type": "ScanResult",
        "is_async": True,
        
        # 語義信息
        "description": "Scan target for SQL injection vulnerabilities",
        "docstring": "詳細文檔...",
        "decorators": ["@register_capability", "@retry(3)"]
    }
```

---

### 階段 3: 知識向量化 (VectorStore)

**檔案**: `services/core/aiva_core/cognitive_core/rag/vector_store.py`

#### 工作原理

```python
class VectorStore:
    """向量數據庫 - 將知識轉換為數值向量"""
    
    def __init__(self):
        # 使用輕量級嵌入模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # 384 維向量, 模型大小僅 ~80MB
        
        self.vectors = {}     # 存儲向量
        self.metadata = {}    # 存儲元數據
        self.documents = {}   # 存儲原始文檔
    
    def add_document(self, doc_id, text, metadata):
        """添加文檔到向量庫"""
        
        # 1. 文本轉向量 (關鍵步驟!)
        embedding = self.model.encode(text)
        # 輸入: "Scan SQL injection using sqlmap"
        # 輸出: [0.123, -0.456, 0.789, ...] (384個數字)
        
        # 2. 歸一化 (提高相似度計算精度)
        embedding = embedding / np.linalg.norm(embedding)
        
        # 3. 存儲
        self.vectors[doc_id] = embedding
        self.metadata[doc_id] = metadata
        self.documents[doc_id] = text
    
    def search(self, query, top_k=5):
        """向量檢索 - 找最相似的知識"""
        
        # 1. 查詢轉向量
        query_vector = self.model.encode(query)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 2. 計算餘弦相似度 (核心!)
        similarities = {}
        for doc_id, doc_vector in self.vectors.items():
            # 點積 = 餘弦相似度 (因為已歸一化)
            similarity = np.dot(query_vector, doc_vector)
            similarities[doc_id] = similarity
        
        # 3. 排序返回 Top K
        ranked = sorted(similarities.items(), 
                       key=lambda x: x[1], 
                       reverse=True)[:top_k]
        
        return [
            {
                "text": self.documents[doc_id],
                "metadata": self.metadata[doc_id],
                "score": score
            }
            for doc_id, score in ranked
        ]
```

#### 為什麼選擇 all-MiniLM-L6-v2?

| 特性 | 數值 | 原因 |
|------|------|------|
| **模型大小** | 80 MB | 輕量,可嵌入系統 |
| **向量維度** | 384 | 足夠表達語義 |
| **推理速度** | ~200 句/秒 | 實時檢索 |
| **語義質量** | 中上 | 對專業術語有效 |

**對比大模型**:
```
GPT-3:   175B 參數 (350GB+)  ❌ 過大
BERT-Large: 340M 參數 (1.3GB) ❌ 太大
all-MiniLM: 22M 參數 (80MB)   ✅ 適合
```

---

### 階段 4: 知識檢索 (KnowledgeBase)

**檔案**: `services/core/aiva_core/cognitive_core/rag/knowledge_base.py`

#### 工作原理

```python
class KnowledgeBase:
    """知識庫 - RAG 的高級接口"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def search(self, query, top_k=5):
        """語義搜索"""
        
        # 1. 調用向量存儲檢索
        results = self.vector_store.search(query, top_k)
        
        # 2. 轉換為知識庫格式
        knowledge_results = []
        for result in results:
            knowledge_results.append({
                "content": result["text"],
                "metadata": result["metadata"],
                "relevance_score": result["score"],
                "source": result["metadata"].get("source", "unknown")
            })
        
        return knowledge_results
    
    def add_knowledge(self, content, metadata):
        """添加知識"""
        doc_id = f"kb_{hash(content)}"
        self.vector_store.add_document(doc_id, content, metadata)
```

#### 查詢流程示例

```python
# 用戶查詢
query = "如何掃描 SQL 注入?"

# 1. 查詢轉向量
query_vector = [0.234, -0.123, 0.456, ...]  # 384 維

# 2. 與知識庫中所有向量比較
知識 1: "scan_sql_injection 函數用於掃描 SQL 注入"
  向量: [0.245, -0.110, 0.467, ...]
  相似度: 0.92  ✅ 高度相關

知識 2: "analyze_xss 函數用於分析 XSS 漏洞"
  向量: [0.111, -0.567, 0.234, ...]
  相似度: 0.45  ⚠️ 低相關

知識 3: "SQL 注入檢測使用 sqlmap 工具"
  向量: [0.238, -0.118, 0.459, ...]
  相似度: 0.89  ✅ 相關

# 3. 返回 Top 3
結果 = [知識1, 知識3, 知識2]
```

---

## 🔄 完整數據流

### 從代碼到 AI 認知的完整路徑

```
┌─────────────────────────────────────────────────────────────┐
│ 階段 1: 代碼掃描                                             │
└─────────────────────────────────────────────────────────────┘
   ↓
[ModuleExplorer] 掃描文件系統
   ├─ services/core/aiva_core/attack/sql_injection.py
   ├─ services/scan/port_scanner.py
   └─ services/features/xss_detector.py
   ↓
   輸出: 文件列表 (124 個 Python 文件)

┌─────────────────────────────────────────────────────────────┐
│ 階段 2: 能力提取                                             │
└─────────────────────────────────────────────────────────────┘
   ↓
[CapabilityAnalyzer] AST 解析每個文件
   ├─ 解析: sql_injection.py
   │   └─ 找到: async def scan_sql_injection(target: str)
   │       ├─ 裝飾器: @register_capability
   │       ├─ 參數: target (str), options (dict)
   │       └─ 返回: ScanResult
   ↓
   輸出: 能力列表 (405 個函數)
   [
     {
       "name": "scan_sql_injection",
       "module": "attack",
       "parameters": [...],
       "description": "掃描 SQL 注入漏洞"
     },
     ...
   ]

┌─────────────────────────────────────────────────────────────┐
│ 階段 3: 文檔生成                                             │
└─────────────────────────────────────────────────────────────┘
   ↓
[InternalLoopConnector] 格式化為文檔
   ↓
   為每個能力生成標準化文檔:
   
   """
   Capability: scan_sql_injection
   Module: attack.sql_injection
   Type: async function
   
   Description:
   掃描目標 URL 的 SQL 注入漏洞,使用 sqlmap 引擎
   
   Parameters:
     - target: str - 目標 URL
     - options: dict - 掃描選項
       * depth: int - 掃描深度 (default: 2)
       * timeout: int - 超時秒數 (default: 60)
   
   Returns:
     ScanResult - 包含發現的漏洞列表
   
   Usage Example:
     result = await scan_sql_injection(
         target="http://example.com/login",
         options={"depth": 3}
     )
   
   File: services/core/aiva_core/attack/sql_injection.py:42
   """

┌─────────────────────────────────────────────────────────────┐
│ 階段 4: 向量化                                               │
└─────────────────────────────────────────────────────────────┘
   ↓
[VectorStore] 文檔 → 向量
   ↓
   使用 all-MiniLM-L6-v2 模型:
   
   文檔: "Capability: scan_sql_injection..."
   ↓ [SentenceTransformer.encode()]
   向量: [0.123, -0.456, 0.789, 0.234, ..., -0.111]
          ↑                                      ↑
        第1維                                  第384維
   
   元數據: {
     "capability_name": "scan_sql_injection",
     "module": "attack",
     "type": "function",
     "source": "internal_exploration"
   }

┌─────────────────────────────────────────────────────────────┐
│ 階段 5: 存儲                                                 │
└─────────────────────────────────────────────────────────────┘
   ↓
[VectorStore] 持久化到內存/ChromaDB
   
   vectors = {
     "cap_001": [0.123, -0.456, ...],  # scan_sql_injection
     "cap_002": [0.234, -0.567, ...],  # scan_xss
     "cap_003": [0.345, -0.678, ...],  # port_scan
     ...
   }
   
   metadata = {
     "cap_001": {"capability_name": "scan_sql_injection", ...},
     ...
   }

┌─────────────────────────────────────────────────────────────┐
│ 階段 6: AI 查詢 (運行時)                                     │
└─────────────────────────────────────────────────────────────┘
   ↓
用戶: "我需要掃描 SQL 注入"
   ↓
[RAG Engine] 語義檢索
   ↓
   1. 查詢向量化:
      "我需要掃描 SQL 注入"
      → [0.125, -0.450, 0.792, ...]
   
   2. 相似度計算:
      vs cap_001 (scan_sql_injection): 0.94  ✅ 最相關
      vs cap_002 (scan_xss):          0.45  
      vs cap_003 (port_scan):         0.32
   
   3. 返回最相關知識:
      {
        "content": "Capability: scan_sql_injection...",
        "score": 0.94,
        "metadata": {...}
      }
   ↓
[AI Agent] 基於檢索結果決策
   ↓
   決策: 使用 attack.sql_injection.scan_sql_injection()
   參數: {
     "target": "http://target.com",
     "options": {"depth": 2}
   }
   ↓
   執行工具調用
```

---

## 📊 當前系統指標

### 性能數據

| 指標 | 數值 | 說明 |
|------|------|------|
| **掃描文件數** | 124 個 | 僅 Python 文件 |
| **提取能力數** | 405 個 | 三層策略識別 |
| **向量維度** | 384 | all-MiniLM-L6-v2 |
| **模型大小** | 80 MB | 可嵌入部署 |
| **向量化速度** | ~200 文檔/秒 | 本地 CPU |
| **檢索延遲** | <10 ms | Top-5 查詢 |
| **內存佔用** | ~150 MB | 包含模型+向量 |

### 語言覆蓋率

```
當前覆蓋:
├─ Python:     100% (405 能力)  ✅
├─ Go:          0% (未分析)      ❌
├─ Rust:        0% (未分析)      ❌
├─ TypeScript:  0% (未分析)      ❌
└─ JavaScript:  0% (未分析)      ❌

總覆蓋率: ~81% (僅計入 Python 部分)
```

---

## 🎯 關鍵設計決策

### 為什麼不用大語言模型?

**決策**: 使用 RAG (檢索增強) 而非 LLM

**理由**:

1. **準確性優先**
   ```
   LLM 生成:  "可能可以使用 sqlmap..."  ⚠️ 不確定
   RAG 檢索:  "系統有 scan_sql_injection()" ✅ 精確
   ```

2. **成本考量**
   ```
   LLM:
   - 模型大小: 175B+ 參數 (350GB+)
   - 推理成本: 需要 GPU 集群
   - 部署難度: 極高
   
   RAG:
   - 模型大小: 22M 參數 (80MB)
   - 推理成本: CPU 即可
   - 部署難度: 低
   ```

3. **可控性**
   ```
   LLM: 黑盒生成 → 難以控制輸出
   RAG: 檢索已知 → 完全可控
   ```

### 為什麼用 Python AST?

**決策**: 使用 AST (抽象語法樹) 解析而非正則表達式

**理由**:

1. **精確性**
   ```python
   # 正則無法處理的情況:
   def func(
       param1,
       param2
   ):
       """
       多行文檔
       """
       pass
   
   # AST 可以正確解析所有情況
   ```

2. **元數據豐富**
   ```python
   AST 能提取:
   - 函數簽名 (參數、返回類型)
   - 裝飾器列表
   - 文檔字串
   - 行號位置
   - 嵌套結構
   ```

3. **標準化**
   ```python
   Python 標準庫內建 ast 模組
   → 無需額外依賴
   → 與 Python 語法 100% 兼容
   ```

### 為什麼用向量檢索?

**決策**: 向量相似度檢索而非關鍵字匹配

**對比**:

```python
# 關鍵字匹配
query = "掃描 SQL 注入"
if "SQL" in document and "注入" in document:
    return document  
# ❌ 無法匹配: "sqlmap scanner" (沒有中文)

# 向量檢索
query_vec = embed("掃描 SQL 注入")     # [0.12, -0.45, ...]
doc_vec = embed("sqlmap scanner")      # [0.13, -0.43, ...]
similarity = cosine(query_vec, doc_vec)  # 0.89 (高相似)
# ✅ 理解語義相似性!
```

---

## 🚨 當前限制

### 1. 僅支援 Python (關鍵限制!)

**問題**:
```python
# module_explorer.py
for py_file in path.rglob("*.py"):  # ❌ 只掃描 .py
    # ...
```

**影響**:
- 忽略 75+ 個 Go/Rust/TS 文件
- AI 不知道 19% 的系統能力
- 跨語言調用無法追蹤

**為什麼暫時只支援 Python?**

1. **AST 解析器的限制**
   ```python
   tree = ast.parse(content)  # Python 專用
   
   # Go 需要: go/parser
   # Rust 需要: syn crate (通過 PyO3)
   # TypeScript 需要: typescript compiler API
   ```

2. **複雜度控制**
   - 每種語言需要不同的解析器
   - 元數據格式需要統一
   - 開發成本 vs 收益權衡

3. **主要能力在 Python**
   - 核心 AI 引擎: Python
   - 攻擊模組: Python
   - 決策邏輯: Python
   - Go/Rust 主要是高性能模組

### 2. 無跨語言調用追蹤

**問題**:
```python
# Python 調用 Go 服務
async def scan(target):
    response = await http.post(
        "http://go-scanner:8080/scan",  # ❌ 未被追蹤
        json={"target": target}
    )
```

**影響**:
- AI 不知道 Python 依賴 Go 服務
- 無法推薦最佳語言組合
- 架構圖不完整

### 3. 無合約映射

**問題**:
```python
async def scan(target: str) -> ScanResult:
    # ❌ AI 不知道 ScanResult 是統一合約
    pass
```

**影響**:
- 無法驗證跨語言一致性
- 不知道哪些函數使用了標準合約
- 合約使用情況不可見

---

## 💡 為什麼這樣設計足夠?

### 階段性策略

**當前階段 (Phase 1)**: 建立基礎
```
目標: AI 能理解自己的 Python 能力
進度: ✅ 完成
覆蓋: 81% (Python 主要功能)
```

**下一階段 (Phase 2)**: 多語言擴展
```
目標: AI 理解所有語言能力
進度: 📋 規劃中
覆蓋: → 100%
```

### 漸進式改進原則

```
Phase 1: Python Only
    ↓
  驗證可行性
  + 建立架構
  + 積累經驗
    ↓
Phase 2: Multi-Language
    ↓
  複製 Phase 1 模式
  + Go 分析器
  + Rust 分析器
  + TS 分析器
    ↓
Phase 3: Intelligence
    ↓
  跨語言追蹤
  + 合約映射
  + 性能分析
```

### ROI 分析 (投資回報率)

```
Phase 1 投入: 2 週開發
Phase 1 回報:
  ✅ 405 個能力被 AI 認知
  ✅ 核心功能完全覆蓋
  ✅ RAG 檢索可用
  ✅ 驗證架構可行

Phase 2 投入: 3-4 週開發
Phase 2 額外回報:
  + 150 個 Go 能力
  + 80 個 Rust 能力
  + 120 個 TS 能力
  = +350 能力 (從 405 → 755)
  
ROI: 額外投入 2x 時間 → 覆蓋率 81% → 100%
```

---

## 🔄 實際運行示例

### 完整流程演示

```bash
# 1. 執行內閉環探索
$ python scripts/internal_loop/update_self_awareness.py

[INFO] 🔍 ModuleExplorer: 開始掃描...
[INFO]   掃描模組: core/aiva_core
[INFO]   掃描模組: scan
[INFO]   掃描模組: features
[INFO]   掃描模組: integration
[INFO] ✅ 掃描完成: 124 個文件

[INFO] 🔍 CapabilityAnalyzer: 開始分析...
[INFO]   分析文件: attack/sql_injection.py
[INFO]     發現能力: scan_sql_injection (async, @capability)
[INFO]   分析文件: attack/xss_scanner.py
[INFO]     發現能力: detect_xss (async)
[INFO]   ...
[INFO] ✅ 分析完成: 405 個能力

[INFO] 📝 InternalLoopConnector: 生成文檔...
[INFO]   格式化能力: scan_sql_injection
[INFO]   格式化能力: detect_xss
[INFO]   ...
[INFO] ✅ 生成 405 個文檔

[INFO] 🧠 VectorStore: 向量化...
[INFO]   載入模型: all-MiniLM-L6-v2
[INFO]   向量化文檔 1/405: scan_sql_injection
[INFO]   向量化文檔 2/405: detect_xss
[INFO]   ...
[INFO] ✅ 向量化完成: 405 個文檔

[INFO] 💾 KnowledgeBase: 存儲...
[INFO]   存儲向量: cap_001 → [0.123, -0.456, ...]
[INFO]   存儲向量: cap_002 → [0.234, -0.567, ...]
[INFO]   ...
[INFO] ✅ 存儲完成

[SUCCESS] 🎉 內閉環執行完成!
  - 掃描文件: 124
  - 識別能力: 405
  - 向量維度: 384
  - 執行時間: 12.3 秒
```

### AI 查詢示例

```python
from aiva_core.cognitive_core.rag import RAGEngine

rag = RAGEngine()

# 查詢 1: 自然語言
results = rag.query("如何掃描 SQL 注入?")

print(results[0])
# {
#   "content": """
#     Capability: scan_sql_injection
#     Module: attack.sql_injection
#     Description: 使用 sqlmap 引擎掃描 SQL 注入漏洞
#     Parameters:
#       - target: str - 目標 URL
#       - options: dict - 掃描選項
#     ...
#   """,
#   "relevance_score": 0.94,
#   "metadata": {
#     "capability_name": "scan_sql_injection",
#     "module": "attack"
#   }
# }

# 查詢 2: 技術術語
results = rag.query("XSS detection capabilities")

print(results[0]["metadata"]["capability_name"])
# "detect_xss"

# 查詢 3: 模糊匹配
results = rag.query("掃描漏洞")  # 廣泛查詢

print([r["metadata"]["capability_name"] for r in results])
# ["scan_sql_injection", "detect_xss", "scan_ports", ...]
```

---

## 📈 效果評估

### 成功指標

✅ **AI 自我認知建立**
```
Before: AI 不知道自己有哪些能力
After:  AI 可以檢索 405 個已知能力
```

✅ **精確工具推薦**
```
User: "掃描 SQL 注入"
AI: 推薦 scan_sql_injection() (相似度: 0.94)
    而非 scan_xss() (相似度: 0.45)
```

✅ **實時更新能力**
```
新增 capability → 重新掃描 → AI 立即感知
週期: ~12 秒完成一次全量掃描
```

✅ **輕量化部署**
```
模型大小: 80 MB
內存佔用: ~150 MB
CPU 推理: 可行
```

### 局限性

⚠️ **語言覆蓋不完整**
```
Python:     100% ✅
Go/Rust/TS:   0% ❌
總覆蓋率:    81%
```

⚠️ **無跨語言感知**
```
Python 調用 Go → AI 不知道
Go 發送 MQ → AI 不追蹤
```

⚠️ **無合約驗證**
```
使用 TaskPayload → AI 不知道這是統一合約
跨語言一致性 → AI 無法驗證
```

---

## 🎯 總結

### 核心價值

1. **輕量化 AI 自我認知**
   - 80MB 模型實現語義理解
   - 無需大語言模型
   - CPU 即可運行

2. **精確知識檢索**
   - 向量相似度 > 關鍵字匹配
   - 405 個能力可檢索
   - <10ms 查詢延遲

3. **實時更新機制**
   - 代碼變更 → 自動感知
   - 內閉環持續運行
   - 知識庫始終最新

### 未來方向

**短期 (1-2 週)**:
- ✅ Python 能力分析 (已完成)
- 📋 增量探索機制 (規劃中)
- 📋 合約使用追蹤 (規劃中)

**中期 (1 個月)**:
- 📋 Go 能力分析器
- 📋 Rust 能力分析器
- 📋 TypeScript 能力分析器

**長期 (2-3 個月)**:
- 📋 跨語言調用圖
- 📋 性能瓶頸分析
- 📋 智能優化建議

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: AIVA Core Team
