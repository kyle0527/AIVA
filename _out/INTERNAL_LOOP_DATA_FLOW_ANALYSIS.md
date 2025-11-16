# 內閉環數據流分析報告
**日期**: 2025-11-16  
**問題**: `update_self_awareness.py` 執行時找到 405 個能力,但寫入 RAG 時全部失敗

## 📊 數據流程追蹤

### 階段 1: 能力掃描 ✅ (成功)
```
ModuleExplorer → CapabilityAnalyzer
- 掃描了 4 個模組
- 找到了 405 個能力函數
```

**關鍵代碼**: `capability_analyzer._extract_capability_info()`
```python
# 返回結構:
{
    'name': node.name,
    'parameters': [...],  # ✅ list[dict]
    'return_type': str,   # ✅ str or None
    'decorators': [...],  # ✅ list[str]
    'docstring': str,     # ✅ str or None
    'description': str    # ✅ str
}
```

### 階段 2: 轉換為文檔 ✅ (成功)
```
InternalLoopConnector._convert_to_documents()
- 輸入: 405 個 capability dicts
- 輸出: 405 個 document dicts
```

**關鍵代碼**: `internal_loop_connector._convert_to_documents()`
```python
# 返回結構:
{
    "content": str,      # ✅ 字串
    "metadata": {        # ✅ 字典
        "type": "capability",
        "capability_name": cap["name"],
        "module": cap["module"],
        "file_path": cap["file_path"],  # ⚠️ 可能是 Path 物件!
        "is_async": cap.get("is_async", False),
        "parameters_count": len(cap["parameters"]),
        "source": "internal_exploration",
        "sync_timestamp": datetime.now(timezone.utc).isoformat()
    }
}
```

### 階段 3: 注入到 RAG ❌ (失敗)
```
InternalLoopConnector._inject_to_rag() 
→ KnowledgeBase.add_knowledge()
→ VectorStore.add_document()
→ model.encode()  # ❌ 這裡出錯!
```

## 🔍 問題根源分析

### 錯誤訊息
```
Failed to add knowledge: 'str' object has no attribute 'items'
```

### 可能原因

#### 原因 1: `file_path` 是 Path 物件 ⭐⭐⭐⭐⭐
```python
# capability_analyzer 返回的數據:
capability = {
    "file_path": Path("C:/D/fold7/AIVA-git/services/..."),  # Path 物件!
    ...
}

# _convert_to_documents 直接使用:
metadata = {
    "file_path": cap["file_path"],  # 傳入了 Path 物件
    ...
}
```

**問題**: `sentence_transformers` 或某個內部處理可能期望所有 metadata 值都是可 JSON 序列化的基本類型

#### 原因 2: metadata 被當作參數傳遞時的類型問題
```python
# knowledge_base.py
self.vector_store.add_document(doc_id, content, metadata)

# vector_store.py  
def add_document(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None):
    model = self._get_embedding_model()
    embedding = model.encode(text, ...)  # ← 這裡不應該碰 metadata
    self.metadata[doc_id] = metadata or {}  # ← 但錯誤發生在這之前
```

## 🛠️ 修復方案

### 方案 1: 確保 metadata 所有值都是基本類型 (✅ 推薦)

**位置**: `internal_loop_connector.py` 的 `_convert_to_documents()`

```python
def _convert_to_documents(self, capabilities: list[dict]) -> list[dict]:
    documents = []
    
    for cap in capabilities:
        # 構建可讀的能力描述
        params_str = ", ".join(
            f"{p['name']}: {p.get('annotation', 'Any')}" 
            for p in cap["parameters"]
        )
        
        content_parts = [...]
        content = "\n".join(content_parts)
        
        doc = {
            "content": content,
            "metadata": {
                "type": "capability",
                "capability_name": cap["name"],
                "module": cap["module"],
                "file_path": str(cap["file_path"]),  # ⭐ 轉換為字串!
                "is_async": bool(cap.get("is_async", False)),
                "parameters_count": int(len(cap["parameters"])),
                "source": "internal_exploration",
                "sync_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        documents.append(doc)
    
    return documents
```

### 方案 2: 在 _inject_to_rag 中清理 metadata (已實現但不夠)

目前的實現:
```python
for i, doc in enumerate(documents):
    try:
        metadata_dict = {}
        for key, value in doc["metadata"].items():
            if isinstance(value, (str, int, float, bool)):
                metadata_dict[key] = value
            elif value is None:
                metadata_dict[key] = None
            else:
                metadata_dict[key] = str(value)  # 轉字串
```

**問題**: 如果 `doc["metadata"]` 本身不是 dict,而是別的東西,`items()` 會失敗!

### 方案 3: 檢查 capability_analyzer 返回的數據類型

**位置**: `capability_analyzer.py` 的 `_extract_capability_info()`

確保返回:
```python
return {
    "name": node.name,
    "parameters": parameters,
    "return_type": return_type,
    "decorators": decorators,
    "docstring": docstring,
    "description": description,
    "module": module_name,         # ✅ str
    "file_path": str(file_path),   # ⭐ 轉為 str
    "is_async": isinstance(node, ast.AsyncFunctionDef)  # ✅ bool
}
```

## 📝 立即修復步驟

1. ✅ **修復 `_convert_to_documents()`**: 確保所有 metadata 值都是基本類型
2. ✅ **驗證**: 重新執行 `update_self_awareness.py`
3. ✅ **確認**: 檢查是否成功寫入 405 個文檔

## 🎯 根本原因總結

**Python Path 物件不能直接序列化到某些後端系統**

- `capability_analyzer` 使用 `Path` 物件來表示檔案路徑
- 這些 `Path` 物件被直接放入 `metadata` 字典
- 當 `VectorStore` 或 embedding model 嘗試處理這些數據時失敗
- 錯誤訊息 `'str' object has no attribute 'items'` 可能是因為某個內部處理將 Path 轉為 str,然後錯誤地嘗試對 str 呼叫 `.items()`

**解決方案**: 在數據流的早期階段(capability 提取或文檔轉換時)就將所有非基本類型轉換為字串。
