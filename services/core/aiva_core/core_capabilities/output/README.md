# 📤 Output - 輸出轉換系統

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **代碼量**: 1 個 Python 檔案，約 20 行代碼  
> **角色**: AIVA 的「格式轉換器」- 將結果轉換為函數調用格式

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心功能](#核心功能)
- [使用範例](#使用範例)

---

## 🎯 模組概述

**Output** 子模組負責將處理後的掃描結果轉換為可執行的函數調用格式，方便後續的自動化處理和 API 整合。

### 核心能力
1. **格式轉換** - 將結果轉換為函數調用格式
2. **序列化** - 支援 JSON、Python 函數等格式
3. **API 整合** - 方便與外部系統對接

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **to_functions.py** | 20 | 輸出轉函數調用 - 格式轉換工具 | ✅ 生產 |
| **__init__.py** | - | 模組初始化 | - |

---

## 🔧 核心功能

### ToFunctions - 輸出轉換器

**檔案**: `to_functions.py` (20 行)

將掃描結果轉換為函數調用格式，支援多種輸出模式。

#### 基本用法

```python
def to_function_call(
    result: Dict[str, Any],
    format: str = "json"
) -> str:
    """將結果轉換為函數調用格式
    
    Args:
        result: 處理後的結果
        format: 輸出格式 (json, python, curl)
        
    Returns:
        str: 函數調用字串
    """
    
    if format == "json":
        return json.dumps(result, indent=2)
    elif format == "python":
        return f"handle_scan_result({result})"
    elif format == "curl":
        return generate_curl_command(result)
```

---

## 🚀 使用範例

### JSON 格式輸出

```python
from core_capabilities.output import to_function_call

result = {
    "scan_id": "scan-001",
    "findings": [...],
    "summary": {...}
}

# 轉換為 JSON
json_output = to_function_call(result, format="json")
print(json_output)
```

### Python 函數調用格式

```python
# 轉換為 Python 函數調用
python_call = to_function_call(result, format="python")
# 輸出: handle_scan_result({'scan_id': 'scan-001', ...})

# 可直接執行
exec(python_call)
```

### cURL 命令格式

```python
# 轉換為 cURL 命令（用於 API 調用）
curl_cmd = to_function_call(result, format="curl")
# 輸出:
# curl -X POST https://api.example.com/results \
#   -H "Content-Type: application/json" \
#   -d '{"scan_id": "scan-001", ...}'
```

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Processing 子模組](../processing/README.md) - 結果處理
- [Plugins 子模組](../plugins/README.md) - AI 摘要插件

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
