# Output 輸出模組

> **路徑**: `services/core/aiva_core/core_capabilities/output`  
> **狀態**: ✅ 正常 | **Python 文件數**: 2 | **最後更新**: 2026-04-05

## 概述

負責將函數調用負載包裝成標準化的 AIVA 消息格式，用於模組間通訊。

## 核心組件

### to_functions.py
**主要函數：**
- `to_function_message()` - 將函數負載包裝成 AIVA 消息
  - 輸入：Topic、FunctionTaskPayload、trace_id、correlation_id
  - 輸出：完整的 AivaMessage（含正確的 Header）

### __init__.py
- 模組初始化和導出

## 消息格式

```python
AivaMessage(
    header=MessageHeader(
        message_id=new_id("msg"),
        trace_id=trace_id,
        correlation_id=correlation_id,
        source_module=ModuleName.CORE,
    ),
    topic=topic,
    payload=payload.model_dump(),
)
```

## 依賴關係

- `aiva_common.enums.modules` - ModuleName, Topic
- `aiva_common.schemas` - AivaMessage, FunctionTaskPayload, MessageHeader
- `aiva_common.utils` - new_id 生成器
