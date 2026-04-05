# API 統一函數調用模組

> **路徑**: `service_backbone/api/`  
> **狀態**: ✅ 正常 | **Python 文件數**: 5 | **最後更新**: 2026-04-05  
> **父模組**: [Service Backbone](../README.md)

## 概述

提供統一的函數調用接口和 AI 服務能力，支援跨模組的動態能力調用，解決「規劃器如何實際調用工具」的架構問題。

## 核心組件

### ai_service.py ⭐ 新增
- `AIService` - AI 持續運行服務
  - API 模式 (FastAPI)
  - 監控模式 (系統健康監控)
  - 互動模式 (CLI 互動)

### app.py
- `AIVAServer` - FastAPI 主應用伺服器
- `create_app()` - 應用工廠函數
- `start_server()` - 伺服器啟動函數

### sse.py
- Server-Sent Events 長連接傳輸端點
- 用於實時更新 AI 處理狀態

### __init__.py
- 模組初始化和導出

## 調用流程

```
任務執行器 (TaskExecutor)
        ↓
UnifiedFunctionCaller
        ↓
CapabilityRegistry（查找能力）
        ↓
實際能力模組執行
        ↓
FunctionCallResult（封裝結果）
```

## 使用方式

```python
caller = UnifiedFunctionCaller()
result = await caller.call_function(
    module="features.scanner",
    function="execute_scan",
    params={"target": "http://example.com"}
)
```

## 依賴關係

- `core_capabilities.capability_registry` - 能力註冊表
- 支援異步調用 (`async/await`)
