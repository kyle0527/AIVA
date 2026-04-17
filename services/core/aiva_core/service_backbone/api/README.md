# API 統一函數調用模組

> **路徑**: `service_backbone/api/`  
> **狀態**: ✅ 正常 | **Python 文件數**: 5 | **最後更新**: 2026-04-05  
> **父模組**: [Service Backbone](../README.md)

## 概述

提供統一的函數調用接口和 AI 服務能力，支援跨模組的動態能力調用，解決「規劃器如何實際調用工具」的架構問題。

## 📄 檔案詳細資訊 (Files Details)

### `ai_service.py`
**說明**: AIVA AI 持續運作服務

**類別 (Classes)**:
- `AIService` - AI 持續運作服務
**函式 (Functions)**:
- `signal_handler()` - 處理中斷信號
- `main()` - 主函數

### `app.py`
**說明**: 無特定描述。


### `sse.py`
**說明**: SSE (Server-Sent Events) 端點處理器

**類別 (Classes)**:
- `LogEvent` - 日誌事件
- `StatusEvent` - 狀態事件

### `unified_function_caller.py`
**說明**: 統一功能調用器 - 跨語言模組調用系統

**類別 (Classes)**:
- `FunctionCallResult` - 功能調用結果
- `ModuleEndpoint` - 模組端點配置
- `UnifiedFunctionCaller` - 統一功能調用器
**函式 (Functions)**:
- `get_unified_caller()` - 獲取統一調用器實例

