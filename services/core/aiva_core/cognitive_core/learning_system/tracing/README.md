# Tracing 執行追蹤模組

> **路徑**: `cognitive_core/learning_system/tracing`  
> **狀態**: ✅ 正常 | **Python 文件數**: 3 | **最後更新**: 2026-04-05

## 概述

記錄任務執行過程中的所有詳細信息，提供統一的追蹤記錄介面。支援多種追蹤類型，用於調試、分析和強化學習回饋。

## 📄 檔案詳細資訊 (Files Details)

### `trace_recorder.py`
**說明**: Trace Recorder - 軌跡記錄器

**類別 (Classes)**:
- `TraceType` - 軌跡類型
- `TraceEntry` - 軌跡條目
- `ExecutionTrace` - 執行軌跡
- `TraceRecorder` - 軌跡記錄器

### `unified_tracer.py`
**說明**: Unified Tracer - 統一追蹤介面

**類別 (Classes)**:
- `TraceType` - 追蹤類型枚舉
- `ExecutionTrace` - 執行追蹤記錄
- `UnifiedTracer` - 統一追蹤記錄器
**函式 (Functions)**:
- `get_global_tracer()` - 獲取全局統一追蹤記錄器實例
- `record_execution_trace()` - 記錄執行追蹤（便利函數）

