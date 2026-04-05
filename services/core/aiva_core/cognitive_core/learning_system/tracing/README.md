# Tracing 執行追蹤模組

> **路徑**: `cognitive_core/learning_system/tracing`  
> **狀態**: ✅ 正常 | **Python 文件數**: 3 | **最後更新**: 2026-04-05

## 概述

記錄任務執行過程中的所有詳細信息，提供統一的追蹤記錄介面。支援多種追蹤類型，用於調試、分析和強化學習回饋。

## 核心組件

### trace_recorder.py

- `TraceType` - 軌跡類型枚舉
  - TASK_START/TASK_END - 任務開始/結束
  - HTTP_REQUEST/HTTP_RESPONSE - HTTP 請求/回應
  - RPC_CALL/RPC_RESPONSE - RPC 調用/回應
  - LOG, ERROR, TOOL_OUTPUT, DECISION, VALIDATION
- `TraceEntry` - 軌跡條目，記錄單一執行步驟
- `ExecutionTrace` - 執行軌跡，包含完整執行記錄
- `TraceRecorder` - 軌跡記錄器
  - 開始/結束軌跡記錄
  - 記錄各類型軌跡
  - 按任務或類型查詢軌跡

### unified_tracer.py

- `TraceType` - 追蹤類型枚舉（擴展版）
  - EXECUTION, AST_ANALYSIS, FUNCTION_CALL
  - VARIABLE_ACCESS, CONTROL_FLOW
  - SESSION_START/SESSION_END
- `ExecutionTrace` - 執行追蹤記錄數據類
- `UnifiedTracer` - 統一追蹤記錄器
  - 整合 trace_recorder 和 RabbitMQ 追蹤
  - 會話管理
  - 符合 aiva_common 規範

### execution_tracer.py

- 向後相容模組
- 重新導出 trace_recorder 組件
- `get_global_recorder()` - 獲取全局記錄器
- `record_execution_trace()` - 便捷記錄方法

## 依賴關係

- 內部依賴：
  - `aiva_common.schemas` (SessionState, TraceRecord)
  - `aiva_common.error_handling`
- 外部依賴：`dataclasses`, `uuid`, `json`, `logging`

## 使用範例

```python
from cognitive_core.learning_system.tracing import (
    TraceRecorder, TraceType, UnifiedTracer
)

# 基本軌跡記錄
recorder = TraceRecorder()
trace = recorder.start_trace(plan_id="plan_001")

recorder.record(
    trace_session_id=trace.trace_session_id,
    trace_type=TraceType.TASK_START,
    content={"task_name": "掃描端口"},
    task_id="task_001"
)

# 結束軌跡
recorder.end_trace(trace.trace_session_id)

# 統一追蹤器
tracer = UnifiedTracer()
tracer.start_session("session_001")
tracer.record_trace(
    trace_type=TraceType.EXECUTION,
    module_name="scanner",
    function_name="scan_ports"
)

# 便捷方法
from cognitive_core.learning_system.tracing.execution_tracer import record_execution_trace

record_execution_trace(
    trace_session_id="session_001",
    trace_type=TraceType.HTTP_REQUEST,
    content={"url": "http://target.com", "method": "GET"}
)
```
