"""
Execution Tracer - 執行追蹤器模組

為了向後相容性，這個模組重新導出 execution.trace_recorder 中的組件
新代碼應該直接使用 execution.trace_recorder
"""

# 重新導出 trace_recorder 中的組件以維持向後相容性
from .execution.trace_recorder import (
    ExecutionTrace,
    TraceType,
    TraceRecorder,
    get_global_recorder,
    record_execution_trace
)

__all__ = [
    "ExecutionTrace",
    "TraceType", 
    "TraceRecorder",
    "get_global_recorder",
    "record_execution_trace"
]