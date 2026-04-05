"""
Execution Tracer - 執行追蹤器模組

為了向後相容性，這個模組重新導出 trace_recorder 中的組件
新代碼應該直接使用 trace_recorder
"""

from typing import Any

# 重新導出 trace_recorder 中的組件以維持向後相容性
from .trace_recorder import (
    ExecutionTrace,
    TraceType,
    TraceRecorder,
)

# 全局記錄器實例（向後相容）
_global_recorder: TraceRecorder | None = None


def get_global_recorder() -> TraceRecorder:
    """獲取全局軌跡記錄器實例"""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = TraceRecorder()
    return _global_recorder


def record_execution_trace(
    trace_session_id: str,
    trace_type: TraceType,
    content: dict[str, Any],
    task_id: str | None = None
) -> None:
    """便捷方法：記錄執行軌跡"""
    recorder = get_global_recorder()
    recorder.record(
        trace_session_id=trace_session_id,
        trace_type=trace_type,
        content=content,
        task_id=task_id
    )


__all__ = [
    "ExecutionTrace",
    "TraceType", 
    "TraceRecorder",
    "get_global_recorder",
    "record_execution_trace"
]