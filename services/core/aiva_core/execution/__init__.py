"""
Execution Module - 攻擊計畫執行模組

提供攻擊計畫的執行、追蹤和管理功能
使用統一追蹤器整合trace_recorder和trace_logger功能
"""

from .attack_plan_mapper import AttackPlanMapper
from .plan_executor import PlanExecutor
from .unified_tracer import (
    UnifiedTracer,
    TraceType,
    ExecutionTrace,
    get_global_tracer,
    record_execution_trace,
)

# 向後相容性別名
TraceLogger = UnifiedTracer
TraceRecorder = UnifiedTracer

__all__ = [
    "PlanExecutor", 
    "AttackPlanMapper",
    "UnifiedTracer",
    "TraceType",
    "ExecutionTrace",
    "get_global_tracer",
    "record_execution_trace",
    # 向後相容性
    "TraceLogger", 
    "TraceRecorder",
]
