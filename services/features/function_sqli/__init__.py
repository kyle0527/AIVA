"""
SQL Injection Detection Module

SQL 注入檢測功能模組。

遵循 README 規範：
- 移除 ImportError fallback 機制
- 確保依賴可用，導入失敗時明確報錯
"""

__version__ = "1.0.0"

# 導入核心組件 - 遵循 README 規範，不使用 try/except fallback
from .smart_detection_manager import SmartDetectionManager
from .task_queue import SqliTaskQueue, QueuedTask
from .detection_models import DetectionModels
from .exceptions import SQLiException
from .payload_wrapper_encoder import PayloadWrapperEncoder
from .telemetry import SqliExecutionTelemetry

# HackingTool 整合組件
from .hackingtool_config import (
    HACKINGTOOL_SQL_CONFIGS, 
    HackingToolSQLIntegrator,
    sql_integrator
)
from .hackingtool_manager import sql_tool_manager
from .engines.hackingtool_engine import HackingToolDetectionEngine

__all__ = [
    "SmartDetectionManager",
    "SqliTaskQueue",
    "QueuedTask",
    "DetectionModels",
    "SQLiException",
    "PayloadWrapperEncoder",
    "SqliExecutionTelemetry",
    # HackingTool 整合組件
    "HACKINGTOOL_SQL_CONFIGS",
    "HackingToolSQLIntegrator",
    "sql_integrator",
    "sql_tool_manager",
    "HackingToolDetectionEngine",
]
