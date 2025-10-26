"""
統一日誌格式器
按照 AIVA 日誌記錄標準實現跨模組統一的日誌格式

符合 services/features/docs/LOGGING_STANDARDS.md 規範
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(str, Enum):
    """日誌級別枚舉"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIVALogFormatter(logging.Formatter):
    """
    AIVA 統一日誌格式器
    
    產生結構化 JSON 日誌，便於自動化分析和跨語言一致性
    """
    
    def __init__(
        self, 
        service_name: str,
        module_name: str,
        include_trace: bool = True,
        include_metrics: bool = False
    ):
        """
        初始化格式器
        
        Args:
            service_name: 服務名稱 (e.g., "aiva-core", "scan-engine")
            module_name: 模組名稱 (e.g., "multilang_coordinator", "bio_neuron")
            include_trace: 是否包含追蹤信息
            include_metrics: 是否包含性能指標
        """
        super().__init__()
        self.service_name = service_name
        self.module_name = module_name
        self.include_trace = include_trace
        self.include_metrics = include_metrics
        
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日誌記錄為統一的 JSON 結構
        
        Args:
            record: 日誌記錄
            
        Returns:
            格式化的 JSON 字串
        """
        # 基礎日誌結構
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "module": self.module_name,
            "message": record.getMessage(),
            "logger": record.name
        }
        
        # 添加執行上下文
        if hasattr(record, 'task_id'):
            log_entry["task_id"] = record.task_id
            
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
            
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        # 添加追蹤信息
        if self.include_trace:
            log_entry["trace"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
                "thread": record.thread,
                "process": record.process
            }
        
        # 添加異常信息
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # 添加自定義字段
        if hasattr(record, 'extra_fields'):
            log_entry["extra"] = record.extra_fields
        
        # 添加性能指標
        if self.include_metrics and hasattr(record, 'metrics'):
            log_entry["metrics"] = record.metrics
            
        # 添加 AI 相關字段
        ai_fields = {}
        for attr in ['confidence', 'model_version', 'prediction', 'accuracy']:
            if hasattr(record, attr):
                ai_fields[attr] = getattr(record, attr)
                
        if ai_fields:
            log_entry["ai"] = ai_fields
        
        return json.dumps(log_entry, ensure_ascii=False)


class CrossLanguageLogManager:
    """
    跨語言日誌管理器
    確保 Python、Go、Rust、TypeScript 模組使用一致的日誌格式
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._formatters: Dict[str, AIVALogFormatter] = {}
        
    def get_logger(
        self, 
        module_name: str,
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = False,
        log_file_path: Optional[str] = None
    ) -> logging.Logger:
        """
        獲取統一格式的日誌器
        
        Args:
            module_name: 模組名稱
            level: 日誌級別
            enable_console: 啟用控制台輸出
            enable_file: 啟用文件輸出
            log_file_path: 日誌文件路徑
            
        Returns:
            配置好的日誌器
        """
        logger_name = f"{self.service_name}.{module_name}"
        logger = logging.getLogger(logger_name)
        
        # 避免重複配置
        if logger.handlers:
            return logger
            
        logger.setLevel(getattr(logging, level.value))
        
        # 創建格式器
        if module_name not in self._formatters:
            self._formatters[module_name] = AIVALogFormatter(
                service_name=self.service_name,
                module_name=module_name,
                include_trace=True,
                include_metrics=True
            )
        
        formatter = self._formatters[module_name]
        
        # 控制台處理器
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件處理器
        if enable_file and log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def log_with_context(
        self,
        logger: logging.Logger,
        level: LogLevel,
        message: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        帶上下文的日誌記錄
        
        Args:
            logger: 日誌器
            level: 日誌級別
            message: 日誌消息
            task_id: 任務 ID
            session_id: 會話 ID
            metrics: 性能指標
            **kwargs: 額外字段
        """
        extra = {}
        
        if task_id:
            extra['task_id'] = task_id
        if session_id:
            extra['session_id'] = session_id
        if metrics:
            extra['metrics'] = metrics
        if kwargs:
            extra['extra_fields'] = kwargs
            
        log_method = getattr(logger, level.value.lower())
        log_method(message, extra=extra)


def create_unified_logger(service_name: str, module_name: str) -> logging.Logger:
    """
    創建統一格式的日誌器（便捷函數）
    
    Args:
        service_name: 服務名稱
        module_name: 模組名稱
        
    Returns:
        配置好的日誌器
    """
    log_manager = CrossLanguageLogManager(service_name)
    return log_manager.get_logger(module_name)


def log_ai_decision(
    logger: logging.Logger,
    decision: str,
    confidence: float,
    model_version: str,
    task_id: Optional[str] = None,
    **context
) -> None:
    """
    記錄 AI 決策日誌（專用函數）
    
    Args:
        logger: 日誌器
        decision: 決策內容
        confidence: 信心度
        model_version: 模型版本
        task_id: 任務 ID
        **context: 其他上下文
    """
    extra = {
        'confidence': confidence,
        'model_version': model_version,
        'prediction': decision
    }
    
    if task_id:
        extra['task_id'] = task_id
    if context:
        extra['extra_fields'] = context
        
    logger.info(f"AI Decision: {decision}", extra=extra)


def log_cross_language_call(
    logger: logging.Logger,
    source_lang: str,
    target_lang: str,
    function_name: str,
    parameters: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    execution_time: Optional[float] = None
) -> None:
    """
    記錄跨語言調用日誌
    
    Args:
        logger: 日誌器
        source_lang: 來源語言
        target_lang: 目標語言
        function_name: 函數名稱
        parameters: 調用參數
        result: 調用結果
        error: 錯誤信息
        execution_time: 執行時間
    """
    extra = {
        'extra_fields': {
            'cross_language_call': {
                'source': source_lang,
                'target': target_lang,
                'function': function_name,
                'parameters_hash': hash(str(parameters)),
                'has_result': result is not None,
                'has_error': error is not None,
                'execution_time_ms': execution_time * 1000 if execution_time else None
            }
        }
    }
    
    if result:
        message = f"Cross-language call {source_lang}->{target_lang}::{function_name} completed successfully"
        logger.info(message, extra=extra)
    elif error:
        message = f"Cross-language call {source_lang}->{target_lang}::{function_name} failed: {error}"
        logger.error(message, extra=extra)
    else:
        message = f"Cross-language call {source_lang}->{target_lang}::{function_name} initiated"
        logger.debug(message, extra=extra)


# 預設日誌管理器實例
_default_log_manager = CrossLanguageLogManager("aiva-core")

def get_aiva_logger(module_name: str) -> logging.Logger:
    """獲取 AIVA 核心日誌器（便捷函數）"""
    return _default_log_manager.get_logger(module_name)