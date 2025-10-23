"""
SQL Injection Detection Module

SQL 注入檢測功能模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .smart_detection_manager import SmartDetectionManager
    from .task_queue import TaskQueue
    from .detection_models import DetectionModels
    from .exceptions import SQLiException
    from .payload_wrapper_encoder import PayloadWrapperEncoder
    from .result_binder_publisher import ResultBinderPublisher
    from .telemetry import Telemetry
    
    __all__ = [
        "SmartDetectionManager",
        "TaskQueue", 
        "DetectionModels",
        "SQLiException",
        "PayloadWrapperEncoder",
        "ResultBinderPublisher",
        "Telemetry"
    ]
except ImportError as e:
    # 如果某些模組不存在，只導入可用的
    try:
        from .smart_detection_manager import SmartDetectionManager
        __all__ = ["SmartDetectionManager"]
    except ImportError:
        __all__ = []
