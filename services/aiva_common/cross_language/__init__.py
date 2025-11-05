"""
AIVA Cross-Language Communication Module
AIVA 跨語言通訊模組

此模組提供完整的跨語言架構支持：

核心組件：
- 統一數據合約通信服務
- 統一錯誤處理系統
- 語言適配器基類

適配器：
- PythonAdapter: Python 語言適配器（內建）
- RustAdapter: Rust 語言適配器（高性能）
- GoAdapter: Go 語言適配器（微服務）

協議支持：
- 統一數據合約: JSON 標準格式
- RabbitMQ: 服務間通訊
- 統一錯誤碼: 跨語言錯誤映射

使用範例：
```python
from aiva_common.schemas.messaging import AivaMessage, AIVARequest
from aiva_common.cross_language import create_rust_adapter

# 創建統一消息
request = AIVARequest(
    request_id="req_123",
    source_module="python_scanner",
    target_module="rust_engine",
    request_type="security_scan",
    payload={"url": "http://example.com"}
)

# 使用 Rust 適配器處理
async with create_rust_adapter() as rust_adapter:
    result = await rust_adapter.process_request(request)
    print("Scan result:", result.payload)
```
"""

from .adapters import (
    GoAdapter,
    GoConfig,
    RustAdapter,
    RustConfig,
    create_go_adapter,
    create_rust_adapter,
)
from .core import (
    ConnectionPool,
    CrossLanguageConfig,
    CrossLanguageService,
    LanguageAdapter,
    MessageRegistry,
    PythonAdapter,
    cross_language_method,
    get_cross_language_service,
    init_cross_language_service,
)
from .errors import (
    AIVAError,
    AIVAErrorCode,
    AIVAException,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    LanguageErrorMapper,
    create_error_context,
    error_handler,
    get_error_handler,
    handle_error,
)

__all__ = [
    # 核心組件
    "CrossLanguageService",
    "CrossLanguageConfig",
    "LanguageAdapter",
    "PythonAdapter",
    "ConnectionPool",
    "MessageRegistry",
    "get_cross_language_service",
    "init_cross_language_service",
    "cross_language_method",
    # 錯誤處理
    "ErrorHandler",
    "AIVAError",
    "AIVAErrorCode",
    "ErrorSeverity",
    "ErrorContext",
    "LanguageErrorMapper",
    "AIVAException",
    "get_error_handler",
    "create_error_context",
    "handle_error",
    "error_handler",
    # 適配器
    "RustAdapter",
    "RustConfig",
    "create_rust_adapter",
    "GoAdapter",
    "GoConfig",
    "create_go_adapter",
]

# 版本信息
__version__ = "1.0.0"
__author__ = "AIVA Development Team"
__description__ = "AIVA Cross-Language Communication Framework"
