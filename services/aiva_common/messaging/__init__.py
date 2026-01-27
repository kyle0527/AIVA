"""
AIVA Messaging 模組
消息傳遞：MQ 抽象層、重試處理、主題管理
"""

from .compatibility_layer import (
    CompatibilityLayer,
    MessageFormat,
    UnifiedMessageBroker,
)
from .mq import (
    AbstractBroker,
    MQMessage,
    RabbitBroker,
    get_broker,
)
from .retry_handler import (
    RetryHandler,
    RetryPolicy,
)
from .unified_topic_manager import (
    UnifiedTopicManager,
    topic_manager,
)

__all__ = [
    # Compatibility Layer
    "CompatibilityLayer",
    "MessageFormat",
    "UnifiedMessageBroker",
    # MQ
    "AbstractBroker",
    "MQMessage",
    "RabbitBroker",
    "get_broker",
    # Retry Handler
    "RetryHandler",
    "RetryPolicy",
    # Topic Manager
    "UnifiedTopicManager",
    "topic_manager",
]
