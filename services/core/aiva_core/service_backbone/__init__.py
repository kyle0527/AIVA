"""AIVA Service Backbone 服務骨幹

提供 AIVA 的基礎設施服務，包括：
- Context Management: 上下文管理
- Messaging: 消息代理
- Coordination: 服務協調
"""

# Context Management
from services.core.aiva_core.service_backbone.context_manager import ContextManager

# Messaging
from services.core.aiva_core.service_backbone.messaging.message_broker import (
    MessageBroker,
)

# Coordination
from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
    AIVACoreServiceCoordinator,
)

__all__ = [
    "ContextManager",
    "MessageBroker",
    "AIVACoreServiceCoordinator",
]
