"""AIVA Core Capabilities 核心能力模組

提供 AIVA 的核心能力，包括：
- Dialog Assistant: 對話助理
- Attack Execution: 攻擊執行
- Capability Registry: 能力註冊
"""

# Dialog Assistant
from services.core.aiva_core.core_capabilities.dialog.assistant import (
    AIVADialogAssistant,
)

# Attack Execution
from services.core.aiva_core.core_capabilities.attack.attack_executor import (
    AttackExecutor,
)
from services.core.aiva_core.core_capabilities.attack.exploit_manager_legacy import (
    ExploitManager,
)
from services.core.aiva_core.core_capabilities.attack.payload_generator import (
    PayloadGenerator,
)
from services.core.aiva_core.core_capabilities.attack.attack_chain import AttackChain
from services.core.aiva_core.core_capabilities.attack.attack_validator import (
    AttackValidator,
)

# Capability Registry
from services.core.aiva_core.core_capabilities.capability_registry import (
    CapabilityRegistry,
)

__all__ = [
    # Dialog
    "AIVADialogAssistant",
    # Attack
    "AttackExecutor",
    "ExploitManager",
    "PayloadGenerator",
    "AttackChain",
    "AttackValidator",
    # Capability
    "CapabilityRegistry",
]
