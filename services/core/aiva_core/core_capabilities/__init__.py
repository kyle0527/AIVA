"""AIVA Core Capabilities 核心能力模組

提供 AIVA 的核心能力，包括：
- Dialog Assistant: 對話助理
- Attack Orchestration: 攻擊編排 (執行已移至 features 模組)
- Capability Registry: 能力註冊

注意：AttackExecutor 已移至 services.features.function_exploit.executor
"""

# 延遲導入避免循環依賴和缺失文件問題
def __getattr__(name):
    """延遲導入機制"""
    if name == "AIVACommandProcessor":
        from services.core.aiva_core.core_capabilities.dialog.assistant import AIVACommandProcessor
        return AIVACommandProcessor
    elif name == "AttackExecutor":
        # 已移至 features 模組
        from services.features.function_exploit.executor.attack_executor import AttackExecutor
        return AttackExecutor
    elif name == "AttackChain":
        from services.core.aiva_core.core_capabilities.attack.attack_chain import AttackChain
        return AttackChain
    elif name == "ExploitOrchestrator":
        from services.core.aiva_core.core_capabilities.attack.exploit_orchestrator import ExploitOrchestrator
        return ExploitOrchestrator
    elif name == "CapabilityRegistry":
        from services.core.aiva_core.core_capabilities.capability_registry import CapabilityRegistry
        return CapabilityRegistry
    elif name == "TaskContext":
        from services.core.aiva_core.core_capabilities.task_context import TaskContext
        return TaskContext
    raise AttributeError(f"module 'core_capabilities' has no attribute '{name}'")
