"""
AIVA Attack Module
攻擊編排模組

此模組負責攻擊流程的編排和協調，包括:
- 攻擊鏈編排
- 漏洞利用編排
- 攻擊驗證
- 載荷生成

實際文件位置:
- AttackExecutor: services.features.function_exploit.executor.attack_executor
- AttackValidator: services.features.function_exploit.validators.attack_validator  
- PayloadGenerator: services.features.function_exploit.generators.payload_generator
"""

# 本地模組
from .attack_chain import AttackChain
from .exploit_orchestrator import ExploitOrchestrator

# 延遲導入函數 - 避免循環導入
def get_attack_executor():
    """獲取 AttackExecutor"""
    from services.features.function_exploit.executor.attack_executor import AttackExecutor
    return AttackExecutor

def get_attack_validator():
    """獲取 AttackValidator"""
    from services.features.function_exploit.validators.attack_validator import AttackValidator
    return AttackValidator

def get_payload_generator():
    """獲取 PayloadGenerator"""
    from services.features.function_exploit.generators.payload_generator import PayloadGenerator
    return PayloadGenerator

__all__ = [
    "AttackChain",
    "ExploitOrchestrator",
    "get_attack_executor",
    "get_attack_validator",
    "get_payload_generator",
]

__version__ = "2.0.0"
__module_name__ = "aiva_attack"
