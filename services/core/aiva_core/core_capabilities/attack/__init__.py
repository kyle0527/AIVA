"""
AIVA Attack Module
攻擊執行模組

此模組負責實際的安全測試攻擊執行，包括:
- 攻擊計劃執行
- 漏洞利用管理
- Payload 生成
- 攻擊鏈編排
- 結果驗證

五大模組架構中的核心攻擊模組
"""

from .attack_chain import AttackChain
from .attack_executor import AttackExecutor
from .attack_validator import AttackValidator
from .exploit_orchestrator import ExploitOrchestrator, ExploitManager
from .payload_generator import PayloadGenerator

__all__ = [
    "AttackExecutor",
    "ExploitManager",
    "ExploitOrchestrator",
    "PayloadGenerator",
    "AttackChain",
    "AttackValidator",
]

__version__ = "1.0.0"
__module_name__ = "aiva_attack"
