"""
Wordlist Generator Module

提供密碼字典生成能力，支援多種生成策略和自訂規則。

風險等級: L1 (資訊收集)
模組版本: 1.0.0
"""

from .handler import WordlistGeneratorCommandHandler
from .manager import WordlistGeneratorManager
from .models import (
    CharsetType,
    # Enums
    GenerationStrategy,
    # Data Models
    GeneratorConfig,
    GeneratorResult,
    WordlistFormat,
    WordlistStats,
)

__all__ = [
    "WordlistGeneratorManager",
    "WordlistGeneratorCommandHandler",
    "GenerationStrategy",
    "CharsetType",
    "WordlistFormat",
    "GeneratorConfig",
    "GeneratorResult",
    "WordlistStats",
]

__version__ = "1.0.0"
__risk_level__ = "L1"
