"""
Wordlist Generator Module Data Models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ==================== Enums ====================

class GenerationStrategy(str, Enum):
    """生成策略"""
    COMBINATION = "combination"
    CUPP_PROFILE = "cupp_profile"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    PERMUTATION = "permutation"
    MARKOV_CHAIN = "markov_chain"


class CharsetType(str, Enum):
    """字符集類型"""
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    DIGITS = "digits"
    SPECIAL = "special"
    ALPHA = "alpha"
    ALPHANUMERIC = "alphanumeric"
    ALL = "all"
    CUSTOM = "custom"


class WordlistFormat(str, Enum):
    """字典格式"""
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    COMPRESSED = "compressed"


# ==================== Data Models ====================

@dataclass
class GeneratorConfig:
    """生成器配置"""
    strategy: GenerationStrategy
    output_file: str
    
    # Combination Settings
    charset: str | None = None
    min_length: int = 8
    max_length: int = 12
    
    # CUPP Settings (Profile-based)
    target_name: str | None = None
    birthdate: str | None = None
    keywords: list[str] = field(default_factory=list)
    
    # Rule Settings
    rules: list[str] = field(default_factory=list)
    base_words: list[str] = field(default_factory=list)
    
    # Advanced Options
    deduplicate: bool = True
    sort: bool = False
    max_words: int | None = None
    
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratorResult:
    """生成結果"""
    success: bool
    output_file: str
    
    # Statistics
    total_words: int = 0
    unique_words: int = 0
    file_size_mb: float = 0.0
    
    # Timing
    generation_time: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    error: str | None = None
    log_file: str = ""


@dataclass
class WordlistStats:
    """字典統計"""
    file_path: str
    
    # Basic Stats
    total_words: int = 0
    unique_words: int = 0
    file_size_mb: float = 0.0
    
    # Length Distribution
    min_length: int = 0
    max_length: int = 0
    avg_length: float = 0.0
    length_distribution: dict[int, int] = field(default_factory=dict)
    
    # Character Stats
    charset_used: list[str] = field(default_factory=list)
    has_lowercase: bool = False
    has_uppercase: bool = False
    has_digits: bool = False
    has_special: bool = False
    
    # Quality Metrics
    complexity_score: float = 0.0
    entropy_score: float = 0.0
