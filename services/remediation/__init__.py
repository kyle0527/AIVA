"""
Remediation Module - 自動修復模組

提供自動修復、補丁生成、代碼修正、配置建議等功能。
"""

from .code_fixer import CodeFixer
from .config_recommender import ConfigRecommender
from .patch_generator import PatchGenerator
from .report_generator import ReportGenerator

__all__ = [
    "PatchGenerator",
    "CodeFixer",
    "ConfigRecommender",
    "ReportGenerator",
]

__version__ = "1.0.0"
