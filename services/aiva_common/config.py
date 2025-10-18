"""
AIVA 配置管理 - 向後相容模組
建議使用 aiva_common.config.unified_config
"""

# 向後相容導入
from .config.unified_config import Settings, get_legacy_settings as get_settings

__all__ = ['Settings', 'get_settings']
