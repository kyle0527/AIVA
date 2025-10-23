"""
AIVA Configuration Template Module

配置範本管理和生成模組。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .config_template_manager import ConfigTemplateManager
    __all__ = ["ConfigTemplateManager"]
except ImportError:
    __all__ = []
