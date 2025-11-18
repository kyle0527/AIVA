"""
AIVA Core Crawling Engine

高效能網頁爬蟲和內容抓取引擎。
"""

__version__ = "1.0.0"

# 導入核心組件
try:
    from .static_content_parser import StaticContentParser
    __all__ = ["StaticContentParser"]
except ImportError:
    __all__ = []
