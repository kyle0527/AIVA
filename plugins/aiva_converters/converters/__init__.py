"""Converters package - 各類格式轉換器"""

try:
    from .sarif_converter import SARIFConverter
except ImportError as e:
    print(f"Warning: Failed to import SARIFConverter: {e}")
    SARIFConverter = None

try:
    from .task_converter import TaskConverter
except ImportError as e:
    print(f"Warning: Failed to import TaskConverter: {e}")
    TaskConverter = None

try:
    from .docx_to_md_converter import DocxToMarkdownConverter
except ImportError as e:
    print(f"Warning: Failed to import DocxToMarkdownConverter: {e}")
    DocxToMarkdownConverter = None

__all__ = ["SARIFConverter", "TaskConverter", "DocxToMarkdownConverter"]
