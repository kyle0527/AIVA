"""
AIVA Rich Console

提供全域單例 Console 實例
"""

from rich.console import Console
from rich.traceback import install

from .themes import AIVA_THEME

# 全域 Console 實例
_console: Console | None = None

def init_console(width: int = 120, force_terminal: bool = True) -> Console:
    """初始化 Rich Console
    
    Args:
        width: 控制台寬度
        force_terminal: 強制終端模式
        
    Returns:
        Console 實例
    """
    global _console
    
    if _console is None:
        _console = Console(
            width=width,
            theme=AIVA_THEME,
            force_terminal=force_terminal,
            legacy_windows=False,
        )
        
        # 安裝美化異常處理
        install(
            console=_console,
            show_locals=True,
            width=width,
            extra_lines=3,
            theme="monokai",
        )
    
    return _console

def get_console() -> Console:
    """取得 Console 實例
    
    Returns:
        Console 實例（若未初始化則自動初始化）
    """
    global _console
    if _console is None:
        return init_console()
    return _console

# 建立預設實例
console = get_console()

__all__ = ["console", "init_console", "get_console"]
