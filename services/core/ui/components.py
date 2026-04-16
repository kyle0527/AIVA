"""
AIVA UI 組件

提供可重用的 UI 組件函數
"""

from typing import Any

from rich.align import Align
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table

from .rich_console import get_console
from .themes import ICONS, PANEL_STYLES, TABLE_STYLES

# 使用函數獲取 console，確保主題正確套用
console = get_console()

def show_banner(
    title: str = "AIVA",
    subtitle: str = "AI-powered Vulnerability Assessment",
    version: str = "v3.0"
) -> None:
    """顯示 AIVA 品牌橫幅
    
    Args:
        title: 主標題
        subtitle: 副標題
        version: 版本資訊
    """
    banner_text = f"""
[aiva.title]{title}[/aiva.title]
[aiva.subtitle]{subtitle}[/aiva.subtitle]
[aiva.muted]{version}[/aiva.muted]
"""
    
    panel = Panel(
        Align.center(banner_text.strip()),
        **PANEL_STYLES["banner"],
        title=f"[aiva.accent]{ICONS['sparkle']} Welcome {ICONS['sparkle']}[/aiva.accent]",
    )
    
    console.print()
    console.print(panel)
    console.print()

def show_menu(
    title: str,
    options: list[tuple[str, str, str]],
    style: str = "main_menu"
) -> None:
    """顯示選單
    
    Args:
        title: 選單標題
        options: 選項列表 [(key, name, description), ...]
        style: 表格樣式名稱
    """
    table_config = TABLE_STYLES.get(style, TABLE_STYLES["main_menu"])
    
    table = Table(
        title=f"[aiva.title]{title}[/aiva.title]",
        **table_config
    )
    
    table.add_column("選項", justify="center", style="aiva.accent", width=8)
    table.add_column("功能", style="bold", width=20)
    table.add_column("說明", style="aiva.muted")
    
    for key, name, desc in options:
        # 特殊處理退出選項
        if key in ["0", "Q", "q"]:
            table.add_row(
                f"[aiva.error]{key}[/aiva.error]",
                f"[bold aiva.error]{name}[/bold aiva.error]",
                desc
            )
        else:
            table.add_row(key, name, desc)
    
    console.print()
    console.print(table)
    console.print()

def show_table(
    title: str,
    columns: list[tuple[str, dict]],
    rows: list[list[Any]],
    style: str = "data"
) -> None:
    """顯示數據表格
    
    Args:
        title: 表格標題
        columns: 列定義 [(name, config), ...]
        rows: 行數據
        style: 表格樣式名稱
    """
    table_config = TABLE_STYLES.get(style, TABLE_STYLES["data"])
    
    table = Table(
        title=f"[aiva.title]{title}[/aiva.title]",
        **table_config
    )
    
    # 添加列
    for col_name, col_config in columns:
        table.add_column(col_name, **col_config)
    
    # 添加行
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    
    console.print()
    console.print(table)
    console.print()

def show_panel(
    content: str,
    title: str | None = None,
    style: str = "info",
    icon: str | None = None
) -> None:
    """顯示面板
    
    Args:
        content: 面板內容
        title: 面板標題
        style: 面板樣式
        icon: 圖標（可選）
    """
    panel_config = PANEL_STYLES.get(style, PANEL_STYLES["info"])
    
    if icon:
        title = f"{icon} {title}" if title else icon
    
    panel = Panel(
        content,
        title=title,
        **panel_config
    )
    
    console.print()
    console.print(panel)
    console.print()

def show_progress(
    description: str = "處理中...",
    total: int | None = None
) -> Progress:
    """建立進度條
    
    Args:
        description: 進度描述
        total: 總步驟數（None 為不確定進度）
        
    Returns:
        Progress 實例
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[aiva.info]{task.description}[/aiva.info]"),
        BarColumn(complete_style="aiva.success", finished_style="aiva.success"),
        TextColumn("[aiva.accent]{task.percentage:>3.0f}%[/aiva.accent]"),
        TimeRemainingColumn(),
        console=console,
    )
    
    return progress

def confirm_action(
    message: str,
    default: bool = False,
    icon: str = ICONS["warning"]
) -> bool:
    """確認操作
    
    Args:
        message: 確認訊息
        default: 預設值
        icon: 圖標
        
    Returns:
        用戶確認結果
    """
    styled_message = f"[aiva.warning]{icon} {message}[/aiva.warning]"
    return Confirm.ask(styled_message, default=default)

def show_success(message: str) -> None:
    """顯示成功訊息"""
    show_panel(
        f"[aiva.success]{message}[/aiva.success]",
        style="success",
        icon=ICONS["success"]
    )

def show_error(message: str, details: str | None = None) -> None:
    """顯示錯誤訊息"""
    content = f"[aiva.error]{message}[/aiva.error]"
    if details:
        content += f"\n\n[aiva.muted]詳細資訊：{details}[/aiva.muted]"
    
    show_panel(content, style="error", icon=ICONS["error"])

def show_warning(message: str) -> None:
    """顯示警告訊息"""
    show_panel(
        f"[aiva.warning]{message}[/aiva.warning]",
        style="warning",
        icon=ICONS["warning"]
    )

def show_info(message: str) -> None:
    """顯示資訊訊息"""
    show_panel(
        f"[aiva.info]{message}[/aiva.info]",
        style="info",
        icon=ICONS["info"]
    )

def clear_screen() -> None:
    """清除螢幕"""
    console.clear()

def print_divider(char: str = "─", style: str = "aiva.muted") -> None:
    """打印分隔線"""
    console.print(char * console.width, style=style)

__all__ = [
    "show_banner",
    "show_menu",
    "show_table",
    "show_panel",
    "show_progress",
    "confirm_action",
    "show_success",
    "show_error",
    "show_warning",
    "show_info",
    "clear_screen",
    "print_divider",
]
