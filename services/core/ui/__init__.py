"""
UI Panel - AIVA 核心 Web 控制面板

提供图形化界面来管理和控制扫描任务、AI 代理、漏洞检测等功能。

核心组件：
- Dashboard: 主控制面板类
- start_ui_server: 启动标准 UI 服务器
- start_auto_server: 自动端口选择服务器
- AIVARichCLI: Rich 终端交互界面
- WebSocketManager: WebSocket 连接管理
- utils: 通用工具函数

使用示例：
    >>> from services.core.ui import Dashboard, start_ui_server
    >>> 
    >>> # 创建控制面板
    >>> dashboard = Dashboard(mode="hybrid")
    >>> 
    >>> # 启动 UI 服务器
    >>> start_ui_server(mode="hybrid", port=8080)

更多文档请参阅 README.md
"""

from .auto_server import start_auto_server
from .dashboard import Dashboard
from .rich_cli import AIVARichCLI
from .server import start_ui_server
from .websocket_manager import WebSocketManager, get_websocket_manager

__all__ = [
    "Dashboard",
    "start_ui_server",
    "start_auto_server",
    "AIVARichCLI",
    "WebSocketManager",
    "get_websocket_manager",
]

__version__ = "2.0.0"
__author__ = "AIVA Development Team"
