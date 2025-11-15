"""Server - AIVA UI 面板 Web 伺服器
使用 FastAPI 提供 RESTful API 和 Web 介面
"""

import logging
import socket
from typing import TYPE_CHECKING, Any

from services.aiva_common.error_handling import (
    AIVAError,
    ErrorSeverity,
    ErrorType,
    create_error_context,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
MODULE_NAME = "aiva_core.ui_panel.server"


def find_free_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """尋找可用的端口號.

    Args:
        start_port: 起始端口號
        max_attempts: 最大嘗試次數

    Returns:
        可用的端口號

    Raises:
        RuntimeError: 找不到可用的端口
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    msg = f"無法在 {start_port}-{start_port + max_attempts - 1} 範圍內找到可用端口"
    raise AIVAError(
        msg,
        error_type=ErrorType.SYSTEM,
        severity=ErrorSeverity.HIGH,
        context=create_error_context(
            module=MODULE_NAME,
            function="find_free_port",
            start_port=start_port,
            max_attempts=max_attempts
        )
    )


def start_ui_server(
    mode: str = "hybrid",
    host: str = "127.0.0.1",
    port: int | None = None,
) -> None:
    """啟動 UI 面板伺服器.

    Args:
        mode: 運作模式 (ui/ai/hybrid)
        host: 綁定的主機位址
        port: 指定的端口號 (None 表示自動選擇)
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError:
        logger.error("錯誤: 需要安裝 FastAPI")
        logger.error("請執行: pip install fastapi uvicorn")
        return

    from .dashboard import Dashboard

    # 建立 FastAPI 應用
    app = FastAPI(title="AIVA Control Panel", version="1.0.0")

    # 初始化控制面板
    dashboard = Dashboard(mode=mode)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """首頁."""
        stats = dashboard.get_stats()
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIVA Control Panel</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                h1 {{
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stat-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .stat-card .value {{
                    font-size: 32px;
                    font-weight: bold;
                }}
                .mode-badge {{
                    display: inline-block;
                    padding: 5px 15px;
                    background: #4CAF50;
                    color: white;
                    border-radius: 20px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f5f5f5;
                    border-radius: 8px;
                }}
                .api-link {{
                    display: inline-block;
                    margin: 5px 10px 5px 0;
                    padding: 10px 20px;
                    background: #667eea;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    transition: background 0.3s;
                }}
                .api-link:hover {{
                    background: #764ba2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>[啟動] AIVA Control Panel</h1>
                <p>運作模式: <span class="mode-badge">{stats['mode_display']}</span></p>

                <div class="stats">
                    <div class="stat-card">
                        <h3>掃描任務</h3>
                        <div class="value">{stats['total_tasks']}</div>
                    </div>
                    <div class="stat-card">
                        <h3>漏洞檢測</h3>
                        <div class="value">{stats['total_detections']}</div>
                    </div>
                    <div class="stat-card">
                        <h3>AI 狀態</h3>
                        <div class="value">{'[V]' if stats['ai_enabled'] else '[X]'}</div>
                    </div>
                    {f'''
                    <div class="stat-card">
                        <h3>AI 知識庫</h3>
                        <div class="value">{stats.get('ai_chunks', 0)}</div>
                    </div>
                    ''' if stats['ai_enabled'] else ''}
                </div>

                <div class="section">
                    <h2>[API] API 端點</h2>
                    <a href="/api/stats" class="api-link">統計資訊</a>
                    <a href="/api/tasks" class="api-link">掃描任務</a>
                    <a href="/api/detections" class="api-link">檢測結果</a>
                    <a href="/docs" class="api-link">API 文檔</a>
                </div>

                <div class="section">
                    <h2>[說明] 使用說明</h2>
                    <p><strong>當前模式:</strong> {stats['mode_display']}</p>
                    <ul>
                        <li><strong>UI 模式:</strong> 使用傳統的 REST API 操作</li>
                        <li><strong>AI 模式:</strong> 所有操作都透過 AI 代理執行</li>
                        <li><strong>混合模式:</strong> 可以選擇使用 UI 或 AI</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

    @app.get("/api/stats")
    async def get_stats() -> dict[str, Any]:
        """獲取統計資訊."""
        return dashboard.get_stats()

    @app.get("/api/tasks")
    async def get_tasks() -> list[dict[str, Any]]:
        """獲取所有掃描任務."""
        return dashboard.get_tasks()

    @app.get("/api/detections")
    async def get_detections() -> list[dict[str, Any]]:
        """獲取所有檢測結果."""
        return dashboard.get_detections()

    @app.post("/api/scan")
    async def create_scan(
        target_url: str, scan_type: str = "full", use_ai: bool | None = None
    ) -> dict[str, Any]:
        """建立掃描任務."""
        return dashboard.create_scan_task(target_url, scan_type, use_ai)

    @app.post("/api/detect")
    async def detect_vuln(
        vuln_type: str, target: str, use_ai: bool | None = None
    ) -> dict[str, Any]:
        """執行漏洞檢測."""
        return dashboard.detect_vulnerability(vuln_type, target, use_ai)

    @app.get("/api/code/read")
    async def read_code(path: str, use_ai: bool | None = None) -> dict[str, Any]:
        """讀取程式碼."""
        return dashboard.read_code(path, use_ai)

    @app.get("/api/code/analyze")
    async def analyze_code(path: str, use_ai: bool | None = None) -> dict[str, Any]:
        """分析程式碼."""
        return dashboard.analyze_code(path, use_ai)

    @app.get("/api/ai/history")
    async def get_ai_history() -> list[dict[str, Any]]:
        """獲取 AI 執行歷史."""
        return dashboard.get_ai_history()

    # 自動選擇可用端口
    if port is None:
        try:
            port = find_free_port()
            logger.info(f"自動選擇可用端口: {port}")
        except RuntimeError as e:
            logger.error(f"端口選擇失敗: {e}")
            return

    # 啟動伺服器
    logger.info(f"\n{'='*60}")
    logger.info("   啟動 AIVA UI 面板伺服器")
    logger.info(f"{'='*60}")
    logger.info(f"位址: http://{host}:{port}")
    logger.info(f"API 文檔: http://{host}:{port}/docs")
    logger.info(f"{'='*60}\n")

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        logger.error("錯誤: 需要安裝 uvicorn")
        logger.error("請執行: pip install uvicorn")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"端口 {port} 已被佔用，嘗試自動選擇其他端口...")
            try:
                new_port = find_free_port(port + 1)
                logger.info(f"使用新端口: {new_port}")
                uvicorn.run(app, host=host, port=new_port, log_level="info")
            except (RuntimeError, ImportError) as retry_error:
                logger.error(f"重試失敗: {retry_error}")
        else:
            logger.error(f"伺服器啟動失敗: {e}")
