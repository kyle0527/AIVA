"""Improved UI for AIVA Control Panel.

This module defines a FastAPI application that provides a polished
dashboard for monitoring scan tasks and vulnerability detections.  The
page preserves the original look and feel of the existing AIVA control
panel while adding searchable tables for tasks and detection results.

The script can be executed directly (e.g. via ``uvicorn improved_ui:app``)
or imported as a module.
"""

import sys
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Add the path to the AIVA project so that ``services`` can be imported.
# When running this script from outside the project root, this ensures
# that ``services.core.aiva_core.ui_panel.dashboard`` is discoverable.
PROJECT_ROOT = "/home/oai/share/AIVA/AIVA-master"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity, create_error_context

MODULE_NAME = "ui_panel.improved_ui"

try:
    from services.core.ui.dashboard import Dashboard  # type: ignore
    from services.aiva_common.schemas import APIResponse  # type: ignore
except Exception as exc:  # pragma: no cover
    raise AIVAError(
        message="Unable to import Dashboard from AIVA project. Ensure the PROJECT_ROOT is correct and the services package is installed.",
        error_type=ErrorType.SYSTEM,
        severity=ErrorSeverity.CRITICAL,
        context=create_error_context(
            module=MODULE_NAME,
            function="<module>",
            project_root=PROJECT_ROOT
        )
    ) from exc


from .utils import build_task_row, build_detection_row, format_result_preview


def _build_task_rows(tasks: list) -> str:
    """構建任務表格行"""
    if not tasks:
        return "<tr><td colspan=6 style='text-align:center; color:#888;'>無任務</td></tr>"
    
    return "".join(build_task_row(task) for task in tasks)


def _build_detection_rows(detections: list) -> str:
    """構建檢測結果表格行"""
    if not detections:
        return "<tr><td colspan=5 style='text-align:center; color:#888;'>無檢測結果</td></tr>"
    
    return "".join(build_detection_row(det) for det in detections)


def _build_history_rows(ai_history: list) -> str:
    """構建AI歷史記錄表格行"""
    if not ai_history:
        return "<tr><td colspan=4 style='text-align:center; color:#888;'>無 AI 歷史紀錄</td></tr>"
    
    rows = ""
    for record in ai_history:
        status = record.get("status", "-")
        tool_used = record.get("tool_used", "-")
        confidence = record.get("confidence", "-")
        result = record.get("result", "-")
        conf_display = f"{confidence:.2%}" if isinstance(confidence, float) else str(confidence)
        result_preview = format_result_preview(result, max_length=50)
        rows += f"<tr><td>{status}</td><td>{tool_used}</td><td>{conf_display}</td><td>{result_preview}</td></tr>"
    return rows


def _build_ai_card(stats: dict[str, Any]) -> str:
    """構建AI卡片HTML"""
    if not stats.get("ai_enabled"):
        return ""
    return f"""
                <div class="stat-card">
                    <h3>AI 知識庫</h3>
                    <div class="value">{stats.get('ai_chunks', 0)}</div>
                </div>
            """


def _build_index_html(dashboard: Dashboard) -> str:
    """Construct the HTML for the index page.

    The page shows overall statistics, a list of scan tasks and a list of
    detection results. Search boxes allow filtering of the task and
    detection tables.
    """
    stats: dict[str, Any] = dashboard.get_stats()
    tasks = dashboard.get_tasks()
    detections = dashboard.get_detections()
    ai_history = dashboard.get_ai_history()

    task_rows = _build_task_rows(tasks)
    detection_rows = _build_detection_rows(detections)
    history_rows = _build_history_rows(ai_history)
    ai_card_html = _build_ai_card(stats)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AIVA Control Panel - Improved UI</title>
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
                margin: 40px 0;
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
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                padding: 8px 12px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background: #667eea;
                color: white;
            }}
            tr:hover {{
                background: #f1f1f1;
            }}
            .search-input {{
                width: 100%;
                padding: 8px 12px;
                margin-bottom: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
        </style>
        <script>
        // Generic table filter function.  Filter terms across specified columns.
        function filterTable(inputId, tableId, colIndices) {{
            var input = document.getElementById(inputId);
            var filter = input.value.toLowerCase();
            var table = document.getElementById(tableId);
            var rows = table.getElementsByTagName('tr');
            for (var i = 1; i < rows.length; i++) {{
                var show = false;
                for (var j = 0; j < colIndices.length; j++) {{
                    var idx = colIndices[j];
                    var cell = rows[i].getElementsByTagName('td')[idx];
                    if (cell && cell.innerHTML.toLowerCase().indexOf(filter) > -1) {{
                        show = true;
                        break;
                    }}
                }}
                rows[i].style.display = show ? '' : 'none';
            }}
        }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>AIVA Control Panel</h1>
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
                {ai_card_html}
            </div>

            <div class="section">
                <h2>🎯 AI 自主攻擊控制台</h2>
                <form id="aiAttackForm" onsubmit="startAIAttack(event)" style="margin-bottom: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end;">
                        <div>
                            <label for="targetUrl" style="display: block; margin-bottom: 5px; font-weight: bold;">
                                目標網址（輸入靶場 URL）
                            </label>
                            <input type="url" id="targetUrl" name="targetUrl" class="search-input" 
                                   placeholder="http://localhost:3000 或 http://localhost:3001" 
                                   required style="margin: 0;">
                            <small style="color: #666;">例如: http://localhost:3000 (juice-shop-live)</small>
                        </div>
                        <button type="submit" style="padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 16px;">
                            🚀 啟動 AI 自主攻擊
                        </button>
                    </div>
                </form>
                <div id="attackStatus" style="display: none; padding: 15px; background: #e7f3ff; border-left: 4px solid #667eea; border-radius: 4px; margin-bottom: 20px;">
                    <strong>攻擊狀態:</strong> <span id="statusText">準備中...</span>
                    <div style="margin-top: 10px;">
                        <strong>目標:</strong> <span id="currentTarget">-</span>
                    </div>
                </div>
                
                <h3 style="margin-top: 30px;">📊 API 端點</h3>
                <a href="/api/stats" class="api-link">統計資訊</a>
                <a href="/api/tasks" class="api-link">掃描任務</a>
                <a href="/api/detections" class="api-link">檢測結果</a>
                <a href="/docs" class="api-link">API 文檔</a>
            </div>
            
            <script>
            async function startAIAttack(event) {{
                event.preventDefault();
                const targetUrl = document.getElementById('targetUrl').value;
                const statusDiv = document.getElementById('attackStatus');
                const statusText = document.getElementById('statusText');
                const currentTarget = document.getElementById('currentTarget');
                
                // 顯示狀態
                statusDiv.style.display = 'block';
                statusText.textContent = 'AI 正在分析目標...';
                currentTarget.textContent = targetUrl;
                
                try {{
                    // 調用 AI 自主攻擊 API
                    const response = await fetch('/api/ai/attack', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            target_url: targetUrl,
                            attack_mode: 'full',
                            use_ai: true
                        }})
                    }});
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {{
                        statusText.textContent = '✅ AI 攻擊已啟動！任務ID: ' + result.task_id;
                        statusDiv.style.background = '#d4edda';
                        statusDiv.style.borderColor = '#28a745';
                        
                        // 5秒後重新載入頁面以顯示新結果
                        setTimeout(() => {{
                            window.location.reload();
                        }}, 3000);
                    }} else {{
                        statusText.textContent = '❌ 攻擊失敗: ' + result.message;
                        statusDiv.style.background = '#f8d7da';
                        statusDiv.style.borderColor = '#dc3545';
                    }}
                }} catch (error) {{
                    statusText.textContent = '❌ 錯誤: ' + error.message;
                    statusDiv.style.background = '#f8d7da';
                    statusDiv.style.borderColor = '#dc3545';
                }}
            }}
            </script>

            <div class="section">
                <h2>掃描任務</h2>
                <input type="text" id="taskFilter" class="search-input" onkeyup="filterTable('taskFilter','tasksTable',[0,1,2,3,4,5])" placeholder="搜尋任務...">
                <table id="tasksTable">
                    <tr>
                        <th>任務ID</th>
                        <th>目標</th>
                        <th>類型</th>
                        <th>狀態</th>
                        <th>來源</th>
                        <th>AI</th>
                    </tr>
                    {task_rows}
                </table>
            </div>

            <div class="section">
                <h2>檢測結果</h2>
                <input type="text" id="detFilter" class="search-input" onkeyup="filterTable('detFilter','detectionsTable',[0,1,2,3,4])" placeholder="搜尋結果...">
                <table id="detectionsTable">
                    <tr>
                        <th>漏洞類型</th>
                        <th>目標</th>
                        <th>狀態</th>
                        <th>方法</th>
                        <th>結果</th>
                    </tr>
                    {detection_rows}
                </table>
            </div>

            <div class="section">
                <h2>AI 歷史</h2>
                <input type="text" id="historyFilter" class="search-input"
                       onkeyup="filterTable('historyFilter','historyTable',[0,1,2,3])" placeholder="搜尋 AI 歷史...">
                <table id="historyTable">
                    <tr>
                        <th>狀態</th>
                        <th>使用工具</th>
                        <th>信心度</th>
                        <th>結果</th>
                    </tr>
                    {history_rows}
                </table>
            </div>

            <div class="section">
                <h2>使用說明</h2>
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
    return html


def create_app(mode: str = "ui") -> FastAPI:
    """Create a FastAPI app with the improved UI.

    Args:
        mode: Operational mode for the Dashboard (ui/ai/hybrid).
            預設使用 ui 模式以避免 AICommander 初始化失敗

    Returns:
        FastAPI instance ready for serving the UI and API endpoints.
    """
    dashboard = Dashboard(mode=mode)
    app = FastAPI(title="AIVA Control Panel (Improved)", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _build_index_html(dashboard)

    @app.get("/api/stats")
    async def api_stats() -> dict[str, Any]:
        return dashboard.get_stats()

    @app.get("/api/tasks")
    async def api_tasks() -> list[dict[str, Any]]:
        return dashboard.get_tasks()

    @app.get("/api/detections")
    async def api_detections() -> list[dict[str, Any]]:
        return dashboard.get_detections()

    @app.post("/api/scan")
    async def api_create_scan(
        target_url: str, scan_type: str = "full", use_ai: bool | None = None
    ) -> dict[str, Any]:
        return dashboard.create_scan_task(target_url, scan_type, use_ai)

    @app.post("/api/detect")
    async def api_detect_vuln(
        vuln_type: str, target: str, use_ai: bool | None = None
    ) -> dict[str, Any]:
        return dashboard.detect_vulnerability(vuln_type, target, use_ai)

    @app.get("/api/ai/history")
    async def api_ai_history() -> list[dict[str, Any]]:
        return dashboard.get_ai_history()

    @app.post("/api/ai/attack")
    async def api_ai_attack(
        target_url: str, attack_mode: str = "full", use_ai: bool = True
    ) -> dict[str, Any]:
        """AI 自主攻擊端點 - 讓 AI 自動調整程序攻擊指定目標"""
        try:
            # 使用 AI 模式創建全面掃描任務
            result = dashboard.create_scan_task(
                target_url=target_url,
                scan_type=attack_mode,
                use_ai=True  # 強制使用 AI
            )
            
            return {
                "status": "success",
                "message": f"AI 已開始自主攻擊目標: {target_url}",
                "task_id": result.get("task_id"),
                "target": target_url,
                "mode": attack_mode,
                "ai_enabled": True
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "target": target_url
            }

    return app


# Create a default app instance for uvicorn ``--reload`` convenience.
app = create_app()
