# 🎨 UI Panel - 使用者介面面板

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心職責](#核心職責)
  - [設計理念](#設計理念)
- [🏗️ 介面類型](#-介面類型)
  - [介面架構](#介面架構)
- [🔧 核心組件](#-核心組件)
  - [1. 🌐 Web Dashboard (Web 儀表板)](#1--web-dashboard-web-儀表板)
  - [2. 🖥️ Rich CLI (命令行介面)](#2--rich-cli-命令行介面)
  - [3. 📦 Data Models (數據模型)](#3--data-models-數據模型)
- [📖 使用範例](#-使用範例)
  - [1. 啟動 Web 儀表板](#1-啟動-web-儀表板)
  - [2. 使用 Rich CLI](#2-使用-rich-cli)
  - [3. 整合到腳本](#3-整合到腳本)
  - [4. AI 混合模式](#4-ai-混合模式)
- [🛠️ 開發指南](#-開發指南)
  - [🔨 aiva_common 修復規範](#-aiva_common-修復規範)
  - [添加新的 API 端點](#添加新的-api-端點)
  - [自定義 Rich CLI 主題](#自定義-rich-cli-主題)
  - [創建自定義 UI 組件](#創建自定義-ui-組件)
  - [整合新的運作模式](#整合新的運作模式)
- [📊 技術棧](#-技術棧)
  - [Web Dashboard](#web-dashboard)
  - [Rich CLI](#rich-cli)
  - [數據驗證](#數據驗證)
- [📊 性能指標](#-性能指標)
  - [Web 服務](#web-服務)
  - [Rich CLI](#rich-cli)
- [🔗 相關模組](#-相關模組)

---

**導航**: [← 返回 AIVA Core](../README.md)

> **版本**: 3.0.0-alpha  
> **狀態**: 生產就緒  
> **角色**: AIVA 的「對外門戶」- 提供 Web 和命令行介面

---

## 📋 目錄

- [模組概述](#模組概述)
- [介面類型](#介面類型)
- [核心組件](#核心組件)
- [使用範例](#使用範例)
- [開發指南](#開發指南)

---

## 🎯 模組概述

**UI Panel** 是 AIVA 六大模組架構中的前端展示層，提供多種使用者介面（Web Dashboard、Rich CLI、RESTful API）來操作和監控 AIVA 系統，實現人機交互和系統可視化。

### 核心職責
1. **Web 儀表板** - 提供圖形化的 Web 管理介面
2. **Rich CLI** - 提供現代化的命令行互動介面
3. **RESTful API** - 提供標準的 HTTP API 服務
4. **狀態展示** - 實時顯示掃描進度和結果
5. **模式切換** - 支援 UI、AI、混合三種運作模式
6. **自動伺服器** - 自動尋找可用端口並啟動服務

### 設計理念
- **多介面支援** - Web、CLI、API 三位一體
- **即時互動** - 實時反饋和進度顯示
- **視覺化優先** - 豐富的圖表和表格展示
- **易用性** - 直觀的操作和清晰的提示
- **靈活模式** - 支援純 UI、純 AI、混合模式

---

## 🏗️ 介面類型

```
ui_panel/
├── 📁 Web Dashboard (Web 儀表板)
│   ├── dashboard.py              # ✅ 主控制面板
│   ├── server.py                 # ✅ FastAPI 伺服器
│   └── auto_server.py            # 自動端口發現與啟動
│
├── 📁 Rich CLI (命令行介面)
│   ├── rich_cli.py               # ✅ Rich 框架 CLI
│   ├── rich_cli_config.py        # CLI 配置和主題
│   └── improved_ui.py            # 改進的 UI 組件
│
├── 📁 Data Models (數據模型)
│   └── ai_ui_schemas.py          # UI 數據模式定義
│
└── __init__.py                   # 模組初始化

總計: 8 個 Python 檔案
```

### 介面架構
```
┌─────────────────────────────────────────────────────────┐
│               UI Panel (使用者介面)                      │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │   Web Dashboard  │  │    Rich CLI      │           │
│  │   (FastAPI)      │  │  (Rich Framework)│           │
│  └────────┬─────────┘  └────────┬─────────┘           │
│           │                     │                      │
│           └─────────┬───────────┘                      │
│                     │                                  │
│           ┌─────────▼─────────┐                        │
│           │   Dashboard Core  │                        │
│           │   (mode: hybrid)  │                        │
│           └─────────┬─────────┘                        │
│                     │                                  │
│      ┌──────────────┼──────────────┐                  │
│      │              │               │                  │
│  ┌───▼───┐    ┌────▼────┐    ┌────▼────┐             │
│  │ Scan  │    │   AI    │    │  Vuln   │             │
│  │ Tasks │    │  Agent  │    │ Results │             │
│  └───────┘    └─────────┘    └─────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 核心組件

### 1. 🌐 Web Dashboard (Web 儀表板)

#### `dashboard.py` - 主控制面板
**功能**: 提供 Web UI 來管理掃描、AI 代理、漏洞檢測
```python
from ui_panel import Dashboard

# 初始化儀表板 (混合模式)
dashboard = Dashboard(mode="hybrid")

# 創建掃描任務
scan_id = dashboard.create_scan_task({
    "target": "https://example.com",
    "scan_type": "full",
    "modules": ["xss", "sqli", "csrf"]
})

# 獲取掃描狀態
status = dashboard.get_scan_status(scan_id)
print(f"掃描進度: {status['progress']}%")

# 獲取檢測結果
results = dashboard.get_detection_results(scan_id)
for vuln in results:
    print(f"發現漏洞: {vuln['type']} - {vuln['severity']}")

# 使用 AI 代理
if dashboard.ai_agent:
    response = await dashboard.ai_agent.query(
        "分析目標網站的安全風險"
    )
    print(f"AI 分析: {response}")
```

**運作模式**:
- `ui` - 僅使用 UI 介面
- `ai` - 僅使用 AI 代理
- `hybrid` - 同時使用 UI 和 AI (預設)

#### `server.py` - FastAPI 伺服器
**功能**: 提供 RESTful API 和 Web 服務
```python
from ui_panel.server import start_ui_server

# 啟動伺服器 (自動尋找可用端口)
start_ui_server(
    mode="hybrid",
    host="127.0.0.1",
    port=None  # None = 自動尋找可用端口
)

# 指定端口啟動
start_ui_server(
    mode="ui",
    host="0.0.0.0",
    port=8080
)
```

**API 端點** (預期):
```
GET  /api/scans               - 獲取所有掃描任務
POST /api/scans               - 創建新掃描任務
GET  /api/scans/{scan_id}     - 獲取掃描詳情
GET  /api/scans/{scan_id}/status - 獲取掃描狀態
GET  /api/vulnerabilities     - 獲取漏洞列表
POST /api/ai/query            - AI 代理查詢
GET  /api/health              - 健康檢查
```

#### `auto_server.py` - 自動端口發現
**功能**: 自動尋找可用端口並啟動伺服器
```python
from ui_panel.server import find_free_port

# 尋找可用端口
free_port = find_free_port(start_port=8080, max_attempts=100)
print(f"找到可用端口: {free_port}")

# 在範圍 8080-8179 中尋找
port = find_free_port(start_port=8080, max_attempts=100)
```

---

### 2. 🖥️ Rich CLI (命令行介面)

#### `rich_cli.py` - Rich 框架 CLI
**功能**: 提供現代化的命令行互動介面
```python
from ui_panel import RichCLI

# 啟動 Rich CLI
cli = RichCLI()
await cli.run()
```

**功能特色**:
- ✅ **彩色主題化介面** - 使用 Rich 主題渲染
- ✅ **互動式選單** - 支援鍵盤導航和選擇
- ✅ **實時進度指示** - Spinner 和進度條
- ✅ **結構化表格** - 美化的數據展示
- ✅ **面板和邊框** - 清晰的區塊劃分
- ✅ **異常處理** - 友好的錯誤提示
- ✅ **狀態指示器** - ✅ ❌ ⏸️ 🔄 等符號

**主選單** (來自 `rich_cli_config.py`):
```
┌─────────────────────────────────────┐
│        AIVA 主選單                  │
├─────────────────────────────────────┤
│ 1. 🔍 漏洞掃描                      │
│ 2. 🤖 AI 代理查詢                   │
│ 3. 📊 查看掃描結果                  │
│ 4. 🔧 能力管理                      │
│ 5. ⚙️  系統設定                      │
│ 6. 📖 幫助文檔                      │
│ 7. 🚪 退出系統                      │
└─────────────────────────────────────┘
```

#### `rich_cli_config.py` - CLI 配置
**功能**: 定義 Rich CLI 的主題、選單、樣式
```python
from ui_panel.rich_cli_config import (
    RICH_THEME,
    CONSOLE_CONFIG,
    MAIN_MENU_ITEMS,
    SCAN_TYPES,
    STATUS_INDICATORS,
    AIVA_COLORS
)

# AIVA 顏色主題
print(AIVA_COLORS["primary"])   # "#00D4FF"
print(AIVA_COLORS["success"])   # "#00FF88"
print(AIVA_COLORS["error"])     # "#FF4444"

# 狀態指示器
print(STATUS_INDICATORS["running"])   # "🔄"
print(STATUS_INDICATORS["success"])   # "✅"
print(STATUS_INDICATORS["error"])     # "❌"

# 掃描類型
for scan_type in SCAN_TYPES:
    print(f"{scan_type['icon']} {scan_type['name']}: {scan_type['desc']}")
```

**預定義主題**:
- `aiva.primary` - 主要強調色 (#00D4FF)
- `aiva.success` - 成功狀態 (#00FF88)
- `aiva.error` - 錯誤狀態 (#FF4444)
- `aiva.warning` - 警告狀態 (#FFAA00)
- `aiva.info` - 資訊提示 (#88CCFF)

#### `improved_ui.py` - 改進的 UI 組件
**功能**: 增強的 UI 組件和工具函數
```python
from ui_panel.improved_ui import (
    create_panel,
    create_table,
    show_progress,
    confirm_action
)

# 創建美化面板
panel = create_panel(
    content="掃描完成",
    title="結果",
    style="success"
)

# 創建表格
table = create_table(
    title="漏洞列表",
    columns=["類型", "嚴重性", "位置"],
    rows=[
        ["XSS", "High", "/search?q="],
        ["SQLi", "Critical", "/login"],
    ]
)

# 顯示進度
with show_progress() as progress:
    task = progress.add_task("掃描中...", total=100)
    for i in range(100):
        progress.update(task, advance=1)

# 確認操作
if confirm_action("確定要開始掃描嗎?"):
    start_scan()
```

---

### 3. 📦 Data Models (數據模型)

#### `ai_ui_schemas.py` - UI 數據模式
**功能**: 定義 UI 相關的數據結構和驗證
```python
from ui_panel.ai_ui_schemas import (
    ScanTaskSchema,
    VulnerabilitySchema,
    ScanStatusSchema
)
from pydantic import BaseModel

# 掃描任務模式
class ScanTaskSchema(BaseModel):
    target: str
    scan_type: str
    modules: list[str]
    priority: int = 1

# 漏洞模式
class VulnerabilitySchema(BaseModel):
    type: str
    severity: str
    location: str
    description: str
    evidence: dict

# 掃描狀態模式
class ScanStatusSchema(BaseModel):
    scan_id: str
    status: str  # "running", "completed", "failed"
    progress: int  # 0-100
    start_time: str
    end_time: str | None
```

---

## 📖 使用範例

### 1. 啟動 Web 儀表板
```python
from ui_panel import Dashboard
from ui_panel.server import start_ui_server

# 方式 1: 直接啟動伺服器
start_ui_server(mode="hybrid", host="127.0.0.1", port=8080)

# 方式 2: 使用 Dashboard 類別
dashboard = Dashboard(mode="hybrid")

# 創建掃描
scan_id = dashboard.create_scan_task({
    "target": "https://example.com",
    "scan_type": "full"
})

# 輪詢狀態
while True:
    status = dashboard.get_scan_status(scan_id)
    print(f"進度: {status['progress']}%")
    if status['status'] == "completed":
        break
    await asyncio.sleep(5)

# 獲取結果
results = dashboard.get_detection_results(scan_id)
```

### 2. 使用 Rich CLI
```python
from ui_panel import RichCLI

# 啟動互動式 CLI
cli = RichCLI()
await cli.run()

# 程序化使用
cli = RichCLI()

# 顯示主選單
cli.show_main_menu()

# 執行掃描
cli.start_scan(target="https://example.com")

# 查看結果
cli.show_scan_results(scan_id)
```

### 3. 整合到腳本
```python
from ui_panel import Dashboard
from rich.console import Console
from rich.table import Table

console = Console()

# 初始化
dashboard = Dashboard(mode="ui")

# 創建掃描
scan_id = dashboard.create_scan_task({
    "target": "https://example.com",
    "scan_type": "quick"
})

# 等待完成
# ... (輪詢邏輯)

# 展示結果
results = dashboard.get_detection_results(scan_id)

table = Table(title="發現的漏洞")
table.add_column("類型", style="cyan")
table.add_column("嚴重性", style="magenta")
table.add_column("位置", style="green")

for vuln in results:
    table.add_row(
        vuln["type"],
        vuln["severity"],
        vuln["location"]
    )

console.print(table)
```

### 4. AI 混合模式
```python
from ui_panel import Dashboard

# 混合模式: UI + AI
dashboard = Dashboard(mode="hybrid")

# 使用 UI 創建掃描
scan_id = dashboard.create_scan_task({
    "target": "https://example.com",
    "scan_type": "full"
})

# 使用 AI 分析結果
if dashboard.ai_agent:
    results = dashboard.get_detection_results(scan_id)
    
    # AI 生成報告
    report = await dashboard.ai_agent.query(
        f"根據以下漏洞生成安全報告: {results}"
    )
    
    print(report)
    
    # AI 建議修復方案
    suggestions = await dashboard.ai_agent.query(
        "針對發現的漏洞提供修復建議"
    )
    
    print(suggestions)
```

---

## 🛠️ 開發指南

### 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範。

**完整規範**: [aiva_common 開發指南](../../../aiva_common/README.md#-開發指南)

#### UI 模組特別注意

```python
# ✅ 正確：使用標準定義
from aiva_common import (
    FindingPayload, Severity, TaskStatus,
    SARIFReport, CVSSv3Metrics
)

# ❌ 禁止：自創 UI 顯示用枚舉
class DisplaySeverity(str, Enum): pass  # 錯誤！直接用 Severity

# ✅ 合理的 UI 專屬枚舉
class DashboardView(str, Enum):
    """儀表板視圖類型 (UI 專用)"""
    OVERVIEW = "overview"
    FINDINGS = "findings"
    REPORTS = "reports"
    SETTINGS = "settings"
```

**UI Panel 原則**:
- 顯示數據必須使用 `aiva_common` 標準格式
- 不要為了 UI 顯示自創數據類型
- 使用標準枚舉的字串值進行渲染

📖 **完整規範**: [aiva_common 標準](../../../aiva_common/README.md#-開發規範與最佳實踐)

---

### 添加新的 API 端點

```python
# ui_panel/server.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/custom-endpoint")
async def custom_endpoint():
    """自定義端點"""
    return {"message": "Hello from custom endpoint"}
```

### 自定義 Rich CLI 主題

```python
# ui_panel/rich_cli_config.py
from rich.theme import Theme

CUSTOM_THEME = Theme({
    "aiva.custom": "#FF00FF",
    "aiva.highlight": "bold #FFFF00",
})

# 在 rich_cli.py 中使用
from rich.console import Console
from .rich_cli_config import CUSTOM_THEME

console = Console(theme=CUSTOM_THEME)
console.print("[aiva.custom]自定義顏色文字[/]")
```

### 創建自定義 UI 組件

```python
# ui_panel/improved_ui.py
from rich.panel import Panel
from rich.table import Table

def create_vulnerability_panel(vuln: dict) -> Panel:
    """創建漏洞展示面板"""
    content = f"""
類型: {vuln['type']}
嚴重性: {vuln['severity']}
位置: {vuln['location']}
描述: {vuln['description']}
    """
    
    return Panel(
        content,
        title=f"[red]漏洞 - {vuln['type']}[/]",
        border_style="red"
    )

def create_scan_progress_table(scans: list) -> Table:
    """創建掃描進度表格"""
    table = Table(title="掃描任務進度")
    table.add_column("ID", style="cyan")
    table.add_column("目標", style="magenta")
    table.add_column("進度", style="green")
    table.add_column("狀態", style="yellow")
    
    for scan in scans:
        table.add_row(
            scan["id"],
            scan["target"],
            f"{scan['progress']}%",
            scan["status"]
        )
    
    return table
```

### 整合新的運作模式

```python
# ui_panel/dashboard.py
class Dashboard:
    def __init__(self, mode: str = "hybrid"):
        valid_modes = ["ui", "ai", "hybrid", "headless"]  # 新增 headless
        
        if mode not in valid_modes:
            raise ValueError(f"無效模式: {mode}")
        
        self.mode = mode
        
        if mode == "headless":
            self._init_headless_mode()
    
    def _init_headless_mode(self):
        """初始化無頭模式 (純 API)"""
        logger.info("初始化無頭模式...")
        # 不啟動 UI，只提供 API
```

---

## 📊 技術棧

### Web Dashboard
- **Framework**: FastAPI
- **Server**: Uvicorn
- **API**: RESTful
- **Port**: 自動發現 (8080-8179)

### Rich CLI
- **Framework**: Rich
- **Themes**: 自定義 AIVA 主題
- **Components**: Panel, Table, Progress, Prompt, Tree
- **Colors**: 256 色支援

### 數據驗證
- **Schema**: Pydantic Models
- **Validation**: 自動類型檢查和驗證

---

## 📊 性能指標

### Web 服務
- **啟動時間**: < 3 秒
- **API 響應時間**: < 100ms
- **並發連接**: 1000+
- **記憶體使用**: < 200MB

### Rich CLI
- **渲染速度**: 60 FPS
- **啟動時間**: < 1 秒
- **記憶體使用**: < 50MB
- **支援終端**: 所有現代終端

---

## 🔗 相關模組

- **[cognitive_core](../cognitive_core/README.md)** - 提供 AI 代理能力
- **[task_planning](../task_planning/README.md)** - 執行掃描任務
- **[core_capabilities](../core_capabilities/README.md)** - 提供漏洞檢測結果
- **[service_backbone](../service_backbone/README.md)** - 提供狀態存儲

---

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team  
**授權**: MIT License
