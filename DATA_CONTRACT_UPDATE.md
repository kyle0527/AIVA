# AIVA 數據合約更新報告

**更新日期**: 2025-10-13

**版本**: 1.1.0

**更新範圍**: AI 引擎與 UI 面板模組

---

## 📋 更新概述

### 新增模組

- **AI 引擎模組** (`services/core/aiva_core/ai_engine/`)
  - 生物啟發式神經網路決策引擎
  - RAG 知識庫檢索系統
  - 9 種工具系統 (CodeReader, CodeWriter, CodeAnalyzer, ScanTrigger, XSS/SQLi/SSRF/IDOR Detector, CommandExecutor)

- **UI 面板模組** (`services/core/aiva_core/ui_panel/`)
  - Web 控制面板 (FastAPI)
  - 支援三種運作模式: UI-only, AI-only, Hybrid
  - RESTful API 端點

### 數據合約影響

之前的數據合約主要涵蓋：

- 訊息協議 (MessageHeader, AivaMessage)
- 掃描相關 (ScanStartPayload, ScanCompletedPayload)
- 功能任務 (FunctionTaskPayload)
- 漏洞報告 (FindingPayload)

新增需求：

- AI 代理請求/響應格式
- 工具執行標準格式
- RAG 檢索數據結構
- UI 面板 API 合約
- 控制面板統計資訊

---

## 🆕 新增數據合約

已創建 `services/core/aiva_core/ai_ui_schemas.py`，包含 22 個新數據模型，摘要如下：

### AI 引擎相關（8 個）

1. ToolExecutionRequest — 工具執行請求

```python
tool_name: str              # 工具名稱 (9 種工具)
parameters: dict            # 工具參數
trace_id: str | None        # 追蹤 ID
```

1. ToolExecutionResult — 工具執行結果

```python
status: Literal["success", "error", "pending"]
tool_name: str
result: dict | None
error: str | None
execution_time_ms: int | None
```

1. AIAgentQuery — AI 代理查詢請求

```python
query: str                  # 自然語言查詢 (1-1000 字元)
context: dict | None        # 額外上下文
use_rag: bool               # 是否使用 RAG (預設 True)
max_tokens: int             # 最大 token 數 (64-2048)
```

1. AIAgentResponse — AI 代理響應

```python
status: Literal["success", "uncertain", "error"]
query: str
tool_used: str | None
confidence: float          # 決策信心度 (0-1)
result: dict | None
context_chunks: list[RAGChunk] | None
message: str | None
timestamp: datetime
```

1. RAGChunk — RAG 檢索片段

```python
path: str                  # 檔案路徑
content: str               # 程式碼內容 (≤5000 字元)
type: Literal["FunctionDef", "ClassDef", "Module"]
name: str                  # 函式/類別名稱
score: int                 # 相關度分數 (≥0)
```

1. KnowledgeBaseStats — 知識庫統計

```python
total_chunks: int
total_keywords: int
indexed_files: int
last_indexed: datetime | None
```

### UI 面板相關（9 個）

1. ScanTaskRequest — 掃描任務請求

```python
target_url: str           # 必須以 http:// 或 https:// 開頭
scan_type: Literal["full", "quick", "custom"]
use_ai: bool | None       # None=自動決定
custom_config: dict | None
```

1. ScanTaskResponse — 掃描任務響應

```python
task_id: str
target: str
scan_type: str
status: Literal["pending", "running", "completed", "failed"]
created_by: Literal["ui", "ai"]
created_at: datetime
ai_result: AIAgentResponse | None
```

1. VulnerabilityDetectionRequest — 漏洞檢測請求

```python
vuln_type: Literal["xss", "sqli", "ssrf", "idor"]
target: str
use_ai: bool | None
parameters: dict | None
```

1. VulnerabilityDetectionResponse — 漏洞檢測響應

```python
vuln_type: str
target: str
status: Literal["pending", "completed", "failed"]
method: Literal["ui", "ai"]
findings: list[dict]
ai_result: AIAgentResponse | None
detected_at: datetime
```

1. CodeOperationRequest — 程式碼操作請求

```python
operation: Literal["read", "analyze", "write"]
path: str                 # 防止路徑遍歷攻擊
use_ai: bool | None
content: str | None       # write 操作使用
```

1. CodeOperationResponse — 程式碼操作響應

```python
status: Literal["success", "error"]
operation: str
path: str
content: str | None
analysis: dict | None
method: Literal["ui", "ai"]
error: str | None
```

1. DashboardStats — 控制面板統計

```python
mode: Literal["ui", "ai", "hybrid"]
mode_display: str
total_tasks: int
total_detections: int
ai_enabled: bool
ai_chunks: int | None
ai_keywords: int | None
ai_history_count: int | None
uptime_seconds: int | None
```

1. UIServerConfig — UI 伺服器配置

```python
mode: Literal["ui", "ai", "hybrid"]
host: str                 # 預設 127.0.0.1
port: int                 # 1024-65535
codebase_path: str
enable_cors: bool
debug: bool
```

### 工具結果標準格式（5 個）

1. CodeReadResult — 程式碼讀取結果
1. CodeWriteResult — 程式碼寫入結果
1. CodeAnalysisResult — 程式碼分析結果
1. CommandExecutionResult — 命令執行結果

---

## 🔍 驗證規則增強

所有新合約都包含完整的 Pydantic v2 驗證，包含：

### 數值範圍驗證

```python
# 埠號
port: int = Field(ge=1024, le=65535)

# 信心度
confidence: float = Field(ge=0.0, le=1.0)

# 分數
score: int = Field(ge=0)
```

## 字串長度驗證

```python
# 查詢長度
query: str = Field(min_length=1, max_length=1000)

# 程式碼片段
content: str = Field(max_length=5000)
```

## 自訂驗證器

```python
@field_validator("target_url")
@classmethod
def validate_url(cls, v: str) -> str:
    if not v.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return v

@field_validator("path")
@classmethod
def validate_path(cls, v: str) -> str:
    # 防止路徑遍歷
    if ".." in v or v.startswith("/"):
        raise ValueError("Invalid path: directory traversal not allowed")
    return v.strip()

@field_validator("tool_name")
@classmethod
def validate_tool_name(cls, v: str) -> str:
    valid_tools = {
        "CodeReader", "CodeWriter", "CodeAnalyzer",
        "ScanTrigger", "XSSDetector", "SQLiDetector",
        "SSRFDetector", "IDORDetector", "CommandExecutor",
    }
    if v not in valid_tools:
        raise ValueError(f"Invalid tool name: {v}")
    return v
```

---

## 📊 數據合約完整性對比

### 更新前

| 類別 | 數量 | 覆蓋率 |
|------|------|--------|
| 訊息協議 | 2 | 100% |
| 掃描相關 | 5 | 100% |
| 功能任務 | 4 | 100% |
| 漏洞報告 | 6 | 100% |
| 反饋狀態 | 4 | 100% |
| **AI 引擎** | **0** | **0%** ❌ |
| **UI 面板** | **0** | **0%** ❌ |
| **總計** | **21** | **75%** |

### 更新後

| 類別 | 數量 | 覆蓋率 |
|------|------|--------|
| 訊息協議 | 2 | 100% ✅ |
| 掃描相關 | 5 | 100% ✅ |
| 功能任務 | 4 | 100% ✅ |
| 漏洞報告 | 6 | 100% ✅ |
| 反饋狀態 | 4 | 100% ✅ |
| **AI 引擎** | **8** | **100%** ✅ |
| **UI 面板** | **9** | **100%** ✅ |
| **工具結果** | **5** | **100%** ✅ |
| **總計** | **43** | **100%** ✅ |

---

## 🎯 使用建議

### 導入新合約

```python
# AI 引擎
from services.core.aiva_core.ai_ui_schemas import (
    AIAgentQuery,
    AIAgentResponse,
    RAGChunk,
    ToolExecutionRequest,
    ToolExecutionResult,
)

# UI 面板
from services.core.aiva_core.ai_ui_schemas import (
    ScanTaskRequest,
    ScanTaskResponse,
    VulnerabilityDetectionRequest,
    DashboardStats,
)
```

## 替換現有 dict 返回值

**修改前**（未規範）：

```python
def create_scan_task(target: str) -> dict[str, Any]:
    return {
        "task_id": "scan_123",
        "target": target,
        "status": "pending",
    }
```

**修改後**（使用數據合約）：

```python
def create_scan_task(target: str) -> ScanTaskResponse:
    return ScanTaskResponse(
        task_id="scan_123",
        target=target,
        scan_type="full",
        status="pending",
        created_by="ui",
    )
```

### 3. API 端點使用

```python
from fastapi import FastAPI
from services.core.aiva_core.ai_ui_schemas import (
    ScanTaskRequest,
    ScanTaskResponse,
)

app = FastAPI()

@app.post("/api/scan", response_model=ScanTaskResponse)
async def create_scan(request: ScanTaskRequest) -> ScanTaskResponse:
    # Pydantic 自動驗證 request
    # 自動生成 API 文檔
    return dashboard.create_scan_task(
        target_url=request.target_url,
        scan_type=request.scan_type,
    )
```

---

## ✅ 後續建議

### 立即行動

1. ✅ 已完成：創建 `ai_ui_schemas.py`（22 個新合約）
1. 🔄 進行中：將 UI 面板代碼改用新合約
1. 📝 待辦：將 AI 引擎代碼改用新合約

### 中期目標

1. 更新 FastAPI 端點使用新合約（自動生成 OpenAPI 文檔）
1. 添加單元測試驗證所有合約
1. 生成 JSON Schema 供前端使用

### 長期規劃

1. 當 AI 引擎成熟時，將其移出成獨立的第五模組
1. 為獨立的 AI 模組創建專用數據合約檔案
1. 建立模組間數據合約版本控制機制

---

## 📝 變更歷史

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| 1.0.0 | 2025-10-13 | 初始版本 - 核心模組、掃描、功能、集成 |
| **1.1.0** | **2025-10-13** | **新增 AI 引擎與 UI 面板合約 (22 個新模型)** |

---

## 結論

數據合約已完整涵蓋所有現有模組，包括新增的 AI 引擎和 UI 面板。建議盡快將現有代碼遷移到使用新合約，以獲得 Pydantic 的自動驗證和類型安全優勢。

---
