"""
AI Engine 與 UI Panel 數據合約

定義 AI 代理、工具系統、UI 控制面板的標準數據結構
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# AI 引擎相關數據合約
# ============================================================================


class ToolExecutionRequest(BaseModel):
    """工具執行請求."""

    tool_name: str = Field(description="工具名稱")
    parameters: dict[str, str | int | bool | list[str]] = Field(
        default_factory=dict,
        description="工具參數",
    )
    trace_id: str | None = Field(default=None, description="追蹤 ID")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """驗證工具名稱."""
        valid_tools = {
            "CodeReader",
            "CodeWriter",
            "CodeAnalyzer",
            "ScanTrigger",
            "XSSDetector",
            "SQLiDetector",
            "SSRFDetector",
            "IDORDetector",
            "CommandExecutor",
        }
        if v not in valid_tools:
            raise ValueError(f"Invalid tool name: {v}. Valid tools: {valid_tools}")
        return v


class ToolExecutionResult(BaseModel):
    """工具執行結果."""

    status: Literal["success", "error", "pending"] = Field(description="執行狀態")
    tool_name: str = Field(description="工具名稱")
    result: dict[str, str | int | list[str]] | None = Field(
        default=None,
        description="執行結果數據",
    )
    error: str | None = Field(default=None, description="錯誤訊息")
    execution_time_ms: int | None = Field(default=None, description="執行時間(毫秒)")
    trace_id: str | None = Field(default=None, description="追蹤 ID")

    @field_validator("execution_time_ms")
    @classmethod
    def validate_execution_time(cls, v: int | None) -> int | None:
        """驗證執行時間."""
        if v is not None and v < 0:
            raise ValueError("Execution time cannot be negative")
        return v


class AIAgentQuery(BaseModel):
    """AI 代理查詢請求."""

    query: str = Field(description="自然語言查詢", min_length=1, max_length=1000)
    context: dict[str, str] | None = Field(default=None, description="額外上下文")
    use_rag: bool = Field(default=True, description="是否使用 RAG 檢索")
    max_tokens: int = Field(default=512, ge=64, le=2048, description="最大 token 數")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """驗證查詢字串."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class AIAgentResponse(BaseModel):
    """AI 代理響應."""

    status: Literal["success", "uncertain", "error"] = Field(description="執行狀態")
    query: str = Field(description="原始查詢")
    tool_used: str | None = Field(default=None, description="使用的工具名稱")
    confidence: float = Field(ge=0.0, le=1.0, description="決策信心度 (0-1)")
    result: dict[str, str | int | list[dict[str, str]]] | None = Field(
        default=None,
        description="執行結果",
    )
    context_chunks: list[RAGChunk] | None = Field(
        default=None,
        description="檢索到的上下文片段",
    )
    message: str | None = Field(default=None, description="狀態訊息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="時間戳")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """驗證信心度."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class RAGChunk(BaseModel):
    """RAG 檢索的程式碼片段."""

    path: str = Field(description="檔案路徑")
    content: str = Field(description="程式碼內容", max_length=5000)
    type: Literal["FunctionDef", "ClassDef", "Module"] = Field(description="程式碼類型")
    name: str = Field(description="函式/類別名稱")
    score: int = Field(ge=0, description="檢索相關度分數")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        """驗證分數."""
        if v < 0:
            raise ValueError("Score cannot be negative")
        return v


class KnowledgeBaseStats(BaseModel):
    """知識庫統計資訊."""

    total_chunks: int = Field(ge=0, description="程式碼片段總數")
    total_keywords: int = Field(ge=0, description="關鍵字總數")
    indexed_files: int = Field(ge=0, description="已索引檔案數")
    last_indexed: datetime | None = Field(default=None, description="最後索引時間")


# ============================================================================
# UI 面板相關數據合約
# ============================================================================


class ScanTaskRequest(BaseModel):
    """掃描任務建立請求."""

    target_url: str = Field(description="目標 URL")
    scan_type: Literal["full", "quick", "custom"] = Field(
        default="full",
        description="掃描類型",
    )
    use_ai: bool | None = Field(default=None, description="是否使用 AI (None=自動)")
    custom_config: dict[str, str | int | bool] | None = Field(
        default=None,
        description="自訂配置",
    )

    @field_validator("target_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """驗證 URL 格式."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class ScanTaskResponse(BaseModel):
    """掃描任務響應."""

    task_id: str = Field(description="任務 ID")
    target: str = Field(description="目標 URL")
    scan_type: str = Field(description="掃描類型")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        description="任務狀態"
    )
    created_by: Literal["ui", "ai"] = Field(description="建立方式")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="建立時間"
    )
    ai_result: AIAgentResponse | None = Field(default=None, description="AI 執行結果")


class VulnerabilityDetectionRequest(BaseModel):
    """漏洞檢測請求."""

    vuln_type: Literal["xss", "sqli", "ssrf", "idor"] = Field(description="漏洞類型")
    target: str = Field(description="檢測目標")
    use_ai: bool | None = Field(default=None, description="是否使用 AI")
    parameters: dict[str, str] | None = Field(default=None, description="檢測參數")

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        """驗證目標."""
        if not v.strip():
            raise ValueError("Target cannot be empty")
        return v.strip()


class VulnerabilityDetectionResponse(BaseModel):
    """漏洞檢測響應."""

    vuln_type: str = Field(description="漏洞類型")
    target: str = Field(description="檢測目標")
    status: Literal["pending", "completed", "failed"] = Field(description="檢測狀態")
    method: Literal["ui", "ai"] = Field(description="檢測方式")
    findings: list[dict[str, str]] = Field(default_factory=list, description="發現列表")
    ai_result: AIAgentResponse | None = Field(default=None, description="AI 結果")
    detected_at: datetime = Field(
        default_factory=datetime.utcnow, description="檢測時間"
    )


class CodeOperationRequest(BaseModel):
    """程式碼操作請求."""

    operation: Literal["read", "analyze", "write"] = Field(description="操作類型")
    path: str = Field(description="檔案路徑")
    use_ai: bool | None = Field(default=None, description="是否使用 AI")
    content: str | None = Field(default=None, description="寫入內容 (write 操作)")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """驗證路徑."""
        if not v.strip():
            raise ValueError("Path cannot be empty")
        # 防止路徑遍歷
        if ".." in v or v.startswith("/"):
            raise ValueError("Invalid path: directory traversal not allowed")
        return v.strip()


class CodeOperationResponse(BaseModel):
    """程式碼操作響應."""

    status: Literal["success", "error"] = Field(description="操作狀態")
    operation: str = Field(description="操作類型")
    path: str = Field(description="檔案路徑")
    content: str | None = Field(default=None, description="檔案內容")
    analysis: dict[str, int] | None = Field(default=None, description="分析結果")
    method: Literal["ui", "ai"] = Field(description="執行方式")
    error: str | None = Field(default=None, description="錯誤訊息")


class DashboardStats(BaseModel):
    """控制面板統計資訊."""

    mode: Literal["ui", "ai", "hybrid"] = Field(description="運作模式")
    mode_display: str = Field(description="模式顯示名稱")
    total_tasks: int = Field(ge=0, description="總任務數")
    total_detections: int = Field(ge=0, description="總檢測數")
    ai_enabled: bool = Field(description="AI 是否啟用")
    ai_chunks: int | None = Field(default=None, ge=0, description="AI 知識庫片段數")
    ai_keywords: int | None = Field(default=None, ge=0, description="AI 關鍵字數")
    ai_history_count: int | None = Field(
        default=None, ge=0, description="AI 執行歷史數"
    )
    uptime_seconds: int | None = Field(default=None, ge=0, description="運行時間(秒)")


class UIServerConfig(BaseModel):
    """UI 伺服器配置."""

    mode: Literal["ui", "ai", "hybrid"] = Field(
        default="hybrid",
        description="運作模式",
    )
    host: str = Field(default="127.0.0.1", description="綁定主機")
    port: int = Field(default=8080, ge=1024, le=65535, description="綁定埠號")
    codebase_path: str = Field(description="程式碼庫根目錄路徑")
    enable_cors: bool = Field(default=False, description="是否啟用 CORS")
    debug: bool = Field(default=False, description="是否啟用偵錯模式")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """驗證埠號."""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v


# ============================================================================
# 工具結果標準格式
# ============================================================================


class CodeReadResult(BaseModel):
    """程式碼讀取結果."""

    status: Literal["success", "error"] = Field(description="狀態")
    path: str = Field(description="檔案路徑")
    content: str | None = Field(default=None, description="檔案內容")
    lines: int | None = Field(default=None, ge=0, description="總行數")
    error: str | None = Field(default=None, description="錯誤訊息")


class CodeWriteResult(BaseModel):
    """程式碼寫入結果."""

    status: Literal["success", "error"] = Field(description="狀態")
    path: str = Field(description="檔案路徑")
    bytes_written: int | None = Field(default=None, ge=0, description="寫入位元組數")
    error: str | None = Field(default=None, description="錯誤訊息")


class CodeAnalysisResult(BaseModel):
    """程式碼分析結果."""

    status: Literal["success", "error"] = Field(description="狀態")
    path: str = Field(description="檔案路徑")
    total_lines: int | None = Field(default=None, ge=0, description="總行數")
    imports: int | None = Field(default=None, ge=0, description="導入語句數")
    functions: int | None = Field(default=None, ge=0, description="函式數")
    classes: int | None = Field(default=None, ge=0, description="類別數")
    error: str | None = Field(default=None, description="錯誤訊息")


class CommandExecutionResult(BaseModel):
    """命令執行結果."""

    status: Literal["success", "error"] = Field(description="狀態")
    command: str = Field(description="執行的命令")
    stdout: str | None = Field(default=None, description="標準輸出")
    stderr: str | None = Field(default=None, description="標準錯誤")
    returncode: int | None = Field(default=None, description="返回碼")
    error: str | None = Field(default=None, description="錯誤訊息")
