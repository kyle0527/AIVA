"""
AI 統一指令系統 - 取代 RabbitMQ 的直接指揮架構

設計原則:
1. AI 作為統一指揮中心，直接下達命令給各模組
2. 使用數據合約確保類型安全和一致性
3. 同步/異步混合模式，保持靈活性
4. 完全移除 RabbitMQ 依賴，簡化部署和調試

架構流程:
    User → Core (AI) → AICommand → Module Handler → Result → Core (AI)
    
vs 舊架構:
    User → Core (AI) → RabbitMQ → Module Worker → RabbitMQ → Core (AI)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Protocol, runtime_checkable

from pydantic import BaseModel, Field, PrivateAttr


# ==================== 命令類型枚舉 ====================

@runtime_checkable
class CommandCallback(Protocol):
    """命令執行回調接口
    
    用於實時通知 UI 或其他系統命令執行進度、狀態變更和中間結果。
    """
    
    async def on_progress(
        self, 
        command_id: str,
        progress: float,  # 0.0 - 1.0
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """進度更新回調
        
        Args:
            command_id: 命令 ID
            progress: 進度值 (0.0 - 1.0)
            message: 進度消息
            metadata: 額外的元數據
        """
        ...
    
    async def on_status_change(
        self,
        command_id: str,
        old_status: "CommandStatus",
        new_status: "CommandStatus",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """狀態變更回調
        
        Args:
            command_id: 命令 ID
            old_status: 舊狀態
            new_status: 新狀態
            metadata: 額外的元數據
        """
        ...
    
    async def on_partial_result(
        self,
        command_id: str,
        result_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """中間結果回調
        
        Args:
            command_id: 命令 ID
            result_type: 結果類型 (vulnerability_found, url_discovered, etc.)
            data: 結果數據
            metadata: 額外的元數據
        """
        ...


class CommandType(str, Enum):
    """AI 命令類型 - 對應五大模組的主要功能"""
    
    # Scan 模組命令
    SCAN_PHASE0 = "scan_phase0"              # Phase 0 快速偵察
    SCAN_PHASE1 = "scan_phase1"              # Phase 1 深度掃描
    SCAN_COMPREHENSIVE = "scan_comprehensive" # 完整掃描（包含 Phase 0 + Phase 1）
    
    # Features 模組命令
    FEATURE_XSS_TEST = "feature_xss_test"    # XSS 測試
    FEATURE_SQLI_TEST = "feature_sqli_test"  # SQL 注入測試
    FEATURE_SSRF_TEST = "feature_ssrf_test"  # SSRF 測試
    FEATURE_IDOR_TEST = "feature_idor_test"  # IDOR 測試
    FEATURE_BIZLOGIC_TEST = "feature_bizlogic_test"  # 業務邏輯測試
    
    # Payload Generator 命令
    FEATURE_PAYLOAD_GENERATE = "feature_payload_generate"  # 生成各類 Payload
    
    # Wordlist Generator 命令
    FEATURE_WORDLIST_GENERATE = "feature_wordlist_generate"  # 生成和管理字典文件
    
    # Social Engineering 命令
    FEATURE_PHISHING_CAMPAIGN = "feature_phishing_campaign"  # 釣魚攻擊活動
    FEATURE_OSINT_COLLECT = "feature_osint_collect"  # OSINT 資訊收集
    
    # Integration 模組命令
    INTEGRATION_COORDINATE = "integration_coordinate"  # 協調多個功能測試
    INTEGRATION_REPORT = "integration_report"          # 生成報告
    
    # Core 內部命令
    CORE_ANALYZE = "core_analyze"            # AI 分析
    CORE_DECIDE = "core_decide"              # AI 決策
    
    # ===== 網路搜索命令 (新增) =====
    # 漏洞數據庫搜索
    SEARCH_EXPLOIT_DB = "search_exploit_db"           # ExploitDB 搜索
    SEARCH_CVE_DETAILS = "search_cve_details"         # CVE Details 搜索
    SEARCH_CWE_INFO = "search_cwe_info"               # CWE 信息查詢
    SEARCH_CAPEC_PATTERNS = "search_capec_patterns"   # CAPEC 攻擊模式
    
    # 威脅情報搜索
    SEARCH_THREAT_INTEL = "search_threat_intel"       # 威脅情報查詢
    SEARCH_IOC = "search_ioc"                         # IOC（入侵指標）搜索
    SEARCH_MALWARE_ANALYSIS = "search_malware_analysis"  # 惡意軟件分析
    
    # 開源情報搜索 (OSINT)
    SEARCH_GOOGLE = "search_google"                   # Google 搜索
    SEARCH_DUCKDUCKGO = "search_duckduckgo"           # DuckDuckGo 搜索
    SEARCH_GITHUB = "search_github"                   # GitHub 代碼/Issue 搜索
    SEARCH_STACKOVERFLOW = "search_stackoverflow"     # StackOverflow 搜索
    SEARCH_SHODAN = "search_shodan"                   # Shodan 設備搜索
    SEARCH_CENSYS = "search_censys"                   # Censys 資產搜索
    
    # 社交工程搜索
    SEARCH_EMAIL_BREACH = "search_email_breach"       # 郵箱洩露查詢
    SEARCH_DOMAIN_INFO = "search_domain_info"         # 域名信息查詢
    SEARCH_WHOIS = "search_whois"                     # WHOIS 查詢
    
    # AI 輔助搜索
    RAG_SEARCH_INTERNAL = "rag_search_internal"       # RAG 內部知識庫搜索
    RAG_SEARCH_EXTERNAL = "rag_search_external"       # RAG 外部網路搜索
    
    # 通用命令
    HEALTH_CHECK = "health_check"            # 健康檢查
    STATUS_QUERY = "status_query"            # 狀態查詢


class CommandStatus(str, Enum):
    """命令執行狀態"""
    PENDING = "pending"       # 待執行
    RUNNING = "running"       # 執行中
    COMPLETED = "completed"   # 成功完成
    FAILED = "failed"         # 執行失敗
    TIMEOUT = "timeout"       # 執行超時
    CANCELLED = "cancelled"   # 已取消


class CommandPriority(int, Enum):
    """命令優先級"""
    LOW = 1       # 低優先級（批量掃描、後台任務）
    NORMAL = 5    # 正常優先級（一般掃描）
    HIGH = 8      # 高優先級（用戶交互任務）
    URGENT = 10   # 緊急（安全事件響應）


# ==================== 核心命令模型 ====================

class AICommand(BaseModel):
    """AI 統一指令格式
    
    這是 AI 向各模組下達命令的標準格式，取代了 RabbitMQ 消息。
    
    使用示例:
        # Phase 0 掃描命令
        command = AICommand(
            command_id="scan_abc123_phase0",
            command_type=CommandType.SCAN_PHASE0,
            target_module="scan",
            payload={
                "scan_id": "scan_abc123",
                "targets": ["https://example.com"],
                "max_depth": 3
            },
            priority=CommandPriority.HIGH,
            timeout=600
        )
    """
    
    # 基本屬性
    command_id: str = Field(..., description="唯一命令 ID，用於追蹤和關聯")
    command_type: CommandType = Field(..., description="命令類型")
    target_module: str = Field(..., description="目標模組 (scan/features/integration/core)")
    
    # 命令內容
    payload: Dict[str, Any] = Field(..., description="命令負載，使用對應的 Payload 模型")
    
    # 執行控制
    priority: CommandPriority = Field(default=CommandPriority.NORMAL, description="執行優先級")
    timeout: int = Field(default=300, ge=1, le=3600, description="超時時間（秒）")
    
    # 追蹤信息
    trace_id: Optional[str] = Field(None, description="追蹤 ID，用於關聯相關命令")
    session_id: Optional[str] = Field(None, description="會話 ID")
    parent_command_id: Optional[str] = Field(None, description="父命令 ID（用於子命令）")
    
    # 元數據
    metadata: Dict[str, Any] = Field(default_factory=dict, description="額外的元數據")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="創建時間")
    
    # ===== 回調配置 (擴展) =====
    enable_callbacks: bool = Field(
        default=False,
        description="是否啟用實時回調機制"
    )
    
    callback_url: Optional[str] = Field(None, description="完成後的回調 URL（可選，用於 HTTP 回調）")
    
    ui_update_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="UI 更新間隔（秒）"
    )
    
    # 回調處理器（僅在 Python 內部使用，不序列化）
    _callback_handler: Optional[CommandCallback] = PrivateAttr(default=None)
    
    def set_callback(self, callback: CommandCallback) -> None:
        """設置回調處理器
        
        Args:
            callback: 實現了 CommandCallback 協議的回調處理器
        """
        self._callback_handler = callback
    
    def get_callback(self) -> Optional[CommandCallback]:
        """獲取回調處理器"""
        return self._callback_handler
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return self.model_dump()


class AICommandResult(BaseModel):
    """命令執行結果
    
    這是模組執行命令後返回給 AI 的標準格式。
    
    使用示例:
        result = AICommandResult(
            command_id="scan_abc123_phase0",
            status=CommandStatus.COMPLETED,
            result={
                "scan_id": "scan_abc123",
                "assets": [...],
                "summary": {...}
            },
            execution_time=45.2,
            success=True
        )
    """
    
    # 關聯信息
    command_id: str = Field(..., description="對應的命令 ID")
    
    # 執行狀態
    status: CommandStatus = Field(..., description="執行狀態")
    success: bool = Field(..., description="是否成功執行")
    
    # 執行結果
    result: Dict[str, Any] = Field(default_factory=dict, description="執行結果，使用對應的 Result 模型")
    
    # 執行信息
    execution_time: float = Field(..., ge=0, description="執行時間（秒）")
    started_at: Optional[datetime] = Field(None, description="開始執行時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    
    # 錯誤信息
    error: Optional[str] = Field(None, description="錯誤信息（如果失敗）")
    error_code: Optional[str] = Field(None, description="錯誤代碼")
    error_details: Optional[Dict[str, Any]] = Field(None, description="詳細錯誤信息")
    
    # 性能指標
    metrics: Dict[str, Any] = Field(default_factory=dict, description="性能指標（可選）")
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return self.model_dump()


# ==================== 批量命令處理 ====================

class AICommandBatch(BaseModel):
    """批量命令
    
    用於同時下達多個相關命令，支持並行或順序執行。
    
    使用示例:
        batch = AICommandBatch(
            batch_id="batch_xyz789",
            commands=[command1, command2, command3],
            execution_mode="parallel",
            max_concurrent=3
        )
    """
    
    batch_id: str = Field(..., description="批次 ID")
    commands: List[AICommand] = Field(..., description="命令列表")
    
    # 執行模式
    execution_mode: str = Field(default="parallel", description="執行模式: parallel/sequential")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="最大並發數（parallel 模式）")
    
    # 錯誤處理
    continue_on_error: bool = Field(default=False, description="遇到錯誤是否繼續執行")
    
    # 元數據
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AICommandBatchResult(BaseModel):
    """批量命令執行結果"""
    
    batch_id: str = Field(..., description="批次 ID")
    total_commands: int = Field(..., description="總命令數")
    completed: int = Field(default=0, description="已完成數")
    failed: int = Field(default=0, description="失敗數")
    
    results: List[AICommandResult] = Field(default_factory=list, description="所有結果")
    
    total_time: float = Field(..., description="總執行時間（秒）")
    success_rate: float = Field(..., ge=0, le=1, description="成功率")


# ==================== 命令執行上下文 ====================

class CommandContext(BaseModel):
    """命令執行上下文
    
    提供命令執行時需要的環境信息和配置。
    """
    
    # 追蹤信息
    trace_id: Optional[str] = Field(None, description="追蹤 ID，用於關聯相關操作")
    
    # 身份信息
    user_id: Optional[str] = Field(None, description="用戶 ID")
    session_id: Optional[str] = Field(None, description="會話 ID")
    
    # 環境配置
    environment: str = Field(default="local", description="運行環境: local/docker/cloud")
    
    # 資源限制
    max_memory_mb: Optional[int] = Field(None, description="最大內存限制（MB）")
    max_cpu_percent: Optional[int] = Field(None, ge=1, le=100, description="最大 CPU 使用率（%）")
    
    # 調試選項
    debug_mode: bool = Field(default=False, description="是否啟用調試模式")
    verbose: bool = Field(default=False, description="是否輸出詳細日誌")
    
    # 自定義配置
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="自定義配置")


# ==================== 導出所有模型 ====================

__all__ = [
    # 枚舉
    "CommandType",
    "CommandStatus",
    "CommandPriority",
    
    # 核心模型
    "AICommand",
    "AICommandResult",
    
    # 批量處理
    "AICommandBatch",
    "AICommandBatchResult",
    
    # 上下文
    "CommandContext",
]
