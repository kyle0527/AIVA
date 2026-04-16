"""
雙閉環架構專用 Schema

定義內部閉環和外部閉環使用的 Pydantic 模型，遵循 aiva_common v2.0 標準。

設計理念：
- 內閉環: AI 自我認知 → 知道有哪些能力、問題、解法（通過 RAG 查詢）
- 外閉環: 使用內閉環發現的能力 → 實戰執行 → 學習優化

能力分類系統：
- 掃描能力 (Scanning): 端口掃描、漏洞掃描、服務識別
- 攻擊能力 (Attacking): SQL注入、XSS、業務邏輯漏洞
- 分析能力 (Analysis): 數據分析、結果解析、偏差分析
- 輔助能力 (Utility): 編碼/解碼、加密/解密、數據轉換
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ==================== 能力分類系統 ====================

class CapabilityCategory(str, Enum):
    """能力類別"""
    SCANNING = "scanning"  # 掃描能力
    ATTACKING = "attacking"  # 攻擊能力
    ANALYSIS = "analysis"  # 分析能力
    UTILITY = "utility"  # 輔助能力
    REPORTING = "reporting"  # 報告生成
    INTEGRATION = "integration"  # 整合協調


class CapabilitySubCategory(str, Enum):
    """能力子類別（詳細分類）"""
    # Scanning 子類別
    PORT_SCAN = "port_scan"
    VULN_SCAN = "vulnerability_scan"
    SERVICE_DETECT = "service_detection"
    WEB_CRAWL = "web_crawling"

    # Attacking 子類別
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    SSRF = "ssrf"
    BUSINESS_LOGIC = "business_logic"
    AUTH_BYPASS = "auth_bypass"

    # Analysis 子類別
    DEVIATION_ANALYSIS = "deviation_analysis"
    RESULT_PARSING = "result_parsing"
    PATTERN_MATCHING = "pattern_matching"

    # Utility 子類別
    ENCODING = "encoding"
    ENCRYPTION = "encryption"
    DATA_TRANSFORM = "data_transformation"

    # Reporting 子類別
    REPORT_GEN = "report_generation"
    METRICS = "metrics"

    # Integration 子類別
    COORDINATION = "coordination"
    ORCHESTRATION = "orchestration"


class CapabilityComplexity(int, Enum):
    """能力複雜度"""
    TRIVIAL = 1  # 簡單：單一操作，無依賴
    SIMPLE = 2  # 簡單：多個操作，少量依賴
    MODERATE = 3  # 中等：需要配置，有依賴
    COMPLEX = 4  # 複雜：多步驟，多依賴
    ADVANCED = 5  # 高級：涉及 AI 決策、多模組協調


# ==================== 能力範圍管理 (v3.0) ====================

class CapabilityScope(str, Enum):
    """能力範圍
    
    定義能力可用的範圍層級：
    - CORE: 核心內部能力（只能 aiva_core 使用）
    - SERVICE: 服務級能力（services/* 可用）
    - GLOBAL: 全局能力（整個項目可用）
    - EXTERNAL: 外部工具能力（調用第三方工具）
    """
    CORE = "core"              # 核心內部能力（如 AST 分析、RAG 管理）
    SERVICE = "service"        # 服務級能力（如 AI 決策、多引擎協調）
    GLOBAL = "global"          # 全局能力（如 SQL 注入、XSS 檢測）
    EXTERNAL = "external"      # 外部工具能力（如 sqlmap, XSStrike）


class CapabilityVisibility(str, Enum):
    """能力可見性
    
    定義能力的可見性級別：
    - PUBLIC: AI Commander 可直接調用
    - INTERNAL: 只能內部調用（模組間調用）
    - SYSTEM: 系統級（需特殊權限）
    - DEPRECATED: 已棄用（不推薦使用）
    """
    PUBLIC = "public"          # AI Commander 可直接調用
    INTERNAL = "internal"      # 只能內部調用
    SYSTEM = "system"          # 系統級（需特殊權限）
    DEPRECATED = "deprecated"  # 已棄用


class CapabilityAccessLevel(str, Enum):
    """能力訪問級別
    
    定義能力的訪問控制級別：
    - L0_SYSTEM: 系統管理（如服務重啟、配置更新）
    - L1_SERVICE: 服務協調（如多引擎協調、任務分發）
    - L2_MODULE: 模組功能（如 SQL 注入檢測、XSS 掃描）
    - L3_INTERNAL: 內部實現（如 AST 分析、數據解析）
    """
    L0_SYSTEM = "system"       # 系統管理（如服務重啟）
    L1_SERVICE = "service"     # 服務協調（如多引擎協調）
    L2_MODULE = "module"       # 模組功能（如 SQL 注入檢測）
    L3_INTERNAL = "internal"   # 內部實現（如 AST 分析）


class CLIMaturityLevel(str, Enum):
    """CLI 成熟度級別
    
    定義能力 CLI 接口的成熟度：
    - NONE: 無 CLI 接口
    - ALPHA: 早期 CLI（有基本命令，未測試）
    - BETA: 測試中（有完整命令，部分測試）
    - STABLE: 穩定版（完整測試，文檔齊全）
    """
    NONE = "none"              # 無 CLI 接口
    ALPHA = "alpha"            # 早期 CLI（未測試）
    BETA = "beta"              # 測試中
    STABLE = "stable"          # 穩定版


# ==================== 參數定義 ====================

class ParameterDefinition(BaseModel):
    """能力參數定義"""
    name: str = Field(..., description="參數名稱")
    type: str = Field(..., description="參數類型 (str, int, bool, list, dict 等)")
    required: bool = Field(default=True, description="是否必需")
    default: Any | None = Field(None, description="默認值")
    description: str = Field(..., description="參數描述")
    example: Any | None = Field(None, description="範例值")
    constraints: dict[str, Any] | None = Field(None, description="約束條件 (min, max, enum 等)")


class ReturnDefinition(BaseModel):
    """能力返回值定義"""
    type: str = Field(..., description="返回類型")
    description: str = Field(..., description="返回值描述")
    example: Any | None = Field(None, description="範例返回值")
    structure: dict[str, Any] | None = Field(None, description="返回值結構（若為複雜對象）")


# ==================== 調用元數據（新增）====================

class InvocationInfo(BaseModel):
    """能力調用元數據
    
    描述如何調用一個能力：
    - protocol: 調用協議 (http, grpc, direct, websocket)
    - endpoint: 調用端點 URL 或路徑
    - module_arg: 模組參數名
    - function_arg: 函數參數名
    - parameter_mapping: 參數映射關係
    """
    protocol: Literal["http", "grpc", "direct", "websocket", "unified_caller"] = Field(
        ...,
        description="調用協議"
    )
    endpoint: str = Field(..., description="調用端點 (URL 或 direct://module.function)")
    module_arg: str | None = Field(None, description="模組參數名")
    function_arg: str | None = Field(None, description="函數參數名")
    parameter_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="參數名映射 {external_name: internal_name}"
    )
    timeout_seconds: int = Field(default=30, description="超時時間（秒）", ge=1)
    retry_count: int = Field(default=0, description="重試次數", ge=0)


class ChangeType(str, Enum):
    """變更類型"""
    ADDED = "added"  # 新增
    MODIFIED = "modified"  # 修改
    DELETED = "deleted"  # 刪除
    UNCHANGED = "unchanged"  # 未變更


# ==================== 圖形拼接數據流 Schema ====================

class DataFlowPathNode(BaseModel):
    """數據流路徑節點
    
    表示數據流中的一個函數調用點
    """
    node_id: str = Field(..., description="節點唯一標識")
    function_signature: str = Field(..., description="函數簽名 (ClassName.method_name)")
    file_path: str = Field(..., description="檔案路徑")
    module_name: str = Field(..., description="模組名稱")
    is_async: bool = Field(default=False, description="是否為異步函數")
    parameters: list[ParameterDefinition] = Field(default_factory=list, description="參數列表")
    line_number: int = Field(default=0, description="行號")


class CompleteDataFlow(BaseModel):
    """完整數據流
    
    從入口點到出口點的完整調用路徑 (基於圖形拼接生成)
    """
    flow_id: str = Field(..., description="數據流唯一標識")
    entry_point: str = Field(..., description="入口點函數簽名")
    exit_point: str | None = Field(None, description="出口點函數簽名")
    path: list[DataFlowPathNode] = Field(..., description="完整調用路徑")
    path_length: int = Field(..., description="路徑長度 (節點數)")
    crosses_files: bool = Field(default=False, description="是否跨檔案")
    file_count: int = Field(default=0, description="涉及的檔案數量")

    @field_validator("path_length", mode="before")
    @classmethod
    def set_path_length(cls, v: Any, info) -> int:
        """自動設置路徑長度"""
        if "path" in info.data:
            return len(info.data["path"])
        return v


class BrokenChainDiagnosis(BaseModel):
    """斷鏈診斷記錄
    
    記錄無法拼接的調用點,用於自我修復診斷
    """
    caller_signature: str = Field(..., description="調用方函數簽名")
    caller_file: str = Field(..., description="調用方檔案路徑")
    missing_function: str = Field(..., description="缺失的被調用函數")
    line_number: int = Field(..., description="調用位置行號")
    possible_causes: list[str] = Field(default_factory=list, description="可能原因")
    severity: Literal["error", "warning", "info"] = Field(
        default="warning",
        description="嚴重程度"
    )
    auto_fix_suggestion: str | None = Field(None, description="自動修復建議")


class DataFlowAnalysisSnapshot(BaseModel):
    """數據流分析快照
    
    保存某一時刻的完整分析結果,支援增量更新
    """
    snapshot_id: str = Field(..., description="快照唯一標識")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="分析時間")
    total_files_analyzed: int = Field(..., description="分析的檔案總數")
    total_functions_found: int = Field(..., description="發現的函數總數")
    entry_points: list[str] = Field(..., description="所有入口點列表")
    complete_flows: list[CompleteDataFlow] = Field(
        default_factory=list,
        description="所有完整數據流"
    )
    broken_chains: list[BrokenChainDiagnosis] = Field(
        default_factory=list,
        description="所有斷鏈診斷"
    )
    file_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="檔案哈希映射 (用於增量更新)"
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="統計資訊"
    )

    def get_flows_by_entry(self, entry_point: str) -> list[CompleteDataFlow]:
        """獲取特定入口點的所有數據流"""
        return [flow for flow in self.complete_flows if flow.entry_point == entry_point]

    def get_broken_chains_by_severity(self, severity: str) -> list[BrokenChainDiagnosis]:
        """按嚴重程度過濾斷鏈"""
        return [chain for chain in self.broken_chains if chain.severity == severity]


# ==================== 能力使用範例 ====================

class CapabilityUsageExample(BaseModel):
    """能力使用範例"""
    title: str = Field(..., description="範例標題")
    description: str = Field(..., description="範例描述")
    input: dict[str, Any] = Field(..., description="輸入參數範例")
    expected_output: Any | None = Field(None, description="預期輸出")
    code_snippet: str | None = Field(None, description="代碼範例")
    notes: str | None = Field(None, description="注意事項")


# ==================== 內部閉環：能力記錄 ====================

class ModuleCapability(BaseModel):
    """模組能力記錄（內部閉環核心數據結構）
    
    記錄 AI 系統可用的所有能力，包括：
    - 能力基本信息（名稱、描述、位置）
    - 能力分類（類別、子類別、複雜度）
    - 五大模組分類（aiva_module, sub_module, entry_point）
    - 使用方法（參數、返回值、範例）
    - 健康狀態（可用性、性能、錯誤率）
    """

    # 基本信息
    capability_id: str = Field(..., description="能力唯一標識符")
    name: str = Field(..., description="能力名稱")
    module: str = Field(..., description="模組路徑")
    function: str = Field(..., description="函數名稱")
    language: str = Field(..., description="程式語言 (python, typescript, rust, go 等)")
    file_path: str | None = Field(None, description="檔案路徑")
    description: str | None = Field(None, description="能力描述")

    # 能力分類
    category: CapabilityCategory = Field(..., description="能力類別")
    sub_category: CapabilitySubCategory | None = Field(None, description="能力子類別")
    complexity: CapabilityComplexity = Field(
        default=CapabilityComplexity.MODERATE,
        description="能力複雜度"
    )
    tags: list[str] = Field(default_factory=list, description="標籤列表")

    # 五大模組分類（cognitive_core 整合了 external_learning）
    aiva_module: str | None = Field(
        None,
        description="AIVA 五大模組分類: cognitive_core (含 learning_system), internal_exploration, task_planning, core_capabilities, service_backbone"
    )
    sub_module: str | None = Field(
        None,
        description="子模組名稱，例如: neural, rag, decision (for cognitive_core), python_tools, go_tools (for internal_exploration)"
    )
    entry_point: str | None = Field(
        None,
        description="主要入口點: AICommander, CapabilityOrchestrator, app.py, ExternalLoopConnector, ScanResultProcessor, InternalLoopConnector"
    )

    # 使用方法
    parameters: list[ParameterDefinition] = Field(
        default_factory=list,
        description="參數列表"
    )
    return_info: ReturnDefinition | None = Field(None, description="返回值信息")
    usage_examples: list[CapabilityUsageExample] = Field(
        default_factory=list,
        description="使用範例"
    )

    # ✅ 新增：調用元數據
    invocation: InvocationInfo | None = Field(None, description="調用元數據（如何調用此能力）")

    # 依賴信息
    dependencies: list[str] = Field(default_factory=list, description="依賴的其他能力")
    prerequisites: list[str] = Field(default_factory=list, description="前置條件")

    # 健康狀態
    health_score: float = Field(
        default=1.0,
        description="健康分數 (0-1)",
        ge=0,
        le=1
    )
    availability: float = Field(
        default=1.0,
        description="可用性 (0-1)",
        ge=0,
        le=1
    )
    avg_latency_ms: float | None = Field(None, description="平均延遲(毫秒)", ge=0)
    error_rate: float = Field(
        default=0.0,
        description="錯誤率 (0-1)",
        ge=0,
        le=1
    )
    last_used: datetime | None = Field(None, description="最後使用時間")

    # ==================== ✅ v3.0 新增: 範圍管理 ====================

    # 範圍分類
    scope: CapabilityScope = Field(
        default=CapabilityScope.CORE,
        description="能力範圍：CORE(核心內部)/SERVICE(服務級)/GLOBAL(全局)/EXTERNAL(外部工具)"
    )
    visibility: CapabilityVisibility = Field(
        default=CapabilityVisibility.INTERNAL,
        description="能力可見性：PUBLIC(可直接調用)/INTERNAL(內部)/SYSTEM(系統級)/DEPRECATED(已棄用)"
    )
    access_level: CapabilityAccessLevel = Field(
        default=CapabilityAccessLevel.L3_INTERNAL,
        description="訪問級別：L0_SYSTEM(系統)/L1_SERVICE(服務)/L2_MODULE(模組)/L3_INTERNAL(內部)"
    )

    # 可用性條件
    available_in: list[str] = Field(
        default_factory=list,
        description="可用的服務路徑列表，例如：['core', 'features/sqli', 'scan']"
    )
    depends_on_services: list[str] = Field(
        default_factory=list,
        description="依賴的服務列表，例如：['core/rag', 'scan', 'features']"
    )

    # CLI 相關
    has_cli: bool = Field(
        default=False,
        description="是否有 CLI 接口"
    )
    cli_command: str | None = Field(
        None,
        description="CLI 命令，例如：'aiva sqli scan' 或 'aiva xss detect'"
    )
    cli_maturity: CLIMaturityLevel = Field(
        default=CLIMaturityLevel.NONE,
        description="CLI 成熟度：NONE(無)/ALPHA(早期)/BETA(測試)/STABLE(穩定)"
    )

    # 元數據
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="創建時間"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="更新時間"
    )
    version: str = Field(default="1.0.0", description="能力版本")


class CapabilitySummary(BaseModel):
    """能力摘要（用於快速查詢）"""
    total_capabilities: int = Field(..., description="總能力數", ge=0)
    by_category: dict[str, int] = Field(..., description="按類別統計")
    by_complexity: dict[str, int] = Field(..., description="按複雜度統計")
    healthy_count: int = Field(..., description="健康能力數", ge=0)
    unhealthy_count: int = Field(..., description="不健康能力數", ge=0)
    avg_health_score: float = Field(..., description="平均健康分數", ge=0, le=1)


class InternalLoopSyncResult(BaseModel):
    """內部閉環同步結果"""
    modules_scanned: int = Field(..., description="掃描的模組數", ge=0)
    capabilities_found: int = Field(..., description="發現的能力數", ge=0)
    capabilities: list[ModuleCapability] = Field(..., description="能力列表")
    summary: CapabilitySummary | None = Field(None, description="能力摘要")
    documents_added: int = Field(..., description="添加到 RAG 的文檔數", ge=0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="同步時間"
    )
    success: bool = Field(..., description="是否成功")
    error: str | None = Field(None, description="錯誤信息")

    def calculate_summary(self) -> CapabilitySummary:
        """計算能力摘要"""
        by_category = {}
        by_complexity = {}
        healthy = 0
        unhealthy = 0
        total_health = 0.0

        for cap in self.capabilities:
            # 按類別統計
            cat = cap.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            # 按複雜度統計
            comp = str(cap.complexity.value)
            by_complexity[comp] = by_complexity.get(comp, 0) + 1

            # 健康狀態統計
            if cap.health_score >= 0.7:
                healthy += 1
            else:
                unhealthy += 1
            total_health += cap.health_score

        avg_health = total_health / len(self.capabilities) if self.capabilities else 0.0

        return CapabilitySummary(
            total_capabilities=len(self.capabilities),
            by_category=by_category,
            by_complexity=by_complexity,
            healthy_count=healthy,
            unhealthy_count=unhealthy,
            avg_health_score=avg_health
        )


# ==================== 內部閉環：問題記錄 ====================

class SystemIssue(BaseModel):
    """系統問題記錄（內部閉環：問題追蹤）"""
    issue_id: str = Field(..., description="問題唯一標識符")
    title: str = Field(..., description="問題標題")
    description: str = Field(..., description="問題描述")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ...,
        description="嚴重程度"
    )
    affected_capabilities: list[str] = Field(
        default_factory=list,
        description="受影響的能力ID列表"
    )
    root_cause: str | None = Field(None, description="根本原因")
    potential_solutions: list[str] = Field(
        default_factory=list,
        description="潛在解決方案"
    )
    status: Literal["open", "investigating", "resolved", "ignored"] = Field(
        default="open",
        description="問題狀態"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="發現時間"
    )
    resolved_at: datetime | None = Field(None, description="解決時間")


# ==================== 內部閉環：RAG 查詢 ====================

class RAGQueryRequest(BaseModel):
    """RAG 查詢請求"""
    query: str = Field(..., description="查詢內容")
    query_type: Literal[
        "capability_search",  # 搜索能力
        "problem_solution",  # 查找問題解法
        "usage_example",  # 查找使用範例
        "general"  # 一般查詢
    ] = Field(default="general", description="查詢類型")
    filters: dict[str, Any] | None = Field(None, description="過濾條件")
    top_k: int = Field(default=5, description="返回結果數", ge=1, le=50)


class RAGQueryResult(BaseModel):
    """RAG 查詢結果"""
    query: str = Field(..., description="原始查詢")
    results: list[dict[str, Any]] = Field(..., description="查詢結果列表")
    total_found: int = Field(..., description="總找到結果數", ge=0)
    relevance_scores: list[float] = Field(..., description="相關性分數列表")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="查詢時間"
    )


# ==================== 外部閉環：執行計劃與軌跡 ====================

class ExecutionStep(BaseModel):
    """執行步驟"""
    step_id: str = Field(..., description="步驟ID")
    action: str = Field(..., description="動作類型")
    capability_id: str | None = Field(None, description="使用的能力ID")
    parameters: dict[str, Any] = Field(default_factory=dict, description="參數")
    expected_duration: float | None = Field(None, description="預期耗時(秒)", ge=0)
    dependencies: list[str] = Field(default_factory=list, description="依賴的步驟ID")


class ExecutionPlan(BaseModel):
    """執行計劃 (AST)"""
    plan_id: str = Field(..., description="計劃ID")
    objective: str = Field(..., description="目標描述")
    steps: list[ExecutionStep] = Field(..., description="計劃步驟")
    expected_duration: float | None = Field(None, description="預期總耗時(秒)", ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="創建時間"
    )
    metadata: dict[str, Any] | None = Field(None, description="元數據")


class ExecutionTrace(BaseModel):
    """執行軌跡記錄"""
    trace_id: str = Field(..., description="軌跡ID")
    step_id: str = Field(..., description="對應步驟ID")
    capability_id: str | None = Field(None, description="使用的能力ID")
    status: Literal["success", "failed", "skipped", "timeout"] = Field(
        ...,
        description="執行狀態"
    )
    duration: float = Field(..., description="實際耗時(秒)", ge=0)
    start_time: datetime = Field(..., description="開始時間")
    end_time: datetime = Field(..., description="結束時間")
    output: Any | None = Field(None, description="輸出結果")
    error: str | None = Field(None, description="錯誤信息")
    metrics: dict[str, Any] | None = Field(None, description="性能指標")


# ==================== 外部閉環：偏差分析 ====================

class DeviationRecord(BaseModel):
    """偏差記錄（外部閉環：學習材料）"""
    deviation_id: str = Field(..., description="偏差ID")
    type: Literal[
        "incomplete_execution",  # 未完成執行
        "execution_failures",  # 執行失敗
        "slow_execution",  # 執行緩慢
        "unexpected_output",  # 意外輸出
        "timeout",  # 超時
        "logic_error"  # 邏輯錯誤
    ] = Field(..., description="偏差類型")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ...,
        description="嚴重程度"
    )
    score: float = Field(..., description="偏差分數", ge=0)

    # 偏差詳情
    expected: dict[str, Any] | None = Field(None, description="預期結果")
    actual: dict[str, Any] | None = Field(None, description="實際結果")
    affected_steps: list[str] = Field(default_factory=list, description="受影響步驟")
    affected_capabilities: list[str] = Field(
        default_factory=list,
        description="受影響能力"
    )

    # 原因分析
    root_cause: str | None = Field(None, description="根本原因")
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="促成因素"
    )

    # 改進建議
    recommendations: list[str] = Field(default_factory=list, description="改進建議")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="記錄時間"
    )


class DeviationAnalysisResult(BaseModel):
    """偏差分析結果"""
    plan_id: str = Field(..., description="計劃ID")
    deviations: list[DeviationRecord] = Field(..., description="偏差列表")
    total_deviations: int = Field(..., description="總偏差數", ge=0)
    is_significant: bool = Field(..., description="是否顯著（需要訓練）")
    significance_score: float = Field(..., description="顯著性分數", ge=0)
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="分析時間"
    )


# ==================== 外部閉環：訓練與優化 ====================

class TrainingDataSample(BaseModel):
    """訓練數據樣本"""
    sample_id: str = Field(..., description="樣本ID")
    plan: ExecutionPlan = Field(..., description="執行計劃")
    trace: list[ExecutionTrace] = Field(..., description="執行軌跡")
    deviations: list[DeviationRecord] = Field(..., description="偏差記錄")
    outcome: Literal["success", "partial_success", "failure"] = Field(
        ...,
        description="結果"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="創建時間"
    )


class ModelTrainingResult(BaseModel):
    """模型訓練結果"""
    training_id: str = Field(..., description="訓練ID")
    samples_used: int = Field(..., description="使用的樣本數", ge=0)
    training_duration: float = Field(..., description="訓練耗時(秒)", ge=0)
    metrics: dict[str, float] = Field(..., description="訓練指標（loss, accuracy 等）")
    new_weights_path: str | None = Field(None, description="新權重文件路徑")
    new_weights_version: str | None = Field(None, description="新權重版本號")
    improvements: dict[str, float] | None = Field(None, description="改進指標")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="訓練時間"
    )


class ExternalLoopProcessResult(BaseModel):
    """外部閉環處理結果"""
    plan_id: str = Field(..., description="計劃ID")
    deviations_found: int = Field(..., description="發現的偏差數", ge=0)
    deviations_significant: bool = Field(..., description="是否顯著")
    deviations: list[DeviationRecord] = Field(..., description="偏差列表")
    training_triggered: bool = Field(..., description="是否觸發訓練")
    training_result: ModelTrainingResult | None = Field(None, description="訓練結果")
    weights_updated: bool = Field(..., description="權重是否更新")
    new_weights_version: str | None = Field(None, description="新權重版本")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="處理時間"
    )
    success: bool = Field(..., description="是否成功")
    error: str | None = Field(None, description="錯誤信息")


# ==================== AICommand 整合 ====================

class DualLoopCommand(BaseModel):
    """雙閉環專用命令（整合到 AICommand 架構）"""
    command_id: str = Field(..., description="命令ID")
    command_type: Literal[
        # 內閉環命令
        "sync_capabilities",  # 同步能力到 RAG
        "query_capability",  # 查詢能力
        "search_solution",  # 搜索問題解法
        "get_usage_example",  # 獲取使用範例
        "report_issue",  # 報告問題

        # 外閉環命令
        "process_execution_result",  # 處理執行結果
        "analyze_deviation",  # 分析偏差
        "trigger_training",  # 觸發訓練
        "update_weights"  # 更新權重
    ] = Field(..., description="命令類型")
    parameters: dict[str, Any] = Field(..., description="命令參數")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="命令時間"
    )
    metadata: dict[str, Any] | None = Field(None, description="元數據")


class DualLoopCommandResult(BaseModel):
    """雙閉環命令執行結果"""
    command_id: str = Field(..., description="命令ID")
    success: bool = Field(..., description="是否成功")
    data: dict[str, Any] | None = Field(None, description="返回數據")
    error: str | None = Field(None, description="錯誤信息")
    execution_time: float = Field(..., description="執行耗時(秒)", ge=0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="完成時間"
    )
