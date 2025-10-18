"""
AIVA Function Models - 功能測試模組

此文件包含與主動功能測試、漏洞驗證、漏洞利用、POC執行相關的所有數據模型。

職責範圍：
1. 功能測試任務配置 (FunctionTaskTarget, FunctionTaskContext, FunctionTaskPayload)
2. 測試執行和結果 (TestExecution, FunctionExecutionResult)
3. 漏洞利用載荷和結果 (ExploitPayload, ExploitResult)
4. 後滲透測試 (PostExTestPayload, PostExResultPayload)
5. API安全測試 (APISchemaPayload, APITestCase)
6. 專項測試 (BizLogic, AuthZ, OAST)
7. 敏感數據檢測和JS分析
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

from ..aiva_common.enums import (
    Confidence,
    PostExTestType,
    Severity,
    TaskStatus,
)

# ==================== 功能測試任務 ====================


class FunctionTaskTarget(BaseModel):
    """功能測試目標"""

    url: HttpUrl = Field(description="目標URL")
    method: str = Field(default="GET", description="HTTP方法")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP標頭")
    cookies: dict[str, str] = Field(default_factory=dict, description="Cookie")
    parameters: dict[str, str] = Field(default_factory=dict, description="參數")
    body: str | None = Field(default=None, description="請求體")
    auth_required: bool = Field(default=False, description="是否需要認證")


class FunctionTaskContext(BaseModel):
    """功能測試上下文"""

    session_data: dict[str, Any] = Field(default_factory=dict, description="會話數據")
    cookies: dict[str, str] = Field(default_factory=dict, description="Cookie數據")
    previous_results: list[str] = Field(default_factory=list, description="前序結果")


class FunctionTaskTestConfig(BaseModel):
    """功能測試配置"""

    timeout: int = Field(default=30, ge=1, le=300, description="超時時間(秒)")
    retry_count: int = Field(default=3, ge=0, le=10, description="重試次數")
    delay_between_requests: float = Field(default=1.0, ge=0, description="請求間延遲(秒)")
    follow_redirects: bool = Field(default=True, description="是否跟隨重定向")
    verify_ssl: bool = Field(default=True, description="是否驗證SSL")


class FunctionTaskPayload(BaseModel):
    """功能測試任務載荷"""

    task_id: str = Field(description="任務ID")
    function_name: str = Field(description="功能名稱")  # e.g., "xss", "sqli", "ssrf"
    target: FunctionTaskTarget = Field(description="測試目標")
    context: FunctionTaskContext = Field(default_factory=FunctionTaskContext, description="測試上下文")
    config: FunctionTaskTestConfig = Field(
        default_factory=FunctionTaskTestConfig, description="測試配置"
    )
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v

    @field_validator("function_name")
    @classmethod
    def validate_function_name(cls, v: str) -> str:
        allowed = {
            "xss",
            "sqli",
            "ssrf",
            "xxe",
            "idor",
            "csrf",
            "lfi",
            "rfi",
            "cmdi",
            "deserialization",
            "postex",
        }
        if v not in allowed:
            raise ValueError(f"function_name must be one of {allowed}")
        return v


class EnhancedFunctionTaskTarget(BaseModel):
    """增強功能測試目標"""

    url: HttpUrl = Field(description="目標URL")
    method: str = Field(default="GET", description="HTTP方法")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP標頭")
    cookies: dict[str, str] = Field(default_factory=dict, description="Cookie")
    parameters: dict[str, str] = Field(default_factory=dict, description="參數")
    body: str | None = Field(default=None, description="請求體")
    auth_required: bool = Field(default=False, description="是否需要認證")


# ==================== 測試執行和結果 ====================


class FunctionTelemetry(BaseModel):
    """功能測試遙測數據"""

    function_name: str = Field(description="功能名稱")
    execution_time: float = Field(ge=0, description="執行時間(秒)")
    requests_sent: int = Field(ge=0, description="發送請求數")
    vulnerabilities_found: int = Field(ge=0, description="發現漏洞數")
    errors_encountered: int = Field(ge=0, description="遇到錯誤數")
    cpu_usage: float | None = Field(default=None, description="CPU使用率")
    memory_usage: int | None = Field(default=None, description="內存使用(MB)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="其他遙測數據")


class ExecutionError(BaseModel):
    """執行錯誤"""

    error_type: str = Field(description="錯誤類型")
    error_message: str = Field(description="錯誤消息")
    stack_trace: str | None = Field(default=None, description="堆棧跟蹤")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recoverable: bool = Field(default=True, description="是否可恢復")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class FunctionExecutionResult(BaseModel):
    """功能執行結果"""

    task_id: str = Field(description="任務ID")
    function_name: str = Field(description="功能名稱")
    status: TaskStatus = Field(description="執行狀態")
    success: bool = Field(description="是否成功")
    vulnerabilities: list[dict[str, Any]] = Field(default_factory=list, description="發現的漏洞")
    telemetry: FunctionTelemetry = Field(description="遙測數據")
    errors: list[ExecutionError] = Field(default_factory=list, description="錯誤列表")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class TestExecution(BaseModel):
    """測試執行記錄"""

    execution_id: str = Field(description="執行ID")
    test_case_id: str = Field(description="測試案例ID")
    target_url: str = Field(description="目標URL")

    # 執行配置
    timeout: int = Field(default=30, ge=1, description="超時時間(秒)")
    retry_attempts: int = Field(default=3, ge=0, description="重試次數")

    # 執行狀態
    status: TaskStatus = Field(description="執行狀態")
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = Field(default=None, description="結束時間")
    duration: float | None = Field(default=None, ge=0.0, description="執行時間(秒)")

    # 執行結果
    success: bool = Field(description="是否成功")
    vulnerability_found: bool = Field(description="是否發現漏洞")
    confidence_level: Confidence = Field(description="結果置信度")

    # 詳細信息
    request_data: dict[str, Any] = Field(default_factory=dict, description="請求數據")
    response_data: dict[str, Any] = Field(default_factory=dict, description="響應數據")
    evidence: list[str] = Field(default_factory=list, description="證據列表")
    error_message: str | None = Field(default=None, description="錯誤消息")

    # 資源使用
    cpu_usage: float | None = Field(default=None, description="CPU使用率")
    memory_usage: int | None = Field(default=None, description="內存使用(MB)")
    network_traffic: int | None = Field(default=None, description="網絡流量(bytes)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 漏洞利用 ====================


class ExploitPayload(BaseModel):
    """漏洞利用載荷"""

    payload_id: str = Field(description="載荷ID")
    payload_type: str = Field(description="載荷類型")  # "xss", "sqli", "command_injection"
    payload_content: str = Field(description="載荷內容")

    # 載荷屬性
    encoding: str = Field(default="none", description="編碼方式")
    obfuscation: bool = Field(default=False, description="是否混淆")
    bypass_technique: str | None = Field(default=None, description="繞過技術")

    # 適用條件
    target_technology: list[str] = Field(default_factory=list, description="目標技術")
    required_context: dict[str, Any] = Field(default_factory=dict, description="所需上下文")

    # 效果評估
    effectiveness_score: float = Field(ge=0.0, le=1.0, description="效果評分")
    detection_evasion: float = Field(ge=0.0, le=1.0, description="逃避檢測能力")

    # 使用統計
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")
    usage_count: int = Field(ge=0, description="使用次數")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ExploitResult(BaseModel):
    """漏洞利用結果"""

    result_id: str = Field(description="結果ID")
    exploit_id: str = Field(description="利用ID")
    target_id: str = Field(description="目標ID")

    # 利用狀態
    success: bool = Field(description="利用是否成功")
    severity: Severity = Field(description="嚴重程度")
    impact_level: str = Field(description="影響級別")  # "critical", "high", "medium", "low"

    # 利用細節
    exploit_technique: str = Field(description="利用技術")
    payload_used: str = Field(description="使用的載荷")
    execution_time: float = Field(ge=0.0, description="執行時間(秒)")

    # 獲得的訪問
    access_gained: dict[str, Any] = Field(default_factory=dict, description="獲得的訪問權限")
    data_extracted: list[str] = Field(default_factory=list, description="提取的數據")
    system_impact: str | None = Field(default=None, description="系統影響")

    # 檢測規避
    detection_bypassed: bool = Field(description="是否繞過檢測")
    artifacts_left: list[str] = Field(default_factory=list, description="留下的痕跡")

    # 修復驗證
    remediation_verified: bool = Field(default=False, description="修復是否已驗證")
    retest_required: bool = Field(default=True, description="是否需要重測")

    # 時間戳
    executed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 後滲透測試 ====================


class PostExTestPayload(BaseModel):
    """後滲透測試載荷"""

    test_id: str = Field(description="測試ID")
    test_type: PostExTestType = Field(description="測試類型")
    target_system: str = Field(description="目標系統")
    access_level: str = Field(description="訪問級別")
    credentials: dict[str, str] = Field(default_factory=dict, description="憑據信息")
    test_config: dict[str, Any] = Field(default_factory=dict, description="測試配置")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class PostExResultPayload(BaseModel):
    """後滲透測試結果載荷"""

    test_id: str = Field(description="測試ID")
    test_type: PostExTestType = Field(description="測試類型")
    success: bool = Field(description="是否成功")
    findings: list[dict[str, Any]] = Field(default_factory=list, description="發現列表")
    data_collected: dict[str, Any] = Field(default_factory=dict, description="收集的數據")
    privilege_escalation: bool = Field(default=False, description="是否提權成功")
    lateral_movement: bool = Field(default=False, description="是否橫向移動")
    persistence_achieved: bool = Field(default=False, description="是否建立持久化")
    execution_time: float = Field(ge=0, description="執行時間(秒)")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== API 安全測試 ====================


class APISchemaPayload(BaseModel):
    """API Schema載荷"""

    api_id: str = Field(description="API ID")
    base_url: HttpUrl = Field(description="API基礎URL")
    schema_type: str = Field(description="Schema類型")  # "openapi", "swagger", "graphql", "rest"
    schema_content: dict[str, Any] = Field(description="Schema內容")
    authentication: dict[str, Any] = Field(default_factory=dict, description="認證配置")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class APITestCase(BaseModel):
    """API測試案例"""

    test_case_id: str = Field(description="測試案例ID")
    endpoint: str = Field(description="API端點")
    method: str = Field(description="HTTP方法")
    test_type: str = Field(description="測試類型")
    parameters: dict[str, Any] = Field(default_factory=dict, description="測試參數")
    expected_result: dict[str, Any] = Field(default_factory=dict, description="預期結果")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class APISecurityTestPayload(BaseModel):
    """API安全測試載荷"""

    test_id: str = Field(description="測試ID")
    api_id: str = Field(description="API ID")
    test_cases: list[APITestCase] = Field(description="測試案例列表")
    test_config: dict[str, Any] = Field(default_factory=dict, description="測試配置")
    authentication: dict[str, Any] = Field(default_factory=dict, description="認證信息")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== OAST (帶外應用安全測試) ====================


class OastEvent(BaseModel):
    """OAST事件"""

    event_id: str = Field(description="事件ID")
    probe_id: str = Field(description="探針ID")
    event_type: str = Field(description="事件類型")  # "dns", "http", "smtp"
    source_ip: str = Field(description="來源IP")
    payload: dict[str, Any] = Field(description="事件載荷")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class OastProbe(BaseModel):
    """OAST探針"""

    probe_id: str = Field(description="探針ID")
    probe_type: str = Field(description="探針類型")  # "dns", "http", "smtp"
    unique_identifier: str = Field(description="唯一標識符")
    callback_url: HttpUrl | None = Field(default=None, description="回調URL")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = Field(default=None, description="過期時間")
    events_received: list[str] = Field(default_factory=list, description="接收到的事件ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 業務邏輯和授權測試 ====================


class BizLogicTestPayload(BaseModel):
    """業務邏輯測試載荷"""

    test_id: str = Field(description="測試ID")
    business_flow: str = Field(description="業務流程名稱")
    test_scenarios: list[dict[str, Any]] = Field(description="測試場景列表")
    workflow_steps: list[dict[str, Any]] = Field(description="工作流步驟")
    expected_behavior: dict[str, Any] = Field(description="預期行為")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class BizLogicResultPayload(BaseModel):
    """業務邏輯測試結果載荷"""

    test_id: str = Field(description="測試ID")
    business_flow: str = Field(description="業務流程名稱")
    vulnerabilities_found: list[dict[str, Any]] = Field(default_factory=list, description="發現的漏洞")
    workflow_violations: list[str] = Field(default_factory=list, description="工作流違規")
    bypass_detected: bool = Field(description="是否檢測到繞過")
    severity: Severity = Field(description="嚴重程度")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AuthZCheckPayload(BaseModel):
    """授權檢查載荷"""

    check_id: str = Field(description="檢查ID")
    user_context: dict[str, Any] = Field(description="用戶上下文")
    resource: str = Field(description="資源標識")
    action: str = Field(description="操作")
    expected_permission: bool = Field(description="預期權限")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AuthZAnalysisPayload(BaseModel):
    """授權分析載荷"""

    analysis_id: str = Field(description="分析ID")
    target_application: str = Field(description="目標應用")
    user_roles: list[str] = Field(description="用戶角色列表")
    resources: list[str] = Field(description="資源列表")
    test_matrix: list[dict[str, Any]] = Field(description="測試矩陣")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class AuthZResultPayload(BaseModel):
    """授權結果載荷"""

    analysis_id: str = Field(description="分析ID")
    vulnerabilities: list[dict[str, Any]] = Field(default_factory=list, description="授權漏洞")
    privilege_escalation_paths: list[dict] = Field(default_factory=list, description="提權路徑")
    unauthorized_access: list[dict] = Field(default_factory=list, description="未授權訪問")
    severity: Severity = Field(description="總體嚴重程度")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 敏感數據檢測 ====================


class SensitiveMatch(BaseModel):
    """敏感數據匹配"""

    match_type: str = Field(description="匹配類型")  # "credit_card", "ssn", "api_key", "password"
    matched_value: str = Field(description="匹配值(脫敏)")
    location: str = Field(description="發現位置")
    context: str = Field(description="上下文")
    confidence: Confidence = Field(description="置信度")
    severity: Severity = Field(description="嚴重程度")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class JavaScriptAnalysisResult(BaseModel):
    """JavaScript分析結果"""

    file_url: HttpUrl = Field(description="文件URL")
    file_size: int = Field(ge=0, description="文件大小(bytes)")
    analysis_type: str = Field(description="分析類型")
    sensitive_data: list[SensitiveMatch] = Field(default_factory=list, description="敏感數據")
    api_endpoints: list[str] = Field(default_factory=list, description="API端點")
    hidden_parameters: list[str] = Field(default_factory=list, description="隱藏參數")
    security_headers: dict[str, Any] = Field(default_factory=dict, description="安全標頭")
    vulnerabilities: list[dict[str, Any]] = Field(default_factory=list, description="發現的漏洞")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


__all__ = [
    # 功能測試任務
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "FunctionTaskPayload",
    "EnhancedFunctionTaskTarget",
    # 測試執行和結果
    "FunctionTelemetry",
    "ExecutionError",
    "FunctionExecutionResult",
    "TestExecution",
    # 漏洞利用
    "ExploitPayload",
    "ExploitResult",
    # 後滲透測試
    "PostExTestPayload",
    "PostExResultPayload",
    # API 安全測試
    "APISchemaPayload",
    "APITestCase",
    "APISecurityTestPayload",
    # OAST
    "OastEvent",
    "OastProbe",
    # 業務邏輯和授權
    "BizLogicTestPayload",
    "BizLogicResultPayload",
    "AuthZCheckPayload",
    "AuthZAnalysisPayload",
    "AuthZResultPayload",
    # 敏感數據檢測
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
]
