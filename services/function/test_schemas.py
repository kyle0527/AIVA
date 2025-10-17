"""
AIVA 功能測試模式定義

包含功能測試、漏洞利用、測試結果等相關的數據模式。
屬於 function 模組的業務特定定義。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.aiva_common.enums import Confidence, ExploitType, Severity, TestStatus
from services.aiva_common.standards import CVSSv3Metrics
from pydantic import BaseModel, Field, HttpUrl, field_validator

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

    task_id: str = Field(description="任務ID", pattern=r"^task_[a-zA-Z0-9_]+$")
    scan_id: str = Field(description="掃描ID", pattern=r"^scan_[a-zA-Z0-9_]+$")
    function_type: str = Field(description="功能類型")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")

    target: FunctionTaskTarget = Field(description="測試目標")
    context: FunctionTaskContext = Field(default_factory=FunctionTaskContext, description="測試上下文")
    config: FunctionTaskTestConfig = Field(default_factory=FunctionTaskTestConfig, description="測試配置")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v.startswith("task_"):
            raise ValueError("task_id must start with 'task_'")
        return v

    @field_validator("scan_id")
    @classmethod
    def validate_scan_id(cls, v: str) -> str:
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("priority must be between 1 and 10")
        return v


# ==================== 測試結果和漏洞利用 ====================


class TestResult(BaseModel):
    """測試結果"""

    task_id: str = Field(description="任務ID")
    test_name: str = Field(description="測試名稱")
    status: TestStatus = Field(description="測試狀態")

    # 結果詳情
    success: bool = Field(description="是否成功")
    message: str | None = Field(default=None, description="結果消息")
    details: dict[str, Any] = Field(default_factory=dict, description="詳細結果")

    # 性能指標
    execution_time: float = Field(description="執行時間(秒)")
    memory_usage: int | None = Field(default=None, description="內存使用(bytes)")

    # 錯誤信息
    error_code: str | None = Field(default=None, description="錯誤代碼")
    error_message: str | None = Field(default=None, description="錯誤消息")
    stack_trace: str | None = Field(default=None, description="堆棧跟[U+8E2A]")

    # 時間戳
    started_at: datetime = Field(description="開始時間")
    finished_at: datetime = Field(description="結束時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ExploitConfiguration(BaseModel):
    """漏洞利用配置"""

    exploit_type: ExploitType = Field(description="利用類型")
    payload: str = Field(description="攻擊載荷")
    target_parameter: str | None = Field(default=None, description="目標參數")

    # 利用選項
    encode_payload: bool = Field(default=True, description="是否編碼載荷")
    use_proxy: bool = Field(default=False, description="是否使用代理")
    bypass_waf: bool = Field(default=False, description="是否繞過WAF")

    # 驗證配置
    success_indicators: list[str] = Field(default_factory=list, description="成功指標")
    failure_indicators: list[str] = Field(default_factory=list, description="失敗指標")


class ExploitResult(BaseModel):
    """漏洞利用結果"""

    exploit_id: str = Field(description="利用ID", pattern=r"^exploit_[a-zA-Z0-9_]+$")
    task_id: str = Field(description="關聯任務ID")
    target_url: str = Field(description="目標URL")

    # 利用詳情
    exploit_type: ExploitType = Field(description="利用類型")
    payload_used: str = Field(description="使用的載荷")
    success: bool = Field(description="是否成功")

    # 結果數據
    response_data: str | None = Field(default=None, description="響應數據")
    extracted_data: dict[str, Any] = Field(default_factory=dict, description="提取的數據")

    # 影響評估
    severity: Severity = Field(description="嚴重程度")
    confidence: Confidence = Field(description="置信度")
    cvss_score: float | None = Field(default=None, ge=0.0, le=10.0, description="CVSS分數")
    cvss_metrics: CVSSv3Metrics | None = Field(default=None, description="CVSS指標")

    # 證據
    request_evidence: str | None = Field(default=None, description="請求證據")
    response_evidence: str | None = Field(default=None, description="響應證據")
    screenshot: str | None = Field(default=None, description="截圖(base64)")

    # 時間戳
    exploited_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 專用測試類型 ====================


class AuthZTestPayload(BaseModel):
    """授權測試載荷"""

    task_id: str = Field(description="任務ID")
    target_url: str = Field(description="目標URL")

    # 測試配置
    test_type: str = Field(description="測試類型")  # "horizontal", "vertical", "function_level"
    user_roles: list[str] = Field(description="測試用戶角色")
    test_endpoints: list[str] = Field(description="測試端點")

    # 認證信息
    auth_tokens: dict[str, str] = Field(default_factory=dict, description="認證令牌")
    session_cookies: dict[str, str] = Field(default_factory=dict, description="會話Cookie")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class APISecurityTestPayload(BaseModel):
    """API安全測試載荷"""

    task_id: str = Field(description="任務ID")
    api_endpoint: str = Field(description="API端點")
    api_specification: dict[str, Any] | None = Field(default=None, description="API規範")

    # 測試類型
    test_categories: list[str] = Field(description="測試分類")
    # ["authentication", "authorization", "input_validation", "rate_limiting", "injection"]

    # API配置
    api_key: str | None = Field(default=None, description="API密鑰")
    oauth_token: str | None = Field(default=None, description="OAuth令牌")

    # 測試參數
    fuzz_parameters: bool = Field(default=True, description="是否模糊測試參數")
    test_business_logic: bool = Field(default=True, description="是否測試業務邏輯")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class PostExTestPayload(BaseModel):
    """後滲透測試載荷 - 僅限授權測試環境"""

    task_id: str = Field(description="任務ID")
    target_system: str = Field(description="目標系統")

    # 測試範圍
    test_scope: list[str] = Field(description="測試範圍")
    # ["privilege_escalation", "lateral_movement", "persistence", "data_exfiltration"]

    # 安全約束
    authorized_only: bool = Field(default=True, description="僅授權環境")
    data_protection: bool = Field(default=True, description="數據保護")
    cleanup_required: bool = Field(default=True, description="需要清理")

    # 憑據信息
    initial_access: dict[str, Any] = Field(description="初始訪問信息")
    target_accounts: list[str] = Field(default_factory=list, description="目標帳戶")

    # 限制條件
    time_limit: int = Field(default=3600, description="時間限制(秒)")
    network_restrictions: list[str] = Field(default_factory=list, description="網路限制")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 測試統計和報告 ====================


class TestStatistics(BaseModel):
    """測試統計信息"""

    total_tests: int = Field(description="總測試數")
    passed_tests: int = Field(description="通過測試數")
    failed_tests: int = Field(description="失敗測試數")
    skipped_tests: int = Field(description="跳過測試數")

    # 漏洞統計
    critical_vulnerabilities: int = Field(default=0, description="嚴重漏洞數")
    high_vulnerabilities: int = Field(default=0, description="高危漏洞數")
    medium_vulnerabilities: int = Field(default=0, description="中危漏洞數")
    low_vulnerabilities: int = Field(default=0, description="低危漏洞數")

    # 性能指標
    total_execution_time: float = Field(description="總執行時間(秒)")
    average_response_time: float = Field(description="平均響應時間(秒)")

    # 時間戳
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "FunctionTaskPayload",
    "TestResult",
    "ExploitConfiguration",
    "ExploitResult",
    "AuthZTestPayload",
    "APISecurityTestPayload",
    "PostExTestPayload",
    "TestStatistics",
]
