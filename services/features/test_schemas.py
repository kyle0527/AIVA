"""
AIVA 功能測試模式定義

包含功能測試、漏洞利用、測試結果等相關的數據模式。
屬於 features 模組的測試輔助定義。

注意：基礎 Schema 類已從 aiva_common 導入，本文件僅保留
features 模組專屬的測試配置類。
"""



from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


from services.aiva_common.schemas import (
    CVSSv3Metrics,
    ExploitPayload,
    ExploitResult,
    FunctionTaskContext,
    FunctionTaskPayload,
    FunctionTaskTarget,
)

# ==================== Features 專屬的測試配置類 ====================


class FunctionTaskTestConfig(BaseModel):
    """
    功能測試配置
    
    Features 模組專屬,用於詳細的測試參數配置。
    與 aiva_common 中的基礎配置互補。
    """

    timeout: int = Field(default=30, ge=1, le=300, description="超時時間(秒)")
    retry_count: int = Field(default=3, ge=0, le=10, description="重試次數")
    delay_between_requests: float = Field(default=1.0, ge=0, description="請求間延遲(秒)")
    follow_redirects: bool = Field(default=True, description="是否跟隨重定向")
    verify_ssl: bool = Field(default=True, description="是否驗證SSL")

# ==================== 測試結果 ====================


class TestResult(BaseModel):
    """
    測試結果
    
    Features 模組專屬,用於詳細的測試執行結果記錄。
    """

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
    stack_trace: str | None = Field(default=None, description="堆棧跟踪")

    # 時間戳
    started_at: datetime = Field(description="開始時間")
    finished_at: datetime = Field(description="結束時間")

    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class ExploitConfiguration(BaseModel):
    """
    漏洞利用配置
    
    Features 模組專屬,定義漏洞利用的詳細配置。
    與 aiva_common.schemas.ExploitPayload 配合使用。
    """

    exploit_type: VulnerabilityType = Field(description="利用類型")
    payload: str = Field(description="攻擊載荷")
    target_parameter: str | None = Field(default=None, description="目標參數")

    # 利用選項
    encode_payload: bool = Field(default=True, description="是否編碼載荷")
    use_proxy: bool = Field(default=False, description="是否使用代理")
    bypass_waf: bool = Field(default=False, description="是否繞過WAF")

    # 驗證配置
    success_indicators: list[str] = Field(default_factory=list, description="成功指標")
    failure_indicators: list[str] = Field(default_factory=list, description="失敗指標")


# ==================== 專用測試類型 ====================


class AuthZTestPayload(BaseModel):
    """
    授權測試載荷
    
    Features 模組專屬,用於授權測試的詳細配置。
    """

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
    """
    API安全測試載荷
    
    Features 模組專屬,用於 API 安全測試配置。
    與 features.models.APISecurityTestPayload 互補。
    """

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
    """
    後滲透測試載荷 - 僅限授權測試環境
    
    Features 模組專屬,用於後滲透測試的安全約束配置。
    與 features.models.PostExTestPayload 互補。
    """

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
    """
    測試統計信息
    
    Features 模組專屬,用於測試執行的統計報告。
    """

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
    # ==================== 從 aiva_common 導入的共享類 ====================
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskPayload",
    "ExploitPayload",
    "ExploitResult",
    "CVSSv3Metrics",
    # ==================== Features 專屬類 ====================
    "FunctionTaskTestConfig",
    "TestResult",
    "ExploitConfiguration",
    "AuthZTestPayload",
    "APISecurityTestPayload",
    "PostExTestPayload",
    "TestStatistics",
]
