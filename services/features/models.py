"""
AIVA Function Models - 功能測試模組

此文件包含與主動功能測試、漏洞驗證、漏洞利用、POC執行相關的所有數據模型。

職責範圍：
1. Features 專屬的增強配置類 (EnhancedFunctionTaskTarget)
2. 後滲透測試 (PostExTestPayload, PostExResultPayload)
3. API安全測試 (APISchemaPayload, APITestCase)
4. 專項測試 (BizLogic, AuthZ)
5. 敏感數據檢測和JS分析

共享類導入自 aiva_common:
- FunctionTaskTarget, FunctionTaskContext, FunctionTaskPayload
- TestExecution, FunctionExecutionResult
- ExploitPayload, ExploitResult
- OastEvent, OastProbe
- AuthZCheckPayload, AuthZAnalysisPayload, AuthZResultPayload
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
from ..aiva_common.schemas import (
    AuthZAnalysisPayload,
    AuthZCheckPayload,
    AuthZResultPayload,
    ExploitPayload,
    ExploitResult,
    FunctionExecutionResult,
    FunctionTaskContext,
    FunctionTaskPayload,
    FunctionTaskTarget,
    OastEvent,
    OastProbe,
    TestExecution,
)

# ==================== Features 專屬的增強類 ====================

class EnhancedFunctionTaskTarget(FunctionTaskTarget):
    """
    增強功能測試目標
    
    Features 模組專屬的擴展類,添加額外的測試配置。
    基礎功能來自 aiva_common.schemas.FunctionTaskTarget。
    """

    pass  # 當前無額外欄位,保留作為未來擴展點


# ==================== 功能測試遙測數據 ====================

class FunctionTelemetry(BaseModel):
    """
    功能測試遙測數據
    
    Features 模組專屬的執行遙測統計。
    與 aiva_common.schemas.FunctionExecutionResult 互補使用。
    """

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
    """
    執行錯誤
    
    Features 模組專屬的錯誤追蹤模型。
    """

    error_type: str = Field(description="錯誤類型")
    error_message: str = Field(description="錯誤消息")
    stack_trace: str | None = Field(default=None, description="堆棧跟蹤")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recoverable: bool = Field(default=True, description="是否可恢復")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 後滲透測試 ====================

class PostExTestPayload(BaseModel):
    """
    後滲透測試載荷
    
    Features 模組專屬,用於後滲透階段的測試配置。
    """

    test_id: str = Field(description="測試ID")
    test_type: PostExTestType = Field(description="測試類型")
    target_system: str = Field(description="目標系統")
    access_level: str = Field(description="訪問級別")
    credentials: dict[str, str] = Field(default_factory=dict, description="憑據信息")
    test_config: dict[str, Any] = Field(default_factory=dict, description="測試配置")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class PostExResultPayload(BaseModel):
    """
    後滲透測試結果載荷
    
    Features 模組專屬,用於報告後滲透測試的結果。
    """

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
    """
    API Schema載荷
    
    Features 模組專屬,用於 API 測試的 Schema 定義。
    """

    api_id: str = Field(description="API ID")
    base_url: HttpUrl = Field(description="API基礎URL")
    schema_type: str = Field(description="Schema類型")  # "openapi", "swagger", "graphql", "rest"
    schema_content: dict[str, Any] = Field(description="Schema內容")
    authentication: dict[str, Any] = Field(default_factory=dict, description="認證配置")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class APITestCase(BaseModel):
    """
    API測試案例
    
    Features 模組專屬,定義單個 API 端點的測試配置。
    """

    test_case_id: str = Field(description="測試案例ID")
    endpoint: str = Field(description="API端點")
    method: str = Field(description="HTTP方法")
    test_type: str = Field(description="測試類型")
    parameters: dict[str, Any] = Field(default_factory=dict, description="測試參數")
    expected_result: dict[str, Any] = Field(default_factory=dict, description="預期結果")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class APISecurityTestPayload(BaseModel):
    """
    API安全測試載荷
    
    Features 模組專屬,用於執行 API 安全測試。
    """

    test_id: str = Field(description="測試ID")
    api_id: str = Field(description="API ID")
    test_cases: list[APITestCase] = Field(description="測試案例列表")
    test_config: dict[str, Any] = Field(default_factory=dict, description="測試配置")
    authentication: dict[str, Any] = Field(default_factory=dict, description="認證信息")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 業務邏輯測試 ====================


class BizLogicTestPayload(BaseModel):
    """
    業務邏輯測試載荷
    
    Features 模組專屬,用於測試業務流程的邏輯缺陷。
    """

    test_id: str = Field(description="測試ID")
    business_flow: str = Field(description="業務流程名稱")
    test_scenarios: list[dict[str, Any]] = Field(description="測試場景列表")
    workflow_steps: list[dict[str, Any]] = Field(description="工作流步驟")
    expected_behavior: dict[str, Any] = Field(description="預期行為")
    priority: int = Field(default=5, ge=1, le=10, description="優先級")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class BizLogicResultPayload(BaseModel):
    """
    業務邏輯測試結果載荷
    
    Features 模組專屬,報告業務邏輯測試的發現。
    """

    test_id: str = Field(description="測試ID")
    business_flow: str = Field(description="業務流程名稱")
    vulnerabilities_found: list[dict[str, Any]] = Field(default_factory=list, description="發現的漏洞")
    workflow_violations: list[str] = Field(default_factory=list, description="工作流違規")
    bypass_detected: bool = Field(description="是否檢測到繞過")
    severity: Severity = Field(description="嚴重程度")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


# ==================== 敏感數據檢測 ====================

class SensitiveMatch(BaseModel):
    """
    敏感數據匹配
    
    Features 模組專屬,用於報告發現的敏感數據。
    """

    match_type: str = Field(description="匹配類型")  # "credit_card", "ssn", "api_key", "password"
    matched_value: str = Field(description="匹配值(脫敏)")
    location: str = Field(description="發現位置")
    context: str = Field(description="上下文")
    confidence: Confidence = Field(description="置信度")
    severity: Severity = Field(description="嚴重程度")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元數據")


class JavaScriptAnalysisResult(BaseModel):
    """
    JavaScript分析結果
    
    Features 模組專屬,用於報告前端代碼分析的結果。
    """

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
    # ==================== 從 aiva_common 導入的共享類 ====================
    # 功能測試任務 (來自 aiva_common.schemas.tasks)
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskPayload",
    # 測試執行 (來自 aiva_common.schemas.tasks)
    "TestExecution",
    # 執行結果 (來自 aiva_common.schemas.telemetry)
    "FunctionExecutionResult",
    # 漏洞利用 (來自 aiva_common.schemas.tasks)
    "ExploitPayload",
    "ExploitResult",
    # OAST (來自 aiva_common.schemas.telemetry)
    "OastEvent",
    "OastProbe",
    # 授權測試 (來自 aiva_common.schemas.tasks)
    "AuthZCheckPayload",
    "AuthZAnalysisPayload",
    "AuthZResultPayload",
    # ==================== Features 模組專屬類 ====================
    # 增強配置
    "EnhancedFunctionTaskTarget",
    # 遙測數據
    "FunctionTelemetry",
    "ExecutionError",
    # 後滲透測試
    "PostExTestPayload",
    "PostExResultPayload",
    # API 安全測試
    "APISchemaPayload",
    "APITestCase",
    "APISecurityTestPayload",
    # 業務邏輯測試
    "BizLogicTestPayload",
    "BizLogicResultPayload",
    # 敏感數據檢測
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
]
