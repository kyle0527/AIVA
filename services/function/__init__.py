"""
AIVA Function - 功能模組

這是 AIVA 的功能模組包，包含各種漏洞檢測和測試功能。

模組包含:
- function_sqli: SQL 注入檢測
- function_xss: XSS 漏洞檢測
- function_ssrf: SSRF 漏洞檢測
- function_idor: IDOR 漏洞檢測
- function_sast_rust: 靜態代碼分析 (Rust)
- function_sca_go: 軟件成分分析 (Go)
- function_authn_go: 認證測試 (Go)
- function_crypto_go: 加密測試 (Go)
- function_cspm_go: 雲安全態勢管理 (Go)
- function_postex: 後滲透測試
- common: 通用工具和設施
"""

__version__ = "1.0.0"

# 從 aiva_common 導入共享基礎設施
from ..aiva_common.enums import (
    Confidence,
    Severity,
    TaskStatus,
    VulnerabilityType,
)
from ..aiva_common.schemas import CVSSv3Metrics

# 從本模組導入功能測試相關模型
from .models import (
    APISchemaPayload,
    APISecurityTestPayload,
    APITestCase,
    AuthZAnalysisPayload,
    AuthZCheckPayload,
    AuthZResultPayload,
    BizLogicResultPayload,
    BizLogicTestPayload,
    EnhancedFunctionTaskTarget,
    ExecutionError,
    ExploitPayload,
    ExploitResult,
    FunctionExecutionResult,
    FunctionTaskContext,
    FunctionTaskPayload,
    FunctionTaskTarget,
    FunctionTaskTestConfig,
    FunctionTelemetry,
    JavaScriptAnalysisResult,
    OastEvent,
    OastProbe,
    PostExResultPayload,
    PostExTestPayload,
    SensitiveMatch,
    TestExecution,
)

__all__ = [
    # 來自 aiva_common
    "Confidence",
    "CVSSv3Metrics",
    "Severity",
    "TaskStatus",
    "VulnerabilityType",
    # 來自本模組
    "APISchemaPayload",
    "APISecurityTestPayload",
    "APITestCase",
    "AuthZAnalysisPayload",
    "AuthZCheckPayload",
    "AuthZResultPayload",
    "BizLogicResultPayload",
    "BizLogicTestPayload",
    "EnhancedFunctionTaskTarget",
    "ExecutionError",
    "ExploitPayload",
    "ExploitResult",
    "FunctionExecutionResult",
    "FunctionTaskContext",
    "FunctionTaskPayload",
    "FunctionTaskTarget",
    "FunctionTaskTestConfig",
    "FunctionTelemetry",
    "JavaScriptAnalysisResult",
    "OastEvent",
    "OastProbe",
    "PostExResultPayload",
    "PostExTestPayload",
    "SensitiveMatch",
    "TestExecution",
]
