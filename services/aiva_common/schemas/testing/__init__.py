"""
AIVA Testing Domain Schemas
===========================

測試執行領域模型，包含：
- API安全測試
- 任務執行框架
- 測試場景與策略

此領域專注於各類安全測試的執行和管理。
"""

from .api_testing import *
from .tasks import *
from .scenarios import *

__all__ = [
    # API測試 (api_standards.py的測試部分)
    "APISecurityTest",
    "APIVulnerabilityFinding", 
    # 任務執行 (tasks.py)
    "ScanStartPayload",
    "ScanCompletedPayload",
    "FunctionTaskPayload", 
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "FeedbackEventPayload",
    "TaskUpdatePayload",
    "ConfigUpdatePayload",
    "ThreatIntelLookupPayload",
    "ThreatIntelResultPayload",
    "AuthZCheckPayload", 
    "AuthZAnalysisPayload",
    "AuthZResultPayload",
    "RemediationGeneratePayload",
    "RemediationResultPayload",
    "PostExTestPayload",
    "PostExResultPayload",
    "BizLogicTestPayload",
    "BizLogicResultPayload",
    "APISchemaPayload", 
    "APITestCase",
    "APISecurityTestPayload",
    "EASMDiscoveryPayload",
    "EASMDiscoveryResult", 
    "ScenarioTestResult",
    "ExploitPayload",
    "TestExecution",
    "ExploitResult",
    "TestStrategy",
]