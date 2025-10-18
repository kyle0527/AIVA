"""
AIVA Features - 高價值功能模組

這是 AIVA 的增強功能模組包，包含專門針對 Bug Bounty 和滲透測試設計的
高價值安全檢測功能，重點關注能在實戰中獲得高額獎金的漏洞類型。

傳統模組包含:
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

高價值模組 (新增):
- mass_assignment: Mass Assignment / 權限提升檢測
- jwt_confusion: JWT 混淆攻擊檢測  
- oauth_confusion: OAuth/OIDC 配置錯誤檢測
- graphql_authz: GraphQL 權限缺陷檢測
- ssrf_oob: SSRF with OOB 檢測
- base: 統一的基礎架構和介面

使用前請確保設置 ALLOWLIST_DOMAINS 環境變數以避免意外掃描！

快速開始：
    from services.features.high_value_guide import HighValueFeatureManager
    
    manager = HighValueFeatureManager()
    result = manager.run_mass_assignment_test(
        target="https://app.example.com",
        update_endpoint="/api/profile/update", 
        auth_headers={"Authorization": "Bearer token"}
    )
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
    # 高價值功能模組 (新增)
    "FeatureBase",
    "FeatureRegistry", 
    "SafeHttp",
    "FeatureResult",
    "Finding",
    "FeatureStepExecutor",
    "get_available_features",
    "create_feature_executor",
]

# 導入基礎架構
try:
    from .base import FeatureBase, FeatureRegistry, SafeHttp, FeatureResult, Finding
    from .feature_step_executor import FeatureStepExecutor
    
    # 自動註冊所有高價值功能模組
    def _register_high_value_features():
        """自動註冊所有高價值功能模組"""
        try:
            # 導入以觸發 @FeatureRegistry.register 裝飾器
            from .mass_assignment import worker
            from .jwt_confusion import worker  
            from .oauth_confusion import worker
            from .graphql_authz import worker
            from .ssrf_oob import worker
            
            registered = list(FeatureRegistry.list_features().keys())
            print(f"✅ 已註冊高價值功能模組: {registered}")
            return registered
            
        except ImportError as e:
            print(f"⚠️  部分高價值功能模組導入失敗: {e}")
            return []
    
    # 自動註冊
    _available_features = _register_high_value_features()
    
    def get_available_features():
        """取得所有可用的功能模組列表"""
        return FeatureRegistry.list_features()
    
    def create_feature_executor(**kwargs):
        """創建功能執行器的便利函數"""
        return FeatureStepExecutor(**kwargs)
        
    # 自動註冊高價值功能模組
    def _register_high_value_features():
        """自動註冊所有高價值功能模組"""
        registered = []
        try:
            # Mass Assignment 檢測
            from .mass_assignment.worker import MassAssignmentWorker
            registered.append("mass_assignment")
        except ImportError as e:
            print(f"⚠️  Mass Assignment 模組不可用: {e}")
        
        try:
            # JWT 混淆檢測
            from .jwt_confusion.worker import JwtConfusionWorker
            registered.append("jwt_confusion")
        except ImportError as e:
            print(f"⚠️  JWT Confusion 模組不可用: {e}")
        
        try:
            # OAuth 混淆檢測
            from .oauth_confusion.worker import OAuthConfusionWorker
            registered.append("oauth_confusion")
        except ImportError as e:
            print(f"⚠️  OAuth Confusion 模組不可用: {e}")
        
        try:
            # GraphQL 權限檢測
            from .graphql_authz.worker import GraphQLAuthzWorker
            registered.append("graphql_authz")
        except ImportError as e:
            print(f"⚠️  GraphQL AuthZ 模組不可用: {e}")
        
        try:
            # SSRF OOB 檢測
            from .ssrf_oob.worker import SsrfOobWorker
            registered.append("ssrf_oob")
        except ImportError as e:
            print(f"⚠️  SSRF OOB 模組不可用: {e}")
        
        print(f"✅ 已註冊 {len(registered)} 個高價值功能模組: {', '.join(registered)}")
        return registered
    
    # 執行註冊
    _available_features = _register_high_value_features()

except ImportError as e:
    print(f"⚠️  高價值功能模組不可用: {e}")
    _available_features = []
    
    # 提供空的替代函數
    def get_available_features():
        return {}
    
    def create_feature_executor(**kwargs):
        raise ImportError("高價值功能模組未正確安裝")
