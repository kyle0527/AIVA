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

# ==================== 從 aiva_common 導入共享基礎設施 ====================
from ..aiva_common.enums import (
    Confidence,
    Severity,
    TaskStatus,
    VulnerabilityType,
)
from ..aiva_common.schemas import (
    AuthZAnalysisPayload,
    AuthZCheckPayload,
    AuthZResultPayload,
    CVSSv3Metrics,
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

# ==================== 從 aiva_common 導入 Features 使用的類 ====================
# 注意: 這些類實際定義在 aiva_common.schemas 中，而非本地 models 模組
from ..aiva_common.schemas.tasks import (
    APISchemaPayload,
    APISecurityTestPayload,
    APITestCase,
)
from ..aiva_common.schemas.base import ExecutionError
from ..aiva_common.schemas.telemetry import FunctionTelemetry

# ==================== 本模組專屬類型（待實現或移除） ====================
# 以下類型暫時註釋，保留作為未來功能預留
# from .models import (
#     BizLogicResultPayload,
#     BizLogicTestPayload,
#     EnhancedFunctionTaskTarget,
#     JavaScriptAnalysisResult,
#     PostExResultPayload,
#     PostExTestPayload,
#     SensitiveMatch,
# )

__all__ = [
    # ==================== 來自 aiva_common ====================
    # 枚舉類
    "Confidence",
    "Severity",
    "TaskStatus",
    "VulnerabilityType",
    # 共享 Schema
    "CVSSv3Metrics",
    "FunctionTaskTarget",
    "FunctionTaskContext",
    "FunctionTaskPayload",
    "TestExecution",
    "FunctionExecutionResult",
    "ExploitPayload",
    "ExploitResult",
    "OastEvent",
    "OastProbe",
    "AuthZCheckPayload",
    "AuthZAnalysisPayload",
    "AuthZResultPayload",
    # ==================== Features 使用的類（來自 aiva_common） ====================
    "FunctionTelemetry",
    "ExecutionError",
    "APISchemaPayload",
    "APITestCase",
    "APISecurityTestPayload",
    # ==================== 本模組專屬類（預留，待實現） ====================
    # "EnhancedFunctionTaskTarget",
    # "PostExTestPayload",
    # "PostExResultPayload",
    # "BizLogicTestPayload",
    # "BizLogicResultPayload",
    # "SensitiveMatch",
    # "JavaScriptAnalysisResult",
    # ==================== 高價值功能模組（預留，待實現 base 模組） ====================
    # "FeatureBase",
    # "FeatureRegistry",
    # "SafeHttp",
    # "FeatureResult",
    # "Finding",
    "FeatureStepExecutor",
    "get_available_features",
    "create_feature_executor",
]

# 導入基礎架構
try:
    from .base import FeatureRegistry, FeatureResult, Finding
    from .feature_step_executor import FeatureStepExecutor
    
    _BASE_AVAILABLE = True
except ImportError as e:
    # 如果 base 模組不可用，提供基本功能
    _BASE_AVAILABLE = False
    FeatureStepExecutor = None
    
    def _register_high_value_features() -> list[str]:
        """空實現，當 base 模組不可用時"""
        return []
    
    def get_available_features() -> dict:
        """空實現，當 base 模組不可用時"""
        return {}

if _BASE_AVAILABLE:
    def _register_high_value_features() -> list[str]:
        """
        自動註冊所有高價值功能模組
        
        Returns:
            已註冊的功能模組名稱列表
        """
        from .base import get_global_registry
        
        registry = get_global_registry()
        registered = registry.list_features()
        
        if registered:
            print(f"[OK] 已註冊 {len(registered)} 個高價值功能模組: {', '.join(registered)}")
        
        return registered
    
    # 執行註冊
    _available_features = _register_high_value_features()
    
    def get_available_features() -> dict:
        """
        取得所有可用的功能模組列表
        
        Returns:
            功能名稱到功能類別的映射字典
        """
        from .base import get_global_registry
        return get_global_registry().list_features()
    
    def create_feature_executor(**kwargs):
        """
        創建功能執行器的便利函數
        
        Args:
            **kwargs: 傳遞給 FeatureStepExecutor 的參數
            
        Returns:
            配置好的 FeatureStepExecutor 實例
        """
        if FeatureStepExecutor:
            return FeatureStepExecutor(**kwargs)
        else:
            raise ImportError("FeatureStepExecutor 不可用")
else:
    def create_feature_executor(**kwargs):
        """空實現"""
        raise ImportError("base 模組不可用，無法創建 FeatureStepExecutor")
