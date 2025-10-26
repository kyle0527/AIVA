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

# ==================== 從本模組導入 Features 專屬類 ====================
from .models import (
    APISchemaPayload,
    APISecurityTestPayload,
    APITestCase,
    BizLogicResultPayload,
    BizLogicTestPayload,
    EnhancedFunctionTaskTarget,
    ExecutionError,
    FunctionTelemetry,
    JavaScriptAnalysisResult,
    PostExResultPayload,
    PostExTestPayload,
    SensitiveMatch,
)

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
    # ==================== 來自本模組 (Features 專屬) ====================
    "EnhancedFunctionTaskTarget",
    "FunctionTelemetry",
    "ExecutionError",
    "PostExTestPayload",
    "PostExResultPayload",
    "APISchemaPayload",
    "APITestCase",
    "APISecurityTestPayload",
    "BizLogicTestPayload",
    "BizLogicResultPayload",
    "SensitiveMatch",
    "JavaScriptAnalysisResult",
    # ==================== 高價值功能模組 ====================
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
    
    def _register_high_value_features() -> list[str]:
        """
        自動註冊所有高價值功能模組
        
        遵循 README 規範：
        - ✅ 使用明確的類導入（不使用 import worker）
        - ✅ 移除過度的 ImportError 處理
        - ✅ 確保依賴可用而非使用 fallback
        
        Returns:
            已註冊的功能模組名稱列表
            
        Raises:
            ImportError: 當必要的功能模組無法導入時
        """
        registered: list[str] = []
        
        # 按照 README 建議，使用明確的類導入
        # 如果導入失敗，應該讓錯誤明確顯示而非靜默處理
        from .mass_assignment.worker import MassAssignmentWorker
        from .jwt_confusion.worker import JwtConfusionWorker
        from .oauth_confusion.worker import OAuthConfusionWorker
        from .graphql_authz.worker import GraphQLAuthzWorker
        from .ssrf_oob.worker import SsrfOobWorker
        
        # 從 FeatureRegistry 獲取實際註冊的功能列表
        registered = list(FeatureRegistry.list_features().keys())
        print(f"[OK] 已註冊 {len(registered)} 個高價值功能模組: {', '.join(registered)}")
        
        return registered
    
    # 執行註冊
    _available_features = _register_high_value_features()
    
    def get_available_features() -> dict[str, type[FeatureBase]]:
        """
        取得所有可用的功能模組列表
        
        Returns:
            功能名稱到功能類別的映射字典
        """
        return FeatureRegistry.list_features()
    
    def create_feature_executor(**kwargs) -> FeatureStepExecutor:
        """
        創建功能執行器的便利函數
        
        Args:
            **kwargs: 傳遞給 FeatureStepExecutor 的參數
            
        Returns:
            配置好的 FeatureStepExecutor 實例
        """
        return FeatureStepExecutor(**kwargs)

except ImportError as e:
    # 遵循 README 原則：不使用 fallback，讓錯誤明確顯示
    import sys
    print(f"[FAIL] 高價值功能模組導入失敗: {e}", file=sys.stderr)
    print(f"   請確保 aiva_common 和所有依賴已正確安裝", file=sys.stderr)
    raise  # 重新拋出異常，不要靜默處理
