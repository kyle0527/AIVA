"""
AIVA Features - 功能模組

這是 AIVA 的增強功能模組包，符合 aiva_common 規範，包含安全檢測功能。所有模組都支持 CLI 調用。

✅ **已符合 aiva_common 規範的模組**:
- function_xss: XSS 漏洞檢測 (CLI + CommandHandler)
- function_sqli: SQL 注入檢測 (CLI + CommandHandler) 
- function_ssrf: SSRF 漏洞檢測 (CLI + CommandHandler)
- function_forensic: 數字取證分析 (CLI + CommandHandler)
- function_wordlist_generator: 字典生成器 (CLI + CommandHandler)

🔧 **待完善命令系統集成的模組**:
- function_reverse_engineering: 逆向工程分析
- function_steganography: 隱寫術檢測
- function_crypto: 密碼學分析
- function_idor: IDOR 漏洞檢測
- function_postex: 後滲透測試
- function_social_engineering: 社會工程
- function_payload_generator: 載荷生成器
- function_exploit_framework: 漏洞利用框架
- function_bizlogic: 業務邏輯漏洞
- function_web_scanner: Web 掃描器
- function_ddos: DDoS 測試
- function_authn_go: 認證測試 (Go)

🎯 **推薦使用方式（CLI 模式）**:

    # 方式 1: 直接通過 CLI 調用（推薦）
    aiva-cli function_xss_test --target https://example.com --params '{"check_reflected": true}'
    
    # 方式 2: 通過 Python 模組調用
    python -m services.features.function_xss.xss_detector --target https://example.com

📦 **向後兼容（AICommand 模式）**:

    # 仍然支持 AICommand 接口（向後兼容）
    from services.features.function_xss.command_handler import XSSCommandHandler
    from aiva_common.schemas.commands import AICommand, CommandType
    
    xss_handler = XSSCommandHandler()
    command = AICommand(
        command_type=CommandType.FEATURE_XSS_TEST,
        payload={"target_url": "https://example.com"},
        target_module="features.xss"
    )
    result = await xss_handler.handle_command(command)

🔒 **安全注意事項**:
使用前請確保設置適當的目標白名單以避免意外掃描！

架構更新（2026-01-08）：
- ✅ 優先使用 CLI 模式調用（符合 aiva_common 規範）
- ✅ 保留 AICommand 接口作為向後兼容層
- ✅ 所有能力通過 Manifest 註冊和發現
"""

__version__ = "2.0.0"  # 升級到 v2.0，符合 aiva_common 規範

# ==================== 從 aiva_common 導入共享基礎設施 ====================
from aiva_common.enums import (
    Confidence,
    Severity, 
    ThreatLevel,  # 使用統一的 ThreatLevel
    TaskStatus,
    VulnerabilityType,
)
from aiva_common.schemas import (
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
try:
    from aiva_common.schemas.tasks import (
        APISchemaPayload,
        APISecurityTestPayload,
        APITestCase,
    )
    from aiva_common.schemas.base import ExecutionError
    from aiva_common.schemas.telemetry import FunctionTelemetry
except ImportError:
    # 如果相對導入失敗，先跳過這些導入
    pass

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

