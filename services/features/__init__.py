"""
AIVA Features - 功能模組

這是 AIVA 的增強功能模組包，符合 aiva_common 規範，包含安全檢測功能。所有模組都支持 CLI 調用。

✅ **已符合 aiva_common 規範的模組**:
- function_xss: XSS 漏洞檢測 (CLI 驅動，保留 CommandHandler 過渡期)
- function_sqli: SQL 注入檢測 (CLI 驅動)
- function_ssrf: SSRF 漏洞檢測 (CLI 驅動)
- function_forensic: 數字取證分析 (CLI 驅動)
- function_wordlist_generator: 字典生成器 (CLI 驅動)

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
]

# 📝 註記 (2026-02-09)
# base 資料夾已歸檔至 _archive/base_feature_infrastructure/
# 原因：10 個功能模組都未使用 FeatureRegistry、FeatureResult 等標準接口
# 各功能模組使用各自的架構：CommandHandler、純CLI、Detector 等
# FeatureStepExecutor 因無註冊模組而無法使用，已一併歸檔

