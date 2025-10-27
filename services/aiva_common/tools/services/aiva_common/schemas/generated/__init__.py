"""
AIVA Schema 自動生成模組
======================

此模組包含所有由 core_schema_sot.yaml 自動生成的 Schema 定義

⚠️  請勿手動修改此模組中的檔案
🔄  如需更新，請修改 core_schema_sot.yaml 後重新生成
"""

# 基礎類型
from .base_types import *

# 訊息通訊
from .messaging import *

# 任務管理
from .tasks import *

# 發現結果
from .findings import *

__version__ = "1.0.0"
__generated_at__ = "2025-10-27T13:23:51.608788"

__all__ = [
    # 基礎類型
    "MessageHeader",
    "Target", 
    "Vulnerability",
    
    # 訊息通訊
    "AivaMessage",
    "AIVARequest",
    "AIVAResponse",
    
    # 任務管理
    "FunctionTaskPayload",
    "FunctionTaskTarget", 
    "FunctionTaskContext",
    "FunctionTaskTestConfig",
    "ScanTaskPayload",
    
    # 發現結果
    "FindingPayload",
    "FindingEvidence",
    "FindingImpact", 
    "FindingRecommendation",
]