"""
AIVA Scan Models - 掃描協調器模組數據模型

遵循 aiva_common 規範：
1. 優先使用 aiva_common 的標準 Schema
2. 禁止重複定義，遵循單一數據來源原則  
3. 只在 aiva_common 沒有的情況下才定義新的模型
4. 所有新模型都要有明確的業務場景和必要性說明

職責範圍：
- 只定義掃描協調器特有的、aiva_common 中不存在的數據模型
- 重新導出 aiva_common 中的常用 Schema 供模組使用

注意：大部分掃描相關的標準 Schema 已在 aiva_common 中定義，應優先使用。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# ==================== 從 aiva_common 重新導出標準 Schema (避免重複定義) ====================

# 枚舉
from ...aiva_common.enums import (
    AssetType,
    Confidence,
    Severity,
    VulnerabilityStatus,
    VulnerabilityType,
)

# 基礎 Schema
from ...aiva_common.schemas import (
    Asset,
    Authentication,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    Fingerprints,
    RateLimit,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
    Vulnerability,
)

# 增強 Schema
from ...aiva_common.schemas.enhanced import (
    EnhancedScanScope,
    EnhancedScanRequest,
)

# 資產 Schema
from ...aiva_common.schemas.assets import (
    AssetInventoryItem,
    AssetLifecyclePayload,
    DiscoveredAsset,
    EASMAsset,
)

# 引用 Schema
from ...aiva_common.schemas.references import (
    TechnicalFingerprint,
    VulnerabilityDiscovery,
)

# 任務 Schema
from ...aiva_common.schemas.tasks import (
    EASMDiscoveryPayload,
)

# 分析 Schema
from ...aiva_common.schemas.findings import (
    JavaScriptAnalysisResult,
)

# ==================== 協調器特有模型 (aiva_common 中沒有的) ====================

class ScanCoordinationMetadata(BaseModel):
    """掃描協調元數據 - 協調器專用
    
    用於追蹤多引擎協調的內部狀態和控制信息。
    這是協調器特有的控制平面數據，aiva_common 中沒有對應模型。
    """
    
    coordination_id: str = Field(description="協調ID")
    scan_request_id: str = Field(description="關聯的掃描請求ID")
    coordination_strategy: str = Field(description="協調策略")  # "sequential", "parallel", "adaptive"
    engine_assignments: dict[str, list[str]] = Field(default_factory=dict, description="引擎任務分配")
    priority_queue: list[str] = Field(default_factory=list, description="優先級隊列")
    resource_allocation: dict[str, Any] = Field(default_factory=dict, description="資源分配")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    estimated_completion: datetime | None = Field(default=None, description="預計完成時間")
    metadata: dict[str, Any] = Field(default_factory=dict, description="額外元數據")

class EngineStatus(BaseModel):
    """引擎狀態 - 協調器專用
    
    追蹤各引擎的運行狀態和性能指標。
    這是協調器內部使用的監控數據，aiva_common 中沒有對應模型。
    """
    
    engine_id: str = Field(description="引擎ID")
    engine_type: str = Field(description="引擎類型")  # "python", "typescript", "rust", "go"
    status: str = Field(description="狀態")  # "idle", "busy", "error", "offline"
    current_tasks: list[str] = Field(default_factory=list, description="當前任務")
    performance_metrics: dict[str, float] = Field(default_factory=dict, description="性能指標")
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))

class MultiEngineCoordinationResult(BaseModel):
    """多引擎協調結果 - 協調器專用
    
    彙總多個引擎的掃描結果和協調過程的整體狀態。
    這是協調器特有的結果聚合模型，aiva_common 中沒有對應模型。
    """
    
    coordination_id: str = Field(description="協調ID")
    participating_engines: list[str] = Field(description="參與引擎")
    results_by_engine: dict[str, Any] = Field(default_factory=dict, description="各引擎結果")
    aggregated_findings: list[dict] = Field(default_factory=list, description="聚合發現")
    completion_status: str = Field(description="完成狀態")
    total_duration: float = Field(ge=0.0, description="總耗時(秒)")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

# ==================== 重新導出清單 ====================
__all__ = [
    # ========== 從 aiva_common 重新導出 (標準 Schema，優先使用) ==========
    # 枚舉
    "AssetType",
    "Confidence", 
    "Severity",
    "VulnerabilityStatus",
    "VulnerabilityType",
    
    # 基礎 Schema
    "Asset",
    "Authentication",
    "CVEReference",
    "CVSSv3Metrics", 
    "CWEReference",
    "Fingerprints",
    "RateLimit",
    "ScanCompletedPayload",
    "ScanStartPayload",
    "Summary",
    "Vulnerability",
    
    # 增強 Schema (來自 aiva_common.schemas.enhanced)
    "EnhancedScanScope",
    "EnhancedScanRequest",
    
    # 資產 Schema (來自 aiva_common.schemas.assets)  
    "AssetInventoryItem",
    "AssetLifecyclePayload",
    "DiscoveredAsset",
    "EASMAsset",
    
    # 引用 Schema (來自 aiva_common.schemas.references)
    "TechnicalFingerprint",
    "VulnerabilityDiscovery",
    
    # 任務 Schema (來自 aiva_common.schemas.tasks)
    "EASMDiscoveryPayload",
    
    # 分析 Schema (來自 aiva_common.schemas.findings)
    "JavaScriptAnalysisResult",
    
    # ========== 協調器特有模型 (aiva_common 中沒有的) ==========
    "ScanCoordinationMetadata",
    "EngineStatus", 
    "MultiEngineCoordinationResult",
]
