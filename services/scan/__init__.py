"""
AIVA Scan - 多語言統一掃描引擎

重構後的模組架構（遵循 aiva_common 規範）：
- engines/: 四個語言引擎模組 (python, typescript, rust, go)  
- coordinators/: 協調器和掃描配置管理
  - scan_models.py: 掃描特定的數據模型擴展
  - multi_engine_coordinator.py: 多引擎協調器
  - unified_scan_engine.py: 統一掃描引擎

設計原則：
- 優先使用 aiva_common 的標準 Schema
- 只在必要時擴展模組特定的數據模型  
- 避免重複定義，遵循單一數據來源原則
"""

__version__ = "1.2.0"  # 重構完成版本

# 從 aiva_common 導入共享基礎設施 (標準優先)
from ..aiva_common.enums import (
    AssetType,
    ScanStatus,
    Severity,
)
from ..aiva_common.schemas import (
    Asset,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    Fingerprints,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
    Vulnerability,
)

# 從 coordinators 導入掃描特定擴展 (只有 aiva_common 沒有的才定義)
from .coordinators.scan_models import (
    ScanCoordinationMetadata,
    EngineStatus,
    MultiEngineCoordinationResult,
)

# 導入各引擎模組（延遲導入以避免循環依賴）
try:
    from . import engines
    from . import coordinators
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"部分模組導入失敗（可能處於重構中）: {e}")

__all__ = [
    # ========== 從 aiva_common 導入（共享標準，優先使用） ==========
    # 枚舉
    "AssetType",
    "ScanStatus",
    "Severity",
    # Schema - 基礎模型
    "Asset",
    "Fingerprints",
    "Summary",
    # Schema - 掃描任務
    "ScanStartPayload", 
    "ScanCompletedPayload",
    # Schema - 漏洞
    "Vulnerability",
    # Schema - 標準引用
    "CVEReference",
    "CVSSv3Metrics",
    "CWEReference",
    
    # ========== 協調器特有模型（只有新定義的） ==========
    "ScanCoordinationMetadata",
    "EngineStatus",
    "MultiEngineCoordinationResult",
    
    # ========== 模組架構 ==========
    "engines",         # 四個語言引擎模組
    "coordinators",    # 協調器模組
]
