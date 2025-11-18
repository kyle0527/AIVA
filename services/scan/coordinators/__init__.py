"""
AIVA Scan 協調器模組

協調器模組負責管理和協調四個語言引擎的掃描工作，提供統一的掃描介面。

核心組件：
- scan_models.py: 協調器數據模型 (遵循 aiva_common 規範)
- multi_engine_coordinator.py: 多引擎協調器  
- unified_scan_engine.py: 統一掃描引擎
- scan_orchestrator.py: 掃描編排器
- target_generators/: 目標生成器 (動態掃描配置)

遵循 aiva_common 規範：
- 優先使用 aiva_common 的標準 Schema
- 禁止重複定義，遵循單一數據來源原則
- 只定義協調器特有的、必要的擴展模型
"""

# 導入數據模型
from .scan_models import (
    # 從 aiva_common 重新導出的標準 Schema
    Asset,
    ScanStartPayload, 
    ScanCompletedPayload,
    Vulnerability,
    # 協調器特有模型
    ScanCoordinationMetadata,
    EngineStatus,
    MultiEngineCoordinationResult,
)

# 導入協調器組件（延遲導入避免循環依賴）
try:
    from .multi_engine_coordinator import MultiEngineCoordinator
    from .unified_scan_engine import UnifiedScanEngine
    # ScanOrchestrator 已移至 engines/python_engine/ 目錄
    from ..engines.python_engine.scan_orchestrator import ScanOrchestrator
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"協調器組件導入失敗（可能處於開發中）: {e}")

__all__ = [
    # ========== 數據模型 ==========
    # 標準 Schema (來自 aiva_common)
    "Asset",
    "ScanStartPayload",
    "ScanCompletedPayload", 
    "Vulnerability",
    # 協調器特有模型
    "ScanCoordinationMetadata",
    "EngineStatus",
    "MultiEngineCoordinationResult",
    
    # ========== 協調器組件 ==========
    "MultiEngineCoordinator",
    "UnifiedScanEngine",
    "ScanOrchestrator",
]