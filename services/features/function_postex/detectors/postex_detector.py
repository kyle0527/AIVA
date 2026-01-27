"""
function_postex.detectors.postex_detector
Post-Exploitation 主檢測器

整合權限提升、橫向移動、持久性檢測引擎。
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from aiva_common.utils import get_logger
from ..engines.privilege_escalation import PrivilegeEscalationEngine, PrivEscVector
from ..engines.lateral_movement import LateralMovementEngine, LateralMovementVector
from ..engines.persistence import PersistenceEngine, PersistenceVector

logger = get_logger(__name__)


@dataclass
class PostExResult:
    """後滲透測試結果"""
    privilege_escalation: List[PrivEscVector]
    lateral_movement: List[LateralMovementVector]
    persistence: List[PersistenceVector]
    summary: Dict[str, Any]


class PostExDetector:
    """Post-Exploitation 檢測器"""
    
    def __init__(self, safe_mode: bool = False):
        """
        初始化後滲透檢測器
        
        Args:
            safe_mode: 安全模式（僅檢測，不執行攻擊）
        """
        self.safe_mode = safe_mode
        self.privesc_engine = PrivilegeEscalationEngine(safe_mode)
        self.lateral_engine = LateralMovementEngine(safe_mode)
        self.persistence_engine = PersistenceEngine(safe_mode)
        logger.info(f"PostEx 檢測器初始化 (SafeMode: {safe_mode})")
    
    def scan_full(self, target: str = "localhost", network: str = None) -> PostExResult:
        """
        執行完整後滲透掃描
        
        Args:
            target: 目標系統
            network: 網路範圍（用於橫向移動）
        
        Returns:
            PostExResult: 檢測結果
        """
        logger.info(f"開始 PostEx 完整掃描 (target: {target})")
        
        # 1. 權限提升檢測
        logger.info("執行權限提升檢測...")
        privesc_vectors = self.privesc_engine.scan(target)
        
        # 2. 橫向移動檢測
        lateral_vectors = []
        if network:
            logger.info(f"執行橫向移動檢測 (network: {network})...")
            lateral_vectors = self.lateral_engine.scan_network(network)
        
        # 3. 持久性檢測
        logger.info("執行持久性檢測...")
        persistence_vectors = self.persistence_engine.scan()
        
        # 生成摘要
        summary = self._generate_summary(privesc_vectors, lateral_vectors, persistence_vectors)
        
        result = PostExResult(
            privilege_escalation=privesc_vectors,
            lateral_movement=lateral_vectors,
            persistence=persistence_vectors,
            summary=summary
        )
        
        logger.info(f"PostEx 掃描完成: {summary}")
        return result
    
    def scan_privesc_only(self, target: str = "localhost") -> List[PrivEscVector]:
        """僅執行權限提升掃描"""
        return self.privesc_engine.scan(target)
    
    def scan_lateral_only(self, network: str) -> List[LateralMovementVector]:
        """僅執行橫向移動掃描"""
        return self.lateral_engine.scan_network(network)
    
    def scan_persistence_only(self) -> List[PersistenceVector]:
        """僅執行持久性掃描"""
        return self.persistence_engine.scan()
    
    def _generate_summary(
        self,
        privesc: List[PrivEscVector],
        lateral: List[LateralMovementVector],
        persistence: List[PersistenceVector]
    ) -> Dict[str, Any]:
        """生成掃描摘要"""
        return {
            "total_vectors": len(privesc) + len(lateral) + len(persistence),
            "privilege_escalation_count": len(privesc),
            "lateral_movement_count": len(lateral),
            "persistence_count": len(persistence),
            "critical_count": sum(1 for v in privesc if v.severity.name == "CRITICAL"),
            "high_count": sum(1 for v in privesc if v.severity.name == "HIGH"),
            "safe_mode": self.safe_mode
        }
