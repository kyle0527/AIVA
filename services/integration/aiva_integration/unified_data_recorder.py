"""
AIVA V2 統一資料存取介面 - 適配器模式

此模組提供 V1 (AIOperationRecorder) 和 V2 (ExperienceRepository) 之間的橋接，
實現平滑遷移而不破壞現有程式碼。

⚠️ 使用指南:
- 新功能開發: 直接使用 ExperienceRepository (V2)
- 現有程式碼遷移: 使用此適配器進行漸進式遷移
- 最終目標: 完全移除 AIOperationRecorder (V1)
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from services.integration.aiva_integration.ai_operation_recorder import (
    AIOperationRecorder,
)
from services.integration.aiva_integration.reception import ExperienceRepository

logger = logging.getLogger(__name__)


class UnifiedDataRecorder:
    """統一資料記錄器 - V1/V2 適配器
    
    此適配器優先使用 V2 ExperienceRepository，
    在失敗時自動降級到 V1 AIOperationRecorder。
    
    範例使用:
        recorder = UnifiedDataRecorder(
            database_url="sqlite:///aiva_experience.db",
            legacy_output_dir="logs"
        )
        
        # 記錄操作 (自動選擇 V2 或 V1)
        recorder.record_operation(
            operation_type="scan",
            details={"target": "example.com"},
            status="success"
        )
        
        # V2 專用方法 (建議使用)
        recorder.save_experience_v2(
            plan_id="plan_001",
            attack_type="sqli",
            ...
        )
    """

    def __init__(
        self,
        database_url: str = "sqlite:///aiva_experience.db",
        legacy_output_dir: str = "logs",
        use_v2_primary: bool = True,
        enable_v1_fallback: bool = True,
    ):
        """初始化統一資料記錄器
        
        Args:
            database_url: V2 資料庫連接字串
            legacy_output_dir: V1 JSON 輸出目錄
            use_v2_primary: 是否優先使用 V2 (建議: True)
            enable_v1_fallback: V2 失敗時是否降級到 V1 (建議: True for migration)
        """
        self.use_v2_primary = use_v2_primary
        self.enable_v1_fallback = enable_v1_fallback
        
        # V2: ExperienceRepository (推薦)
        self.v2_repo: Optional[ExperienceRepository] = None
        try:
            self.v2_repo = ExperienceRepository(database_url)
            logger.info(f"✅ V2 ExperienceRepository initialized: {database_url}")
        except Exception as e:
            logger.warning(f"⚠️ V2 ExperienceRepository init failed: {e}")
            if not enable_v1_fallback:
                raise
        
        # V1: AIOperationRecorder (棄用中)
        self.v1_recorder: Optional[AIOperationRecorder] = None
        if enable_v1_fallback:
            try:
                self.v1_recorder = AIOperationRecorder(
                    output_dir=legacy_output_dir,
                    enable_realtime=False  # 降低資源消耗
                )
                logger.info(f"📝 V1 AIOperationRecorder initialized (fallback): {legacy_output_dir}")
            except Exception as e:
                logger.error(f"❌ V1 AIOperationRecorder init failed: {e}")
        
        # 統計資訊
        self.stats = {
            "v2_success": 0,
            "v2_failure": 0,
            "v1_fallback": 0,
            "total_operations": 0,
        }

    def record_operation(
        self,
        operation_type: str,
        details: dict[str, Any],
        status: str = "success",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """記錄操作 (統一介面 - 自動選擇 V2/V1)
        
        Args:
            operation_type: 操作類型 (e.g., "scan", "attack", "analyze")
            details: 操作詳細資訊
            status: 操作狀態 ("success", "failure", "pending")
            metadata: 額外元資料
        
        Returns:
            是否成功記錄
        """
        self.stats["total_operations"] += 1
        
        # 優先使用 V2
        if self.use_v2_primary and self.v2_repo:
            try:
                self._record_v2(operation_type, details, status, metadata)
                self.stats["v2_success"] += 1
                return True
            except Exception as e:
                logger.error(f"V2 recording failed: {e}")
                self.stats["v2_failure"] += 1
                
                # 降級到 V1
                if self.enable_v1_fallback and self.v1_recorder:
                    logger.warning("Falling back to V1 AIOperationRecorder")
                    return self._record_v1(operation_type, details, status, metadata)
                else:
                    raise
        
        # 使用 V1 (當 V2 不可用或 use_v2_primary=False)
        elif self.v1_recorder:
            self.stats["v1_fallback"] += 1
            return self._record_v1(operation_type, details, status, metadata)
        
        else:
            raise RuntimeError("No recording backend available (V2 and V1 both unavailable)")

    def _record_v2(
        self,
        operation_type: str,
        details: dict[str, Any],
        status: str,
        metadata: Optional[dict[str, Any]],
    ) -> None:
        """使用 V2 ExperienceRepository 記錄"""
        if not self.v2_repo:
            raise RuntimeError("V2 ExperienceRepository not available")
        
        # 轉換為 V2 格式
        plan_id = details.get("plan_id", f"op_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        attack_type = operation_type
        
        execution_trace = {
            "status": status,
            "trace_session_id": details.get("session_id", ""),
            "details": details,
        }
        
        metrics = {
            "completion_rate": 1.0 if status == "success" else 0.0,
            "success_steps": 1 if status == "success" else 0,
            "failed_steps": 0 if status == "success" else 1,
            "error_count": 0 if status == "success" else 1,
        }
        
        feedback = {
            "reward": 10 if status == "success" else 0,
            "status": status,
        }
        
        self.v2_repo.save_experience(
            plan_id=plan_id,
            attack_type=attack_type,
            ast_graph={},  # 若有需要可從 details 提取
            execution_trace=execution_trace,
            metrics=metrics,
            feedback=feedback,
            target_info=details.get("target_info"),
            metadata=metadata,
        )
        
        logger.debug(f"V2 recorded: {operation_type} - {status}")

    def _record_v1(
        self,
        operation_type: str,
        details: dict[str, Any],
        status: str,
        metadata: Optional[dict[str, Any]],
    ) -> bool:
        """使用 V1 AIOperationRecorder 記錄"""
        if not self.v1_recorder:
            raise RuntimeError("V1 AIOperationRecorder not available")
        
        # 發出棄用警告
        warnings.warn(
            "Using deprecated V1 AIOperationRecorder. Please migrate to V2 ExperienceRepository.",
            DeprecationWarning,
            stacklevel=3,
        )
        
        self.v1_recorder.log_operation(
            operation_type=operation_type,
            details=details,
            status=status,
            metadata=metadata or {},
        )
        
        logger.debug(f"V1 recorded (fallback): {operation_type} - {status}")
        return True

    # V2 專用方法 (建議直接使用)
    
    def save_experience_v2(
        self,
        plan_id: str,
        attack_type: str,
        ast_graph: dict[str, Any],
        execution_trace: dict[str, Any],
        metrics: dict[str, Any],
        feedback: dict[str, Any],
        target_info: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """直接使用 V2 save_experience (推薦)
        
        這是推薦的方法，因為它直接使用 V2 API，
        提供完整的型別安全和功能。
        """
        if not self.v2_repo:
            raise RuntimeError("V2 ExperienceRepository not available")
        
        return self.v2_repo.save_experience(
            plan_id=plan_id,
            attack_type=attack_type,
            ast_graph=ast_graph,
            execution_trace=execution_trace,
            metrics=metrics,
            feedback=feedback,
            target_info=target_info,
            metadata=metadata,
        )
    
    def query_experiences_v2(
        self,
        attack_type: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        """查詢經驗記錄 (V2)"""
        if not self.v2_repo:
            raise RuntimeError("V2 ExperienceRepository not available")
        
        return self.v2_repo.query_experiences(
            attack_type=attack_type,
            min_score=min_score,
            max_score=max_score,
            limit=limit,
            offset=offset,
        )
    
    # 統計與監控
    
    def get_stats(self) -> dict[str, Any]:
        """獲取統計資訊"""
        return {
            **self.stats,
            "v2_available": self.v2_repo is not None,
            "v1_available": self.v1_recorder is not None,
            "primary_backend": "v2" if self.use_v2_primary else "v1",
            "v2_success_rate": (
                self.stats["v2_success"] / self.stats["total_operations"]
                if self.stats["total_operations"] > 0
                else 0.0
            ),
        }
    
    def print_stats(self):
        """列印統計資訊"""
        stats = self.get_stats()
        print("\n=== Unified Data Recorder Statistics ===")
        print(f"Total Operations: {stats['total_operations']}")
        print(f"V2 Success: {stats['v2_success']}")
        print(f"V2 Failure: {stats['v2_failure']}")
        print(f"V1 Fallback: {stats['v1_fallback']}")
        print(f"V2 Success Rate: {stats['v2_success_rate']:.2%}")
        print(f"Primary Backend: {stats['primary_backend']}")
        print(f"V2 Available: {stats['v2_available']}")
        print(f"V1 Available: {stats['v1_available']}")
        print("=" * 40)


# 便利函數

_global_recorder: Optional[UnifiedDataRecorder] = None


def get_unified_recorder(
    database_url: str = "sqlite:///aiva_experience.db",
    legacy_output_dir: str = "logs",
) -> UnifiedDataRecorder:
    """獲取全域統一記錄器單例"""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = UnifiedDataRecorder(
            database_url=database_url,
            legacy_output_dir=legacy_output_dir,
        )
    return _global_recorder


# 範例使用
if __name__ == "__main__":
    # 基本使用
    recorder = UnifiedDataRecorder()
    
    # 記錄操作 (自動使用 V2 或降級到 V1)
    recorder.record_operation(
        operation_type="scan",
        details={
            "target": "example.com",
            "scan_type": "vulnerability",
        },
        status="success",
    )
    
    # V2 專用方法 (推薦)
    recorder.save_experience_v2(
        plan_id="plan_001",
        attack_type="sqli",
        ast_graph={},
        execution_trace={"target": "example.com"},
        metrics={"completion_rate": 1.0},
        feedback={"reward": 10},
    )
    
    # 查看統計
    recorder.print_stats()
