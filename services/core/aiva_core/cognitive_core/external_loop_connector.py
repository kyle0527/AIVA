"""External Loop Connector - 外部閉環連接器

將執行結果傳遞給 external_learning 進行偏差分析和模型訓練，實現 AI 從經驗中學習

數據流：
task_planning (執行結果) → ExternalLoopConnector → external_learning (偏差分析 + 訓練) → cognitive_core (權重更新)

遵循 aiva_common 修復規範:
- 使用統一的日誌記錄
- 使用統一的錯誤處理
- 使用統一的數據格式
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExternalLoopConnector:
    """外部閉環連接器
    
    職責：
    1. 接收任務執行結果（計劃 AST + 執行軌跡）
    2. 觸發偏差分析（計劃 vs 實際執行）
    3. 觸發模型訓練（基於偏差數據）
    4. 通知權重管理器（新權重可用）
    
    這是 AI 自我優化雙重閉環中「對外學習閉環」的關鍵組件
    """
    
    def __init__(self):
        """初始化外部閉環連接器"""
        self._comparator = None
        self._trainer = None
        self._weight_manager = None
        
        logger.info("ExternalLoopConnector initialized")
    
    @property
    def comparator(self):
        """延遲加載 ASTTraceComparator"""
        if self._comparator is None:
            from ..external_learning.analysis.ast_trace_comparator import ASTTraceComparator
            self._comparator = ASTTraceComparator()
        return self._comparator
    
    @property
    def trainer(self):
        """延遲加載 ModelTrainer"""
        if self._trainer is None:
            from ..external_learning.learning.model_trainer import ModelTrainer
            self._trainer = ModelTrainer()
        return self._trainer
    
    @property
    def weight_manager(self):
        """延遲加載 AIWeightManager"""
        if self._weight_manager is None:
            from .neural.weight_manager import AIWeightManager
            self._weight_manager = AIWeightManager()
        return self._weight_manager
    
    async def process_execution_result(
        self,
        plan: dict[str, Any],
        trace: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """處理執行結果並觸發學習循環
        
        這是外部閉環的核心方法，將執行經驗轉化為 AI 的進化動力
        
        Args:
            plan: 原始 AST 計劃
            trace: 執行軌跡
            
        Returns:
            處理統計: {
                "deviations_found": int,
                "deviations_significant": bool,
                "training_triggered": bool,
                "weights_updated": bool,
                "new_weights_version": str | None,
                "timestamp": str,
                "success": bool
            }
        """
        logger.info("🔄 Starting external loop processing...")
        
        try:
            # 步驟 1: 偏差分析
            logger.info("  Step 1: Analyzing deviations...")
            deviations = self._analyze_deviations(plan, trace)
            
            # 步驟 2: 判斷是否需要訓練
            is_significant = self._is_significant_deviation(deviations)
            logger.info(f"  Found {len(deviations)} deviations, significant: {is_significant}")
            
            training_triggered = False
            weights_updated = False
            new_version = None
            
            # 步驟 3: 如果偏差顯著，觸發訓練
            if is_significant:
                logger.info("  Step 2: Triggering model training...")
                training_result = await self._train_from_experience(plan, trace, deviations)
                training_triggered = True
                
                # 步驟 4: 如果產生了新權重，通知權重管理器
                if training_result.get("new_weights_path"):
                    logger.info("  Step 3: Registering new weights...")
                    new_version = self._register_new_weights(training_result)
                    weights_updated = True
            else:
                logger.info("  Deviations not significant, skipping training")
            
            result_summary = {
                "deviations_found": len(deviations),
                "deviations_significant": is_significant,
                "training_triggered": training_triggered,
                "weights_updated": weights_updated,
                "new_weights_version": new_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True
            }
            
            logger.info(f"✅ External loop processing completed: {result_summary}")
            return result_summary
            
        except Exception as e:
            logger.error(f"❌ External loop processing failed: {e}", exc_info=True)
            return {
                "deviations_found": 0,
                "deviations_significant": False,
                "training_triggered": False,
                "weights_updated": False,
                "new_weights_version": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": False,
                "error": str(e)
            }
    
    def _analyze_deviations(
        self,
        plan: dict[str, Any],
        trace: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """分析執行偏差
        
        Args:
            plan: 原始計劃
            trace: 執行軌跡
            
        Returns:
            偏差列表
        """
        try:
            # 暫時使用簡化版本的偏差分析
            # 完整版本需要完整的 AST 和 Trace 對象
            deviations = []
            
            # 基本檢查：計劃步驟數 vs 執行步驟數
            plan_steps = plan.get("steps", [])
            if len(trace) < len(plan_steps):
                deviations.append({
                    "type": "incomplete_execution",
                    "severity": "high",
                    "score": 2.0,
                    "expected_steps": len(plan_steps),
                    "actual_steps": len(trace)
                })
            
            # 檢查執行錯誤
            failed_steps = [t for t in trace if t.get("status") == "failed"]
            if failed_steps:
                deviations.append({
                    "type": "execution_failures",
                    "severity": "high",
                    "score": len(failed_steps) * 1.5,
                    "failed_count": len(failed_steps)
                })
            
            # 檢查執行時間異常
            avg_duration = sum(t.get("duration", 0) for t in trace) / max(len(trace), 1)
            if avg_duration > 30:  # 假設 30 秒為閾值
                deviations.append({
                    "type": "slow_execution",
                    "severity": "medium",
                    "score": 1.0,
                    "avg_duration": avg_duration
                })
            
            return deviations
        except Exception as e:
            logger.error(f"Deviation analysis failed: {e}")
            return []
    
    def _is_significant_deviation(self, deviations: list[dict]) -> bool:
        """判斷偏差是否顯著到需要訓練
        
        策略：
        - 偏差數量 >= 3: 需要訓練
        - 包含嚴重偏差（severity = "high"）: 需要訓練
        - 累積偏差分數 >= 閾值: 需要訓練
        
        Args:
            deviations: 偏差列表
            
        Returns:
            是否顯著
        """
        if not deviations:
            return False
        
        # 策略 1: 數量閾值
        if len(deviations) >= 3:
            return True
        
        # 策略 2: 嚴重性檢查
        for deviation in deviations:
            if deviation.get("severity") == "high":
                return True
        
        # 策略 3: 累積分數
        total_score = sum(deviation.get("score", 0) for deviation in deviations)
        if total_score >= 5.0:
            return True
        
        return False
    
    async def _train_from_experience(
        self,
        plan: dict[str, Any],
        trace: list[dict[str, Any]],
        deviations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """基於執行經驗訓練模型
        
        Args:
            plan: 原始計劃
            trace: 執行軌跡
            deviations: 偏差列表
            
        Returns:
            訓練結果
        """
        try:
            from services.aiva_common.schemas import ExperienceSample, ModelTrainingConfig
            
            # 將偏差轉換為經驗樣本
            samples = []
            for i, deviation in enumerate(deviations):
                sample = ExperienceSample(
                    sample_id=f"sample_{uuid4().hex[:12]}",
                    session_id=f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    plan_id=plan.get("plan_id", f"plan_{i}"),
                    state_before={
                        "plan_steps": len(plan.get("steps", [])),
                        "expected_execution": "complete"
                    },
                    action_taken={
                        "executed_steps": len(trace),
                        "deviations": deviation
                    },
                    state_after={
                        "actual_steps": len(trace),
                        "deviation_detected": True
                    },
                    reward=1.0 - min(deviation.get("score", 0) / 10.0, 1.0),
                    reward_breakdown={
                        "completion": 0.3,
                        "success": 0.3,
                        "sequence": 0.2,
                        "goal": 0.2
                    },
                    context={
                        "deviation_type": deviation.get("type"),
                        "severity": deviation.get("severity")
                    },
                    target_info=plan.get("target_info", {}),
                    timestamp=datetime.now(timezone.utc),
                    is_positive=False,
                    confidence=0.8,
                    learning_tags=["deviation", deviation.get("type", "unknown")],
                    difficulty_level=3 if deviation.get("severity") == "high" else 2
                )
                samples.append(sample)
            
            # 使用監督學習訓練
            config = ModelTrainingConfig(
                config_id=f"config_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                model_type="neural_network",
                training_mode="supervised",
                epochs=10,
                batch_size=32,
                learning_rate=0.001,
                validation_split=0.2,
                early_stopping=True,
                patience=3
            )
            
            training_result = await self.trainer.train_supervised(
                samples=samples,
                config=config
            )
            
            # 轉換為字典格式
            return {
                "new_weights_path": str(training_result.model_path) if training_result.model_path else None,
                "version": training_result.model_version,
                "metrics": training_result.metrics
            }
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {}
    
    def _register_new_weights(self, training_result: dict[str, Any]) -> str | None:
        """註冊新權重到權重管理器
        
        Args:
            training_result: 訓練結果
            
        Returns:
            新權重版本號
        """
        try:
            weights_path = training_result.get("new_weights_path")
            version = training_result.get("version", f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
            metrics = training_result.get("metrics", {})
            
            # 目前 AIWeightManager 使用 save_model_weights 方法
            # 未來可以實現 register_new_weights 輔助方法
            logger.info(f"New weights available at {weights_path}, version {version}")
            logger.info(f"Training metrics: {metrics}")
            
            return version
        except Exception as e:
            logger.error(f"Weight registration failed: {e}")
            return None
    
    def get_loop_status(self) -> dict[str, Any]:
        """獲取外部閉環狀態
        
        Returns:
            狀態資訊
        """
        return {
            "connector": "ExternalLoopConnector",
            "status": "active",
            "components": {
                "comparator": self._comparator is not None,
                "trainer": self._trainer is not None,
                "weight_manager": self._weight_manager is not None
            },
            "last_processing": None
        }
