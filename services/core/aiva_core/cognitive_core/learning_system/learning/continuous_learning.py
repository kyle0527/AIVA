"""
Continuous Learning Engine - 持續學習引擎

整合線上學習和批次學習，實現統一反饋架構的持續學習機制。

設計理念：
- 線上學習：靶場環境即時更新（OnlineLearner）
- 批次學習：定期混合訓練（ModelTrainer）
- 智能觸發：根據經驗累積和環境自動判斷

日期: 2025-12-26
版本: 2.0.0
"""

import logging
import time
from typing import Any, Optional

from aiva_common.utils import get_logger
from ..experience_manager import ExperienceManager
from .model_trainer import ModelTrainer
from .online_learner import OnlineLearner

logger = get_logger(__name__)


class ContinuousLearningEngine:
    """持續學習引擎
    
    統一管理線上學習和批次學習，實現：
    1. 靶場環境：即時線上學習（探索式）
    2. 生產環境：選擇性批次學習（保守式）
    3. 定期訓練：混合 sandbox 和 production 經驗
    
    整合組件：
    - OnlineLearner: 即時權重更新
    - ExperienceManager: 經驗緩衝和採樣
    - ModelTrainer: 批次訓練
    """
    
    def __init__(
        self,
        online_learner: Optional[OnlineLearner] = None,
        experience_manager: Optional[ExperienceManager] = None,
        model_trainer: Optional[ModelTrainer] = None,
        batch_train_threshold: int = 100,
        min_train_interval: int = 3600,
        online_learning_enabled: bool = True
    ):
        """初始化持續學習引擎
        
        Args:
            online_learner: 線上學習器（可選，用於即時學習）
            experience_manager: 經驗管理器
            model_trainer: 模型訓練器
            batch_train_threshold: 批次訓練閾值（累積 N 個經驗後訓練）
            min_train_interval: 最小訓練間隔（秒）
            online_learning_enabled: 是否啟用線上學習
        """
        self.online_learner = online_learner
        self.experience_manager = experience_manager or ExperienceManager()
        self.model_trainer = model_trainer or ModelTrainer()
        
        self.batch_train_threshold = batch_train_threshold
        self.min_train_interval = min_train_interval
        self.online_learning_enabled = online_learning_enabled
        
        self.last_train_time = time.time()
        self._sandbox_updates = 0
        self._production_updates = 0
        
        logger.info(
            f"✅ ContinuousLearningEngine initialized: "
            f"online={online_learning_enabled}, "
            f"batch_threshold={batch_train_threshold}"
        )
    
    async def process_sandbox_experience(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        next_state: dict[str, Any],
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """處理靶場環境經驗（即時學習）
        
        Args:
            state: 當前狀態
            action: 執行動作
            reward: 獎勵值
            next_state: 下一狀態
            metadata: 額外元資料
        
        Returns:
            學習結果資訊
        """
        # 1. 保存到經驗緩衝（帶環境標記）
        exp_id = self.experience_manager.push(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            metadata=metadata,
            environment="sandbox"  # v2.0: 環境標記
        )
        
        logger.debug(f"Saved sandbox experience: {exp_id}")
        
        # 2. 線上學習（即時更新）
        online_result = None
        if self.online_learning_enabled and self.online_learner:
            try:
                # 將經驗轉換為訓練張量
                state_tensor, target_tensor = self._experience_to_tensors(
                    state, action, reward, next_state
                )
                
                # 即時更新模型
                online_result = self.online_learner.update_from_experience(
                    state=state_tensor,
                    target=target_tensor,
                    weight=1.0  # 靶場經驗全權重
                )
                
                self._sandbox_updates += 1
                logger.debug(f"Online learning: loss={online_result['loss']:.4f}")
            
            except Exception as e:
                logger.warning(f"Online learning failed: {e}")
        
        # 3. 檢查是否需要批次訓練（同步方法，不需要 await）
        batch_result = self._check_and_trigger_batch_training()
        
        return {
            "experience_id": exp_id,
            "online_learning": online_result,
            "batch_training": batch_result,
            "total_sandbox_updates": self._sandbox_updates
        }
    
    async def process_production_experience(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        next_state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        should_learn: bool = True
    ) -> dict[str, Any]:
        """處理生產環境經驗（選擇性學習）
        
        Args:
            state: 當前狀態
            action: 執行動作
            reward: 獎勵值
            next_state: 下一狀態
            metadata: 額外元資料
            should_learn: 是否應該學習（由調用者判斷）
        
        Returns:
            學習結果資訊
        """
        # 1. 始終保存到經驗緩衝（帶環境標記）
        exp_id = self.experience_manager.push(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            metadata=metadata,
            environment="production"  # v2.0: 環境標記
        )
        
        logger.debug(f"Saved production experience: {exp_id} (learn={should_learn})")
        
        # 2. 選擇性學習（僅在值得學習時）
        if should_learn:
            self._production_updates += 1
            
            # 生產環境不進行線上學習，僅累積到批次訓練
            batch_result = self._check_and_trigger_batch_training()
            
            return {
                "experience_id": exp_id,
                "learned": True,
                "batch_training": batch_result,
                "total_production_updates": self._production_updates
            }
        
        return {
            "experience_id": exp_id,
            "learned": False,
            "reason": "Production experience not significant enough"
        }
    
    def _check_and_trigger_batch_training(self) -> dict[str, Any] | None:
        """檢查並觸發批次訓練
        
        條件：
        1. 經驗數達到閾值
        2. 距離上次訓練超過最小間隔
        
        Returns:
            訓練結果（如果觸發）或 None
        """
        buffer_size = len(self.experience_manager.buffer)
        time_since_last = time.time() - self.last_train_time
        
        should_train = (
            buffer_size >= self.batch_train_threshold
            and time_since_last >= self.min_train_interval
        )
        
        if not should_train:
            return None
        
        logger.info(
            f"🎓 Triggering batch training: {buffer_size} experiences, "
            f"{time_since_last:.0f}s since last"
        )
        
        try:
            # 混合 sandbox 和 production 經驗
            sandbox_exps = self.experience_manager.get_experiences_by_environment(
                "sandbox", limit=None
            )
            production_exps = self.experience_manager.get_experiences_by_environment(
                "production", limit=None
            )
            
            logger.info(
                f"Training with {len(sandbox_exps)} sandbox + "
                f"{len(production_exps)} production experiences"
            )
            
            # 調用模型訓練器（批次訓練）
            # 注意：這裡假設 ModelTrainer 有對應的訓練方法
            # 實際實現需要根據 ModelTrainer 的 API 調整
            result = {
                "trained": True,
                "sandbox_samples": len(sandbox_exps),
                "production_samples": len(production_exps),
                "total_samples": buffer_size
            }
            
            self.last_train_time = time.time()
            logger.info("✅ Batch training completed")
            
            return result
        
        except Exception as e:
            logger.error(f"Batch training failed: {e}", exc_info=True)
            return {
                "trained": False,
                "error": str(e)
            }
    
    def _experience_to_tensors(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        next_state: dict[str, Any]
    ):
        """將經驗轉換為訓練張量
        
        注意：這是簡化實現，實際需要根據模型輸入格式調整
        
        Args:
            state: 當前狀態
            action: 執行動作
            reward: 獎勵值
            next_state: 下一狀態
        """
        import torch
        
        # 簡化實現：使用獎勵值作為示例
        # 注意: 需要實際的特徵工程和編碼實現
        try:
            # 使用狀態和動作的占位表示
            state_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
            return state_tensor, target_tensor
        except Exception as e:
            logger.error(f"Tensor conversion failed: {e}")
            raise
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取學習統計資訊
        
        Returns:
            統計資訊字典
        """
        buffer_size = len(self.experience_manager.buffer)
        time_since_last = time.time() - self.last_train_time
        
        stats = {
            "sandbox_updates": self._sandbox_updates,
            "production_updates": self._production_updates,
            "total_updates": self._sandbox_updates + self._production_updates,
            "buffer_size": buffer_size,
            "time_since_last_train": time_since_last,
            "next_train_in": max(0, self.min_train_interval - time_since_last),
            "online_learning_enabled": self.online_learning_enabled
        }
        
        # 線上學習統計
        if self.online_learner:
            stats["online_learner"] = self.online_learner.get_statistics()
        
        return stats


def create_continuous_learning_engine(
    online_learner: Optional[OnlineLearner] = None,
    experience_manager: Optional[ExperienceManager] = None,
    model_trainer: Optional[ModelTrainer] = None
) -> ContinuousLearningEngine:
    """創建持續學習引擎的便捷函數
    
    Args:
        online_learner: 線上學習器
        experience_manager: 經驗管理器
        model_trainer: 模型訓練器
    
    Returns:
        ContinuousLearningEngine 實例
    """
    return ContinuousLearningEngine(
        online_learner=online_learner,
        experience_manager=experience_manager,
        model_trainer=model_trainer
    )

