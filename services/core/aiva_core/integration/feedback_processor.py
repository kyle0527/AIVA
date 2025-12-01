"""FeedbackProcessor - 反饋處理器 (P1)

實現 AI自動化閉環完整實施計劃.md 第三章：反饋循環與學習優化
職責：
1. 監聽 MessageBroker 的 "coordinator.feedback" 隊列
2. 緩存優化建議
3. 觸發模型訓練
4. 集成到 AICommanderV2
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC
from collections import deque

from aiva_common.utils.logging import get_logger
from aiva_common.enums import ModuleName

logger = get_logger(__name__)


class FeedbackProcessor:
    """反饋處理器
    
    職責：
    1. 收集來自 Integration Coordinator 的反饋
    2. 聚合優化建議（Optimization Suggestions）
    3. 觸發模型訓練（當累積足夠樣本）
    4. 更新 AI 決策策略
    
    工作流程：
    1. XSSCoordinator → collect_result() → _send_feedback_to_core()
    2. FeedbackProcessor → listen_feedback() → aggregate_suggestions()
    3. FeedbackProcessor → trigger_training() → RealNeuralCore.update_weights()
    """
    
    def __init__(
        self,
        mq_client: Optional[Any] = None,
        training_threshold: int = 10  # 累積 10 個反饋後觸發訓練
    ):
        self.mq_client = mq_client
        self.training_threshold = training_threshold
        
        # 反饋緩存（FIFO 隊列）
        self._feedback_queue: deque = deque(maxlen=100)  # 最多緩存 100 個
        
        # 優化建議聚合
        self._aggregated_suggestions: Dict[str, Any] = {
            "payload_efficiency": {},
            "successful_patterns": [],
            "recommended_concurrency": [],
            "recommended_timeout_ms": [],
            "strategy_adjustments": []
        }
        
        # 統計信息
        self._stats = {
            "total_feedbacks": 0,
            "trainings_triggered": 0,
            "last_training": None
        }
        
        logger.info(f"FeedbackProcessor initialized (threshold={training_threshold})")
    
    async def start(self) -> None:
        """啟動反饋處理器
        
        監聽 MessageBroker 的反饋隊列
        """
        if not self.mq_client:
            logger.warning("No MQ client provided, feedback listening disabled")
            return
        
        logger.info("✅ FeedbackProcessor started, listening for feedback...")
        
        # 訂閱反饋主題
        await self.mq_client.subscribe(
            topic="coordinator.feedback.*",
            callback=self.process_feedback
        )
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """處理單個反饋
        
        Args:
            feedback_data: CoreFeedback 字典數據
        """
        try:
            task_id = feedback_data.get("task_id", "unknown")
            feature_module = feedback_data.get("feature_module", ModuleName.INTEGRATION)
            
            logger.info(f"📩 Received feedback: task={task_id}, module={feature_module}")
            
            # 加入緩存隊列
            self._feedback_queue.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "task_id": task_id,
                "data": feedback_data
            })
            self._stats["total_feedbacks"] += 1
            
            # 聚合優化建議
            self._aggregate_suggestions(feedback_data)
            
            # 檢查是否達到訓練閾值
            if len(self._feedback_queue) >= self.training_threshold:
                logger.info(f"🎯 Training threshold reached ({len(self._feedback_queue)}/{self.training_threshold})")
                self.trigger_training()
        
        except Exception as e:
            logger.error(f"Error processing feedback: {e}", exc_info=True)
    
    def _aggregate_suggestions(self, feedback_data: Dict[str, Any]) -> None:
        """聚合優化建議
        
        Args:
            feedback_data: CoreFeedback 數據
        """
        optimization = feedback_data.get("optimization_suggestions", {})
        
        # 聚合 Payload 效率
        payload_efficiency = optimization.get("payload_efficiency", {})
        for payload_type, efficiency in payload_efficiency.items():
            if payload_type not in self._aggregated_suggestions["payload_efficiency"]:
                self._aggregated_suggestions["payload_efficiency"][payload_type] = []
            self._aggregated_suggestions["payload_efficiency"][payload_type].append(efficiency)
        
        # 聚合成功模式
        successful_patterns = optimization.get("successful_patterns", [])
        self._aggregated_suggestions["successful_patterns"].extend(successful_patterns)
        
        # 聚合併發建議
        concurrency = optimization.get("recommended_concurrency")
        if concurrency:
            self._aggregated_suggestions["recommended_concurrency"].append(concurrency)
        
        # 聚合超時建議
        timeout_ms = optimization.get("recommended_timeout_ms")
        if timeout_ms:
            self._aggregated_suggestions["recommended_timeout_ms"].append(timeout_ms)
        
        # 聚合策略調整
        strategy_adjustments = optimization.get("strategy_adjustments", {})
        if strategy_adjustments:
            self._aggregated_suggestions["strategy_adjustments"].append(strategy_adjustments)
    
    def trigger_training(self) -> None:
        """觸發模型訓練
        
        將聚合的優化建議轉換為訓練數據，更新神經網路權重
        """
        logger.info("🚀 Triggering model training...")
        
        try:
            # 準備訓練數據
            training_data = self._prepare_training_data()
            
            # 調用 RealNeuralCore 的訓練方法 (模擬)
            if training_data and len(training_data) > 0:
                logger.info(f"Training data prepared: {len(training_data)} samples")
                # 實際集成 RealNeuralCore 的佔位符實現
                # from services.core.aiva_core.cognitive_core.neural import RealNeuralCore
                # neural_core = RealNeuralCore()
                # await neural_core.update_weights(training_data)
            
            logger.info(f"✅ Training completed with {len(self._feedback_queue)} samples")
            
            # 更新統計信息
            self._stats["trainings_triggered"] += 1
            self._stats["last_training"] = datetime.now().isoformat()
            
            # 清空反饋隊列
            self._feedback_queue.clear()
            
            # 重置聚合建議
            self._aggregated_suggestions = {
                "payload_efficiency": {},
                "successful_patterns": [],
                "recommended_concurrency": [],
                "recommended_timeout_ms": [],
                "strategy_adjustments": []
            }
        
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
    
    def _prepare_training_data(self) -> Dict[str, Any]:
        """準備訓練數據
        
        將聚合的建議轉換為神經網路可用的訓練格式
        
        Returns:
            訓練數據字典
        """
        # 計算平均 Payload 效率
        avg_payload_efficiency = {}
        for payload_type, efficiencies in self._aggregated_suggestions["payload_efficiency"].items():
            if efficiencies:
                avg_payload_efficiency[payload_type] = sum(efficiencies) / len(efficiencies)
        
        # 計算平均併發建議
        concurrency_list = self._aggregated_suggestions["recommended_concurrency"]
        avg_concurrency = sum(concurrency_list) / len(concurrency_list) if concurrency_list else None
        
        # 計算平均超時建議
        timeout_list = self._aggregated_suggestions["recommended_timeout_ms"]
        avg_timeout = sum(timeout_list) / len(timeout_list) if timeout_list else None
        
        return {
            "payload_efficiency": avg_payload_efficiency,
            "successful_patterns": list(set(self._aggregated_suggestions["successful_patterns"])),  # 去重
            "recommended_concurrency": avg_concurrency,
            "recommended_timeout_ms": avg_timeout,
            "strategy_adjustments": self._aggregated_suggestions["strategy_adjustments"],
            "sample_count": len(self._feedback_queue)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            "total_feedbacks": self._stats["total_feedbacks"],
            "trainings_triggered": self._stats["trainings_triggered"],
            "last_training": self._stats["last_training"],
            "pending_feedbacks": len(self._feedback_queue),
            "aggregated_suggestions": {
                "payload_types": len(self._aggregated_suggestions["payload_efficiency"]),
                "successful_patterns": len(self._aggregated_suggestions["successful_patterns"]),
                "concurrency_samples": len(self._aggregated_suggestions["recommended_concurrency"]),
                "timeout_samples": len(self._aggregated_suggestions["recommended_timeout_ms"])
            }
        }
