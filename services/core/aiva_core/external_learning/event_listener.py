"""External Learning Event Listener - 外部學習事件監聽器

監聽 TASK_COMPLETED 事件並觸發學習流程，實現 AI 從執行經驗中學習

這是外部閉環的關鍵組件，連接執行系統和學習系統

使用方式:
    from services.core.aiva_core.external_learning.event_listener import ExternalLearningListener
    
    listener = ExternalLearningListener()
    await listener.start_listening()
"""

import asyncio
import logging
from typing import Any

from services.aiva_common.enums import Topic
from services.aiva_common.schemas import AivaMessage

logger = logging.getLogger(__name__)


class ExternalLearningListener:
    """外部學習監聽器
    
    職責：
    1. 監聽 TASK_COMPLETED 事件
    2. 提取執行數據（計劃 AST + 軌跡）
    3. 觸發 ExternalLoopConnector 處理
    4. 實現異步學習（不阻塞主流程）
    """
    
    def __init__(self):
        """初始化監聽器"""
        self._broker = None
        self._connector = None
        self._is_running = False
        self._processing_count = 0
        
        logger.info("ExternalLearningListener initialized")
    
    @property
    def broker(self):
        """延遲加載 MessageBroker"""
        if self._broker is None:
            from ...service_backbone.messaging.message_broker import MessageBroker
            self._broker = MessageBroker()
        return self._broker
    
    @property
    def connector(self):
        """延遲加載 ExternalLoopConnector"""
        if self._connector is None:
            from ...cognitive_core.external_loop_connector import ExternalLoopConnector
            self._connector = ExternalLoopConnector()
        return self._connector
    
    async def start_listening(self):
        """開始監聽任務完成事件
        
        這個方法會阻塞當前協程，持續監聽事件
        """
        logger.info("=" * 60)
        logger.info("👂 External Learning Listener Starting...")
        logger.info("=" * 60)
        
        try:
            # 訂閱 TASK_COMPLETED 事件
            await self.broker.subscribe(
                topic=Topic.TASK_COMPLETED,
                callback=self._on_task_completed,
            )
            
            self._is_running = True
            logger.info("✅ Listening for TASK_COMPLETED events")
            logger.info("   Waiting for execution results to learn from...")
            
            # 保持運行
            while self._is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Received shutdown signal")
            await self.stop_listening()
        except Exception as e:
            logger.error(f"❌ Listener failed: {e}", exc_info=True)
            raise
    
    async def stop_listening(self):
        """停止監聽"""
        logger.info("🛑 Stopping External Learning Listener...")
        self._is_running = False
        
        # 取消訂閱
        if self._broker:
            await self.broker.unsubscribe(Topic.TASK_COMPLETED)
        
        logger.info(f"✅ Listener stopped (processed {self._processing_count} events)")
    
    async def _on_task_completed(self, message: AivaMessage | dict[str, Any]):
        """處理任務完成事件
        
        Args:
            message: 任務完成消息
        """
        self._processing_count += 1
        
        try:
            # 解析消息
            if isinstance(message, AivaMessage):
                payload = message.payload
                plan_id = message.header.trace_id
            else:
                payload = message
                plan_id = payload.get("plan_id", "unknown")
            
            logger.info("=" * 60)
            logger.info(f"📥 Received TASK_COMPLETED event #{self._processing_count}")
            logger.info(f"   Plan ID: {plan_id}")
            logger.info("=" * 60)
            
            # 提取執行數據
            plan = payload.get("plan_ast", {})
            trace = payload.get("execution_trace", [])
            result = payload.get("result", {})
            
            # 驗證數據完整性
            if not plan or not trace:
                logger.warning("   ⚠️  Incomplete event data, skipping")
                return
            
            logger.info(f"   Plan steps: {len(plan.get('steps', []))}")
            logger.info(f"   Trace records: {len(trace)}")
            
            # 觸發學習處理（異步，不阻塞）
            asyncio.create_task(self._process_learning(plan, trace, result, plan_id))
            
        except Exception as e:
            logger.error(f"❌ Failed to process TASK_COMPLETED event: {e}", exc_info=True)
    
    async def _process_learning(
        self,
        plan: dict[str, Any],
        trace: list[dict[str, Any]],
        result: dict[str, Any],
        plan_id: str,
    ):
        """處理學習流程（異步）
        
        Args:
            plan: 計劃 AST
            trace: 執行軌跡
            result: 執行結果
            plan_id: 計劃 ID
        """
        try:
            logger.info(f"\n🧠 Processing learning for plan {plan_id}...")
            
            # 使用 ExternalLoopConnector 處理
            processing_result = await self.connector.process_execution_result(
                plan=plan,
                trace=trace,
                result=result,
            )
            
            # 顯示處理結果
            logger.info("\n" + "=" * 60)
            logger.info(f"✅ Learning Processing Completed for {plan_id}")
            logger.info("=" * 60)
            logger.info(f"   Deviations found:       {processing_result['deviations_found']}")
            logger.info(f"   Deviations significant: {processing_result['deviations_significant']}")
            logger.info(f"   Training triggered:     {processing_result['training_triggered']}")
            logger.info(f"   Weights updated:        {processing_result['weights_updated']}")
            
            if processing_result.get("new_weights_version"):
                logger.info(f"   New weights version:    {processing_result['new_weights_version']}")
            
            logger.info("=" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Learning processing failed for plan {plan_id}: {e}", exc_info=True)
    
    def get_status(self) -> dict[str, Any]:
        """獲取監聽器狀態
        
        Returns:
            狀態資訊
        """
        return {
            "listener": "ExternalLearningListener",
            "is_running": self._is_running,
            "events_processed": self._processing_count,
            "broker_initialized": self._broker is not None,
            "connector_initialized": self._connector is not None,
        }


async def main():
    """主函數 - 獨立運行監聽器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA External Learning Listener")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日誌級別",
    )
    
    args = parser.parse_args()
    
    # 配置日誌
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 啟動監聽器
    listener = ExternalLearningListener()
    
    try:
        await listener.start_listening()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        await listener.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
