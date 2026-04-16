"""Learning System Event Listener - 統一經驗學習事件監聽器

監聽 LOG_RESULTS_ALL 事件並觸發學習流程，實現 AI 從執行經驗中學習

架構原則：
- 外部監聽器使用同步處理（異步只在 AI 內部）
- 接收外部模組的執行結果（XSS/SQLi/SSRF/...）
- 使用 ModuleKnowledgeManager 匹配已知情況
- 傳遞給 ExternalLoopConnector 進行學習

數據流：
    外部模組 → MQ (LOG_RESULTS_ALL) → ExternalLearningListener
        → ModuleKnowledgeManager (匹配知識)
        → ExternalLoopConnector (學習處理)
        → AI 核心

使用方式:
    from services.core.aiva_core.cognitive_core.learning_system.event_listener import ExternalLearningListener
    
    listener = ExternalLearningListener()
    listener.start_listening()  # 同步阻塞
"""

import logging
from typing import Any

from aiva_common.enums import Topic
from aiva_common.schemas import AivaMessage, FindingPayload

logger = logging.getLogger(__name__)


class ExternalLearningListener:
    """外部學習監聽器（同步版本）
    
    職責：
    1. 監聽 LOG_RESULTS_ALL 事件（外部模組執行結果）
    2. 使用 ModuleKnowledgeManager 匹配已知情況
    3. 提取執行數據並轉換為學習格式
    4. 觸發 ExternalLoopConnector 處理（同步）
    5. AI 內部的異步處理由 connector 負責
    
    架構原則：
    - 外部接口同步（符合 AIVA 架構規範）
    - 異步只在 AI 內部使用
    """
    
    def __init__(self):
        """初始化監聽器"""
        self._broker = None
        self._connector = None
        self._knowledge_manager = None
        self._is_running = False
        self._processing_count = 0
        
        logger.info("ExternalLearningListener initialized (sync mode)")
    
    @property
    def broker(self):
        """延遲加載 MessageBroker"""
        if self._broker is None:
            # 修復: 正確的相對導入路徑 (learning_system → aiva_core → service_backbone)
            from ...service_backbone.messaging.message_broker import MessageBroker
            self._broker = MessageBroker()
        return self._broker
    
    @property
    def connector(self):
        """延遲加載 ExternalLoopConnector"""
        if self._connector is None:
            # 修復: 正確的相對導入路徑 (learning_system → cognitive_core)
            from ..external_loop_connector import ExternalLoopConnector
            self._connector = ExternalLoopConnector()
        return self._connector
    
    @property
    def knowledge_manager(self):
        """延遲加載 ModuleKnowledgeManager"""
        if self._knowledge_manager is None:
            from .knowledge.module_knowledge_manager import ModuleKnowledgeManager
            self._knowledge_manager = ModuleKnowledgeManager(
                knowledge_base_dir='./data/knowledge_base'
            )
            self._knowledge_manager.load_all_knowledge()
        return self._knowledge_manager
    
    def start_listening(self):
        """開始監聽執行結果（同步阻塞）
        
        監聽 LOG_RESULTS_ALL 事件，接收外部模組的執行結果
        這個方法會阻塞當前線程，持續監聽事件
        """
        logger.info("=" * 60)
        logger.info("👂 External Learning Listener Starting (Sync Mode)...")
        logger.info("=" * 60)
        
        try:
            # ✅ 修復：監聽正確的 Topic
            logger.info(f"📡 Subscribing to: {Topic.LOG_RESULTS_ALL}")
            
            # 同步訂閱（MessageBroker 提供同步接口）
            self.broker.subscribe_sync(
                topic=Topic.LOG_RESULTS_ALL,
                callback=self._on_result_received,
                queue_name="external_learning.results"
            )
            
            self._is_running = True
            logger.info("✅ Listening for LOG_RESULTS_ALL events")
            logger.info("   Modules: XSS, SQLi, SSRF, IDOR, ...")
            logger.info("   Waiting for execution results to learn from...")
            logger.info("   Press Ctrl+C to stop")
            logger.info("=" * 60)
            
            # 保持運行（同步阻塞）
            while self._is_running:
                pass  # 消息處理由 broker 回調觸發
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Received shutdown signal")
            self.stop_listening()
        except Exception as e:
            logger.error(f"❌ Listener failed: {e}", exc_info=True)
            raise
    
    def stop_listening(self):
        """停止監聽（同步）"""
        logger.info("🛑 Stopping External Learning Listener...")
        self._is_running = False
        
        # 斷開連接（同步）
        if self._broker:
            self.broker.disconnect_sync()
        
        logger.info(f"✅ Listener stopped (processed {self._processing_count} events)")
    
    def _on_result_received(self, message: AivaMessage):
        """MessageBroker 同步回調
        
        接收外部模組的執行結果（FindingPayload）
        
        Args:
            message: AIVA 消息（包含 FindingPayload）
        """
        self._processing_count += 1
        
        try:
            logger.info("=" * 60)
            logger.info(f"📥 Received Result #{self._processing_count}")
            logger.info(f"   Source: {message.header.source_module}")
            logger.info(f"   Topic: {message.topic}")
            logger.info("=" * 60)
            
            # 解析 FindingPayload
            if isinstance(message.payload, FindingPayload):
                self._process_finding(message)
            else:
                logger.warning(f"   ⚠️  Unknown payload type: {type(message.payload)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process message: {e}", exc_info=True)
    
    def _process_finding(self, message: AivaMessage):
        """處理漏洞發現結果（同步）
        
        流程：
        1. 構建執行上下文（sent_data + received_data）
        2. 使用 ModuleKnowledgeManager 匹配已知情況
        3. 生成學習建議
        4. 傳遞給 ExternalLoopConnector
        
        Args:
            message: AIVA 消息（包含 FindingPayload）
        """
        try:
            finding: FindingPayload = message.payload
            
            # 1. 構建執行上下文
            from .knowledge.module_knowledge_manager import ExecutionContext
            
            context = ExecutionContext(
                module_name=str(message.header.source_module),
                target_url=finding.location.url,
                sent_data={
                    'method': finding.location.method,
                    'parameter': finding.location.parameter,
                    'payload': finding.evidence.payload if finding.evidence else None,
                },
                received_data={
                    'status_code': finding.evidence.response.status_code if finding.evidence else None,
                    'vulnerability_type': finding.vulnerability.name,
                    'severity': finding.vulnerability.severity,
                    'confidence': finding.vulnerability.confidence,
                },
                timestamp=finding.discovered_at or '',
                execution_id=finding.finding_id
            )
            
            logger.info(f"   Module: {context.module_name}")
            logger.info(f"   Target: {context.target_url}")
            logger.info(f"   Vuln: {finding.vulnerability.name} ({finding.vulnerability.severity})")
            
            # 2. 使用 ModuleKnowledgeManager 匹配
            logger.info("\n🔍 Matching knowledge base...")
            recommendation = self.knowledge_manager.generate_recommendation(context)
            
            if recommendation.knowledge_match and recommendation.knowledge_match.matched:
                logger.info(f"✅ Matched: {recommendation.knowledge_match.scenario_name}")
                logger.info(f"   Confidence: {recommendation.confidence:.2f}")
                logger.info(f"   Category: {recommendation.knowledge_match.category}")
            elif recommendation.requires_rag:
                logger.info("❓ Unknown situation, RAG triggered")
                logger.info(f"   Query: {recommendation.rag_query}")
            else:
                logger.info("⚠️  No match, no RAG")
            
            # 3. 傳遞給 ExternalLoopConnector（AI 內部異步處理）
            logger.info("\n🧠 Triggering learning process...")
            self.connector.process_external_finding(
                context=context,
                recommendation=recommendation
            )
            
            logger.info("✅ Processing complete\n")
            
        except Exception as e:
            logger.error(f"❌ Failed to process finding: {e}", exc_info=True)
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取監聽器統計信息
        
        Returns:
            統計資訊字典
        """
        stats = {
            "listener": "ExternalLearningListener",
            "mode": "sync",
            "is_running": self._is_running,
            "events_processed": self._processing_count,
            "broker_initialized": self._broker is not None,
            "connector_initialized": self._connector is not None,
            "knowledge_manager_initialized": self._knowledge_manager is not None,
        }
        
        # 添加知識庫統計
        if self._knowledge_manager:
            stats["knowledge_stats"] = self.knowledge_manager.get_statistics()
        
        return stats
    
def main():
    """主函數 - 獨立運行監聽器（同步）"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AIVA External Learning Listener (Sync Mode)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日誌級別",
    )
    parser.add_argument(
        "--knowledge-base",
        default="./data/knowledge_base",
        help="知識庫目錄路徑",
    )
    
    args = parser.parse_args()
    
    # 配置日誌
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 啟動監聽器（同步阻塞）
    listener = ExternalLearningListener()
    
    try:
        listener.start_listening()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        listener.stop_listening()
        
        # 顯示最終統計
        stats = listener.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("📊 Final Statistics:")
        logger.info(f"   Events processed: {stats['events_processed']}")
        if 'knowledge_stats' in stats:
            logger.info(f"   Modules loaded: {stats['knowledge_stats']['total_modules']}")
            logger.info(f"   Scenarios: {stats['knowledge_stats']['total_scenarios']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

