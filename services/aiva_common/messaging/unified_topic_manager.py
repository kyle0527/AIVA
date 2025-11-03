"""
AIVA 統一 Topic 管理器 - V2 架構增強版
=======================================

實現枚舉化 Topic 管理，統一所有 MQ 通信主題命名規則
基於 core_schema_sot.yaml 生成，與 Schema 保持一致

功能:
- 統一 Topic 命名規則
- 支援多種路由策略  
- 向後兼容性維護
- 自動映射與驗證
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass
import logging

from services.aiva_common.enums import Topic, ModuleName
from services.aiva_common.schemas.generated.messaging import AivaMessage

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """路由策略枚舉"""
    BROADCAST = "broadcast"      # 廣播到所有訂閱者
    DIRECT = "direct"           # 直接發送到指定模組
    FANOUT = "fanout"          # 扇出到多個目標
    ROUND_ROBIN = "round_robin" # 輪詢負載均衡


@dataclass
class TopicMetadata:
    """Topic 元資料"""
    category: str              # 類別 (tasks, results, events, commands)
    module: str               # 所屬模組
    action: str               # 動作類型
    description: str          # 描述
    default_routing: RoutingStrategy  # 預設路由策略
    priority: int = 5         # 預設優先級


class UnifiedTopicManager:
    """統一 Topic 管理器
    
    負責:
    1. Topic 命名規則統一
    2. 路由策略管理
    3. 向後兼容性維護
    4. Topic 映射與驗證
    """
    
    def __init__(self):
        self._topic_metadata = self._initialize_topic_metadata()
        self._legacy_mappings = self._initialize_legacy_mappings()
    
    def _initialize_topic_metadata(self) -> Dict[str, TopicMetadata]:
        """初始化 Topic 元資料"""
        return {
            # Scan Topics
            Topic.TASK_SCAN_START: TopicMetadata(
                category="tasks", module="scan", action="start",
                description="掃描任務啟動", 
                default_routing=RoutingStrategy.DIRECT, priority=7
            ),
            Topic.RESULTS_SCAN_COMPLETED: TopicMetadata(
                category="results", module="scan", action="completed",
                description="掃描結果完成",
                default_routing=RoutingStrategy.BROADCAST, priority=6
            ),
            
            # Function Topics  
            Topic.TASK_FUNCTION_START: TopicMetadata(
                category="tasks", module="function", action="start",
                description="函式測試任務啟動",
                default_routing=RoutingStrategy.DIRECT, priority=7
            ),
            Topic.RESULTS_FUNCTION_COMPLETED: TopicMetadata(
                category="results", module="function", action="completed", 
                description="函式測試結果完成",
                default_routing=RoutingStrategy.BROADCAST, priority=6
            ),
            
            # AI Topics
            Topic.TASK_AI_TRAINING_START: TopicMetadata(
                category="tasks", module="ai", action="training.start",
                description="AI 訓練任務啟動",
                default_routing=RoutingStrategy.DIRECT, priority=8
            ),
            Topic.EVENT_AI_EXPERIENCE_CREATED: TopicMetadata(
                category="events", module="ai", action="experience.created",
                description="AI 經驗創建事件", 
                default_routing=RoutingStrategy.BROADCAST, priority=5
            ),
            
            # General Topics
            Topic.FINDING_DETECTED: TopicMetadata(
                category="findings", module="core", action="detected",
                description="漏洞發現通知",
                default_routing=RoutingStrategy.BROADCAST, priority=9
            ),
            Topic.COMMAND_TASK_CANCEL: TopicMetadata(
                category="commands", module="core", action="task.cancel", 
                description="任務取消命令",
                default_routing=RoutingStrategy.DIRECT, priority=8
            ),
        }
    
    def _initialize_legacy_mappings(self) -> Dict[str, str]:
        """初始化舊版 Topic 映射"""
        return {
            # 舊版 routing key 映射到新版 Topic
            "scan.start": Topic.TASK_SCAN_START,
            "scan.result": Topic.RESULTS_SCAN_COMPLETED,
            "function.start": Topic.TASK_FUNCTION_START,
            "function.result": Topic.RESULTS_FUNCTION_COMPLETED,
            "finding.new": Topic.FINDING_DETECTED,
            "task.cancel": Topic.COMMAND_TASK_CANCEL,
        }
    
    def get_topic_metadata(self, topic: str) -> Optional[TopicMetadata]:
        """獲取 Topic 元資料"""
        return self._topic_metadata.get(topic)
    
    def get_routing_strategy(self, topic: str) -> RoutingStrategy:
        """獲取 Topic 的預設路由策略"""
        metadata = self.get_topic_metadata(topic)
        return metadata.default_routing if metadata else RoutingStrategy.BROADCAST
    
    def get_priority(self, topic: str) -> int:
        """獲取 Topic 的預設優先級"""
        metadata = self.get_topic_metadata(topic)
        return metadata.priority if metadata else 5
    
    def normalize_topic(self, legacy_topic: str) -> str:
        """標準化 Topic（支援舊版映射）"""
        # 如果是新版 Topic，直接返回
        if legacy_topic in self._topic_metadata:
            return legacy_topic
        
        # 如果是舊版，進行映射
        if legacy_topic in self._legacy_mappings:
            new_topic = self._legacy_mappings[legacy_topic]
            logger.info(f"🔄 Topic 映射: {legacy_topic} -> {new_topic}")
            return new_topic
        
        # 未知 Topic，記錄警告
        logger.warning(f"⚠️  未知 Topic: {legacy_topic}")
        return legacy_topic
    
    def validate_topic(self, topic: str) -> bool:
        """驗證 Topic 是否有效"""
        return topic in self._topic_metadata or topic in self._legacy_mappings
    
    def get_topics_by_category(self, category: str) -> List[str]:
        """根據類別獲取 Topic 列表"""
        return [
            topic for topic, metadata in self._topic_metadata.items()
            if metadata.category == category
        ]
    
    def get_topics_by_module(self, module: str) -> List[str]:
        """根據模組獲取 Topic 列表"""
        return [
            topic for topic, metadata in self._topic_metadata.items()
            if metadata.module == module
        ]
    
    def create_enhanced_message(
        self,
        topic: str,
        payload: Dict,
        source_module: str,
        target_module: Optional[str] = None,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> AivaMessage:
        """創建增強版 AivaMessage
        
        自動填充路由策略、優先級等元資料
        """
        import uuid
        from datetime import datetime
        from services.aiva_common.schemas.generated.base_types import MessageHeader
        
        # 標準化 Topic
        normalized_topic = self.normalize_topic(topic)
        
        # 獲取 Topic 元資料
        metadata = self.get_topic_metadata(normalized_topic)
        routing_strategy = metadata.default_routing if metadata else RoutingStrategy.BROADCAST
        priority = metadata.priority if metadata else 5
        
        # 創建 MessageHeader
        header = MessageHeader(
            message_id=str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            correlation_id=correlation_id,
            source_module=source_module,
            target_module=target_module,
            timestamp=datetime.now(),
            version="1.1"
        )
        
        # 創建 AivaMessage
        return AivaMessage(
            header=header,
            topic=normalized_topic,
            schema_version="1.1",
            source_module=source_module,
            target_module=target_module,
            trace_id=header.trace_id,
            correlation_id=correlation_id,
            routing_strategy=routing_strategy.value,
            priority=priority,
            payload=payload,
            metadata=kwargs.get('metadata', {}),
            ttl_seconds=kwargs.get('ttl_seconds')
        )
    
    def get_migration_report(self) -> Dict[str, any]:
        """生成遷移報告"""
        return {
            "total_topics": len(self._topic_metadata),
            "legacy_mappings": len(self._legacy_mappings),
            "categories": list({m.category for m in self._topic_metadata.values()}),
            "modules": list({m.module for m in self._topic_metadata.values()}),
            "routing_strategies": {
                strategy.value: len([
                    m for m in self._topic_metadata.values() 
                    if m.default_routing == strategy
                ])
                for strategy in RoutingStrategy
            }
        }


# 全域 Topic 管理器實例
topic_manager = UnifiedTopicManager()