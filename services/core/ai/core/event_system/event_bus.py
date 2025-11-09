"""
AIVA Event-Driven Communication System v2.0
事件驅動通訊系統

實現高性能、可擴展的事件匯流排，支援模組間異步通訊、
事件溯源、消息路由和訂閱管理。

Author: AIVA Team  
Created: 2025-11-09
Version: 2.0.0
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Set, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import weakref
import traceback

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 事件系統核心數據類型 ====================

class EventPriority(Enum):
    """事件優先級"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class EventStatus(Enum):
    """事件狀態"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AIEvent:
    """統一 AI 事件格式"""
    # 事件識別 (必需參數)
    event_type: str
    source_module: str
    data: Dict[str, Any]
    
    # 可選參數 (有默認值)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_version: str = "v2.0"
    
    # 關聯資訊
    correlation_id: Optional[str] = None  # 關聯其他事件
    causation_id: Optional[str] = None    # 引發此事件的原因
    
    # 控制資訊
    priority: EventPriority = EventPriority.NORMAL
    ttl_seconds: float = 300.0  # 生存時間
    retry_count: int = 0
    max_retries: int = 3
    
    # 路由資訊
    target_modules: Optional[List[str]] = None  # 目標模組 (None = 廣播)
    
    # 元數據
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """檢查事件是否已過期"""
        elapsed = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds
    
    def can_retry(self) -> bool:
        """檢查是否可以重試"""
        return self.retry_count < self.max_retries

@dataclass
class EventSubscription:
    """事件訂閱"""
    # 必需參數
    module_name: str
    event_types: List[str]
    handler: Callable[[AIEvent], Any]
    
    # 可選參數
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filter_func: Optional[Callable[[AIEvent], bool]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True
    
    def matches(self, event: AIEvent) -> bool:
        """檢查事件是否匹配訂閱"""
        if not self.active:
            return False
            
        # 檢查事件類型
        type_match = any(
            event_type == event.event_type or 
            event.event_type.startswith(event_type + ".")
            for event_type in self.event_types
        )
        
        if not type_match:
            return False
            
        # 檢查過濾條件
        if self.filter_func and not self.filter_func(event):
            return False
            
        return True

@dataclass
class EventStats:
    """事件統計"""
    total_published: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    avg_processing_time: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# ==================== 事件存儲介面 ====================

class IEventStore(ABC):
    """事件存儲介面"""
    
    @abstractmethod
    async def store_event(self, event: AIEvent) -> None:
        """存儲事件"""
        pass
    
    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[AIEvent]:
        """獲取事件"""
        pass
    
    @abstractmethod
    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[AIEvent]:
        """根據類型獲取事件"""
        pass
    
    @abstractmethod
    async def get_events_by_correlation(self, correlation_id: str) -> List[AIEvent]:
        """根據關聯 ID 獲取事件"""
        pass

class MemoryEventStore(IEventStore):
    """記憶體事件存儲 (開發/測試用)"""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: Dict[str, AIEvent] = {}
        self.events_by_type: Dict[str, List[str]] = defaultdict(list)
        self.events_by_correlation: Dict[str, List[str]] = defaultdict(list)
        self.event_order = deque(maxlen=max_events)
        
    async def store_event(self, event: AIEvent) -> None:
        """存儲事件到記憶體"""
        self._handle_capacity_limit()
        self._store_new_event(event)
        self._update_event_indexes(event)
        logger.debug(f"事件已存儲: {event.event_type}:{event.event_id}")
    
    def _handle_capacity_limit(self) -> None:
        """處理容量限制，移除最舊事件"""
        if len(self.events) >= self.max_events and self.event_order:
            oldest_id = self.event_order.popleft()
            if oldest_id in self.events:
                old_event = self.events.pop(oldest_id)
                self._cleanup_event_indexes(oldest_id, old_event)
    
    def _cleanup_event_indexes(self, event_id: str, event: AIEvent) -> None:
        """清理舊事件的索引"""
        # 清理類型索引
        if event.event_type in self.events_by_type:
            if event_id in self.events_by_type[event.event_type]:
                self.events_by_type[event.event_type].remove(event_id)
        
        # 清理關聯索引
        if event.correlation_id and event.correlation_id in self.events_by_correlation:
            if event_id in self.events_by_correlation[event.correlation_id]:
                self.events_by_correlation[event.correlation_id].remove(event_id)
    
    def _store_new_event(self, event: AIEvent) -> None:
        """存儲新事件"""
        self.events[event.event_id] = event
        self.event_order.append(event.event_id)
    
    def _update_event_indexes(self, event: AIEvent) -> None:
        """更新事件索引"""
        # 更新類型索引
        self.events_by_type[event.event_type].append(event.event_id)
        
        # 更新關聯索引
        if event.correlation_id:
            self.events_by_correlation[event.correlation_id].append(event.event_id)
    
    async def get_event(self, event_id: str) -> Optional[AIEvent]:
        """獲取單個事件"""
        return self.events.get(event_id)
    
    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[AIEvent]:
        """根據類型獲取事件"""
        event_ids = self.events_by_type.get(event_type, [])[-limit:]
        return [self.events[eid] for eid in event_ids if eid in self.events]
    
    async def get_events_by_correlation(self, correlation_id: str) -> List[AIEvent]:
        """根據關聯 ID 獲取事件"""
        event_ids = self.events_by_correlation.get(correlation_id, [])
        return [self.events[eid] for eid in event_ids if eid in self.events]

# ==================== 事件發布者和訂閱者介面 ====================

class IEventPublisher(ABC):
    """事件發布介面"""
    
    @abstractmethod
    async def publish(self, event: AIEvent) -> None:
        """發布事件"""
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[AIEvent]) -> None:
        """批量發布事件"""
        pass

class IEventSubscriber(ABC):
    """事件訂閱介面"""
    
    @abstractmethod
    async def subscribe(
        self, 
        module_name: str, 
        event_types: List[str], 
        handler: Callable[[AIEvent], Any],
        filter_func: Optional[Callable[[AIEvent], bool]] = None
    ) -> str:
        """訂閱事件類型，返回訂閱 ID"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """取消訂閱"""
        pass

# ==================== 事件匯流排核心實現 ====================

class AIEventBus(IEventPublisher, IEventSubscriber):
    """AI 事件匯流排"""
    
    def __init__(self, event_store: Optional[IEventStore] = None):
        self.event_store = event_store or MemoryEventStore()
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.subscriptions_by_type: Dict[str, List[str]] = defaultdict(list)
        
        # 事件隊列 (按優先級分隊列)
        self.event_queues = {
            EventPriority.CRITICAL: asyncio.Queue(),
            EventPriority.HIGH: asyncio.Queue(), 
            EventPriority.NORMAL: asyncio.Queue(),
            EventPriority.LOW: asyncio.Queue()
        }
        
        # 統計資訊
        self.stats = EventStats()
        
        # 處理器控制
        self._running = False
        self._processors: List[asyncio.Task] = []
        
        # 弱引用，防止記憶體洩漏
        self._weak_handlers = weakref.WeakSet()
        
        logger.info("AI EventBus 初始化完成")
    
    def start(self, num_processors: int = 4):
        """啟動事件處理器"""
        if self._running:
            return
            
        self._running = True
        
        # 啟動事件處理器
        for _ in range(num_processors):
            processor = asyncio.create_task(self._process_events())
            self._processors.append(processor)
            
        logger.info(f"事件處理器已啟動 ({num_processors} 個處理器)")
    
    async def stop(self):
        """停止事件處理器"""
        if not self._running:
            return
            
        self._running = False
        
        # 取消所有處理器
        for processor in self._processors:
            processor.cancel()
        
        # 等待處理器結束
        await asyncio.gather(*self._processors, return_exceptions=True)
        self._processors.clear()
        
        logger.info("事件處理器已停止")
    
    async def publish(self, event: AIEvent) -> None:
        """發布單個事件"""
        try:
            # 存儲事件
            await self.event_store.store_event(event)
            
            # 加入對應優先級隊列
            await self.event_queues[event.priority].put(event)
            
            self.stats.total_published += 1
            self.stats.last_updated = datetime.now(timezone.utc)
            
            logger.debug(f"事件已發布: {event.event_type}:{event.event_id}")
            
        except Exception as e:
            logger.error(f"發布事件失敗: {str(e)}")
            raise
    
    async def publish_batch(self, events: List[AIEvent]) -> None:
        """批量發布事件"""
        for event in events:
            await self.publish(event)
    
    async def subscribe(
        self, 
        module_name: str, 
        event_types: List[str], 
        handler: Callable[[AIEvent], Any],
        filter_func: Optional[Callable[[AIEvent], bool]] = None
    ) -> str:
        """訂閱事件類型"""
        subscription = EventSubscription(
            module_name=module_name,
            event_types=event_types,
            handler=handler,
            filter_func=filter_func
        )
        
        # 存儲訂閱
        self.subscriptions[subscription.subscription_id] = subscription
        
        # 更新索引
        for event_type in event_types:
            self.subscriptions_by_type[event_type].append(subscription.subscription_id)
        
        # 註冊弱引用
        self._weak_handlers.add(handler)
        
        logger.info(f"模組 {module_name} 訂閱事件: {event_types}")
        
        return subscription.subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """取消訂閱"""
        if subscription_id not in self.subscriptions:
            logger.warning(f"訂閱 ID 不存在: {subscription_id}")
            return
        
        subscription = self.subscriptions.pop(subscription_id)
        
        # 清理索引
        for event_type in subscription.event_types:
            if subscription_id in self.subscriptions_by_type[event_type]:
                self.subscriptions_by_type[event_type].remove(subscription_id)
        
        logger.info(f"取消訂閱: {subscription.module_name} - {subscription.event_types}")
    
    async def _process_events(self):
        """事件處理器主迴圈"""
        logger.debug("事件處理器開始運行")
        
        while self._running:
            try:
                # 按優先級處理事件
                event = await self._get_next_event()
                
                if event is None:
                    continue
                
                await self._handle_event(event)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"事件處理器錯誤: {str(e)}")
                await asyncio.sleep(0.1)  # 錯誤後稍微暫停
        
        logger.debug("事件處理器結束運行")
    
    async def _get_next_event(self) -> Optional[AIEvent]:
        """獲取下一個要處理的事件 (按優先級)"""
        # 優先級順序
        priority_order = [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.NORMAL,
            EventPriority.LOW
        ]
        
        for priority in priority_order:
            try:
                # 非阻塞方式嘗試獲取事件
                event = self.event_queues[priority].get_nowait()
                return event
            except asyncio.QueueEmpty:
                continue
        
        # 如果所有隊列都空，等待一個事件
        # 使用 asyncio.wait_for 避免無限等待
        try:
            tasks = [
                asyncio.create_task(queue.get()) 
                for queue in self.event_queues.values()
            ]
            
            done, pending = await asyncio.wait(
                tasks, 
                timeout=1.0, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 取消未完成的任務
            for task in pending:
                task.cancel()
            
            if done:
                return await done.pop()
                
        except asyncio.TimeoutError:
            pass
        
        return None
    
    async def _handle_event(self, event: AIEvent):
        """處理單個事件"""
        start_time = time.time()
        
        try:
            # 檢查事件是否過期
            if event.is_expired():
                logger.warning(f"事件已過期: {event.event_type}:{event.event_id}")
                self.stats.total_timeout += 1
                return
            
            # 查找匹配的訂閱
            matching_subscriptions = self._find_matching_subscriptions(event)
            
            if not matching_subscriptions:
                logger.debug(f"沒有訂閱者的事件: {event.event_type}")
                return
            
            # 並行處理所有訂閱
            tasks = []
            for subscription in matching_subscriptions:
                task = asyncio.create_task(
                    self._execute_handler(subscription, event)
                )
                tasks.append(task)
            
            # 等待所有處理器完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 統計處理結果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count
            
            if error_count > 0:
                logger.warning(f"事件處理部分失敗: {event.event_type} - {error_count}/{len(results)}")
                self.stats.total_failed += error_count
            
            self.stats.total_processed += success_count
            
            # 更新平均處理時間
            processing_time = (time.time() - start_time) * 1000
            self._update_avg_processing_time(processing_time)
            
        except Exception as e:
            logger.error(f"處理事件時發生錯誤: {str(e)}")
            self.stats.total_failed += 1
    
    def _find_matching_subscriptions(self, event: AIEvent) -> List[EventSubscription]:
        """查找匹配的訂閱"""
        matching = []
        
        # 檢查直接匹配的事件類型
        subscription_ids = self.subscriptions_by_type.get(event.event_type, [])
        
        # 檢查通配符匹配 (例如 "ai.*" 匹配 "ai.perception.scan.completed")
        parts = event.event_type.split('.')
        for i in range(len(parts)):
            partial_type = '.'.join(parts[:i+1])
            subscription_ids.extend(self.subscriptions_by_type.get(partial_type, []))
        
        # 去重並檢查匹配條件
        for subscription_id in set(subscription_ids):
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                if subscription.matches(event):
                    matching.append(subscription)
        
        return matching
    
    async def _execute_handler(self, subscription: EventSubscription, event: AIEvent):
        """執行事件處理器"""
        try:
            # 呼叫處理器
            result = subscription.handler(event)
            
            # 如果返回的是協程或任務，等待完成
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Task):
                await result
                
        except Exception as e:
            logger.error(f"事件處理器執行失敗: {subscription.module_name} - {str(e)}")
            raise
    
    def _update_avg_processing_time(self, processing_time: float):
        """更新平均處理時間"""
        if self.stats.total_processed == 0:
            self.stats.avg_processing_time = processing_time
        else:
            # 指數移動平均
            alpha = 0.1
            self.stats.avg_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.stats.avg_processing_time
            )
    
    def get_stats(self) -> EventStats:
        """獲取事件統計"""
        self.stats.last_updated = datetime.now(timezone.utc)
        return self.stats
    
    async def get_event_history(self, event_type: str, limit: int = 100) -> List[AIEvent]:
        """獲取事件歷史"""
        return await self.event_store.get_events_by_type(event_type, limit)

# ==================== 事件路由器 ====================

class EventRouter:
    """事件路由器 - 支援複雜的路由邏輯"""
    
    def __init__(self, event_bus: AIEventBus):
        self.event_bus = event_bus
        self.routing_rules: List[Dict[str, Any]] = []
    
    def add_routing_rule(
        self, 
        rule_name: str,
        condition: Callable[[AIEvent], bool],
        action: Callable[[AIEvent], Union[AIEvent, List[AIEvent], None]]
    ):
        """添加路由規則"""
        rule = {
            'name': rule_name,
            'condition': condition,
            'action': action
        }
        self.routing_rules.append(rule)
        logger.info(f"路由規則已添加: {rule_name}")
    
    def route_event(self, event: AIEvent) -> List[AIEvent]:
        """路由事件 - 返回路由後的事件列表"""
        routed_events = [event]  # 預設包含原事件
        
        for rule in self.routing_rules:
            try:
                if rule['condition'](event):
                    result = rule['action'](event)
                    
                    if result is None:
                        continue
                    elif isinstance(result, AIEvent):
                        routed_events.append(result)
                    elif isinstance(result, list):
                        routed_events.extend(result)
                        
            except Exception as e:
                logger.error(f"路由規則執行失敗 {rule['name']}: {str(e)}")
        
        return routed_events

# ==================== 高階事件模式 ====================

class EventAggregator:
    """事件聚合器 - 聚合相關事件"""
    
    def __init__(self, event_bus: AIEventBus):
        self.event_bus = event_bus
        self.aggregation_rules: Dict[str, Dict[str, Any]] = {}
    
    def add_aggregation_rule(
        self,
        rule_name: str,
        trigger_event_types: List[str],
        aggregation_window: float,  # 秒
        aggregator_func: Callable[[List[AIEvent]], AIEvent]
    ):
        """添加聚合規則"""
        self.aggregation_rules[rule_name] = {
            'trigger_types': trigger_event_types,
            'window': aggregation_window,
            'aggregator': aggregator_func,
            'pending_events': [],
            'last_trigger': None
        }
        
        # 訂閱觸發事件
        for event_type in trigger_event_types:
            task = asyncio.create_task(self.event_bus.subscribe(
                f"aggregator_{rule_name}",
                [event_type],
                lambda e, rule=rule_name: self._handle_aggregation_event(rule, e)
            ))
            # 保存任務引用避免垃圾回收
            self.aggregation_rules[rule_name]['subscription_task'] = task
    
    async def _handle_aggregation_event(self, rule_name: str, event: AIEvent):
        """處理聚合事件"""
        if rule_name not in self.aggregation_rules:
            return
        
        rule = self.aggregation_rules[rule_name]
        now = datetime.now(timezone.utc)
        
        # 檢查聚合窗口
        if rule['last_trigger'] is None:
            rule['last_trigger'] = now
            rule['pending_events'] = [event]
        else:
            elapsed = (now - rule['last_trigger']).total_seconds()
            
            if elapsed < rule['window']:
                # 在窗口內，添加事件
                rule['pending_events'].append(event)
            else:
                # 窗口結束，執行聚合
                if rule['pending_events']:
                    aggregated_event = rule['aggregator'](rule['pending_events'])
                    await self.event_bus.publish(aggregated_event)
                
                # 重置窗口
                rule['last_trigger'] = now
                rule['pending_events'] = [event]

# ==================== 測試和示例 ====================

async def test_event_system():
    """測試事件系統"""
    
    print("🧪 測試 AI 事件系統")
    print("=" * 50)
    
    # 創建事件匯流排
    event_bus = AIEventBus()
    event_bus.start()
    
    # 測試事件計數器
    received_events = []
    
    async def test_handler(event: AIEvent):
        received_events.append(event)
        print(f"📨 接收到事件: {event.event_type} from {event.source_module}")
        await asyncio.sleep(0.1)  # 模擬處理時間
    
    # 訂閱事件
    subscription_id = await event_bus.subscribe(
        "test_module",
        ["ai.perception.*", "ai.knowledge.*"],
        test_handler
    )
    
    print(f"✅ 訂閱成功，ID: {subscription_id}")
    
    # 發布測試事件
    test_events = [
        AIEvent(
            event_type="ai.perception.scan.completed",
            source_module="perception_v2",
            data={"scan_results": "analysis complete"},
            priority=EventPriority.HIGH
        ),
        AIEvent(
            event_type="ai.knowledge.rag.search",
            source_module="knowledge_v2", 
            data={"query": "find similar documents", "results": 5},
            priority=EventPriority.NORMAL
        ),
        AIEvent(
            event_type="ai.decision.made",
            source_module="decision_v2",
            data={"decision": "execute action"},
            priority=EventPriority.CRITICAL
        )
    ]
    
    print(f"\n📤 發布 {len(test_events)} 個測試事件...")
    
    for event in test_events:
        await event_bus.publish(event)
        await asyncio.sleep(0.1)  # 稍微間隔
    
    # 等待處理完成
    await asyncio.sleep(1)
    
    # 檢查結果
    print("\n📊 測試結果:")
    print(f"  發布事件數: {len(test_events)}")
    print(f"  接收事件數: {len(received_events)}")
    print(f"  匹配的事件: {[e.event_type for e in received_events]}")
    
    # 獲取統計
    stats = event_bus.get_stats()
    print("\n📈 事件統計:")
    print(f"  總發布數: {stats.total_published}")
    print(f"  總處理數: {stats.total_processed}")
    print(f"  平均處理時間: {stats.avg_processing_time:.2f}ms")
    
    # 測試事件歷史
    history = await event_bus.get_event_history("ai.perception.scan.completed")
    print(f"\n📚 事件歷史 (ai.perception.scan.completed): {len(history)} 條")
    
    # 取消訂閱
    await event_bus.unsubscribe(subscription_id)
    print("✅ 取消訂閱成功")
    
    # 停止事件匯流排
    await event_bus.stop()
    print("✅ 事件系統已停止")

if __name__ == "__main__":
    asyncio.run(test_event_system())