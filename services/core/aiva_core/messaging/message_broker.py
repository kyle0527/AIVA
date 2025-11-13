"""Message Broker - 消息代理

統一管理 RabbitMQ 連接和消息路由
"""

import asyncio
from collections.abc import Callable
from contextlib import suppress
import json
import logging
from typing import Any

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)

from services.aiva_common.config import get_settings
from services.aiva_common.enums import ModuleName
from services.aiva_common.schemas import AivaMessage

logger = logging.getLogger(__name__)


class MessageBroker:
    """消息代理

    負責 RabbitMQ 連接管理和消息發布/訂閱
    """

    def __init__(self, module_name: ModuleName = ModuleName.CORE) -> None:
        """初始化消息代理

        Args:
            module_name: 當前模組名稱
        """
        self.module_name = module_name
        self.config = get_settings()
        self.connection: AbstractRobustConnection | None = None
        self.channel: AbstractChannel | None = None
        self.exchanges: dict[str, AbstractExchange] = {}
        self.queues: dict[str, AbstractQueue] = {}
        self.consumers: dict[str, asyncio.Task[Any]] = {}

        logger.info(f"MessageBroker initialized for module {module_name.value}")

    async def connect(self) -> None:
        """建立 RabbitMQ 連接"""
        if self.connection and not self.connection.is_closed:
            logger.warning("Already connected to RabbitMQ")
            return

        try:
            # 從配置獲取 RabbitMQ 連接信息
            rabbitmq_url = self.config.rabbitmq_url

            self.connection = await aio_pika.connect_robust(rabbitmq_url)
            self.channel = await self.connection.channel()

            # 設置 QoS
            await self.channel.set_qos(prefetch_count=10)

            logger.info(f"Connected to RabbitMQ at {rabbitmq_url}")

            # 聲明交換機
            await self._declare_exchanges()

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def _declare_exchanges(self) -> None:
        """聲明所有需要的交換機"""
        if not self.channel:
            raise RuntimeError("Channel not initialized")

        exchange_names = [
            "aiva.tasks",  # 任務交換機
            "aiva.results",  # 結果交換機
            "aiva.events",  # 事件交換機
            "aiva.feedback",  # 反饋交換機
        ]

        for name in exchange_names:
            exchange = await self.channel.declare_exchange(
                name=name,
                type=aio_pika.ExchangeType.TOPIC,
                durable=True,
            )
            self.exchanges[name] = exchange
            logger.debug(f"Declared exchange: {name}")

    async def publish_message(
        self,
        exchange_name: str,
        routing_key: str,
        message: AivaMessage | dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        """發布消息

        Args:
            exchange_name: 交換機名稱
            routing_key: 路由鍵
            message: 消息內容
            correlation_id: 關聯 ID（用於請求-響應模式）
        """
        if not self.channel:
            raise RuntimeError("Channel not initialized")

        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found")

        # 轉換為字典
        if isinstance(message, AivaMessage):
            message_dict = message.model_dump()
        else:
            message_dict = message

        # 序列化為 JSON
        body = json.dumps(message_dict, default=str).encode()

        # 創建消息
        aio_message = aio_pika.Message(
            body=body,
            content_type="application/json",
            correlation_id=correlation_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        # 發布消息
        await exchange.publish(
            message=aio_message,
            routing_key=routing_key,
        )

        logger.debug(
            f"Published message to {exchange_name} with routing key {routing_key}"
        )

    async def subscribe(
        self,
        queue_name: str,
        routing_keys: list[str],
        exchange_name: str,
        callback: Callable[[AbstractIncomingMessage], Any],
        auto_ack: bool = False,
    ) -> None:
        """訂閱消息

        Args:
            queue_name: 隊列名稱
            routing_keys: 路由鍵列表
            exchange_name: 交換機名稱
            callback: 消息處理回調函數
            auto_ack: 是否自動確認
        """
        if not self.channel:
            raise RuntimeError("Channel not initialized")

        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found")

        # 聲明隊列
        queue = await self.channel.declare_queue(
            name=queue_name,
            durable=True,
            arguments={"x-message-ttl": 86400000},  # 24小時 TTL
        )
        self.queues[queue_name] = queue

        # 綁定路由鍵
        for routing_key in routing_keys:
            await queue.bind(exchange=exchange, routing_key=routing_key)
            logger.debug(f"Bound queue {queue_name} to {routing_key}")

        # 開始消費
        await queue.consume(callback, no_ack=auto_ack)

        logger.info(
            f"Subscribed to {queue_name} on {exchange_name} "
            f"with routing keys: {routing_keys}"
        )

    async def create_rpc_client(
        self,
        exchange_name: str,
        timeout: float = 30.0,
    ) -> RPCClient:
        """創建 RPC 客戶端

        Args:
            exchange_name: 交換機名稱
            timeout: 超時時間

        Returns:
            RPC 客戶端
        """
        if not self.channel:
            raise RuntimeError("Channel not initialized")

        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found")

        return RPCClient(
            channel=self.channel,
            exchange=exchange,
            timeout=timeout,
        )

    async def disconnect(self) -> None:
        """斷開連接"""
        # 停止所有消費者
        for _consumer_tag, task in self.consumers.items():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        # 關閉通道和連接
        if self.channel:
            await self.channel.close()

        if self.connection:
            await self.connection.close()

        logger.info("Disconnected from RabbitMQ")


class RPCClient:
    """RPC 客戶端

    實現請求-響應模式的消息通信
    """

    def __init__(
        self,
        channel: AbstractChannel,
        exchange: AbstractExchange,
        timeout: float = 30.0,
    ) -> None:
        """初始化 RPC 客戶端

        Args:
            channel: 通道
            exchange: 交換機
            timeout: 超時時間
        """
        self.channel = channel
        self.exchange = exchange
        self.timeout = timeout
        self.futures: dict[str, asyncio.Future] = {}
        self.callback_queue: AbstractQueue | None = None

    async def setup(self) -> None:
        """設置回調隊列"""
        # 聲明臨時回調隊列
        self.callback_queue = await self.channel.declare_queue(
            exclusive=True,
            auto_delete=True,
        )

        # 開始消費回調消息
        await self.callback_queue.consume(self._on_response, no_ack=True)

    async def _on_response(self, message: AbstractIncomingMessage) -> None:
        """處理響應消息

        Args:
            message: 響應消息
        """
        correlation_id = message.correlation_id
        if not correlation_id or correlation_id not in self.futures:
            logger.warning(
                f"Received response with unknown correlation_id: {correlation_id}"
            )
            return

        future = self.futures.pop(correlation_id)

        try:
            body = json.loads(message.body.decode())
            future.set_result(body)
        except Exception as e:
            future.set_exception(e)

    async def call(
        self,
        routing_key: str,
        message: dict[str, Any],
        correlation_id: str,
    ) -> dict[str, Any]:
        """發送 RPC 請求並等待響應

        Args:
            routing_key: 路由鍵
            message: 請求消息
            correlation_id: 關聯 ID

        Returns:
            響應消息

        Raises:
            asyncio.TimeoutError: 超時
        """
        if not self.callback_queue:
            await self.setup()

        # 創建 Future
        future: asyncio.Future = asyncio.Future()
        self.futures[correlation_id] = future

        # 發送請求
        body = json.dumps(message, default=str).encode()
        aio_message = aio_pika.Message(
            body=body,
            content_type="application/json",
            correlation_id=correlation_id,
            reply_to=self.callback_queue.name if self.callback_queue else None,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await self.exchange.publish(
            message=aio_message,
            routing_key=routing_key,
        )

        # 等待響應
        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result
        except TimeoutError:
            self.futures.pop(correlation_id, None)
            raise


# ==================== 事件驅動系統增強 (整合自 AI 模組) ====================

import uuid
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Union


class EventPriority(Enum):
    """事件優先級 (整合自 AI 模組)"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AIVAEvent:
    """AIVA 統一事件格式 (整合自 AI 模組)"""
    # 事件識別 (必需參數)
    event_type: str
    source_module: str
    data: Dict[str, Any]
    
    # 可選參數 (有默認值)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_version: str = "v2.0"
    
    # 關聯資訊
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    # 控制資訊
    priority: EventPriority = EventPriority.NORMAL
    ttl_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    
    # 路由資訊
    target_modules: Optional[List[str]] = None
    
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
    """事件訂閱 (整合自 AI 模組)"""
    module_name: str
    event_types: List[str]
    handler: Callable[[AIVAEvent], Any]
    
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filter_func: Optional[Callable[[AIVAEvent], bool]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True
    
    def matches(self, event: AIVAEvent) -> bool:
        """檢查事件是否匹配訂閱"""
        if not self.active:
            return False
        
        # 檢查事件類型匹配 (支持通配符)
        type_match = any(
            self._match_pattern(pattern, event.event_type) 
            for pattern in self.event_types
        )
        
        if not type_match:
            return False
        
        # 應用自定義過濾器
        if self.filter_func and not self.filter_func(event):
            return False
        
        return True
    
    def _match_pattern(self, pattern: str, event_type: str) -> bool:
        """模式匹配 (支持通配符)"""
        if pattern == "*":
            return True
        
        # 支持 prefix.* 格式的通配符
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return event_type.startswith(prefix)
        
        return pattern == event_type


class EnhancedMessageBroker(MessageBroker):
    """增強的消息代理 (整合事件驅動系統)
    
    在原有 RabbitMQ 功能基礎上添加高性能事件匯流排
    """
    
    def __init__(self, module_name: ModuleName = ModuleName.CORE):
        super().__init__(module_name)
        
        # 事件系統組件
        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._event_store: deque[AIVAEvent] = deque(maxlen=10000)  # 內存事件存儲
        self._processing = False
        self._event_queue: asyncio.Queue[AIVAEvent] = asyncio.Queue()
        self._processors: List[asyncio.Task] = []
        
        logger.info(f"EnhancedMessageBroker initialized with event system for {module_name.value}")
    
    async def start_event_system(self, num_processors: int = 2) -> None:
        """啟動事件處理系統
        
        Args:
            num_processors: 事件處理器數量
        """
        if self._processing:
            return
        
        self._processing = True
        
        # 啟動事件處理器
        for i in range(num_processors):
            processor = asyncio.create_task(self._process_events(f"processor-{i}"))
            self._processors.append(processor)
        
        logger.info(f"Event system started with {num_processors} processors")
    
    async def stop_event_system(self) -> None:
        """停止事件處理系統"""
        self._processing = False
        
        # 停止所有處理器
        for processor in self._processors:
            processor.cancel()
        
        await asyncio.gather(*self._processors, return_exceptions=True)
        self._processors.clear()
        
        logger.info("Event system stopped")
    
    async def publish_event(self, event: AIVAEvent) -> None:
        """發布事件
        
        Args:
            event: 要發布的事件
        """
        # 存儲事件到內存
        self._event_store.append(event)
        
        # 添加到處理隊列
        await self._event_queue.put(event)
        
        logger.debug(f"Event published: {event.event_type} from {event.source_module}")
    
    async def subscribe_event(self, 
                             module_name: str,
                             event_types: List[str],
                             handler: Callable[[AIVAEvent], Any],
                             filter_func: Optional[Callable[[AIVAEvent], bool]] = None) -> str:
        """訂閱事件
        
        Args:
            module_name: 模組名稱
            event_types: 事件類型列表 (支持通配符)
            handler: 事件處理函數
            filter_func: 可選的過濾函數
            
        Returns:
            訂閱ID
        """
        subscription = EventSubscription(
            module_name=module_name,
            event_types=event_types,
            handler=handler,
            filter_func=filter_func
        )
        
        # 將訂閱添加到所有匹配的事件類型
        for event_type in event_types:
            self._subscriptions[event_type].append(subscription)
        
        logger.info(f"Event subscription created: {module_name} -> {event_types}")
        return subscription.subscription_id
    
    async def unsubscribe_event(self, subscription_id: str) -> bool:
        """取消事件訂閱
        
        Args:
            subscription_id: 訂閱ID
            
        Returns:
            是否成功取消訂閱
        """
        found = False
        
        for event_type_subs in self._subscriptions.values():
            for sub in event_type_subs[:]:  # 創建副本以避免修改中的列表
                if sub.subscription_id == subscription_id:
                    event_type_subs.remove(sub)
                    found = True
        
        if found:
            logger.info(f"Event subscription cancelled: {subscription_id}")
        
        return found
    
    async def _process_events(self, processor_name: str) -> None:
        """事件處理器
        
        Args:
            processor_name: 處理器名稱
        """
        logger.info(f"Event processor {processor_name} started")
        
        try:
            while self._processing:
                try:
                    # 獲取事件 (帶超時以支持優雅停止)
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    
                    # 檢查事件是否過期
                    if event.is_expired():
                        logger.warning(f"Event expired: {event.event_type} (age: {event.timestamp})")
                        continue
                    
                    # 處理事件
                    await self._handle_event(event, processor_name)
                    
                except asyncio.TimeoutError:
                    continue  # 超時，繼續下一輪循環
                except Exception as e:
                    logger.error(f"Event processor {processor_name} error: {e}")
                    
        except asyncio.CancelledError:
            logger.info(f"Event processor {processor_name} cancelled")
            raise  # 重新拋出 CancelledError
        finally:
            logger.info(f"Event processor {processor_name} stopped")
    
    async def _handle_event(self, event: AIVAEvent, processor_name: str) -> None:
        """處理單個事件
        
        Args:
            event: 要處理的事件
            processor_name: 處理器名稱
        """
        handlers_called = 0
        
        # 找到所有匹配的訂閱
        matching_subs = []
        for event_type, subscriptions in self._subscriptions.items():
            for sub in subscriptions:
                if sub.matches(event):
                    matching_subs.append(sub)
        
        # 按優先級排序處理器
        matching_subs.sort(key=lambda s: event.priority.value, reverse=True)
        
        # 調用所有匹配的處理器
        for sub in matching_subs:
            try:
                if asyncio.iscoroutinefunction(sub.handler):
                    await sub.handler(event)
                else:
                    sub.handler(event)
                handlers_called += 1
                
            except Exception as e:
                logger.error(
                    f"Event handler error in {sub.module_name}: {e} "
                    f"(event: {event.event_type})"
                )
        
        if handlers_called > 0:
            logger.debug(
                f"Event {event.event_type} processed by {handlers_called} handlers "
                f"(processor: {processor_name})"
            )
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """獲取事件系統統計信息
        
        Returns:
            統計信息字典
        """
        total_subscriptions = sum(len(subs) for subs in self._subscriptions.values())
        
        return {
            "total_events_stored": len(self._event_store),
            "pending_events": self._event_queue.qsize(),
            "total_subscriptions": total_subscriptions,
            "event_types_subscribed": len(self._subscriptions),
            "processors_running": len([p for p in self._processors if not p.done()]),
            "processing_active": self._processing
        }


# 全域增強消息代理實例
_enhanced_broker: Optional[EnhancedMessageBroker] = None


def get_enhanced_message_broker(module_name: ModuleName = ModuleName.CORE) -> EnhancedMessageBroker:
    """獲取全域增強消息代理實例
    
    Args:
        module_name: 模組名稱
        
    Returns:
        增強消息代理實例
    """
    global _enhanced_broker
    if _enhanced_broker is None:
        _enhanced_broker = EnhancedMessageBroker(module_name)
    return _enhanced_broker


# 便捷函數
async def publish_aiva_event(event_type: str, source_module: str, data: Dict[str, Any], 
                            priority: EventPriority = EventPriority.NORMAL,
                            target_modules: Optional[List[str]] = None) -> None:
    """發布 AIVA 事件的便捷函數
    
    Args:
        event_type: 事件類型
        source_module: 源模組名稱
        data: 事件數據
        priority: 事件優先級
        target_modules: 目標模組列表
    """
    broker = get_enhanced_message_broker()
    event = AIVAEvent(
        event_type=event_type,
        source_module=source_module,
        data=data,
        priority=priority,
        target_modules=target_modules
    )
    await broker.publish_event(event)


async def subscribe_aiva_events(module_name: str, 
                               event_types: List[str],
                               handler: Callable[[AIVAEvent], Any]) -> str:
    """訂閱 AIVA 事件的便捷函數
    
    Args:
        module_name: 模組名稱
        event_types: 事件類型列表
        handler: 事件處理函數
        
    Returns:
        訂閱ID
    """
    broker = get_enhanced_message_broker()
    return await broker.subscribe_event(module_name, event_types, handler)
