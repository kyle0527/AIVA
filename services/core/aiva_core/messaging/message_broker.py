"""
Message Broker - 消息代理

統一管理 RabbitMQ 連接和消息路由
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
import json
import logging
from typing import Any

import aio_pika
from aio_pika import Channel, Connection, Exchange, Queue
from aio_pika.abc import AbstractIncomingMessage


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
        self.connection: Connection | None = None
        self.channel: Channel | None = None
        self.exchanges: dict[str, Exchange] = {}
        self.queues: dict[str, Queue] = {}
        self.consumers: dict[str, asyncio.Task] = {}

        logger.info(f"MessageBroker initialized for module {module_name.value}")

    async def connect(self) -> None:
        """建立 RabbitMQ 連接"""
        if self.connection and not self.connection.is_closed:
            logger.warning("Already connected to RabbitMQ")
            return

        try:
            # 從配置獲取 RabbitMQ 連接信息
            rabbitmq_url = self.config.get(
                "rabbitmq_url", "amqp://guest:guest@localhost:5672/"
            )

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
        channel: Channel,
        exchange: Exchange,
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
        self.callback_queue: Queue | None = None

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
