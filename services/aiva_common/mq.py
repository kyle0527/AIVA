"""AIVA Message Queue (MQ) abstraction layer.

This module provides a unified interface for RabbitMQ message broker operations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import get_settings
from .enums import Topic

if TYPE_CHECKING:
    import aio_pika
else:
    # Handle optional aio_pika dependency for runtime
    try:
        import aio_pika  # type: ignore[no-redef]
    except ModuleNotFoundError:  # optional dependency
        aio_pika = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MQMessage:
    """Message Queue 消息封裝類別。

    Attributes:
        body: 消息主體（字節）
        routing_key: 路由鍵，用於消息分發

    """

    body: bytes
    routing_key: str | None = None


class AbstractBroker(ABC):
    """Message Broker 抽象基類。

    定義消息代理的標準介面，支援連接、發布、訂閱和關閉操作。
    """

    @abstractmethod
    async def connect(self) -> None:
        """建立與消息代理的連接。

        Raises:
            NotImplementedError: 子類必須實現此方法

        """
        raise NotImplementedError

    @abstractmethod
    async def publish(self, topic: Topic, body: bytes) -> None:
        """發布消息到指定主題。

        Args:
            topic: 消息主題
            body: 消息內容（字節）

        Raises:
            NotImplementedError: 子類必須實現此方法

        """
        raise NotImplementedError

    @abstractmethod
    async def subscribe(self, topic: Topic) -> AsyncIterator[MQMessage]:
        """訂閱指定主題的消息。

        Args:
            topic: 要訂閱的主題

        Yields:
            MQMessage: 接收到的消息

        Raises:
            NotImplementedError: 子類必須實現此方法

        """
        raise NotImplementedError
        # Note: yield statement removed - subclasses must implement async
        # generator

    @abstractmethod
    async def close(self) -> None:
        """關閉與消息代理的連接。

        Raises:
            NotImplementedError: 子類必須實現此方法

        """
        raise NotImplementedError


class RabbitBroker(AbstractBroker):
    """RabbitMQ 消息代理實現。

    使用 aio_pika 提供完整的 RabbitMQ 功能，支援持久化、確認機制等。

    Attributes:
        _settings: AIVA 配置設置
        _connection: RabbitMQ 連接
        _channel: RabbitMQ 通道
        _exchange: RabbitMQ 交換機

    """

    def __init__(self) -> None:
        """初始化 RabbitMQ Broker。

        Raises:
            RuntimeError: 當 aio_pika 未安裝時

        """
        if aio_pika is None:
            raise RuntimeError("aio_pika is not installed")
        self._settings = get_settings()
        self._connection: Any = None
        self._channel: Any = None
        self._exchange: Any = None

    async def connect(self) -> None:
        """建立 RabbitMQ 連接並配置通道和交換機。

        配置包括:
        - 使用 robust 連接（自動重連）
        - 啟用發布者確認
        - 設置 QoS prefetch_count
        - 聲明 TOPIC 類型的交換機

        """
        assert aio_pika is not None
        self._connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
        self._channel = await self._connection.channel(publisher_confirms=True)
        await self._channel.set_qos(prefetch_count=10)
        self._exchange = await self._channel.declare_exchange(
            self._settings.exchange_name, aio_pika.ExchangeType.TOPIC
        )

    async def publish(self, topic: Topic, body: bytes) -> None:
        """發布持久化消息到 RabbitMQ。

        Args:
            topic: 消息主題（作為路由鍵）
            body: 消息內容（字節）

        """
        assert aio_pika is not None and self._exchange is not None
        msg = aio_pika.Message(body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT)
        await self._exchange.publish(msg, routing_key=str(topic))

    async def subscribe(self, topic: Topic) -> AsyncIterator[MQMessage]:
        """訂閱 RabbitMQ 主題消息。

        自動聲明臨時隊列並綁定到指定主題。

        Args:
            topic: 要訂閱的主題

        Yields:
            MQMessage: 接收到的消息

        """
        if self._channel is None or self._exchange is None:
            await self.connect()
        assert (
            aio_pika is not None
            and self._channel is not None
            and self._exchange is not None
        )
        queue = await self._channel.declare_queue(
            exclusive=True, durable=False, auto_delete=True
        )
        await queue.bind(self._exchange, routing_key=str(topic))
        async with queue.iterator() as it:
            async for message in it:
                async with message.process():
                    yield MQMessage(
                        body=message.body,
                        routing_key=getattr(message, "routing_key", None),
                    )

    async def close(self) -> None:
        """關閉 RabbitMQ 連接。"""
        if self._connection:
            await self._connection.close()


async def get_broker() -> AbstractBroker:
    """獲取可用的消息代理實例。

    必須成功連接 RabbitMQ，否則拋出異常。

    Returns:
        AbstractBroker: RabbitBroker 實例

    Raises:
        RuntimeError: aio_pika 模組未安裝
        Exception: RabbitMQ 連接失敗

    """
    if aio_pika is None:
        raise RuntimeError(
            "aio_pika module not available. "
            "Install aio_pika: pip install aio-pika"
        )
    
    broker = RabbitBroker()
    await broker.connect()
    return broker
