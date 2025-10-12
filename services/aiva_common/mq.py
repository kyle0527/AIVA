from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aio_pika
else:
    try:
        import aio_pika
    except ModuleNotFoundError:  # optional dependency
        aio_pika = None

from .config import get_settings
from .enums import Topic


@dataclass(slots=True)
class MQMessage:
    body: bytes
    routing_key: str | None = None


class AbstractBroker(ABC):
    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def publish(self, topic: Topic, body: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    async def subscribe(self, topic: Topic) -> AsyncIterator[MQMessage]:
        """Return an async iterator of MQMessage for the given topic."""
        raise NotImplementedError
        yield  # Make this an async generator for type checking

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class RabbitBroker(AbstractBroker):
    def __init__(self) -> None:
        if aio_pika is None:
            raise RuntimeError("aio_pika is not installed")
        self._settings = get_settings()
        self._connection: Any = None
        self._channel: Any = None
        self._exchange: Any = None

    async def connect(self) -> None:
        assert aio_pika is not None
        self._connection = await aio_pika.connect_robust(self._settings.rabbitmq_url)
        self._channel = await self._connection.channel(publisher_confirms=True)
        await self._channel.set_qos(prefetch_count=10)
        self._exchange = await self._channel.declare_exchange(
            self._settings.exchange_name, aio_pika.ExchangeType.TOPIC
        )

    async def publish(self, topic: Topic, body: bytes) -> None:
        assert aio_pika is not None and self._exchange is not None
        msg = aio_pika.Message(body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT)
        await self._exchange.publish(msg, routing_key=str(topic))

    async def subscribe(self, topic: Topic) -> AsyncIterator[MQMessage]:
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
        if self._connection:
            await self._connection.close()


class InMemoryBroker(AbstractBroker):
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[bytes]] = {}

    async def connect(self) -> None:
        return None

    async def publish(self, topic: Topic, body: bytes) -> None:
        q = self._queues.setdefault(str(topic), asyncio.Queue())
        await q.put(body)

    async def subscribe(self, topic: Topic) -> AsyncIterator[MQMessage]:
        q = self._queues.setdefault(str(topic), asyncio.Queue())
        while True:
            body = await q.get()
            yield MQMessage(body=body, routing_key=str(topic))

    async def close(self) -> None:
        self._queues.clear()


async def get_broker() -> AbstractBroker:
    if aio_pika is not None:
        try:
            broker = RabbitBroker()
            await broker.connect()
            return broker
        except Exception:
            pass
    # fallback
    return InMemoryBroker()
