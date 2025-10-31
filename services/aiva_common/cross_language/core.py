"""
AIVA Cross-Language Communication Framework
統一的跨語言通訊核心模組

此模組提供：
1. gRPC 服務基礎框架
2. Protocol Buffers 消息處理
3. 跨語言錯誤映射
4. 統一的配置管理
5. 連接池與重試機制
"""

import asyncio
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import google.protobuf.message
from google.protobuf.json_format import MessageToDict, ParseDict
from grpc import aio as grpc_aio

from ..utils.logging import setup_logger
from ..utils.retry import RetryConfig, retry_async

# 類型定義
T = TypeVar("T")
MessageType = TypeVar("MessageType", bound=google.protobuf.message.Message)

logger = setup_logger(__name__)


@dataclass
class CrossLanguageConfig:
    """跨語言通訊配置"""

    grpc_host: str = "localhost"
    grpc_port: int = 50051
    max_workers: int = 10
    max_message_length: int = 4 * 1024 * 1024  # 4MB
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 5000
    keepalive_permit_without_calls: bool = True
    max_connection_idle_ms: int = 300000  # 5 minutes
    max_connection_age_ms: int = 1800000  # 30 minutes
    retry_config: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_attempts=3, delay=1.0, max_delay=10.0, backoff_factor=2.0
        )
    )


class LanguageAdapter(Generic[T]):
    """語言適配器基類"""

    def __init__(self, language: str):
        self.language = language
        self.logger = setup_logger(f"adapter.{language}")

    async def serialize(self, data: T) -> bytes:
        """序列化數據為 Protocol Buffers"""
        raise NotImplementedError

    async def deserialize(self, data: bytes, message_type: type) -> T:
        """反序列化 Protocol Buffers 數據"""
        raise NotImplementedError

    async def convert_error(self, error: Exception) -> dict[str, Any]:
        """將本地錯誤轉換為統一格式"""
        raise NotImplementedError


class PythonAdapter(LanguageAdapter[google.protobuf.message.Message]):
    """Python 語言適配器"""

    def __init__(self):
        super().__init__("python")

    async def serialize(self, data: google.protobuf.message.Message) -> bytes:
        """序列化 Protobuf 消息"""
        try:
            return data.SerializeToString()
        except Exception as e:
            self.logger.error(f"Python serialization failed: {e}")
            raise

    async def deserialize(
        self, data: bytes, message_type: type
    ) -> google.protobuf.message.Message:
        """反序列化 Protobuf 消息"""
        try:
            message = message_type()
            message.ParseFromString(data)
            return message
        except Exception as e:
            self.logger.error(f"Python deserialization failed: {e}")
            raise

    async def convert_error(self, error: Exception) -> dict[str, Any]:
        """轉換 Python 錯誤"""
        error_info = {
            "language": "python",
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "module": getattr(error, "__module__", "unknown"),
        }

        # 映射常見 Python 錯誤到 AIVA 錯誤碼
        error_mapping = {
            "ImportError": "ERROR_CODE_PYTHON_IMPORT_ERROR",
            "SyntaxError": "ERROR_CODE_PYTHON_SYNTAX_ERROR",
            "RuntimeError": "ERROR_CODE_PYTHON_RUNTIME_ERROR",
            "FileNotFoundError": "ERROR_CODE_FILE_NOT_FOUND",
            "PermissionError": "ERROR_CODE_FILE_ACCESS_DENIED",
            "TimeoutError": "ERROR_CODE_TIMEOUT",
            "ConnectionError": "ERROR_CODE_CONNECTION_REFUSED",
        }

        error_info["aiva_error_code"] = error_mapping.get(
            type(error).__name__, "ERROR_CODE_UNKNOWN"
        )

        return error_info


class ConnectionPool:
    """gRPC 連接池管理"""

    def __init__(self, config: CrossLanguageConfig):
        self.config = config
        self.logger = setup_logger("connection_pool")
        self._channels: dict[str, grpc_aio.Channel] = {}
        self._lock = asyncio.Lock()

    async def get_channel(self, target: str) -> grpc_aio.Channel:
        """獲取或創建 gRPC 通道"""
        async with self._lock:
            if target not in self._channels:
                self._channels[target] = await self._create_channel(target)
            return self._channels[target]

    async def _create_channel(self, target: str) -> grpc_aio.Channel:
        """創建新的 gRPC 通道"""
        options = [
            ("grpc.keepalive_time_ms", self.config.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
            (
                "grpc.keepalive_permit_without_calls",
                self.config.keepalive_permit_without_calls,
            ),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ("grpc.max_connection_idle_ms", self.config.max_connection_idle_ms),
            ("grpc.max_connection_age_ms", self.config.max_connection_age_ms),
            ("grpc.max_send_message_length", self.config.max_message_length),
            ("grpc.max_receive_message_length", self.config.max_message_length),
        ]

        channel = grpc_aio.insecure_channel(target, options=options)
        self.logger.info(f"Created gRPC channel to {target}")
        return channel

    async def close_all(self):
        """關閉所有連接"""
        async with self._lock:
            for target, channel in self._channels.items():
                await channel.close()
                self.logger.info(f"Closed gRPC channel to {target}")
            self._channels.clear()


class MessageRegistry:
    """消息類型註冊表"""

    def __init__(self):
        self._message_types: dict[str, type] = {}
        self.logger = setup_logger("message_registry")

    def register(self, message_name: str, message_type: type):
        """註冊消息類型"""
        self._message_types[message_name] = message_type
        self.logger.debug(f"Registered message type: {message_name}")

    def get_type(self, message_name: str) -> type | None:
        """獲取消息類型"""
        return self._message_types.get(message_name)

    def list_types(self) -> list[str]:
        """列出所有註冊的消息類型"""
        return list(self._message_types.keys())


class CrossLanguageService:
    """跨語言服務核心"""

    def __init__(self, config: CrossLanguageConfig | None = None):
        self.config = config or CrossLanguageConfig()
        self.logger = setup_logger("cross_language_service")

        # 初始化組件
        self.connection_pool = ConnectionPool(self.config)
        self.message_registry = MessageRegistry()
        self.adapters: dict[str, LanguageAdapter] = {}

        # 註冊 Python 適配器
        self.register_adapter("python", PythonAdapter())

        # 服務狀態
        self._running = False
        self._server: grpc_aio.Server | None = None

    def register_adapter(self, language: str, adapter: LanguageAdapter):
        """註冊語言適配器"""
        self.adapters[language] = adapter
        self.logger.info(f"Registered adapter for {language}")

    async def start_server(self, servicers: list[Any]):
        """啟動 gRPC 服務器"""
        if self._running:
            self.logger.warning("Server is already running")
            return

        self._server = grpc_aio.server(
            ThreadPoolExecutor(max_workers=self.config.max_workers)
        )

        # 添加服務
        for servicer in servicers:
            # 這裡需要根據實際的 servicer 類型來添加
            # 例如: add_AIServiceServicer_to_server(servicer, self._server)
            pass

        listen_addr = f"{self.config.grpc_host}:{self.config.grpc_port}"
        self._server.add_insecure_port(listen_addr)

        await self._server.start()
        self._running = True
        self.logger.info(f"gRPC server started on {listen_addr}")

    async def stop_server(self):
        """停止 gRPC 服務器"""
        if not self._running or not self._server:
            return

        await self._server.stop(grace=30)
        await self.connection_pool.close_all()
        self._running = False
        self.logger.info("gRPC server stopped")

    @asynccontextmanager
    async def get_client_stub(self, stub_class: type, target: str):
        """獲取客戶端存根"""
        channel = await self.connection_pool.get_channel(target)
        stub = stub_class(channel)
        try:
            yield stub
        finally:
            # 連接由連接池管理，不需要在這裡關閉
            pass

    async def call_service(
        self,
        stub_class: type,
        method_name: str,
        request: google.protobuf.message.Message,
        target: str,
        timeout: float = 30.0,
    ) -> google.protobuf.message.Message:
        """調用遠程服務"""
        async with self.get_client_stub(stub_class, target) as stub:
            method = getattr(stub, method_name)

            try:
                response = await retry_async(
                    method, self.config.retry_config, request, timeout=timeout
                )
                return response
            except Exception as e:
                self.logger.error(f"Service call failed: {e}")
                raise

    async def broadcast_message(
        self,
        message: google.protobuf.message.Message,
        targets: list[str],
        stub_class: type,
        method_name: str,
    ) -> dict[str, google.protobuf.message.Message | Exception]:
        """廣播消息到多個目標"""
        results = {}

        tasks = []
        for target in targets:
            task = asyncio.create_task(
                self._safe_call_service(stub_class, method_name, message, target)
            )
            tasks.append((target, task))

        for target, task in tasks:
            try:
                results[target] = await task
            except Exception as e:
                results[target] = e
                self.logger.error(f"Broadcast to {target} failed: {e}")

        return results

    async def _safe_call_service(
        self,
        stub_class: type,
        method_name: str,
        request: google.protobuf.message.Message,
        target: str,
    ) -> google.protobuf.message.Message:
        """安全調用服務（用於廣播）"""
        try:
            return await self.call_service(stub_class, method_name, request, target)
        except Exception as e:
            # 轉換錯誤為統一格式
            adapter = self.adapters.get("python")
            if adapter:
                error_info = await adapter.convert_error(e)
                self.logger.error(f"Service call error: {error_info}")
            raise

    def to_dict(self, message: google.protobuf.message.Message) -> dict[str, Any]:
        """將 Protobuf 消息轉換為字典"""
        return MessageToDict(message, preserving_proto_field_name=True)

    def from_dict(
        self, data: dict[str, Any], message_type: type
    ) -> google.protobuf.message.Message:
        """從字典創建 Protobuf 消息"""
        message = message_type()
        ParseDict(data, message)
        return message

    async def health_check(self, target: str) -> bool:
        """健康檢查"""
        try:
            channel = await self.connection_pool.get_channel(target)
            await channel.channel_ready()
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed for {target}: {e}")
            return False

    def get_service_info(self) -> dict[str, Any]:
        """獲取服務信息"""
        return {
            "running": self._running,
            "config": {
                "host": self.config.grpc_host,
                "port": self.config.grpc_port,
                "max_workers": self.config.max_workers,
            },
            "adapters": list(self.adapters.keys()),
            "connections": len(self.connection_pool._channels),
            "message_types": len(self.message_registry._message_types),
        }


# 全局服務實例
_global_service: CrossLanguageService | None = None


def get_cross_language_service() -> CrossLanguageService:
    """獲取全局跨語言服務實例"""
    global _global_service
    if _global_service is None:
        _global_service = CrossLanguageService()
    return _global_service


def init_cross_language_service(
    config: CrossLanguageConfig | None = None,
) -> CrossLanguageService:
    """初始化全局跨語言服務"""
    global _global_service
    _global_service = CrossLanguageService(config)
    return _global_service


# 便利函數
async def call_ai_service(
    request_data: dict[str, Any], target: str = "localhost:50051"
) -> dict[str, Any]:
    """調用 AI 服務的便利函數"""
    service = get_cross_language_service()
    # 這裡需要根據實際的 AI 服務接口來實現
    # 例如：
    # from .generated.aiva_services_pb2 import AIRequest
    # from .generated.aiva_services_pb2_grpc import AIServiceStub
    #
    # request = service.from_dict(request_data, AIRequest)
    # response = await service.call_service(AIServiceStub, "ProcessTask", request, target)
    # return service.to_dict(response)
    pass


async def call_security_scanner(
    request_data: dict[str, Any], target: str = "localhost:50052"
) -> dict[str, Any]:
    """調用安全掃描服務的便利函數"""
    service = get_cross_language_service()
    # 類似於 AI 服務的實現
    pass


# 裝飾器
def cross_language_method(timeout: float = 30.0):
    """跨語言方法裝飾器"""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except TimeoutError:
                logger.error(f"Method {func.__name__} timed out after {timeout}s")
                raise
            except Exception as e:
                service = get_cross_language_service()
                adapter = service.adapters.get("python")
                if adapter:
                    error_info = await adapter.convert_error(e)
                    logger.error(f"Method {func.__name__} failed: {error_info}")
                raise
            finally:
                execution_time = time.time() - start_time
                logger.debug(
                    f"Method {func.__name__} executed in {execution_time:.3f}s"
                )

        return wrapper

    return decorator


if __name__ == "__main__":
    # 示例用法
    async def main():
        service = init_cross_language_service()

        # 啟動服務器
        await service.start_server([])

        try:
            # 保持服務運行
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await service.stop_server()

    asyncio.run(main())
