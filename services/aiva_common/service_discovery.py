"""
AIVA Service Discovery System
AIVA 服務發現系統

實施 TODO 項目 10: 創建服務發現機制
- 動態服務註冊和發現
- 服務健康檢查
- 負載均衡和故障轉移
- 服務路由和版本管理
- 分布式服務協調

特性：
1. 自動服務註冊和反註冊
2. 心跳檢測和健康監控
3. 服務元數據和標籤管理
4. 服務間依賴關係管理
5. 動態配置和服務更新
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp

from ..aiva_common.config_manager import ConfigManager, get_config_manager
from ..aiva_common.error_handling import (
    ErrorHandler,
)


class ServiceStatus(Enum):
    """服務狀態"""

    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """健康檢查類型"""

    HTTP = "http"
    TCP = "tcp"
    COMMAND = "command"
    CUSTOM = "custom"


@dataclass
class ServiceEndpoint:
    """服務端點"""

    host: str
    port: int
    protocol: str = "http"
    path: str = "/"

    @property
    def url(self) -> str:
        """獲取完整 URL"""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"

    def __str__(self) -> str:
        return self.url


@dataclass
class HealthCheck:
    """健康檢查配置"""

    type: HealthCheckType
    endpoint: ServiceEndpoint | None = None
    command: str | None = None
    interval: int = 30  # 檢查間隔 (秒)
    timeout: int = 10  # 超時時間 (秒)
    retries: int = 3  # 重試次數
    custom_checker: Callable[[], bool] | None = None


@dataclass
class ServiceMetadata:
    """服務元數據"""

    tags: set[str] = field(default_factory=set)
    version: str = "1.0.0"
    description: str = ""
    dependencies: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    environment: str = "production"
    region: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": list(self.tags),
            "version": self.version,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "capabilities": list(self.capabilities),
            "environment": self.environment,
            "region": self.region,
        }


@dataclass
class ServiceRegistration:
    """服務註冊信息"""

    service_id: str
    service_name: str
    endpoints: list[ServiceEndpoint]
    metadata: ServiceMetadata
    health_check: HealthCheck | None = None
    registration_time: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: ServiceStatus = ServiceStatus.STARTING
    weight: int = 100  # 負載均衡權重

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "endpoints": [
                {
                    "host": ep.host,
                    "port": ep.port,
                    "protocol": ep.protocol,
                    "path": ep.path,
                }
                for ep in self.endpoints
            ],
            "metadata": self.metadata.to_dict(),
            "registration_time": self.registration_time,
            "last_heartbeat": self.last_heartbeat,
            "status": self.status.value,
            "weight": self.weight,
        }


class ServiceRegistry:
    """
    服務註冊表

    管理所有已註冊的服務信息，提供服務發現功能
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._services: dict[str, ServiceRegistration] = {}
        self._service_index: dict[str, set[str]] = {}  # 服務名到 ID 的索引
        self._tag_index: dict[str, set[str]] = {}  # 標籤到服務 ID 的索引
        self._listeners: list[
            Callable[[str, ServiceRegistration, ServiceStatus], None]
        ] = []
        self._lock = asyncio.Lock()

    async def register_service(self, registration: ServiceRegistration) -> bool:
        """
        註冊服務

        Args:
            registration: 服務註冊信息

        Returns:
            是否註冊成功
        """
        async with self._lock:
            try:
                service_id = registration.service_id
                service_name = registration.service_name

                # 添加到主索引
                self._services[service_id] = registration

                # 更新服務名索引
                if service_name not in self._service_index:
                    self._service_index[service_name] = set()
                self._service_index[service_name].add(service_id)

                # 更新標籤索引
                for tag in registration.metadata.tags:
                    if tag not in self._tag_index:
                        self._tag_index[tag] = set()
                    self._tag_index[tag].add(service_id)

                self.logger.info(f"服務已註冊: {service_name} ({service_id})")

                # 通知監聽器
                self._notify_listeners(service_id, registration, ServiceStatus.STARTING)

                return True

            except Exception as e:
                self.logger.error(f"服務註冊失敗: {e}")
                return False

    async def deregister_service(self, service_id: str) -> bool:
        """
        反註冊服務

        Args:
            service_id: 服務 ID

        Returns:
            是否反註冊成功
        """
        async with self._lock:
            try:
                if service_id not in self._services:
                    return False

                registration = self._services[service_id]
                service_name = registration.service_name

                # 從主索引移除
                del self._services[service_id]

                # 更新服務名索引
                if service_name in self._service_index:
                    self._service_index[service_name].discard(service_id)
                    if not self._service_index[service_name]:
                        del self._service_index[service_name]

                # 更新標籤索引
                for tag in registration.metadata.tags:
                    if tag in self._tag_index:
                        self._tag_index[tag].discard(service_id)
                        if not self._tag_index[tag]:
                            del self._tag_index[tag]

                self.logger.info(f"服務已反註冊: {service_name} ({service_id})")

                # 通知監聽器
                self._notify_listeners(service_id, registration, ServiceStatus.STOPPED)

                return True

            except Exception as e:
                self.logger.error(f"服務反註冊失敗: {e}")
                return False

    async def update_service_status(
        self, service_id: str, status: ServiceStatus
    ) -> bool:
        """
        更新服務狀態

        Args:
            service_id: 服務 ID
            status: 新狀態

        Returns:
            是否更新成功
        """
        async with self._lock:
            if service_id not in self._services:
                return False

            registration = self._services[service_id]
            old_status = registration.status
            registration.status = status
            registration.last_heartbeat = time.time()

            if old_status != status:
                self.logger.info(
                    f"服務狀態已更新: {registration.service_name} ({service_id}) {old_status.value} -> {status.value}"
                )
                self._notify_listeners(service_id, registration, status)

            return True

    async def heartbeat(self, service_id: str) -> bool:
        """
        服務心跳

        Args:
            service_id: 服務 ID

        Returns:
            是否心跳成功
        """
        async with self._lock:
            if service_id not in self._services:
                return False

            self._services[service_id].last_heartbeat = time.time()
            return True

    def get_service(self, service_id: str) -> ServiceRegistration | None:
        """獲取服務信息"""
        return self._services.get(service_id)

    def discover_services(
        self,
        service_name: str | None = None,
        tags: set[str] | None = None,
        status: ServiceStatus | None = None,
        healthy_only: bool = True,
    ) -> list[ServiceRegistration]:
        """
        發現服務

        Args:
            service_name: 服務名過濾
            tags: 標籤過濾
            status: 狀態過濾
            healthy_only: 只返回健康的服務

        Returns:
            匹配的服務列表
        """
        candidates = set()

        # 根據服務名過濾
        if service_name:
            candidates = self._service_index.get(service_name, set()).copy()
        else:
            candidates = set(self._services.keys())

        # 根據標籤過濾
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self._tag_index:
                    if not tag_matches:
                        tag_matches = self._tag_index[tag].copy()
                    else:
                        tag_matches &= self._tag_index[tag]
            candidates &= tag_matches

        # 應用狀態和健康過濾
        results = []
        for service_id in candidates:
            service = self._services[service_id]

            # 狀態過濾
            if status and service.status != status:
                continue

            # 健康過濾
            if healthy_only and service.status != ServiceStatus.HEALTHY:
                continue

            results.append(service)

        # 按權重排序
        results.sort(key=lambda s: s.weight, reverse=True)
        return results

    def get_service_by_capability(self, capability: str) -> list[ServiceRegistration]:
        """根據能力發現服務"""
        results = []
        for service in self._services.values():
            if capability in service.metadata.capabilities:
                results.append(service)

        results.sort(key=lambda s: s.weight, reverse=True)
        return results

    def list_all_services(self) -> list[ServiceRegistration]:
        """列出所有服務"""
        return list(self._services.values())

    def get_service_count(self) -> int:
        """獲取服務總數"""
        return len(self._services)

    def get_healthy_service_count(self) -> int:
        """獲取健康服務數量"""
        return sum(
            1 for s in self._services.values() if s.status == ServiceStatus.HEALTHY
        )

    def add_status_listener(
        self, listener: Callable[[str, ServiceRegistration, ServiceStatus], None]
    ):
        """添加狀態變更監聽器"""
        self._listeners.append(listener)

    def remove_status_listener(
        self, listener: Callable[[str, ServiceRegistration, ServiceStatus], None]
    ):
        """移除狀態變更監聽器"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(
        self, service_id: str, registration: ServiceRegistration, status: ServiceStatus
    ):
        """通知狀態變更監聽器"""
        for listener in self._listeners:
            try:
                listener(service_id, registration, status)
            except Exception as e:
                self.logger.error(f"狀態監聽器執行失敗: {e}")

    def cleanup_stale_services(self, max_age: int = 300) -> int:
        """
        清理過期服務

        Args:
            max_age: 最大存活時間 (秒)

        Returns:
            清理的服務數量
        """
        current_time = time.time()
        stale_services = []

        for service_id, service in self._services.items():
            if current_time - service.last_heartbeat > max_age:
                stale_services.append(service_id)

        cleaned_count = 0
        for service_id in stale_services:
            asyncio.create_task(self.deregister_service(service_id))
            cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"清理了 {cleaned_count} 個過期服務")

        return cleaned_count


class HealthMonitor:
    """
    健康監控器

    定期檢查已註冊服務的健康狀態
    """

    def __init__(self, registry: ServiceRegistry):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = registry
        self.error_handler = ErrorHandler()
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

    async def start_monitoring(self, check_interval: int = 30):
        """
        開始健康監控

        Args:
            check_interval: 檢查間隔 (秒)
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(check_interval))
        self.logger.info(f"健康監控已啟動，檢查間隔: {check_interval}秒")

    async def stop_monitoring(self):
        """停止健康監控"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("健康監控已停止")

    async def _monitor_loop(self, check_interval: int):
        """監控循環"""
        while self._monitoring:
            try:
                services = self.registry.list_all_services()

                # 並行檢查所有服務
                health_check_tasks = []
                for service in services:
                    if service.health_check:
                        task = asyncio.create_task(self._check_service_health(service))
                        health_check_tasks.append(task)

                # 等待所有健康檢查完成
                if health_check_tasks:
                    await asyncio.gather(*health_check_tasks, return_exceptions=True)

                # 清理過期服務
                self.registry.cleanup_stale_services()

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"健康監控循環出錯: {e}")
                await asyncio.sleep(5)  # 出錯時短暫休息

    async def _check_service_health(self, service: ServiceRegistration):
        """檢查單個服務健康狀態"""
        try:
            health_check = service.health_check
            is_healthy = False

            if health_check.type == HealthCheckType.HTTP:
                is_healthy = await self._http_health_check(health_check)
            elif health_check.type == HealthCheckType.TCP:
                is_healthy = await self._tcp_health_check(health_check)
            elif health_check.type == HealthCheckType.COMMAND:
                is_healthy = await self._command_health_check(health_check)
            elif health_check.type == HealthCheckType.CUSTOM:
                is_healthy = await self._custom_health_check(health_check)

            # 更新服務狀態
            new_status = (
                ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
            )
            await self.registry.update_service_status(service.service_id, new_status)

        except Exception as e:
            self.logger.error(f"服務健康檢查失敗 {service.service_name}: {e}")
            await self.registry.update_service_status(
                service.service_id, ServiceStatus.UNKNOWN
            )

    async def _http_health_check(self, health_check: HealthCheck) -> bool:
        """HTTP 健康檢查"""
        if not health_check.endpoint:
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=health_check.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_check.endpoint.url) as response:
                    return 200 <= response.status < 400
        except Exception:
            return False

    async def _tcp_health_check(self, health_check: HealthCheck) -> bool:
        """TCP 健康檢查"""
        if not health_check.endpoint:
            return False

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    health_check.endpoint.host, health_check.endpoint.port
                ),
                timeout=health_check.timeout,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _command_health_check(self, health_check: HealthCheck) -> bool:
        """命令健康檢查"""
        if not health_check.command:
            return False

        try:
            process = await asyncio.create_subprocess_shell(
                health_check.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                await asyncio.wait_for(process.wait(), timeout=health_check.timeout)
                return process.returncode == 0
            except TimeoutError:
                process.kill()
                return False
        except Exception:
            return False

    async def _custom_health_check(self, health_check: HealthCheck) -> bool:
        """自定義健康檢查"""
        if not health_check.custom_checker:
            return False

        try:
            if asyncio.iscoroutinefunction(health_check.custom_checker):
                return await health_check.custom_checker()
            else:
                return health_check.custom_checker()
        except Exception:
            return False


class ServiceDiscoveryManager:
    """
    服務發現管理器

    提供完整的服務發現和管理功能
    """

    def __init__(self, config_manager: ConfigManager | None = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager or get_config_manager()

        # 核心組件
        self.registry = ServiceRegistry()
        self.health_monitor = HealthMonitor(self.registry)

        # 當前服務信息
        self.current_service: ServiceRegistration | None = None

        # 狀態
        self.is_running = False

    async def start(self):
        """啟動服務發現管理器"""
        if self.is_running:
            return

        try:
            # 啟動健康監控
            monitor_interval = self.config_manager.get(
                "service_discovery.health_check_interval", 30
            )
            await self.health_monitor.start_monitoring(monitor_interval)

            self.is_running = True
            self.logger.info("服務發現管理器啟動成功")

        except Exception as e:
            self.logger.error(f"服務發現管理器啟動失敗: {e}")
            raise

    async def stop(self):
        """停止服務發現管理器"""
        if not self.is_running:
            return

        try:
            # 反註冊當前服務
            if self.current_service:
                await self.registry.deregister_service(self.current_service.service_id)

            # 停止健康監控
            await self.health_monitor.stop_monitoring()

            self.is_running = False
            self.logger.info("服務發現管理器已停止")

        except Exception as e:
            self.logger.error(f"服務發現管理器停止失敗: {e}")

    async def register_current_service(
        self,
        service_name: str,
        endpoints: list[ServiceEndpoint],
        metadata: ServiceMetadata | None = None,
        health_check: HealthCheck | None = None,
    ) -> str:
        """
        註冊當前服務

        Args:
            service_name: 服務名稱
            endpoints: 服務端點列表
            metadata: 服務元數據
            health_check: 健康檢查配置

        Returns:
            服務 ID
        """
        service_id = f"{service_name}-{uuid.uuid4().hex[:8]}"

        if metadata is None:
            metadata = ServiceMetadata(
                version="2.0.0",
                description=f"AIVA {service_name} Service",
                environment=self.config_manager.get("system.environment", "production"),
            )

        registration = ServiceRegistration(
            service_id=service_id,
            service_name=service_name,
            endpoints=endpoints,
            metadata=metadata,
            health_check=health_check,
        )

        success = await self.registry.register_service(registration)
        if success:
            self.current_service = registration
            # 設置為健康狀態
            await self.registry.update_service_status(service_id, ServiceStatus.HEALTHY)
            self.logger.info(f"當前服務已註冊: {service_name} ({service_id})")

        return service_id

    def discover_service(
        self,
        service_name: str,
        tags: set[str] | None = None,
        load_balance: bool = True,
    ) -> ServiceRegistration | None:
        """
        發現單個服務實例

        Args:
            service_name: 服務名稱
            tags: 標籤過濾
            load_balance: 是否進行負載均衡選擇

        Returns:
            服務實例，如果沒找到返回 None
        """
        services = self.registry.discover_services(
            service_name=service_name, tags=tags, healthy_only=True
        )

        if not services:
            return None

        if load_balance:
            # 基於權重的隨機選擇
            import random

            total_weight = sum(s.weight for s in services)
            if total_weight == 0:
                return random.choice(services)

            rand = random.randint(0, total_weight - 1)
            current_weight = 0
            for service in services:
                current_weight += service.weight
                if rand < current_weight:
                    return service

        return services[0]

    def discover_services(
        self,
        service_name: str | None = None,
        tags: set[str] | None = None,
        capability: str | None = None,
    ) -> list[ServiceRegistration]:
        """發現多個服務實例"""
        if capability:
            return self.registry.get_service_by_capability(capability)
        else:
            return self.registry.discover_services(
                service_name=service_name, tags=tags, healthy_only=True
            )

    async def send_heartbeat(self):
        """發送心跳"""
        if self.current_service:
            await self.registry.heartbeat(self.current_service.service_id)

    def get_discovery_status(self) -> dict[str, Any]:
        """獲取服務發現狀態"""
        return {
            "is_running": self.is_running,
            "current_service": (
                self.current_service.to_dict() if self.current_service else None
            ),
            "total_services": self.registry.get_service_count(),
            "healthy_services": self.registry.get_healthy_service_count(),
            "service_index_size": len(self.registry._service_index),
            "tag_index_size": len(self.registry._tag_index),
        }


# 全局服務發現管理器實例
_global_discovery_manager: ServiceDiscoveryManager | None = None


def get_service_discovery_manager(
    config_manager: ConfigManager | None = None,
) -> ServiceDiscoveryManager:
    """獲取全局服務發現管理器實例"""
    global _global_discovery_manager

    if _global_discovery_manager is None:
        _global_discovery_manager = ServiceDiscoveryManager(config_manager)

    return _global_discovery_manager
