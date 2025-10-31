"""AIVA Core Service Coordinator - 核心服務協調器
從 aiva_core_v2 遷移到核心模組

AI驅動的系統核心引擎和跨模組協調中心
這是當AI組件不在時的備用核心服務實現
"""

import logging
import time
from typing import Any

# 從 aiva_common 導入共享組件
from ...aiva_common.config_manager import (
    ConfigChangeEvent,
    get_config_manager,
)
from ...aiva_common.cross_language import (
    error_handler,
    get_cross_language_service,
)
from ...aiva_common.monitoring import (
    MetricType,
    get_monitoring_service,
    trace_operation,
)
from ...aiva_common.monitoring_log_handler import setup_monitoring_logging
from ...aiva_common.security import get_security_manager
from ...aiva_common.security_middleware import (
    create_security_middleware,
)
from .command_router import (
    CommandContext,
    ExecutionResult,
    get_command_router,
)
from .context_manager import get_context_manager
from .execution_planner import get_execution_planner


class AIVACoreServiceCoordinator:
    """AIVA 核心服務協調器

    這是當AI組件不可用時的備用核心服務實現，
    提供基本的命令處理、上下文管理和執行編排功能
    """

    def __init__(self):
        self.logger = logging.getLogger("aiva_core_service_coordinator")

        # 基本狀態
        self.is_running = False
        self.service_id: str | None = None
        self.startup_time: float | None = None

        # 核心組件
        self._initialize_core_components()

        # 共享服務（從 aiva_common）
        self._initialize_shared_services()

        # 設置監控和配置
        self._setup_monitoring_and_config()

    def _initialize_core_components(self):
        """初始化核心組件"""
        try:
            # 核心三大件：路由、上下文、執行
            self.command_router = get_command_router()
            self.context_manager = get_context_manager()
            self.execution_planner = get_execution_planner()

            self.logger.info("核心組件初始化完成")

        except Exception as e:
            self.logger.error(f"核心組件初始化失敗: {e}")
            raise

    def _initialize_shared_services(self):
        """初始化共享服務"""
        try:
            # 配置管理器 - 最高優先級
            self.config_manager = get_config_manager()
            self.config_manager.add_change_listener(self._on_config_changed)

            # 跨語言服務
            self.cross_lang_service = get_cross_language_service()

            # 監控系統
            self.monitoring_service = get_monitoring_service()

            # 安全框架
            self.security_manager = get_security_manager()
            self.security_middleware = create_security_middleware(self.security_manager)

            self.logger.info("共享服務初始化完成")

        except Exception as e:
            self.logger.error(f"共享服務初始化失敗: {e}")
            raise

    def _setup_monitoring_and_config(self):
        """設置監控和配置"""
        try:
            # 設置監控日誌
            setup_monitoring_logging(
                "aiva_core_coordinator", monitoring_service=self.monitoring_service
            )

            # 應用初始配置
            self._apply_initial_config()

            # 配置安全中間件
            self._configure_security_middleware()

            self.logger.info("監控和配置設置完成")

        except Exception as e:
            self.logger.error(f"監控和配置設置失敗: {e}")

    def _apply_initial_config(self):
        """應用初始配置"""
        try:
            # 獲取服務配置
            service_config = self.config_manager.get_config("core_service", {})

            # 設置服務ID
            self.service_id = service_config.get(
                "service_id", f"aiva_core_{int(time.time())}"
            )

            # 設置日誌級別
            log_level = service_config.get("log_level", "INFO")
            logging.getLogger().setLevel(
                getattr(logging, log_level.upper(), logging.INFO)
            )

            self.logger.info(f"應用初始配置完成，服務ID: {self.service_id}")

        except Exception as e:
            self.logger.error(f"應用初始配置失敗: {e}")

    def _configure_security_middleware(self):
        """配置安全中間件"""
        try:
            # 配置 CORS
            self.security_middleware.configure_cors(
                allowed_origins=["http://localhost:3000", "https://aiva.app"],
                allowed_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allowed_headers=["Content-Type", "Authorization", "X-API-Key"],
                allow_credentials=True,
            )

            # 添加白名單路徑
            whitelist_paths = ["/health", "/status", "/metrics", "/docs"]
            for path in whitelist_paths:
                self.security_middleware.add_whitelist_path(path)

            self.logger.info("安全中間件配置完成")

        except Exception as e:
            self.logger.error(f"安全中間件配置失敗: {e}")

    def _on_config_changed(self, event: ConfigChangeEvent):
        """配置變更事件處理"""
        self.logger.info(f"配置變更: {event.key} = {event.new_value}")

        # 根據配置變更調整行為
        if event.key.startswith("core_service."):
            self._apply_initial_config()
        elif event.key.startswith("security."):
            self._configure_security_middleware()

    async def start(self):
        """啟動核心服務協調器"""
        if self.is_running:
            self.logger.warning("服務已在運行中")
            return

        try:
            self.startup_time = time.time()

            # 啟動共享服務
            await self._start_shared_services()

            # 啟動核心組件
            await self._start_core_components()

            self.is_running = True
            self.logger.info(f"AIVA 核心服務協調器啟動成功 (ID: {self.service_id})")

            # 記錄啟動指標
            self.monitoring_service.record_metric(
                "core_service.startup", 1, MetricType.COUNTER
            )

        except Exception as e:
            self.logger.error(f"服務啟動失敗: {e}")
            await self._cleanup_on_failure()
            raise

    async def _start_shared_services(self):
        """啟動共享服務"""
        # 啟動跨語言服務
        await self.cross_lang_service.start_server([])

        # 啟動監控服務
        await self.monitoring_service.start()

        # 啟動安全管理器
        await self.security_manager.start()

        self.logger.debug("共享服務啟動完成")

    async def _start_core_components(self):
        """啟動核心組件"""
        # 核心組件通常不需要顯式啟動，它們是無狀態的
        # 但可以在這裡進行一些初始化檢查

        # 檢查命令路由器
        available_commands = self.command_router.get_available_commands()
        self.logger.info(f"命令路由器就緒，支持 {len(available_commands)} 個命令")

        # 檢查上下文管理器
        stats = await self.context_manager.get_context_stats()
        self.logger.debug(f"上下文管理器就緒: {stats}")

        # 檢查執行計劃器
        exec_stats = await self.execution_planner.get_execution_stats()
        self.logger.debug(f"執行計劃器就緒: {exec_stats}")

        self.logger.debug("核心組件啟動檢查完成")

    async def stop(self):
        """停止核心服務協調器"""
        if not self.is_running:
            self.logger.warning("服務未在運行")
            return

        try:
            # 停止核心組件
            await self._stop_core_components()

            # 停止共享服務
            await self._stop_shared_services()

            self.is_running = False

            # 計算運行時間
            if self.startup_time:
                uptime = time.time() - self.startup_time
                self.logger.info(f"AIVA 核心服務協調器停止 (運行時間: {uptime:.1f}秒)")
            else:
                self.logger.info("AIVA 核心服務協調器停止")

            # 記錄停止指標
            self.monitoring_service.record_metric(
                "core_service.shutdown", 1, MetricType.COUNTER
            )

        except Exception as e:
            self.logger.error(f"服務停止失敗: {e}")
            raise

    async def _stop_core_components(self):
        """停止核心組件"""
        # 清理過期的上下文和會話
        await self.context_manager.cleanup_expired_contexts()
        await self.context_manager.cleanup_expired_sessions()

        self.logger.debug("核心組件停止完成")

    async def _stop_shared_services(self):
        """停止共享服務"""
        # 停止安全管理器
        await self.security_manager.stop()

        # 停止跨語言服務
        await self.cross_lang_service.stop_server()

        # 停止監控服務（最後停止）
        await self.monitoring_service.stop()

        self.logger.debug("共享服務停止完成")

    async def _cleanup_on_failure(self):
        """失敗時清理資源"""
        try:
            if hasattr(self, "monitoring_service"):
                await self.monitoring_service.stop()
            if hasattr(self, "security_manager"):
                await self.security_manager.stop()
            if hasattr(self, "cross_lang_service"):
                await self.cross_lang_service.stop_server()
        except Exception:
            pass  # 忽略清理過程中的錯誤

    @error_handler("aiva_core_coordinator")
    async def process_command(
        self,
        command: str,
        args: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """處理命令的主要入口點

        這是核心服務的主要功能，負責：
        1. 創建命令上下文
        2. 智能路由命令
        3. 創建執行計劃
        4. 執行命令
        5. 記錄歷史
        """
        # 創建命令上下文
        context = CommandContext(
            command=command,
            args=args,
            user_id=user_id,
            session_id=session_id,
            request_id=f"req_{int(time.time())}_{id(self)}",
        )

        with trace_operation("process_command", {"command": command}):
            try:
                # 1. 智能路由命令
                route_info = self.command_router.route_command(context)
                self.logger.info(
                    f"Command '{command}' routed to {route_info['type'].value} "
                    f"(AI: {route_info['requires_ai']}, Priority: {route_info['priority']})"
                )

                # 2. 創建執行上下文
                context_id = await self.context_manager.create_context(context)

                # 3. 創建執行計劃
                plan = await self.execution_planner.create_execution_plan(
                    context, route_info
                )
                self.logger.debug(
                    f"Created execution plan {plan['plan_id']} with {len(plan['steps'])} steps"
                )

                # 4. 執行計劃
                result = await self.execution_planner.execute_plan(plan)

                # 5. 記錄執行歷史
                await self.context_manager.add_history(
                    context_id,
                    {
                        "action": "command_processed",
                        "command": command,
                        "route_type": route_info["type"].value,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "plan_id": plan["plan_id"],
                    },
                )

                # 6. 記錄性能指標
                self.monitoring_service.record_metric(
                    f"command.{command}.execution_time",
                    result.execution_time,
                    MetricType.HISTOGRAM,
                )

                self.monitoring_service.record_metric(
                    f"command.{command}.success",
                    1 if result.success else 0,
                    MetricType.COUNTER,
                )

                return result

            except Exception as e:
                self.logger.error(f"Command processing failed for '{command}': {e}")

                # 記錄失敗指標
                self.monitoring_service.record_metric(
                    f"command.{command}.error", 1, MetricType.COUNTER
                )

                # 重新拋出異常，讓 error_handler 裝飾器處理
                raise

    async def get_service_status(self) -> dict[str, Any]:
        """獲取服務狀態"""
        if not self.is_running:
            return {"status": "stopped", "service_id": self.service_id, "uptime": 0}

        uptime = time.time() - self.startup_time if self.startup_time else 0

        # 獲取各組件狀態
        context_stats = await self.context_manager.get_context_stats()
        execution_stats = await self.execution_planner.get_execution_stats()
        command_stats = self.command_router.get_command_stats()

        return {
            "status": "running",
            "service_id": self.service_id,
            "uptime": uptime,
            "startup_time": self.startup_time,
            "components": {
                "command_router": {
                    "available_commands": len(
                        self.command_router.get_available_commands()
                    ),
                    "command_stats": command_stats,
                },
                "context_manager": context_stats,
                "execution_planner": execution_stats,
            },
            "shared_services": {
                "config_manager": "active",
                "monitoring_service": "active",
                "security_manager": "active",
                "cross_lang_service": "active",
            },
        }

    async def health_check(self) -> dict[str, Any]:
        """健康檢查"""
        health_status = {
            "healthy": True,
            "timestamp": time.time(),
            "service_id": self.service_id,
            "checks": {},
        }

        # 檢查基本狀態
        health_status["checks"]["service_running"] = {
            "status": "pass" if self.is_running else "fail",
            "message": (
                "Service is running" if self.is_running else "Service is not running"
            ),
        }

        # 檢查核心組件
        try:
            context_stats = await self.context_manager.get_context_stats()
            health_status["checks"]["context_manager"] = {
                "status": "pass",
                "message": f"Active contexts: {context_stats['active_contexts']}",
            }
        except Exception as e:
            health_status["checks"]["context_manager"] = {
                "status": "fail",
                "message": f"Context manager error: {e}",
            }
            health_status["healthy"] = False

        # 檢查執行計劃器
        try:
            exec_stats = await self.execution_planner.get_execution_stats()
            health_status["checks"]["execution_planner"] = {
                "status": "pass",
                "message": f"Success rate: {exec_stats['success_rate']:.1f}%",
            }
        except Exception as e:
            health_status["checks"]["execution_planner"] = {
                "status": "fail",
                "message": f"Execution planner error: {e}",
            }
            health_status["healthy"] = False

        return health_status


# 全局核心服務協調器實例
_core_service_coordinator_instance = None


def get_core_service_coordinator() -> AIVACoreServiceCoordinator:
    """獲取核心服務協調器實例"""
    global _core_service_coordinator_instance
    if _core_service_coordinator_instance is None:
        _core_service_coordinator_instance = AIVACoreServiceCoordinator()
    return _core_service_coordinator_instance


# 便捷函數
async def process_command(
    command: str,
    args: dict[str, Any] = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> ExecutionResult:
    """便捷的命令處理函數

    Args:
        command: 要執行的命令
        args: 命令參數
        user_id: 用戶ID
        session_id: 會話ID

    Returns:
        ExecutionResult: 執行結果
    """
    coordinator = get_core_service_coordinator()
    return await coordinator.process_command(command, args or {}, user_id, session_id)


async def initialize_core_module() -> dict[str, Any]:
    """初始化核心模組

    這個函數負責啟動整個 AIVA 核心模組，包括：
    1. 初始化核心服務協調器
    2. 啟動所有核心組件
    3. 設置監控和配置
    4. 驗證系統就緒狀態

    Returns:
        Dict[str, Any]: 初始化結果和狀態資訊
    """
    try:
        # 1. 獲取核心服務協調器
        coordinator = get_core_service_coordinator()

        # 2. 啟動服務
        await coordinator.start()

        # 3. 驗證服務狀態
        status = await coordinator.get_service_status()
        health = await coordinator.health_check()

        # 4. 返回初始化結果
        init_result = {
            "success": True,
            "service_id": coordinator.service_id,
            "startup_time": coordinator.startup_time,
            "status": status,
            "health": health,
            "message": "AIVA 核心模組初始化成功",
        }

        logging.getLogger("aiva_core_module").info(
            f"核心模組初始化完成 - 服務ID: {coordinator.service_id}"
        )

        return init_result

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"AIVA 核心模組初始化失敗: {e}",
        }

        logging.getLogger("aiva_core_module").error(f"核心模組初始化失敗: {e}")

        raise RuntimeError(f"核心模組初始化失敗: {e}") from e


async def shutdown_core_module() -> dict[str, Any]:
    """關閉核心模組

    這個函數負責優雅地關閉整個 AIVA 核心模組，包括：
    1. 停止核心服務協調器
    2. 清理所有資源
    3. 記錄關閉統計

    Returns:
        Dict[str, Any]: 關閉結果和統計資訊
    """
    try:
        # 1. 獲取核心服務協調器
        coordinator = get_core_service_coordinator()

        # 2. 記錄關閉前狀態
        pre_shutdown_status = await coordinator.get_service_status()

        # 3. 停止服務
        await coordinator.stop()

        # 4. 返回關閉結果
        shutdown_result = {
            "success": True,
            "service_id": coordinator.service_id,
            "uptime": pre_shutdown_status.get("uptime", 0),
            "message": "AIVA 核心模組關閉成功",
        }

        logging.getLogger("aiva_core_module").info(
            f"核心模組關閉完成 - 服務ID: {coordinator.service_id}"
        )

        return shutdown_result

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"AIVA 核心模組關閉失敗: {e}",
        }

        logging.getLogger("aiva_core_module").error(f"核心模組關閉失敗: {e}")

        return error_result
