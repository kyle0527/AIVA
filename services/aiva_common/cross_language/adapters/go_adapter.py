"""
AIVA Go Language Adapter
Go 語言適配器

此模組提供：
1. Go 程序執行接口
2. Go 錯誤映射
3. 數據類型轉換
4. 並發處理協調
5. 微服務通信支持
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import LanguageAdapter
from ..errors import (
    AIVAError,
    AIVAErrorCode,
    ErrorContext,
    ErrorSeverity,
    LanguageErrorMapper,
)

logger = logging.getLogger(__name__)


@dataclass
class GoConfig:
    """Go 適配器配置"""

    go_executable: str = "go"
    workspace_path: str = "../../go_services"
    build_timeout: int = 120  # 2 minutes
    execution_timeout: int = 60
    max_memory_mb: int = 512
    enable_race_detection: bool = True
    build_mode: str = "release"  # release or debug


class GoPanicError(Exception):
    """Go Panic 錯誤"""

    pass


class GoBuildError(Exception):
    """Go 構建錯誤"""

    pass


class GoRuntimeError(Exception):
    """Go 運行時錯誤"""

    pass


class GoAdapter(LanguageAdapter[dict[str, Any]]):
    """Go 語言適配器"""

    def __init__(self, config: GoConfig | None = None):
        super().__init__("go")
        self.config = config or GoConfig()
        self.is_initialized = False
        self._executables: dict[str, Path] = {}
        self._setup_error_mapper()

    def _setup_error_mapper(self):
        """設置 Go 錯誤映射"""
        self.error_mapper = LanguageErrorMapper("go")

        # 註冊 Go 特定錯誤映射
        go_error_mappings = {
            "panic": AIVAErrorCode.GO_PANIC,
            "build_error": AIVAErrorCode.INTERNAL_ERROR,  # 使用通用錯誤碼
            "runtime_error": AIVAErrorCode.JAVASCRIPT_RUNTIME_ERROR,  # 使用類似的錯誤碼
            "network_error": AIVAErrorCode.NETWORK_UNREACHABLE,
            "timeout": AIVAErrorCode.TIMEOUT,
            "permission_denied": AIVAErrorCode.PERMISSION_DENIED,
        }

        for error_type, error_code in go_error_mappings.items():
            setattr(self.error_mapper, f"_{error_type}_code", error_code)

    async def initialize(self) -> bool:
        """初始化 Go 適配器"""
        if self.is_initialized:
            return True

        try:
            # 檢查 Go 環境
            if not await self._check_go_environment():
                self.logger.error("Go environment not available")
                return False

            # 構建 Go 程序
            if not await self._build_go_programs():
                self.logger.error("Failed to build Go programs")
                return False

            self.is_initialized = True
            self.logger.info("Go adapter initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Go adapter initialization failed: {e}")
            return False

    async def _check_go_environment(self) -> bool:
        """檢查 Go 環境"""
        try:
            result = await asyncio.create_subprocess_exec(
                self.config.go_executable,
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                version = stdout.decode().strip()
                self.logger.info(f"Go version: {version}")
                return True
            else:
                self.logger.error(f"Go not available: {stderr.decode()}")
                return False

        except FileNotFoundError:
            self.logger.error("Go executable not found")
            return False
        except Exception as e:
            self.logger.error(f"Error checking Go environment: {e}")
            return False

    async def _build_go_programs(self) -> bool:
        """構建 Go 程序"""
        try:
            workspace_path = Path(self.config.workspace_path).resolve()
            if not workspace_path.exists():
                self.logger.warning(f"Go workspace not found: {workspace_path}")
                return True  # 不是必須的，可以動態構建

            # 尋找並構建 Go 程序
            go_modules = list(workspace_path.glob("*/go.mod"))

            for module_file in go_modules:
                module_dir = module_file.parent
                program_name = module_dir.name

                if await self._build_single_program(module_dir, program_name):
                    self.logger.info(f"Built Go program: {program_name}")
                else:
                    self.logger.warning(f"Failed to build Go program: {program_name}")

            return True

        except Exception as e:
            self.logger.error(f"Error building Go programs: {e}")
            return False

    async def _build_single_program(self, module_dir: Path, program_name: str) -> bool:
        """構建單個 Go 程序"""
        try:
            # 構建命令
            build_args = [self.config.go_executable, "build"]

            if self.config.enable_race_detection:
                build_args.append("-race")

            # 輸出文件
            output_file = module_dir / f"{program_name}"
            if os.name == "nt":  # Windows
                output_file = output_file.with_suffix(".exe")

            build_args.extend(["-o", str(output_file)])

            # 執行構建
            process = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=module_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.build_timeout
                )

                if process.returncode == 0:
                    self._executables[program_name] = output_file
                    return True
                else:
                    self.logger.error(
                        f"Go build failed for {program_name}: {stderr.decode()}"
                    )
                    return False

            except TimeoutError:
                process.kill()
                self.logger.error(f"Go build timed out for {program_name}")
                return False

        except Exception as e:
            self.logger.error(f"Error building Go program {program_name}: {e}")
            return False

    async def serialize(self, data: dict[str, Any]) -> bytes:
        """序列化數據"""
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            return json_str.encode("utf-8")
        except Exception as e:
            self.logger.error(f"Go serialization failed: {e}")
            raise

    async def deserialize(
        self, data: bytes, message_type: type = dict
    ) -> dict[str, Any]:
        """反序列化數據"""
        try:
            json_str = data.decode("utf-8")
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Go deserialization failed: {e}")
            raise

    async def convert_error(self, error: Exception) -> dict[str, Any]:
        """轉換 Go 錯誤"""
        error_info = {
            "language": "go",
            "type": type(error).__name__,
            "message": str(error),
        }

        # 檢查是否為 Go 特定錯誤
        if isinstance(error, GoPanicError):
            error_info["aiva_error_code"] = "ERROR_CODE_GO_PANIC"
            error_info["is_panic"] = "True"
        elif isinstance(error, GoBuildError):
            error_info["aiva_error_code"] = "ERROR_CODE_INTERNAL_ERROR"
            error_info["is_build_error"] = "True"
        elif isinstance(error, GoRuntimeError):
            error_info["aiva_error_code"] = "ERROR_CODE_JAVASCRIPT_RUNTIME_ERROR"
            error_info["is_runtime_error"] = "True"
        else:
            # 通用錯誤映射
            error_mapping = {
                "TimeoutError": "ERROR_CODE_TIMEOUT",
                "ConnectionError": "ERROR_CODE_CONNECTION_REFUSED",
                "PermissionError": "ERROR_CODE_PERMISSION_DENIED",
                "FileNotFoundError": "ERROR_CODE_FILE_NOT_FOUND",
            }
            error_info["aiva_error_code"] = error_mapping.get(
                type(error).__name__, "ERROR_CODE_UNKNOWN"
            )

        return error_info

    async def execute_go_program(
        self, program_name: str, args: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """執行 Go 程序"""
        if not self.is_initialized:
            if not await self.initialize():
                raise GoRuntimeError("Go adapter not initialized")

        timeout = timeout or self.config.execution_timeout

        try:
            # 檢查程序是否存在
            executable = self._executables.get(program_name)
            if not executable or not executable.exists():
                # 嘗試動態構建
                if not await self._build_dynamic_program(program_name):
                    raise GoRuntimeError(f"Go program not found: {program_name}")
                executable = self._executables.get(program_name)
                if not executable:
                    raise GoRuntimeError(f"Failed to build Go program: {program_name}")

            # 準備輸入數據
            input_data = json.dumps(args).encode("utf-8")

            # 執行程序
            process = await asyncio.create_subprocess_exec(
                str(executable),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data), timeout=timeout
                )

                if process.returncode == 0:
                    # 解析輸出
                    result = json.loads(stdout.decode("utf-8"))
                    self.logger.debug(
                        f"Go program {program_name} executed successfully"
                    )
                    return result
                else:
                    error_msg = stderr.decode("utf-8")
                    if "panic:" in error_msg:
                        raise GoPanicError(f"Go panic in {program_name}: {error_msg}")
                    else:
                        raise GoRuntimeError(f"Go program failed: {error_msg}")

            except TimeoutError:
                process.kill()
                raise GoRuntimeError(f"Go program {program_name} timed out")

        except Exception as e:
            self.logger.error(f"Go program execution failed: {e}")
            error_context = ErrorContext(
                service_name="go_adapter",
                function_name=program_name,
                file_name=__file__,
                line_number=0,
                language="go",
                timestamp=time.time(),
            )

            aiva_error = AIVAError(
                error_code=AIVAErrorCode.JAVASCRIPT_RUNTIME_ERROR,
                message=str(e),
                severity=ErrorSeverity.HIGH,
                context=error_context,
                original_error=e,
            )

            raise GoRuntimeError(str(aiva_error)) from e

    async def _build_dynamic_program(self, program_name: str) -> bool:
        """動態構建 Go 程序"""
        try:
            workspace_path = Path(self.config.workspace_path).resolve()
            program_dir = workspace_path / program_name

            if not program_dir.exists():
                self.logger.error(f"Program directory not found: {program_dir}")
                return False

            return await self._build_single_program(program_dir, program_name)

        except Exception as e:
            self.logger.error(f"Dynamic build failed for {program_name}: {e}")
            return False

    async def execute_microservice(
        self,
        service_name: str,
        endpoint: str,
        data: dict[str, Any],
        method: str = "POST",
    ) -> dict[str, Any]:
        """執行微服務調用"""
        args = {
            "service_name": service_name,
            "endpoint": endpoint,
            "method": method,
            "data": data,
        }

        try:
            result = await self.execute_go_program("microservice_client", args)
            self.logger.info(f"Microservice {service_name} called successfully")
            return result
        except Exception as e:
            self.logger.error(f"Microservice call failed: {e}")
            raise

    async def start_gateway_service(
        self, port: int = 8080, services: list[str] | None = None
    ) -> dict[str, Any]:
        """啟動 API 網關服務"""
        args = {"port": port, "services": services or []}

        try:
            result = await self.execute_go_program("api_gateway", args)
            self.logger.info(f"API Gateway started on port {port}")
            return result
        except Exception as e:
            self.logger.error(f"API Gateway startup failed: {e}")
            raise

    async def execute_data_processing(
        self, data: dict[str, Any], processing_type: str = "standard"
    ) -> dict[str, Any]:
        """執行數據處理（Go 實現）"""
        args = {"data": data, "processing_type": processing_type}

        try:
            result = await self.execute_go_program("data_processor", args)
            self.logger.info("Data processing completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise

    async def cleanup(self):
        """清理資源"""
        try:
            # 停止正在運行的 Go 程序
            # 這裡可以添加進程管理邏輯

            self._executables.clear()
            self.is_initialized = False
            self.logger.info("Go adapter cleaned up")

        except Exception as e:
            self.logger.error(f"Go adapter cleanup failed: {e}")

    def get_adapter_info(self) -> dict[str, Any]:
        """獲取適配器信息"""
        return {
            "language": self.language,
            "initialized": self.is_initialized,
            "config": {
                "go_executable": self.config.go_executable,
                "workspace_path": self.config.workspace_path,
                "build_mode": self.config.build_mode,
                "enable_race_detection": self.config.enable_race_detection,
            },
            "available_programs": list(self._executables.keys()),
            "built_programs": len(self._executables),
        }

    def list_available_programs(self) -> list[str]:
        """列出可用的 Go 程序"""
        return list(self._executables.keys())

    async def health_check(self) -> dict[str, Any]:
        """健康檢查"""
        try:
            # 檢查基本的 Go 程序執行
            test_args = {"test": True}
            result = await self.execute_go_program("health_checker", test_args)

            return {
                "status": "healthy",
                "go_version": await self._get_go_version(),
                "available_programs": len(self._executables),
                "test_result": result,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "available_programs": len(self._executables),
            }

    async def _get_go_version(self) -> str:
        """獲取 Go 版本"""
        try:
            result = await asyncio.create_subprocess_exec(
                self.config.go_executable,
                "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return stdout.decode().strip()
            else:
                return "unknown"

        except Exception:
            return "unknown"

    async def __aenter__(self):
        """異步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        await self.cleanup()


# 便利函數
async def create_go_adapter(config: GoConfig | None = None) -> GoAdapter:
    """創建並初始化 Go 適配器"""
    adapter = GoAdapter(config)
    if await adapter.initialize():
        return adapter
    else:
        raise GoRuntimeError("Failed to initialize Go adapter")


async def execute_go_microservice(
    service_name: str,
    endpoint: str,
    data: dict[str, Any],
    config: GoConfig | None = None,
) -> dict[str, Any]:
    """執行 Go 微服務的便利函數"""
    async with GoAdapter(config) as adapter:
        return await adapter.execute_microservice(service_name, endpoint, data)


async def start_go_gateway(
    port: int = 8080,
    services: list[str] | None = None,
    config: GoConfig | None = None,
) -> dict[str, Any]:
    """啟動 Go API 網關的便利函數"""
    async with GoAdapter(config) as adapter:
        return await adapter.start_gateway_service(port, services)


if __name__ == "__main__":
    # 測試 Go 適配器
    async def test_go_adapter():
        try:
            adapter = await create_go_adapter()

            # 測試健康檢查
            health = await adapter.health_check()
            print("Health check:", health)

            # 測試程序執行
            if adapter.list_available_programs():
                program_name = adapter.list_available_programs()[0]
                result = await adapter.execute_go_program(program_name, {"test": True})
                print(f"Program {program_name} result:", result)

            await adapter.cleanup()

        except Exception as e:
            print(f"Test failed: {e}")

    asyncio.run(test_go_adapter())
