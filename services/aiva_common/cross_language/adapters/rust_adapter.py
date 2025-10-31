"""
AIVA Rust Language Adapter
Rust 語言適配器

此模組提供：
1. Rust FFI 接口封裝
2. Rust 錯誤映射
3. 數據類型轉換
4. 並發處理協調
5. 內存管理優化
"""

import asyncio
import ctypes
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
class RustConfig:
    """Rust 適配器配置"""

    rust_executable: str = "cargo"
    workspace_path: str = "../../target/release"
    ffi_library_name: str = "libaiva_rust_ffi"
    build_timeout: int = 300  # 5 minutes
    execution_timeout: int = 60
    max_memory_mb: int = 1024
    enable_optimizations: bool = True
    debug_mode: bool = False


class RustFFIError(Exception):
    """Rust FFI 錯誤"""

    pass


class RustPanicError(Exception):
    """Rust Panic 錯誤"""

    pass


class RustAdapter(LanguageAdapter[dict[str, Any]]):
    """Rust 語言適配器"""

    def __init__(self, config: RustConfig | None = None):
        super().__init__("rust")
        self.config = config or RustConfig()
        self.ffi_lib: ctypes.CDLL | None = None
        self.is_initialized = False
        self._setup_error_mapper()

    def _setup_error_mapper(self):
        """設置 Rust 錯誤映射"""
        self.error_mapper = LanguageErrorMapper("rust")

        # 註冊 Rust 特定錯誤映射
        rust_error_mappings = {
            "panic": AIVAErrorCode.RUST_PANIC,
            "compilation_error": AIVAErrorCode.RUST_COMPILATION_FAILED,
            "ffi_error": AIVAErrorCode.RUST_PANIC,  # FFI 錯誤通常類似 panic
            "memory_error": AIVAErrorCode.RESOURCE_EXHAUSTED,
            "io_error": AIVAErrorCode.INTERNAL_ERROR,
            "parse_error": AIVAErrorCode.INVALID_ARGUMENT,
            "timeout": AIVAErrorCode.TIMEOUT,
        }

        for error_type, error_code in rust_error_mappings.items():
            # 這裡我們使用字符串映射，因為 Rust 錯誤不是 Python 異常類型
            setattr(self.error_mapper, f"_{error_type}_code", error_code)

    async def initialize(self) -> bool:
        """初始化 Rust 適配器"""
        if self.is_initialized:
            return True

        try:
            # 檢查 Rust 環境
            if not await self._check_rust_environment():
                self.logger.error("Rust environment not available")
                return False

            # 構建 Rust 庫
            if not await self._build_rust_library():
                self.logger.error("Failed to build Rust library")
                return False

            # 加載 FFI 庫
            if not await self._load_ffi_library():
                self.logger.error("Failed to load FFI library")
                return False

            # 設置 FFI 函數簽名
            self._setup_ffi_functions()

            self.is_initialized = True
            self.logger.info("Rust adapter initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Rust adapter initialization failed: {e}")
            return False

    async def _check_rust_environment(self) -> bool:
        """檢查 Rust 環境"""
        try:
            result = await asyncio.create_subprocess_exec(
                self.config.rust_executable,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                version = stdout.decode().strip()
                self.logger.info(f"Rust version: {version}")
                return True
            else:
                self.logger.error(f"Rust not available: {stderr.decode()}")
                return False

        except FileNotFoundError:
            self.logger.error("Rust executable not found")
            return False
        except Exception as e:
            self.logger.error(f"Error checking Rust environment: {e}")
            return False

    async def _build_rust_library(self) -> bool:
        """構建 Rust 庫"""
        try:
            workspace_path = Path(self.config.workspace_path).resolve()
            if not workspace_path.exists():
                self.logger.error(f"Rust workspace not found: {workspace_path}")
                return False

            # 構建命令
            build_args = [self.config.rust_executable, "build"]
            if self.config.enable_optimizations:
                build_args.append("--release")

            # 執行構建
            self.logger.info("Building Rust library...")
            process = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.build_timeout
                )

                if process.returncode == 0:
                    self.logger.info("Rust library built successfully")
                    return True
                else:
                    self.logger.error(f"Rust build failed: {stderr.decode()}")
                    return False

            except TimeoutError:
                process.kill()
                self.logger.error("Rust build timed out")
                return False

        except Exception as e:
            self.logger.error(f"Error building Rust library: {e}")
            return False

    async def _load_ffi_library(self) -> bool:
        """加載 FFI 庫"""
        try:
            # 尋找庫文件
            workspace_path = Path(self.config.workspace_path).resolve()

            # 根據平台確定庫文件擴展名
            if os.name == "nt":  # Windows
                lib_ext = ".dll"
            elif os.name == "posix":
                if os.uname().sysname == "Darwin":  # macOS
                    lib_ext = ".dylib"
                else:  # Linux
                    lib_ext = ".so"
            else:
                lib_ext = ".so"

            lib_name = self.config.ffi_library_name + lib_ext
            lib_path = workspace_path / lib_name

            if not lib_path.exists():
                self.logger.error(f"FFI library not found: {lib_path}")
                return False

            # 加載庫
            self.ffi_lib = ctypes.CDLL(str(lib_path))
            self.logger.info(f"Loaded FFI library: {lib_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading FFI library: {e}")
            return False

    def _setup_ffi_functions(self):
        """設置 FFI 函數簽名"""
        if not self.ffi_lib:
            return

        try:
            # 設置常用 FFI 函數的簽名

            # 初始化函數
            self.ffi_lib.rust_init.argtypes = []
            self.ffi_lib.rust_init.restype = ctypes.c_int

            # 清理函數
            self.ffi_lib.rust_cleanup.argtypes = []
            self.ffi_lib.rust_cleanup.restype = None

            # 執行掃描函數
            self.ffi_lib.rust_execute_scan.argtypes = [ctypes.c_char_p]
            self.ffi_lib.rust_execute_scan.restype = ctypes.c_char_p

            # 處理數據函數
            self.ffi_lib.rust_process_data.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
            self.ffi_lib.rust_process_data.restype = ctypes.c_char_p

            # 獲取最後錯誤函數
            self.ffi_lib.rust_last_error.argtypes = []
            self.ffi_lib.rust_last_error.restype = ctypes.c_char_p

            # 釋放字符串函數
            self.ffi_lib.rust_free_string.argtypes = [ctypes.c_char_p]
            self.ffi_lib.rust_free_string.restype = None

            self.logger.debug("FFI function signatures set up")

        except AttributeError as e:
            self.logger.warning(f"Some FFI functions not available: {e}")
        except Exception as e:
            self.logger.error(f"Error setting up FFI functions: {e}")

    async def serialize(self, data: dict[str, Any]) -> bytes:
        """序列化數據"""
        try:
            # 將字典轉換為 JSON 字符串，然後編碼為字節
            json_str = json.dumps(data, ensure_ascii=False)
            return json_str.encode("utf-8")
        except Exception as e:
            self.logger.error(f"Rust serialization failed: {e}")
            raise

    async def deserialize(
        self, data: bytes, message_type: type = dict
    ) -> dict[str, Any]:
        """反序列化數據"""
        try:
            # 將字節解碼為 JSON 字符串，然後解析為字典
            json_str = data.decode("utf-8")
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Rust deserialization failed: {e}")
            raise

    async def convert_error(self, error: Exception) -> dict[str, Any]:
        """轉換 Rust 錯誤"""
        error_info = {
            "language": "rust",
            "type": type(error).__name__,
            "message": str(error),
        }

        # 檢查是否為 Rust 特定錯誤
        if isinstance(error, RustPanicError):
            error_info["aiva_error_code"] = "ERROR_CODE_RUST_PANIC"
            error_info["is_panic"] = "True"
        elif isinstance(error, RustFFIError):
            error_info["aiva_error_code"] = "ERROR_CODE_RUST_PANIC"
            error_info["is_ffi_error"] = "True"
        else:
            # 通用錯誤映射
            error_mapping = {
                "OSError": "ERROR_CODE_INTERNAL_ERROR",
                "MemoryError": "ERROR_CODE_RESOURCE_EXHAUSTED",
                "TimeoutError": "ERROR_CODE_TIMEOUT",
            }
            error_info["aiva_error_code"] = error_mapping.get(
                type(error).__name__, "ERROR_CODE_UNKNOWN"
            )

        return error_info

    async def execute_rust_function(
        self, function_name: str, args: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """執行 Rust 函數"""
        if not self.is_initialized:
            if not await self.initialize():
                raise RustFFIError("Rust adapter not initialized")

        if not self.ffi_lib:
            raise RustFFIError("FFI library not loaded")

        timeout = timeout or self.config.execution_timeout

        try:
            # 序列化參數
            args_json = json.dumps(args).encode("utf-8")

            # 檢查 FFI 函數是否存在並調用
            if function_name == "execute_scan" and hasattr(
                self.ffi_lib, "rust_execute_scan"
            ):
                result_ptr = self.ffi_lib.rust_execute_scan(args_json)
            elif function_name == "process_data" and hasattr(
                self.ffi_lib, "rust_process_data"
            ):
                result_ptr = self.ffi_lib.rust_process_data(args_json, len(args_json))
            else:
                raise RustFFIError(
                    f"Unknown or unavailable Rust function: {function_name}"
                )

            if not result_ptr:
                # 獲取錯誤信息
                if hasattr(self.ffi_lib, "rust_last_error"):
                    error_ptr = self.ffi_lib.rust_last_error()
                    if error_ptr:
                        error_msg = ctypes.string_at(error_ptr).decode("utf-8")
                        if hasattr(self.ffi_lib, "rust_free_string"):
                            self.ffi_lib.rust_free_string(error_ptr)
                        raise RustPanicError(f"Rust function failed: {error_msg}")
                raise RustFFIError("Rust function returned null")

            # 獲取結果
            result_json = ctypes.string_at(result_ptr).decode("utf-8")
            if hasattr(self.ffi_lib, "rust_free_string"):
                self.ffi_lib.rust_free_string(result_ptr)

            # 解析結果
            result = json.loads(result_json)

            self.logger.debug(f"Rust function {function_name} executed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Rust function execution failed: {e}")
            error_context = ErrorContext(
                service_name="rust_adapter",
                function_name=function_name,
                file_name=__file__,
                line_number=0,
                language="rust",
                timestamp=time.time(),
            )

            aiva_error = AIVAError(
                error_code=AIVAErrorCode.RUST_PANIC,
                message=str(e),
                severity=ErrorSeverity.HIGH,
                context=error_context,
                original_error=e,
            )

            raise RustFFIError(str(aiva_error)) from e

    async def execute_security_scan(
        self,
        target: str,
        scan_type: str = "vulnerability",
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """執行安全掃描（Rust 實現）"""
        args = {"target": target, "scan_type": scan_type, "options": options or {}}

        try:
            result = await self.execute_rust_function("execute_scan", args)
            self.logger.info(f"Security scan completed for {target}")
            return result
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            raise

    async def process_scan_results(
        self,
        raw_results: dict[str, Any],
        processing_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """處理掃描結果（Rust 實現）"""
        args = {"raw_results": raw_results, "options": processing_options or {}}

        try:
            result = await self.execute_rust_function("process_data", args)
            self.logger.info("Scan results processed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Scan results processing failed: {e}")
            raise

    async def cleanup(self):
        """清理資源"""
        try:
            if self.ffi_lib and hasattr(self.ffi_lib, "rust_cleanup"):
                self.ffi_lib.rust_cleanup()

            self.ffi_lib = None
            self.is_initialized = False
            self.logger.info("Rust adapter cleaned up")

        except Exception as e:
            self.logger.error(f"Rust adapter cleanup failed: {e}")

    def get_adapter_info(self) -> dict[str, Any]:
        """獲取適配器信息"""
        return {
            "language": self.language,
            "initialized": self.is_initialized,
            "config": {
                "rust_executable": self.config.rust_executable,
                "workspace_path": self.config.workspace_path,
                "ffi_library_name": self.config.ffi_library_name,
                "debug_mode": self.config.debug_mode,
            },
            "ffi_available": self.ffi_lib is not None,
        }

    async def __aenter__(self):
        """異步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        await self.cleanup()


# 便利函數
async def create_rust_adapter(config: RustConfig | None = None) -> RustAdapter:
    """創建並初始化 Rust 適配器"""
    adapter = RustAdapter(config)
    if await adapter.initialize():
        return adapter
    else:
        raise RustFFIError("Failed to initialize Rust adapter")


async def execute_rust_scan(
    target: str, scan_type: str = "vulnerability", config: RustConfig | None = None
) -> dict[str, Any]:
    """執行 Rust 掃描的便利函數"""
    async with RustAdapter(config) as adapter:
        return await adapter.execute_security_scan(target, scan_type)


if __name__ == "__main__":
    # 測試 Rust 適配器
    async def test_rust_adapter():
        try:
            adapter = await create_rust_adapter()

            # 測試掃描功能
            result = await adapter.execute_security_scan("http://example.com")
            print("Scan result:", result)

            await adapter.cleanup()

        except Exception as e:
            print(f"Test failed: {e}")

    asyncio.run(test_rust_adapter())
