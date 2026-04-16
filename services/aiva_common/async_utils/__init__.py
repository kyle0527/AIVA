"""
現代化異步支援模組
基於 asyncio 和 contextvar 的最佳實踐
"""

import asyncio
import functools
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar, copy_context
from typing import (
    Any,
    Dict,
    Optional,
    TypeVar,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel, Field

# 類型變數
T = TypeVar("T")
AsyncCallable = Callable[..., Awaitable[T]]


# 上下文變數
request_id_context: ContextVar[str] = ContextVar("request_id", default="")
trace_id_context: ContextVar[str] = ContextVar("trace_id", default="")
user_id_context: ContextVar[str] = ContextVar("user_id", default="")


class AsyncContext(BaseModel):
    """異步上下文模型"""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    start_time: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AsyncTaskManager:
    """異步任務管理器"""

    def __init__(self):
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._results: dict[str, Any] = {}

    async def submit_task(
        self, task_id: str, coro: Awaitable[T], context: AsyncContext | None = None
    ) -> asyncio.Task[T]:
        """提交異步任務"""
        if context:
            # 在指定上下文中執行
            ctx = copy_context()
            ctx[request_id_context] = context.request_id
            ctx[trace_id_context] = context.trace_id
            if context.user_id:
                ctx[user_id_context] = context.user_id

            task = asyncio.create_task(
                ctx.run(self._run_with_context, coro), name=task_id
            )
        else:
            task = asyncio.create_task(coro, name=task_id)

        self._tasks[task_id] = task
        return task

    async def _run_with_context(self, coro: Awaitable[T]) -> T:
        """在上下文中執行協程"""
        return await coro

    async def wait_for_task(self, task_id: str, timeout: float | None = None) -> Any:
        """等待任務完成"""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self._results[task_id] = result
            return result
        except TimeoutError:
            task.cancel()
            raise
        finally:
            # 清理已完成的任務
            if task.done():
                del self._tasks[task_id]

    async def wait_for_all(self, timeout: float | None = None) -> dict[str, Any]:
        """等待所有任務完成"""
        if not self._tasks:
            return {}

        tasks = list(self._tasks.values())
        task_ids = list(self._tasks.keys())

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )

            result_dict = {}
            for task_id, result in zip(task_ids, results, strict=False):
                if isinstance(result, Exception):
                    result_dict[task_id] = {"error": str(result)}
                else:
                    result_dict[task_id] = result
                    self._results[task_id] = result

            return result_dict

        except TimeoutError:
            # 取消所有未完成的任務
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            # 清理所有任務
            self._tasks.clear()

    def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if not task.done():
                task.cancel()
                return True
            del self._tasks[task_id]
        return False

    def cancel_all(self) -> int:
        """取消所有任務"""
        cancelled_count = 0
        for task_id in list(self._tasks.keys()):
            if self.cancel_task(task_id):
                cancelled_count += 1
        return cancelled_count

    def get_task_status(self, task_id: str) -> str | None:
        """獲取任務狀態"""
        if task_id not in self._tasks:
            if task_id in self._results:
                return "completed"
            return None

        task = self._tasks[task_id]
        if task.done():
            if task.cancelled():
                return "cancelled"
            elif task.exception():
                return "failed"
            else:
                return "completed"
        else:
            return "running"

    def get_active_tasks(self) -> dict[str, str]:
        """獲取活動任務列表"""
        return {
            task_id: self.get_task_status(task_id) or "unknown"
            for task_id in self._tasks.keys()
        }


class AsyncLimitedExecutor:
    """有限制的異步執行器"""

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_tasks = 0

    async def submit(self, coro: Awaitable[T]) -> T:
        """提交協程執行"""
        async with self._semaphore:
            self._active_tasks += 1
            try:
                return await coro
            finally:
                self._active_tasks -= 1

    @property
    def active_count(self) -> int:
        """活動任務數量"""
        return self._active_tasks

    @property
    def available_slots(self) -> int:
        """可用執行槽位"""
        return self.max_concurrency - self._active_tasks


class AsyncRetry:
    """異步重試器"""

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.exceptions = exceptions

    async def execute(self, coro_func: Callable[[], Awaitable[T]]) -> T:
        """執行帶重試的協程"""
        last_exception = None
        current_delay = self.delay

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await coro_func()
            except self.exceptions as e:
                last_exception = e

                if attempt == self.max_attempts:
                    break

                await asyncio.sleep(current_delay)
                current_delay = min(current_delay * self.backoff_factor, self.max_delay)

        # 重試失敗，拋出最後一次異常
        raise last_exception


# 裝飾器支援
def async_timeout(seconds: float):
    """異步超時裝飾器"""

    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """異步重試裝飾器"""

    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry = AsyncRetry(
                max_attempts=max_attempts,
                delay=delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
            )
            return await retry.execute(lambda: func(*args, **kwargs))

        return wrapper

    return decorator


def async_context(request_id: str | None = None, trace_id: str | None = None):
    """異步上下文裝飾器"""

    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ctx = copy_context()

            # 設置上下文變數
            if request_id:
                ctx[request_id_context] = request_id
            if trace_id:
                ctx[trace_id_context] = trace_id

            # 在上下文中執行
            return await ctx.run(func, *args, **kwargs)

        return wrapper

    return decorator


@asynccontextmanager
async def async_context_manager(
    context: AsyncContext,
) -> AsyncGenerator[AsyncContext, None]:
    """異步上下文管理器"""
    # 保存舊的上下文值
    old_request_id = request_id_context.get("")
    old_trace_id = trace_id_context.get("")
    old_user_id = user_id_context.get("")

    try:
        # 設置新的上下文值
        request_id_context.set(context.request_id)
        trace_id_context.set(context.trace_id)
        if context.user_id:
            user_id_context.set(context.user_id)

        yield context

    finally:
        # 恢復舊的上下文值
        request_id_context.set(old_request_id)
        trace_id_context.set(old_trace_id)
        user_id_context.set(old_user_id)


# 輔助函數
def get_current_request_id() -> str:
    """獲取當前請求 ID"""
    return request_id_context.get("")


def get_current_trace_id() -> str:
    """獲取當前追蹤 ID"""
    return trace_id_context.get("")


def get_current_user_id() -> str:
    """獲取當前用戶 ID"""
    return user_id_context.get("")


async def run_in_background(coro: Awaitable[T]) -> asyncio.Task[T]:
    """在背景執行協程"""
    return asyncio.create_task(coro)


async def gather_with_concurrency(
    max_concurrency: int, *coroutines: Awaitable[Any]
) -> list[Any]:
    """限制並發數量的 gather"""
    executor = AsyncLimitedExecutor(max_concurrency)
    tasks = [executor.submit(coro) for coro in coroutines]
    return await asyncio.gather(*tasks)


# 全域實例
default_task_manager = AsyncTaskManager()


# ============================================
# 進程管理 (Process Management)
# ============================================
# 解決殭屍進程和 Event Loop 阻塞問題


class AsyncProcessManager:
    """異步子進程管理器
    
    解決的問題：
    1. 殭屍進程管理 - 主程式崩潰時自動清理子進程
    2. Event Loop 阻塞 - 使用 asyncio.create_subprocess_exec 避免阻塞
    3. 即時 Stderr 串流 - 可選即時讀取進程輸出
    4. 進程組管理 - 支援 Windows/Unix 跨平台
    
    使用範例：
        manager = AsyncProcessManager()
        result = await manager.run_command(
            ["nmap", "-sV", "target.com"],
            timeout=300,
            stream_output=True
        )
    """

    def __init__(self, cleanup_on_exit: bool = True):
        """初始化進程管理器
        
        Args:
            cleanup_on_exit: 是否在程式退出時自動清理子進程
        """
        import atexit
        import signal
        import sys

        self._active_processes: dict[int, asyncio.subprocess.Process] = {}
        self._process_info: dict[int, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_registered = False

        if cleanup_on_exit:
            self._register_cleanup_handlers()

    def _register_cleanup_handlers(self) -> None:
        """註冊清理處理器"""
        import atexit
        import signal
        import sys

        if self._cleanup_registered:
            return

        # atexit 處理器
        atexit.register(self._sync_cleanup_all)

        # Windows 不支援 SIGTERM，需要特殊處理
        if sys.platform != "win32":
            # Unix 信號處理
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(sig, self._signal_handler)
                except (OSError, ValueError):
                    pass  # 某些環境無法設置信號處理器

        self._cleanup_registered = True

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """信號處理器"""
        self._sync_cleanup_all()

    def _sync_cleanup_all(self) -> None:
        """同步清理所有進程（用於 atexit）"""
        import os
        import signal
        import sys

        for pid, proc in list(self._active_processes.items()):
            try:
                if proc.returncode is None:
                    if sys.platform == "win32":
                        proc.terminate()
                    else:
                        # Unix: 發送 SIGTERM 給進程組
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError):
                            proc.terminate()
            except Exception:
                pass

        self._active_processes.clear()
        self._process_info.clear()

    async def run_command(
        self,
        cmd: list[str],
        timeout: float = 60.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stream_output: bool = False,
        output_callback: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        """異步執行命令（不阻塞 Event Loop）
        
        Args:
            cmd: 命令和參數列表（避免 shell=True）
            timeout: 超時秒數
            cwd: 工作目錄
            env: 環境變數
            stream_output: 是否即時串流輸出
            output_callback: 輸出回調函數 (stream_type: "stdout"|"stderr", line: str)
        
        Returns:
            {
                "exit_code": int,
                "stdout": str,
                "stderr": str,
                "timed_out": bool,
                "duration": float,
                "pid": int
            }
        """
        import os
        import sys
        import time

        start_time = time.time()

        # 構建進程參數
        process_kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }

        if cwd:
            process_kwargs["cwd"] = cwd

        if env:
            # 合併環境變數
            merged_env = os.environ.copy()
            merged_env.update(env)
            process_kwargs["env"] = merged_env

        # Unix: 創建新的進程組以便清理
        if sys.platform != "win32":
            process_kwargs["start_new_session"] = True

        try:
            # 創建異步子進程
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                **process_kwargs
            )

            # 追蹤進程
            async with self._lock:
                self._active_processes[proc.pid] = proc
                self._process_info[proc.pid] = {
                    "cmd": cmd,
                    "start_time": start_time,
                    "cwd": cwd,
                }

            try:
                if stream_output and output_callback:
                    # 即時串流輸出
                    stdout_lines: list[str] = []
                    stderr_lines: list[str] = []

                    async def read_stream(
                        stream: asyncio.StreamReader | None,
                        stream_type: str,
                        lines_list: list[str]
                    ) -> None:
                        if stream is None:
                            return
                        while True:
                            line = await stream.readline()
                            if not line:
                                break
                            decoded = line.decode("utf-8", errors="replace").rstrip()
                            lines_list.append(decoded)
                            if output_callback:
                                output_callback(stream_type, decoded)

                    # 並行讀取 stdout 和 stderr
                    await asyncio.wait_for(
                        asyncio.gather(
                            read_stream(proc.stdout, "stdout", stdout_lines),
                            read_stream(proc.stderr, "stderr", stderr_lines),
                        ),
                        timeout=timeout
                    )

                    await proc.wait()
                    stdout = "\n".join(stdout_lines)
                    stderr = "\n".join(stderr_lines)
                    timed_out = False
                else:
                    # 標準等待模式
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout
                    )
                    stdout = stdout_bytes.decode("utf-8", errors="replace")
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
                    timed_out = False

            except TimeoutError:
                # 超時處理
                await self._terminate_process(proc)
                stdout = ""
                stderr = f"Process timed out after {timeout} seconds"
                timed_out = True

            duration = time.time() - start_time

            return {
                "exit_code": proc.returncode or -1,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
                "duration": duration,
                "pid": proc.pid,
            }

        finally:
            # 清理追蹤
            async with self._lock:
                self._active_processes.pop(proc.pid, None)
                self._process_info.pop(proc.pid, None)

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        """終止進程（帶有優雅關閉）"""
        import os
        import signal
        import sys

        if proc.returncode is not None:
            return

        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                # Unix: 先發送 SIGTERM 給進程組
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()

            # 等待進程結束
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                # 強制終止
                if sys.platform == "win32":
                    proc.kill()
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()

        except Exception:
            pass

    async def cleanup_all(self) -> int:
        """清理所有活動進程
        
        Returns:
            清理的進程數量
        """
        cleaned = 0
        async with self._lock:
            for pid, proc in list(self._active_processes.items()):
                await self._terminate_process(proc)
                cleaned += 1
            self._active_processes.clear()
            self._process_info.clear()
        return cleaned

    def get_active_processes(self) -> list[dict[str, Any]]:
        """獲取活動進程列表"""
        import time

        result = []
        for pid, info in self._process_info.items():
            result.append({
                "pid": pid,
                "cmd": info["cmd"],
                "running_time": time.time() - info["start_time"],
                "cwd": info.get("cwd"),
            })
        return result

    # ============================================
    # 遙測數據解析 (Telemetry Parsing)
    # ============================================
    # 解決 AI 學習迴圈「感官缺失」問題

    def _parse_telemetry(
        self,
        stdout: str,
        stderr: str,
        exit_code: int
    ) -> dict[str, Any]:
        """智能解析 CLI 輸出的遙測數據
        
        提取 HTTP 狀態碼、WAF 特徵、響應時間等資訊，
        供 AI 學習系統使用。
        
        Args:
            stdout: 標準輸出
            stderr: 標準錯誤
            exit_code: 退出碼
        
        Returns:
            遙測數據字典
        """
        import re

        telemetry = {
            "http_status_codes": self._extract_http_codes(stdout, stderr),
            "waf_triggered": self._detect_waf_patterns(stderr),
            "bypassed_protection": self._detect_bypass_success(stdout),
            "response_times": self._extract_response_times(stdout),
            "connection_issues": self._detect_connection_issues(stderr),
            "error_patterns": self._extract_error_patterns(stderr),
        }

        return telemetry

    def _extract_http_codes(self, stdout: str, stderr: str) -> list[int]:
        """提取 HTTP 狀態碼"""
        import re

        combined = stdout + stderr
        # 匹配常見的 HTTP 狀態碼格式
        patterns = [
            r'HTTP/\d\.\d\s+(\d{3})',  # HTTP/1.1 200
            r'Status:\s*(\d{3})',       # Status: 403
            r'(\d{3})\s+(?:OK|Forbidden|Not Found|Internal Server Error)',  # 403 Forbidden
        ]

        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            codes.extend(int(m) for m in matches)

        return list(set(codes))  # 去重

    def _detect_waf_patterns(self, stderr: str) -> bool:
        """檢測 WAF 攔截特徵
        
        識別常見的 WAF/防護系統特徵：
        - Cloudflare, Imperva, AWS WAF, mod_security
        - 403 Forbidden, 406 Not Acceptable
        - Connection Reset, SSL handshake failure
        """
        waf_keywords = [
            # HTTP 狀態
            "403 forbidden",
            "406 not acceptable",
            "429 too many requests",
            # WAF 廠商
            "cloudflare",
            "imperva",
            "incapsula",
            "mod_security",
            "modsecurity",
            "aws waf",
            "f5 big-ip",
            "barracuda",
            "fortiweb",
            # 連線異常（WAF 主動斷線）
            "connection reset by peer",
            "ssl handshake failure",
            "ssl: wrong_version_number",
            # WAF 特徵頁面
            "access denied",
            "request blocked",
            "security violation",
            "captcha",
        ]

        stderr_lower = stderr.lower()
        return any(keyword in stderr_lower for keyword in waf_keywords)

    def _detect_bypass_success(self, stdout: str) -> bool:
        """檢測成功繞過防護的特徵"""
        bypass_indicators = [
            "vulnerability found",
            "exploit successful",
            "shell spawned",
            "authentication bypassed",
            "sql injection successful",
            "xss executed",
            "rce achieved",
            "payload executed",
        ]

        stdout_lower = stdout.lower()
        return any(indicator in stdout_lower for indicator in bypass_indicators)

    def _extract_response_times(self, stdout: str) -> list[float]:
        """提取響應時間（毫秒）"""
        import re

        # 匹配常見的時間格式
        patterns = [
            r'(\d+\.?\d*)\s*ms',           # 123.45 ms
            r'time[=:]\s*(\d+\.?\d*)',     # time=123.45
            r'latency[=:]\s*(\d+\.?\d*)',  # latency=123
            r'(\d+\.?\d*)\s*seconds?',     # 1.23 seconds
        ]

        times = []
        for pattern in patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            for match in matches:
                try:
                    time_value = float(match)
                    # 轉換為毫秒
                    if 'second' in pattern:
                        time_value *= 1000
                    times.append(time_value)
                except ValueError:
                    pass

        return times

    def _detect_connection_issues(self, stderr: str) -> bool:
        """檢測連線問題"""
        connection_errors = [
            "connection refused",
            "connection timeout",
            "network unreachable",
            "no route to host",
            "connection aborted",
            "broken pipe",
        ]

        stderr_lower = stderr.lower()
        return any(error in stderr_lower for error in connection_errors)

    def _extract_error_patterns(self, stderr: str) -> list[str]:
        """提取錯誤模式類型"""
        error_types = []

        error_mapping = {
            "timeout": ["timeout", "timed out"],
            "permission_denied": ["permission denied", "access denied", "forbidden"],
            "not_found": ["not found", "404"],
            "connection_error": ["connection refused", "connection reset"],
            "ssl_error": ["ssl error", "certificate verify failed"],
            "authentication_error": ["authentication failed", "unauthorized", "401"],
        }

        stderr_lower = stderr.lower()
        for error_type, keywords in error_mapping.items():
            if any(kw in stderr_lower for kw in keywords):
                error_types.append(error_type)

        return error_types

    async def run_command_with_telemetry(
        self,
        cmd: list[str],
        timeout: float = 60.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stream_output: bool = False,
        output_callback: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        """異步執行命令並附帶遙測數據解析
        
        相比 run_command()，額外提供：
        - HTTP 狀態碼提取
        - WAF 檢測
        - 繞過檢測
        - 響應時間分析
        
        用於 AI 學習系統的獎勵計算。
        
        Args:
            cmd: 命令和參數列表
            timeout: 超時秒數
            cwd: 工作目錄
            env: 環境變數
            stream_output: 是否即時串流輸出
            output_callback: 輸出回調函數
        
        Returns:
            {
                "exit_code": int,
                "stdout": str,
                "stderr": str,
                "timed_out": bool,
                "duration": float,
                "pid": int,
                "telemetry": {  # 額外的遙測數據
                    "http_status_codes": list[int],
                    "waf_triggered": bool,
                    "bypassed_protection": bool,
                    "response_times": list[float],
                    "connection_issues": bool,
                    "error_patterns": list[str]
                }
            }
        """
        # 先執行命令
        result = await self.run_command(
            cmd=cmd,
            timeout=timeout,
            cwd=cwd,
            env=env,
            stream_output=stream_output,
            output_callback=output_callback
        )

        # 解析遙測數據
        telemetry = self._parse_telemetry(
            stdout=result["stdout"],
            stderr=result["stderr"],
            exit_code=result["exit_code"]
        )

        result["telemetry"] = telemetry
        return result


# 全域進程管理器實例
default_process_manager = AsyncProcessManager()


__all__ = [
    "AsyncContext",
    "AsyncTaskManager",
    "AsyncLimitedExecutor",
    "AsyncRetry",
    "AsyncProcessManager",
    "async_timeout",
    "async_retry",
    "async_context",
    "async_context_manager",
    "get_current_request_id",
    "get_current_trace_id",
    "get_current_user_id",
    "run_in_background",
    "gather_with_concurrency",
    "default_task_manager",
    "default_process_manager",
    "request_id_context",
    "trace_id_context",
    "user_id_context",
]
