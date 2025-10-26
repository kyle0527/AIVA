"""
現代化異步支援模組
基於 asyncio 和 contextvar 的最佳實踐
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar, copy_context
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# 類型變數
T = TypeVar('T')
AsyncCallable = Callable[..., Awaitable[T]]


# 上下文變數
request_id_context: ContextVar[str] = ContextVar('request_id', default='')
trace_id_context: ContextVar[str] = ContextVar('trace_id', default='')
user_id_context: ContextVar[str] = ContextVar('user_id', default='')


class AsyncContext(BaseModel):
    """異步上下文模型"""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    start_time: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AsyncTaskManager:
    """異步任務管理器"""
    
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, Any] = {}
    
    async def submit_task(
        self, 
        task_id: str, 
        coro: Awaitable[T],
        context: Optional[AsyncContext] = None
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
                ctx.run(self._run_with_context, coro),
                name=task_id
            )
        else:
            task = asyncio.create_task(coro, name=task_id)
        
        self._tasks[task_id] = task
        return task
    
    async def _run_with_context(self, coro: Awaitable[T]) -> T:
        """在上下文中執行協程"""
        return await coro
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """等待任務完成"""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self._results[task_id] = result
            return result
        except asyncio.TimeoutError:
            task.cancel()
            raise
        finally:
            # 清理已完成的任務
            if task.done():
                del self._tasks[task_id]
    
    async def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """等待所有任務完成"""
        if not self._tasks:
            return {}
        
        tasks = list(self._tasks.values())
        task_ids = list(self._tasks.keys())
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            result_dict = {}
            for task_id, result in zip(task_ids, results):
                if isinstance(result, Exception):
                    result_dict[task_id] = {"error": str(result)}
                else:
                    result_dict[task_id] = result
                    self._results[task_id] = result
            
            return result_dict
            
        except asyncio.TimeoutError:
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
    
    def get_task_status(self, task_id: str) -> Optional[str]:
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
    
    def get_active_tasks(self) -> Dict[str, str]:
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
        exceptions: tuple = (Exception,)
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
    exceptions: tuple = (Exception,)
):
    """異步重試裝飾器"""
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry = AsyncRetry(
                max_attempts=max_attempts,
                delay=delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions
            )
            return await retry.execute(lambda: func(*args, **kwargs))
        return wrapper
    return decorator


def async_context(request_id: Optional[str] = None, trace_id: Optional[str] = None):
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
async def async_context_manager(context: AsyncContext) -> AsyncGenerator[AsyncContext, None]:
    """異步上下文管理器"""
    # 保存舊的上下文值
    old_request_id = request_id_context.get('')
    old_trace_id = trace_id_context.get('')
    old_user_id = user_id_context.get('')
    
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
    return request_id_context.get('')


def get_current_trace_id() -> str:
    """獲取當前追蹤 ID"""
    return trace_id_context.get('')


def get_current_user_id() -> str:
    """獲取當前用戶 ID"""
    return user_id_context.get('')


async def run_in_background(coro: Awaitable[T]) -> asyncio.Task[T]:
    """在背景執行協程"""
    return asyncio.create_task(coro)


async def gather_with_concurrency(
    max_concurrency: int,
    *coroutines: Awaitable[Any]
) -> list[Any]:
    """限制並發數量的 gather"""
    executor = AsyncLimitedExecutor(max_concurrency)
    tasks = [executor.submit(coro) for coro in coroutines]
    return await asyncio.gather(*tasks)


# 全域實例
default_task_manager = AsyncTaskManager()


__all__ = [
    "AsyncContext",
    "AsyncTaskManager", 
    "AsyncLimitedExecutor",
    "AsyncRetry",
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
    "request_id_context",
    "trace_id_context",
    "user_id_context",
]