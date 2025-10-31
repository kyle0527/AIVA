"""記憶體管理模組
拆分自 optimized_core.py 的記憶體管理部分
"""

import asyncio
from contextlib import asynccontextmanager
import gc
import time
from typing import Any
import weakref


class ComponentPool:
    """組件對象池 - 避免頻繁建立/銷毀物件"""

    def __init__(self, component_class: type, pool_size: int = 10):
        self.component_class = component_class
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self.active_components = set()

        # 預先建立池子中的物件
        for _ in range(pool_size):
            component = component_class()
            self.pool.put_nowait(component)

    @asynccontextmanager
    async def get_component(self):
        """取得組件（上下文管理器）"""
        component = await self.pool.get()
        self.active_components.add(id(component))

        try:
            yield component
        finally:
            # 重置組件狀態
            if hasattr(component, "reset"):
                component.reset()

            self.active_components.discard(id(component))
            await self.pool.put(component)

    async def get_component_async(self):
        """異步取得組件"""
        return await self.pool.get()

    def return_component(self, component):
        """歸還組件到池中"""
        if hasattr(component, "reset"):
            component.reset()

        try:
            self.pool.put_nowait(component)
            self.active_components.discard(id(component))
        except asyncio.QueueFull:
            # 如果池子滿了，直接丟棄組件
            pass

    def get_pool_stats(self) -> dict[str, int]:
        """獲取池子統計資訊"""
        return {
            "pool_size": self.pool_size,
            "available": self.pool.qsize(),
            "active": len(self.active_components),
            "utilization": len(self.active_components) / self.pool_size,
        }


class MemoryManager:
    """智慧記憶體管理器"""

    def __init__(self, gc_threshold_mb: int = 512):
        self.gc_threshold_mb = gc_threshold_mb
        self.weak_refs = weakref.WeakSet()
        self.gc_stats = {
            "collections": 0,
            "objects_collected": 0,
            "last_collection": time.time(),
        }

    async def start_monitoring(self):
        """啟動記憶體監控"""
        while True:
            current_memory = self._get_memory_usage_mb()

            if current_memory > self.gc_threshold_mb:
                await self._force_cleanup()

            # 每30秒檢查一次
            await asyncio.sleep(30)

    async def _force_cleanup(self):
        """強制清理記憶體"""
        print("Memory threshold exceeded, forcing cleanup...")

        before_count = len(gc.get_objects())

        # 執行垃圾回收
        collected = gc.collect()

        after_count = len(gc.get_objects())

        # 更新統計
        self.gc_stats["collections"] += 1
        self.gc_stats["objects_collected"] += before_count - after_count
        self.gc_stats["last_collection"] = time.time()

        print(
            f"GC completed: {collected} cycles, {before_count - after_count} objects freed"
        )

    def _get_memory_usage_mb(self) -> float:
        """獲取當前記憶體使用量（MB）"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # 如果沒有 psutil，使用 tracemalloc
            import tracemalloc

            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                return current / 1024 / 1024
        return 0.0

    def register_weak_ref(self, obj):
        """註冊弱引用"""
        self.weak_refs.add(obj)

    def get_memory_stats(self) -> dict[str, Any]:
        """獲取記憶體統計"""
        return {
            "current_memory_mb": self._get_memory_usage_mb(),
            "threshold_mb": self.gc_threshold_mb,
            "weak_refs_count": len(self.weak_refs),
            "gc_stats": self.gc_stats.copy(),
        }
