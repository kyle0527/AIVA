"""
AIVA Stream Processing System
AIVA 流處理系統

專門用於實時數據流處理的組件，支持：
- 實時事件流處理
- 窗口聚合操作
- 流式機器學習
- 複雜事件處理 (CEP)
- 流式 SQL 查詢
"""

import asyncio
import logging
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    TypeVar,
)

from ..config_manager import ConfigManager, get_config_manager
from ..error_handling import AIVAError, ErrorHandler, ErrorSeverity, ErrorType
from .data_pipeline import PipelineMetrics

T = TypeVar("T")


class WindowType(Enum):
    """窗口類型"""

    TUMBLING = "tumbling"  # 翻滾窗口
    SLIDING = "sliding"  # 滑動窗口
    SESSION = "session"  # 會話窗口
    COUNT = "count"  # 計數窗口


class StreamEventType(Enum):
    """流事件類型"""

    DATA = "data"
    WATERMARK = "watermark"
    BARRIER = "barrier"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """流事件"""

    event_id: str
    event_type: StreamEventType
    timestamp: float
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """用於優先隊列排序"""
        return self.timestamp < other.timestamp


@dataclass
class WindowResult:
    """窗口結果"""

    window_id: str
    window_start: float
    window_end: float
    data: Any
    record_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TimeWindow:
    """時間窗口"""

    def __init__(
        self,
        window_type: WindowType,
        size_ms: int,
        slide_ms: int | None = None,
        session_timeout_ms: int | None = None,
    ):
        self.window_type = window_type
        self.size_ms = size_ms
        self.slide_ms = slide_ms or size_ms  # 默認為翻滾窗口
        self.session_timeout_ms = session_timeout_ms or 300000  # 5分鐘

        self.windows: dict[str, list[StreamEvent]] = {}
        self.window_metadata: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}({window_type.value})"
        )

    def add_event(self, event: StreamEvent) -> list[WindowResult]:
        """添加事件到窗口"""
        results = []

        if self.window_type == WindowType.TUMBLING:
            results.extend(self._handle_tumbling_window(event))
        elif self.window_type == WindowType.SLIDING:
            results.extend(self._handle_sliding_window(event))
        elif self.window_type == WindowType.SESSION:
            results.extend(self._handle_session_window(event))
        elif self.window_type == WindowType.COUNT:
            results.extend(self._handle_count_window(event))

        return results

    def _handle_tumbling_window(self, event: StreamEvent) -> list[WindowResult]:
        """處理翻滾窗口"""
        window_start = (int(event.timestamp * 1000) // self.size_ms) * self.size_ms
        window_end = window_start + self.size_ms
        window_id = f"tumbling_{window_start}_{window_end}"

        if window_id not in self.windows:
            self.windows[window_id] = []
            self.window_metadata[window_id] = {
                "start": window_start,
                "end": window_end,
                "type": "tumbling",
            }

        self.windows[window_id].append(event)

        # 檢查窗口是否應該關閉
        current_time_ms = time.time() * 1000
        if current_time_ms >= window_end:
            return self._close_window(window_id)

        return []

    def _handle_sliding_window(self, event: StreamEvent) -> list[WindowResult]:
        """處理滑動窗口"""
        event_time_ms = int(event.timestamp * 1000)
        results = []

        # 計算所有包含此事件的窗口
        for window_start in range(
            event_time_ms - self.size_ms + self.slide_ms,
            event_time_ms + 1,
            self.slide_ms,
        ):
            if window_start < 0:
                continue

            window_end = window_start + self.size_ms
            window_id = f"sliding_{window_start}_{window_end}"

            if window_id not in self.windows:
                self.windows[window_id] = []
                self.window_metadata[window_id] = {
                    "start": window_start,
                    "end": window_end,
                    "type": "sliding",
                }

            # 只有在時間範圍內的事件才添加到窗口
            if window_start <= event_time_ms < window_end:
                self.windows[window_id].append(event)

        # 檢查是否有窗口需要關閉
        current_time_ms = time.time() * 1000
        windows_to_close = []
        for window_id, metadata in self.window_metadata.items():
            if current_time_ms >= metadata["end"]:
                windows_to_close.append(window_id)

        for window_id in windows_to_close:
            results.extend(self._close_window(window_id))

        return results

    def _handle_session_window(self, event: StreamEvent) -> list[WindowResult]:
        """處理會話窗口"""
        event_time_ms = int(event.timestamp * 1000)
        results = []

        # 查找現有的會話窗口
        matching_window = None
        for window_id, metadata in self.window_metadata.items():
            if metadata["type"] != "session":
                continue

            # 檢查事件是否在會話超時時間內
            if event_time_ms - metadata["last_event_time"] <= self.session_timeout_ms:
                matching_window = window_id
                break

        if matching_window:
            # 擴展現有會話
            self.windows[matching_window].append(event)
            self.window_metadata[matching_window]["last_event_time"] = event_time_ms
            self.window_metadata[matching_window]["end"] = event_time_ms
        else:
            # 創建新會話
            window_id = f"session_{event_time_ms}_{uuid.uuid4().hex[:8]}"
            self.windows[window_id] = [event]
            self.window_metadata[window_id] = {
                "start": event_time_ms,
                "end": event_time_ms,
                "last_event_time": event_time_ms,
                "type": "session",
            }

        # 檢查過期的會話
        current_time_ms = time.time() * 1000
        expired_sessions = []
        for window_id, metadata in self.window_metadata.items():
            if metadata["type"] == "session":
                if (
                    current_time_ms - metadata["last_event_time"]
                    > self.session_timeout_ms
                ):
                    expired_sessions.append(window_id)

        for window_id in expired_sessions:
            results.extend(self._close_window(window_id))

        return results

    def _handle_count_window(self, event: StreamEvent) -> list[WindowResult]:
        """處理計數窗口"""
        # 使用單一計數窗口
        window_id = "count_window"

        if window_id not in self.windows:
            self.windows[window_id] = []
            self.window_metadata[window_id] = {
                "type": "count",
                "target_count": self.size_ms,  # 這裡 size_ms 表示計數
            }

        self.windows[window_id].append(event)

        # 當達到目標計數時關閉窗口
        if len(self.windows[window_id]) >= self.size_ms:
            result = self._close_window(window_id)

            # 重新創建計數窗口
            self.windows[window_id] = []

            return result

        return []

    def _close_window(self, window_id: str) -> list[WindowResult]:
        """關閉窗口並生成結果"""
        if window_id not in self.windows:
            return []

        events = self.windows[window_id]
        metadata = self.window_metadata[window_id]

        if not events:
            # 清理空窗口
            del self.windows[window_id]
            del self.window_metadata[window_id]
            return []

        # 創建窗口結果
        result = WindowResult(
            window_id=window_id,
            window_start=(
                metadata["start"] / 1000 if "start" in metadata else events[0].timestamp
            ),
            window_end=(
                metadata["end"] / 1000 if "end" in metadata else events[-1].timestamp
            ),
            data=[event.data for event in events],
            record_count=len(events),
            metadata=metadata.copy(),
        )

        # 清理窗口
        del self.windows[window_id]
        del self.window_metadata[window_id]

        self.logger.debug(f"關閉窗口 {window_id}，包含 {len(events)} 個事件")

        return [result]


class StreamAggregator:
    """流聚合器"""

    def __init__(self, aggregation_functions: dict[str, Callable[[list[Any]], Any]]):
        self.aggregation_functions = aggregation_functions
        self.logger = logging.getLogger(self.__class__.__name__)

    def aggregate_window(self, window_result: WindowResult) -> dict[str, Any]:
        """聚合窗口結果"""
        aggregated = {}

        try:
            for name, func in self.aggregation_functions.items():
                aggregated[name] = func(window_result.data)
        except Exception as e:
            self.logger.error(f"聚合計算失敗: {e}")
            aggregated["error"] = str(e)

        return aggregated


class StreamProcessor:
    """
    流處理器

    提供實時流數據處理能力
    """

    def __init__(self, name: str, config_manager: ConfigManager | None = None):
        self.name = name
        self.config_manager = config_manager or get_config_manager()
        self.logger = logging.getLogger(f"{self.__class__.__name__}({name})")

        # 事件隊列和處理
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.windows: dict[str, TimeWindow] = {}
        self.aggregators: dict[str, StreamAggregator] = {}

        # 處理狀態
        self.is_running = False
        self.processor_task: asyncio.Task | None = None

        # 指標
        self.metrics = PipelineMetrics()
        self.events_processed = 0
        self.windows_closed = 0

        # 錯誤處理
        self.error_handler = ErrorHandler()

    def add_window(
        self,
        window_name: str,
        window_type: WindowType,
        size_ms: int,
        slide_ms: int | None = None,
        session_timeout_ms: int | None = None,
    ):
        """添加時間窗口"""
        window = TimeWindow(
            window_type=window_type,
            size_ms=size_ms,
            slide_ms=slide_ms,
            session_timeout_ms=session_timeout_ms,
        )
        self.windows[window_name] = window
        self.logger.info(f"添加窗口: {window_name} ({window_type.value})")

    def add_aggregator(
        self,
        aggregator_name: str,
        aggregation_functions: dict[str, Callable[[list[Any]], Any]],
    ):
        """添加聚合器"""
        aggregator = StreamAggregator(aggregation_functions)
        self.aggregators[aggregator_name] = aggregator
        self.logger.info(f"添加聚合器: {aggregator_name}")

    async def start(self):
        """啟動流處理器"""
        if self.is_running:
            return

        self.is_running = True
        self.metrics.start_time = time.time()
        self.processor_task = asyncio.create_task(self._process_events())

        self.logger.info(f"流處理器已啟動: {self.name}")

    async def stop(self):
        """停止流處理器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"流處理器已停止: {self.name}")

    async def emit_event(self, data: Any, timestamp: float | None = None) -> str:
        """發送事件到流處理器"""
        event_id = f"event_{uuid.uuid4().hex[:8]}"
        event = StreamEvent(
            event_id=event_id,
            event_type=StreamEventType.DATA,
            timestamp=timestamp or time.time(),
            data=data,
        )

        await self.event_queue.put(event)
        return event_id

    async def emit_watermark(self, timestamp: float):
        """發送水位線標記"""
        watermark = StreamEvent(
            event_id=f"watermark_{uuid.uuid4().hex[:8]}",
            event_type=StreamEventType.WATERMARK,
            timestamp=timestamp,
        )

        await self.event_queue.put(watermark)

    async def _process_events(self):
        """處理事件循環"""
        while self.is_running:
            try:
                # 等待事件，帶超時以避免無限等待
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                await self._handle_event(event)
                self.events_processed += 1

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"處理事件失敗: {e}")
                await self.error_handler.handle_error(
                    AIVAError(
                        error_type=ErrorType.PROCESSING_ERROR,
                        message=f"流處理事件失敗: {self.name}",
                        details={"error": str(e)},
                        severity=ErrorSeverity.MEDIUM,
                    )
                )

    async def _handle_event(self, event: StreamEvent):
        """處理單個事件"""
        if event.event_type == StreamEventType.DATA:
            await self._process_data_event(event)
        elif event.event_type == StreamEventType.WATERMARK:
            await self._process_watermark(event)
        elif event.event_type == StreamEventType.HEARTBEAT:
            await self._process_heartbeat(event)

    async def _process_data_event(self, event: StreamEvent):
        """處理數據事件"""
        # 將事件發送到所有窗口
        for window_name, window in self.windows.items():
            try:
                window_results = window.add_event(event)

                # 處理窗口結果
                for result in window_results:
                    await self._handle_window_result(window_name, result)
                    self.windows_closed += 1

            except Exception as e:
                self.logger.error(f"窗口 {window_name} 處理事件失敗: {e}")

    async def _process_watermark(self, event: StreamEvent):
        """處理水位線"""
        self.logger.debug(f"處理水位線: {event.timestamp}")
        # 可以用於觸發延遲的窗口關閉

    async def _process_heartbeat(self, event: StreamEvent):
        """處理心跳"""
        self.logger.debug(f"處理心跳: {event.timestamp}")

    async def _handle_window_result(self, window_name: str, result: WindowResult):
        """處理窗口結果"""
        # 應用聚合器
        for aggregator_name, aggregator in self.aggregators.items():
            try:
                aggregated_data = aggregator.aggregate_window(result)

                self.logger.info(
                    f"窗口聚合完成: {window_name}/{aggregator_name}, "
                    f"記錄數: {result.record_count}, "
                    f"時間範圍: {result.window_start:.2f}-{result.window_end:.2f}"
                )

                # 這裡可以添加回調或事件發送邏輯
                await self._emit_aggregated_result(
                    window_name, aggregator_name, aggregated_data, result
                )

            except Exception as e:
                self.logger.error(f"聚合器 {aggregator_name} 處理失敗: {e}")

    async def _emit_aggregated_result(
        self,
        window_name: str,
        aggregator_name: str,
        aggregated_data: dict[str, Any],
        window_result: WindowResult,
    ):
        """發送聚合結果"""
        # 子類可以重寫此方法來處理聚合結果
        pass

    def get_metrics(self) -> dict[str, Any]:
        """獲取流處理器指標"""
        uptime = time.time() - self.metrics.start_time if self.metrics.start_time else 0

        return {
            "name": self.name,
            "is_running": self.is_running,
            "events_processed": self.events_processed,
            "windows_closed": self.windows_closed,
            "uptime": uptime,
            "event_rate": self.events_processed / uptime if uptime > 0 else 0,
            "windows_count": len(self.windows),
            "aggregators_count": len(self.aggregators),
            "queue_size": self.event_queue.qsize(),
        }


# 預定義聚合函數
class StreamAggregations:
    """流聚合函數集合"""

    @staticmethod
    def count(data: list[Any]) -> int:
        """計數"""
        return len(data)

    @staticmethod
    def sum_numeric(data: list[Any]) -> int | float:
        """數值求和"""
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        return sum(numeric_data)

    @staticmethod
    def average_numeric(data: list[Any]) -> float:
        """數值平均"""
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        return statistics.mean(numeric_data) if numeric_data else 0

    @staticmethod
    def min_numeric(data: list[Any]) -> int | float | None:
        """數值最小值"""
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        return min(numeric_data) if numeric_data else None

    @staticmethod
    def max_numeric(data: list[Any]) -> int | float | None:
        """數值最大值"""
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        return max(numeric_data) if numeric_data else None

    @staticmethod
    def distinct_count(data: list[Any]) -> int:
        """去重計數"""
        return len(set(str(x) for x in data))

    @staticmethod
    def first(data: list[Any]) -> Any:
        """第一個值"""
        return data[0] if data else None

    @staticmethod
    def last(data: list[Any]) -> Any:
        """最後一個值"""
        return data[-1] if data else None

    @staticmethod
    def collect_list(data: list[Any]) -> list[Any]:
        """收集為列表"""
        return data.copy()

    @staticmethod
    def collect_set(data: list[Any]) -> list[Any]:
        """收集為去重集合"""
        return list(set(str(x) for x in data))


# 便利函數
def create_stream_processor(
    name: str, window_size_seconds: int = 60, slide_seconds: int | None = None
) -> StreamProcessor:
    """創建簡單的流處理器"""
    processor = StreamProcessor(name)

    # 添加翻滾窗口
    processor.add_window(
        "default_window",
        WindowType.TUMBLING,
        window_size_seconds * 1000,  # 轉換為毫秒
        (slide_seconds * 1000) if slide_seconds else None,
    )

    # 添加基本聚合器
    processor.add_aggregator(
        "basic_stats",
        {
            "count": StreamAggregations.count,
            "sum": StreamAggregations.sum_numeric,
            "average": StreamAggregations.average_numeric,
            "min": StreamAggregations.min_numeric,
            "max": StreamAggregations.max_numeric,
            "first": StreamAggregations.first,
            "last": StreamAggregations.last,
        },
    )

    return processor
