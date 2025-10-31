"""
AIVA Data Pipeline System
AIVA 數據流管道系統

實施 TODO 項目 11: 實現數據流管道
- 流式數據處理
- 批量數據處理
- 實時數據分析
- 跨語言數據交換
- 數據轉換和路由

特性：
1. 多種處理模式：流式、批量、實時
2. 數據源適配器：文件、數據庫、消息隊列、API
3. 處理器鏈：支持管道式數據處理
4. 錯誤處理和重試機制
5. 性能監控和指標收集
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

try:
    from ..config_manager import ConfigManager, ConfigScope, get_config_manager
    from ..error_handling import AIVAError, ErrorHandler, ErrorSeverity, ErrorType
except ImportError:
    # 如果無法導入，創建模擬類
    class ErrorHandler:
        @staticmethod
        async def handle_error(error, error_type="GENERAL", severity="ERROR"):
            # 簡單記錄錯誤
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error: {error}, Type: {error_type}, Severity: {severity}")
            pass

    class ErrorType:
        DATA_PROCESSING_ERROR = "DATA_PROCESSING_ERROR"
        PIPELINE_ERROR = "PIPELINE_ERROR"
        PROCESSING_ERROR = "PROCESSING_ERROR"

    class ErrorSeverity:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    class AIVAError(Exception):
        pass

    class ConfigManager:
        def __init__(self):
            self.settings = {}

        def get_setting(self, key, default=None, scope=None):
            return self.settings.get(key, default)

        def update_setting(self, key, value, scope=None):
            self.settings[key] = value

    def get_config_manager():
        return ConfigManager()

    class ConfigScope:
        PIPELINE = "pipeline"


T = TypeVar("T", contravariant=True)
U = TypeVar("U", covariant=True)


class ProcessingMode(Enum):
    """處理模式"""

    STREAMING = "streaming"  # 流式處理
    BATCH = "batch"  # 批量處理
    REALTIME = "realtime"  # 實時處理
    HYBRID = "hybrid"  # 混合模式


class DataSourceType(Enum):
    """數據源類型"""

    FILE = "file"
    DATABASE = "database"
    API = "api"
    QUEUE = "queue"
    STREAM = "stream"
    MEMORY = "memory"
    CUSTOM = "custom"


class PipelineStatus(Enum):
    """管道狀態"""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class DataRecord:
    """數據記錄"""

    id: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str | None = None
    processing_history: list[str] = field(default_factory=list)

    def add_processing_step(self, step_name: str):
        """添加處理步驟記錄"""
        self.processing_history.append(f"{step_name}@{time.time()}")

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "id": self.id,
            "data": (
                self.data
                if isinstance(self.data, (dict, list, str, int, float, bool))
                else str(self.data)
            ),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "source": self.source,
            "processing_history": self.processing_history,
        }


@dataclass
class PipelineMetrics:
    """管道指標"""

    records_processed: int = 0
    records_failed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    start_time: float | None = None
    last_update_time: float = field(default_factory=time.time)

    def update_metrics(self, processing_time: float, success: bool = True):
        """更新指標"""
        if success:
            self.records_processed += 1
        else:
            self.records_failed += 1

        self.total_processing_time += processing_time
        total_records = self.records_processed + self.records_failed

        if total_records > 0:
            self.average_processing_time = self.total_processing_time / total_records
            self.error_rate = self.records_failed / total_records

        current_time = time.time()
        if self.start_time:
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.throughput_per_second = self.records_processed / elapsed

        self.last_update_time = current_time


class DataSource(Protocol):
    """數據源接口"""

    def read_data(self) -> AsyncIterator[DataRecord]:
        """讀取數據 - 異步生成器方法"""
        ...

    async def close(self):
        """關閉數據源"""
        ...


@runtime_checkable
class DataSink(Protocol):
    """數據輸出接口"""

    async def write_data(self, record: DataRecord) -> bool:
        """寫入數據"""
        ...

    async def close(self):
        """關閉數據輸出"""
        ...


@runtime_checkable
class DataProcessor(Protocol[T, U]):
    """數據處理器接口"""

    async def process(self, record: T) -> U:
        """處理數據"""
        ...


class FileDataSource:
    """文件數據源"""

    def __init__(self, file_path: Path, format_type: str = "json"):
        self.file_path = Path(file_path)
        self.format_type = format_type.lower()
        self.logger = logging.getLogger(f"{self.__class__.__name__}({file_path})")

    async def read_data(self) -> AsyncIterator[DataRecord]:
        """讀取文件數據"""
        try:
            if not self.file_path.exists():
                self.logger.error(f"文件不存在: {self.file_path}")
                return

            with open(self.file_path, encoding="utf-8") as f:
                if self.format_type == "json":
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                yield DataRecord(
                                    id=f"file_{self.file_path.stem}_{i}",
                                    data=item,
                                    source=str(self.file_path),
                                    metadata={"line_number": i, "format": "json"},
                                )
                                await asyncio.sleep(0)  # 讓出控制權
                        else:
                            yield DataRecord(
                                id=f"file_{self.file_path.stem}_0",
                                data=data,
                                source=str(self.file_path),
                                metadata={"format": "json"},
                            )
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON 解析錯誤: {e}")

                elif self.format_type == "text":
                    line_number = 0
                    for line in f:
                        line = line.strip()
                        if line:
                            yield DataRecord(
                                id=f"file_{self.file_path.stem}_{line_number}",
                                data=line,
                                source=str(self.file_path),
                                metadata={"line_number": line_number, "format": "text"},
                            )
                            line_number += 1
                            await asyncio.sleep(0)  # 讓出控制權

        except Exception as e:
            self.logger.error(f"讀取文件失敗: {e}")

    async def close(self):
        """關閉文件數據源"""
        pass


class MemoryDataSource:
    """內存數據源"""

    def __init__(self, data: list[Any]):
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    async def read_data(self) -> AsyncIterator[DataRecord]:
        """讀取內存數據"""
        for i, item in enumerate(self.data):
            yield DataRecord(
                id=f"memory_{i}", data=item, source="memory", metadata={"index": i}
            )
            await asyncio.sleep(0)  # 讓出控制權

    async def close(self):
        """關閉內存數據源"""
        pass


class QueueDataSource:
    """隊列數據源（流式處理）"""

    def __init__(self, queue_name: str = "default"):
        self.queue_name = queue_name
        self.queue: Queue = Queue()
        self.is_closed = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}({queue_name})")

    def put_data(self, data: Any, metadata: dict[str, Any] | None = None):
        """向隊列添加數據"""
        if not self.is_closed:
            record = DataRecord(
                id=f"queue_{self.queue_name}_{uuid.uuid4().hex[:8]}",
                data=data,
                source=f"queue_{self.queue_name}",
                metadata=metadata or {},
            )
            self.queue.put(record)

    async def read_data(self) -> AsyncIterator[DataRecord]:
        """讀取隊列數據"""
        while not self.is_closed or not self.queue.empty():
            try:
                # 非阻塞讀取
                record = self.queue.get_nowait()
                yield record
                self.queue.task_done()
                await asyncio.sleep(0)  # 讓出控制權
            except Empty:
                # 如果隊列為空，短暫等待
                await asyncio.sleep(0.1)

    async def close(self):
        """關閉隊列數據源"""
        self.is_closed = True


class FileDataSink:
    """文件數據輸出"""

    def __init__(
        self, file_path: Path, format_type: str = "json", append: bool = False
    ):
        self.file_path = Path(file_path)
        self.format_type = format_type.lower()
        self.append = append
        self.logger = logging.getLogger(f"{self.__class__.__name__}({file_path})")
        self._file_handle = None
        self._records_written = 0

        # 確保目錄存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def write_data(self, record: DataRecord) -> bool:
        """寫入數據到文件"""
        try:
            if self._file_handle is None:
                mode = "a" if self.append else "w"
                self._file_handle = open(self.file_path, mode, encoding="utf-8")

            if self.format_type == "json":
                json.dump(record.to_dict(), self._file_handle, ensure_ascii=False)
                self._file_handle.write("\n")
            elif self.format_type == "text":
                self._file_handle.write(f"{record.data}\n")

            self._file_handle.flush()
            self._records_written += 1
            return True

        except Exception as e:
            self.logger.error(f"寫入文件失敗: {e}")
            return False

    async def close(self):
        """關閉文件輸出"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        self.logger.info(f"文件輸出已關閉，共寫入 {self._records_written} 條記錄")


class MemoryDataSink:
    """內存數據輸出"""

    def __init__(self):
        self.data: list[DataRecord] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    async def write_data(self, record: DataRecord) -> bool:
        """寫入數據到內存"""
        try:
            self.data.append(record)
            return True
        except Exception as e:
            self.logger.error(f"寫入內存失敗: {e}")
            return False

    def get_data(self) -> list[DataRecord]:
        """獲取所有數據"""
        return self.data.copy()

    async def close(self):
        """關閉內存輸出"""
        self.logger.info(f"內存輸出已關閉，共收集 {len(self.data)} 條記錄")


class TransformProcessor:
    """數據轉換處理器"""

    def __init__(self, transform_func: Callable[[Any], Any], name: str = "transform"):
        self.transform_func = transform_func
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}({name})")

    async def process(self, record: DataRecord) -> DataRecord:
        """處理數據記錄"""
        try:
            start_time = time.time()

            # 執行轉換
            transformed_data = self.transform_func(record.data)

            # 創建新記錄
            new_record = DataRecord(
                id=record.id,
                data=transformed_data,
                metadata=record.metadata.copy(),
                timestamp=record.timestamp,
                source=record.source,
                processing_history=record.processing_history.copy(),
            )

            new_record.add_processing_step(self.name)
            new_record.metadata[f"{self.name}_processing_time"] = (
                time.time() - start_time
            )

            return new_record

        except Exception as e:
            self.logger.error(f"數據轉換失敗: {e}")
            # 返回原記錄，但標記錯誤
            record.metadata[f"{self.name}_error"] = str(e)
            record.add_processing_step(f"{self.name}_error")
            return record


class FilterProcessor:
    """數據過濾處理器"""

    def __init__(self, filter_func: Callable[[Any], bool], name: str = "filter"):
        self.filter_func = filter_func
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}({name})")

    async def process(self, record: DataRecord) -> DataRecord | None:
        """處理數據記錄（過濾）"""
        try:
            if self.filter_func(record.data):
                record.add_processing_step(f"{self.name}_passed")
                return record
            else:
                record.add_processing_step(f"{self.name}_filtered")
                return None

        except Exception as e:
            self.logger.error(f"數據過濾失敗: {e}")
            record.metadata[f"{self.name}_error"] = str(e)
            record.add_processing_step(f"{self.name}_error")
            return record


class AggregateProcessor:
    """數據聚合處理器"""

    def __init__(
        self,
        window_size: int = 100,
        aggregate_func: Callable[[list[Any]], Any] | None = None,
    ):
        self.window_size = window_size
        self.aggregate_func = aggregate_func or self._default_aggregate
        self.window: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _default_aggregate(self, data_list: list[Any]) -> dict[str, Any]:
        """默認聚合函數"""
        return {
            "count": len(data_list),
            "first_item": data_list[0] if data_list else None,
            "last_item": data_list[-1] if data_list else None,
            "timestamp": time.time(),
        }

    async def process(self, record: DataRecord) -> DataRecord | None:
        """處理數據記錄（聚合）"""
        try:
            self.window.append(record.data)

            # 當窗口滿時進行聚合
            if len(self.window) >= self.window_size:
                aggregated_data = self.aggregate_func(list(self.window))

                # 創建聚合記錄
                aggregate_record = DataRecord(
                    id=f"aggregate_{uuid.uuid4().hex[:8]}",
                    data=aggregated_data,
                    metadata={
                        "aggregate_window_size": self.window_size,
                        "source_records": len(self.window),
                    },
                    source="aggregator",
                )

                aggregate_record.add_processing_step("aggregate")

                # 清空窗口的一半，保持滑動窗口效果
                for _ in range(self.window_size // 2):
                    if self.window:
                        self.window.popleft()

                return aggregate_record

            return None

        except Exception as e:
            self.logger.error(f"數據聚合失敗: {e}")
            return None


class DataPipeline:
    """
    數據管道

    支持多種處理模式和複雜的數據處理鏈
    """

    def __init__(
        self,
        name: str,
        mode: ProcessingMode = ProcessingMode.STREAMING,
        config_manager: ConfigManager | None = None,
    ):
        self.name = name
        self.mode = mode
        self.config_manager = config_manager or get_config_manager()
        self.logger = logging.getLogger(f"{self.__class__.__name__}({name})")

        # 管道組件
        self.data_sources: list[DataSource] = []
        self.processors: list[DataProcessor] = []
        self.data_sinks: list[DataSink] = []

        # 管道狀態
        self.status = PipelineStatus.CREATED
        self.metrics = PipelineMetrics()
        self.error_handler = ErrorHandler()

        # 並發控制
        max_tasks = self.config_manager.get_setting("pipeline.max_concurrent_tasks", 10)
        batch_size = self.config_manager.get_setting("pipeline.batch_size", 100)
        self.max_concurrent_tasks: int = max_tasks if max_tasks is not None else 10
        self.batch_size: int = batch_size if batch_size is not None else 100

        # 控制變量
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._running_tasks: set[asyncio.Task] = set()

    def add_source(self, source: DataSource):
        """添加數據源"""
        self.data_sources.append(source)
        self.logger.info(f"添加數據源: {type(source).__name__}")

    def add_processor(self, processor: DataProcessor):
        """添加處理器"""
        self.processors.append(processor)
        self.logger.info(f"添加處理器: {type(processor).__name__}")

    def add_sink(self, sink: DataSink):
        """添加數據輸出"""
        self.data_sinks.append(sink)
        self.logger.info(f"添加數據輸出: {type(sink).__name__}")

    async def start(self):
        """啟動管道"""
        if self.status != PipelineStatus.CREATED:
            self.logger.warning(f"管道狀態不正確，無法啟動: {self.status}")
            return

        self.status = PipelineStatus.STARTING
        self.metrics.start_time = time.time()
        self.logger.info(f"啟動數據管道: {self.name} (模式: {self.mode.value})")

        try:
            if self.mode == ProcessingMode.STREAMING:
                await self._run_streaming_mode()
            elif self.mode == ProcessingMode.BATCH:
                await self._run_batch_mode()
            elif self.mode == ProcessingMode.REALTIME:
                await self._run_realtime_mode()
            elif self.mode == ProcessingMode.HYBRID:
                await self._run_hybrid_mode()

            self.status = PipelineStatus.COMPLETED
            self.logger.info(f"數據管道完成: {self.name}")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            self.logger.error(f"數據管道執行失敗: {e}")
            error_msg = f"數據管道執行失敗: {self.name} - {str(e)}"
            await self.error_handler.handle_error(
                error_msg,
                error_type=ErrorType.PROCESSING_ERROR,
                severity=ErrorSeverity.HIGH,
            )
        finally:
            await self._cleanup()

    async def stop(self):
        """停止管道"""
        self.logger.info(f"停止數據管道: {self.name}")
        self.status = PipelineStatus.STOPPING
        self._stop_event.set()

        # 等待所有任務完成
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        self.status = PipelineStatus.STOPPED

    async def pause(self):
        """暫停管道"""
        self.logger.info(f"暫停數據管道: {self.name}")
        self.status = PipelineStatus.PAUSED
        self._pause_event.clear()

    async def resume(self):
        """恢復管道"""
        self.logger.info(f"恢復數據管道: {self.name}")
        self.status = PipelineStatus.RUNNING
        self._pause_event.set()

    async def _run_streaming_mode(self):
        """流式處理模式"""
        self.status = PipelineStatus.RUNNING
        self._pause_event.set()

        # 為每個數據源創建處理任務
        source_tasks = []
        for source in self.data_sources:
            task = asyncio.create_task(self._process_source_streaming(source))
            self._running_tasks.add(task)
            source_tasks.append(task)

        # 等待所有源處理完成
        await asyncio.gather(*source_tasks, return_exceptions=True)

    async def _process_source_streaming(self, source: DataSource):
        """處理單個數據源（流式）"""
        try:
            async for record in source.read_data():
                # 檢查停止和暫停信號
                if self._stop_event.is_set():
                    break

                await self._pause_event.wait()

                # 處理記錄
                await self._process_record(record)

        except Exception as e:
            self.logger.error(f"流式處理數據源失敗: {e}")
        finally:
            await source.close()

    async def _run_batch_mode(self):
        """批量處理模式"""
        self.status = PipelineStatus.RUNNING

        # 收集所有數據
        all_records = []
        for source in self.data_sources:
            try:
                async for record in source.read_data():
                    all_records.append(record)
                await source.close()
            except Exception as e:
                self.logger.error(f"讀取數據源失敗: {e}")

        # 分批處理
        total_batches = (len(all_records) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(all_records), self.batch_size):
            batch = all_records[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            self.logger.info(
                f"處理批次 {batch_num}/{total_batches}, 記錄數: {len(batch)}"
            )

            # 並發處理批次中的記錄
            tasks = [self._process_record(record) for record in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

            if self._stop_event.is_set():
                break

    async def _run_realtime_mode(self):
        """實時處理模式"""
        self.status = PipelineStatus.RUNNING
        self._pause_event.set()

        # 實時模式類似流式，但有更嚴格的延遲要求
        await self._run_streaming_mode()

    async def _run_hybrid_mode(self):
        """混合處理模式"""
        self.status = PipelineStatus.RUNNING

        # 混合模式結合批量和流式處理
        # 可以根據數據源類型選擇不同的處理方式
        batch_sources = []
        stream_sources = []

        for source in self.data_sources:
            # 默認所有數據源都支持流式處理
            # 可以根據具體實現類型做更精細的判斷
            if isinstance(source, (QueueDataSource,)):
                stream_sources.append(source)
            else:
                batch_sources.append(source)

        # 同時處理批量和流式數據源
        tasks = []

        if batch_sources:
            self.data_sources = batch_sources
            task = asyncio.create_task(self._run_batch_mode())
            tasks.append(task)

        if stream_sources:
            for source in stream_sources:
                task = asyncio.create_task(self._process_source_streaming(source))
                self._running_tasks.add(task)
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_record(self, record: DataRecord):
        """處理單條記錄"""
        start_time = time.time()
        success = True

        try:
            current_record = record

            # 通過處理器鏈處理記錄
            for processor in self.processors:
                if current_record is None:
                    break

                if hasattr(processor, "process"):
                    result = await processor.process(current_record)
                    current_record = result
                else:
                    self.logger.warning(
                        f"處理器 {type(processor).__name__} 沒有 process 方法"
                    )

            # 輸出到所有接收器
            if current_record is not None:
                for sink in self.data_sinks:
                    try:
                        await sink.write_data(current_record)
                    except Exception as e:
                        self.logger.error(f"寫入數據接收器失敗: {e}")
                        success = False

        except Exception as e:
            self.logger.error(f"處理記錄失敗: {e}")
            success = False

        # 更新指標
        processing_time = time.time() - start_time
        self.metrics.update_metrics(processing_time, success)

    async def _cleanup(self):
        """清理資源"""
        # 關閉所有數據源
        for source in self.data_sources:
            try:
                await source.close()
            except Exception as e:
                self.logger.error(f"關閉數據源失敗: {e}")

        # 關閉所有數據接收器
        for sink in self.data_sinks:
            try:
                await sink.close()
            except Exception as e:
                self.logger.error(f"關閉數據接收器失敗: {e}")

        # 清理運行中的任務
        self._running_tasks.clear()

    def get_metrics(self) -> dict[str, Any]:
        """獲取管道指標"""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "status": self.status.value,
            "records_processed": self.metrics.records_processed,
            "records_failed": self.metrics.records_failed,
            "total_processing_time": self.metrics.total_processing_time,
            "average_processing_time": self.metrics.average_processing_time,
            "throughput_per_second": self.metrics.throughput_per_second,
            "error_rate": self.metrics.error_rate,
            "uptime": (
                time.time() - self.metrics.start_time if self.metrics.start_time else 0
            ),
            "sources_count": len(self.data_sources),
            "processors_count": len(self.processors),
            "sinks_count": len(self.data_sinks),
        }


class PipelineManager:
    """
    管道管理器

    管理多個數據管道的生命週期
    """

    def __init__(self, config_manager: ConfigManager | None = None):
        self.config_manager = config_manager or get_config_manager()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.pipelines: dict[str, DataPipeline] = {}
        self.pipeline_tasks: dict[str, asyncio.Task] = {}

        # 全局指標
        self.total_pipelines_created = 0
        self.active_pipelines = 0

    def create_pipeline(
        self, name: str, mode: ProcessingMode = ProcessingMode.STREAMING
    ) -> DataPipeline:
        """創建新管道"""
        if name in self.pipelines:
            raise ValueError(f"管道 {name} 已存在")

        pipeline = DataPipeline(name, mode, self.config_manager)
        self.pipelines[name] = pipeline
        self.total_pipelines_created += 1

        self.logger.info(f"創建數據管道: {name} (模式: {mode.value})")
        return pipeline

    async def start_pipeline(self, name: str):
        """啟動管道"""
        if name not in self.pipelines:
            raise ValueError(f"管道 {name} 不存在")

        if name in self.pipeline_tasks:
            self.logger.warning(f"管道 {name} 已在運行")
            return

        pipeline = self.pipelines[name]
        task = asyncio.create_task(pipeline.start())
        self.pipeline_tasks[name] = task
        self.active_pipelines += 1

        self.logger.info(f"啟動管道: {name}")

    async def stop_pipeline(self, name: str):
        """停止管道"""
        if name not in self.pipelines:
            raise ValueError(f"管道 {name} 不存在")

        pipeline = self.pipelines[name]
        await pipeline.stop()

        if name in self.pipeline_tasks:
            task = self.pipeline_tasks[name]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.pipeline_tasks[name]
            self.active_pipelines -= 1

        self.logger.info(f"停止管道: {name}")

    async def stop_all_pipelines(self):
        """停止所有管道"""
        pipeline_names = list(self.pipeline_tasks.keys())
        for name in pipeline_names:
            await self.stop_pipeline(name)

        self.logger.info("所有管道已停止")

    def get_pipeline(self, name: str) -> DataPipeline | None:
        """獲取管道"""
        return self.pipelines.get(name)

    def list_pipelines(self) -> list[str]:
        """列出所有管道名稱"""
        return list(self.pipelines.keys())

    def get_pipeline_status(self, name: str) -> PipelineStatus | None:
        """獲取管道狀態"""
        pipeline = self.pipelines.get(name)
        return pipeline.status if pipeline else None

    def get_all_metrics(self) -> dict[str, Any]:
        """獲取所有管道指標"""
        return {
            "manager_stats": {
                "total_pipelines_created": self.total_pipelines_created,
                "active_pipelines": self.active_pipelines,
                "registered_pipelines": len(self.pipelines),
            },
            "pipelines": {
                name: pipeline.get_metrics()
                for name, pipeline in self.pipelines.items()
            },
        }


# 全局管道管理器實例
_global_pipeline_manager: PipelineManager | None = None


def get_pipeline_manager(
    config_manager: ConfigManager | None = None,
) -> PipelineManager:
    """獲取全局管道管理器實例"""
    global _global_pipeline_manager

    if _global_pipeline_manager is None:
        _global_pipeline_manager = PipelineManager(config_manager)

    return _global_pipeline_manager


# 便利函數
def create_simple_pipeline(
    name: str,
    source_data: list[Any],
    transform_func: Callable[[Any], Any] | None = None,
    filter_func: Callable[[Any], bool] | None = None,
) -> DataPipeline:
    """創建簡單的數據管道"""
    manager = get_pipeline_manager()
    pipeline = manager.create_pipeline(name, ProcessingMode.BATCH)

    # 添加內存數據源
    source = MemoryDataSource(source_data)
    pipeline.add_source(source)

    # 添加轉換處理器
    if transform_func:
        processor = TransformProcessor(transform_func, "transform")
        pipeline.add_processor(processor)

    # 添加過濾處理器
    if filter_func:
        filter_processor = FilterProcessor(filter_func, "filter")
        pipeline.add_processor(filter_processor)

    # 添加內存輸出
    sink = MemoryDataSink()
    pipeline.add_sink(sink)

    return pipeline
