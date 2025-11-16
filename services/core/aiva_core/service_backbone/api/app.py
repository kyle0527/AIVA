"""AIVA Core API - 系統唯一入口點

職責:
1. FastAPI 應用程序主入口 - 系統的唯一啟動點
2. 持有 CoreServiceCoordinator 作為狀態管理器
3. 提供 RESTful API 端點
4. 啟動內部閉環和外部學習後台任務

架構層次:
    app.py (FastAPI)          ← 唯一主入口
        ↓ 持有
    CoreServiceCoordinator    ← 狀態管理器和服務工廠
        ↓ 管理
    各功能服務 (Decision, Planning, Execution...)
"""

import asyncio
from collections import Counter
from typing import Any

from fastapi import FastAPI
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from services.aiva_common.config import get_settings
from services.aiva_common.enums.modules import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, ScanCompletedPayload
from services.aiva_common.utils import get_logger
from services.core.aiva_core.external_learning.analysis.dynamic_strategy_adjustment import (
    StrategyAdjuster,
)
from services.core.aiva_core.core_capabilities.analysis.initial_surface import InitialAttackSurface
from services.core.aiva_core.task_planning.executor.execution_status_monitor import (
    ExecutionStatusMonitor,
)
from services.core.aiva_core.task_planning.planner.task_generator import TaskGenerator
from services.core.aiva_core.task_planning.executor.task_queue_manager import TaskQueueManager
from services.core.aiva_core.core_capabilities.ingestion.scan_module_interface import ScanModuleInterface
from services.core.aiva_core.core_capabilities.processing import ScanResultProcessor
from services.core.aiva_core.service_backbone.state.session_state_manager import SessionStateManager

# ✅ 引入 CoreServiceCoordinator 作為狀態管理器
from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
    AIVACoreServiceCoordinator,
)

# ✅ 引入內部閉環和外部學習組件
from services.core.aiva_core.internal_exploration.connectors.update_self_awareness import (
    periodic_update,
)
from services.core.aiva_core.external_learning.connectors.external_loop_connector import (
    ExternalLoopConnector,
)

app = FastAPI(
    title="AIVA Core Engine - 智慧分析與協調中心",
    description="核心分析引擎：攻擊面分析、策略生成、任務協調 (系統唯一入口)",
    version="3.0.0",
)
logger = get_logger(__name__)

# ✅ 全局協調器實例（狀態管理器，非主線程）
coordinator: AIVACoreServiceCoordinator | None = None

# ✅ 全局後台任務引用（防止垃圾回收）
_background_tasks: list[asyncio.Task] = []


def _count_tasks_by_type(tasks: list) -> dict[str, int]:
    """統計各類型任務的數量"""
    return dict(Counter(topic.value for topic, _ in tasks))


# 核心組件初始化 - 按照架構文檔的五大子系統
# 1. 資料接收與預處理
scan_interface = ScanModuleInterface()

# 2. 分析與策略引擎
surface_analyzer = InitialAttackSurface()
strategy_adjuster = StrategyAdjuster()

# 3. 任務協調與執行
task_generator = TaskGenerator()
task_queue_manager = TaskQueueManager()
execution_monitor = ExecutionStatusMonitor()

# 4. 狀態與知識庫管理
session_state_manager = SessionStateManager()

# 5. 掃描結果處理器 (新增 - 封裝七階段處理流程)
scan_result_processor = ScanResultProcessor(
    scan_interface=scan_interface,
    surface_analyzer=surface_analyzer,
    strategy_adjuster=strategy_adjuster,
    task_generator=task_generator,
    task_queue_manager=task_queue_manager,
    session_state_manager=session_state_manager,
)




@app.on_event("startup")
async def startup() -> None:
    """啟動核心引擎服務 - 系統唯一啟動點
    
    啟動流程:
    1. 初始化 CoreServiceCoordinator（狀態管理器）
    2. 啟動內部閉環更新（後台任務）
    3. 啟動外部學習監聽器（後台任務）
    4. 啟動掃描結果處理（後台任務）
    5. 啟動功能結果處理（後台任務）
    6. 啟動執行狀態監控（後台任務）
    """
    global coordinator
    
    logger.info("🚀 [啟動] AIVA Core Engine starting up...")
    
    # ✅ Step 1: 初始化協調器（作為狀態管理器，非主線程）
    coordinator = AIVACoreServiceCoordinator()
    await coordinator.start()
    logger.info("✅ [啟動] CoreServiceCoordinator initialized (state manager mode)")
    
    # ✅ Step 2: 啟動內部閉環更新（P0 問題一）
    _background_tasks.append(asyncio.create_task(
        periodic_update(),
        name="internal_loop_update"
    ))
    logger.info("✅ [啟動] Internal exploration loop started")
    
    # ✅ Step 3: 啟動外部學習監聽器（P0 問題二）
    external_connector = ExternalLoopConnector()
    _background_tasks.append(asyncio.create_task(
        external_connector.start_listening(),
        name="external_learning_loop"
    ))
    logger.info("✅ [啟動] External learning listener started")
    
    # ✅ Step 4-6: 啟動核心處理循環
    logger.info("[統計] Initializing analysis components...")
    logger.info("[循環] Starting message processing loops...")
    
    _background_tasks.append(asyncio.create_task(
        process_scan_results(),
        name="scan_results_processor"
    ))
    
    _background_tasks.append(asyncio.create_task(
        process_function_results(),
        name="function_results_processor"
    ))
    
    _background_tasks.append(asyncio.create_task(
        monitor_execution_status(),
        name="execution_monitor"
    ))
    
    logger.info("✅ [啟動] All background tasks started")
    logger.info("🎉 [啟動] AIVA Core Engine ready to accept requests!")


@app.on_event("shutdown")
async def shutdown() -> None:
    """關閉核心引擎服務"""
    global coordinator
    
    logger.info("🛑 [關閉] AIVA Core Engine shutting down...")
    
    if coordinator:
        await coordinator.stop()
        logger.info("✅ [關閉] CoreServiceCoordinator stopped")
    
    logger.info("👋 [關閉] AIVA Core Engine shutdown complete")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": "aiva-core-engine",
        "components": {
            "scan_interface": "active",
            "analysis_engine": "active",
            "task_coordinator": "active",
            "state_manager": "active",
        },
    }


@app.get("/status/{scan_id}")
async def get_scan_status(scan_id: str) -> dict[str, str]:
    """獲取掃描狀態"""
    return session_state_manager.get_session_status(scan_id)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def _process_single_scan_with_retry(
    payload: ScanCompletedPayload, trace_id: str
) -> None:
    """可重試的掃描處理邏輯

    Args:
        payload: 掃描完成載荷
        trace_id: 追蹤 ID

    Raises:
        Exception: 當所有重試都失敗時拋出
    """
    broker = await get_broker()
    await scan_result_processor.process(payload, broker, trace_id)


async def process_scan_results() -> None:
    """處理掃描模組回傳的結果 - 核心分析與策略生成
    這是第3階段: 核心分析與建議的主要邏輯
    """
    logger.info("[API] Starting scan results processor...")
    broker = await get_broker()

    aiterator = broker.subscribe(Topic.RESULTS_SCAN_COMPLETED)
    if hasattr(aiterator, "__await__"):
        aiterator = await aiterator  # type: ignore[misc]

    async for mqmsg in aiterator:  # type: ignore[misc]
        msg = AivaMessage.model_validate_json(mqmsg.body)
        payload = ScanCompletedPayload(**msg.payload)
        scan_id = payload.scan_id

        try:
            # 使用重試機制處理掃描結果
            await _process_single_scan_with_retry(payload, msg.header.trace_id)

        except RetryError as retry_err:
            # 所有重試都失敗
            logger.error(
                f"[失敗] All retries exhausted for scan {scan_id}: {retry_err}",
                exc_info=True,
            )
            # 更新掃描狀態為失敗
            session_state_manager.update_session_status(
                scan_id,
                "failed",
                {
                    "error": str(retry_err),
                    "error_type": "retry_exhausted",
                    "retry_attempts": 3,
                },
            )

        except Exception as e:
            # 非預期的錯誤
            logger.error(
                f"[失敗] Unexpected error processing scan {scan_id}: {e}",
                exc_info=True,
            )
            # 更新掃描狀態為失敗
            session_state_manager.update_session_status(
                scan_id,
                "failed",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )


async def process_function_results() -> None:
    """處理功能模組回傳的結果 - 用於下一輪優化
    實現動態學習與策略調整
    """
    logger.info("[循環] Starting function results processor...")
    broker = await get_broker()

    # 監聽所有功能模組的結果
    aiterator = broker.subscribe(Topic.LOG_RESULTS_ALL)
    if hasattr(aiterator, "__await__"):
        aiterator = await aiterator  # type: ignore[misc]

    async for mqmsg in aiterator:  # type: ignore[misc]
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            result_data = msg.payload

            # 提取相關資訊
            scan_id = result_data.get("scan_id")
            vulnerability_info = result_data.get("vulnerability", {})

            logger.info(f"[統計] Received result from {msg.header.source_module}")

            # 回饋給策略調整器，用於改善下次決策
            feedback_data = {
                "scan_id": scan_id,
                "module": msg.header.source_module,
                "vulnerability": vulnerability_info,
                "success": vulnerability_info.get("confidence") == "CONFIRMED",
            }

            # 更新長期知識庫
            strategy_adjuster.learn_from_result(feedback_data)

        except Exception as e:
            logger.error(f"[失敗] Error processing function result: {e}")


async def monitor_execution_status() -> None:
    """監控執行狀態與效能"""
    settings = get_settings()
    logger.info(
        f"[📈] Starting execution status monitor "
        f"(interval: {settings.core_monitor_interval}s)..."
    )

    while True:
        try:
            # 使用配置的監控間隔
            await asyncio.sleep(settings.core_monitor_interval)

            # 獲取系統健康狀態
            system_status = execution_monitor.get_system_health()

            # 檢查是否有異常情況需要處理
            if system_status.get("status") != "healthy":
                logger.warning(f"[警告] System health issue: {system_status}")

        except Exception as e:
            logger.error(f"[失敗] Error in status monitoring: {e}")
