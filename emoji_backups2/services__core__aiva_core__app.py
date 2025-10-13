from __future__ import annotations

import asyncio
from collections import Counter
import json
from typing import Any

from fastapi import FastAPI

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, ScanCompletedPayload
from services.aiva_common.utils import get_logger
from services.core.aiva_core.analysis.dynamic_strategy_adjustment import (
    StrategyAdjuster,
)
from services.core.aiva_core.analysis.initial_surface import InitialAttackSurface

# from services.core.aiva_core.analysis.test_strategy_generation import StrategyGenerator  # noqa: E501
from services.core.aiva_core.execution.execution_status_monitor import (
    ExecutionStatusMonitor,
)
from services.core.aiva_core.execution.task_generator import TaskGenerator
from services.core.aiva_core.execution.task_queue_manager import TaskQueueManager
from services.core.aiva_core.ingestion.scan_module_interface import ScanModuleInterface
from services.core.aiva_core.output.to_functions import to_function_message
from services.core.aiva_core.state.session_state_manager import SessionStateManager

app = FastAPI(
    title="AIVA Core Engine - 智慧分析與協調中心",
    description="核心分析引擎：攻擊面分析、策略生成、任務協調",
    version="1.0.0",
)
logger = get_logger(__name__)


def _count_tasks_by_type(tasks: list) -> dict[str, int]:
    """統計各類型任務的數量"""
    return dict(Counter(topic.value for topic, _ in tasks))


# 核心組件初始化 - 按照架構文檔的五大子系統
# 1. 資料接收與預處理
scan_interface = ScanModuleInterface()

# 2. 分析與策略引擎
surface_analyzer = InitialAttackSurface()
# strategy_generator = StrategyGenerator()  # Removed: legacy code
strategy_adjuster = StrategyAdjuster()

# 3. 任務協調與執行
task_generator = TaskGenerator()
task_queue_manager = TaskQueueManager()
execution_monitor = ExecutionStatusMonitor()

# 4. 狀態與知識庫管理
session_state_manager = SessionStateManager()


@app.on_event("startup")
async def startup() -> None:
    """啟動核心引擎服務"""
    logger.info("[啟動] AIVA Core Engine starting up...")
    logger.info("[統計] Initializing analysis components...")
    logger.info("[循環] Starting message processing loops...")

    # 啟動各種處理任務
    asyncio.create_task(process_scan_results())
    asyncio.create_task(process_function_results())
    asyncio.create_task(monitor_execution_status())


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


async def process_scan_results() -> None:
    """
    處理掃描模組回傳的結果 - 核心分析與策略生成
    這是第3階段: 核心分析與建議的主要邏輯
    """
    logger.info("[API] Starting scan results processor...")
    broker = await get_broker()

    aiterator = broker.subscribe(Topic.RESULTS_SCAN_COMPLETED)
    if hasattr(aiterator, "__await__"):
        aiterator = await aiterator  # type: ignore[misc]

    async for mqmsg in aiterator:  # type: ignore[misc]
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            payload = ScanCompletedPayload(**msg.payload)
            scan_id = payload.scan_id

            logger.info(f"🔍 [Stage 1/7] Processing scan results for {scan_id}")

            # ===== 階段1：資料接收與預處理 (Data Ingestion) =====
            await scan_interface.process_scan_data(payload)  # 處理但不需要暫存
            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 1,
                    "total_assets": len(payload.assets),
                    "urls_found": payload.summary.urls_found,
                    "forms_found": payload.summary.forms_found,
                    "apis_found": payload.summary.apis_found,
                },
            )
            logger.info(
                f"[接收] [Stage 1/7] Data ingested - "
                f"Assets: {len(payload.assets)}, "
                f"URLs: {payload.summary.urls_found}, "
                f"Forms: {payload.summary.forms_found}"
            )

            # ===== 階段2：初步攻擊面分析 (Initial Attack Surface Analysis) =====
            logger.info(f"🔍 [Stage 2/7] Analyzing attack surface for {scan_id}")
            # 直接使用payload而非processed_data
            attack_surface = surface_analyzer.analyze(payload)
            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 2,
                    "attack_surface": attack_surface,
                    "high_risk_count": attack_surface.get("high_risk_assets", 0),
                    "medium_risk_count": attack_surface.get("medium_risk_assets", 0),
                },
            )
            logger.info(
                f"[列表] [Stage 2/7] Attack surface identified - "
                f"High risk: {attack_surface.get('high_risk_assets', 0)}, "
                f"Medium risk: {attack_surface.get('medium_risk_assets', 0)}"
            )

            # ===== 階段3：測試策略生成 (Test Strategy Generation) =====
            logger.info(f"[目標] [Stage 3/7] Generating test strategy for {scan_id}")
            # Legacy strategy generator removed - using direct strategy
            base_strategy = {"test_plans": [], "strategy_type": "default"}
            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 3,
                    "base_strategy": base_strategy,
                    "planned_tests": len(base_strategy.get("test_plans", [])),
                },
            )
            logger.info(
                f"[記錄] [Stage 3/7] Base strategy generated - "
                f"Tests: {len(base_strategy.get('test_plans', []))}"
            )

            # ===== 階段4：動態策略調整 (Dynamic Strategy Adjustment) =====
            logger.info(
                f"[設定] [Stage 4/7] Adjusting strategy based on context for {scan_id}"
            )
            session_context = session_state_manager.get_session_context(scan_id)
            # 將fingerprints整合到context中
            enriched_context = {**session_context, "fingerprints": payload.fingerprints}
            adjusted_strategy = strategy_adjuster.adjust(
                base_strategy, enriched_context
            )
            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 4,
                    "adjusted_strategy": adjusted_strategy,
                    "optimizations_applied": adjusted_strategy.get("optimizations", []),
                },
            )
            logger.info(
                f"[調整] [Stage 4/7] Strategy adjusted - "
                f"Optimizations: {len(adjusted_strategy.get('optimizations', []))}"
            )

            # ===== 階段5：任務生成 (Task Generation) =====
            logger.info(f"[快速] [Stage 5/7] Generating tasks for {scan_id}")
            # 將generator轉為list以便重複使用
            tasks = list(task_generator.from_strategy(adjusted_strategy, payload))
            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 5,
                    "total_tasks": len(tasks),
                    "tasks_by_type": _count_tasks_by_type(tasks),
                },
            )
            logger.info(
                f"📦 [Stage 5/7] Tasks generated - "
                f"Total: {len(tasks)}, "
                f"Types: {_count_tasks_by_type(tasks)}"
            )

            # Stage 6: Task Queue Management & Distribution
            logger.info(f"📤 [Stage 6/7] Dispatching tasks for {scan_id}")
            dispatched_count = 0
            for topic, task_payload in tasks:
                # 將任務加入佇列管理
                task_queue_manager.enqueue_task(topic, task_payload)

                # 生成並發送功能模組任務
                out = to_function_message(
                    topic,
                    task_payload,
                    trace_id=msg.header.trace_id,
                    correlation_id=scan_id,
                )
                await broker.publish(
                    topic, json.dumps(out.model_dump()).encode("utf-8")
                )
                dispatched_count += 1

            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 6,
                    "dispatched_tasks": dispatched_count,
                    "pending_tasks": len(tasks),
                },
            )
            logger.info(f"[啟動] [Stage 6/7] Dispatched {dispatched_count} tasks")

            # ===== 階段7：執行狀態監控 (Execution Status Monitoring) =====
            logger.info(f"[監控] [Stage 7/7] Monitoring execution for {scan_id}")
            session_state_manager.update_context(
                scan_id,
                {
                    "stage": 7,
                    "status": "monitoring",
                    # 修正：將欄位名稱改為更能反映其內容的名稱
                    "scan_duration_seconds": payload.summary.scan_duration_seconds,
                    # 注意：核心引擎無法得知確切開始時間，因此移除 "start_time" 欄位
                },
            )
            session_state_manager.update_session_status(
                scan_id,
                "analysis_completed",
                {
                    "tasks_dispatched": dispatched_count,
                    "monitoring_active": True,
                },
            )

            logger.info(f"[已] [Stage 7/7] All stages completed for {scan_id}")

        except Exception as e:
            logger.error(f"[失敗] Error processing scan results: {e}")


async def process_function_results() -> None:
    """
    處理功能模組回傳的結果 - 用於下一輪優化
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
    logger.info("📈 Starting execution status monitor...")

    while True:
        try:
            # 每30秒檢查一次系統狀態
            await asyncio.sleep(30)

            # 獲取系統健康狀態
            system_status = execution_monitor.get_system_health()

            # 檢查是否有異常情況需要處理
            if system_status.get("status") != "healthy":
                logger.warning(f"[警告] System health issue: {system_status}")

        except Exception as e:
            logger.error(f"[失敗] Error in status monitoring: {e}")
