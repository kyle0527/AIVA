"""掃描結果處理器 - 七階段處理流程

此模組封裝了核心引擎處理掃描結果的完整七階段流程,
提高了程式碼的可讀性和可維護性。
"""

import json
from typing import TYPE_CHECKING

from services.aiva_common.schemas import ScanCompletedPayload
from services.aiva_common.utils import get_logger
from services.aiva_common.mq import AbstractBroker
from services.core.aiva_core.core_capabilities.ingestion.scan_module_interface import ScanModuleInterface
from services.core.aiva_core.external_learning.analysis.dynamic_strategy_adjustment import StrategyAdjuster
from services.core.aiva_core.core_capabilities.analysis.initial_surface import InitialAttackSurface
from services.core.aiva_core.task_planning.planner.task_generator import TaskGenerator
from services.core.aiva_core.task_planning.executor.task_queue_manager import TaskQueueManager
from services.core.aiva_core.service_backbone.state.session_state_manager import SessionStateManager

if TYPE_CHECKING:
    pass  # 保留為將來的僅類型檢查導入

logger = get_logger(__name__)


class ScanResultProcessor:
    """掃描結果處理器 - 負責執行七階段處理流程

    七個階段:
    1. 資料接收與預處理 (Data Ingestion)
    2. 初步攻擊面分析 (Initial Attack Surface Analysis)
    3. 測試策略生成 (Test Strategy Generation)
    4. 動態策略調整 (Dynamic Strategy Adjustment)
    5. 任務生成 (Task Generation)
    6. 任務佇列管理與分發 (Task Queue Management & Distribution)
    7. 執行狀態監控 (Execution Status Monitoring)
    """

    def __init__(
        self,
        scan_interface: ScanModuleInterface,
        surface_analyzer: InitialAttackSurface,
        strategy_adjuster: StrategyAdjuster,
        task_generator: TaskGenerator,
        task_queue_manager: TaskQueueManager,
        session_state_manager: SessionStateManager,
    ):
        """初始化處理器

        Args:
            scan_interface: 掃描模組介面
            surface_analyzer: 攻擊面分析器
            strategy_adjuster: 策略調整器
            task_generator: 任務生成器
            task_queue_manager: 任務佇列管理器
            session_state_manager: 會話狀態管理器
        """
        self.scan_interface = scan_interface
        self.surface_analyzer = surface_analyzer
        self.strategy_adjuster = strategy_adjuster
        self.task_generator = task_generator
        self.task_queue_manager = task_queue_manager
        self.session_state_manager = session_state_manager

    async def stage_1_ingest_data(self, payload: ScanCompletedPayload) -> None:
        """階段1: 資料接收與預處理 (Data Ingestion)

        Args:
            payload: 掃描完成載荷
        """
        scan_id = payload.scan_id
        logger.info(f"[🔍] [Stage 1/7] Processing scan results for {scan_id}")

        await self.scan_interface.process_scan_data(payload)
        self.session_state_manager.update_context(
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

    async def stage_2_analyze_surface(
        self, payload: ScanCompletedPayload
    ) -> dict[str, int]:
        """階段2: 初步攻擊面分析 (Initial Attack Surface Analysis)

        Args:
            payload: 掃描完成載荷

        Returns:
            攻擊面分析結果
        """
        scan_id = payload.scan_id
        logger.info(f"[🔍] [Stage 2/7] Analyzing attack surface for {scan_id}")

        attack_surface = self.surface_analyzer.analyze(payload)

        # 安全地訪問 AttackSurfaceAnalysis 的屬性
        high_risk_count = getattr(attack_surface, "high_risk_assets", 0)
        medium_risk_count = getattr(attack_surface, "medium_risk_assets", 0)

        self.session_state_manager.update_context(
            scan_id,
            {
                "stage": 2,
                "attack_surface": attack_surface,
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
            },
        )
        logger.info(
            f"[列表] [Stage 2/7] Attack surface identified - "
            f"High risk: {high_risk_count}, "
            f"Medium risk: {medium_risk_count}"
        )
        return attack_surface

    async def stage_3_generate_strategy(self, scan_id: str) -> dict:
        """階段3: 測試策略生成 (Test Strategy Generation)

        Args:
            scan_id: 掃描 ID

        Returns:
            基礎策略
        """
        logger.info(f"[目標] [Stage 3/7] Generating test strategy for {scan_id}")

        # Legacy strategy generator removed - using direct strategy
        base_strategy = {"test_plans": [], "strategy_type": "default"}
        self.session_state_manager.update_context(
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
        return base_strategy

    async def stage_4_adjust_strategy(
        self, scan_id: str, base_strategy: dict, payload: ScanCompletedPayload
    ) -> dict:
        """階段4: 動態策略調整 (Dynamic Strategy Adjustment)

        Args:
            scan_id: 掃描 ID
            base_strategy: 基礎策略
            payload: 掃描完成載荷

        Returns:
            調整後的策略
        """
        logger.info(
            f"[設定] [Stage 4/7] Adjusting strategy based on context for {scan_id}"
        )

        session_context = self.session_state_manager.get_session_context(scan_id)
        # 將 fingerprints 整合到 context 中
        enriched_context = {**session_context, "fingerprints": payload.fingerprints}
        adjusted_strategy = self.strategy_adjuster.adjust(
            base_strategy, enriched_context
        )

        self.session_state_manager.update_context(
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
        return adjusted_strategy

    async def stage_5_generate_tasks(
        self, scan_id: str, adjusted_strategy: dict, payload: ScanCompletedPayload
    ) -> list:
        """階段5: 任務生成 (Task Generation)

        Args:
            scan_id: 掃描 ID
            adjusted_strategy: 調整後的策略
            payload: 掃描完成載荷

        Returns:
            生成的任務列表
        """
        logger.info(f"[快速] [Stage 5/7] Generating tasks for {scan_id}")

        # 將 generator 轉為 list 以便重複使用
        tasks = list(self.task_generator.from_strategy(adjusted_strategy, payload))

        # 統計任務類型
        from collections import Counter

        tasks_by_type = dict(Counter(topic.value for topic, _ in tasks))

        self.session_state_manager.update_context(
            scan_id,
            {
                "stage": 5,
                "total_tasks": len(tasks),
                "tasks_by_type": tasks_by_type,
            },
        )
        logger.info(
            f"[📦] [Stage 5/7] Tasks generated - "
            f"Total: {len(tasks)}, "
            f"Types: {tasks_by_type}"
        )
        return tasks

    async def stage_6_dispatch_tasks(
        self,
        scan_id: str,
        tasks: list,
        broker: AbstractBroker,
        trace_id: str,
    ) -> int:
        """階段6: 任務佇列管理與分發 (Task Queue Management & Distribution)

        Args:
            scan_id: 掃描 ID
            tasks: 任務列表
            broker: 訊息代理
            trace_id: 追蹤 ID

        Returns:
            已分發的任務數量
        """
        from services.core.aiva_core.output.to_functions import to_function_message

        logger.info(f"[📤] [Stage 6/7] Dispatching tasks for {scan_id}")

        dispatched_count = 0
        for topic, task_payload in tasks:
            # 將任務加入佇列管理
            self.task_queue_manager.enqueue_task(topic, task_payload)

            # 生成並發送功能模組任務
            out = to_function_message(
                topic,
                task_payload,
                trace_id=trace_id,
                correlation_id=scan_id,
            )
            await broker.publish(topic, json.dumps(out.model_dump()).encode("utf-8"))
            dispatched_count += 1

        self.session_state_manager.update_context(
            scan_id,
            {
                "stage": 6,
                "dispatched_tasks": dispatched_count,
                "pending_tasks": len(tasks),
            },
        )
        logger.info(f"[啟動] [Stage 6/7] Dispatched {dispatched_count} tasks")
        return dispatched_count

    async def stage_7_monitor_execution(
        self, scan_id: str, payload: ScanCompletedPayload, dispatched_count: int
    ) -> None:
        """階段7: 執行狀態監控 (Execution Status Monitoring)

        Args:
            scan_id: 掃描 ID
            payload: 掃描完成載荷
            dispatched_count: 已分發的任務數量
        """
        logger.info(f"[監控] [Stage 7/7] Monitoring execution for {scan_id}")

        self.session_state_manager.update_context(
            scan_id,
            {
                "stage": 7,
                "status": "monitoring",
                "scan_duration_seconds": payload.summary.scan_duration_seconds,
            },
        )
        self.session_state_manager.update_session_status(
            scan_id,
            "analysis_completed",
            {
                "tasks_dispatched": dispatched_count,
                "monitoring_active": True,
            },
        )

        logger.info(f"[已] [Stage 7/7] All stages completed for {scan_id}")

    async def process(
        self, payload: ScanCompletedPayload, broker: AbstractBroker, trace_id: str
    ) -> None:
        """執行完整的七階段處理流程

        Args:
            payload: 掃描完成載荷
            broker: 訊息代理
            trace_id: 追蹤 ID
        """
        scan_id = payload.scan_id

        # 階段1: 資料接收與預處理
        await self.stage_1_ingest_data(payload)

        # 階段2: 初步攻擊面分析
        await self.stage_2_analyze_surface(payload)

        # 階段3: 測試策略生成
        base_strategy = await self.stage_3_generate_strategy(scan_id)

        # 階段4: 動態策略調整
        adjusted_strategy = await self.stage_4_adjust_strategy(
            scan_id, base_strategy, payload
        )

        # 階段5: 任務生成
        tasks = await self.stage_5_generate_tasks(scan_id, adjusted_strategy, payload)

        # 階段6: 任務佇列管理與分發
        dispatched_count = await self.stage_6_dispatch_tasks(
            scan_id, tasks, broker, trace_id
        )

        # 階段7: 執行狀態監控
        await self.stage_7_monitor_execution(scan_id, payload, dispatched_count)
