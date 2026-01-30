"""AIVA Core API - 系統唯一對外入口點

職責:
1. FastAPI 應用程序 - 系統唯一對外接口（已從 AI 系統移出）
2. 接收所有外部請求並進行雙路處理：
   - 第一路：整合模組儲存（數據庫持久化）
   - 第二路：AI 分析處理（aiva_core 系統）
3. 調用 AI 決策引擎（EnhancedDecisionAgent）
4. AI 決策後下令給掃描模組或功能模組
5. 提供 RESTful API 端點（POST /scan 等）

架構定位:
    外部請求
        ↓
    app.py (唯一入口) ← 不屬於 AI，是對外門戶
        ↓
    ┌──────────────┬──────────────┐
    ↓              ↓
 整合模組儲存   aiva_core (AI 系統)
    ↓              ↓
 數據庫      EnhancedDecisionAgent
                   ↓
              【AI 決策並下令】
                   ↓
            掃描/功能模組執行

統一處理流程（適用於所有13個步驟）:
    1. POST /scan 接收外部請求
    2. 雙路處理：儲存 + AI 分析
    3. AI 決策分析（EnhancedDecisionAgent）
    4. AI 下令給目標模組（掃描/功能）
    5. 模組執行並返回結果到 MQ
    6. ScanResultProcessor 處理結果
    7. ExternalLoopConnector 學習更新

[2026-01-20] app.py 已移出 AI 系統，作為核心模組唯一對外入口
[2026-01-19] 新增 POST /scan 端點，整合認知核心 AI 決策
"""

import asyncio
from collections import Counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from aiva_common.config import get_settings
from aiva_common.enums.modules import Topic
from aiva_common.mq import get_broker
from aiva_common.schemas import AivaMessage, ScanCompletedPayload, Phase0CompletedPayload
from aiva_common.utils import get_logger
# 模組整合: external_learning → cognitive_core/learning_system (2026-01-03)
from services.core.aiva_core.cognitive_core.learning_system.analysis.dynamic_strategy_adjustment import (
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

# ✅ 引入認知核心 - AI 決策引擎
from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import (
    EnhancedDecisionAgent,
    DecisionContext,
)

# ✅ 引入 CoreServiceCoordinator 作為狀態管理器
from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
    AIVACoreServiceCoordinator,
)

# ✅ 引入內部閉環和外部學習組件
# ⚠️ 暫時註釋：這些模組尚未實現
# from services.core.aiva_core.internal_exploration.connectors.update_self_awareness import (
#     periodic_update,
# )
# from services.core.aiva_core.external_learning.connectors.external_loop_connector import (
#     ExternalLoopConnector,
# )

# ==================== 請求/響應模型 ====================

class ScanRequest(BaseModel):
    """掃描請求"""
    target: str
    scan_type: str = "comprehensive"  # comprehensive, quick, deep
    max_depth: int = 3
    timeout: int = 1800

class ScanResponse(BaseModel):
    """掃描響應"""
    scan_id: str
    status: str
    message: str
    target: str
    estimated_time: int  # 預估時間（秒）

# ==================== FastAPI 應用 ====================

app = FastAPI(
    title="AIVA System - 唯一對外入口",
    description="系統唯一對外接口：接收外部請求，雙路處理（儲存 + AI 決策），AI 下令執行模組",
    version="3.0.0",
)
logger = get_logger(__name__)

# ✅ 全局協調器實例（整合模組儲存管理）
coordinator: AIVACoreServiceCoordinator | None = None

# ✅ 全局認知核心實例（AI 決策引擎 - aiva_core 系統）
decision_agent: EnhancedDecisionAgent | None = None

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
    2. 初始化 EnhancedDecisionAgent（認知核心）
    3. 啟動內部閉環更新（後台任務）
    4. 啟動外部學習監聽器（後台任務）
    5. 啟動掃描結果處理（後台任務）
    6. 啟動功能結果處理（後台任務）
    7. 啟動執行狀態監控（後台任務）
    """
    global coordinator, decision_agent
    
    logger.info("🚀 [啟動] AIVA Core Engine starting up...")
    
    # ✅ Step 1: 初始化協調器（作為狀態管理器，非主線程）
    coordinator = AIVACoreServiceCoordinator()
    await coordinator.start()
    logger.info("✅ [啟動] CoreServiceCoordinator initialized (state manager mode)")
    
    # ✅ Step 2: 初始化認知核心（AI 決策引擎）
    decision_agent = EnhancedDecisionAgent()
    logger.info("✅ [啟動] EnhancedDecisionAgent initialized (AI decision engine)")
    
    # ⚠️ Step 2-3: 內部閉環和外部學習（暫時禁用 - 模組尚未實現）
    # _background_tasks.append(asyncio.create_task(
    #     periodic_update(),
    #     name="internal_loop_update"
    # ))
    # logger.info("✅ [啟動] Internal exploration loop started")
    # 
    # external_connector = ExternalLoopConnector()
    # _background_tasks.append(asyncio.create_task(
    #     external_connector.start_listening(),
    #     name="external_learning_loop"
    # ))
    # logger.info("✅ [啟動] External learning listener started")
    logger.info("⚠️  [啟動] Internal/External loops disabled (modules not implemented)")
    
    # ✅ Step 4-6: 啟動核心處理循環
    logger.info("[統計] Initializing analysis components...")
    logger.info("[循環] Starting message processing loops...")
    
    # Phase0 結果處理器 (優先於標準掃描)
    _background_tasks.append(asyncio.create_task(
        process_phase0_results(),
        name="phase0_results_processor"
    ))
    
    # 標準掃描結果處理器 (處理 Phase1 和傳統掃描)
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
    
    logger.info("✅ [啟動] All background tasks started (including Phase0 processor)")
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


@app.post("/scan", response_model=ScanResponse)
async def start_scan(request: ScanRequest) -> ScanResponse:
    """唯一對外入口 - 接收掃描請求並進行雙路處理
    
    統一處理流程（適用於所有請求）:
    1. 接收外部請求（目標網址）
    2. 第一路：整合模組儲存（數據庫持久化）
    3. 第二路：AI 分析處理
       - 調用 EnhancedDecisionAgent（aiva_core）進行決策
       - AI 決策後下令給掃描模組
    4. 返回 scan_id 供後續查詢
    
    注意：掃描不是直接觸發，而是由 AI 決策後下令執行
    """
    if not decision_agent:
        raise HTTPException(status_code=503, detail="Decision agent not initialized")
    
    # 生成 scan_id
    scan_id = f"scan_{uuid4().hex[:8]}"
    trace_id = f"trace_{uuid4().hex[:8]}"
    
    logger.info(f"🎯 [Scan Request] {scan_id} - Target: {request.target}")
    
    try:
        # 構建決策上下文
        context = DecisionContext()
        context.target_info = {
            "type": "web",
            "value": request.target,
            "id": scan_id,
            "scan_type": request.scan_type,
        }
        context.available_tools = ["nmap", "nikto", "sqlmap", "xsser", "dirb"]
        
        # 🧠 調用認知核心進行初步分析
        logger.info(f"🧠 [AI Analysis] {scan_id} - Analyzing target with cognitive core...")
        initial_decision = await decision_agent.make_enhanced_decision(
            context=context,
            use_embedded_knowledge=True
        )
        
        logger.info(
            f"🧠 [AI Decision] {scan_id} - "
            f"Action: {initial_decision.action}, "
            f"Confidence: {initial_decision.confidence:.0%}"
        )
        
        # 根據 AI 決策發送 Phase0 命令到 MQ
        broker = await get_broker()
        await scan_interface.send_phase0_command(
            broker=broker,
            scan_id=scan_id,
            targets=[request.target],
            trace_id=trace_id,
            timeout_seconds=request.timeout,
        )
        
        # 初始化會話狀態
        session_state_manager.update_session_status(
            session_id=scan_id,
            status="phase0_started",
            additional_data={
                "target": request.target,
                "scan_type": request.scan_type,
                "ai_decision": initial_decision.action,
                "confidence": initial_decision.confidence,
                "reasoning": initial_decision.reasoning,
            },
        )
        
        logger.info(f"✅ [Scan Started] {scan_id} - Phase0 command sent to MQ")
        
        return ScanResponse(
            scan_id=scan_id,
            status="started",
            message=f"Scan initiated with AI decision: {initial_decision.action}",
            target=request.target,
            estimated_time=600,  # Phase0 預估 10 分鐘
        )
        
    except Exception as e:
        logger.error(f"❌ [Scan Failed] {scan_id} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start scan: {str(e)}")


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


async def process_phase0_results() -> None:
    """處理 Phase0 快速偵察結果 - AI 決策與引擎選擇
    
    流程:
    1. 接收 Phase0 結果 (scan.phase0.completed)
    2. AI 分析決策是否需要 Phase1
    3. 如果需要 Phase1, 選擇適合的引擎
    4. 發送 Phase1 命令 (tasks.scan.phase1)
    """
    logger.info("[Phase0] Starting Phase0 results processor...")
    broker = await get_broker()

    aiterator = broker.subscribe(Topic.RESULTS_SCAN_PHASE0_COMPLETED)
    if hasattr(aiterator, "__await__"):
        aiterator = await aiterator  # type: ignore[misc]

    async for mqmsg in aiterator:  # type: ignore[misc]
        msg = AivaMessage.model_validate_json(mqmsg.body)
        payload = Phase0CompletedPayload(**msg.payload)
        scan_id = payload.scan_id

        try:
            # 根据 aiva_common/schemas/tasks.py 的实际字段结构
            assets_count = len(payload.assets) if payload.assets else 0
            fingerprints = payload.fingerprints.model_dump() if payload.fingerprints else {}
            tech_count = len(fingerprints.get("technologies", []))
            
            logger.info(
                f"[Phase0] Received results for {scan_id} - "
                f"Assets: {assets_count}, "
                f"Technologies: {tech_count}, "
                f"Status: {payload.status}"
            )

            # 處理 Phase0 結果並決策
            need_phase1, reason, selected_engines = await scan_result_processor.process_phase0(
                payload, broker, msg.header.trace_id
            )

            if not need_phase1:
                logger.info(
                    f"[Phase0] Scan {scan_id} does not need Phase1. Reason: {reason}"
                )
                # 輕量級分析流程：Phase0 結果已足夠，直接生成報告
                logger.info(f"[Phase0] Generating lightweight report for {scan_id}")
                continue

            logger.info(
                f"[Phase0] Scan {scan_id} needs Phase1 with engines: {selected_engines}. Reason: {reason}"
            )

            # 發送 Phase1 命令
            # 從 Phase0 payload 的 assets 中提取目標 URL (使用 value 字段)
            targets = [asset.value for asset in payload.assets if asset.value] if payload.assets else []
            
            # Fallback: 從會話上下文獲取
            if not targets:
                session_context = session_state_manager.get_session_context(scan_id)
                targets = session_context.get("targets", [])

            await scan_interface.send_phase1_command(
                broker=broker,
                scan_id=scan_id,
                targets=targets,
                trace_id=msg.header.trace_id,
                phase0_result=payload,
                selected_engines=selected_engines,
                max_depth=3,
                max_urls=1000,
            )

            logger.info(
                f"[Phase0] Phase1 command sent for {scan_id} with engines {selected_engines}"
            )

        except Exception as e:
            logger.error(
                f"[Phase0] Error processing Phase0 result for {scan_id}: {e}",
                exc_info=True,
            )
            session_state_manager.update_session_status(
                scan_id,
                "phase0_failed",
                {"error": str(e), "error_type": type(e).__name__},
            )


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

