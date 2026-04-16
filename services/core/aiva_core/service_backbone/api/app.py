"""AIVA Core API - 程序與 AI 的溝通接口

職責:
1. FastAPI 應用程序 - 程序與 AI 溝通的接口（可插拔式設計）
2. 持有 CoreServiceCoordinator 作為狀態管理器
3. 持有 EnhancedDecisionAgent 作為認知核心（AI 決策引擎）
4. 提供 RESTful API 端點（接收 main.py 轉發的請求）
5. 啟動內部閉環和外部學習後台任務
6. 監聽 MQ 結果並觸發 AI 決策

架構層次（三層）:
    main.py                         ← 整個程序對外接收信息
        ↓ 轉發
    app.py (本模塊)                  ← 程序與 AI 溝通接口
        ↓ 持有
    ┌─ CoreServiceCoordinator       ← 狀態管理器和服務工廠
    │       ↓ 管理
    │   各功能服務 (Decision, Planning, Execution...)
    │
    └─ EnhancedDecisionAgent        ← 認知核心（AI 決策引擎）
            ↓ 整合
        雙CLI架構 + embedded_knowledge

完整掃描流程（雙路分離）:
    0. 外部 → main.py → 轉發給 app.py
    1. POST /scan 接收 main.py 轉發的請求
    2. 雙路處理（分離執行）:
       ├─ 路徑1: 整合模組存儲（實時記錄所有數據）
       └─ 路徑2: EnhancedDecisionAgent AI 分析 → 下令執行
    3. 發送 Phase0 命令到 MQ
    4. 掃描 Workers 執行並返回結果到 MQ
    5. process_phase0_results() 監聽結果
    6. 調用 ScanResultProcessor (含 AI 決策)
    7. AI 決定是否需要 Phase1 深度掃描
    8. 發送 Phase1 命令或生成報告

學習系統（異步獨立）:
    - 任務結束後，從整合模組讀取完整數據（訓練+實際，統一格式）
    - 本次記錄與歷史記錄比對和評估
    - 學習目標：如何調整參數才能讓響應變好
    - 參數優化：掃描參數、Payload 調整、檢測閾值等
    - 不在任務執行期間介入（安全考慮，避免修改運行中的代碼）
    - 離線優化模型和知識庫

[2026-01-20] 明確學習目標：參數優化以改善響應品質
[2026-01-20] 明確雙路分離架構：存儲與規劃獨立，學習系統異步比對評估歷史數據
[2026-01-19] 新增 POST /scan 端點，整合認知核心 AI 決策

[2026-01-20] 明確雙路分離架構：存儲與規劃獨立，學習系統異步運行
[2026-01-19] 新增 POST /scan 端點，整合認知核心 AI 決策
"""

import asyncio
from collections import Counter
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, HTTPException

# 路徑自動處理（支援直接運行和透過 __main__.py 運行）
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[5]  # AIVA-git/ (app.py -> api -> service_backbone -> aiva_core -> core -> services -> AIVA-git)
_SERVICES_DIR = _PROJECT_ROOT / "services"
_CORE_DIR = _SERVICES_DIR / "core"

# 智能路徑添加：僅在需要時添加
for p in [str(_CORE_DIR), str(_SERVICES_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:

# 整合模块数据管理
from services.integration.simple_data_manager import get_data_manager
from pydantic import BaseModel, HttpUrl
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from aiva_common.config import get_settings
from aiva_common.schemas import AivaMessage, ScanCompletedPayload, Phase0CompletedPayload
from aiva_common.schemas.commands import AICommand, AICommandResult, CommandType, CommandStatus, CommandPriority
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

# ✅ 引入 Rich UI 組件
from services.core.ui import (
    console,
    init_console,
    show_banner,
    show_panel,
)

# ✅ 引入 SSE 路由
try:
    from services.core.aiva_core.service_backbone.api.sse import router as sse_router
    SSE_AVAILABLE = True
except ImportError:
    logger = get_logger(__name__)  # 提前定義 logger
    logger.warning("⚠️  SSE 模組未安裝（需要 sse-starlette），Dashboard 即時功能將不可用")
    SSE_AVAILABLE = False
    sse_router = None

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
    title="AIVA Core Engine - 智慧分析與協調中心",
    description="核心分析引擎：攻擊面分析、策略生成、任務協調 (系統唯一入口)",
    version="3.0.0",
)
logger = get_logger(__name__)

# ✅ 註冊 SSE 路由（如果可用）
if SSE_AVAILABLE and sse_router:
    app.include_router(sse_router)
    logger.info("✅ SSE 路由已註冊: /api/logs/stream, /api/status/stream")

# ✅ 全局協調器實例（狀態管理器，非主線程）
coordinator: AIVACoreServiceCoordinator | None = None

# ✅ 全局認知核心實例（AI 決策引擎）
decision_agent: EnhancedDecisionAgent | None = None

# ✅ 全局指揮官實例（任務指揮中心）
commander: Any = None

# ✅ 全局消息代理實例（RabbitMQ）
broker: Any = None

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
    """啟動核心引擎服務 - 系統唯一啟動點（修正版）

    啟動流程（修正指揮鏈）:
    1. 初始化 CoreServiceCoordinator（狀態管理器）
    2. 初始化 EnhancedDecisionAgent（認知核心 - 大腦）
    3. 初始化 RabbitMQ Broker（消息隊列）
    4. 初始化執行能力（UnifiedExecutor, MultiEngineCoordinator - 四肢）
    5. 初始化 CommanderCoordinator（指揮官 - 神經中樞）
    6. 連接大腦、四肢到指揮官（修復指揮鏈斷裂）
    7. 啟動內部閉環更新（後台任務）
    8. 啟動執行狀態監控（後台任務）
    """
    global coordinator, decision_agent, commander, broker

    logger.info("🚀 [啟動] AIVA Core Engine starting up...")

    # ✅ Step 1: 初始化協調器（作為狀態管理器，非主線程）
    coordinator = AIVACoreServiceCoordinator()
    await coordinator.start()
    logger.info("✅ [啟動] CoreServiceCoordinator initialized (state manager mode)")

    # ✅ Step 2: 初始化認知核心（AI 決策引擎 - 大腦）
    decision_agent = EnhancedDecisionAgent()
    logger.info("✅ [啟動] EnhancedDecisionAgent initialized (Cognitive Core - Brain)")

    # ✅ Step 3: 初始化 RabbitMQ Broker（消息隊列）- 可選，5秒超時
    logger.info("🔧 [啟動] Initializing RabbitMQ Broker (optional, 5s timeout)...")
    import asyncio
    import os
    broker = None  # 預設為 None，沒有 RabbitMQ 也能運行

    try:
        # 先檢查 RabbitMQ 端口是否可達
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 5672))
        sock.close()

        if result != 0:
            logger.warning("⚠️  [啟動] RabbitMQ 端口 5672 無法連接，跳過消息隊列初始化")
            logger.info("📝 [啟動] 系統將在本地直接執行模式下運行（不需要 RabbitMQ）")
        else:
            from aiva_common.messaging import RabbitBroker

            # 設置環境變量（如果未設置）
            if 'RABBITMQ_URL' not in os.environ:
                os.environ['RABBITMQ_URL'] = 'amqp://localhost:5672/'

            broker = RabbitBroker()
            await asyncio.wait_for(broker.connect(), timeout=5.0)
            logger.info(f"✅ [啟動] RabbitMQ Broker connected: {os.environ['RABBITMQ_URL']}")
    except asyncio.TimeoutError:
        logger.warning("⚠️  [啟動] RabbitMQ 連接超時（5秒），跳過")
        broker = None
    except Exception as e:
        logger.warning(f"⚠️  [啟動] RabbitMQ 初始化失敗: {e}")
        broker = None

    if broker is None:
        logger.info("📝 [啟動] 系統將使用本地直接執行模式（無需消息隊列）")

    # ✅ Step 4: 初始化執行能力（四肢） - 核心修復！
    logger.info("🔧 [啟動] Initializing execution capabilities (Limbs)...")
    try:
        from services.core.aiva_core.task_planning.unified_executor import UnifiedAttackExecutor
        unified_executor = UnifiedAttackExecutor()
        logger.info("✅ [啟動] UnifiedAttackExecutor initialized")
    except Exception as e:
        logger.warning(f"⚠️  [啟動] UnifiedAttackExecutor failed: {e}, using None")
        unified_executor = None

    try:
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        multilang_coordinator = MultiEngineCoordinator()
        logger.info("✅ [啟動] MultiEngineCoordinator initialized")
    except Exception as e:
        logger.warning(f"⚠️  [啟動] MultiEngineCoordinator failed: {e}, using None")
        multilang_coordinator = None

    # ✅ Step 5: 初始化指揮官（神經中樞） - 核心修復！
    logger.info("🔧 [啟動] Initializing CommanderCoordinator (Command Center)...")
    try:
        from services.core.aiva_core.task_planning.commander import CommanderCoordinator

        # ✅ 依賴注入：將大腦、四肢、狀態管理器連接到指揮官
        commander = CommanderCoordinator(
            data_directory=_CORE_DIR / "data" / "commander",
            learning_enabled=True,
            unified_executor=unified_executor,              # ✅ 注入執行器
            multilang_coordinator=multilang_coordinator,    # ✅ 注入多語言協調器
            internal_loop_connector=decision_agent.internal_connector if decision_agent else None,  # ✅ 注入內部閉環
            session_state_manager=session_state_manager,    # ✅ 注入狀態管理器（進度追蹤）
        )

        # Commander 已經初始化完成，直接可用（不需要註冊到 coordinator）
        logger.info("✅ [啟動] CommanderCoordinator ready - Command chain is complete!")
    except Exception as e:
        logger.error(f"❌ [啟動] CommanderCoordinator initialization failed: {e}")
        commander = None

    # ✅ Step 3: 啟動內部閉環（Internal Loop）
    # 內部閉環已在 EnhancedDecisionAgent 中初始化 (self.internal_connector)
    if decision_agent.internal_connector is not None:
        try:
            # 執行一次同步以確保 RAG 知識庫是最新的
            logger.info("🔄 [啟動] Syncing Internal Loop RAG knowledge base...")
            # ✅ 使用正確的方法名稱: sync_capabilities_to_rag
            sync_result = await decision_agent.internal_connector.sync_capabilities_to_rag(
                force_refresh=False  # 啟動時不強制刷新
            )

            if sync_result.success:
                logger.info(
                    f"✅ [啟動] Internal Loop synced successfully - "
                    f"Capabilities found: {sync_result.capabilities_found}, "
                    f"Documents added: {sync_result.documents_added}"
                )
            else:
                logger.warning(f"⚠️  [啟動] Internal Loop sync failed: {sync_result.error}")

        except Exception as e:
            logger.warning(f"⚠️  [啟動] Internal Loop sync failed: {e}")
    else:
        logger.info("⚠️  [啟動] Internal Loop connector not available (fallback mode)")

    # ⚠️ Step 4: 外部學習（External Loop） - 延後啟用
    # 外部閉環已在 EnhancedDecisionAgent 中初始化 (self.external_connector)
    # 但學習循環需要執行結果數據，暫時不啟動背景任務
    if decision_agent.external_connector is not None:
        logger.info("✅ [啟動] External Loop connector available (on-demand mode)")
    else:
        logger.info("⚠️  [啟動] External Loop connector not available")

    # ✅ Step 5: 啟動執行狀態監控（唯一的背景任務）
    logger.info("[統計] Initializing analysis components...")

    # v2.0: 移除 MQ 監聽循環，改用 AICommand 直接執行
    # 只保留執行狀態監控
    _background_tasks.append(asyncio.create_task(
        monitor_execution_status(),
        name="execution_monitor"
    ))

    logger.info("✅ [啟動] Background monitor started (v2.0 AICommand mode)")
    logger.info("🎉 [啟動] AIVA Core Engine ready to accept requests!")


@app.on_event("shutdown")
async def shutdown() -> None:
    """關閉核心引擎服務"""
    global coordinator, _background_tasks

    logger.info("🛑 [關閉] AIVA Core Engine shutting down...")

    # 取消所有背景任務
    for task in _background_tasks:
        if not task.done():
            task.cancel()

    # 等待任務完成
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)
        logger.info("✅ [關閉] Background tasks cancelled")

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
async def get_scan_status(scan_id: str) -> dict[str, Any]:
    """獲取掃描狀態

    Args:
        scan_id: 掃描任務 ID

    Returns:
        掃描狀態資訊（含進度、階段、發現數量等）
    """
    status = session_state_manager.get_session_status(scan_id)
    if not status or status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    return status


@app.get("/api/scans")
async def list_scans(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> dict[str, Any]:
    """查詢歷史掃描列表

    Args:
        status: 篩選狀態（completed, running, failed 等）
        limit: 返回數量限制
        offset: 分頁偏移

    Returns:
        {
            "total": 總數,
            "scans": [掃描列表]
        }
    """
    try:
        # 從 SessionStateManager 取得所有掃描記錄
        # NOTE: SessionStateManager 需要實作 list_all_sessions() 方法
        # all_scans = session_state_manager.list_all_sessions()
        # 暫時返回空列表
        all_scans = []

        # 篩選狀態
        if status:
            all_scans = [s for s in all_scans if s.get("status") == status]

        # 分頁
        total = len(all_scans)
        scans = all_scans[offset:offset+limit]

        return {
            "total": total,
            "scans": scans,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"❌ [API] Failed to list scans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scans/{scan_id}/results")
async def get_scan_results(scan_id: str) -> dict[str, Any]:
    """取得完整掃描結果（含漏洞詳情）

    Args:
        scan_id: 掃描任務 ID

    Returns:
        完整掃描結果（含 vulnerabilities, assets, technologies 等）
    """
    try:
        # 從整合模組取得完整結果
        # NOTE: SimpleDataManager 需要實作 get_task_result(scan_id) 方法
        # data_manager = get_data_manager()
        # results = data_manager.get_task_result(scan_id)
        # 暫時從 SessionStateManager 取得
        status = session_state_manager.get_session_status(scan_id)
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
        results = status

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [API] Failed to get results for {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scans/{scan_id}/stop")
async def stop_scan(scan_id: str) -> dict[str, str]:
    """停止執行中的掃描

    Args:
        scan_id: 掃描任務 ID

    Returns:
        操作結果訊息
    """
    try:
        # 更新狀態為 cancelled
        session_state_manager.update_session_status(
            scan_id,
            "cancelled",
            {"cancelled_at": datetime.now().isoformat()}
        )

        # 如果有 coordinator，通知停止
        if coordinator:
            # NOTE: CommanderCoordinator 需要實作 cancel_scan(scan_id) 方法
            logger.info(f"⏹️ [API] Scan {scan_id} stop requested via coordinator")

        logger.info(f"⏹️ [API] Scan {scan_id} stopped")
        return {
            "scan_id": scan_id,
            "status": "stopped",
            "message": "Scan stop requested"
        }

    except Exception as e:
        logger.error(f"❌ [API] Failed to stop scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/scans/{scan_id}")
async def delete_scan(scan_id: str) -> dict[str, str]:
    """刪除掃描記錄

    Args:
        scan_id: 掃描任務 ID

    Returns:
        操作結果訊息
    """
    try:
        # NOTE: 刪除功能需要實作：
        #   1. SimpleDataManager.delete_task(scan_id)
        #   2. SessionStateManager.delete_session(scan_id)

        logger.info(f"🗑️ [API] Scan {scan_id} delete requested (not implemented yet)")
        return {
            "scan_id": scan_id,
            "status": "pending",
            "message": "Delete functionality not yet implemented"
        }

    except Exception as e:
        logger.error(f"❌ [API] Failed to delete scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan", response_model=ScanResponse)
async def start_scan(request: ScanRequest) -> ScanResponse:
    """啟動掃描 - AI 自主運作模式

    數據流: main.py → app.py (本函數) → Commander → AI 決策 → 執行器

    流程（符合手冊規範）:
    1. 接收 main.py 轉發的目標網址
    2. 委託給 Commander（指揮官）
    3. Commander 詢問 DecisionAgent（AI 大腦）
    4. AI 決策後，Commander 指揮執行器（四肢）
    5. 這就是真正的「AI 自主運作」
    """
    # 生成 scan_id
    scan_id = f"scan_{uuid4().hex[:8]}"
    trace_id = f"trace_{uuid4().hex[:8]}"

    logger.info(f"🎯 [Scan Request] {scan_id} - Target: {request.target}")

    # ============ 模式選擇：Commander 自主模式 vs 舊模式 ============
    if commander:
        # ✅ 新模式：完整 AI 自主運作（符合手冊規範）
        logger.info(f"🎖️ [Commander Mode] {scan_id} - AI autonomous operation enabled")

        try:
            # 保存到整合模組
            data_manager = get_data_manager()
            data_manager.save_task_data(
                capability="two_phase_scan",
                task_id=scan_id,
                target=request.target,
                request_data={
                    "scan_type": request.scan_type,
                    "max_depth": request.max_depth,
                    "timeout": request.timeout
                },
                metadata={
                    "trace_id": trace_id,
                    "source": "external_request",
                    "mode": "commander_autonomous"
                }
            )
            logger.info(f"💾 [Storage] {scan_id} - Saved to integration module")

            # 構建 Commander 執行上下文
            from services.core.aiva_core.task_planning.commander.types import AITaskType

            command_context = {
                "user_input": f"Scan target {request.target} with {request.scan_type} mode",
                "target": request.target,
                "targets": [request.target],  # ✅ 修復：添加targets列表
                "broker": broker,  # ✅ 修復：添加broker實例
                "scan_id": scan_id,
                "trace_id": trace_id,
                "scan_type": request.scan_type,
                "max_depth": request.max_depth,
                "timeout": request.timeout,
            }

            # 🎖️ 委託給 Commander：讓 AI 接管一切
            # Commander 會：
            # 1. 詢問 DecisionAgent（AI 大腦）該怎麼做
            # 2. 根據 AI 決策選擇工具和策略
            # 3. 指揮 UnifiedExecutor（四肢）執行
            logger.info(f"🎖️ [Delegating] {scan_id} - Handing over to Commander...")

            # 任務完成回調：記錄結果或錯誤
            def task_done_callback(task: asyncio.Task) -> None:
                try:
                    result = task.result()
                    logger.info(f"✅ [Task Complete] {scan_id} - Result: success={result.get('success', False)}")
                    if result.get('vulnerabilities'):
                        logger.info(f"   🔍 Found {len(result['vulnerabilities'])} vulnerabilities")
                except asyncio.CancelledError:
                    logger.warning(f"⚠️ [Task Cancelled] {scan_id}")
                    session_state_manager.update_session_status(scan_id, "cancelled")
                except Exception as e:
                    logger.error(f"❌ [Task Failed] {scan_id} - Error: {e}")
                    session_state_manager.update_session_status(scan_id, "failed", {"error": str(e)})

            # 保存 task 引用以防止被垃圾回收
            scan_task = asyncio.create_task(
                commander.execute_command(
                    task_type=AITaskType.TWO_PHASE_SCAN,
                    context=command_context
                ),
                name=f"commander_scan_{scan_id}"
            )
            # 將 task 加入集合以防止過早回收
            if not hasattr(app.state, 'background_tasks'):
                app.state.background_tasks = set()
            app.state.background_tasks.add(scan_task)
            scan_task.add_done_callback(task_done_callback)
            scan_task.add_done_callback(app.state.background_tasks.discard)

            return ScanResponse(
                scan_id=scan_id,
                status="started",
                message="AI Autonomous Scan Initiated (Commander Mode)",
                target=request.target,
                estimated_time=600
            )

        except Exception as e:
            logger.error(f"❌ [Commander Error] {scan_id} - {e}")
            raise HTTPException(status_code=500, detail=f"Commander execution failed: {str(e)}")

    else:
        # ⚠️ 舊模式：降級處理（向後兼容）
        logger.warning(f"⚠️  [Legacy Mode] {scan_id} - Commander not available, using direct execution")

        if not decision_agent:
            raise HTTPException(status_code=503, detail="System not fully initialized")

        try:
            # ============ 第一路：整合模組存儲 ============
            data_manager = get_data_manager()
            data_manager.save_task_data(
                capability="phase0",  # 按能力分类
                task_id=scan_id,
                target=request.target,
                request_data={
                    "scan_type": request.scan_type,
                    "max_depth": request.max_depth,
                    "timeout": request.timeout
                },
                metadata={
                    "trace_id": trace_id,
                    "source": "external_request",
                    "mode": "legacy_direct"
                }
            )
            logger.info(f"💾 [存儲] {scan_id} - 已保存到整合模組")

            # ============ 第二路：AI 分析決策 ============
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

            # v2.0: 使用 AICommand 直接執行（取代 MQ）
            command = AICommand(
                command_id=f"{scan_id}_phase0",
                command_type=CommandType.SCAN_PHASE0,
                target_module="scan",
                payload={
                    "scan_id": scan_id,
                    "targets": [request.target],
                    "trace_id": trace_id,
                    "timeout": request.timeout,
                    "ai_decision": initial_decision.action,
                },
                priority=CommandPriority.NORMAL,  # ✅ 使用枚舉值
                timeout=request.timeout,
                # ✅ 添加必要的參數
                trace_id=trace_id,
                session_id=scan_id,
                parent_command_id=None,
                callback_url=None,
            )

            # 異步執行 Phase0（不阻塞響應）
            # ✅ 保存 task 以防止過早被垃圾回收
            task = asyncio.create_task(
                _execute_phase0_async(command, scan_id, request.target),
                name=f"phase0_{scan_id}"
            )
            _background_tasks.append(task)

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

            logger.info(f"✅ [Scan Started] {scan_id} - Phase0 task created (Legacy AICommand mode)")

            return ScanResponse(
                scan_id=scan_id,
                status="started",
                message=f"Scan initiated (Legacy Mode) with AI decision: {initial_decision.action}",
                target=request.target,
                estimated_time=600,  # Phase0 預估 10 分鐘
            )

        except Exception as e:
            logger.error(f"❌ [Scan Failed] {scan_id} - {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to start scan: {str(e)}")


# ==================== v2.0 AICommand 異步執行函數 ====================

async def _execute_phase0_async(command: AICommand, scan_id: str, target: str) -> None:
    """異步執行 Phase0 掃描

    v2.0 架構：使用 AICommand 直接執行，取代 MQ 監聽

    Args:
        command: Phase0 掃描命令
        scan_id: 掃描 ID
        target: 目標 URL
    """
    import subprocess
    import json
    from datetime import datetime

    logger.info(f"[Phase0] Starting async execution for {scan_id}")
    start_time = datetime.now()

    try:
        # 使用 CLI 執行器進行 Phase0 掃描
        # 這裡調用 scan_interface 的內部方法直接執行
        result = await scan_interface.execute_phase0_direct(
            scan_id=scan_id,
            targets=[target],
            timeout=command.timeout or 600,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # 構建 AICommandResult
        command_result = AICommandResult(
            command_id=command.command_id,
            status=CommandStatus.COMPLETED if result.get("success") else CommandStatus.FAILED,
            success=result.get("success", False),
            result=result,
            execution_time=execution_time,
            started_at=start_time,
            completed_at=datetime.now(),
            # ✅ 添加必要的錯誤參數
            error=str(result.get("error")) if not result.get("success") else None,
            error_code=None,
            error_details=None,
        )

        logger.info(
            f"[Phase0] Completed for {scan_id} - "
            f"Success: {command_result.success}, "
            f"Time: {execution_time:.1f}s"
        )

        # 處理 Phase0 結果並決策是否需要 Phase1
        await _process_phase0_result(scan_id, result)

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[Phase0] Failed for {scan_id}: {e}", exc_info=True)

        session_state_manager.update_session_status(
            scan_id,
            "phase0_failed",
            {"error": str(e), "error_type": type(e).__name__},
        )


async def _process_phase0_result(scan_id: str, result: dict) -> None:
    """處理 Phase0 結果並決策是否需要 Phase1

    Args:
        scan_id: 掃描 ID
        result: Phase0 執行結果
    """
    try:
        assets = result.get("assets", [])
        technologies = result.get("technologies", [])

        logger.info(
            f"[Phase0] Processing results for {scan_id} - "
            f"Assets: {len(assets)}, Technologies: {len(technologies)}"
        )

        # AI 決策是否需要 Phase1
        if decision_agent:
            context = DecisionContext()
            context.target_info = {
                "type": "phase0_result",
                "scan_id": scan_id,
                "assets_count": len(assets),
                "technologies": technologies,
            }

            decision = await decision_agent.make_enhanced_decision(
                context=context,
                use_embedded_knowledge=True
            )

            need_phase1 = decision.confidence > 0.5 and len(assets) > 0

            if need_phase1:
                logger.info(f"[Phase0] {scan_id} needs Phase1 scan")
                # 創建 Phase1 任務
                phase1_command = AICommand(
                    command_id=f"{scan_id}_phase1",
                    command_type=CommandType.SCAN_PHASE1,
                    target_module="scan",
                    payload={
                        "scan_id": scan_id,
                        "targets": [a.get("value", a) if isinstance(a, dict) else a for a in assets],
                        "phase0_result": result,
                        "selected_engines": decision.params.get("engines", ["nikto", "dirb"]),
                    },
                    priority=CommandPriority.NORMAL,  # ✅ 使用枚舉值
                    timeout=1800,
                    # ✅ 添加必要的參數
                    trace_id=f"trace_{scan_id}",
                    session_id=scan_id,
                    parent_command_id=f"{scan_id}_phase0",
                    callback_url=None,
                )
                # ✅ 保存 task 以防止過早被垃圾回收
                task = asyncio.create_task(
                    _execute_phase1_async(phase1_command, scan_id),
                    name=f"phase1_{scan_id}"
                )
                _background_tasks.append(task)
                session_state_manager.update_session_status(
                    scan_id, "phase1_started", {"engines": decision.params.get("engines", [])}
                )
            else:
                logger.info(f"[Phase0] {scan_id} completed, no Phase1 needed")
                session_state_manager.update_session_status(
                    scan_id, "completed", {"result": "phase0_sufficient"}
                )
        else:
            # 沒有 AI 決策代理時，預設完成
            session_state_manager.update_session_status(
                scan_id, "completed", {"result": "phase0_only"}
            )

    except Exception as e:
        logger.error(f"[Phase0] Error processing result for {scan_id}: {e}", exc_info=True)
        session_state_manager.update_session_status(
            scan_id, "failed", {"error": str(e)}
        )


async def _execute_phase1_async(command: AICommand, scan_id: str) -> None:
    """異步執行 Phase1 深度掃描

    Args:
        command: Phase1 掃描命令
        scan_id: 掃描 ID
    """
    from datetime import datetime

    logger.info(f"[Phase1] Starting async execution for {scan_id}")
    start_time = datetime.now()

    try:
        payload = command.payload or {}
        result = await scan_interface.execute_phase1_direct(
            scan_id=scan_id,
            targets=payload.get("targets", []),
            engines=payload.get("selected_engines", []),
            timeout=command.timeout or 1800,
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"[Phase1] Completed for {scan_id} - "
            f"Time: {execution_time:.1f}s"
        )

        # 更新狀態
        session_state_manager.update_session_status(
            scan_id, "completed", {"phase1_result": result}
        )

        # 回饋學習
        strategy_adjuster.learn_from_result({
            "scan_id": scan_id,
            "phase": "phase1",
            "result": result,
            "execution_time": execution_time,
        })

    except Exception as e:
        logger.error(f"[Phase1] Failed for {scan_id}: {e}", exc_info=True)
        session_state_manager.update_session_status(
            scan_id, "phase1_failed", {"error": str(e)}
        )


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


# ============================================================================
# 啟動配置
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description="AIVA AI 智能滲透測試系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 啟動服務模式（HTTP API）
  python app.py

  # 直接掃描目標（CLI 模式）
  python app.py --target http://localhost:3000

  # 指定端口啟動
  python app.py --port 9000
"""
    )

    # 服務模式參數
    parser.add_argument("--port", "-p", type=int, default=8000, help="服務端口 (預設: 8000)")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="綁定地址 (預設: 0.0.0.0)")
    parser.add_argument("--reload", "-r", action="store_true", help="開發模式（自動重載）")

    # CLI 掃描模式參數
    parser.add_argument("--target", "-t", type=str, help="直接掃描目標 URL（CLI 模式）")
    parser.add_argument("--scan-type", "-s", choices=["quick", "full", "passive"], default="full", help="掃描類型")
    parser.add_argument("--depth", "-d", type=int, default=3, help="掃描深度 (預設: 3)")

    args = parser.parse_args()

    # ========================================
    # CLI 模式：直接掃描目標
    # ========================================
    if args.target:
        import asyncio

        # 初始化 Rich UI
        init_console()

        # 顯示 AIVA Banner
        show_banner(
            title="AIVA 智能滲透測試系統",
            subtitle="AI-Powered Vulnerability Intelligence & Assessment",
            version="v4.4.1"
        )
        console.print("[cyan bold]🚀 模式: CLI 掃描模式[/cyan bold]\n")

        # 顯示掃描配置
        from rich.table import Table
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("項目", style="cyan", width=10)
        config_table.add_column("值", style="white")
        config_table.add_row("🎯 目標", f"[bold]{args.target}[/bold]")
        config_table.add_row("📋 類型", args.scan_type)
        config_table.add_row("📊 深度", str(args.depth))
        console.print(config_table)
        console.print()


        async def run_cli_scan():
            """執行 CLI 掃描"""
            from rich.progress import Progress, SpinnerColumn, TextColumn

            # 初始化系統（調用 startup）
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("正在初始化 AIVA 系統...", total=None)
                await startup()
                progress.update(task, description="✅ 系統初始化完成")

            console.print()

            # 創建掃描請求
            request = ScanRequest(
                target=args.target,
                scan_type=args.scan_type,
                max_depth=args.depth,
            )

            # 執行掃描
            console.print(f"[cyan]🔍 開始掃描 {args.target}...[/cyan]")
            console.print()

            try:
                result = await start_scan(request)
                console.print()
                console.print(
                    f"[green bold]✅ 掃描完成![/green bold]\n\n"
                    f"  [bold]Scan ID:[/bold] {result.scan_id}\n"
                    f"  [bold]狀態:[/bold] {result.status}\n" +
                    (f"  [bold]訊息:[/bold] {result.message}" if hasattr(result, 'message') else "")
                )
            except Exception as e:
                from services.core.ui.components import show_error
                show_error(f"掃描失敗: {str(e)}")
            finally:
                # 清理
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("正在清理資源...", total=None)
                    await shutdown()
                    progress.update(task, description="✅ 清理完成")


        asyncio.run(run_cli_scan())

    # ========================================
    # 服務模式：啟動 HTTP API
    # ========================================
    else:
        # 初始化 Rich UI
        init_console()

        # 顯示 AIVA Banner
        show_banner(
            title="AIVA 智能滲透測試系統",
            subtitle="AI-Powered Vulnerability Intelligence & Assessment",
            version="v4.4.1"
        )
        console.print("[cyan bold]🌐 模式: HTTP 服務模式[/cyan bold]\n")

        # 服務信息面板
        from rich.table import Table
        service_info = Table(show_header=False, box=None, padding=(0, 2))
        service_info.add_column("項目", style="cyan bold", width=12)
        service_info.add_column("值", style="white")
        service_info.add_row("🌐 服務地址", f"http://{args.host}:{args.port}")
        service_info.add_row("📚 API 文檔", f"http://{args.host}:{args.port}/docs")
        service_info.add_row("💚 健康檢查", f"http://localhost:{args.port}/health")
        console.print(service_info)
        console.print()

        # 使用說明
        show_panel(
            "[bold]使用方式:[/bold]\n\n"
            f"[cyan]→[/cyan] curl http://localhost:{args.port}/health\n"
            f"[cyan]→[/cyan] curl -X POST http://localhost:{args.port}/scan -d '...'\n\n"
            "[bold]或使用 CLI 模式:[/bold]\n\n"
            "[cyan]→[/cyan] python app.py --target http://localhost:3000",
            style="info",
            title="快速開始"
        )
        console.print()


        # 直接傳入 app 對象（不用字串路徑，避免模組找不到的問題）
        uvicorn.run(
            app,  # 直接用 app 對象
            host=args.host,
            port=args.port,
            # reload 模式需要字串路徑，但會有 Windows 子進程問題，所以禁用
            log_level="info"
        )


