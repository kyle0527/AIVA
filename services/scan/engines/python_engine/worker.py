"""
重構的 Scan Worker
使用 ScanOrchestrator 進行統一的掃描編排
支持 Phase0/Phase1 兩階段掃描流程
"""


import json

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    Phase0CompletedPayload,
    Phase0StartPayload,
    Phase1CompletedPayload,
    Phase1StartPayload,
    ScanCompletedPayload,
    ScanStartPayload,
)
from services.aiva_common.utils import get_logger, new_id

from .scan_orchestrator import ScanOrchestrator

logger = get_logger(__name__)


async def run() -> None:
    """
    Worker 主函數，訂閱掃描任務並使用 ScanOrchestrator 執行
    支持三種掃描模式：
    1. 標準掃描 (TASK_SCAN_START)
    2. Phase0 快速偵察 (TASK_SCAN_PHASE0)
    3. Phase1 深度掃描 (TASK_SCAN_PHASE1)
    """
    broker = await get_broker()

    # 創建 orchestrator 實例（共享）
    orchestrator = ScanOrchestrator()

    # 訂閱三種掃描任務
    logger.info("Worker started, subscribing to scan topics...")

    async for mqmsg in broker.subscribe(
        Topic.TASK_SCAN_START,
        Topic.TASK_SCAN_PHASE0,
        Topic.TASK_SCAN_PHASE1,
    ):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)

            # 根據 topic 分發到不同的處理函數
            if msg.topic == Topic.TASK_SCAN_PHASE0:
                await _handle_phase0(orchestrator, msg, broker)
            elif msg.topic == Topic.TASK_SCAN_PHASE1:
                await _handle_phase1(orchestrator, msg, broker)
            else:  # TASK_SCAN_START
                await _handle_standard_scan(orchestrator, msg, broker)

        except Exception as exc:  # pragma: no cover
            logger.exception("Scan task failed: %s", exc)


async def _handle_standard_scan(
    orchestrator: ScanOrchestrator,
    msg: AivaMessage,
    broker,
) -> None:
    """處理標準掃描任務"""
    req = ScanStartPayload(**msg.payload)
    logger.info(f"Processing standard scan: {req.scan_id}")

    payload = await orchestrator.execute_scan(req)

    out = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=msg.header.trace_id,
            correlation_id=req.scan_id,
            source_module=ModuleName.SCAN,
        ),
        topic=Topic.RESULTS_SCAN_COMPLETED,
        payload=payload.model_dump(),
    )
    await broker.publish(
        Topic.RESULTS_SCAN_COMPLETED,
        json.dumps(out.model_dump()).encode("utf-8"),
    )
    logger.info(f"Standard scan completed: {req.scan_id}")


async def _handle_phase0(
    orchestrator: ScanOrchestrator,
    msg: AivaMessage,
    broker,
) -> None:
    """處理 Phase0 快速偵察任務"""
    req = Phase0StartPayload(**msg.payload)
    logger.info(f"Processing Phase0 scan: {req.scan_id}")

    payload = await orchestrator.execute_phase0(req)

    out = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=msg.header.trace_id,
            correlation_id=req.scan_id,
            source_module=ModuleName.SCAN,
        ),
        topic=Topic.RESULTS_SCAN_PHASE0_COMPLETED,
        payload=payload.model_dump(),
    )
    await broker.publish(
        Topic.RESULTS_SCAN_PHASE0_COMPLETED,
        json.dumps(out.model_dump()).encode("utf-8"),
    )
    logger.info(
        f"Phase0 completed: {req.scan_id}, "
        f"technologies: {len(payload.discovered_technologies)}, "
        f"sensitive: {len(payload.sensitive_data_found)}"
    )


async def _handle_phase1(
    orchestrator: ScanOrchestrator,
    msg: AivaMessage,
    broker,
) -> None:
    """處理 Phase1 深度掃描任務"""
    req = Phase1StartPayload(**msg.payload)
    logger.info(f"Processing Phase1 scan: {req.scan_id}")

    payload = await orchestrator.execute_phase1(req)

    out = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=msg.header.trace_id,
            correlation_id=req.scan_id,
            source_module=ModuleName.SCAN,
        ),
        topic=Topic.RESULTS_SCAN_COMPLETED,  # Phase1 完成後發送最終結果
        payload=payload.model_dump(),
    )
    await broker.publish(
        Topic.RESULTS_SCAN_COMPLETED,
        json.dumps(out.model_dump()).encode("utf-8"),
    )
    logger.info(
        f"Phase1 completed: {req.scan_id}, "
        f"assets: {len(payload.complete_asset_list)}, "
        f"engines: {payload.engines_used}"
    )


# 保留向後兼容的函數
async def _perform_scan_with_orchestrator(
    req: ScanStartPayload,
    _trace_id: str
) -> ScanCompletedPayload:
    """
    使用 ScanOrchestrator 執行掃描(向後兼容)

    Args:
        req: 掃描請求
        _trace_id: 追蹤 ID (未使用)

    Returns:
        掃描完成的 Payload
    """
    orchestrator = ScanOrchestrator()
    payload = await orchestrator.execute_scan(req)
    logger.info(f"Scan orchestration completed: {req.scan_id}")
    return payload
