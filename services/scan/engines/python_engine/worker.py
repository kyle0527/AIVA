"""
重構的 Scan Worker
使用 ScanOrchestrator 進行統一的掃描編排
"""


import json

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    ScanCompletedPayload,
    ScanStartPayload,
)
from services.aiva_common.utils import get_logger, new_id

from .scan_orchestrator import ScanOrchestrator

logger = get_logger(__name__)


async def run() -> None:
    """
    Worker 主函數，訂閱掃描任務並使用 ScanOrchestrator 執行
    """
    broker = await get_broker()
    async for mqmsg in broker.subscribe(Topic.TASK_SCAN_START):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            req = ScanStartPayload(**msg.payload)

            # 使用 ScanOrchestrator 執行掃描
            payload = await _perform_scan_with_orchestrator(req, msg.header.trace_id)

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
            logger.info(f"Scan completed successfully: {req.scan_id}")
        except Exception as exc:  # pragma: no cover
            logger.exception("Scan failed: %s", exc)


async def _perform_scan_with_orchestrator(
    req: ScanStartPayload,
    trace_id: str
) -> ScanCompletedPayload:
    """
    使用 ScanOrchestrator 執行掃描

    Args:
        req: 掃描請求
        trace_id: 追蹤 ID

    Returns:
        掃描完成的 Payload
    """
    # 創建 ScanOrchestrator 實例（不需要傳遞參數）
    orchestrator = ScanOrchestrator()

    # 執行掃描並直接獲取 ScanCompletedPayload
    payload = await orchestrator.execute_scan(req)

    logger.info(
        f"Scan orchestration completed: {req.scan_id}"
    )

    return payload
