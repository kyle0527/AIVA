import asyncio
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, FunctionTaskPayload
from services.aiva_common.enums import Topic
from services.aiva_common.utils import get_logger
from services.features.function_crypto.detector.crypto_detector import CryptoDetector

logger = get_logger(__name__)
detector = CryptoDetector()

async def run() -> None:
    broker = await get_broker()
    async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_CRYPTO):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        task = FunctionTaskPayload(**msg.payload)
        await _process_task(task, broker)

async def _process_task(task: FunctionTaskPayload, broker=None) -> None:
    try:
        target_content = None
        if isinstance(task.target.url, str):
            target_content = task.target.url
            try:
                if (target_content.startswith("/") or ":" in target_content) and "\n" not in target_content:
                    with open(target_content, "r", encoding="utf-8", errors="ignore") as f:
                        target_content = f.read()
            except Exception as e:
                logger.warning("File read failed; using raw string", extra={"error": str(e), "task_id": task.task_id})
        else:
            target_content = str(task.target.url)

        if not target_content:
            logger.error("No target content", extra={"task_id": task.task_id})
            return

        findings = detector.detect(target_content, task.task_id, task.scan_id)
        for f in findings:
            await broker.publish(Topic.FINDING_DETECTED, f.model_dump_json().encode("utf-8"), correlation_id=task.task_id)

        await broker.publish(Topic.STATUS_TASK_UPDATE, {
            "task_id": task.task_id, "scan_id": task.scan_id, "status":"COMPLETED", "worker_id":"crypto_worker",
            "details": {"findings_count": len(findings)}
        })
        logger.info("crypto_completed", extra={"task_id": task.task_id, "count": len(findings)})
    except Exception as exc:
        logger.exception("crypto_failed", extra={"task_id": task.task_id})
        if broker:
            await broker.publish(Topic.STATUS_TASK_UPDATE, {
                "task_id": task.task_id, "scan_id": task.scan_id, "status":"FAILED", "error": str(exc)
            })
