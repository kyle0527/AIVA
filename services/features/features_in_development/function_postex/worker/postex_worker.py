import asyncio
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, PostExTaskPayload
from services.aiva_common.enums import Topic
from services.aiva_common.utils import get_logger
from services.features.function_postex.detector.postex_detector import PostExDetector

logger = get_logger(__name__)
detector = PostExDetector()

async def run() -> None:
    broker = await get_broker()
    async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_POSTEX):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        task = PostExTaskPayload(**msg.payload)
        await _process_task(task, broker)

async def _process_task(task: PostExTaskPayload, broker=None) -> None:
    try:
        findings = detector.analyze(task.test_type, task.target, task.task_id, task.scan_id, task.safe_mode, task.authorization_token)
        for f in findings:
            await broker.publish(Topic.FINDING_DETECTED, f.model_dump_json().encode("utf-8"), correlation_id=task.task_id)
        await broker.publish(Topic.STATUS_TASK_UPDATE, {
            "task_id": task.task_id, "scan_id": task.scan_id, "status":"COMPLETED", "worker_id":"postex_worker",
            "details": {"findings_count": len(findings)}
        })
        logger.info("postex_completed", extra={"task_id": task.task_id, "count": len(findings)})
    except Exception as exc:
        logger.exception("postex_failed", extra={"task_id": task.task_id})
        if broker:
            await broker.publish(Topic.STATUS_TASK_UPDATE, {
                "task_id": task.task_id, "scan_id": task.scan_id, "status":"FAILED", "error": str(exc)
            })
