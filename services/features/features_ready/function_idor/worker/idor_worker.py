import os, json
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, MessageHeader, FunctionTaskPayload
from services.aiva_common.enums import Topic, ModuleName
from services.aiva_common.utils import get_logger, new_id
from ..config.idor_config import IdorConfig
from ..detector.idor_detector import IDORDetector

logger = get_logger(__name__)

def _topic(env_name: str, default: str):
    return os.getenv(env_name, default)

async def run():
    broker = await get_broker()
    task_topic = _topic("IDOR_TOPIC_TASK", getattr(Topic,"TASK_FUNCTION_IDOR","TASK_FUNCTION_IDOR"))
    finding_topic = _topic("TOPIC_FINDING", getattr(Topic,"FINDING_DETECTED","FINDING_DETECTED"))
    status_topic = _topic("TOPIC_STATUS", getattr(Topic,"TASK_STATUS","TASK_STATUS"))

    cfg = IdorConfig(
        horizontal_enabled = os.getenv("IDOR_ENABLE_HORIZONTAL","true").lower()=="true",
        vertical_enabled   = os.getenv("IDOR_ENABLE_VERTICAL","true").lower()=="true",
        max_id_variations  = int(os.getenv("IDOR_MAX_VARIATIONS","5")),
        allow_active_network = os.getenv("IDOR_ALLOW_ACTIVE","false").lower()=="true",
        safe_mode          = os.getenv("IDOR_SAFE_MODE","true").lower()=="true",
        request_timeout    = float(os.getenv("IDOR_TIMEOUT","8.0"))
    )
    detector = IDORDetector(cfg)

    async for mqmsg in broker.subscribe(task_topic):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            task = FunctionTaskPayload(**msg.payload)
            trace_id = msg.header.trace_id if msg.header else new_id("trace")

            # IN_PROGRESS
            await broker.publish(status_topic, json.dumps(AivaMessage(
                header=MessageHeader(message_id=new_id("msg"), trace_id=trace_id, source_module=getattr(ModuleName,"FUNC_IDOR","FUNC_IDOR")),
                topic=status_topic, payload={"status":"IN_PROGRESS","target":str(task.target.url)}
            ).model_dump()).encode("utf-8"))

            findings = await detector.analyze(task)
            for f in findings:
                out = AivaMessage(
                    header=MessageHeader(message_id=new_id("msg"), trace_id=trace_id, source_module=getattr(ModuleName,"FUNC_IDOR","FUNC_IDOR")),
                    topic=finding_topic, payload=f.model_dump()
                )
                await broker.publish(finding_topic, json.dumps(out.model_dump()).encode("utf-8"))

            # COMPLETED
            await broker.publish(status_topic, json.dumps(AivaMessage(
                header=MessageHeader(message_id=new_id("msg"), trace_id=trace_id, source_module=getattr(ModuleName,"FUNC_IDOR","FUNC_IDOR")),
                topic=status_topic, payload={"status":"COMPLETED","findings_count":len(findings)}
            ).model_dump()).encode("utf-8"))
        except Exception as e:
            logger.exception("IDOR task failed")
            await broker.publish(status_topic, json.dumps(AivaMessage(
                header=MessageHeader(message_id=new_id("msg"), trace_id=new_id("trace"), source_module=getattr(ModuleName,"FUNC_IDOR","FUNC_IDOR")),
                topic=status_topic, payload={"status":"ERROR","error":str(e)}
            ).model_dump()).encode("utf-8"))
