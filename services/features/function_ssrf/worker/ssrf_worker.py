import os, json
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, MessageHeader, FunctionTaskPayload
from services.aiva_common.enums import Topic, ModuleName
from services.aiva_common.utils import get_logger, new_id
from ..config.ssrf_config import SsrfConfig
from ..detector.ssrf_detector import SSRFDetector

logger = get_logger(__name__)

def _topic(env_name: str, default: str):
    return os.getenv(env_name, default)

async def run():
    broker = await get_broker()
    task_topic = _topic("SSRF_TOPIC_TASK", getattr(Topic,"TASK_FUNCTION_SSRF","TASK_FUNCTION_SSRF"))
    finding_topic = _topic("TOPIC_FINDING", getattr(Topic,"FINDING_DETECTED","FINDING_DETECTED"))
    status_topic = _topic("TOPIC_STATUS", getattr(Topic,"TASK_STATUS","TASK_STATUS"))

    config = SsrfConfig(
        enable_internal_scan = os.getenv("SSRF_ENABLE_INTERNAL","true").lower()=="true",
        enable_cloud_metadata= os.getenv("SSRF_ENABLE_METADATA","true").lower()=="true",
        enable_file_protocol = os.getenv("SSRF_ENABLE_FILE","false").lower()=="true",
        allow_active_network = os.getenv("SSRF_ALLOW_ACTIVE","false").lower()=="true",
        request_timeout      = float(os.getenv("SSRF_TIMEOUT","8.0")),
        max_redirects        = int(os.getenv("SSRF_MAX_REDIRECTS","3")),
        safe_mode            = os.getenv("SSRF_SAFE_MODE","true").lower()=="true"
    )
    detector = SSRFDetector(config)

    async for mqmsg in broker.subscribe(task_topic):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            task = FunctionTaskPayload(**msg.payload)
            trace_id = msg.header.trace_id if msg.header else new_id("trace")
            target_url = str(task.target.url)

            # IN_PROGRESS
            await broker.publish(status_topic, json.dumps(AivaMessage(
                header=MessageHeader(message_id=new_id("msg"), trace_id=trace_id, source_module=getattr(ModuleName,"FUNC_SSRF","FUNC_SSRF")),
                topic=status_topic, payload={"status":"IN_PROGRESS","target":target_url}
            ).model_dump()).encode("utf-8"))

            findings = await detector.analyze(target_url)
            for f in findings:
                f.task_id = task.task_id
                f.scan_id = task.scan_id
                out = AivaMessage(
                    header=MessageHeader(message_id=new_id("msg"), trace_id=trace_id, source_module=getattr(ModuleName,"FUNC_SSRF","FUNC_SSRF")),
                    topic=finding_topic, payload=f.model_dump()
                )
                await broker.publish(finding_topic, json.dumps(out.model_dump()).encode("utf-8"))

            # COMPLETED
            await broker.publish(status_topic, json.dumps(AivaMessage(
                header=MessageHeader(message_id=new_id("msg"), trace_id=trace_id, source_module=getattr(ModuleName,"FUNC_SSRF","FUNC_SSRF")),
                topic=status_topic, payload={"status":"COMPLETED","target":target_url,"findings_count":len(findings)}
            ).model_dump()).encode("utf-8"))
        except Exception as e:
            logger.exception("SSRF task failed")
            await broker.publish(status_topic, json.dumps(AivaMessage(
                header=MessageHeader(message_id=new_id("msg"), trace_id=new_id("trace"), source_module=getattr(ModuleName,"FUNC_SSRF","FUNC_SSRF")),
                topic=status_topic, payload={"status":"ERROR","error":str(e)}
            ).model_dump()).encode("utf-8"))
