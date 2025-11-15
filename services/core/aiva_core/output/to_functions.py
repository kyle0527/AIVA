from services.aiva_common.enums.modules import ModuleName, Topic
from services.aiva_common.schemas import AivaMessage, FunctionTaskPayload, MessageHeader
from services.aiva_common.utils import new_id


def to_function_message(
    topic: Topic,
    payload: FunctionTaskPayload,
    trace_id: str,
    correlation_id: str,
) -> AivaMessage:
    """Wrap function payload into AIVA message with proper headers."""
    return AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=trace_id,
            correlation_id=correlation_id,
            source_module=ModuleName.CORE,
        ),
        topic=topic,
        payload=payload.model_dump(),
    )
