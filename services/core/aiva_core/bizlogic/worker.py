"""BizLogic Worker - 業務邏輯漏洞測試 Worker

監聽業務邏輯測試任務,執行測試並回報結果
"""

import json

from services.aiva_common.enums.modules import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, MessageHeader
from services.aiva_common.utils import get_logger, new_id

from .price_manipulation_tester import PriceManipulationTester
from .race_condition_tester import RaceConditionTester
from .workflow_bypass_tester import WorkflowBypassTester

logger = get_logger(__name__)


async def run() -> None:
    """啟動 BizLogic Worker

    監聽 tasks.function.bizlogic Topic
    """
    logger.info("Starting BizLogic Worker...")

    broker = await get_broker()

    async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_START):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)

            # 只處理 bizlogic 相關的任務
            if msg.payload.get("module") != "bizlogic":
                continue

            logger.info(f"Received bizlogic task: {msg.payload.get('test_type')}")

            # 執行測試
            findings = await _perform_test(msg.payload)

            # 發送結果
            result_msg = AivaMessage(
                header=MessageHeader(
                    message_id=new_id("msg"),
                    trace_id=msg.header.trace_id,
                    correlation_id=msg.payload.get("task_id"),
                    source_module=ModuleName.FUNCTION,
                ),
                topic=Topic.RESULTS_FUNCTION_COMPLETED,
                payload={
                    "task_id": msg.payload.get("task_id"),
                    "scan_id": msg.payload.get("scan_id"),
                    "module": "bizlogic",
                    "findings": [f.model_dump() for f in findings],
                    "status": "completed",
                },
            )

            await broker.publish(
                Topic.RESULTS_FUNCTION_COMPLETED,
                json.dumps(result_msg.model_dump()).encode("utf-8"),
            )

            logger.info(f"BizLogic test completed: {len(findings)} findings reported")

        except Exception as e:
            logger.exception(f"BizLogic worker error: {e}")


async def _perform_test(payload: dict) -> list:
    """執行業務邏輯測試

    Args:
        payload: 任務 Payload

    Returns:
        list: 發現的漏洞列表
    """
    test_type = payload.get("test_type", "price_manipulation")
    target_urls = payload.get("target_urls", {})
    task_id = payload.get("task_id", "task_unknown")
    scan_id = payload.get("scan_id", "scan_unknown")

    findings = []

    # 價格操縱測試
    if test_type == "price_manipulation":
        tester = PriceManipulationTester()
        try:
            product_id = payload.get("product_id")
            findings = await tester.run_all_tests(
                target_urls, task_id=task_id, scan_id=scan_id, product_id=product_id
            )
        finally:
            await tester.close()

    # 工作流程繞過測試
    elif test_type == "workflow_bypass":
        tester = WorkflowBypassTester()
        try:
            workflow_steps = payload.get("workflow_steps", [])
            findings = await tester.test_step_skip(
                workflow_steps, task_id=task_id, scan_id=scan_id
            )
        finally:
            await tester.close()

    # 競爭條件測試
    elif test_type == "race_condition":
        tester = RaceConditionTester()
        try:
            api = payload.get("api_endpoint")
            product_id = payload.get("product_id")
            if api and product_id:
                findings = await tester.test_inventory_race(
                    purchase_api=api,
                    product_id=product_id,
                    task_id=task_id,
                    scan_id=scan_id,
                )
        finally:
            await tester.close()

    return findings
