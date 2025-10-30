

from collections.abc import Iterable

from services.aiva_common.enums import Topic
from services.aiva_common.schemas import (
    FunctionTaskPayload,
    FunctionTaskTarget,
    ScanCompletedPayload,
)


class TaskGenerator:
    """Translate strategies into concrete Function tasks."""

    def from_strategy(
        self, plan: dict, payload: ScanCompletedPayload
    ) -> Iterable[tuple[Topic, FunctionTaskPayload]]:
        tasks: list[tuple[Topic, FunctionTaskPayload]] = []
        for index, x in enumerate(plan.get("xss", [])):
            tasks.append(
                (
                    Topic.TASK_FUNCTION_XSS,
                    FunctionTaskPayload(
                        task_id=f"xss-{payload.scan_id}-{index}",
                        scan_id=payload.scan_id,
                        priority=x["priority"],
                        target=FunctionTaskTarget(
                            url=x["asset"],
                            parameter=x.get("parameter"),
                            parameter_location=x.get("location", "query"),
                            method=x.get("method", "GET"),
                        ),
                    ),
                )
            )
        for index, x in enumerate(plan.get("sqli", [])):
            tasks.append(
                (
                    Topic.TASK_FUNCTION_SQLI,
                    FunctionTaskPayload(
                        task_id=f"sqli-{payload.scan_id}-{index}",
                        scan_id=payload.scan_id,
                        priority=x["priority"],
                        target=FunctionTaskTarget(
                            url=x["asset"],
                            parameter=x.get("parameter"),
                            parameter_location=x.get("location", "query"),
                            method=x.get("method", "GET"),
                        ),
                    ),
                )
            )
        for index, x in enumerate(plan.get("ssrf", [])):
            tasks.append(
                (
                    Topic.TASK_FUNCTION_SSRF,
                    FunctionTaskPayload(
                        task_id=f"ssrf-{payload.scan_id}-{index}",
                        scan_id=payload.scan_id,
                        priority=x["priority"],
                        target=FunctionTaskTarget(
                            url=x["asset"],
                            parameter=x.get("parameter"),
                            parameter_location=x.get("location", "query"),
                            method=x.get("method", "GET"),
                        ),
                    ),
                )
            )
        return tasks
