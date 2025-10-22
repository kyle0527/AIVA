from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlparse, urlunparse

import httpx

from services.aiva_common.enums import Confidence, Severity, Topic, VulnerabilityType
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    FunctionTaskPayload,
    MessageHeader,
    Vulnerability,
)
from services.aiva_common.utils import get_logger, new_id
from services.features.common.worker_statistics import (
    StatisticsCollector,
    ErrorCategory,
)

from .internal_address_detector import InternalAddressDetector
from .oast_dispatcher import OastDispatcher, OastEvent
from .param_semantics_analyzer import (
    OAST_PLACEHOLDER,
    AnalysisPlan,
    ParamSemanticsAnalyzer,
    SsrfTestVector,
)
from .result_publisher import SsrfResultPublisher

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0


@dataclass
class SsrfTelemetry:
    attempts: int = 0
    findings: int = 0
    oast_callbacks: int = 0
    errors: list[str] = field(default_factory=list)

    def to_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "attempts": self.attempts,
            "findings": self.findings,
            "oast_callbacks": self.oast_callbacks,
        }
        if self.errors:
            details["errors"] = self.errors
        return details


@dataclass
class TaskExecutionResult:
    findings: list[FindingPayload]
    telemetry: SsrfTelemetry
    statistics_summary: dict[str, Any] | None = None  # 新增統計摘要


async def run() -> None:
    broker = await get_broker()
    publisher = SsrfResultPublisher(broker)
    analyzer = ParamSemanticsAnalyzer()
    detector = InternalAddressDetector()
    dispatcher = OastDispatcher()

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=DEFAULT_TIMEOUT_SECONDS
    ) as client:
        try:
            async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_SSRF):
                msg = AivaMessage.model_validate_json(mqmsg.body)
                task = FunctionTaskPayload(**msg.payload)
                trace_id = msg.header.trace_id

                await _execute_task(
                    task,
                    trace_id=trace_id,
                    client=client,
                    publisher=publisher,
                    analyzer=analyzer,
                    detector=detector,
                    dispatcher=dispatcher,
                )
        finally:  # pragma: no cover - defensive guard for shutdown
            await dispatcher.close()


async def _execute_task(
    task: FunctionTaskPayload,
    *,
    trace_id: str,
    client: httpx.AsyncClient,
    publisher: SsrfResultPublisher,
    analyzer: ParamSemanticsAnalyzer,
    detector: InternalAddressDetector,
    dispatcher: OastDispatcher,
) -> None:
    await publisher.publish_status(task, "IN_PROGRESS", trace_id=trace_id)

    try:
        result = await process_task(
            task,
            client=client,
            analyzer=analyzer,
            detector=detector,
            dispatcher=dispatcher,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception(
            "Unhandled error while processing SSRF task",
            extra={"task_id": task.task_id},
        )
        await publisher.publish_error(task, exc, trace_id=trace_id)
        return

    for finding in result.findings:
        await publisher.publish_finding(finding, trace_id=trace_id)

    # 記錄統計摘要到日誌
    if result.statistics_summary:
        logger.info(
            "SSRF task completed with statistics",
            extra={
                "task_id": task.task_id,
                "statistics": result.statistics_summary
            }
        )

    await publisher.publish_status(
        task,
        "COMPLETED",
        trace_id=trace_id,
        details=result.telemetry.to_details(),
    )


async def process_task(
    task: FunctionTaskPayload,
    *,
    client: httpx.AsyncClient,
    analyzer: ParamSemanticsAnalyzer | None = None,
    detector: InternalAddressDetector | None = None,
    dispatcher: OastDispatcherLike | None = None,
) -> TaskExecutionResult:
    analyzer = analyzer or ParamSemanticsAnalyzer()
    detector = detector or InternalAddressDetector()
    # Narrow type for static type checkers
    assert isinstance(detector, InternalAddressDetector)
    dispatcher = dispatcher or OastDispatcher()

    # 創建統計數據收集器
    stats_collector = StatisticsCollector(
        task_id=task.task_id,
        worker_type="ssrf"
    )

    plan: AnalysisPlan = analyzer.analyze(task)
    telemetry = SsrfTelemetry()
    findings: list[FindingPayload] = []

    if not plan.vectors:
        logger.debug(
            "No SSRF payloads generated for task", extra={"task_id": task.task_id}
        )
        # 完成統計收集（無測試）
        final_stats = stats_collector.finalize()
        return TaskExecutionResult(
            findings=findings, 
            telemetry=telemetry,
            statistics_summary=stats_collector.get_summary()
        )

    for vector in plan.vectors:
        telemetry.attempts += 1
        
        # 記錄 Payload 測試
        stats_collector.record_payload_test(success=False)
        
        payload = await _resolve_payload(vector, dispatcher, task)
        
        # 記錄 OAST 探針發送（如果適用）
        if vector.requires_oast:
            stats_collector.record_oast_probe()

        try:
            response = await _issue_request(client, task, vector, payload)
            
            # 記錄成功的請求
            stats_collector.record_request(
                success=True,
                timeout=False,
                rate_limited=False
            )
            
        except httpx.TimeoutException as exc:
            logger.warning(
                "SSRF payload request timeout",
                extra={"task_id": task.task_id, "payload": payload, "error": repr(exc)},
            )
            telemetry.errors.append(str(exc))
            
            # 記錄超時錯誤
            stats_collector.record_request(success=False, timeout=True)
            stats_collector.record_error(
                category=ErrorCategory.TIMEOUT,
                message=str(exc),
                request_info={"url": task.url, "payload": payload}
            )
            continue
            
        except httpx.NetworkError as exc:
            logger.warning(
                "SSRF payload network error",
                extra={"task_id": task.task_id, "payload": payload, "error": repr(exc)},
            )
            telemetry.errors.append(str(exc))
            
            # 記錄網絡錯誤
            stats_collector.record_request(success=False)
            stats_collector.record_error(
                category=ErrorCategory.NETWORK,
                message=str(exc),
                request_info={"url": task.url, "payload": payload}
            )
            continue
            
        except Exception as exc:
            logger.warning(
                "Failed to execute SSRF payload",
                extra={"task_id": task.task_id, "payload": payload, "error": repr(exc)},
            )
            telemetry.errors.append(str(exc))
            
            # 記錄未知錯誤
            stats_collector.record_request(success=False)
            stats_collector.record_error(
                category=ErrorCategory.UNKNOWN,
                message=str(exc),
                request_info={"url": task.url, "payload": payload}
            )
            continue

        detection: Any = detector.analyze(response)
        if detection.matched:
            telemetry.findings += 1
            
            # 記錄成功檢測到漏洞
            stats_collector.record_vulnerability(false_positive=False)
            stats_collector.record_payload_test(success=True)
            
            findings.append(
                _build_internal_finding(
                    task=task,
                    vector=vector,
                    payload=payload,
                    response=response,
                    detection_summary=detection.summary(),
                )
            )
            continue

        if vector.requires_oast:
            events = await dispatcher.fetch_events(payload)
            if not events:
                events = await dispatcher.fetch_events(_extract_token(payload))

            if events:
                telemetry.findings += 1
                telemetry.oast_callbacks += len(events)
                
                # 記錄 OAST 回調
                for event in events:
                    stats_collector.record_oast_callback(
                        probe_token=_extract_token(payload),
                        callback_type=event.event_type if hasattr(event, 'event_type') else "unknown",
                        source_ip=event.source_ip if hasattr(event, 'source_ip') else "unknown",
                        payload_info={
                            "url": task.url,
                            "parameter": vector.injection_point,
                            "payload": payload
                        }
                    )
                
                # 記錄成功檢測到漏洞
                stats_collector.record_vulnerability(false_positive=False)
                stats_collector.record_payload_test(success=True)
                
                findings.append(
                    _build_oast_finding(
                        task=task,
                        vector=vector,
                        payload=payload,
                        events=events,
                    )
                )
    
    # 設置 SSRF 特定統計數據
    stats_collector.set_module_specific("total_vectors_tested", len(plan.vectors))
    stats_collector.set_module_specific("internal_detection_tests", 
        sum(1 for v in plan.vectors if not v.requires_oast))
    stats_collector.set_module_specific("oast_tests", 
        sum(1 for v in plan.vectors if v.requires_oast))
    
    # 完成統計數據收集
    final_stats = stats_collector.finalize()

    return TaskExecutionResult(
        findings=findings, 
        telemetry=telemetry,
        statistics_summary=stats_collector.get_summary()
    )


async def _resolve_payload(
    vector: SsrfTestVector,
    dispatcher: OastDispatcherLike,
    task: FunctionTaskPayload,
) -> str:
    payload = vector.payload
    if vector.requires_oast:
        probe = await dispatcher.register(task)
        payload = payload.replace(OAST_PLACEHOLDER, probe.callback_url)
    # Ensure we return a str for type checkers (vector.payload may be Any)
    return str(payload)


async def _issue_request(
    client: httpx.AsyncClient,
    task: FunctionTaskPayload,
    vector: SsrfTestVector,
    payload: str,
) -> httpx.Response:
    target = task.target
    method = (target.method or "GET").upper()
    parameter = vector.parameter or target.parameter
    location = (vector.location or target.parameter_location or "query").lower()

    parsed = urlparse(str(target.url))
    base_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    headers = dict(target.headers)
    cookies = dict(target.cookies)
    data = dict(target.form_data)
    json_data = dict(target.json_data or {}) if target.json_data else None
    content = target.body

    if location in {"query", "url"} and parameter:
        base_params[parameter] = payload
    elif location in {"body", "form"} and parameter:
        data[parameter] = payload
    elif location == "json" and parameter:
        json_data = json_data or {}
        json_data[parameter] = payload
    elif location == "header" and parameter:
        headers[parameter] = payload
    elif location == "cookie" and parameter:
        cookies[parameter] = payload
    elif location == "body_raw":
        if parameter and content:
            content = content.replace(f"{{{{{parameter}}}}}", payload)
        else:
            content = payload

    request_url = urlunparse(parsed._replace(query=""))
    follow_redirects = vector.follow_redirects

    send_data = data if data else None
    send_json = json_data
    send_content = None if send_data is not None or send_json is not None else content

    response = await client.request(
        method,
        request_url,
        params=base_params or None,
        headers=headers or None,
        cookies=cookies or None,
        data=send_data,
        json=send_json,
        content=send_content,
        follow_redirects=follow_redirects,
    )
    return response


def _build_internal_finding(
    *,
    task: FunctionTaskPayload,
    vector: SsrfTestVector,
    payload: str,
    response: httpx.Response,
    detection_summary: str,
) -> FindingPayload:
    severity = _severity_from_summary(detection_summary)

    evidence = FindingEvidence(
        payload=payload,
        response_time_delta=_safe_elapsed(response),
        request=_format_request(response.request),
        response=_format_response(response),
        proof=detection_summary,
    )

    return FindingPayload(
        finding_id=new_id("finding"),
        task_id=task.task_id,
        scan_id=task.scan_id,
        status="VULNERABILITY_FOUND",
        vulnerability=Vulnerability(
            name=VulnerabilityType.SSRF,
            severity=severity,
            confidence=Confidence.CERTAIN,
        ),
        target=FindingTarget(
            url=task.target.url,
            parameter=vector.parameter or task.target.parameter,
            method=task.target.method,
        ),
        strategy=task.strategy,
        evidence=evidence,
        impact=FindingImpact(
            description=("Internal resource accessible through server-side request."),
            business_impact=(
                "Potential exposure of internal services or cloud metadata endpoints."
            ),
        ),
        recommendation=FindingRecommendation(
            fix=(
                "Validate and sanitize user supplied URLs, restrict "
                "outbound network access, and enforce allow-lists for "
                "reachable services."
            ),
            priority="High" if severity is Severity.HIGH else "Medium",
        ),
    )


def _build_oast_finding(
    *,
    task: FunctionTaskPayload,
    vector: SsrfTestVector,
    payload: str,
    events: list[OastEvent],
) -> FindingPayload:
    summary = (
        "; ".join(filter(None, (event.evidence for event in events)))
        or "OAST callback received"
    )

    evidence = FindingEvidence(
        payload=payload,
        proof=f"OAST callbacks received: {summary}",
        request=None,
        response=json.dumps([event.__dict__ for event in events]) if events else None,
    )

    return FindingPayload(
        finding_id=new_id("finding"),
        task_id=task.task_id,
        scan_id=task.scan_id,
        status="VULNERABILITY_FOUND",
        vulnerability=Vulnerability(
            name=VulnerabilityType.SSRF,
            severity=Severity.HIGH,
            confidence=Confidence.CERTAIN,
        ),
        target=FindingTarget(
            url=task.target.url,
            parameter=vector.parameter or task.target.parameter,
            method=task.target.method,
        ),
        strategy=task.strategy,
        evidence=evidence,
        impact=FindingImpact(
            description="Blind SSRF confirmed via out-of-band callback.",
            business_impact=(
                "Enables attackers to pivot into internal network or "
                "exfiltrate sensitive metadata through secondary services."
            ),
        ),
        recommendation=FindingRecommendation(
            fix=(
                "Implement strict allow-lists for outbound requests and "
                "segregate internal services from user controlled fetchers."
            ),
            priority="Critical",
        ),
    )


def _format_request(request: httpx.Request) -> str:
    if request is None:
        return ""
    lines = [f"{request.method} {request.url} HTTP/1.1"]
    for key, value in request.headers.items():
        lines.append(f"{key}: {value}")
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    if body:
        lines.append("")
        lines.append(body[:512])
    return "\n".join(lines)


def _format_response(response: httpx.Response) -> str:
    lines = [f"HTTP {response.status_code}"]
    for key, value in response.headers.items():
        lines.append(f"{key}: {value}")
    snippet = response.text[:512]
    if snippet:
        lines.append("")
        lines.append(snippet)
    return "\n".join(lines)


def _safe_elapsed(response: httpx.Response) -> float | None:
    try:
        elapsed = response.elapsed
    except RuntimeError:
        return None
    if not elapsed:
        return None
    return elapsed.total_seconds()


def _severity_from_summary(summary: str) -> Severity:
    keywords = ["169.254.169.254", "metadata", "security-credentials"]
    if any(keyword in summary for keyword in keywords):
        return Severity.HIGH
    return Severity.MEDIUM


def _extract_token(payload: str) -> str:
    return payload.split("/")[-1]


@runtime_checkable
class OastDispatcherLike(Protocol):
    async def register(
        self, task: FunctionTaskPayload
    ) -> Any:  # returns a probe with callback_url
        ...

    async def fetch_events(self, token: str) -> list[OastEvent]: ...

    async def close(self) -> None: ...


class SsrfWorkerService:
    """SSRF Worker 服務類 - 提供統一的任務處理接口"""
    
    def __init__(self):
        self.oast_dispatcher = None
        self.internal_detector = None
        
    async def process_task(self, task) -> dict:
        """處理 SSRF 檢測任務"""
        # 將 Task 對象轉換為 FunctionTaskPayload
        if hasattr(task, 'target') and task.target:
            # 構建 FunctionTaskPayload
            payload = FunctionTaskPayload(
                header=MessageHeader(
                    message_id=task.task_id,
                    trace_id=task.task_id,
                    source_module="function_ssrf"
                ),
                scan_id=getattr(task, 'scan_id', 'default'),
                target=task.target,
                strategy=getattr(task, 'strategy', 'normal'),
                priority=getattr(task, 'priority', 5)
            )
        else:
            raise ValueError("Task must have a valid target")
            
        # 使用現有的 _execute_task 函數
        return await _execute_task(
            payload,
            oast_dispatcher=self.oast_dispatcher,
            internal_address_detector=self.internal_detector
        )
