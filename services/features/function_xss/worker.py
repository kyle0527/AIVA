from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import HttpUrl, TypeAdapter

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

from .blind_xss_listener_validator import BlindXssEvent, BlindXssListenerValidator
from .dom_xss_detector import DomDetectionResult, DomXssDetector
from .payload_generator import XssPayloadGenerator
from .result_publisher import XssResultPublisher
from .stored_detector import StoredXssDetector, StoredXssResult
from .task_queue import QueuedTask, XssTaskQueue
from .traditional_detector import (
    TraditionalXssDetector,
    XssDetectionResult,
    XssExecutionError,
)

logger = get_logger(__name__)

_HTTP_URL_VALIDATOR = TypeAdapter(HttpUrl)


def _validated_http_url(value: str) -> HttpUrl:
    return _HTTP_URL_VALIDATOR.validate_python(value)


DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass
class XssExecutionTelemetry:
    payloads_sent: int = 0
    reflections: int = 0
    dom_escalations: int = 0
    blind_callbacks: int = 0
    errors: list[str] = field(default_factory=list)

    def to_details(self, findings_count: int) -> dict[str, Any]:
        details: dict[str, Any] = {
            "findings": findings_count,
            "payloads_sent": self.payloads_sent,
            "reflections": self.reflections,
            "dom_escalations": self.dom_escalations,
            "blind_callbacks": self.blind_callbacks,
        }
        if self.errors:
            details["errors"] = self.errors
        return details


@dataclass
class TaskExecutionResult:
    findings: list[FindingPayload]
    telemetry: XssExecutionTelemetry
    statistics_summary: dict[str, Any] | None = None  # 新增統計摘要


async def run() -> None:
    broker = await get_broker()
    publisher = XssResultPublisher(broker)
    queue = XssTaskQueue()
    consumer = asyncio.create_task(_consume_queue(queue, publisher))

    try:
        async for mqmsg in broker.subscribe(Topic.TASK_FUNCTION_XSS):
            msg = AivaMessage.model_validate_json(mqmsg.body)
            task = FunctionTaskPayload(**msg.payload)
            trace_id = msg.header.trace_id
            await queue.put(task, trace_id=trace_id)
    finally:  # pragma: no cover - defensive shutdown guard
        await queue.close()
        await consumer


async def _consume_queue(queue: XssTaskQueue, publisher: XssResultPublisher) -> None:
    while True:
        queued: QueuedTask | None = await queue.get()
        if queued is None:
            return

        await _execute_task(queued, publisher)


async def _execute_task(queued: QueuedTask, publisher: XssResultPublisher) -> None:
    task = queued.task
    trace_id = queued.trace_id

    await publisher.publish_status(task, "IN_PROGRESS", trace_id=trace_id)

    try:
        result = await process_task(task)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception(
            "Unhandled error while processing XSS task",
            extra={"task_id": task.task_id},
        )
        await publisher.publish_error(task, exc, trace_id=trace_id)
        return

    for finding in result.findings:
        await publisher.publish_finding(finding, trace_id=trace_id)

    # 記錄統計摘要
    if result.statistics_summary:
        logger.info(
            "XSS task completed with statistics",
            extra={
                "task_id": task.task_id,
                "statistics": result.statistics_summary
            }
        )

    await publisher.publish_status(
        task,
        "COMPLETED",
        trace_id=trace_id,
        details=result.telemetry.to_details(len(result.findings)),
    )


async def process_task(
    task: FunctionTaskPayload,
    *,
    payload_generator: XssPayloadGenerator | None = None,
    detector: TraditionalXssDetector | None = None,
    dom_detector: DomXssDetector | None = None,
    blind_validator: BlindXssListenerValidator | None = None,
    stored_detector: StoredXssDetector | None = None,
) -> TaskExecutionResult:
    config = task.test_config
    timeout = config.timeout or DEFAULT_TIMEOUT_SECONDS

    # 創建統計數據收集器
    stats_collector = StatisticsCollector(
        task_id=task.task_id,
        worker_type="xss"
    )

    generator = payload_generator or XssPayloadGenerator()
    validator = blind_validator
    blind_payload: str | None = None
    if config.blind_xss:
        validator = validator or BlindXssListenerValidator()
        try:
            blind_payload = await validator.provision_payload(task)
            # 記錄 OAST 探針 (Blind XSS)
            if blind_payload:
                stats_collector.record_oast_probe()
        except Exception as exc:  # pragma: no cover - defensive guard for OAST outages
            logger.exception(
                "Failed to provision blind XSS payload",
                extra={"task_id": task.task_id},
            )
            # 記錄錯誤
            stats_collector.record_error(
                category=ErrorCategory.NETWORK,
                message=f"Failed to provision blind XSS payload: {str(exc)}",
                request_info={"task_id": task.task_id}
            )
            blind_payload = None
            validator = None

    payloads = _build_payloads(task, generator, blind_payload)
    telemetry = XssExecutionTelemetry(payloads_sent=len(payloads))

    if not payloads:
        logger.debug("No payloads produced for task", extra={"task_id": task.task_id})
        # 完成統計
        stats_collector.finalize()
        return TaskExecutionResult(
            findings=[], 
            telemetry=telemetry,
            statistics_summary=stats_collector.get_summary()
        )

    detector = detector or TraditionalXssDetector(task, timeout=timeout)
    
    # 記錄 Payload 測試
    for _ in payloads:
        stats_collector.record_payload_test(success=False)
    
    detections = await detector.execute(payloads)
    
    # 記錄請求統計 (成功的檢測)
    stats_collector.stats.total_requests = len(payloads)
    stats_collector.stats.successful_requests = len(detections)

    errors: list[XssExecutionError] = getattr(detector, "errors", [])
    if errors:
        for error in errors:
            logger.warning(
                "Payload attempt failed",
                extra={
                    "task_id": task.task_id,
                    "payload": error.payload,
                    "vector": error.vector,
                    "error": error.message,
                    "attempts": error.attempts,
                },
            )
            # 記錄錯誤
            stats_collector.record_error(
                category=ErrorCategory.NETWORK if "timeout" not in error.message.lower() else ErrorCategory.TIMEOUT,
                message=error.message,
                request_info={
                    "payload": error.payload,
                    "vector": error.vector,
                    "attempts": error.attempts
                }
            )
        telemetry.errors = [error.to_detail() for error in errors]
        stats_collector.stats.failed_requests = len(errors)

    findings: list[FindingPayload] = []

    dom_engine = dom_detector if config.dom_testing else None
    if dom_engine is None and config.dom_testing:
        dom_engine = DomXssDetector()

    for detection in detections:
        dom_result: DomDetectionResult | None = None
        severity = Severity.MEDIUM
        confidence = Confidence.FIRM

        if dom_engine:
            dom_result = dom_engine.analyze(
                payload=detection.payload, document=detection.response_text
            )
            if dom_result:
                severity = Severity.HIGH
                confidence = Confidence.CERTAIN
                telemetry.dom_escalations += 1

        findings.append(
            _build_finding(
                task,
                detection,
                severity=severity,
                confidence=confidence,
                dom_result=dom_result,
            )
        )
        
        # 記錄漏洞發現
        stats_collector.record_vulnerability(false_positive=False)
        stats_collector.record_payload_test(success=True)

    telemetry.reflections = len(findings)

    # Stored XSS (submit then view) – conservative activation:
    # - if no direct reflections found, but blind or DOM hints exist
    # - or if user requested via payload set name 'stored' (future-compat)
    wants_stored = any(
        (name.lower() == "stored") for name in (task.test_config.payloads or [])
    )
    hinted = (telemetry.dom_escalations > 0) or bool(validator)
    if (not findings and hinted) or wants_stored:
        try:
            sdetector = stored_detector or StoredXssDetector(task, timeout=timeout)
            # Allow caller to pass follow-up view URLs via custom_payloads (optional)
            view_urls = [p for p in (task.custom_payloads or []) if isinstance(p, str)]
            sresults: list[StoredXssResult] = await sdetector.execute(
                payloads, view_urls=view_urls or None
            )
            for sres in sresults:
                findings.append(
                    _build_finding(
                        task,
                        XssDetectionResult(
                            payload=sres.payload,
                            request=sres.request,
                            response_status=sres.response_status,
                            response_headers=sres.response_headers,
                            response_text=sres.response_text,
                        ),
                        severity=Severity.HIGH,
                        confidence=Confidence.CERTAIN,
                        dom_result=None,
                    )
                )
        except Exception:  # pragma: no cover - defensive guard
            logger.exception(
                "Stored XSS detector failed",
                extra={"task_id": task.task_id},
            )

    if validator:
        blind_events = await validator.collect_events()
        for event in blind_events:
            findings.append(_build_blind_finding(task, event))
            
            # 記錄 OAST 回調 (Blind XSS)
            stats_collector.record_oast_callback(
                probe_token=event.token if hasattr(event, 'token') else "blind_xss",
                callback_type="blind_xss",
                source_ip=event.source_ip if hasattr(event, 'source_ip') else "unknown",
                payload_info={
                    "url": task.url,
                    "event_type": event.event_type if hasattr(event, 'event_type') else "unknown"
                }
            )
            
            # 記錄漏洞發現 (Blind XSS)
            stats_collector.record_vulnerability(false_positive=False)

        telemetry.blind_callbacks = len(blind_events)

    # 設置 XSS 特定統計數據
    stats_collector.set_module_specific("reflected_xss_tests", len(detections))
    stats_collector.set_module_specific("dom_xss_escalations", telemetry.dom_escalations)
    stats_collector.set_module_specific("blind_xss_enabled", config.blind_xss)
    stats_collector.set_module_specific("dom_testing_enabled", config.dom_testing)
    stats_collector.set_module_specific("stored_xss_tested", wants_stored or (not findings and hinted))
    
    # 完成統計數據收集
    stats_collector.finalize()

    return TaskExecutionResult(
        findings=findings, 
        telemetry=telemetry,
        statistics_summary=stats_collector.get_summary()
    )


def _build_payloads(
    task: FunctionTaskPayload,
    generator: XssPayloadGenerator,
    blind_payload: str | None,
) -> list[str]:
    config = task.test_config
    payload_sets: Iterable[str] | None = config.payloads or None

    combined_custom: list[str] = []
    if task.custom_payloads:
        combined_custom.extend(task.custom_payloads)
    if config.custom_payloads:
        combined_custom.extend(config.custom_payloads)

    payloads: list[str] = generator.generate(
        payload_sets=payload_sets,
        custom_payloads=combined_custom,
        blind_payload=blind_payload,
    )
    return payloads


def _build_finding(
    task: FunctionTaskPayload,
    detection: XssDetectionResult,
    *,
    severity: Severity,
    confidence: Confidence,
    dom_result: DomDetectionResult | None,
) -> FindingPayload:
    request = detection.request
    request_url: HttpUrl = _validated_http_url(str(request.url))
    evidence = FindingEvidence(
        payload=detection.payload,
        request=_format_request(request),
        response=_format_response(detection, dom_result=dom_result),
        proof=_proof_text(dom_result=dom_result, blind=False),
    )

    return FindingPayload(
        finding_id=new_id("finding"),
        task_id=task.task_id,
        scan_id=task.scan_id,
        status="VULNERABILITY_FOUND",
        vulnerability=Vulnerability(
            name=VulnerabilityType.XSS,
            severity=severity,
            confidence=confidence,
        ),
        target=FindingTarget(
            url=request_url,
            parameter=task.target.parameter,
            method=request.method,
        ),
        strategy=task.strategy,
        evidence=evidence,
        impact=_build_impact(dom_result=dom_result, blind=False),
        recommendation=_build_recommendation(dom_result=dom_result, blind=False),
    )


def _build_blind_finding(
    task: FunctionTaskPayload, event: BlindXssEvent
) -> FindingPayload:
    evidence = FindingEvidence(
        payload=event.token,
        request=event.request,
        response=event.response or event.evidence,
        proof=_proof_text(dom_result=None, blind=True),
    )

    return FindingPayload(
        finding_id=new_id("finding"),
        task_id=task.task_id,
        scan_id=task.scan_id,
        status="VULNERABILITY_FOUND",
        vulnerability=Vulnerability(
            name=VulnerabilityType.XSS,
            severity=Severity.HIGH,
            confidence=Confidence.CERTAIN,
        ),
        target=FindingTarget(
            url=task.target.url,
            parameter=task.target.parameter,
            method=task.target.method,
        ),
        strategy=task.strategy,
        evidence=evidence,
        impact=_build_impact(dom_result=None, blind=True),
        recommendation=_build_recommendation(dom_result=None, blind=True),
    )


def _build_impact(
    *, dom_result: DomDetectionResult | None, blind: bool
) -> FindingImpact:
    if blind:
        return FindingImpact(
            description="Blind XSS payload triggered out-of-band callback.",
            business_impact=(
                "Allows attackers to execute JavaScript on privileged users even when "
                "responses are not immediately visible."
            ),
        )

    if dom_result:
        description = "Payload executed within DOM context (script or event handler)."
    else:
        description = "Payload reflected in HTTP response without contextual encoding."

    return FindingImpact(
        description=description,
        business_impact=(
            "Enables attackers to run arbitrary JavaScript which can "
            "hijack sessions or steal sensitive data from impacted users."
        ),
    )


def _build_recommendation(
    *, dom_result: DomDetectionResult | None, blind: bool
) -> FindingRecommendation:
    if blind:
        fix = (
            "Audit server-side sinks that persist user input and ensure proper "
            "output encoding and Content Security Policy enforcement."
        )
    else:
        fix = (
            "Apply context-aware output encoding, validate input, and deploy a strict "
            "Content Security Policy to prevent script execution."
        )

    priority = "Critical" if blind or dom_result else "High"
    return FindingRecommendation(fix=fix, priority=priority)


def _proof_text(*, dom_result: DomDetectionResult | None, blind: bool) -> str:
    if blind:
        return (
            "Blind callback received from monitoring endpoint confirming exploitation."
        )

    if dom_result:
        return "Payload observed within executable DOM context."

    return "Payload reflected in HTTP response body without sanitization."


def _format_request(request) -> str:
    lines = [f"{request.method} {request.url} HTTP/1.1"]
    for key, value in request.headers.items():
        lines.append(f"{key}: {value}")
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def _format_response(
    detection: XssDetectionResult, *, dom_result: DomDetectionResult | None
) -> str:
    lines = [f"HTTP {detection.response_status}"]
    for key, value in detection.response_headers.items():
        lines.append(f"{key}: {value}")
    snippet = detection.response_text[:512]
    if snippet:
        lines.append("")
        lines.append(snippet)
    if dom_result:
        lines.append("")
        lines.append("DOM Context:")
        lines.append(dom_result.snippet)
    return "\n".join(lines)


def _inject_query(url: str, parameter: str | None, value: str) -> str:
    if not parameter:
        return url
    parts = list(urlparse(url))
    query_pairs = dict(parse_qsl(parts[4], keep_blank_values=True))
    query_pairs[parameter] = value
    parts[4] = urlencode(query_pairs, doseq=True)
    return urlunparse(parts)


class XssWorkerService:
    """XSS Worker 服務類 - 提供統一的任務處理接口"""
    
    def __init__(self):
        self.payload_generator = None
        self.detector = None
        
    async def process_task(self, task) -> dict:
        """處理 XSS 檢測任務"""
        # 將 Task 對象轉換為 FunctionTaskPayload
        if hasattr(task, 'target') and task.target:
            # 構建 FunctionTaskPayload
            payload = FunctionTaskPayload(
                header=MessageHeader(
                    message_id=task.task_id,
                    trace_id=task.task_id,
                    source_module="FunctionXSS"
                ),
                scan_id=getattr(task, 'scan_id', 'default'),
                target=task.target,
                strategy=getattr(task, 'strategy', 'normal'),
                priority=getattr(task, 'priority', 5)
            )
        else:
            raise ValueError("Task must have a valid target")
            
        # 使用現有的 process_task 函數
        return await process_task(
            payload,
            payload_generator=self.payload_generator,
            detector=self.detector
        )
