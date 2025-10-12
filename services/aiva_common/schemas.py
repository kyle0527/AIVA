from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .enums import Confidence, ModuleName, Severity, Topic, VulnerabilityType


class MessageHeader(BaseModel):
    message_id: str
    trace_id: str
    correlation_id: str | None = None
    source_module: ModuleName
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"


class AivaMessage(BaseModel):
    header: MessageHeader
    topic: Topic
    schema_version: str = "1.0"
    payload: dict[str, Any]


class Authentication(BaseModel):
    method: str = "none"
    credentials: dict[str, str] | None = None


class RateLimit(BaseModel):
    requests_per_second: int = 25
    burst: int = 50

    @field_validator("requests_per_second", "burst")
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate limit must be non-negative")
        return v


class ScanScope(BaseModel):
    exclusions: list[str] = []
    include_subdomains: bool = True
    allowed_hosts: list[str] = []


class ScanStartPayload(BaseModel):
    scan_id: str
    targets: list[HttpUrl]
    scope: ScanScope = Field(default_factory=ScanScope)
    authentication: Authentication = Field(default_factory=Authentication)
    strategy: str = "deep"
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    custom_headers: dict[str, str] = {}
    x_forwarded_for: str | None = None


class Asset(BaseModel):
    asset_id: str
    type: str
    value: str
    parameters: list[str] | None = None
    has_form: bool = False


class Summary(BaseModel):
    urls_found: int = 0
    forms_found: int = 0
    apis_found: int = 0
    scan_duration_seconds: int = 0


class Fingerprints(BaseModel):
    web_server: dict[str, str] | None = None
    framework: dict[str, str] | None = None
    language: dict[str, str] | None = None
    waf_detected: bool = False
    waf_vendor: str | None = None


class ScanCompletedPayload(BaseModel):
    scan_id: str
    status: str
    summary: Summary
    assets: list[Asset] = []
    fingerprints: Fingerprints | None = None
    error_info: str | None = None


class FunctionTaskTarget(BaseModel):
    # Accept arbitrary URL-like values; runtime code will cast to str as needed
    url: Any
    parameter: str | None = None
    method: str = "GET"
    parameter_location: str = "query"
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    form_data: dict[str, Any] = Field(default_factory=dict)
    json_data: dict[str, Any] | None = None
    body: str | None = None


class FunctionTaskContext(BaseModel):
    db_type_hint: str | None = None
    waf_detected: bool = False
    related_findings: list[str] | None = None


class FunctionTaskTestConfig(BaseModel):
    payloads: list[str] = Field(default_factory=lambda: ["basic"])
    custom_payloads: list[str] = Field(default_factory=list)
    blind_xss: bool = False
    dom_testing: bool = False
    timeout: float | None = None


class FunctionTaskPayload(BaseModel):
    task_id: str
    scan_id: str
    priority: int = 5
    target: FunctionTaskTarget
    context: FunctionTaskContext = Field(default_factory=FunctionTaskContext)
    strategy: str = "full"
    custom_payloads: list[str] | None = None
    test_config: FunctionTaskTestConfig = Field(default_factory=FunctionTaskTestConfig)


class FeedbackEventPayload(BaseModel):
    task_id: str
    scan_id: str
    event_type: str
    details: dict[str, Any] = {}
    form_url: HttpUrl | None = None


class Vulnerability(BaseModel):
    name: VulnerabilityType
    cwe: str | None = None
    severity: Severity
    confidence: Confidence


class FindingTarget(BaseModel):
    # Accept arbitrary URL-like values
    url: Any
    parameter: str | None = None
    method: str | None = None


class FindingEvidence(BaseModel):
    payload: str | None = None
    response_time_delta: float | None = None
    db_version: str | None = None
    request: str | None = None
    response: str | None = None
    proof: str | None = None


class FindingImpact(BaseModel):
    description: str | None = None
    business_impact: str | None = None


class FindingRecommendation(BaseModel):
    fix: str | None = None
    priority: str | None = None


class FindingPayload(BaseModel):
    finding_id: str
    task_id: str
    scan_id: str
    status: str
    vulnerability: Vulnerability
    target: FindingTarget
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None


class TaskUpdatePayload(BaseModel):
    task_id: str
    scan_id: str
    status: str
    worker_id: str
    details: dict[str, Any] | None = None


class HeartbeatPayload(BaseModel):
    module: ModuleName
    worker_id: str
    capacity: int


class ConfigUpdatePayload(BaseModel):
    update_id: str
    config_items: dict[str, Any] = {}
