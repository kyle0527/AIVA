from __future__ import annotations
from typing import List
from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload, Vulnerability, FindingEvidence, FindingImpact, FindingRecommendation, FindingTarget
from services.aiva_common.utils import new_id, get_logger
from ..config.ssrf_config import SsrfConfig
from ..engine.ssrf_engine import SSRFEngine, SSRFIssue

logger = get_logger(__name__)

class SSRFDetector:
    def __init__(self, config: SsrfConfig):
        self.config = config

    async def analyze(self, target_url: str) -> List[FindingPayload]:
        engine = SSRFEngine(timeout=self.config.request_timeout,
                            max_redirects=self.config.max_redirects,
                            allow_active=self.config.allow_active_network,
                            safe_mode=self.config.safe_mode)
        try:
            issues = await engine.run(
                target_url=target_url,
                enable_internal=self.config.enable_internal_scan,
                enable_metadata=self.config.enable_cloud_metadata,
                enable_file=self.config.enable_file_protocol
            )
            return [self._issue_to_finding(i, target_url) for i in issues]
        finally:
            await engine.close()

    def _issue_to_finding(self, issue: SSRFIssue, target_url: str) -> FindingPayload:
        # Map issue to enums
        if issue.kind.startswith("SSRF_METADATA_EXPOSED"):
            sev = Severity.CRITICAL
        elif issue.kind.startswith("SSRF_INTERNAL_ACCESS"):
            sev = Severity.HIGH
        else:
            sev = Severity.MEDIUM
        vul = Vulnerability(
            name=VulnerabilityType.SERVER_SIDE_REQUEST_FORGERY if hasattr(VulnerabilityType,"SERVER_SIDE_REQUEST_FORGERY") else VulnerabilityType.INFO_LEAK,
            severity=sev,
            confidence=Confidence.POSSIBLE if "POTENTIAL" in issue.kind else Confidence.CERTAIN,
            description=issue.description,
            cwe=issue.cwe
        )
        evidence = FindingEvidence(payload=None, request=None, response=None, proof=issue.evidence or issue.description)
        impact = FindingImpact(description=issue.description, business_impact="SSRF can lead to internal network exposure and credential leaks.")
        rec = FindingRecommendation(fix="Validate outbound targets; enforce allow-list; block metadata endpoints; disable file://", priority=str(sev))
        target = FindingTarget(url=target_url, parameter=None, method="GET")
        return FindingPayload(
            finding_id=new_id("finding"), task_id="", scan_id="", status="potential" if "POTENTIAL" in issue.kind else "confirmed",
            vulnerability=vul, target=target, strategy="ssrf_analysis",
            evidence=evidence, impact=impact, recommendation=rec
        )
