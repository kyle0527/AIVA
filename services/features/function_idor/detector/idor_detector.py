from __future__ import annotations
from typing import List
from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload, Vulnerability, FindingEvidence, FindingImpact, FindingRecommendation, FindingTarget, FunctionTaskPayload
from services.aiva_common.utils import new_id, get_logger
from ..config.idor_config import IdorConfig
from ..engine.idor_engine import IDOREngine, IDORIssue

logger = get_logger(__name__)

class IDORDetector:
    def __init__(self, config: IdorConfig):
        self.config = config

    async def analyze(self, task: FunctionTaskPayload) -> List[FindingPayload]:
        engine = IDOREngine(timeout=self.config.request_timeout, allow_active=self.config.allow_active_network, safe_mode=self.config.safe_mode)
        try:
            url = str(task.target.url)
            ids = engine.extract_ids_from_url(url)
            findings: List[FindingPayload] = []

            # Horizontal
            if self.config.horizontal_enabled:
                user_a = (task.options or {}).get("auth_user") or {}
                user_b = (task.options or {}).get("auth_other") or {}
                for cand in ids:
                    for v in engine.generate_variants(cand.value, self.config.max_id_variations):
                        test_url = engine.replace_id_in_url(url, cand.value, v)
                        issue = await engine.test_horizontal(test_url, user_a, user_b)
                        if issue:
                            findings.append(self._to_finding(issue, task, test_url))

            # Vertical
            if self.config.vertical_enabled:
                low_auth = (task.options or {}).get("auth_user") or {}
                targets = self.config.privileged_urls or (task.options or {}).get("privileged_urls") or []
                for purl in targets:
                    issue = await engine.test_vertical(purl, low_auth)
                    if issue:
                        findings.append(self._to_finding(issue, task, purl))

            return findings
        finally:
            await engine.close()

    def _to_finding(self, issue: IDORIssue, task: FunctionTaskPayload, url: str) -> FindingPayload:
        sev = Severity.CRITICAL if "VERTICAL" in issue.kind else (Severity.HIGH if "HORIZONTAL" in issue.kind and "POTENTIAL" not in issue.kind else Severity.MEDIUM)
        vul = Vulnerability(
            name=VulnerabilityType.PRIVILEGE_ESCALATION if "VERTICAL" in issue.kind else VulnerabilityType.ACCESS_CONTROL,
            severity=sev,
            confidence=Confidence.POSSIBLE if "POTENTIAL" in issue.kind else Confidence.CERTAIN,
            description=issue.description,
            cwe=issue.cwe
        )
        evidence = FindingEvidence(payload=None, request=None, response=None, proof=issue.evidence or issue.description)
        impact = FindingImpact(description=issue.description, business_impact="Unauthorized access to resources due to IDOR.")
        rec = FindingRecommendation(fix="Enforce authorization checks on resource IDs; avoid predictable IDs.", priority=str(sev))
        target = FindingTarget(url=url, parameter=None, method="GET")
        return FindingPayload(
            finding_id=new_id("finding"), task_id=task.task_id, scan_id=task.scan_id,
            status="potential" if "POTENTIAL" in issue.kind else "confirmed",
            vulnerability=vul, target=target, strategy="idor_analysis",
            evidence=evidence, impact=impact, recommendation=rec
        )
