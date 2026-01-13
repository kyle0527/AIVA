from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload, Vulnerability, FindingEvidence, FindingImpact, FindingRecommendation, FindingTarget
from services.aiva_common.utils import new_id, get_logger
from services.features.function_postex.engines.privilege_engine import PrivilegeEscalationTester
from services.features.function_postex.engines.lateral_engine import LateralMovementTester
from services.features.function_postex.engines.persistence_engine import PersistenceChecker

logger = get_logger(__name__)

class PostExDetector:
    def analyze(self, test_type: str, target: str, task_id: str, scan_id: str, safe_mode: bool = True, auth_token: str | None = None) -> list[FindingPayload]:
        findings: list[FindingPayload] = []
        t = (test_type or "").lower()
        if t == "privilege_escalation":
            report = PrivilegeEscalationTester(auth_token, safe_mode).run_full_check()
            for issue in report.get("findings", []):
                findings.append(self._mk_finding(VulnerabilityType.PRIVILEGE_ESCALATION, Severity.CRITICAL, Confidence.POSSIBLE if safe_mode else Confidence.CERTAIN, issue.get("message","Privilege escalation"), "CWE-269", "Fix sudo/SUID misconfigurations", task_id, scan_id, target))
        elif t == "lateral_movement":
            report = LateralMovementTester(auth_token, target, safe_mode).run_full_assessment()
            for test in report.get("tests", []):
                for issue in test.get("findings", []):
                    if issue.get("type") == "credential_reuse":
                        findings.append(self._mk_finding(VulnerabilityType.WEAK_AUTH, Severity.HIGH, Confidence.POSSIBLE, issue.get("message","Credential reuse"), None, "Use unique credentials per host", task_id, scan_id, target))
        elif t == "persistence":
            report = PersistenceChecker(auth_token, safe_mode).run_full_check()
            for issue in report.get("findings", []):
                vt = VulnerabilityType.ACCESS_CONTROL
                desc = "Potential persistence/backdoor"
                if issue.get("issue") == "backdoor_account":
                    desc = "Backdoor user account present"
                if issue.get("issue") == "startup_script_backdoor":
                    desc = "Malicious startup script configured"
                findings.append(self._mk_finding(vt, Severity.CRITICAL, Confidence.POSSIBLE, desc, "CWE-912", "Remove unauthorized persistence", task_id, scan_id, target))
        else:
            logger.warning("Unknown postex test_type", extra={"test_type": test_type})
        return findings

    def _mk_finding(self, vt, severity, confidence, description, cwe, fix, task_id, scan_id, target) -> FindingPayload:
        vulnerability = Vulnerability(name=vt, severity=severity, confidence=confidence, description=description, cwe=cwe)
        evidence = FindingEvidence(proof=description, payload=None, request=None, response=None)
        impact = FindingImpact(description=description, business_impact="Could lead to full compromise if exploited")
        rec = FindingRecommendation(fix=fix, priority=str(severity))
        target_obj = FindingTarget(url=target or "localhost", parameter=None, method="POSTEX")
        return FindingPayload(
            finding_id=new_id("finding"), task_id=task_id, scan_id=scan_id,
            status="confirmed" if confidence.name=='CERTAIN' else "potential",
            vulnerability=vulnerability, target=target_obj, strategy="post_exploitation",
            evidence=evidence, impact=impact, recommendation=rec
        )
