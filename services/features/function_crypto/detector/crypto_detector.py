from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload, Vulnerability, FindingEvidence, FindingImpact, FindingRecommendation, FindingTarget
from services.aiva_common.utils import new_id, get_logger
from services.features.function_crypto.python_wrapper.engine_bridge import scan_code
from services.features.function_crypto.config.crypto_config import CryptoConfig

logger = get_logger(__name__)
config = CryptoConfig()

class CryptoDetector:
    def detect(self, target_code: str, task_id: str, scan_id: str) -> list[FindingPayload]:
        issues = scan_code(target_code)
        findings: list[FindingPayload] = []
        for issue_type, detail in issues:
            if issue_type in ("WEAK_ALGORITHM","WEAK_CIPHER"):
                vuln_name = VulnerabilityType.INFO_LEAK
                severity = Severity.HIGH
                confidence = Confidence.CERTAIN
                cwe_id = "CWE-327"
                description = "Weak or broken cryptographic algorithm in use."
                recommendation_text = "Replace with AES/GCM, SHA-256+ and modern KDFs."
            elif issue_type == "INSECURE_TLS":
                vuln_name = VulnerabilityType.WEAK_AUTH
                severity = Severity.HIGH
                confidence = Confidence.CERTAIN
                cwe_id = "CWE-295"
                description = "Insecure TLS configuration detected."
                recommendation_text = "Enforce TLS>=1.2 and enable certificate verification."
            elif issue_type == "HARDCODED_KEY":
                vuln_name = VulnerabilityType.WEAK_AUTH
                severity = Severity.CRITICAL
                confidence = Confidence.CERTAIN
                cwe_id = "CWE-321"
                description = "Hardcoded cryptographic key or secret found."
                recommendation_text = "Remove hardcoded keys; use a secrets manager."
            elif issue_type == "WEAK_RANDOM":
                vuln_name = VulnerabilityType.INFO_LEAK
                severity = Severity.MEDIUM
                confidence = Confidence.POSSIBLE
                cwe_id = "CWE-338"
                description = "Use of a potentially predictable RNG."
                recommendation_text = "Use `secrets` or OS-backed CSPRNG for security tasks."
            else:
                vuln_name = VulnerabilityType.INFO_LEAK
                severity = Severity.MEDIUM
                confidence = Confidence.FIRM
                cwe_id = None
                description = "Cryptography-related issue detected."
                recommendation_text = "Follow cryptography best practices."

            vulnerability = Vulnerability(
                name=vuln_name, severity=severity, confidence=confidence,
                description=description, cwe=cwe_id
            )
            evidence = FindingEvidence(proof=detail, payload=None, request=None, response=None)
            impact = FindingImpact(description=description, business_impact="Risk of data compromise or bypass.")
            recommendation = FindingRecommendation(fix=recommendation_text, priority=str(severity))
            target = FindingTarget(url=(target_code[:100] + "...") if len(target_code)>100 else target_code, parameter=None, method="STATIC_ANALYSIS")

            finding = FindingPayload(
                finding_id=new_id("finding"),
                task_id=task_id,
                scan_id=scan_id,
                status="confirmed" if confidence==Confidence.CERTAIN else "potential",
                vulnerability=vulnerability, target=target,
                strategy="crypto_analysis",
                evidence=evidence, impact=impact, recommendation=recommendation
            )
            findings.append(finding)
        return findings
