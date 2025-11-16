"""BizLogic Finding 創建輔助函數
"""

from typing import Any

from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import (
    FindingEvidence,
    FindingPayload,
    FindingTarget,
    Vulnerability,
)
from services.aiva_common.utils import new_id


def create_bizlogic_finding(
    vuln_type: VulnerabilityType,
    severity: Severity,
    target_url: str,
    method: str,
    evidence_data: dict[str, Any],
    task_id: str = "task_bizlogic",
    scan_id: str = "scan_bizlogic",
    parameter: str | None = None,
) -> FindingPayload:
    """創建業務邏輯漏洞 Finding

    Args:
        vuln_type: 漏洞類型
        severity: 嚴重程度
        target_url: 目標 URL
        method: HTTP 方法
        evidence_data: 證據數據
        task_id: 任務 ID
        scan_id: 掃描 ID
        parameter: 參數名稱

    Returns:
        FindingPayload 對象
    """
    vulnerability = Vulnerability(
        name=vuln_type,
        severity=severity,
        confidence=Confidence.FIRM,
    )

    target = FindingTarget(
        url=target_url,
        method=method,
        parameter=parameter,
    )

    evidence = FindingEvidence(
        request=str(evidence_data.get("request", {})),
        response=str(evidence_data.get("response", {})),
        proof=evidence_data.get("proof", ""),
    )

    return FindingPayload(
        finding_id=new_id("finding"),
        task_id=task_id,
        scan_id=scan_id,
        status="detected",
        vulnerability=vulnerability,
        target=target,
        evidence=evidence,
    )
