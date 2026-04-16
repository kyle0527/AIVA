from __future__ import annotations

from aiva_common.enums import Confidence, Severity, VulnerabilityType
from aiva_common.schemas import (
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    FunctionTaskPayload,
    Vulnerability,
)
from aiva_common.utils import get_logger, new_id

from ..config.idor_config import IdorConfig
from ..engine.idor_engine import IDOREngine, IDORIssue

logger = get_logger(__name__)

class IDORDetector:
    def __init__(self, config: IdorConfig):
        self.config = config

    async def analyze(self, task: FunctionTaskPayload) -> list[FindingPayload]:
        """IDOR漏洞分析 - 重構版本（降低複雜度）"""
        engine = IDOREngine(
            timeout=self.config.request_timeout, 
            allow_active=self.config.allow_active_network, 
            safe_mode=self.config.safe_mode
        )
        try:
            url = str(task.target.url)
            ids = engine.extract_ids_from_url(url)
            findings: list[FindingPayload] = []

            # 水平測試（抽出成獨立函數）
            if self.config.horizontal_enabled:
                horizontal_findings = await self._perform_horizontal_tests(engine, url, ids, task)
                findings.extend(horizontal_findings)

            # 垂直測試（抽出成獨立函數）
            if self.config.vertical_enabled:
                vertical_findings = await self._perform_vertical_tests(engine, task)
                findings.extend(vertical_findings)

            return findings
        finally:
            await engine.close()

    async def _perform_horizontal_tests(
        self, 
        engine: IDOREngine, 
        url: str, 
        ids: list, 
        task: FunctionTaskPayload
    ) -> list[FindingPayload]:
        """執行水平IDOR測試——降低 analyze 複雜度"""
        findings: list[FindingPayload] = []
        # 使用默認的用戶認證設置——修復 options 屬性不存在的問題
        user_a = {}
        user_b = {}
        
        for cand in ids:
            for v in engine.generate_variants(cand.value, self.config.max_id_variations):
                test_url = engine.replace_id_in_url(url, cand.value, v)
                issue = await engine.test_horizontal(test_url, user_a, user_b)
                if issue:
                    findings.append(self._to_finding(issue, task, test_url))
        
        return findings

    async def _perform_vertical_tests(
        self, 
        engine: IDOREngine, 
        task: FunctionTaskPayload
    ) -> list[FindingPayload]:
        """執行垂直IDOR測試——降低 analyze 複雜度"""
        findings: list[FindingPayload] = []
        # 使用默認設置——修復 options 屬性不存在的問題
        low_auth = {}
        targets = self.config.privileged_urls or []
        
        for purl in targets:
            issue = await engine.test_vertical(purl, low_auth)
            if issue:
                findings.append(self._to_finding(issue, task, purl))
        
        return findings

    def _to_finding(self, issue: IDORIssue, task: FunctionTaskPayload, url: str) -> FindingPayload:
        """將 IDOR 問題轉換為 Finding 物件——簡化版本"""
        # 簡化嚴重程度判斷邏輯
        sev = self._determine_severity(issue.kind)
        vul = self._create_vulnerability(issue, sev)
        evidence = FindingEvidence(
            payload=None, 
            request=None, 
            response=None, 
            proof=issue.evidence or issue.description
        )
        impact = FindingImpact(
            description=issue.description, 
            business_impact="Unauthorized access to resources due to IDOR."
        )
        rec = FindingRecommendation(
            fix="Enforce authorization checks on resource IDs; avoid predictable IDs.", 
            priority=str(sev)
        )
        target = FindingTarget(url=url, parameter=None, method="GET")
        
        return FindingPayload(
            finding_id=new_id("finding"), 
            task_id=task.task_id, 
            scan_id=task.scan_id,
            status="potential" if "POTENTIAL" in issue.kind else "confirmed",
            vulnerability=vul, 
            target=target, 
            strategy="idor_analysis",
            evidence=evidence, 
            impact=impact, 
            recommendation=rec
        )

    def _determine_severity(self, kind: str) -> Severity:
        """根據 IDOR 類型決定嚴重程度"""
        if "VERTICAL" in kind:
            return Severity.CRITICAL
        elif "HORIZONTAL" in kind and "POTENTIAL" not in kind:
            return Severity.HIGH
        else:
            return Severity.MEDIUM

    def _create_vulnerability(self, issue: IDORIssue, severity: Severity) -> Vulnerability:
        """創建漏洞物件"""
        # 使用正確的漏洞類型——修復屬性不存在的問題
        vuln_type = VulnerabilityType.IDOR  # 直接使用 IDOR 類型
        confidence = (
            Confidence.POSSIBLE 
            if "POTENTIAL" in issue.kind 
            else Confidence.CERTAIN
        )
        
        return Vulnerability(
            name=vuln_type,
            severity=severity,
            confidence=confidence,
            description=issue.description,
            cwe=issue.cwe
        )

