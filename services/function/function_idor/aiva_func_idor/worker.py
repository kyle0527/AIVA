"""
IDOR Detection Worker

Main worker for detecting Insecure Direct Object Reference (IDOR) vulnerabilities.
Consumes tasks from message queue and performs horizontal/vertical privilege
escalation testing.
"""

from __future__ import annotations

import json

import httpx
from pydantic import HttpUrl, TypeAdapter

from services.aiva_common.enums import (
    Confidence,
    ModuleName,
    Severity,
    Topic,
    VulnerabilityType,
)
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

from .cross_user_tester import CrossUserTester, CrossUserTestResult
from .resource_id_extractor import ResourceIdExtractor
from .vertical_escalation_tester import (
    PrivilegeLevel,
    VerticalEscalationTester,
    VerticalTestResult,
)

logger = get_logger(__name__)

_HTTP_URL_VALIDATOR = TypeAdapter(HttpUrl)


def _validated_http_url(value: str) -> HttpUrl:
    """Validate and convert string to HttpUrl."""
    return _HTTP_URL_VALIDATOR.validate_python(value)


async def run() -> None:
    """
    Start the IDOR detection worker.

    Subscribes to FUNCTION_IDOR_TASK topic and processes incoming
    detection tasks.
    """
    broker = await get_broker()
    worker = IdorWorker()

    async for mqmsg in broker.subscribe(Topic.FUNCTION_IDOR_TASK):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            task = FunctionTaskPayload(**msg.payload)
            trace_id = msg.header.trace_id

            logger.info(f"收到 IDOR 檢測任務: {task.task_id}")

            # Detect horizontal privilege escalation (IDOR)
            horizontal_findings = await worker.detect_idor(task)

            # Detect vertical privilege escalation (BFLA)
            vertical_findings = await worker.detect_vertical_escalation(task)

            # Combine all findings
            findings = horizontal_findings + vertical_findings

            # Publish findings
            for finding in findings:
                out = AivaMessage(
                    header=MessageHeader(
                        message_id=new_id("msg"),
                        trace_id=trace_id,
                        correlation_id=task.scan_id,
                        source_module=ModuleName.FUNC_IDOR,
                    ),
                    topic=Topic.FINDING_DETECTED,
                    payload=finding.model_dump(),
                )
                await broker.publish(
                    Topic.FINDING_DETECTED,
                    json.dumps(out.model_dump()).encode("utf-8"),
                )

            logger.info(
                f"任務 {task.task_id} 完成，發現 {len(findings)} 個漏洞 "
                f"(橫向: {len(horizontal_findings)}, 垂直: {len(vertical_findings)})"
            )

        except Exception as exc:
            logger.exception(f"IDOR 檢測失敗: {exc}")


class IdorWorker:
    """
    IDOR vulnerability detection worker.

    Detects horizontal and vertical privilege escalation vulnerabilities
    by testing cross-user resource access patterns.

    Example:
        >>> worker = IdorWorker()
        >>> findings = await worker.detect_idor(task)
    """

    def __init__(self):
        """Initialize IDOR worker with detectors."""
        self.extractor = ResourceIdExtractor()
        self.http_client = httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, verify=True
        )
        self.tester = CrossUserTester(self.http_client)
        self.vertical_tester = VerticalEscalationTester(self.http_client)

    async def detect_idor(self, task: FunctionTaskPayload) -> list[FindingPayload]:
        """
        Execute IDOR detection on the target.

        Steps:
        1. Extract resource IDs from URL
        2. Generate test ID variations
        3. Test cross-user access for each variation
        4. Build vulnerability findings for detected issues

        Args:
            task: Function task containing target information

        Returns:
            List of detected vulnerabilities
        """
        findings: list[FindingPayload] = []

        # Step 1: Extract resource IDs from URL
        url_str = str(task.target.url)
        resource_ids = self.extractor.extract_from_url(url_str)

        if not resource_ids:
            logger.info(f"未在 URL 中找到資源 ID: {url_str}")
            return findings

        logger.info(
            f"找到 {len(resource_ids)} 個資源 ID: {[r.value for r in resource_ids]}"
        )

        # Step 2-4: Test each resource ID
        for rid in resource_ids:
            # Generate test ID variations
            test_ids = self.extractor.generate_test_ids(rid, count=5)
            logger.info(f"為 ID '{rid.value}' 生成 {len(test_ids)} 個測試 ID")

            # Get authentication credentials
            owner_auth = self._extract_auth(task)
            test_user_auth = self._get_test_user_auth(task)

            # Test each ID variation
            for test_id in test_ids:
                # Replace ID in URL
                test_url = self.extractor.replace_id_in_url(url_str, rid.value, test_id)
                logger.debug(f"測試 URL: {test_url}")

                # Test for horizontal IDOR
                result = await self.tester.test_horizontal_idor(
                    url=test_url,
                    resource_id=test_id,
                    user_a_auth=owner_auth or {},
                    user_b_auth=test_user_auth or {},
                )

                if result.vulnerable:
                    logger.warning(
                        f"🚨 檢測到 IDOR 漏洞: {test_url} "
                        f"(相似度: {result.similarity_score:.2%})"
                    )
                    findings.append(
                        self._build_finding(
                            task=task,
                            test_url=_validated_http_url(test_url),
                            original_id=rid.value,
                            test_id=test_id,
                            result=result,
                        )
                    )

        return findings

    async def detect_vertical_escalation(
        self, task: FunctionTaskPayload
    ) -> list[FindingPayload]:
        """
        Execute vertical privilege escalation detection.

        Tests if lower-privilege users can access higher-privilege functions.

        Args:
            task: Function task containing target information

        Returns:
            List of detected BFLA vulnerabilities
        """
        findings: list[FindingPayload] = []

        # Determine privilege level required based on URL patterns
        url_str = str(task.target.url)
        required_level = self._infer_required_privilege(url_str)

        if required_level == PrivilegeLevel.USER:
            # No point testing if it's a regular user endpoint
            logger.debug(f"Skipping vertical test for user-level endpoint: {url_str}")
            return findings

        # Get authentication credentials
        user_auth = self._extract_auth(task)

        # Test with different privilege levels
        test_levels = [PrivilegeLevel.GUEST, PrivilegeLevel.USER]

        for test_level in test_levels:
            if test_level == required_level:
                continue  # Skip testing with same level

            # For now, we use the same auth but mark it as different level
            # TODO: Implement proper multi-user credential management
            result = await self.vertical_tester.test_vertical_escalation(
                url=url_str,
                user_auth=user_auth if test_level == PrivilegeLevel.USER else {},
                user_level=test_level,
                required_level=required_level,
                method=task.target.method,
            )

            if result.vulnerable:
                logger.warning(
                    f"🚨 檢測到 BFLA 漏洞: {url_str} "
                    f"(user={test_level.value}, required={required_level.value})"
                )
                findings.append(
                    self._build_vertical_finding(
                        task=task,
                        url=_validated_http_url(url_str),
                        result=result,
                    )
                )

        return findings

    def _infer_required_privilege(self, url: str) -> PrivilegeLevel:
        """
        Infer required privilege level from URL patterns.

        Args:
            url: Target URL

        Returns:
            Inferred privilege level
        """
        url_lower = url.lower()

        # Admin patterns
        if any(
            pattern in url_lower
            for pattern in ["/admin", "/administrator", "/management", "/console"]
        ):
            return PrivilegeLevel.ADMIN

        # Moderator patterns
        if any(pattern in url_lower for pattern in ["/mod", "/moderator", "/moderate"]):
            return PrivilegeLevel.MODERATOR

        # Superadmin patterns
        if any(pattern in url_lower for pattern in ["/superadmin", "/root", "/system"]):
            return PrivilegeLevel.SUPERADMIN

        # Default to user level
        return PrivilegeLevel.USER

    def _build_vertical_finding(
        self,
        task: FunctionTaskPayload,
        url: HttpUrl,
        result: VerticalTestResult,
    ) -> FindingPayload:
        """
        Build a BFLA vulnerability finding from test result.

        Args:
            task: Original task
            url: URL that was tested
            result: Vertical escalation test result

        Returns:
            FindingPayload describing the BFLA vulnerability
        """
        return FindingPayload(
            finding_id=new_id("finding"),
            task_id=task.task_id,
            scan_id=task.scan_id,
            status="CONFIRMED",
            vulnerability=Vulnerability(
                name=VulnerabilityType.BOLA,
                severity=Severity.HIGH,
                confidence=Confidence.CERTAIN,
            ),
            target=FindingTarget(
                url=url,
                parameter=task.target.parameter,
                method=task.target.method,
            ),
            strategy=task.strategy,
            evidence=FindingEvidence(
                payload=(
                    f"Privilege: {result.actual_level.value if result.actual_level else 'Unknown'} → "
                    f"{result.attempted_level.value if result.attempted_level else 'Unknown'}"
                ),
                proof=result.evidence if hasattr(result, 'evidence') else None,
            ),
            impact=FindingImpact(
                description=(
                    f"低權限用戶 ({result.actual_level.value if result.actual_level else 'Unknown'}) 可以訪問需要 "
                    f"{result.attempted_level.value if result.attempted_level else 'Unknown'} 權限的功能。"
                ),
                business_impact="攻擊者可以執行未授權的管理操作，導致系統完全被控制",
            ),
            recommendation=FindingRecommendation(
                fix="實施嚴格的功能級授權檢查，在每個端點驗證用戶權限",
                priority="CRITICAL",
            ),
        )

    def _build_finding(
        self,
        task: FunctionTaskPayload,
        test_url: HttpUrl,
        original_id: str,
        test_id: str,
        result: CrossUserTestResult,
    ) -> FindingPayload:
        """
        Build a vulnerability finding from test result.

        Args:
            task: Original task
            test_url: URL that was tested
            original_id: Original resource ID
            test_id: Test ID that succeeded
            result: Test result with evidence

        Returns:
            FindingPayload describing the vulnerability
        """
        return FindingPayload(
            finding_id=new_id("finding"),
            task_id=task.task_id,
            scan_id=task.scan_id,
            status="CONFIRMED",
            vulnerability=Vulnerability(
                name=VulnerabilityType.IDOR,
                severity=Severity.HIGH,
                confidence=Confidence.CERTAIN,
            ),
            target=FindingTarget(
                url=test_url,
                parameter=task.target.parameter,
                method=task.target.method,
            ),
            strategy=task.strategy,
            evidence=FindingEvidence(
                payload=f"ID: {original_id} → {test_id}",
                proof=result.evidence,
            ),
            impact=FindingImpact(
                description=(
                    f"攻擊者可以通過修改資源 ID ({original_id} → {test_id}) "
                    f"訪問未授權的資源。相似度：{result.similarity_score:.2%}"
                ),
                business_impact="未授權用戶可以訪問、修改或刪除其他用戶的資源",
            ),
            recommendation=FindingRecommendation(
                fix="實施基於角色的訪問控制（RBAC），在服務端驗證用戶是否有權訪問請求的資源",
                priority="HIGH",
            ),
        )

    def _extract_auth(self, task: FunctionTaskPayload) -> dict[str, str]:
        """
        Extract authentication information from task.

        Args:
            task: Function task payload

        Returns:
            Authentication dictionary
        """
        auth: dict[str, str] = {}

        # Extract from task headers if available
        if task.target.headers:
            if "Authorization" in task.target.headers:
                auth["Authorization"] = task.target.headers["Authorization"]
            if "Cookie" in task.target.headers:
                auth["Cookie"] = task.target.headers["Cookie"]

        # Extract from cookies
        if task.target.cookies:
            from urllib.parse import quote

            cookie_str = "; ".join(
                f"{k}={quote(str(v))}" for k, v in task.target.cookies.items()
            )
            auth["Cookie"] = cookie_str

        return auth

    def _get_test_user_auth(self, task: FunctionTaskPayload) -> dict[str, str] | None:
        """
        Get test user authentication credentials.

        In a real implementation, this should:
        1. Create a second test user account
        2. Authenticate as that user
        3. Return credentials

        For now, returns None to test unauthenticated access.

        Args:
            task: Function task payload

        Returns:
            Test user authentication dictionary or None
        """
        _ = task  # Mark as used
        # TODO: Implement proper multi-user testing
        # For now, test unauthenticated access
        return None
