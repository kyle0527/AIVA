"""
Workflow Bypass Tester - 工作流程繞過測試器

測試多步驟業務流程中的邏輯缺陷:
- 步驟跳過
- 狀態驗證繞過
- 強制瀏覽 (Forced Browsing)
- 工作流程順序錯亂
"""

from __future__ import annotations

import httpx

from services.aiva_common.enums import Severity, VulnerabilityType
from services.aiva_common.schemas import FindingPayload
from services.aiva_common.utils import get_logger

from .finding_helper import create_bizlogic_finding

logger = get_logger(__name__)


class WorkflowBypassTester:
    """
    工作流程繞過測試器

    測試多步驟流程(如註冊、結帳、審批)中的業務邏輯缺陷
    """

    def __init__(self, client: httpx.AsyncClient | None = None):
        """
        初始化工作流程繞過測試器

        Args:
            client: HTTP 客戶端
        """
        self.client = client or httpx.AsyncClient(timeout=10.0)

    async def test_step_skip(
        self, workflow_steps: list[dict[str, str]], task_id: str, scan_id: str
    ) -> list[FindingPayload]:
        """
        測試步驟跳過漏洞

        嘗試跳過中間步驟,直接訪問最終步驟

        Args:
            workflow_steps: 工作流程步驟列表
                [
                    {"name": "step1", "url": "/checkout/step1"},
                    {"name": "step2", "url": "/checkout/step2"},
                    {"name": "step3", "url": "/checkout/step3"},
                ]

        Returns:
            list[FindingPayload]: 發現的漏洞
        """
        logger.info(f"Testing workflow step skip for {len(workflow_steps)} steps")
        findings = []

        if len(workflow_steps) < 2:
            return findings

        # 嘗試直接訪問最後一步(跳過中間步驟)
        final_step = workflow_steps[-1]

        try:
            response = await self.client.get(final_step["url"])

            # 如果能直接訪問最終步驟且包含最終步驟內容,可能存在漏洞
            if response.status_code == 200 and any(
                keyword in response.text.lower()
                for keyword in ["confirm", "complete", "submit", "finish"]
            ):
                finding = create_bizlogic_finding(
                    vuln_type=VulnerabilityType.WORKFLOW_BYPASS,
                    severity=Severity.HIGH,
                    target_url=final_step["url"],
                    method="GET",
                    evidence_data={
                        "request": {"url": final_step["url"], "method": "GET"},
                        "response": {"status_code": response.status_code},
                        "proof": f"可以跳過前面步驟直接訪問 {final_step['name']}",
                    },
                    task_id=task_id,
                    scan_id=scan_id,
                )
                findings.append(finding)
                logger.warning(
                    f"Workflow bypass: Can directly access {final_step['name']}"
                )

        except Exception as e:
            logger.debug(f"Step skip test failed: {e}")

        return findings

    async def test_forced_browsing(
        self, protected_urls: list[str], task_id: str, scan_id: str
    ) -> list[FindingPayload]:
        """
        測試強制瀏覽漏洞

        嘗試直接訪問應該受保護的 URL

        Args:
            protected_urls: 應該受保護的 URL 列表

        Returns:
            list[FindingPayload]: 發現的漏洞
        """
        logger.info(f"Testing forced browsing on {len(protected_urls)} URLs")
        findings = []

        for url in protected_urls:
            try:
                response = await self.client.get(url)

                # 如果能訪問應該受保護的資源
                if response.status_code == 200:
                    finding = create_bizlogic_finding(
                        vuln_type=VulnerabilityType.FORCED_BROWSING,
                        severity=Severity.MEDIUM,
                        target_url=url,
                        method="GET",
                        evidence_data={
                            "request": {"url": url, "method": "GET"},
                            "response": {"status_code": response.status_code},
                            "proof": f"可以直接訪問受保護 URL {url}",
                        },
                        task_id=task_id,
                        scan_id=scan_id,
                    )
                    findings.append(finding)
                    logger.warning(f"Forced browsing: Accessed protected URL {url}")

            except Exception as e:
                logger.debug(f"Forced browsing test failed for {url}: {e}")

        return findings

    async def test_state_manipulation(
        self, state_endpoint: str, states: list[str], task_id: str, scan_id: str
    ) -> list[FindingPayload]:
        """
        測試狀態操縱漏洞

        嘗試手動修改訂單/流程狀態

        Args:
            state_endpoint: 狀態更新端點
            states: 要測試的狀態列表

        Returns:
            list[FindingPayload]: 發現的漏洞
        """
        logger.info(f"Testing state manipulation on {state_endpoint}")
        findings = []

        for state in states:
            try:
                response = await self.client.post(
                    state_endpoint, json={"status": state}
                )

                if response.status_code == 200:
                    data = response.json()

                    # 如果狀態被成功修改
                    if data.get("status") == state or data.get("success"):
                        finding = create_bizlogic_finding(
                            vuln_type=VulnerabilityType.STATE_MANIPULATION,
                            severity=Severity.HIGH,
                            target_url=state_endpoint,
                            method="POST",
                            evidence_data={
                                "request": {"url": state_endpoint, "method": "POST", "body": {"status": state}},
                                "response": {"status_code": response.status_code, "data": data},
                                "proof": f"成功將狀態設置為 {state}",
                            },
                            task_id=task_id,
                            scan_id=scan_id,
                        )
                        findings.append(finding)
                        logger.warning(
                            f"State manipulation: Successfully set state to {state}"
                        )

            except Exception as e:
                logger.debug(f"State manipulation test failed for {state}: {e}")

        return findings

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()
