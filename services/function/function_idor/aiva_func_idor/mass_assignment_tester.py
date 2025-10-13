"""
Mass Assignment Tester
巨量賦值漏洞測試器

檢測應用程式是否會不當處理 POST/PUT/PATCH 請求中的額外欄位
"""

import asyncio
from enum import Enum
import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from services.aiva_common.enums import Confidence, Severity, VulnerabilityType
from services.aiva_common.schemas import (
    Authentication,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    FindingTarget,
    Vulnerability,
)
from services.aiva_common.utils import new_id

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """欄位類型"""

    PRIVILEGE = "privilege"  # 權限提升
    ROLE = "role"  # 角色變更
    STATUS = "status"  # 狀態變更
    SENSITIVE = "sensitive"  # 敏感資料


class MassAssignmentPayload(BaseModel):
    """巨量賦值測試載荷"""

    field_name: str
    field_value: Any
    field_type: FieldType
    description: str


class MassAssignmentTestResult(BaseModel):
    """測試結果"""

    endpoint: str
    method: str
    injected_field: str
    injected_value: Any
    normal_status: int
    injected_status: int
    is_vulnerable: bool
    response_evidence: dict[str, Any] = Field(default_factory=dict)


class MassAssignmentTester:
    """巨量賦值測試器"""

    # 常見的危險欄位（可能被用於權限提升）
    DANGEROUS_FIELDS = [
        MassAssignmentPayload(
            field_name="isAdmin",
            field_value=True,
            field_type=FieldType.PRIVILEGE,
            description="管理員標記",
        ),
        MassAssignmentPayload(
            field_name="is_admin",
            field_value=True,
            field_type=FieldType.PRIVILEGE,
            description="管理員標記（蛇形命名）",
        ),
        MassAssignmentPayload(
            field_name="role",
            field_value="admin",
            field_type=FieldType.ROLE,
            description="角色欄位",
        ),
        MassAssignmentPayload(
            field_name="roles",
            field_value=["admin", "superuser"],
            field_type=FieldType.ROLE,
            description="角色陣列",
        ),
        MassAssignmentPayload(
            field_name="permissions",
            field_value=["read", "write", "delete", "admin"],
            field_type=FieldType.PRIVILEGE,
            description="權限陣列",
        ),
        MassAssignmentPayload(
            field_name="is_verified",
            field_value=True,
            field_type=FieldType.STATUS,
            description="驗證狀態",
        ),
        MassAssignmentPayload(
            field_name="is_active",
            field_value=True,
            field_type=FieldType.STATUS,
            description="啟用狀態",
        ),
        MassAssignmentPayload(
            field_name="status",
            field_value="approved",
            field_type=FieldType.STATUS,
            description="審核狀態",
        ),
        MassAssignmentPayload(
            field_name="balance",
            field_value=999999,
            field_type=FieldType.SENSITIVE,
            description="帳戶餘額",
        ),
        MassAssignmentPayload(
            field_name="credit",
            field_value=999999,
            field_type=FieldType.SENSITIVE,
            description="信用額度",
        ),
        MassAssignmentPayload(
            field_name="price",
            field_value=0.01,
            field_type=FieldType.SENSITIVE,
            description="價格操縱",
        ),
    ]

    def __init__(
        self,
        auth: Authentication,
        timeout: int = 10,
        max_concurrent: int = 5,
    ):
        """
        初始化測試器

        Args:
            auth: 認證資訊
            timeout: 請求超時時間（秒）
            max_concurrent: 最大並發數
        """
        self.auth = auth
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def test_endpoint(
        self,
        endpoint: str,
        method: str,
        base_payload: dict[str, Any],
        custom_fields: list[MassAssignmentPayload] | None = None,
    ) -> list[MassAssignmentTestResult]:
        """
        測試端點的巨量賦值漏洞

        Args:
            endpoint: 目標 URL
            method: HTTP 方法（POST, PUT, PATCH）
            base_payload: 正常的請求載荷
            custom_fields: 自訂測試欄位（預設使用 DANGEROUS_FIELDS）

        Returns:
            測試結果列表
        """
        test_fields = custom_fields or self.DANGEROUS_FIELDS
        results = []

        # 1. 發送正常請求作為基準
        normal_response = await self._send_request(endpoint, method, base_payload)
        if not normal_response:
            logger.warning(f"Failed to get baseline response for {endpoint}")
            return results

        # 2. 測試每個危險欄位
        for field in test_fields:
            async with self.semaphore:
                result = await self._test_field_injection(
                    endpoint=endpoint,
                    method=method,
                    base_payload=base_payload,
                    injection_field=field,
                    normal_response=normal_response,
                )
                if result:
                    results.append(result)
                    if result.is_vulnerable:
                        logger.warning(
                            f"Mass Assignment detected: {endpoint} - "
                            f"field '{field.field_name}' accepted"
                        )

        return results

    async def _send_request(
        self,
        endpoint: str,
        method: str,
        payload: dict[str, Any],
    ) -> httpx.Response | None:
        """發送 HTTP 請求"""
        try:
            headers = self._build_headers()
            headers["Content-Type"] = "application/json"

            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                response = await client.request(
                    method=method,
                    url=endpoint,
                    headers=headers,
                    json=payload,
                )
                return response
        except Exception as e:
            logger.warning(f"Request failed: {method} {endpoint}, error: {e}")
            return None

    def _build_headers(self) -> dict[str, str]:
        """建立請求標頭"""
        headers = {}

        # 處理認證
        if self.auth.method == "bearer" and self.auth.credentials:
            token = self.auth.credentials.get("bearer_token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif self.auth.method == "basic" and self.auth.credentials:
            username = self.auth.credentials.get("username")
            password = self.auth.credentials.get("password")
            if username and password:
                import base64

                credentials = f"{username}:{password}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        return headers

    async def _test_field_injection(
        self,
        endpoint: str,
        method: str,
        base_payload: dict[str, Any],
        injection_field: MassAssignmentPayload,
        normal_response: httpx.Response,
    ) -> MassAssignmentTestResult | None:
        """測試注入特定欄位"""
        # 建立注入載荷
        injected_payload = base_payload.copy()
        injected_payload[injection_field.field_name] = injection_field.field_value

        # 發送注入請求
        injected_response = await self._send_request(endpoint, method, injected_payload)
        if not injected_response:
            return None

        # 分析是否存在漏洞
        is_vulnerable = self._is_mass_assignment_vulnerable(
            normal_response=normal_response,
            injected_response=injected_response,
            injection_field=injection_field,
        )

        return MassAssignmentTestResult(
            endpoint=endpoint,
            method=method,
            injected_field=injection_field.field_name,
            injected_value=injection_field.field_value,
            normal_status=normal_response.status_code,
            injected_status=injected_response.status_code,
            is_vulnerable=is_vulnerable,
            response_evidence={
                "normal_response": self._extract_response_data(normal_response),
                "injected_response": self._extract_response_data(injected_response),
                "field_type": injection_field.field_type.value,
            },
        )

    def _is_mass_assignment_vulnerable(
        self,
        normal_response: httpx.Response,
        injected_response: httpx.Response,
        injection_field: MassAssignmentPayload,
    ) -> bool:
        """
        判斷是否存在巨量賦值漏洞

        漏洞條件：
        1. 注入請求成功（200-299）
        2. 回應中包含注入的欄位值
        3. 或回應內容發生顯著變化
        """
        # 如果注入請求失敗，可能不存在漏洞
        if not (200 <= injected_response.status_code < 300):
            return False

        # 如果狀態碼相同，進一步分析回應內容
        try:
            normal_json = normal_response.json()
            injected_json = injected_response.json()

            # 檢查注入的欄位是否出現在回應中
            if injection_field.field_name in injected_json:
                injected_value = injected_json[injection_field.field_name]
                # 如果值匹配，確定存在漏洞
                if injected_value == injection_field.field_value:
                    return True

            # 檢查是否有新增的權限相關欄位
            privilege_fields = ["isAdmin", "is_admin", "role", "roles", "permissions"]
            for field in privilege_fields:
                if field in injected_json and field not in normal_json:
                    return True

        except Exception as e:
            logger.debug(f"Failed to parse JSON response: {e}")

        return False

    def _extract_response_data(self, response: httpx.Response) -> dict[str, Any]:
        """提取回應資料"""
        try:
            return response.json()
        except Exception:
            return {
                "status_code": response.status_code,
                "content_length": len(response.content),
                "headers": dict(response.headers),
            }

    def create_finding(
        self,
        test_result: MassAssignmentTestResult,
        task_id: str,
        scan_id: str,
    ) -> FindingPayload:
        """根據測試結果建立 Finding"""
        finding_id = new_id("finding")

        # 判斷嚴重性
        severity = self._determine_severity(test_result)

        # 建立漏洞物件
        vulnerability = Vulnerability(
            name=VulnerabilityType.BOLA,  # Mass Assignment 也是授權問題
            cwe="CWE-915",  # Improperly Controlled Modification of Dynamically-Determined Object Attributes
            severity=severity,
            confidence=Confidence.FIRM,
        )

        # 建立目標
        target = FindingTarget(
            url=test_result.endpoint,
            method=test_result.method,
            parameter=test_result.injected_field,
        )

        # 建立證據
        evidence = FindingEvidence(
            request=(
                f"{test_result.method} {test_result.endpoint}\n"
                f"Content-Type: application/json\n\n"
                f'{{\n  "{test_result.injected_field}": {json.dumps(test_result.injected_value)}\n}}'
            ),
            response=(
                f"HTTP {test_result.injected_status}\n"
                f"{json.dumps(test_result.response_evidence.get('injected_response', {}), indent=2)}"
            ),
            payload=json.dumps(
                {test_result.injected_field: test_result.injected_value}
            ),
            proof_of_concept=(
                f"1. 發送正常請求到 {test_result.endpoint}\n"
                f"2. 在請求中添加額外欄位 '{test_result.injected_field}': {test_result.injected_value}\n"
                f"3. 伺服器接受並處理了該欄位\n"
                f"4. 可能導致權限提升或未授權的資料修改"
            ),
        )

        # 建立影響
        impact = FindingImpact(
            confidentiality="HIGH",
            integrity="CRITICAL",
            availability="LOW",
            business_impact=(
                "攻擊者可以透過注入額外欄位來修改不應由客戶端控制的屬性，"
                "可能導致權限提升、資料篡改或業務邏輯繞過"
            ),
        )

        # 建立修復建議
        recommendation = FindingRecommendation(
            remediation=(
                "1. 使用白名單（Allowlist）明確定義可接受的欄位\n"
                "2. 使用 DTO (Data Transfer Object) 模式限制可綁定的屬性\n"
                "3. 在 ORM 層標記唯讀欄位（如 Laravel 的 $guarded, Django 的 read_only_fields）\n"
                "4. 永不直接將使用者輸入綁定到模型物件\n"
                "5. 實施嚴格的輸入驗證"
            ),
            references=[
                "https://owasp.org/API-Security/editions/2023/en/0xa3-broken-object-property-level-authorization/",
                "https://cwe.mitre.org/data/definitions/915.html",
                "https://cheatsheetseries.owasp.org/cheatsheets/Mass_Assignment_Cheat_Sheet.html",
            ],
        )

        return FindingPayload(
            finding_id=finding_id,
            task_id=task_id,
            scan_id=scan_id,
            status="detected",
            vulnerability=vulnerability,
            target=target,
            evidence=evidence,
            impact=impact,
            recommendation=recommendation,
        )

    def _determine_severity(self, test_result: MassAssignmentTestResult) -> Severity:
        """根據測試結果判斷嚴重性"""
        evidence = test_result.response_evidence
        field_type = evidence.get("field_type", "")

        # 權限提升類欄位最嚴重
        if field_type == FieldType.PRIVILEGE.value:
            return Severity.CRITICAL

        # 角色變更次之
        if field_type == FieldType.ROLE.value:
            return Severity.HIGH

        # 敏感資料修改
        if field_type == FieldType.SENSITIVE.value:
            return Severity.HIGH

        # 狀態變更
        if field_type == FieldType.STATUS.value:
            return Severity.MEDIUM

        return Severity.MEDIUM


async def main():
    """測試範例"""
    # 模擬認證
    auth = Authentication(
        method="bearer",
        credentials={"bearer_token": "user_token_12345"},
    )

    # 建立測試器
    tester = MassAssignmentTester(auth=auth)

    # 測試端點
    endpoint = "https://example.com/api/v1/users/profile"
    method = "PUT"
    base_payload = {
        "name": "John Doe",
        "email": "john@example.com",
    }

    results = await tester.test_endpoint(endpoint, method, base_payload)

    # 輸出結果
    print(f"\n=== Testing {endpoint} ===")
    for result in results:
        print(f"  Field: {result.injected_field}")
        print(f"  Vulnerable: {result.is_vulnerable}")
        if result.is_vulnerable:
            finding = tester.create_finding(result, "test_task_123")
            print(f"    Finding ID: {finding.finding_id}")
            print(f"    Severity: {finding.vulnerability.severity.value}")


if __name__ == "__main__":
    asyncio.run(main())
