"""
IDOR & Privilege Escalation Testing Module
完整實現水平越權 (Horizontal)、垂直越權 (Vertical) 和資源枚舉 (Enumeration)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import aiohttp


class EscalationType(Enum):
    """越權類型"""
    HORIZONTAL = "horizontal"  # 水平越權 (同級別用戶)
    VERTICAL = "vertical"      # 垂直越權 (權限提升)
    ENUMERATION = "enumeration"  # 資源枚舉


class ResourceType(Enum):
    """資源類型"""
    USER_PROFILE = "user_profile"
    USER_DATA = "user_data"
    ADMIN_PANEL = "admin_panel"
    API_ENDPOINT = "api_endpoint"
    FILE_ACCESS = "file_access"
    DATABASE_RECORD = "database_record"


@dataclass
class TestUser:
    """測試用戶"""
    user_id: str
    username: str
    role: str  # user, admin, guest
    token: Optional[str] = None
    session: Optional[str] = None
    cookies: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class IDORTestCase:
    """IDOR 測試案例"""
    test_id: str
    escalation_type: EscalationType
    resource_type: ResourceType
    url: str
    method: str = "GET"
    params: Dict[str, str] = field(default_factory=dict)
    data: Optional[Dict] = None
    attacker: Optional[TestUser] = None
    victim: Optional[TestUser] = None
    description: str = ""


@dataclass
class IDORFinding:
    """IDOR 發現結果"""
    test_id: str
    escalation_type: EscalationType
    severity: str
    vulnerable: bool
    url: str
    method: str
    description: str
    evidence: Dict
    impact: str
    remediation: str
    cvss_score: float = 0.0


class PrivilegeEscalationTester:
    """權限提升測試器"""

    def __init__(self, target_url: str, logger: Optional[logging.Logger] = None):
        self.target_url = target_url
        self.logger = logger or logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.findings: List[IDORFinding] = []
        
        # ID 模式匹配
        self.id_patterns = [
            r'\b(id|user_id|uid|account_id)=(\d+)',
            r'\b(username|user|account)=([a-zA-Z0-9_-]+)',
            r'/users?/(\d+)',
            r'/profile/(\d+)',
            r'/account/([a-zA-Z0-9_-]+)',
            r'\b(doc_id|file_id|item_id)=(\d+)',
        ]

    async def __aenter__(self):
        """異步上下文管理器進入"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        if self.session:
            await self.session.close()

    async def test_horizontal_escalation(
        self,
        attacker: TestUser,
        victim: TestUser,
        target_url: str,
        method: str = "GET"
    ) -> IDORFinding:
        """
        測試水平越權
        
        場景：低權限用戶 A 試圖訪問同等權限用戶 B 的資源
        例如：用戶 Alice 訪問用戶 Bob 的個人資料
        """
        self.logger.info(
            f"Testing horizontal escalation: {attacker.username} -> {victim.username}"
        )

        # 1. 使用攻擊者身份訪問受害者資源
        attacker_response = await self._make_request(
            url=target_url,
            method=method,
            user=attacker,
            target_user_id=victim.user_id
        )

        # 2. 使用受害者身份訪問自己的資源(正常行為)
        victim_response = await self._make_request(
            url=target_url,
            method=method,
            user=victim,
            target_user_id=victim.user_id
        )

        # 3. 分析結果
        vulnerable = self._analyze_horizontal_access(
            attacker_response,
            victim_response,
            victim
        )

        severity = "HIGH" if vulnerable else "INFO"
        cvss = 7.5 if vulnerable else 0.0

        finding = IDORFinding(
            test_id=f"h_esc_{attacker.user_id}_{victim.user_id}",
            escalation_type=EscalationType.HORIZONTAL,
            severity=severity,
            vulnerable=vulnerable,
            url=target_url,
            method=method,
            description=(
                f"水平越權測試: {attacker.username}({attacker.role}) "
                f"嘗試訪問 {victim.username}({victim.role}) 的資源"
            ),
            evidence={
                "attacker_status": attacker_response.get("status", 0),
                "victim_status": victim_response.get("status", 0),
                "attacker_data": attacker_response.get("body", ""),
                "victim_data": victim_response.get("body", ""),
                "leaked_fields": self._find_leaked_fields(
                    attacker_response.get("body", ""),
                    victim_response.get("body", "")
                ),
            },
            impact=(
                "攻擊者能夠未經授權訪問其他用戶的私密資料，"
                "可能導致隱私洩露、身份盜用或進一步的攻擊。"
                if vulnerable else "未發現漏洞"
            ),
            remediation=(
                "1. 實施嚴格的用戶身份驗證\n"
                "2. 在服務端驗證請求用戶與資源擁有者的關係\n"
                "3. 使用不可預測的資源標識符 (UUID)\n"
                "4. 記錄並監控異常訪問模式"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_vertical_escalation(
        self,
        low_priv_user: TestUser,
        high_priv_user: TestUser,
        admin_url: str,
        method: str = "GET"
    ) -> IDORFinding:
        """
        測試垂直越權
        
        場景：低權限用戶試圖訪問高權限資源
        例如：普通用戶訪問管理員功能
        """
        self.logger.info(
            f"Testing vertical escalation: {low_priv_user.username} -> admin functions"
        )

        # 1. 使用低權限用戶訪問高權限資源
        low_priv_response = await self._make_request(
            url=admin_url,
            method=method,
            user=low_priv_user
        )

        # 2. 使用高權限用戶訪問(正常行為)
        high_priv_response = await self._make_request(
            url=admin_url,
            method=method,
            user=high_priv_user
        )

        # 3. 使用訪客訪問(無認證)
        guest_response = await self._make_request(
            url=admin_url,
            method=method,
            user=None
        )

        # 4. 分析垂直越權
        vulnerable = self._analyze_vertical_access(
            low_priv_response,
            high_priv_response,
            guest_response
        )

        severity = "CRITICAL" if vulnerable else "INFO"
        cvss = 9.1 if vulnerable else 0.0

        finding = IDORFinding(
            test_id=f"v_esc_{low_priv_user.user_id}",
            escalation_type=EscalationType.VERTICAL,
            severity=severity,
            vulnerable=vulnerable,
            url=admin_url,
            method=method,
            description=(
                f"垂直越權測試: {low_priv_user.username}({low_priv_user.role}) "
                f"嘗試訪問 {high_priv_user.role} 級別的資源"
            ),
            evidence={
                "low_priv_status": low_priv_response.get("status", 0),
                "high_priv_status": high_priv_response.get("status", 0),
                "guest_status": guest_response.get("status", 0),
                "low_priv_data": low_priv_response.get("body", ""),
                "privileged_functions": self._extract_admin_functions(
                    low_priv_response.get("body", "")
                ),
            },
            impact=(
                "低權限用戶能夠執行管理員功能，可能導致系統完全被控制、"
                "數據洩露、用戶資料篡改或系統破壞。"
                if vulnerable else "未發現漏洞"
            ),
            remediation=(
                "1. 實施基於角色的訪問控制 (RBAC)\n"
                "2. 在每個敏感操作前驗證用戶權限\n"
                "3. 使用權限管理中間件\n"
                "4. 最小權限原則\n"
                "5. 定期審計權限配置"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_resource_enumeration(
        self,
        user: TestUser,
        base_url: str,
        id_param: str,
        id_range: Tuple[int, int] = (1, 100),
        method: str = "GET"
    ) -> IDORFinding:
        """
        測試資源枚舉
        
        場景：通過遍歷可預測的 ID 來發現並訪問資源
        例如：/api/users/1, /api/users/2, /api/users/3 ...
        """
        self.logger.info(
            f"Testing resource enumeration: {base_url} with {id_param}"
        )

        accessible_resources = []
        forbidden_resources = []
        not_found_resources = []

        start_id, end_id = id_range

        # 並發掃描資源 ID
        tasks = []
        for resource_id in range(start_id, end_id + 1):
            task = self._test_single_resource(
                base_url=base_url,
                resource_id=str(resource_id),
                id_param=id_param,
                user=user,
                method=method
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 統計結果
        for result in results:
            if isinstance(result, Exception):
                continue

            status = result.get("status", 0)
            resource_id = result.get("resource_id", "")

            if status == 200:
                accessible_resources.append({
                    "id": resource_id,
                    "data": result.get("body", "")[:200]  # 截取前 200 字符
                })
            elif status == 403:
                forbidden_resources.append(resource_id)
            elif status == 404:
                not_found_resources.append(resource_id)

        # 判斷是否存在枚舉漏洞
        vulnerable = len(accessible_resources) > 10  # 超過 10 個可訪問資源視為漏洞
        
        severity = "MEDIUM" if vulnerable else "LOW"
        cvss = 5.3 if vulnerable else 2.0

        finding = IDORFinding(
            test_id=f"enum_{id_param}_{start_id}_{end_id}",
            escalation_type=EscalationType.ENUMERATION,
            severity=severity,
            vulnerable=vulnerable,
            url=base_url,
            method=method,
            description=f"資源枚舉測試: 掃描 {id_param} 從 {start_id} 到 {end_id}",
            evidence={
                "total_scanned": end_id - start_id + 1,
                "accessible_count": len(accessible_resources),
                "forbidden_count": len(forbidden_resources),
                "not_found_count": len(not_found_resources),
                "sample_accessible": accessible_resources[:5],  # 顯示前 5 個
                "enumerable_pattern": self._detect_enumeration_pattern(
                    accessible_resources
                ),
            },
            impact=(
                f"攻擊者能夠通過遍歷 ID 發現 {len(accessible_resources)} 個資源，"
                "可能導致敏感數據洩露或進一步攻擊的基礎。"
                if vulnerable else "枚舉風險較低"
            ),
            remediation=(
                "1. 使用 UUID 或不可預測的資源標識符\n"
                "2. 實施速率限制 (Rate Limiting)\n"
                "3. 對連續失敗的訪問實施封鎖\n"
                "4. 不要在錯誤訊息中洩露資源是否存在\n"
                "5. 使用 CAPTCHA 防止自動化掃描"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def _make_request(
        self,
        url: str,
        method: str,
        user: Optional[TestUser] = None,
        target_user_id: Optional[str] = None
    ) -> Dict:
        """發送 HTTP 請求"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        # 替換 URL 中的用戶 ID
        if target_user_id:
            url = self._replace_id_in_url(url, target_user_id)

        # 準備請求頭
        headers = {}
        cookies = {}

        if user:
            if user.token:
                headers["Authorization"] = f"Bearer {user.token}"
            if user.session:
                cookies["session"] = user.session
            headers.update(user.headers)
            cookies.update(user.cookies)

        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                cookies=cookies,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                body = await response.text()
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": body,
                    "cookies": response.cookies,
                }
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return {
                "status": 0,
                "error": str(e),
                "body": "",
            }

    async def _test_single_resource(
        self,
        base_url: str,
        resource_id: str,
        id_param: str,
        user: Optional[TestUser],
        method: str
    ) -> Dict:
        """測試單個資源"""
        url = self._build_url_with_id(base_url, resource_id, id_param)
        result = await self._make_request(url, method, user)
        result["resource_id"] = resource_id
        return result

    def _replace_id_in_url(self, url: str, new_id: str) -> str:
        """替換 URL 中的 ID"""
        for pattern in self.id_patterns:
            url = re.sub(pattern, lambda m: f"{m.group(1)}={new_id}", url)
        return url

    def _build_url_with_id(self, base_url: str, resource_id: str, id_param: str) -> str:
        """構建帶有 ID 的 URL"""
        parsed = urlparse(base_url)
        query_params = parse_qs(parsed.query)
        query_params[id_param] = [resource_id]

        new_query = urlencode(query_params, doseq=True)
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))

    def _analyze_horizontal_access(
        self,
        attacker_response: Dict,
        victim_response: Dict,
        victim: TestUser
    ) -> bool:
        """分析水平訪問結果"""
        # 1. 檢查狀態碼
        if attacker_response.get("status") != 200:
            return False

        # 2. 檢查是否返回受害者數據
        victim_data = victim_response.get("body", "")
        attacker_data = attacker_response.get("body", "")

        # 3. 檢查受害者標識符
        victim_identifiers = [
            victim.user_id,
            victim.username,
        ]

        for identifier in victim_identifiers:
            if identifier and identifier in attacker_data:
                return True

        # 4. 檢查數據相似度
        if self._calculate_similarity(victim_data, attacker_data) > 0.8:
            return True

        return False

    def _analyze_vertical_access(
        self,
        low_priv_response: Dict,
        high_priv_response: Dict,
        guest_response: Dict
    ) -> bool:
        """分析垂直訪問結果"""
        low_status = low_priv_response.get("status", 0)
        high_status = high_priv_response.get("status", 0)
        guest_status = guest_response.get("status", 0)

        # 1. 如果低權限用戶獲得 200，而訪客得到 403/401，可能存在問題
        if low_status == 200 and high_status == 200:
            if guest_status in [401, 403]:
                return True

        # 2. 檢查是否包含管理功能
        low_body = low_priv_response.get("body", "")
        admin_keywords = ["admin", "管理", "delete", "刪除", "modify", "編輯"]

        for keyword in admin_keywords:
            if keyword.lower() in low_body.lower():
                return True

        return False

    def _find_leaked_fields(self, attacker_data: str, victim_data: str) -> List[str]:
        """找出洩露的欄位"""
        leaked = []

        # 簡化實現：檢查關鍵字段
        sensitive_fields = [
            "email", "phone", "address", "ssn", "credit_card",
            "password", "token", "api_key"
        ]

        for field in sensitive_fields:
            if field in attacker_data.lower() and field in victim_data.lower():
                leaked.append(field)

        return leaked

    def _extract_admin_functions(self, response_body: str) -> List[str]:
        """提取管理功能"""
        admin_functions = []

        patterns = [
            r'(delete|remove|edit|modify|admin|manage)_\w+',
            r'/(admin|管理)/\w+',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_body, re.IGNORECASE)
            admin_functions.extend(matches)

        return list(set(admin_functions))

    def _detect_enumeration_pattern(self, resources: List[Dict]) -> str:
        """檢測枚舉模式"""
        if not resources:
            return "無明顯模式"

        ids = [r["id"] for r in resources]

        # 檢查是否為連續數字
        if all(id.isdigit() for id in ids):
            int_ids = sorted([int(id) for id in ids])
            if len(int_ids) > 1:
                diff = int_ids[1] - int_ids[0]
                if all(int_ids[i + 1] - int_ids[i] == diff for i in range(len(int_ids) - 1)):
                    return f"連續數字模式 (間隔 {diff})"

        return "可預測的 ID 模式"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """計算文本相似度（簡化版）"""
        if not text1 or not text2:
            return 0.0

        # 簡單的 Jaccard 相似度
        set1 = set(text1.split())
        set2 = set(text2.split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def generate_report(self, output_path: str) -> None:
        """生成測試報告"""
        report = {
            "summary": {
                "total_tests": len(self.findings),
                "vulnerable_tests": sum(1 for f in self.findings if f.vulnerable),
                "by_severity": self._count_by_severity(),
                "by_type": self._count_by_type(),
            },
            "findings": [
                {
                    "test_id": f.test_id,
                    "escalation_type": f.escalation_type.value,
                    "severity": f.severity,
                    "vulnerable": f.vulnerable,
                    "url": f.url,
                    "method": f.method,
                    "description": f.description,
                    "evidence": f.evidence,
                    "impact": f.impact,
                    "remediation": f.remediation,
                    "cvss_score": f.cvss_score,
                }
                for f in self.findings
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Report generated: {output_path}")

    def _count_by_severity(self) -> Dict[str, int]:
        """按嚴重性計數"""
        counts = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def _count_by_type(self) -> Dict[str, int]:
        """按類型計數"""
        counts = {}
        for finding in self.findings:
            type_name = finding.escalation_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts


# 使用範例
async def main():
    """主程序示例"""
    # 創建測試用戶
    attacker = TestUser(
        user_id="123",
        username="alice",
        role="user",
        token="attacker_token_here"
    )

    victim = TestUser(
        user_id="456",
        username="bob",
        role="user",
        token="victim_token_here"
    )

    admin = TestUser(
        user_id="789",
        username="admin",
        role="admin",
        token="admin_token_here"
    )

    # 開始測試
    async with PrivilegeEscalationTester("https://example.com") as tester:
        # 測試水平越權
        await tester.test_horizontal_escalation(
            attacker=attacker,
            victim=victim,
            target_url="https://example.com/api/user/profile?user_id=456"
        )

        # 測試垂直越權
        await tester.test_vertical_escalation(
            low_priv_user=attacker,
            high_priv_user=admin,
            admin_url="https://example.com/admin/users"
        )

        # 測試資源枚舉
        await tester.test_resource_enumeration(
            user=attacker,
            base_url="https://example.com/api/user/profile",
            id_param="user_id",
            id_range=(1, 100)
        )

        # 生成報告
        tester.generate_report("idor_test_report.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
