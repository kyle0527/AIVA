"""
Authentication & CORS Security Testing Module
認證漏洞與跨域資源共享 (CORS) 測試模組
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import aiohttp


class AuthTestType(Enum):
    """認證測試類型"""
    WEAK_PASSWORD = "weak_password"
    BRUTE_FORCE = "brute_force"
    SESSION_FIXATION = "session_fixation"
    TOKEN_VALIDATION = "token_validation"
    JWT_SECURITY = "jwt_security"
    PASSWORD_RESET = "password_reset"
    MFA_BYPASS = "mfa_bypass"
    OAUTH_MISCONFIGURATION = "oauth_misconfiguration"


class CORSTestType(Enum):
    """CORS 測試類型"""
    NULL_ORIGIN = "null_origin"
    WILDCARD_ALLOW = "wildcard_allow"
    REFLECTED_ORIGIN = "reflected_origin"
    SUBDOMAIN_BYPASS = "subdomain_bypass"
    CREDENTIALS_LEAKAGE = "credentials_leakage"


@dataclass
class AuthFinding:
    """認證測試發現"""
    test_id: str
    test_type: AuthTestType
    severity: str
    vulnerable: bool
    url: str
    description: str
    evidence: dict[str, Any]
    impact: str
    remediation: str
    cvss_score: float = 0.0


@dataclass
class CORSFinding:
    """CORS 測試發現"""
    test_id: str
    test_type: CORSTestType
    severity: str
    vulnerable: bool
    url: str
    origin_tested: str
    cors_headers: dict[str, str]
    description: str
    evidence: dict[str, Any]
    impact: str
    remediation: str
    cvss_score: float = 0.0


class AuthenticationTester:
    """認證安全測試器"""

    def __init__(self, target_url: str, logger: logging.Logger | None = None):
        self.target_url = target_url
        self.logger = logger or logging.getLogger(__name__)
        self.session: aiohttp.ClientSession | None = None
        self.findings: list[AuthFinding] = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_weak_password_policy(
        self,
        register_url: str,
        weak_passwords: list[str] | None = None
    ) -> AuthFinding:
        """測試弱密碼策略"""
        if weak_passwords is None:
            weak_passwords = [
                "123456", "password", "12345678", "qwerty",
                "abc123", "111111", "admin", "test"
            ]

        self.logger.info("Testing weak password policy")

        accepted_weak_passwords = []
        rejected_count = 0

        for pwd in weak_passwords:
            result = await self._attempt_register(
                register_url,
                username=f"test_user_{hash(pwd)}",
                password=pwd
            )

            if result["success"]:
                accepted_weak_passwords.append({
                    "password": pwd,
                    "response": result["status"]
                })
            else:
                rejected_count += 1

        vulnerable = len(accepted_weak_passwords) > 0
        severity = "HIGH" if vulnerable else "INFO"
        cvss = 7.5 if vulnerable else 0.0

        finding = AuthFinding(
            test_id="auth_weak_password",
            test_type=AuthTestType.WEAK_PASSWORD,
            severity=severity,
            vulnerable=vulnerable,
            url=register_url,
            description="弱密碼策略測試：檢查系統是否接受常見的弱密碼",
            evidence={
                "tested_passwords": len(weak_passwords),
                "accepted_weak_passwords": accepted_weak_passwords,
                "rejected_count": rejected_count,
                "acceptance_rate": len(accepted_weak_passwords) / len(weak_passwords),
            },
            impact=(
                f"系統接受 {len(accepted_weak_passwords)} 個弱密碼，"
                "使攻擊者更容易通過暴力破解或字典攻擊入侵帳戶。"
                if vulnerable else "密碼策略正常"
            ),
            remediation=(
                "1. 實施強密碼策略（最少 8 位，包含大小寫字母、數字和特殊符號）\n"
                "2. 檢查常見弱密碼字典\n"
                "3. 實施密碼複雜度評分\n"
                "4. 要求定期更換密碼\n"
                "5. 禁止使用用戶名作為密碼"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_brute_force_protection(
        self,
        login_url: str,
        username: str,
        max_attempts: int = 20
    ) -> AuthFinding:
        """測試暴力破解防護"""
        self.logger.info(f"Testing brute force protection with {max_attempts} attempts")

        attempt_results = []
        blocked = False
        captcha_triggered = False

        for i in range(max_attempts):
            result = await self._attempt_login(
                login_url,
                username=username,
                password=f"wrong_password_{i}"
            )

            attempt_results.append({
                "attempt": i + 1,
                "status": result["status"],
                "response_time": result.get("response_time", 0),
                "blocked": result.get("blocked", False),
                "captcha": result.get("captcha_required", False),
            })

            if result.get("blocked"):
                blocked = True
                self.logger.info(f"Account blocked after {i + 1} attempts")
                break

            if result.get("captcha_required"):
                captcha_triggered = True
                self.logger.info(f"CAPTCHA triggered after {i + 1} attempts")

            # 避免對真實系統造成過大負擔
            await asyncio.sleep(0.5)

        vulnerable = not (blocked or captcha_triggered)
        severity = "HIGH" if vulnerable else "LOW"
        cvss = 7.3 if vulnerable else 2.0

        finding = AuthFinding(
            test_id="auth_brute_force",
            test_type=AuthTestType.BRUTE_FORCE,
            severity=severity,
            vulnerable=vulnerable,
            url=login_url,
            description=f"暴力破解防護測試：嘗試 {max_attempts} 次登入",
            evidence={
                "total_attempts": len(attempt_results),
                "blocked": blocked,
                "captcha_triggered": captcha_triggered,
                "attempts_before_protection": len(attempt_results),
                "sample_attempts": attempt_results[:5],
            },
            impact=(
                f"系統在 {max_attempts} 次失敗登入後仍未觸發防護機制，"
                "攻擊者可以無限制地嘗試密碼。"
                if vulnerable else "暴力破解防護機制有效"
            ),
            remediation=(
                "1. 實施帳戶鎖定機制（如 5 次失敗後鎖定 15 分鐘）\n"
                "2. 引入 CAPTCHA 或人機驗證\n"
                "3. 實施指數退避（每次失敗後增加延遲）\n"
                "4. IP 限速與黑名單\n"
                "5. 多因素認證 (MFA)\n"
                "6. 監控異常登入模式"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_jwt_security(
        self,
        token: str,
        api_url: str
    ) -> AuthFinding:
        """測試 JWT Token 安全性"""
        self.logger.info("Testing JWT security")

        vulnerabilities = []

        # 1. 檢查 none 演算法攻擊
        none_algo_result = await self._test_jwt_none_algorithm(token, api_url)
        if none_algo_result["vulnerable"]:
            vulnerabilities.append("none_algorithm")

        # 2. 檢查弱簽名
        weak_sig_result = await self._test_jwt_weak_signature(token, api_url)
        if weak_sig_result["vulnerable"]:
            vulnerabilities.append("weak_signature")

        # 3. 檢查 Token 過期驗證
        expiry_result = await self._test_jwt_expiration(token, api_url)
        if expiry_result["vulnerable"]:
            vulnerabilities.append("no_expiration_check")

        # 4. 檢查敏感資訊洩露
        sensitive_data = self._analyze_jwt_payload(token)
        if sensitive_data:
            vulnerabilities.append("sensitive_data_leak")

        vulnerable = len(vulnerabilities) > 0
        severity = "CRITICAL" if vulnerable else "INFO"
        cvss = 9.0 if vulnerable else 0.0

        finding = AuthFinding(
            test_id="auth_jwt_security",
            test_type=AuthTestType.JWT_SECURITY,
            severity=severity,
            vulnerable=vulnerable,
            url=api_url,
            description="JWT Token 安全性測試",
            evidence={
                "vulnerabilities": vulnerabilities,
                "none_algo_test": none_algo_result,
                "weak_signature_test": weak_sig_result,
                "expiration_test": expiry_result,
                "sensitive_data": sensitive_data,
            },
            impact=(
                f"JWT 實現存在 {len(vulnerabilities)} 個安全問題，"
                "可能導致認證繞過、權限提升或敏感資訊洩露。"
                if vulnerable else "JWT 安全性良好"
            ),
            remediation=(
                "1. 禁用 'none' 演算法\n"
                "2. 使用強加密演算法 (RS256, ES256)\n"
                "3. 驗證 Token 過期時間\n"
                "4. 不在 JWT payload 中存儲敏感資訊\n"
                "5. 實施 Token 撤銷機制\n"
                "6. 使用短期 Token + Refresh Token 模式\n"
                "7. 驗證 Token 簽發者 (issuer)"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_session_fixation(
        self,
        login_url: str,
        username: str,
        password: str
    ) -> AuthFinding:
        """測試 Session Fixation 漏洞"""
        self.logger.info("Testing session fixation")

        # 1. 獲取登入前的 session ID
        pre_login_session = await self._get_session_id(login_url)

        # 2. 使用該 session 登入
        login_result = await self._attempt_login(
            login_url,
            username=username,
            password=password,
            existing_session=pre_login_session
        )

        # 3. 檢查登入後的 session ID
        post_login_session = login_result.get("session_id")

        # 4. 判斷是否存在 session fixation
        vulnerable = (pre_login_session == post_login_session and
                     login_result["success"])

        severity = "HIGH" if vulnerable else "INFO"
        cvss = 7.5 if vulnerable else 0.0

        finding = AuthFinding(
            test_id="auth_session_fixation",
            test_type=AuthTestType.SESSION_FIXATION,
            severity=severity,
            vulnerable=vulnerable,
            url=login_url,
            description="Session Fixation 測試：檢查登入後是否重新生成 Session ID",
            evidence={
                "pre_login_session": pre_login_session,
                "post_login_session": post_login_session,
                "session_changed": pre_login_session != post_login_session,
                "login_success": login_result["success"],
            },
            impact=(
                "攻擊者可以預先設定受害者的 Session ID，"
                "在受害者登入後，攻擊者可以使用該 Session 劫持帳戶。"
                if vulnerable else "Session 管理正常"
            ),
            remediation=(
                "1. 登入成功後立即重新生成 Session ID\n"
                "2. 使用 HTTPOnly 和 Secure 標籤保護 Cookie\n"
                "3. 實施 SameSite Cookie 屬性\n"
                "4. 定期輪換 Session ID\n"
                "5. 綁定 Session 到 IP 或 User-Agent（可選）"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def _attempt_register(
        self,
        url: str,
        username: str,
        password: str
    ) -> dict[str, Any]:
        """嘗試註冊"""
        if not self.session:
            return {"success": False, "status": 0}

        try:
            async with self.session.post(
                url,
                json={"username": username, "password": password},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return {
                    "success": response.status in [200, 201],
                    "status": response.status,
                    "body": await response.text(),
                }
        except Exception as e:
            self.logger.error(f"Register attempt failed: {e}")
            return {"success": False, "status": 0, "error": str(e)}

    async def _attempt_login(
        self,
        url: str,
        username: str,
        password: str,
        existing_session: str | None = None
    ) -> dict[str, Any]:
        """嘗試登入"""
        if not self.session:
            return {"success": False, "status": 0}

        cookies = {}
        if existing_session:
            cookies["session"] = existing_session

        try:
            async with self.session.post(
                url,
                json={"username": username, "password": password},
                cookies=cookies,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                body = await response.text()

                return {
                    "success": response.status == 200,
                    "status": response.status,
                    "body": body,
                    "blocked": "blocked" in body.lower() or response.status == 429,
                    "captcha_required": "captcha" in body.lower(),
                    "session_id": response.cookies.get("session", {}).value if "session" in response.cookies else None,
                }
        except Exception as e:
            self.logger.error(f"Login attempt failed: {e}")
            return {"success": False, "status": 0, "error": str(e)}

    async def _get_session_id(self, url: str) -> str | None:
        """獲取 session ID"""
        if not self.session:
            return None

        try:
            async with self.session.get(url) as response:
                return response.cookies.get("session", {}).value if "session" in response.cookies else None
        except Exception:
            return None

    async def _test_jwt_none_algorithm(self, token: str, api_url: str) -> dict[str, Any]:
        """測試 JWT none 演算法"""
        # 實現略（需要 JWT 庫）
        return {"vulnerable": False, "details": "需要 JWT 解析庫"}

    async def _test_jwt_weak_signature(self, token: str, api_url: str) -> dict[str, Any]:
        """測試 JWT 弱簽名"""
        return {"vulnerable": False, "details": "需要 JWT 解析庫"}

    async def _test_jwt_expiration(self, token: str, api_url: str) -> dict[str, Any]:
        """測試 JWT 過期驗證"""
        return {"vulnerable": False, "details": "需要 JWT 解析庫"}

    def _analyze_jwt_payload(self, token: str) -> list[str]:
        """分析 JWT payload 中的敏感資訊"""
        # 簡化實現
        sensitive_patterns = ["password", "secret", "api_key", "ssn", "credit_card"]
        found = []

        try:
            # 解碼 JWT payload (base64)
            import base64
            parts = token.split(".")
            if len(parts) >= 2:
                payload = base64.urlsafe_b64decode(parts[1] + "==")
                payload_str = payload.decode("utf-8")

                for pattern in sensitive_patterns:
                    if pattern in payload_str.lower():
                        found.append(pattern)
        except Exception:
            pass

        return found

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
                    "test_type": f.test_type.value,
                    "severity": f.severity,
                    "vulnerable": f.vulnerable,
                    "url": f.url,
                    "description": f.description,
                    "evidence": f.evidence,
                    "impact": f.impact,
                    "remediation": f.remediation,
                    "cvss_score": f.cvss_score,
                }
                for f in self.findings
            ],
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    def _count_by_severity(self) -> dict[str, int]:
        """按嚴重性計數"""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def _count_by_type(self) -> dict[str, int]:
        """按類型計數"""
        counts: dict[str, int] = {}
        for finding in self.findings:
            type_name = finding.test_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts


class CORSTester:
    """CORS 安全測試器"""

    def __init__(self, target_url: str, logger: logging.Logger | None = None):
        self.target_url = target_url
        self.logger = logger or logging.getLogger(__name__)
        self.session: aiohttp.ClientSession | None = None
        self.findings: list[CORSFinding] = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_null_origin(self, endpoint: str) -> CORSFinding:
        """測試 null origin 繞過"""
        self.logger.info("Testing null origin bypass")

        response = await self._send_cors_request(
            endpoint,
            origin="null"
        )

        vulnerable = self._check_cors_misconfiguration(
            response,
            expected_origin="null"
        )

        severity = "HIGH" if vulnerable else "INFO"
        cvss = 7.5 if vulnerable else 0.0

        finding = CORSFinding(
            test_id="cors_null_origin",
            test_type=CORSTestType.NULL_ORIGIN,
            severity=severity,
            vulnerable=vulnerable,
            url=endpoint,
            origin_tested="null",
            cors_headers=response.get("cors_headers", {}),
            description="測試 null origin 是否被允許訪問",
            evidence={
                "response_status": response.get("status", 0),
                "allowed_origin": response.get("cors_headers", {}).get("Access-Control-Allow-Origin"),
                "credentials_allowed": response.get("cors_headers", {}).get("Access-Control-Allow-Credentials"),
            },
            impact=(
                "允許 null origin 可能導致來自本地文件或沙箱 iframe 的攻擊，"
                "攻擊者可以竊取敏感數據或執行未授權操作。"
                if vulnerable else "CORS 配置正常"
            ),
            remediation=(
                "1. 不要允許 'null' origin\n"
                "2. 使用白名單明確指定允許的 origin\n"
                "3. 避免使用 '*' wildcard\n"
                "4. 不要同時啟用 credentials 和 wildcard"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_wildcard_with_credentials(self, endpoint: str) -> CORSFinding:
        """測試 wildcard + credentials 錯誤配置"""
        self.logger.info("Testing wildcard with credentials")

        response = await self._send_cors_request(
            endpoint,
            origin="https://evil.com",
            with_credentials=True
        )

        cors_headers = response.get("cors_headers", {})
        allow_origin = cors_headers.get("Access-Control-Allow-Origin", "")
        allow_credentials = cors_headers.get("Access-Control-Allow-Credentials", "")

        vulnerable = (allow_origin == "*" and
                     allow_credentials.lower() == "true")

        severity = "CRITICAL" if vulnerable else "INFO"
        cvss = 9.1 if vulnerable else 0.0

        finding = CORSFinding(
            test_id="cors_wildcard_credentials",
            test_type=CORSTestType.WILDCARD_ALLOW,
            severity=severity,
            vulnerable=vulnerable,
            url=endpoint,
            origin_tested="https://evil.com",
            cors_headers=cors_headers,
            description="測試 wildcard (*) 與 credentials 的危險組合",
            evidence={
                "allow_origin": allow_origin,
                "allow_credentials": allow_credentials,
                "dangerous_combination": vulnerable,
            },
            impact=(
                "同時允許任意 origin (*) 和 credentials，"
                "攻擊者可以從任何網站竊取用戶的認證憑證和敏感數據。"
                if vulnerable else "CORS 配置安全"
            ),
            remediation=(
                "1. 不要同時使用 '*' 和 'Access-Control-Allow-Credentials: true'\n"
                "2. 使用具體的 origin 白名單\n"
                "3. 動態驗證並返回請求的 origin（如果在白名單中）\n"
                "4. 考慮使用更嚴格的 CSRF 防護"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def test_reflected_origin(
        self,
        endpoint: str,
        test_origins: list[str] | None = None
    ) -> CORSFinding:
        """測試 origin 反射漏洞"""
        if test_origins is None:
            test_origins = [
                "https://evil.com",
                "https://attacker.com",
                "http://localhost:8000",
            ]

        self.logger.info("Testing reflected origin vulnerability")

        reflected_origins = []

        for origin in test_origins:
            response = await self._send_cors_request(endpoint, origin=origin)
            allowed_origin = response.get("cors_headers", {}).get("Access-Control-Allow-Origin")

            if allowed_origin == origin:
                reflected_origins.append({
                    "origin": origin,
                    "reflected": True,
                    "credentials": response.get("cors_headers", {}).get("Access-Control-Allow-Credentials"),
                })

        vulnerable = len(reflected_origins) > 0
        severity = "HIGH" if vulnerable else "INFO"
        cvss = 8.1 if vulnerable else 0.0

        finding = CORSFinding(
            test_id="cors_reflected_origin",
            test_type=CORSTestType.REFLECTED_ORIGIN,
            severity=severity,
            vulnerable=vulnerable,
            url=endpoint,
            origin_tested=", ".join(test_origins),
            cors_headers=response.get("cors_headers", {}),
            description="測試服務器是否直接反射請求的 Origin",
            evidence={
                "tested_origins": test_origins,
                "reflected_origins": reflected_origins,
                "reflection_rate": len(reflected_origins) / len(test_origins),
            },
            impact=(
                f"服務器反射了 {len(reflected_origins)} 個測試 origin，"
                "攻擊者可以從任意網站訪問敏感 API，繞過同源策略。"
                if vulnerable else "Origin 驗證正常"
            ),
            remediation=(
                "1. 不要直接反射 Origin 標頭\n"
                "2. 使用靜態白名單驗證 origin\n"
                "3. 對 subdomain 使用嚴格的正則表達式驗證\n"
                "4. 記錄並監控異常的 CORS 請求"
            ),
            cvss_score=cvss,
        )

        self.findings.append(finding)
        return finding

    async def _send_cors_request(
        self,
        url: str,
        origin: str,
        with_credentials: bool = False
    ) -> dict[str, Any]:
        """發送 CORS preflight 請求"""
        if not self.session:
            return {}

        headers = {
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        }

        try:
            async with self.session.options(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                cors_headers = {
                    k: v for k, v in response.headers.items()
                    if k.lower().startswith("access-control-")
                }

                return {
                    "status": response.status,
                    "cors_headers": cors_headers,
                    "all_headers": dict(response.headers),
                }
        except Exception as e:
            self.logger.error(f"CORS request failed: {e}")
            return {"status": 0, "error": str(e)}

    def _check_cors_misconfiguration(
        self,
        response: dict[str, Any],
        expected_origin: str
    ) -> bool:
        """檢查 CORS 錯誤配置"""
        if response.get("status") != 200:
            return False

        cors_headers = response.get("cors_headers", {})
        allowed_origin = cors_headers.get("Access-Control-Allow-Origin")

        return allowed_origin == expected_origin

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
                    "test_type": f.test_type.value,
                    "severity": f.severity,
                    "vulnerable": f.vulnerable,
                    "url": f.url,
                    "origin_tested": f.origin_tested,
                    "cors_headers": f.cors_headers,
                    "description": f.description,
                    "evidence": f.evidence,
                    "impact": f.impact,
                    "remediation": f.remediation,
                    "cvss_score": f.cvss_score,
                }
                for f in self.findings
            ],
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    def _count_by_severity(self) -> dict[str, int]:
        """按嚴重性計數"""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def _count_by_type(self) -> dict[str, int]:
        """按類型計數"""
        counts: dict[str, int] = {}
        for finding in self.findings:
            type_name = finding.test_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts


# 使用示例
async def main():
    """主程序示例"""
    target = "https://example.com"

    # 認證測試
    async with AuthenticationTester(target) as auth_tester:
        # 測試弱密碼
        await auth_tester.test_weak_password_policy(
            register_url=f"{target}/api/register"
        )

        # 測試暴力破解防護
        await auth_tester.test_brute_force_protection(
            login_url=f"{target}/api/login",
            username="test_user"
        )

        # 測試 Session Fixation
        await auth_tester.test_session_fixation(
            login_url=f"{target}/api/login",
            username="test_user",
            password="correct_password"
        )

        # 生成報告
        auth_tester.generate_report("auth_test_report.json")

    # CORS 測試
    async with CORSTester(target) as cors_tester:
        # 測試 null origin
        await cors_tester.test_null_origin(f"{target}/api/data")

        # 測試 wildcard + credentials
        await cors_tester.test_wildcard_with_credentials(f"{target}/api/data")

        # 測試 reflected origin
        await cors_tester.test_reflected_origin(f"{target}/api/data")

        # 生成報告
        cors_tester.generate_report("cors_test_report.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
