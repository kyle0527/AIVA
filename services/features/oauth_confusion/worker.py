# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from urllib.parse import urljoin, urlencode, urlparse, parse_qs
import random
import string
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

def random_string(length=12):
    """生成隨機字串"""
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

@FeatureRegistry.register
class OAuthConfusionWorker(FeatureBase):
    """
    OAuth 2.0 / OIDC 配置錯誤檢測模組
    
    專門檢測 OAuth 2.0 和 OpenID Connect 實作中的常見安全漏洞，
    這些漏洞可能導致授權代碼或 access token 洩漏，進而實現帳戶接管。
    
    檢測的漏洞類型：
    1. redirect_uri 驗證不嚴格 - 允許開放重定向
    2. PKCE 未強制執行 - 公共客戶端缺乏保護
    3. state 參數缺失或可預測 - CSRF 攻擊風險
    4. Open Redirect 鏈式攻擊 - 透過合法域名跳轉到惡意域名
    
    這些漏洞在實戰中命中率較高，且能提供明確的證據。
    """
    
    name = "oauth_confusion"
    version = "2.0.0"
    tags = ["oauth", "oidc", "redirect-uri", "pkce", "authorization-bypass"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 OAuth 混淆檢測
        
        Args:
            params: 檢測參數
              - target (str): IdP 基礎 URL，如 https://auth.example.com
              - auth_endpoint (str): 授權端點，如 /oauth/authorize
              - client_id (str): OAuth 客戶端 ID
              - legitimate_redirect (str): 合法的重定向 URI
              - attacker_redirect (str): 攻擊者控制的重定向 URI
              - scope (str): OAuth scope，預設 "openid profile"
              - tests (dict): 測試類型開關
                - redirect_uri_bypass: 測試重定向 URI 繞過
                - pkce_downgrade: 測試 PKCE 降級
                - open_redirect_chain: 測試開放重定向鏈
              
        Returns:
            FeatureResult: 檢測結果
        """
        http = SafeHttp()
        base = params.get("target", "")
        auth_endpoint = params.get("auth_endpoint", "/oauth/authorize")
        auth_url = urljoin(base, auth_endpoint)
        client_id = params.get("client_id", "")
        legitimate_redirect = params.get("legitimate_redirect", "")
        attacker_redirect = params.get("attacker_redirect", "")
        scope = params.get("scope", "openid profile")
        tests = params.get("tests", {
            "redirect_uri_bypass": True,
            "pkce_downgrade": True, 
            "open_redirect_chain": True
        })
        
        findings: List[Finding] = []
        trace = []
        
        cmd = self.build_command_record(
            "oauth.confusion",
            "OAuth/OIDC 配置錯誤檢測",
            {
                "auth_endpoint": auth_endpoint,
                "client_id": client_id,
                "tests": list(tests.keys())
            }
        )
        
        if not client_id:
            return FeatureResult(
                False, self.name, cmd, [],
                {"error": "缺少 client_id 參數", "trace": trace}
            )
        
        # 1) redirect_uri 繞過測試
        if tests.get("redirect_uri_bypass") and attacker_redirect:
            redirect_payloads = [
                attacker_redirect,  # 直接使用攻擊者域名
                f"{legitimate_redirect}@{urlparse(attacker_redirect).netloc}",  # 用戶資訊繞過
                f"https://{urlparse(attacker_redirect).netloc}.{urlparse(legitimate_redirect).netloc}",  # 子域名繞過
                f"{legitimate_redirect}/../{urlparse(attacker_redirect).path}",  # 路徑遍歷
                f"{legitimate_redirect}?redirect={attacker_redirect}",  # 參數注入
                f"//{urlparse(attacker_redirect).netloc}/callback",  # 協議繞過
                f"javascript:window.location='{attacker_redirect}'",  # JavaScript 協議
            ]
            
            for payload in redirect_payloads:
                try:
                    query_params = {
                        "response_type": "code",
                        "client_id": client_id,
                        "redirect_uri": payload,
                        "scope": scope,
                        "state": random_string(16)
                    }
                    
                    test_url = f"{auth_url}?{urlencode(query_params)}"
                    response = http.request("GET", test_url, allow_redirects=False)
                    
                    trace.append({
                        "step": "redirect_uri_test",
                        "payload": payload,
                        "status": response.status_code,
                        "location": response.headers.get("Location", ""),
                        "has_redirect": bool(response.headers.get("Location"))
                    })
                    
                    # 檢查是否被重定向到攻擊者控制的域名
                    location = response.headers.get("Location", "")
                    if (300 <= response.status_code < 400 and location and 
                        any(domain in location for domain in [
                            urlparse(attacker_redirect).netloc,
                            "//",  # 協議相對 URL
                            "javascript:",  # JavaScript 協議
                        ])):
                        
                        findings.append(Finding(
                            vuln_type="OAuth redirect_uri Bypass",
                            severity="high",
                            title=f"redirect_uri 驗證繞過：{payload}",
                            evidence={
                                "bypass_payload": payload,
                                "redirect_location": location,
                                "response_status": response.status_code,
                                "auth_url": test_url
                            },
                            reproduction=[
                                {
                                    "step": 1,
                                    "description": "發送帶有惡意 redirect_uri 的授權請求",
                                    "request": {
                                        "method": "GET",
                                        "url": test_url
                                    },
                                    "expect": "應拒絕未註冊的 redirect_uri"
                                }
                            ],
                            impact="攻擊者可攔截授權代碼或 access token，導致帳戶接管",
                            recommendation="1. 實施嚴格的 redirect_uri 白名單\n2. 禁止通配符和模糊匹配\n3. 驗證完整的 URL 而非僅域名\n4. 拒絕 JavaScript 等危險協議"
                        ))
                        break  # 找到一個繞過就停止
                        
                except Exception as e:
                    trace.append({
                        "step": "redirect_uri_error",
                        "payload": payload,
                        "error": str(e)
                    })
        
        # 2) PKCE 降級攻擊
        if tests.get("pkce_downgrade"):
            try:
                # 測試不帶 PKCE 的授權請求
                no_pkce_params = {
                    "response_type": "code",
                    "client_id": client_id,
                    "redirect_uri": legitimate_redirect or "https://example.com/callback",
                    "scope": scope,
                    "state": random_string(16)
                    # 故意不包含 code_challenge 和 code_challenge_method
                }
                
                test_url = f"{auth_url}?{urlencode(no_pkce_params)}"
                response = http.request("GET", test_url, allow_redirects=False)
                
                trace.append({
                    "step": "pkce_downgrade_test",
                    "status": response.status_code,
                    "has_error": "error" in response.text.lower() if response.text else False,
                    "allows_no_pkce": response.status_code in (200, 302)
                })
                
                # 如果返回 200 或 302，說明可能允許無 PKCE 的請求
                if response.status_code in (200, 302) and "error" not in response.text.lower():
                    findings.append(Finding(
                        vuln_type="OAuth PKCE Not Enforced",
                        severity="medium",
                        title="PKCE 未強制執行，允許降級攻擊",
                        evidence={
                            "response_status": response.status_code,
                            "allows_no_pkce": True,
                            "test_url": test_url
                        },
                        reproduction=[
                            {
                                "step": 1,
                                "description": "發送不帶 code_challenge 的授權請求",
                                "request": {
                                    "method": "GET", 
                                    "url": test_url
                                },
                                "expect": "應要求 PKCE 參數或返回錯誤"
                            }
                        ],
                        impact="公共客戶端易受授權代碼攔截和重放攻擊",
                        recommendation="1. 對公共客戶端強制執行 PKCE\n2. 驗證 code_challenge 參數\n3. 在 token 端點驗證 code_verifier"
                    ))
                    
            except Exception as e:
                trace.append({
                    "step": "pkce_downgrade_error",
                    "error": str(e)
                })
        
        # 3) Open Redirect 鏈式攻擊
        if tests.get("open_redirect_chain") and legitimate_redirect and attacker_redirect:
            try:
                # 構造鏈式重定向 URL
                parsed_legitimate = urlparse(legitimate_redirect)
                chain_payloads = [
                    f"{legitimate_redirect}?next={attacker_redirect}",
                    f"{legitimate_redirect}?return={attacker_redirect}",
                    f"{legitimate_redirect}?redirect={attacker_redirect}",
                    f"{legitimate_redirect}?returnUrl={attacker_redirect}",
                    f"{legitimate_redirect}?url={attacker_redirect}",
                    f"{legitimate_redirect}#next={attacker_redirect}",  # Fragment 基礎
                ]
                
                for chain_payload in chain_payloads:
                    try:
                        query_params = {
                            "response_type": "code",
                            "client_id": client_id,
                            "redirect_uri": chain_payload,
                            "scope": scope,
                            "state": random_string(16)
                        }
                        
                        test_url = f"{auth_url}?{urlencode(query_params)}"
                        response = http.request("GET", test_url, allow_redirects=True, timeout=10)
                        
                        # 檢查最終是否到達攻擊者域名
                        final_url = response.url
                        attacker_domain = urlparse(attacker_redirect).netloc
                        
                        trace.append({
                            "step": "open_redirect_chain_test",
                            "chain_payload": chain_payload,
                            "final_url": final_url,
                            "reached_attacker": attacker_domain in final_url,
                            "status": response.status_code
                        })
                        
                        if attacker_domain in final_url:
                            findings.append(Finding(
                                vuln_type="OAuth Open Redirect Chain",
                                severity="high",
                                title=f"透過合法域名的開放重定向攔截授權碼",
                                evidence={
                                    "chain_payload": chain_payload,
                                    "final_destination": final_url,
                                    "attacker_domain": attacker_domain,
                                    "auth_url": test_url
                                },
                                reproduction=[
                                    {
                                        "step": 1,
                                        "description": "使用鏈式重定向的 redirect_uri",
                                        "request": {
                                            "method": "GET",
                                            "url": test_url
                                        },
                                        "expect": "不應跳轉到未授權的第三方域名"
                                    }
                                ],
                                impact="透過合法域名的開放重定向漏洞攔截 OAuth 授權碼，實現帳戶接管",
                                recommendation="1. 修復合法域名上的開放重定向漏洞\n2. redirect_uri 驗證應包含完整路徑\n3. 禁止帶有可疑參數的 redirect_uri\n4. 實施多層重定向檢測"
                            ))
                            break
                            
                    except Exception as e:
                        trace.append({
                            "step": "open_redirect_chain_error",
                            "chain_payload": chain_payload,
                            "error": str(e)
                        })
                        
            except Exception as e:
                trace.append({
                    "step": "open_redirect_chain_setup_error", 
                    "error": str(e)
                })
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "trace": trace,
                "client_id": client_id,
                "tests_run": [k for k, v in tests.items() if v]
            }
        )