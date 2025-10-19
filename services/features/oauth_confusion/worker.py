# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urljoin, urlencode, urlparse, parse_qs
import random
import string
import time
from datetime import datetime
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

def random_string(length=12):
    """生成隨機字串"""
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# v2.5 新增：寬鬆的 3xx 狀態碼列表
RELAXED_REDIRECT_CODES = [301, 302, 303, 307, 308]

# v2.5 新增：PKCE 繞過技術列表
PKCE_BYPASS_TECHNIQUES = [
    {"name": "no_pkce", "omit": ["code_challenge", "code_challenge_method"]},
    {"name": "empty_challenge", "code_challenge": "", "code_challenge_method": "S256"},
    {"name": "wrong_method", "code_challenge": "fake_challenge", "code_challenge_method": "plain"},
    {"name": "null_challenge", "code_challenge": "null", "code_challenge_method": "S256"},
]

@FeatureRegistry.register
class OAuthConfusionWorker(FeatureBase):
    """
    OAuth 2.0 / OIDC 配置錯誤檢測模組 v2.5
    
    專門檢測 OAuth 2.0 和 OpenID Connect 實作中的常見安全漏洞，
    這些漏洞可能導致授權代碼或 access token 洩漏，進而實現帳戶接管。
    
    v2.5 新增功能：
    - Location header 反射檢測：測試響應頭中的 redirect_uri 反射
    - 寬鬆 302 標準：支持多種 3xx 重定向狀態碼檢測
    - PKCE 繞過鏈：多種 PKCE 繞過技術組合測試
    - 多步驟流程追蹤：完整的 OAuth 流程時間軸證據
    
    檢測的漏洞類型：
    1. redirect_uri 驗證不嚴格 - 允許開放重定向
    2. PKCE 未強制執行或可繞過 - 公共客戶端缺乏保護
    3. state 參數缺失或可預測 - CSRF 攻擊風險
    4. Open Redirect 鏈式攻擊 - 透過合法域名跳轉到惡意域名
    5. Location header 反射 - 響應頭洩露敏感信息
    6. 寬鬆重定向處理 - 接受非標準重定向狀態碼
    
    這些漏洞在實戰中命中率較高，且能提供明確的證據。
    """
    
    name = "oauth_confusion"
    version = "2.5.0"
    tags = ["oauth", "oidc", "redirect-uri", "pkce", "authorization-bypass"]
    
    def _test_location_header_reflection(
        self,
        http: SafeHttp,
        auth_url: str,
        client_id: str,
        redirect_uri: str,
        scope: str
    ) -> Optional[Dict[str, Any]]:
        """
        v2.5 新增：Location header 反射檢測
        測試服務器是否在響應頭中反射 redirect_uri
        
        Returns:
            如果發現反射，返回證據字典
        """
        test_marker = f"reflection_test_{random_string(8)}"
        test_redirect = f"{redirect_uri}?marker={test_marker}"
        
        try:
            query_params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": test_redirect,
                "scope": scope,
                "state": random_string(16)
            }
            
            response = http.request(
                "GET",
                f"{auth_url}?{urlencode(query_params)}",
                allow_redirects=False
            )
            
            # 檢查 Location header 是否包含我們的標記
            location = response.headers.get("Location", "")
            if test_marker in location:
                return {
                    "vulnerability": "location_header_reflection",
                    "test_marker": test_marker,
                    "location_header": location,
                    "status_code": response.status_code,
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception:
            pass
        
        return None
    
    def _test_relaxed_redirect_codes(
        self,
        http: SafeHttp,
        auth_url: str,
        client_id: str,
        redirect_uri: str,
        scope: str
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：寬鬆重定向狀態碼檢測
        測試服務器是否接受非標準的 3xx 狀態碼
        
        Returns:
            所有被接受的非標準狀態碼列表
        """
        accepted_codes = []
        
        try:
            query_params = {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "state": random_string(16)
            }
            
            response = http.request(
                "GET",
                f"{auth_url}?{urlencode(query_params)}",
                allow_redirects=False
            )
            
            # 檢查是否使用非標準的重定向狀態碼
            if response.status_code in RELAXED_REDIRECT_CODES:
                location = response.headers.get("Location", "")
                if location:
                    accepted_codes.append({
                        "status_code": response.status_code,
                        "location": location,
                        "is_standard": response.status_code == 302,
                        "timestamp": datetime.utcnow().isoformat()
                    })
        except Exception:
            pass
        
        return accepted_codes
    
    def _test_pkce_bypass_chain(
        self,
        http: SafeHttp,
        auth_url: str,
        client_id: str,
        redirect_uri: str,
        scope: str
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：PKCE 繞過鏈測試
        測試多種 PKCE 繞過技術
        
        Returns:
            成功的繞過技術列表
        """
        successful_bypasses = []
        
        for technique in PKCE_BYPASS_TECHNIQUES:
            try:
                query_params = {
                    "response_type": "code",
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "scope": scope,
                    "state": random_string(16)
                }
                
                # 應用繞過技術
                if technique["name"] != "no_pkce":
                    for key, value in technique.items():
                        if key != "name" and key != "omit":
                            query_params[key] = value
                
                response = http.request(
                    "GET",
                    f"{auth_url}?{urlencode(query_params)}",
                    allow_redirects=False
                )
                
                # 如果返回 200 或重定向，且沒有錯誤，說明繞過成功
                if response.status_code in (200, 302, 303) and "error" not in response.text.lower():
                    successful_bypasses.append({
                        "technique": technique["name"],
                        "status_code": response.status_code,
                        "parameters": {k: v for k, v in technique.items() if k != "name"},
                        "timestamp": datetime.utcnow().isoformat()
                    })
            except Exception:
                continue
        
        return successful_bypasses
    
    def _build_oauth_flow_timeline(
        self,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：OAuth 流程時間軸構建
        構建完整的 OAuth 授權流程時間軸
        
        Returns:
            包含完整流程的時間軸字典
        """
        return {
            "steps": steps,
            "total_steps": len(steps),
            "start_time": steps[0].get("timestamp") if steps else None,
            "end_time": steps[-1].get("timestamp") if steps else None,
            "duration_ms": self._calculate_flow_duration(steps)
        }
    
    def _calculate_flow_duration(self, steps: List[Dict[str, Any]]) -> float:
        """計算流程持續時間（毫秒）"""
        if len(steps) < 2:
            return 0.0
        
        try:
            start = datetime.fromisoformat(steps[0].get("timestamp", ""))
            end = datetime.fromisoformat(steps[-1].get("timestamp", ""))
            return (end - start).total_seconds() * 1000
        except:
            return 0.0

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 OAuth 混淆檢測 (v2.5)
        
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
                - location_header_reflection: v2.5 測試 Location header 反射
                - relaxed_redirect_codes: v2.5 測試寬鬆重定向狀態碼
                - pkce_bypass_chain: v2.5 測試 PKCE 繞過鏈
              
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
            "open_redirect_chain": True,
            "location_header_reflection": True,  # v2.5
            "relaxed_redirect_codes": True,  # v2.5
            "pkce_bypass_chain": True  # v2.5
        })
        
        findings: List[Finding] = []
        trace = []
        
        cmd = self.build_command_record(
            "oauth.confusion.v2.5",
            "OAuth/OIDC 配置錯誤檢測（增強版）",
            {
                "auth_endpoint": auth_endpoint,
                "client_id": client_id,
                "tests": list(tests.keys()),
                "v2.5_features": ["location_header_reflection", "relaxed_redirect_codes", "pkce_bypass_chain"]
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
        
        # v2.5 新增測試
        flow_timeline_steps = []
        
        # 4) Location header 反射檢測
        if tests.get("location_header_reflection", True) and legitimate_redirect:
            flow_timeline_steps.append({
                "stage": "location_header_reflection_test",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            reflection_vuln = self._test_location_header_reflection(
                http, auth_url, client_id, legitimate_redirect, scope
            )
            
            trace.append({
                "step": "location_header_reflection",
                "vulnerability_found": bool(reflection_vuln),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            if reflection_vuln:
                findings.append(Finding(
                    vuln_type="OAuth Location Header Reflection (v2.5)",
                    severity="medium",
                    title="Location header 反射 redirect_uri 參數",
                    evidence={
                        "test_marker": reflection_vuln["test_marker"],
                        "location_header": reflection_vuln["location_header"],
                        "status_code": reflection_vuln["status_code"],
                        "attack_timestamp": reflection_vuln["timestamp"]
                    },
                    reproduction=[
                        {
                            "step": 1,
                            "description": "發送帶有測試標記的 redirect_uri",
                            "request": {
                                "method": "GET",
                                "url": auth_url,
                                "params": {
                                    "client_id": client_id,
                                    "redirect_uri": f"{legitimate_redirect}?marker={reflection_vuln['test_marker']}"
                                }
                            },
                            "expect": "Location header 不應反射未驗證的參數"
                        }
                    ],
                    impact="Location header 反射可能導致 HTTP 標頭注入，攻擊者可構造惡意跳轉",
                    recommendation="1. 驗證並清理 redirect_uri 參數\n2. 使用預定義的重定向 URI\n3. 避免在響應頭中直接反射用戶輸入\n4. 實施嚴格的輸出編碼"
                ))
        
        # 5) 寬鬆重定向狀態碼檢測
        if tests.get("relaxed_redirect_codes", True) and legitimate_redirect:
            flow_timeline_steps.append({
                "stage": "relaxed_redirect_codes_test",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            redirect_codes = self._test_relaxed_redirect_codes(
                http, auth_url, client_id, legitimate_redirect, scope
            )
            
            non_standard_codes = [rc for rc in redirect_codes if not rc["is_standard"]]
            
            trace.append({
                "step": "relaxed_redirect_codes",
                "accepted_codes": [rc["status_code"] for rc in redirect_codes],
                "non_standard_count": len(non_standard_codes),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            if non_standard_codes:
                findings.append(Finding(
                    vuln_type="OAuth Non-Standard Redirect Codes (v2.5)",
                    severity="low",
                    title=f"使用非標準重定向狀態碼：{[rc['status_code'] for rc in non_standard_codes]}",
                    evidence={
                        "non_standard_codes": non_standard_codes,
                        "total_codes_tested": len(redirect_codes)
                    },
                    reproduction=[
                        {
                            "step": 1,
                            "description": "觀察授權流程的重定向狀態碼",
                            "request": {
                                "method": "GET",
                                "url": auth_url,
                                "params": {"client_id": client_id, "redirect_uri": legitimate_redirect}
                            },
                            "expect": f"返回標準的 302 狀態碼，實際返回：{non_standard_codes[0]['status_code']}"
                        }
                    ],
                    impact="使用非標準重定向狀態碼可能導致瀏覽器或代理解析不一致，增加安全風險",
                    recommendation="1. 統一使用 302 Found 作為重定向狀態碼\n2. 遵循 OAuth 2.0 規範\n3. 測試不同瀏覽器和代理的兼容性"
                ))
        
        # 6) PKCE 繞過鏈測試
        if tests.get("pkce_bypass_chain", True) and legitimate_redirect:
            flow_timeline_steps.append({
                "stage": "pkce_bypass_chain_test",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            pkce_bypasses = self._test_pkce_bypass_chain(
                http, auth_url, client_id, legitimate_redirect, scope
            )
            
            trace.append({
                "step": "pkce_bypass_chain",
                "successful_bypasses": len(pkce_bypasses),
                "techniques": [bp["technique"] for bp in pkce_bypasses],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            for bypass in pkce_bypasses:
                findings.append(Finding(
                    vuln_type="OAuth PKCE Bypass Chain (v2.5)",
                    severity="high",
                    title=f"PKCE 繞過技術成功：{bypass['technique']}",
                    evidence={
                        "bypass_technique": bypass["technique"],
                        "status_code": bypass["status_code"],
                        "parameters_used": bypass["parameters"],
                        "attack_timestamp": bypass["timestamp"]
                    },
                    reproduction=[
                        {
                            "step": 1,
                            "description": f"使用 {bypass['technique']} 技術繞過 PKCE",
                            "request": {
                                "method": "GET",
                                "url": auth_url,
                                "params": {
                                    "client_id": client_id,
                                    "redirect_uri": legitimate_redirect,
                                    **bypass["parameters"]
                                }
                            },
                            "expect": "應強制執行 PKCE 驗證，拒絕繞過嘗試"
                        }
                    ],
                    impact=f"使用 {bypass['technique']} 技術可繞過 PKCE 保護，攻擊者可攔截授權碼",
                    recommendation="1. 強制執行 PKCE（RFC 7636）\n2. 驗證 code_challenge 和 code_verifier\n3. 拒絕空值或無效的 PKCE 參數\n4. 對公共客戶端強制要求 PKCE\n5. 實施嚴格的參數驗證"
                ))
        
        # v2.5 構建完整的流程時間軸
        flow_timeline_steps.append({
            "stage": "completion",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        flow_timeline = self._build_oauth_flow_timeline(flow_timeline_steps)
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "version": "2.5.0",
                "trace": trace,
                "client_id": client_id,
                "tests_run": [k for k, v in tests.items() if v],
                "flow_timeline": flow_timeline,
                "v2.5_features": {
                    "location_header_reflection": len([f for f in findings if "Location Header" in f.vuln_type]),
                    "relaxed_redirect_codes": len([f for f in findings if "Non-Standard Redirect" in f.vuln_type]),
                    "pkce_bypass_chain": len([f for f in findings if "PKCE Bypass Chain" in f.vuln_type])
                }
            }
        )