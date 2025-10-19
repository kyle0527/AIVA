# -*- coding: utf-8 -*-
"""
Email Change Bypass 攻擊檢測模組

專門檢測透過各種手段繞過電子郵件變更驗證流程的漏洞，
這類漏洞可能導致帳戶接管，在 Bug Bounty 中屬於高嚴重性漏洞。
"""
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlencode
import random
import string
import time
import json
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

def random_email(domain="example.com"):
    """生成隨機電子郵件地址"""
    prefix = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
    return f"{prefix}@{domain}"

@FeatureRegistry.register
class EmailChangeBypassWorker(FeatureBase):
    """
    Email Change Bypass 檢測
    
    檢測原理：
    正常的電子郵件變更流程應該包含：
    1. 發送驗證連結到新電子郵件
    2. 用戶點擊連結確認
    3. 系統驗證 token 有效性
    4. 更新電子郵件地址
    
    常見的繞過手法：
    
    1. Token 重用漏洞：
       - 同一個 token 可以多次使用
       - token 不會在使用後失效
       
    2. Token 爆破漏洞：
       - token 格式可預測（如自增 ID）
       - 沒有速率限制
       - token 長度不足
       
    3. 競態條件漏洞：
       - 同時發送多個變更請求
       - token 驗證和更新操作不是原子性的
       
    4. 參數污染漏洞：
       - 發送多個 email 參數
       - 驗證使用第一個，更新使用最後一個
       
    5. Host Header 攻擊：
       - 修改 Host 標頭導致驗證連結發送到攻擊者控制的域名
       
    6. Token 洩漏漏洞：
       - token 在 URL 中傳遞
       - token 在 Referer 中洩漏
       - token 記錄在日誌中
       
    這些漏洞在 Bug Bounty 平台屬於 High/Critical 級別。
    """
    
    name = "email_change_bypass"
    version = "1.0.0"
    tags = ["email-change", "account-takeover", "token-bypass", "race-condition", "high-severity"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 Email Change Bypass 檢測
        
        Args:
            params: 檢測參數
              - target (str): 目標基礎 URL
              - change_endpoint (str): 電子郵件變更請求端點
              - verify_endpoint (str): 驗證 token 端點
              - session_token (str): 已登入的 session token
              - current_email (str): 當前電子郵件地址
              - new_email (str): 新的電子郵件地址（攻擊者控制）
              - headers (dict): 額外的 HTTP 標頭
              - method (str): HTTP 方法，預設 POST
              - email_param (str): 電子郵件參數名稱，預設 "email"
              - token_param (str): Token 參數名稱，預設 "token"
              - enable_race_condition (bool): 是否測試競態條件，預設 True
              - enable_parameter_pollution (bool): 是否測試參數污染，預設 True
              - enable_host_header (bool): 是否測試 Host Header 攻擊，預設 True
              
        Returns:
            FeatureResult: 檢測結果，包含發現的漏洞和證據
        """
        http = SafeHttp()
        base = params.get("target", "")
        change_ep = urljoin(base, params.get("change_endpoint", "/api/user/email/change"))
        verify_ep = urljoin(base, params.get("verify_endpoint", "/api/user/email/verify"))
        session_token = params.get("session_token", "")
        current_email = params.get("current_email", "")
        new_email = params.get("new_email", "")
        method = params.get("method", "POST").upper()
        email_param = params.get("email_param", "email")
        token_param = params.get("token_param", "token")
        
        enable_race = params.get("enable_race_condition", True)
        enable_param_pollution = params.get("enable_parameter_pollution", True)
        enable_host_header = params.get("enable_host_header", True)
        
        headers = params.get("headers", {}).copy()
        if session_token:
            headers["Authorization"] = f"Bearer {session_token}"
        
        findings: List[Finding] = []
        trace = []
        
        if not all([base, change_ep, session_token, new_email]):
            cmd = self.build_command_record(
                command="email.change.bypass",
                description="Email change bypass detection (missing params)",
                parameters={"error": "Missing required parameters"}
            )
            return FeatureResult(
                ok=False,
                feature=self.name,
                command_record=cmd,
                findings=[],
                meta={"error": "Missing required parameters: target, change_endpoint, session_token, new_email"}
            )
        
        # 測試 1: Token 重用漏洞
        token_reuse_result = self._test_token_reuse(
            http, change_ep, verify_ep, new_email, email_param, token_param, headers, method
        )
        if token_reuse_result:
            findings.append(token_reuse_result)
            trace.append({"test": "token_reuse", "result": "vulnerable"})
        else:
            trace.append({"test": "token_reuse", "result": "secure"})
        
        # 測試 2: 競態條件漏洞
        if enable_race:
            race_result = self._test_race_condition(
                http, change_ep, new_email, email_param, headers, method
            )
            if race_result:
                findings.append(race_result)
                trace.append({"test": "race_condition", "result": "vulnerable"})
            else:
                trace.append({"test": "race_condition", "result": "secure"})
        
        # 測試 3: 參數污染漏洞
        if enable_param_pollution:
            pollution_result = self._test_parameter_pollution(
                http, change_ep, verify_ep, current_email, new_email, email_param, token_param, headers, method
            )
            if pollution_result:
                findings.append(pollution_result)
                trace.append({"test": "parameter_pollution", "result": "vulnerable"})
            else:
                trace.append({"test": "parameter_pollution", "result": "secure"})
        
        # 測試 4: Host Header 攻擊
        if enable_host_header:
            host_result = self._test_host_header_injection(
                http, change_ep, new_email, email_param, headers, method
            )
            if host_result:
                findings.append(host_result)
                trace.append({"test": "host_header", "result": "vulnerable"})
            else:
                trace.append({"test": "host_header", "result": "secure"})
        
        # 構建命令記錄
        cmd = self.build_command_record(
            command="email.change.bypass",
            description=f"Email change bypass detection on {base}",
            parameters={
                "change_endpoint": change_ep,
                "tests_enabled": {
                    "token_reuse": True,
                    "race_condition": enable_race,
                    "parameter_pollution": enable_param_pollution,
                    "host_header": enable_host_header
                }
            }
        )
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={"trace": trace, "tests_run": len(trace)}
        )
    
    def _test_token_reuse(
        self,
        http: SafeHttp,
        change_ep: str,
        verify_ep: str,
        new_email: str,
        email_param: str,
        token_param: str,
        headers: Dict[str, Any],
        method: str
    ) -> Optional[Finding]:
        """測試 Token 重用漏洞"""
        try:
            # 第一次請求變更
            data1 = {email_param: new_email}
            r1 = http.request(method, change_ep, headers=headers, json=data1)
            
            if r1.status_code not in (200, 201):
                return None
            
            # 嘗試從響應中提取 token
            try:
                resp_data = r1.json()
                token = resp_data.get("token") or resp_data.get("verification_token")
                if not token:
                    return None
            except:
                return None
            
            # 第一次驗證
            verify_data1 = {token_param: token}
            r2 = http.request("POST", verify_ep, headers=headers, json=verify_data1)
            
            if r2.status_code not in (200, 201):
                return None
            
            # 第二次使用相同 token 驗證
            time.sleep(1)  # 給系統一點時間處理
            r3 = http.request("POST", verify_ep, headers=headers, json=verify_data1)
            
            # 如果第二次仍然成功，說明 token 可以重用
            if r3.status_code in (200, 201):
                return Finding(
                    vuln_type="Email Change - Token Reuse",
                    severity="high",
                    title="電子郵件變更 Token 可以重複使用",
                    evidence={
                        "change_request": {"url": change_ep, "email": new_email},
                        "token": token,
                        "first_verify": {"status": r2.status_code, "response": r2.text[:200]},
                        "second_verify": {"status": r3.status_code, "response": r3.text[:200]}
                    },
                    reproduction=[
                        {"step": 1, "request": {"method": method, "url": change_ep, "body": data1}, "description": "發送電子郵件變更請求"},
                        {"step": 2, "response": f"獲得 token: {token}", "description": "從響應中提取驗證 token"},
                        {"step": 3, "request": {"method": "POST", "url": verify_ep, "body": verify_data1}, "description": "第一次驗證 token"},
                        {"step": 4, "request": {"method": "POST", "url": verify_ep, "body": verify_data1}, "description": "第二次使用相同 token 驗證"},
                        {"step": 5, "expect": "第二次驗證應該失敗（token 已失效）", "actual": "第二次驗證仍然成功，token 可重用"}
                    ],
                    impact="攻擊者可以攔截一次驗證連結，然後重複使用 token 變更多個帳戶的電子郵件地址",
                    recommendation=(
                        "1. Token 應該在首次使用後立即失效\n"
                        "2. 實施 token 使用次數限制（僅允許使用一次）\n"
                        "3. 為 token 添加短暫的有效期（如 15 分鐘）\n"
                        "4. 記錄 token 使用歷史，防止重放攻擊"
                    )
                )
        except Exception:
            pass
        
        return None
    
    def _test_race_condition(
        self,
        http: SafeHttp,
        change_ep: str,
        new_email: str,
        email_param: str,
        headers: Dict[str, Any],
        method: str
    ) -> Optional[Finding]:
        """測試競態條件漏洞"""
        try:
            # 生成兩個不同的電子郵件地址
            email1 = new_email
            email2 = random_email("attacker.com")
            
            # 快速發送兩個請求（嘗試觸發競態條件）
            data1 = {email_param: email1}
            data2 = {email_param: email2}
            
            # 並發發送（簡化版本，實際應該使用線程）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(http.request, method, change_ep, headers=headers, json=data1)
                future2 = executor.submit(http.request, method, change_ep, headers=headers, json=data2)
                
                r1 = future1.result()
                r2 = future2.result()
            
            # 如果兩個請求都成功，可能存在競態條件
            if r1.status_code in (200, 201) and r2.status_code in (200, 201):
                # 檢查響應中是否包含兩個不同的 token
                try:
                    token1 = r1.json().get("token")
                    token2 = r2.json().get("token")
                    
                    if token1 and token2 and token1 != token2:
                        return Finding(
                            vuln_type="Email Change - Race Condition",
                            severity="medium",
                            title="電子郵件變更流程存在競態條件漏洞",
                            evidence={
                                "concurrent_requests": 2,
                                "email1": email1,
                                "email2": email2,
                                "token1": token1,
                                "token2": token2,
                                "both_succeeded": True
                            },
                            reproduction=[
                                {"step": 1, "description": "同時發送兩個電子郵件變更請求，使用不同的電子郵件地址"},
                                {"step": 2, "expect": "系統應該拒絕其中一個請求", "actual": "兩個請求都成功，各自獲得不同的 token"},
                                {"step": 3, "impact": "攻擊者可以利用競態條件繞過速率限制或創建多個驗證 token"}
                            ],
                            impact="攻擊者可能利用競態條件繞過單次變更限制，或者創建多個有效 token",
                            recommendation=(
                                "1. 在資料庫層面實施行級鎖定，確保同一用戶的電子郵件變更操作是序列化的\n"
                                "2. 使用分散式鎖（如 Redis）防止並發請求\n"
                                "3. 在創建新 token 前，使舊 token 失效\n"
                                "4. 實施嚴格的速率限制"
                            )
                        )
                except:
                    pass
        except Exception:
            pass
        
        return None
    
    def _test_parameter_pollution(
        self,
        http: SafeHttp,
        change_ep: str,
        verify_ep: str,
        current_email: str,
        new_email: str,
        email_param: str,
        token_param: str,
        headers: Dict[str, Any],
        method: str
    ) -> Optional[Finding]:
        """測試參數污染漏洞"""
        try:
            # 發送包含兩個 email 參數的請求
            # 有些框架會使用第一個，有些會使用最後一個
            polluted_data = {
                email_param: [current_email, new_email]  # 陣列形式
            }
            
            r1 = http.request(method, change_ep, headers=headers, json=polluted_data)
            
            if r1.status_code not in (200, 201):
                return None
            
            try:
                resp_data = r1.json()
                # 檢查響應中提到的是哪個電子郵件
                resp_text = json.dumps(resp_data).lower()
                
                # 如果響應中提到當前電子郵件，但實際上新電子郵件被更新了
                if current_email.lower() in resp_text:
                    return Finding(
                        vuln_type="Email Change - Parameter Pollution",
                        severity="medium",
                        title="電子郵件變更存在參數污染漏洞",
                        evidence={
                            "polluted_request": polluted_data,
                            "response": resp_data,
                            "current_email_in_response": True,
                            "multiple_params_sent": True
                        },
                        reproduction=[
                            {"step": 1, "request": {"method": method, "url": change_ep, "body": polluted_data}, "description": "發送包含多個 email 參數的請求"},
                            {"step": 2, "expect": "系統應該拒絕或正確處理多個參數", "actual": "系統可能使用了錯誤的參數值"}
                        ],
                        impact="攻擊者可能透過參數污染繞過電子郵件驗證或更新錯誤的電子郵件地址",
                        recommendation=(
                            "1. 明確處理多個同名參數的情況，只接受單一值\n"
                            "2. 驗證和更新操作使用相同的參數解析邏輯\n"
                            "3. 記錄異常的參數模式並發出警報"
                        )
                    )
            except:
                pass
        except Exception:
            pass
        
        return None
    
    def _test_host_header_injection(
        self,
        http: SafeHttp,
        change_ep: str,
        new_email: str,
        email_param: str,
        headers: Dict[str, Any],
        method: str
    ) -> Optional[Finding]:
        """測試 Host Header 注入攻擊"""
        try:
            # 修改 Host 標頭為攻擊者控制的域名
            evil_headers = headers.copy()
            evil_headers["Host"] = "evil.attacker.com"
            
            data = {email_param: new_email}
            r = http.request(method, change_ep, headers=evil_headers, json=data)
            
            # 如果請求成功，檢查響應中是否包含惡意 Host
            if r.status_code in (200, 201):
                try:
                    resp_data = r.json()
                    resp_text = json.dumps(resp_data)
                    
                    # 如果響應中包含惡意 Host，說明驗證連結可能被發送到攻擊者域名
                    if "evil.attacker.com" in resp_text:
                        return Finding(
                            vuln_type="Email Change - Host Header Injection",
                            severity="critical",
                            title="電子郵件變更驗證連結存在 Host Header 注入漏洞",
                            evidence={
                                "injected_host": "evil.attacker.com",
                                "response_contains_evil_host": True,
                                "response": resp_data
                            },
                            reproduction=[
                                {"step": 1, "request": {"method": method, "url": change_ep, "headers": evil_headers, "body": data}, "description": "發送帶有惡意 Host 標頭的變更請求"},
                                {"step": 2, "expect": "系統應該使用配置的域名生成驗證連結", "actual": f"系統使用了 Host 標頭 (evil.attacker.com) 生成驗證連結"},
                                {"step": 3, "impact": "驗證連結可能被發送到攻擊者控制的域名，導致 token 洩漏"}
                            ],
                            impact="攻擊者可以透過 Host Header 注入讓驗證連結指向惡意域名，攔截驗證 token 並接管帳戶",
                            recommendation=(
                                "1. 不要使用 Host 標頭構建驗證連結，使用配置的域名\n"
                                "2. 驗證 Host 標頭是否在允許的域名白名單中\n"
                                "3. 使用絕對 URL 而非相對 URL 生成驗證連結\n"
                                "4. 實施 Host Header 驗證中介軟體"
                            )
                        )
                except:
                    pass
        except Exception:
            pass
        
        return None
