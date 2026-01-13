

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import html
import re
from html import unescape
from urllib.parse import unquote_plus, urlencode, urlparse, urlunparse

import httpx

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class XssDetectionResult:
    payload: str
    request: httpx.Request
    response_status: int
    response_headers: dict[str, str]
    response_text: str


@dataclass
class XssExecutionError:
    """Represents a failed payload attempt for resiliency telemetry."""

    payload: str
    vector: str
    message: str
    attempts: int

    def to_detail(self) -> str:
        prefix = f"[{self.vector}]"
        return f"{prefix} {self.payload!r} failed after {self.attempts} attempts: {self.message}"


class TraditionalXssDetector:
    """HTTP-based reflected and stored XSS detector."""

    def __init__(
        self,
        task: FunctionTaskPayload,
        *,
        timeout: float,
        retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._task = task
        self._timeout = timeout
        self._client = client
        self._retries = max(0, retries)
        self._errors: list[XssExecutionError] = []

    async def execute(self, payloads: Sequence[str]) -> list[XssDetectionResult]:
        if not payloads:
            return []

        results: list[XssDetectionResult] = []
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            follow_redirects=True, timeout=self._timeout
        )

        try:
            for payload in payloads:
                response: httpx.Response | None = None
                attempts = 0
                while True:
                    try:
                        method, url, headers, cookies, data, json_data, content = (
                            self._build_request_parts(payload)
                        )
                        response = await client.request(
                            method=method,
                            url=url,
                            headers=headers,
                            cookies=cookies,
                            data=data,
                            json=json_data,
                            content=content,
                        )
                        break
                    except httpx.HTTPError as exc:
                        attempts += 1
                        if attempts > self._retries:
                            vector = (
                                self._task.target.parameter_location or "query"
                            ).lower()
                            self._errors.append(
                                XssExecutionError(
                                    payload=payload,
                                    vector=vector,
                                    message=str(exc),
                                    attempts=attempts,
                                )
                            )
                            response = None
                            break
                        continue

                if response is None:
                    continue

                body_text = response.text or ""
                response_headers = dict(response.headers)
                
                # ✅ 強化驗證：不僅檢查 payload 存在，還要驗證執行上下文
                if _payload_in_response(payload, body_text):
                    # 驗證 1: 檢查是否在可執行上下文
                    if not _verify_execution_context(payload, body_text, response_headers):
                        continue
                    
                    # 驗證 2: 檢查 WAF 是否干擾
                    if _detect_waf_interference(payload, body_text):
                        continue
                    
                    # ✅ 確認為有效 XSS
                    request = response.request
                    results.append(
                        XssDetectionResult(
                            payload=payload,
                            request=request,
                            response_status=response.status_code,
                            response_headers=response_headers,
                            response_text=body_text,
                        )
                    )
        finally:
            if owns_client:
                await client.aclose()

        return results

    @property
    def errors(self) -> list[XssExecutionError]:
        return list(self._errors)

    def _build_request_parts(
        self, payload: str
    ) -> tuple[
        str,
        str,
        dict[str, str] | None,
        dict[str, str] | None,
        dict[str, object] | None,
        dict[str, object] | None,
        bytes | None,
    ]:
        """Build request parts for httpx.AsyncClient.request()."""
        target = self._task.target
        method = (target.method or "GET").upper()
        location = (target.parameter_location or "query").lower()
        parameter = target.parameter
        url = str(target.url)

        headers = dict(target.headers or {})
        cookies = dict(target.cookies or {})
        data: dict[str, object] | None = None
        json_data: dict[str, object] | None = None
        content: bytes | None = None

        if location == "query" or (
            location not in {"form", "json", "header", "cookie", "body"}
            and method == "GET"
        ):
            url = _inject_query(url, parameter, payload)
        elif location == "form":
            form = copy.deepcopy(target.form_data) or {}
            data = _inject_mapping(form, parameter, payload)
        elif location == "json":
            json_base = copy.deepcopy(target.json_data) or {}
            json_data = _inject_mapping(json_base, parameter, payload)
        elif location == "header" and parameter:
            headers[parameter] = payload
        elif location == "cookie" and parameter:
            cookies[parameter] = payload
        elif location == "body":
            content = payload.encode("utf-8")
        elif target.body:
            content = (
                target.body.encode("utf-8")
                if isinstance(target.body, str)
                else target.body
            )

        return (
            method,
            url,
            headers or None,
            cookies or None,
            data,
            json_data,
            content,
        )


def _inject_mapping(
    mapping: dict[str, object], parameter: str | None, payload: str
) -> dict[str, object]:
    if not mapping:
        return {parameter or "xss_probe": payload}

    if parameter and parameter in mapping:
        mapping[parameter] = payload
        return mapping

    if parameter and parameter not in mapping:
        mapping[parameter] = payload
        return mapping

    for key in mapping.keys():
        mapping[key] = payload
    return mapping


def _inject_query(url: str, parameter: str | None, payload: str) -> str:
    parsed = urlparse(url)
    query_items: dict[str, str] = {}
    if parsed.query:
        for pair in parsed.query.split("&"):
            if not pair:
                continue
            if "=" in pair:
                key, value = pair.split("=", 1)
                query_items[key] = value
            else:
                query_items[pair] = ""

    if parameter:
        query_items[parameter] = payload
    else:
        query_items.setdefault("xss_probe", payload)

    new_query = urlencode(query_items, doseq=True)
    parts = list(parsed)
    parts[4] = new_query
    return urlunparse(parts)


def _payload_in_response(payload: str, body_text: str) -> bool:
    if payload in body_text:
        return True

    decoded = unquote_plus(body_text)
    if payload in decoded:
        return True

    html_decoded = unescape(body_text)
    return payload in html_decoded


def _verify_execution_context(payload: str, response_text: str, response_headers: dict[str, str]) -> bool:
    """
    驗證 payload 是否在可執行上下文中
    
    Args:
        payload: XSS payload
        response_text: 響應內容
        response_headers: 響應標頭
        
    Returns:
        bool: True 表示在可執行上下文，False 表示在安全上下文
    """
    # 檢查 1: 是否在安全上下文（HTML 註解、textarea、script 註解等）
    safe_contexts = [
        r'<!--.*?' + re.escape(payload) + r'.*?-->',  # HTML 註解
        r'<textarea[^>]*>.*?' + re.escape(payload),    # Textarea
        r'<script[^>]*><!--.*?' + re.escape(payload),  # Script 註解
        r'<noscript[^>]*>.*?' + re.escape(payload),    # NoScript
        r'<style[^>]*>.*?' + re.escape(payload),       # Style 標籤
    ]
    
    for pattern in safe_contexts:
        if re.search(pattern, response_text, re.DOTALL | re.IGNORECASE):
            return False  # 在安全上下文，非有效 XSS
    
    # 檢查 2: 是否被 HTML 編碼
    encoded_payload = html.escape(payload)
    # 如果找到編碼版本但找不到原始版本，說明被編碼了
    if encoded_payload in response_text and payload not in response_text:
        # 再次確認編碼的 payload 確實存在
        if '&lt;' in encoded_payload or '&gt;' in encoded_payload or '&quot;' in encoded_payload:
            return False  # 被編碼，無法執行
    
    # 檢查 3: CSP (Content Security Policy) 是否阻止執行
    csp = response_headers.get('content-security-policy', '') or response_headers.get('Content-Security-Policy', '')
    if csp:
        csp_lower = csp.lower()
        
        # 檢查是否有 script-src 限制
        if 'script-src' in csp_lower:
            # 如果沒有 'unsafe-inline'，內聯腳本會被阻止
            if "'unsafe-inline'" not in csp_lower:
                # 檢查是否有 nonce 或 hash（這些可能允許特定腳本）
                # 但我們的測試 payload 通常沒有正確的 nonce/hash
                if 'nonce-' in response_text or 'sha256-' in csp_lower or 'sha384-' in csp_lower:
                    return False  # CSP 阻止執行
    
    # 檢查 4: 是否在屬性值內但被正確引號包圍
    # 例如: <input value="<script>alert(1)</script>"> - 這不會執行
    attr_patterns = [
        r'<[^>]*\s+\w+\s*=\s*["\'].*?' + re.escape(payload) + r'.*?["\'][^>]*>',
    ]
    
    for pattern in attr_patterns:
        matches = re.finditer(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            matched_text = match.group(0)
            # 檢查 payload 是否被引號正確包圍，如果在引號內可能不會執行
            if '<script' in payload.lower() and ('"' in matched_text or "'" in matched_text):
                pass  # 繼續其他檢查
    
    return True  # 通過所有檢查，在可執行上下文


def _detect_waf_interference(payload: str, response_text: str) -> bool:
    """
    檢測 WAF 是否干擾了 payload
    
    Args:
        payload: 原始 payload
        response_text: 響應內容
        
    Returns:
        bool: True 表示檢測到 WAF 干擾，False 表示無干擾
    """
    # 常見 WAF 修改模式
    waf_patterns = [
        # Imperva WAF
        (r'<scr<script>ipt>', '<script>'),
        (r'<scri<script>pt>', '<script>'),
        
        # Cloudflare WAF
        (r'<script.*?removed.*?>', '<script>'),
        (r'<script.*?blocked.*?>', '<script>'),
        
        # AWS WAF
        (r'javascript:.*?blocked', 'javascript:'),
        
        # ModSecurity
        (r'<script.*?filtered.*?>', '<script>'),
        
        # 通用過濾
        (r'<script.*?sanitized.*?>', '<script>'),
    ]
    
    for waf_pattern, original_pattern in waf_patterns:
        if original_pattern in payload.lower():
            if re.search(waf_pattern, response_text, re.IGNORECASE):
                return True  # 檢測到 WAF 干擾
    
    # 檢查 WAF 標識性響應
    waf_indicators = [
        'cloudflare',
        'imperva',
        'incapsula',
        'sucuri',
        'barracuda',
        'f5 big-ip',
        'fortiweb',
        'modsecurity',
        'request blocked',
        'access denied',
        'security policy',
    ]
    
    response_lower = response_text.lower()
    for indicator in waf_indicators:
        if indicator in response_lower:
            # 可能是 WAF 阻止頁面
            return True
    
    return False  # 無 WAF 干擾
