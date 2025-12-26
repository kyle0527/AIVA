

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import html
from urllib.parse import urlparse, urlunparse

import httpx

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class StoredXssResult:
    payload: str
    request: httpx.Request
    response_status: int
    response_headers: dict[str, str]
    response_text: str


class StoredXssDetector:
    """Two-step stored XSS detector: submit then view.

    This detector submits candidate payloads using the task's target definition
    and then requests one or more candidate "view" URLs to verify persistence.
    It is intentionally conservative and will only look for a raw payload echo
    (similar to reflected detection) to keep implementation simple and stable.
    
    Requirements:
        view_urls must be explicitly provided (including test range URLs)
    """

    def __init__(
        self,
        task: FunctionTaskPayload,
        *,
        timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        初始化持久化 XSS 檢測器
        
        Args:
            task: 功能任務載荷，包含目標信息
            timeout: HTTP 請求超時時間（秒）
            client: 可選的 httpx 客戶端
        """
        self._task = task
        self._timeout = timeout
        self._client = client

    async def execute(
        self,
        payloads: Sequence[str],
        *,
        view_urls: Iterable[str] | None = None,
    ) -> list[StoredXssResult]:
        if not payloads:
            return []

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            follow_redirects=True, timeout=self._timeout
        )

        try:
            # Submit first payload only (avoid spamming persistent backends)
            payload = payloads[0]
            await self._submit_payload(client, payload)

            candidates = list(view_urls or [])
            if not candidates:
                raise ValueError(
                    "view_urls is required for stored XSS detection. "
                    "Provide the URL(s) where the stored payload should be checked. "
                    "Example: detector.execute(payloads, view_urls=['https://target.com/view'])"
                )

            results: list[StoredXssResult] = []
            for url in candidates:
                if not url:
                    continue
                resp = await client.get(url)
                text = resp.text or ""
                if payload in text:
                    # 驗證是否為真實的持久化 XSS (不是反射或緩存)
                    if self._verify_persistence(payload, text, resp):
                        results.append(
                            StoredXssResult(
                                payload=payload,
                                request=resp.request,
                                response_status=resp.status_code,
                                response_headers=dict(resp.headers),
                                response_text=text,
                            )
                        )
                        break  # one positive hit is enough

            return results
        finally:
            if owns_client:
                await client.aclose()

    async def _submit_payload(
        self, client: httpx.AsyncClient, payload: str
    ) -> httpx.Response:
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

        if location in {"query", "url"}:
            url = self._inject_query(url, parameter, payload)
        elif location == "form":
            data = dict(target.form_data or {})
            if parameter:
                data[parameter] = payload
        elif location == "json":
            base = dict(target.json_data or {})
            if parameter:
                base[parameter] = payload
            json_data = base
        elif location == "header" and parameter:
            headers[parameter] = payload
        elif location == "cookie" and parameter:
            cookies[parameter] = payload
        elif location in {"body", "body_raw"}:
            content = payload.encode("utf-8")

        return await client.request(
            method,
            url,
            headers=headers or None,
            cookies=cookies or None,
            data=data,
            json=json_data,
            content=content,
        )



    def _verify_persistence(
        self,
        payload: str,
        response_text: str,
        response: httpx.Response
    ) -> bool:
        """驗證是否為真實的持久化 XSS，過濾反射型和緩存誤報"""
        # 檢查 1: 確認不是反射型 (檢查 URL 中沒有 payload)
        if payload in str(response.url):
            return False  # URL 中有 payload = 反射型 XSS
        
        # 檢查 2: 確認不是 HTTP 緩存
        cache_headers = [
            "x-cache", "cf-cache-status", "x-varnish-cache",
            "x-proxy-cache", "age"
        ]
        if any(header in response.headers for header in cache_headers):
            # 如果有緩存標頭且是 HIT，可能是緩存的反射型
            cache_status = " ".join(
                str(response.headers.get(h, "")).lower() 
                for h in cache_headers
            )
            if "hit" in cache_status or "from cache" in cache_status:
                return False
        
        # 檢查 3: 確認 payload 在可執行的上下文中 (非註釋/textarea)
        # 使用與 traditional_detector 相同的邏輯
        escaped_payload = html.escape(payload)
        if escaped_payload in response_text and payload not in response_text:
            return False  # Payload 被編碼，無法執行
        
        # 檢查安全上下文 (HTML 註釋、textarea、noscript 等)
        safe_contexts = [
            f"<!--{payload}-->",
            f"<!--{payload}",
            f"{payload}-->",
            f"<textarea>{payload}",
            f"<textarea {payload}",
            f"<noscript>{payload}",
            f"<style>{payload}",
        ]
        if any(ctx in response_text for ctx in safe_contexts):
            return False
        
        # 檢查 4: 驗證多次請求的一致性 (真正持久化的內容應該每次都出現)
        # 注意：這裡只做基本檢查，完整的多次驗證需要在更高層級實作
        
        return True
    
    @staticmethod
    def _inject_query(url: str, parameter: str | None, value: str) -> str:
        if not parameter:
            return url
        parsed = urlparse(url)
        query_items: dict[str, str] = {}
        if parsed.query:
            for pair in parsed.query.split("&"):
                if not pair:
                    continue
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    query_items[key] = val
                else:
                    query_items[pair] = ""
        query_items[parameter] = value
        parts = list(parsed)
        from urllib.parse import urlencode

        parts[4] = urlencode(query_items, doseq=True)
        return urlunparse(parts)
