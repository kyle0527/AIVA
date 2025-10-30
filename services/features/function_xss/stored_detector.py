

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
    """

    def __init__(
        self,
        task: FunctionTaskPayload,
        *,
        timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
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

            # View step: check provided URLs or fallback heuristic
            candidates = list(view_urls or [])
            if not candidates:
                candidates = [self._fallback_view_url()]

            results: list[StoredXssResult] = []
            for url in candidates:
                if not url:
                    continue
                resp = await client.get(url)
                text = resp.text or ""
                if payload in text:
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

    def _fallback_view_url(self) -> str:
        # Strip query to avoid re-submitting; GET base resource
        parsed = urlparse(str(self._task.target.url))
        return urlunparse(parsed._replace(query=""))

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
