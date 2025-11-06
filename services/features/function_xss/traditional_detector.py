

from collections.abc import Sequence
import copy
from dataclasses import dataclass
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
                if _payload_in_response(payload, body_text):
                    request = response.request
                    results.append(
                        XssDetectionResult(
                            payload=payload,
                            request=request,
                            response_status=response.status_code,
                            response_headers=dict(response.headers),
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

    for key in list(mapping.keys()):
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
