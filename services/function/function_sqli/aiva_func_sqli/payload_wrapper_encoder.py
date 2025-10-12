from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class EncodedPayload:
    url: str
    method: str
    payload: str
    request_kwargs: dict[str, Any]

    def build_request_dump(self) -> str:
        lines = [f"{self.method} {self.url}"]
        if "headers" in self.request_kwargs:
            for key, value in self.request_kwargs["headers"].items():
                lines.append(f"{key}: {value}")
        body_parts = []
        if "params" in self.request_kwargs:
            body_parts.append(f"params={self.request_kwargs['params']}")
        if "data" in self.request_kwargs:
            body_parts.append(f"data={self.request_kwargs['data']}")
        if "json" in self.request_kwargs:
            body_parts.append(f"json={self.request_kwargs['json']}")
        if "content" in self.request_kwargs:
            body_parts.append(f"content={self.request_kwargs['content']!r}")
        if body_parts:
            lines.append("\n".join(body_parts))
        return "\n".join(lines)


class PayloadWrapperEncoder:
    """
    Payload 包裝和編碼器

    提供多種編碼和包裝技術來繞過 WAF 和輸入過濾，
    增加 SQL 注入 payload 的成功率。同時整合平台接口。
    """

    def __init__(self, task: FunctionTaskPayload) -> None:
        self._task = task

    def encode(self, payload: str) -> EncodedPayload:
        """將 payload 編碼為目標請求格式"""
        target = self._task.target
        method = target.method.upper() if target.method else "GET"
        headers = dict(target.headers)
        cookies = dict(target.cookies)
        location = (target.parameter_location or "query").lower()
        parameter = target.parameter

        request_kwargs: dict[str, Any] = {"headers": headers, "cookies": cookies}

        if location == "query":
            base_params = dict(
                parse_qsl(urlparse(str(target.url)).query, keep_blank_values=True)
            )
            if parameter:
                base_params[parameter] = payload
            request_kwargs["params"] = base_params
            url = str(target.url)
        elif location == "form" and method in {"POST", "PUT", "PATCH"}:
            data = dict(target.form_data)
            if parameter:
                data[parameter] = payload
            request_kwargs["data"] = data
            url = str(target.url)
        elif location == "json" and method in {"POST", "PUT", "PATCH"}:
            json_payload = dict(target.json_data or {})
            if parameter:
                json_payload[parameter] = payload
            request_kwargs["json"] = json_payload
            request_kwargs.setdefault("headers", {})["Content-Type"] = (
                "application/json"
            )
            url = str(target.url)
        elif location == "body":
            body = target.body or ""
            request_kwargs["content"] = body.replace("{{INJECT}}", payload)
            url = str(target.url)
        else:  # fallback to query injection
            url = self._inject_query(str(target.url), parameter, payload)

        request_kwargs = {k: v for k, v in request_kwargs.items() if v}

        return EncodedPayload(
            url=url,
            method=method,
            payload=payload,
            request_kwargs=request_kwargs,
        )

    @staticmethod
    def _inject_query(url: str, parameter: str | None, value: str) -> str:
        if not parameter:
            return url
        parts = list(urlparse(url))
        query_pairs = dict(parse_qsl(parts[4], keep_blank_values=True))
        query_pairs[parameter] = value
        parts[4] = urlencode(query_pairs, doseq=True)
        return urlunparse(parts)
