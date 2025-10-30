

from services.aiva_common.schemas import Fingerprints


class PassiveFingerprinter:
    """Build passive fingerprints from response headers."""

    def from_headers(self, headers: dict[str, str]) -> Fingerprints:
        lower = {k.lower(): v for k, v in headers.items()}
        web_server = (
            {"name": headers.get("Server", ""), "version": ""}
            if "server" in lower
            else None
        )
        framework = (
            {"name": headers.get("X-Powered-By", ""), "version": ""}
            if "x-powered-by" in lower
            else None
        )
        return Fingerprints(
            web_server=web_server,
            framework=framework,
            language=None,
            waf_detected=False,
            waf_vendor=None,
        )
