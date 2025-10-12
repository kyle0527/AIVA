from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import ipaddress
import re
from typing import Any

import httpx

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class InternalIndicator:
    """Represents a single hint that an internal resource was contacted."""

    source: str
    value: str
    reason: str


@dataclass
class InternalAddressDetection:
    """Aggregated result returned by :class:`InternalAddressDetector`."""

    matched: bool
    indicators: list[InternalIndicator] = field(default_factory=list)

    def summary(self) -> str:
        if not self.indicators:
            return "No indicators collected"
        return "; ".join(
            f"{item.source}: {item.value} ({item.reason})" for item in self.indicators
        )


@dataclass
class InternalAddressDetector:
    """Detect internal addresses and assess SSRF risk.

    This class provides helper methods to analyze a probe response and
    determine whether internal addresses, cloud metadata or internal
    services are accessible.
    """

    _internal_ranges: list[str] = field(
        default_factory=lambda: [
            "127.0.0.0/8",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.0.0/16",
            "::1/128",
        ]
    )
    _internal_service_ports: list[int] = field(
        default_factory=lambda: [80, 443, 22, 21, 25, 53, 3306, 5432, 6379, 9200]
    )
    _special_protocols: list[str] = field(
        default_factory=lambda: ["file://", "ftp://", "gopher://", "ldap://"]
    )

    def analyze(
        self, response: httpx.Response | str | None
    ) -> InternalAddressDetection:
        """Analyze a probe response and return structured findings."""
        text = (
            response.text if isinstance(response, httpx.Response) else (response or "")
        )
        indicators: list[InternalIndicator] = []

        # find IP-like strings in the response and check if they map to internal ranges
        ips = set(re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", text))
        for ip in ips:
            if self.is_internal_address(ip):
                indicators.append(
                    InternalIndicator(
                        source="ip_address", value=ip, reason="internal network range"
                    )
                )

        # cloud metadata checks
        for provider in ("aws", "gcp", "azure", "alibaba"):
            if self._is_metadata_response(text, provider):
                indicators.append(
                    InternalIndicator(
                        source="cloud_metadata",
                        value=provider,
                        reason="metadata service response detected",
                    )
                )

        # protocol support inferred from response text
        for protocol in self._special_protocols:
            if self._is_protocol_supported(text, protocol):
                indicators.append(
                    InternalIndicator(
                        source="protocol",
                        value=protocol.removesuffix("://"),
                        reason="special protocol support detected",
                    )
                )

        # Check for specific metadata indicators
        if "169.254.169.254" in text:
            indicators.append(
                InternalIndicator(
                    source="metadata_ip",
                    value="169.254.169.254",
                    reason="AWS/cloud metadata endpoint accessed",
                )
            )

        matched = len(indicators) > 0
        return InternalAddressDetection(matched=matched, indicators=indicators)

    def _test_internal_services(
        self,
        url: str,
        parameter: str,
        test_function: Callable[..., str | None],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Probe common internal service ports using the provided callable."""
        detected_services: list[dict[str, Any]] = []

        test_host = "127.0.0.1"

        for port in self._internal_service_ports[:10]:
            try:
                test_url = f"http://{test_host}:{port}"
                response = test_function(url, parameter, test_url, **kwargs)

                if self._is_service_response(response, port):
                    service_type = self._identify_service_type(port)
                    detected_services.append(
                        {
                            "host": test_host,
                            "port": port,
                            "service_type": service_type,
                            "url": test_url,
                            "response_preview": response[:100] if response else "",
                        }
                    )
                    logger.info(
                        f"Internal service detected: {service_type} on port {port}"
                    )

            except Exception as e:
                logger.debug(f"Error testing port {port}: {e}")
                continue

        return detected_services

    def _test_protocol_support(
        self,
        url: str,
        parameter: str,
        test_function: Callable[..., str | None],
        **kwargs: Any,
    ) -> list[str]:
        """Check whether special protocol schemes are honoured by the target."""
        supported_protocols: list[str] = []

        for protocol in self._special_protocols:
            try:
                test_url = f"{protocol}127.0.0.1/test"
                response = test_function(url, parameter, test_url, **kwargs)

                if self._is_protocol_supported(response, protocol):
                    supported_protocols.append(protocol.removesuffix("://"))
                    logger.info(f"Protocol support detected: {protocol}")

            except Exception as e:
                logger.debug(f"Error testing protocol {protocol}: {e}")
                continue

        return supported_protocols

    def _is_successful_response(self, response: str | None) -> bool:
        """
        Determine whether the supplied response looks like a
        successful HTTP reply.
        """
        if not response:
            return False

        success_indicators = [
            "<html",
            "<body",
            "<head",
            "<title",
            "server:",
            "content-type:",
            "set-cookie:",
            "apache",
            "nginx",
            "iis",
            "lighttpd",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in success_indicators)

    def _is_metadata_response(self, response: str | None, provider: str) -> bool:
        """
        Verify if the response body contains indicators of
        cloud metadata endpoints.
        """
        if not response:
            return False

        metadata_indicators = {
            "aws": ["ami-", "instance-id", "security-groups", "iam/"],
            "gcp": ["project/", "instance/", "service-accounts/"],
            "azure": ["compute/", "network/", "storage/"],
            "alibaba": ["instance-id", "image-id", "region-id"],
        }

        indicators = metadata_indicators.get(provider, [])
        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in indicators)

    def _is_service_response(self, response: str | None, port: int) -> bool:
        """
        Verify if the response matches heuristics for a service
        listening on the given port.
        """
        if not response:
            return False

        service_indicators = {
            21: ["ftp", "220 "],
            22: ["ssh", "protocol"],
            25: ["smtp", "220 "],
            80: ["http", "<html", "server:"],
            443: ["https", "ssl", "tls"],
            3306: ["mysql", "5."],
            5432: ["postgresql", "psql"],
            6379: ["redis", "redis_version"],
            9200: ["elasticsearch", '"cluster_name"'],
        }

        indicators = service_indicators.get(port, ["server", "service"])
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in indicators)

    def _identify_service_type(self, port: int) -> str:
        service_map = {
            21: "FTP",
            22: "SSH",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            443: "HTTPS",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            9200: "Elasticsearch",
        }
        return service_map.get(port, f"Unknown service on port {port}")

    def _is_protocol_supported(self, response: str | None, protocol: str) -> bool:
        if not response:
            return False

        protocol_indicators = {
            "file://": ["file not found", "permission denied", "directory"],
            "ftp://": ["ftp", "220", "login"],
            "gopher://": ["gopher", "selector"],
            "ldap://": ["ldap", "bind", "search"],
        }

        indicators = protocol_indicators.get(protocol, [])
        if not indicators:
            return True

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in indicators)

    def _assess_risk_level(self, results: dict[str, Any]) -> str:
        risk_score = 0
        if results.get("accessible_addresses"):
            risk_score += len(results["accessible_addresses"]) * 2
        if results.get("cloud_metadata_access"):
            risk_score += len(results["cloud_metadata_access"]) * 5
        if results.get("internal_services"):
            risk_score += len(results["internal_services"]) * 3
        if results.get("protocol_support"):
            risk_score += len(results["protocol_support"]) * 1

        if risk_score >= 10:
            return "critical"
        elif risk_score >= 6:
            return "high"
        elif risk_score >= 3:
            return "medium"
        elif risk_score >= 1:
            return "low"
        else:
            return "info"

    def _generate_evidence(self, results: dict[str, Any]) -> list[str]:
        evidence: list[str] = []
        if results.get("accessible_addresses"):
            count = len(results["accessible_addresses"])
            evidence.append(f"Discovered {count} internal addresses")
        if results.get("cloud_metadata_access"):
            providers = {item["provider"] for item in results["cloud_metadata_access"]}
            evidence.append(
                f"Observed cloud metadata responses from: {', '.join(providers)}"
            )
        if results.get("internal_services"):
            services = [item["service_type"] for item in results["internal_services"]]
            evidence.append(f"Detected internal services: {', '.join(services)}")
        if results.get("protocol_support"):
            protocols = results["protocol_support"]
            evidence.append(f"Supported protocols: {', '.join(protocols)}")
        return evidence

    def is_internal_address(self, address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
            for range_str in self._internal_ranges:
                if ip in ipaddress.ip_network(range_str, strict=False):
                    return True
        except ValueError:
            internal_domains = [
                "localhost",
                "internal",
                "local",
                "private",
                "metadata.google.internal",
            ]
            return any(domain in address.lower() for domain in internal_domains)

        return False
