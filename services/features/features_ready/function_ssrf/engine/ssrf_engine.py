from __future__ import annotations
import asyncio, socket, ipaddress
from dataclasses import dataclass
from typing import Optional, List
import httpx

@dataclass
class SSRFIssue:
    kind: str
    url: str
    description: str
    severity: str = "HIGH"
    cwe: Optional[str] = None
    evidence: Optional[str] = None

class SSRFEngine:
    def __init__(self, *, timeout: float, max_redirects: int, allow_active: bool, safe_mode: bool):
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=max_redirects>0)
        self.allow_active = allow_active and not safe_mode
        self.safe_mode = safe_mode

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def _resolve_ips(host: str) -> list[str]:
        try:
            infos = socket.getaddrinfo(host, None)
            ips = []
            for info in infos:
                ip = info[4][0]
                if ip not in ips:
                    ips.append(ip)
            return ips
        except Exception:
            return []

    @staticmethod
    def _is_internal_ip(ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
            return addr.is_private or addr.is_loopback or addr.is_link_local
        except ValueError:
            return False

    async def check_internal_access(self, url: str) -> list[SSRFIssue]:
        issues: List[SSRFIssue] = []
        # Resolve host
        try:
            host = url.split("://", 1)[-1].split("/", 1)[0].split("@")[-1].split(":")[0]
        except Exception:
            host = ""
        ips = self._resolve_ips(host) if host else []
        internal = any(self._is_internal_ip(ip) for ip in ips)
        if internal:
            if self.allow_active:
                try:
                    r = await self.client.get(url)
                    if r.status_code < 500:
                        issues.append(SSRFIssue(
                            kind="SSRF_INTERNAL_ACCESS",
                            url=url,
                            description=f"Internal host {host} is reachable via SSRF",
                            severity="HIGH",
                            cwe="CWE-918",
                            evidence=f"status={r.status_code}, len={len(r.text)}"
                        ))
                except Exception as e:
                    issues.append(SSRFIssue(
                        kind="SSRF_INTERNAL_POTENTIAL",
                        url=url,
                        description=f"Internal host {host} resolved; active probe failed",
                        severity="MEDIUM",
                        cwe="CWE-918",
                        evidence=str(e)
                    ))
            else:
                issues.append(SSRFIssue(
                    kind="SSRF_INTERNAL_POTENTIAL",
                    url=url,
                    description=f"Internal host {host} resolved (safe_mode: no active probe)",
                    severity="MEDIUM",
                    cwe="CWE-918"
                ))
        return issues

    async def check_cloud_metadata(self) -> list[SSRFIssue]:
        issues: List[SSRFIssue] = []
        meta_targets = [
            ("AWS","http://169.254.169.254/latest/meta-data/"),
            ("GCP","http://metadata.google.internal/computeMetadata/v1/"),
            ("Azure","http://169.254.169.254/metadata/instance?api-version=2021-02-01")
        ]
        headers_map = {
            "GCP": {"Metadata-Flavor": "Google"},
            "Azure": {"Metadata": "true"}
        }
        if not self.allow_active:
            # Passive indication only
            for name, url in meta_targets:
                issues.append(SSRFIssue(
                    kind="SSRF_METADATA_POTENTIAL",
                    url=url,
                    description=f"Cloud metadata endpoint known: {name} ({url}) (safe_mode)",
                    severity="MEDIUM",
                    cwe="CWE-200"
                ))
            return issues
        # Active probes (careful; rely on timeout)
        for name, url in meta_targets:
            try:
                r = await self.client.get(url, headers=headers_map.get(name, {}))
                if r.status_code == 200 and r.text:
                    issues.append(SSRFIssue(
                        kind="SSRF_METADATA_EXPOSED",
                        url=url,
                        description=f"{name} metadata accessible via SSRF",
                        severity="CRITICAL",
                        cwe="CWE-200",
                        evidence=r.text[:200]
                    ))
            except Exception:
                continue
        return issues

    async def check_file_protocol(self, url: str) -> list[SSRFIssue]:
        issues: List[SSRFIssue] = []
        if url.lower().startswith("file://"):
            issues.append(SSRFIssue(
                kind="SSRF_FILE_PROTOCOL",
                url=url,
                description="file:// access via SSRF may expose local files",
                severity="MEDIUM",
                cwe="CWE-73"
            ))
        return issues

    async def run(self, *, target_url: str, enable_internal: bool, enable_metadata: bool, enable_file: bool) -> list[SSRFIssue]:
        tasks = []
        if enable_internal:
            tasks.append(self.check_internal_access(target_url))
        if enable_metadata:
            tasks.append(self.check_cloud_metadata())
        if enable_file:
            tasks.append(self.check_file_protocol(target_url))
        issues = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, list):
                    issues.extend(r)
        return issues
