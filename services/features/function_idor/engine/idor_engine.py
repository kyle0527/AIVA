from __future__ import annotations
import re, httpx
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class IdCandidate:
    value: str
    span: tuple[int,int]

@dataclass
class IDORIssue:
    kind: str
    url: str
    description: str
    severity: str = "HIGH"
    cwe: Optional[str] = None
    evidence: Optional[str] = None

class IDOREngine:
    def __init__(self, *, timeout: float, allow_active: bool, safe_mode: bool):
        self.allow_active = allow_active and not safe_mode
        self.safe_mode = safe_mode
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def extract_ids_from_url(url: str) -> List[IdCandidate]:
        ids: List[IdCandidate] = []
        for m in re.finditer(r'(?<![\w])([0-9]{1,12})(?![\w])', url):
            ids.append(IdCandidate(m.group(1), m.span(1)))
        return ids

    @staticmethod
    def generate_variants(raw: str, count: int) -> List[str]:
        try:
            base = int(raw)
            pool = []
            for d in range(1, count+1):
                pool += [str(base+d), str(max(0, base-d))]
            return list(dict.fromkeys(pool))
        except ValueError:
            return [raw]

    @staticmethod
    def replace_id_in_url(url: str, old: str, new: str) -> str:
        return url.replace(old, new, 1)

    async def test_horizontal(self, url: str, user_a_hdr: dict, user_b_hdr: dict) -> Optional[IDORIssue]:
        if not self.allow_active:
            return IDORIssue(kind="IDOR_HORIZONTAL_POTENTIAL", url=url, description="Potential horizontal IDOR (safe_mode)", cwe="CWE-639", severity="MEDIUM")
        try:
            ra = await self.client.get(url, headers=user_a_hdr)
            rb = await self.client.get(url, headers=user_b_hdr)
            if rb.status_code == 200 and ra.status_code in (200, 403, 401) and rb.text and (rb.text == ra.text or len(rb.text) > 0):
                return IDORIssue(kind="IDOR_HORIZONTAL", url=url, description="User B accessed resource of User A", cwe="CWE-639", severity="HIGH")
        except Exception as e:
            return IDORIssue(kind="IDOR_HORIZONTAL_POTENTIAL", url=url, description=f"Active test failed: {e}", cwe="CWE-639", severity="MEDIUM")
        return None

    async def test_vertical(self, url: str, low_auth_hdr: dict) -> Optional[IDORIssue]:
        if not self.allow_active:
            return IDORIssue(kind="IDOR_VERTICAL_POTENTIAL", url=url, description="Potential vertical privilege escalation (safe_mode)", cwe="CWE-269", severity="MEDIUM")
        try:
            r = await self.client.get(url, headers=low_auth_hdr)
            if r.status_code == 200:
                return IDORIssue(kind="IDOR_VERTICAL", url=url, description="Low-privilege user accessed privileged endpoint", cwe="CWE-269", severity="CRITICAL")
        except Exception as e:
            return IDORIssue(kind="IDOR_VERTICAL_POTENTIAL", url=url, description=f"Active test failed: {e}", cwe="CWE-269", severity="MEDIUM")
        return None
