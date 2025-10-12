from __future__ import annotations

import httpx

from ..authentication_manager import AuthenticationManager
from ..header_configuration import HeaderConfiguration


class HiHttpClient:
    """High-performance HTTP client with redirects and timeouts."""

    def __init__(
        self, auth: AuthenticationManager, headers: HeaderConfiguration
    ) -> None:
        self._auth = auth
        self._headers = headers

    async def get(self, url: str) -> httpx.Response | None:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20.0,
            headers=self._headers.user_headers,
        ) as client:
            try:
                return await client.get(url)
            except Exception:
                return None
