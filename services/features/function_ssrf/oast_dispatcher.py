

from dataclasses import dataclass
import json
import os

import httpx

from services.aiva_common.schemas import FunctionTaskPayload


@dataclass
class OastProbe:
    token: str
    callback_url: str


@dataclass
class OastEvent:
    token: str
    evidence: str | None = None
    request: str | None = None
    response: str | None = None


class OastDispatcher:
    """Thin client for the shared OAST service used for blind SSRF validation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = (
            base_url or os.getenv("OAST_SERVICE_URL") or "http://localhost:8083"
        ).rstrip("/")
        self._client = client
        self._timeout = timeout
        self._tokens: dict[str, OastProbe] = {}

    async def register(self, task: FunctionTaskPayload) -> OastProbe:
        if task.task_id in self._tokens:
            return self._tokens[task.task_id]

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout
        )

        try:
            response = await client.post(
                "/register",
                json={
                    "task_id": task.task_id,
                    "scan_id": task.scan_id,
                    "callback_base_url": self._base_url,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - defensive network guard
            raise RuntimeError(
                "Failed to register SSRF probe with OAST service"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        token = payload.get("token")
        if not token:
            raise RuntimeError("OAST service did not provide a token")

        callback_url = payload.get("callback_url")
        if not callback_url:
            path = payload.get("callback_path") or f"/oast/{token}"
            callback_url = f"{self._base_url}{path}"

        probe = OastProbe(token=token, callback_url=callback_url)
        self._tokens[task.task_id] = probe
        self._tokens[callback_url] = probe
        self._tokens[token] = probe
        return probe

    async def fetch_events(self, token: str) -> list[OastEvent]:
        resolved = self._resolve_token(token)

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout
        )

        try:
            response = await client.get(f"/events/{resolved}")
            if response.status_code == 404:
                return []
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - defensive network guard
            raise RuntimeError("Failed to fetch SSRF OAST events") from exc
        finally:
            if owns_client:
                await client.aclose()

        events: list[OastEvent] = []
        for entry in payload.get("events", []):
            evidence = entry.get("evidence")
            if isinstance(evidence, dict | list):
                evidence_text = json.dumps(evidence)
            elif evidence is None:
                evidence_text = None
            else:
                evidence_text = str(evidence)

            event = OastEvent(
                token=resolved,
                evidence=evidence_text,
                request=entry.get("request"),
                response=entry.get("response"),
            )
            
            # 驗證 OAST 事件的真實性
            if self._validate_oast_event(event):
                events.append(event)

        return events
    
    def _validate_oast_event(self, event: OastEvent) -> bool:
        """驗證 OAST 事件是否為真實 SSRF，過濾虛假回呼"""
        # 檢查是否有請求內容
        if not event.request:
            return False
        
        # 檢查是否包含真實的 HTTP 請求特徵
        request_indicators = [
            "GET ", "POST ", "PUT ", "DELETE ",
            "Host:", "User-Agent:", "Accept:",
            "HTTP/1.", "HTTP/2"
        ]
        if not any(indicator in event.request for indicator in request_indicators):
            return False
        
        # 檢查是否為預期的回呼來源 (非瀏覽器預載入等)
        if event.request and "User-Agent:" in event.request:
            # 排除瀏覽器預載入和掃描器
            browser_preload_patterns = [
                "Chrome-Lighthouse", "Googlebot", "bingbot",
                "Prefetch", "Preload", "Scanner"
            ]
            if any(pattern in event.request for pattern in browser_preload_patterns):
                return False
        
        # 檢查回應內容 (如果有)
        if event.response:
            # 確認有實質回應內容
            if len(event.response.strip()) < 10:
                return False
        
        return True

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    def _resolve_token(self, token: str) -> str:
        if token in self._tokens:
            return self._tokens[token].token

        normalized = token.rstrip("/")
        for probe in self._tokens.values():
            if not isinstance(probe, OastProbe):
                continue
            if normalized.startswith(probe.callback_url.rstrip("/")):
                return probe.token

        parts = normalized.split("/")
        return parts[-1] if parts else token
