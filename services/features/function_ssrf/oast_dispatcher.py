

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

            events.append(
                OastEvent(
                    token=resolved,
                    evidence=evidence_text,
                    request=entry.get("request"),
                    response=entry.get("response"),
                )
            )

        return events

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
