from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import httpx

from services.aiva_common.schemas import FunctionTaskPayload
from services.aiva_common.utils import new_id


@dataclass
class BlindXssEvent:
    """Represents a callback received by the OAST/blind listener."""

    token: str
    request: str | None = None
    response: str | None = None
    evidence: str | None = None


class BlindCallbackStore(Protocol):
    async def register_probe(self, task: FunctionTaskPayload) -> str: ...

    async def fetch_events(self, token: str) -> Iterable[BlindXssEvent]: ...


class _NullBlindCallbackStore:
    """Fallback implementation that issues synthetic payloads."""

    async def register_probe(
        self, task: FunctionTaskPayload
    ) -> str:  # pragma: no cover - trivial
        return f"https://oast.invalid/{new_id('bxss')}"

    async def fetch_events(
        self, token: str
    ) -> Iterable[BlindXssEvent]:  # pragma: no cover - trivial
        return []


class OastHttpCallbackStore:
    """Interact with the shared OAST service over HTTP."""

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
        self._payload_tokens: dict[str, str] = {}

    async def register_probe(self, task: FunctionTaskPayload) -> str:
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
                    "target": str(task.target.url),
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = cast(dict[str, Any], response.json())
        except httpx.HTTPError as exc:  # pragma: no cover - network failure safety
            raise RuntimeError(
                "Failed to register blind XSS probe with OAST service"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        token: str | None = cast(str | None, payload.get("token"))
        if not token:
            raise RuntimeError("OAST service response missing token")

        callback_url: str | None = cast(str | None, payload.get("callback_url"))
        if not callback_url:
            callback_path: str | None = cast(str | None, payload.get("callback_path"))
            callback_path = callback_path or f"/oast/{token}"
            callback_url = f"{self._base_url}{callback_path}"

        # At this point, callback_url must be a string
        assert isinstance(callback_url, str)
        self._payload_tokens[callback_url] = token
        return callback_url

    async def fetch_events(self, token: str) -> Iterable[BlindXssEvent]:
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
            payload: dict[str, Any] = cast(dict[str, Any], response.json())
        except httpx.HTTPError as exc:  # pragma: no cover - network failure safety
            raise RuntimeError(
                "Failed to fetch blind XSS events from OAST service"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        events: list[BlindXssEvent] = []
        for entry in cast(list[dict[str, Any]], payload.get("events", [])):
            evidence: Any | None = entry.get("evidence")
            if isinstance(evidence, dict | list):
                evidence_text: str | None = json.dumps(evidence)
            elif evidence is not None:
                evidence_text = str(evidence)
            else:
                evidence_text = None

            events.append(
                BlindXssEvent(
                    token=token,
                    request=cast(str | None, entry.get("request")),
                    response=cast(str | None, entry.get("response")),
                    evidence=evidence_text,
                )
            )

        return events

    def _resolve_token(self, payload_identifier: str) -> str:
        if payload_identifier in self._payload_tokens:
            return self._payload_tokens[payload_identifier]

        parsed = urlparse(payload_identifier)
        if parsed.path:
            token = parsed.path.rstrip("/").split("/")[-1]
            self._payload_tokens[payload_identifier] = token
            return token

        return payload_identifier


class BlindXssListenerValidator:
    """Utility responsible for provisioning blind payloads and polling callbacks."""

    def __init__(self, store: BlindCallbackStore | None = None) -> None:
        if store is None:
            base_url = os.getenv("OAST_SERVICE_URL")
            store = (
                OastHttpCallbackStore(base_url=base_url)
                if base_url
                else _NullBlindCallbackStore()
            )
        self._store = store
        self._token: str | None = None

    async def provision_payload(self, task: FunctionTaskPayload) -> str:
        if not self._token:
            self._token = await self._store.register_probe(task)
        return self._token

    async def collect_events(self) -> list[BlindXssEvent]:
        if not self._token:
            return []

        events = await self._store.fetch_events(self._token)
        return list(events)
