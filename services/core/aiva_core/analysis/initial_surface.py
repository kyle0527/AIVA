from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from services.aiva_common.schemas import Asset, ScanCompletedPayload


class InitialAttackSurface:
    """Compute initial attack surface from scan results."""

    _SSRF_PARAM_HINTS: tuple[str, ...] = (
        "url",
        "uri",
        "target",
        "dest",
        "redirect",
        "return",
        "callback",
        "webhook",
        "endpoint",
        "host",
        "domain",
    )

    _SSRF_PATH_HINTS: tuple[str, ...] = (
        "proxy",
        "redirect",
        "callback",
        "webhook",
        "fetch",
        "relay",
        "resource",
    )

    def analyze(self, payload: ScanCompletedPayload) -> dict[str, Any]:
        forms = sum(1 for a in payload.assets if a.has_form)
        params = sum(len(a.parameters or []) for a in payload.assets)
        waf = bool(payload.fingerprints and payload.fingerprints.waf_detected)

        asset_details: list[dict[str, Any]] = []
        ssrf_candidates: list[dict[str, Any]] = []

        for asset in payload.assets:
            detail = self._summarize_asset(asset)
            asset_details.append(detail)

            for candidate in self._detect_ssrf_candidates(asset):
                ssrf_candidates.append(candidate)

        return {
            "forms": forms,
            "parameters": params,
            "waf": waf,
            "assets": [a.value for a in payload.assets],
            "asset_details": asset_details,
            "ssrf_candidates": ssrf_candidates,
        }

    def _summarize_asset(self, asset: Asset) -> dict[str, Any]:
        return {
            "asset_id": asset.asset_id,
            "type": asset.type,
            "value": asset.value,
            "parameters": list(asset.parameters or []),
            "has_form": asset.has_form,
        }

    def _detect_ssrf_candidates(self, asset: Asset) -> Iterable[dict[str, Any]]:
        parameters = list(asset.parameters or [])
        if not parameters:
            return []

        location = "body" if asset.has_form else "query"
        path = str(asset.value).lower()
        path_hints = [hint for hint in self._SSRF_PATH_HINTS if hint in path]

        candidates: list[dict[str, Any]] = []
        for parameter in parameters:
            reasons = self._evaluate_parameter(parameter, path_hints)
            if not reasons:
                continue

            priority = 5 if any("callback" in reason for reason in reasons) else 4
            candidates.append(
                {
                    "asset": asset.value,
                    "parameter": parameter,
                    "location": location,
                    "reasons": reasons,
                    "priority": priority,
                }
            )

        return candidates

    def _evaluate_parameter(self, parameter: str, path_hints: list[str]) -> list[str]:
        lower_name = parameter.lower()
        keyword_hits = [hint for hint in self._SSRF_PARAM_HINTS if hint in lower_name]

        reasons: list[str] = []
        if keyword_hits:
            reasons.append(
                f"Parameter '{parameter}' contains SSRF-related keyword(s): "
                f"{', '.join(keyword_hits)}"
            )

        if path_hints:
            reasons.append(
                "URL path includes SSRF hint(s): {hints}".format(
                    hints=", ".join(path_hints)
                )
            )

        return reasons
