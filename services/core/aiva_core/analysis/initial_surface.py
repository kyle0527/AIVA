from typing import Any

from services.aiva_common.schemas import Asset, ScanCompletedPayload
from services.core.aiva_core.business_schemas import (
    AssetAnalysis,
    AttackSurfaceAnalysis,
    IdorCandidate,
    SqliCandidate,
    SsrfCandidate,
    XssCandidate,
)


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

    _XSS_PARAM_HINTS: tuple[str, ...] = (
        "q",
        "query",
        "search",
        "keyword",
        "name",
        "comment",
        "message",
        "text",
        "content",
        "input",
    )

    _SQLI_PARAM_HINTS: tuple[str, ...] = (
        "id",
        "user",
        "username",
        "email",
        "order",
        "sort",
        "filter",
        "category",
        "search",
    )

    _IDOR_PARAM_HINTS: tuple[str, ...] = (
        "id",
        "user_id",
        "account",
        "profile",
        "doc",
        "file",
        "order_id",
    )

    def analyze(self, payload: ScanCompletedPayload) -> AttackSurfaceAnalysis:
        forms = sum(1 for a in payload.assets if a.has_form)
        params = sum(len(a.parameters or []) for a in payload.assets)
        waf = bool(payload.fingerprints and payload.fingerprints.waf_detected)

        # 收集所有候選
        xss_candidates: list[XssCandidate] = []
        sqli_candidates: list[SqliCandidate] = []
        ssrf_candidates: list[SsrfCandidate] = []
        idor_candidates: list[IdorCandidate] = []

        # 資產分層
        high_risk_assets: list[AssetAnalysis] = []
        medium_risk_assets: list[AssetAnalysis] = []
        low_risk_assets: list[AssetAnalysis] = []

        for asset in payload.assets:
            # 檢測各類漏洞候選
            xss_candidates.extend(self._detect_xss_candidates(asset))
            sqli_candidates.extend(self._detect_sqli_candidates(asset))
            ssrf_candidates.extend(self._detect_ssrf_candidates(asset))
            idor_candidates.extend(self._detect_idor_candidates(asset))

            # 計算風險評分並分層
            risk_score = self._calculate_risk_score(asset)
            asset_analysis = AssetAnalysis(
                asset_id=asset.asset_id,
                url=str(asset.value),
                asset_type=asset.type,
                risk_score=risk_score,
                parameters=list(asset.parameters or []),
                has_form=asset.has_form,
            )

            if risk_score >= 70:
                high_risk_assets.append(asset_analysis)
            elif risk_score >= 40:
                medium_risk_assets.append(asset_analysis)
            else:
                low_risk_assets.append(asset_analysis)

        return AttackSurfaceAnalysis(
            scan_id=payload.scan_id,
            total_assets=len(payload.assets),
            forms=forms,
            parameters=params,
            waf_detected=waf,
            high_risk_assets=high_risk_assets,
            medium_risk_assets=medium_risk_assets,
            low_risk_assets=low_risk_assets,
            xss_candidates=xss_candidates,
            sqli_candidates=sqli_candidates,
            ssrf_candidates=ssrf_candidates,
            idor_candidates=idor_candidates,
        )

    def _summarize_asset(self, asset: Asset) -> dict[str, Any]:
        return {
            "asset_id": asset.asset_id,
            "type": asset.type,
            "value": asset.value,
            "parameters": list(asset.parameters or []),
            "has_form": asset.has_form,
        }

    def _calculate_risk_score(self, asset: Asset) -> int:
        """計算資產風險評分 (0-100)"""
        score = 50  # 基準分數

        # 有表單 +20
        if asset.has_form:
            score += 20

        # 參數數量影響
        param_count = len(asset.parameters or [])
        if param_count > 5:
            score += 15
        elif param_count > 2:
            score += 10
        elif param_count > 0:
            score += 5

        # 敏感參數名稱檢測
        parameters = [p.lower() for p in (asset.parameters or [])]
        sensitive_keywords = {"id", "user", "admin", "password", "token", "key"}
        if any(kw in param for param in parameters for kw in sensitive_keywords):
            score += 15

        return min(score, 100)  # 上限 100

    def _detect_xss_candidates(self, asset: Asset) -> list[XssCandidate]:
        """檢測 XSS 漏洞候選"""
        parameters = list(asset.parameters or [])
        if not parameters:
            return []

        location = "body" if asset.has_form else "query"
        candidates: list[XssCandidate] = []

        for parameter in parameters:
            lower_name = parameter.lower()
            keyword_hits = [
                hint for hint in self._XSS_PARAM_HINTS if hint in lower_name
            ]

            if keyword_hits:
                confidence = 0.6 if len(keyword_hits) > 1 else 0.4
                reasons = [
                    f"Parameter '{parameter}' suggests user input: {', '.join(keyword_hits)}"
                ]

                candidates.append(
                    XssCandidate(
                        asset_url=str(asset.value),
                        parameter=parameter,
                        location=location,
                        confidence=confidence,
                        reasons=reasons,
                        xss_type="reflected",
                    )
                )

        return candidates

    def _detect_sqli_candidates(self, asset: Asset) -> list[SqliCandidate]:
        """檢測 SQLi 漏洞候選"""
        parameters = list(asset.parameters or [])
        if not parameters:
            return []

        location = "body" if asset.has_form else "query"
        candidates: list[SqliCandidate] = []

        for parameter in parameters:
            lower_name = parameter.lower()
            keyword_hits = [
                hint for hint in self._SQLI_PARAM_HINTS if hint in lower_name
            ]

            if keyword_hits:
                confidence = 0.7 if "id" in keyword_hits else 0.5
                reasons = [
                    f"Parameter '{parameter}' suggests database query: {', '.join(keyword_hits)}"
                ]

                candidates.append(
                    SqliCandidate(
                        asset_url=str(asset.value),
                        parameter=parameter,
                        location=location,
                        confidence=confidence,
                        reasons=reasons,
                        error_based_possible=True,
                    )
                )

        return candidates

    def _detect_ssrf_candidates(self, asset: Asset) -> list[SsrfCandidate]:
        """檢測 SSRF 漏洞候選"""
        parameters = list(asset.parameters or [])
        if not parameters:
            return []

        location = "body" if asset.has_form else "query"
        path = str(asset.value).lower()
        path_hints = [hint for hint in self._SSRF_PATH_HINTS if hint in path]

        candidates: list[SsrfCandidate] = []
        for parameter in parameters:
            reasons = self._evaluate_parameter(parameter, path_hints)
            if not reasons:
                continue

            confidence = 0.7 if any("callback" in reason for reason in reasons) else 0.5

            candidates.append(
                SsrfCandidate(
                    asset_url=str(asset.value),
                    parameter=parameter,
                    location=location,
                    confidence=confidence,
                    reasons=reasons,
                    target_type="url_parameter",
                )
            )

        return candidates

    def _detect_idor_candidates(self, asset: Asset) -> list[IdorCandidate]:
        """檢測 IDOR 漏洞候選"""
        parameters = list(asset.parameters or [])
        if not parameters:
            return []

        location = "body" if asset.has_form else "query"
        candidates: list[IdorCandidate] = []

        for parameter in parameters:
            lower_name = parameter.lower()
            keyword_hits = [
                hint for hint in self._IDOR_PARAM_HINTS if hint in lower_name
            ]

            if keyword_hits:
                confidence = 0.6 if "id" in keyword_hits else 0.4
                reasons = [
                    f"Parameter '{parameter}' suggests resource reference: {', '.join(keyword_hits)}"
                ]

                # 推斷 ID 模式
                id_pattern = "numeric" if "id" in lower_name else "uuid"

                candidates.append(
                    IdorCandidate(
                        asset_url=str(asset.value),
                        parameter=parameter,
                        location=location,
                        confidence=confidence,
                        reasons=reasons,
                        resource_type="user" if "user" in lower_name else "document",
                        id_pattern=id_pattern,
                    )
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
