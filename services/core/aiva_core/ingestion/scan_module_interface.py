from typing import Any

from services.aiva_common.schemas import ScanCompletedPayload


class ScanModuleInterface:
    """掃描模組介面 - 資料接收與預處理

    負責接收掃描模組的原始數據並進行標準化處理，
    包含格式檢測、資料清理、去重、豐富化等功能。
    """

    async def process_scan_data(self, payload: ScanCompletedPayload) -> dict[str, Any]:
        """處理掃描模組回傳的原始數據

        Args:
            payload: 掃描完成的負載數據

        Returns:
            標準化處理後的資料結構
        """
        processed_data = {
            "scan_id": payload.scan_id,
            "status": payload.status,
            "summary": payload.summary,
            "assets": self._process_assets(payload.assets),
            "fingerprints": self._process_fingerprints(payload.fingerprints),
        }

        return processed_data

    def _process_assets(self, assets: list[Any]) -> list[dict[str, Any]]:
        """處理資產清單，進行分類與標準化

        Args:
            assets: 原始資產清單

        Returns:
            處理後的資產清單
        """
        processed_assets = []

        for asset in assets:
            processed_asset = {
                "asset_id": asset.asset_id,
                "type": asset.type,
                "value": asset.value,
                "parameters": asset.parameters,
                "has_form": asset.has_form,
                "risk_score": self._calculate_risk_score(asset),
                "categories": self._categorize_asset(asset),
            }
            processed_assets.append(processed_asset)

        return processed_assets

    def _process_fingerprints(self, fingerprints: Any) -> dict[str, Any]:
        """處理技術指紋資料

        Args:
            fingerprints: 原始指紋資料

        Returns:
            處理後的指紋資料
        """
        if not fingerprints:
            return {}

        return {
            "web_server": fingerprints.web_server or {},
            "framework": fingerprints.framework or {},
            "language": fingerprints.language or {},
            "waf_detected": fingerprints.waf_detected,
            "waf_vendor": fingerprints.waf_vendor,
        }

    def _calculate_risk_score(self, asset: Any) -> int:
        """計算資產風險分數

        Args:
            asset: 資產物件

        Returns:
            風險分數 (1-10)
        """
        risk_score = 1

        # 基於資產類型調整風險分數
        if asset.type == "URL":
            if asset.has_form:
                risk_score += 3  # 有表單的頁面風險較高
            if asset.parameters:
                risk_score += len(asset.parameters)  # 參數越多風險越高
        elif asset.type == "API":
            risk_score += 2  # API端點通常風險較高

        return min(risk_score, 10)  # 最高10分

    def _categorize_asset(self, asset: Any) -> list[str]:
        """資產分類

        Args:
            asset: 資產物件

        Returns:
            資產類別標籤列表
        """
        categories: list[str] = []

        if not hasattr(asset, "value") or not asset.value:
            return categories

        url_path = str(asset.value).lower()

        # 功能分類
        if any(keyword in url_path for keyword in ["login", "signin", "auth"]):
            categories.append("authentication")
        if any(keyword in url_path for keyword in ["admin", "manage", "dashboard"]):
            categories.append("administration")
        if any(keyword in url_path for keyword in ["api", "rest", "graphql"]):
            categories.append("api")
        if any(keyword in url_path for keyword in ["upload", "file", "download"]):
            categories.append("file_handling")
        if any(keyword in url_path for keyword in ["search", "query"]):
            categories.append("search")
        if asset.has_form:
            categories.append("form_input")
        if asset.parameters:
            categories.append("parameterized")

        return categories
