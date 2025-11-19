import json
from typing import Any

from services.aiva_common.enums import Topic
from services.aiva_common.mq import AbstractBroker
from services.aiva_common.schemas import (
    ScanCompletedPayload,
    Phase0StartPayload,
    Phase0CompletedPayload,
    Phase1StartPayload,
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScanModuleInterface:
    """掃描模組介面 - 資料接收與預處理

    負責接收掃描模組的原始數據並進行標準化處理，
    包含格式檢測、資料清理、去重、豐富化等功能。
    """

    def process_scan_data(self, payload: ScanCompletedPayload) -> dict[str, Any]:
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

    # ==================== Phase0/Phase1 兩階段掃描接口 ====================

    async def send_phase0_command(
        self,
        broker: AbstractBroker,
        scan_id: str,
        targets: list[str],
        trace_id: str,
        timeout_seconds: int = 600,
    ) -> None:
        """發送 Phase0 快速偵察命令

        Args:
            broker: 消息代理
            scan_id: 掃描 ID
            targets: 目標列表
            trace_id: 追蹤 ID
            timeout_seconds: 超時時間 (預設 10 分鐘)
        """
        phase0_payload = Phase0StartPayload(
            scan_id=scan_id,
            targets=targets,
            timeout_seconds=timeout_seconds,
        )

        message = {
            "trace_id": trace_id,
            "correlation_id": scan_id,
            "payload": phase0_payload.model_dump(),
        }

        await broker.publish(
            Topic.TASK_SCAN_PHASE0,
            json.dumps(message).encode("utf-8"),
        )

        logger.info(
            f"[Phase0] Command sent to {Topic.TASK_SCAN_PHASE0.value} "
            f"(scan_id={scan_id}, targets={len(targets)})"
        )

    async def send_phase1_command(
        self,
        broker: AbstractBroker,
        scan_id: str,
        targets: list[str],
        trace_id: str,
        phase0_result: Phase0CompletedPayload,
        selected_engines: list[str],
        max_depth: int = 3,
        max_urls: int = 1000,
        timeout_seconds: int = 1800,
    ) -> None:
        """發送 Phase1 深度掃描命令

        Args:
            broker: 消息代理
            scan_id: 掃描 ID
            targets: 目標列表
            trace_id: 追蹤 ID
            phase0_result: Phase0 結果
            selected_engines: 選中的引擎列表
            max_depth: 最大爬取深度 (預設 3)
            max_urls: 最大 URL 數量 (預設 1000)
            timeout_seconds: 超時時間 (預設 30 分鐘)
        """
        phase1_payload = Phase1StartPayload(
            scan_id=scan_id,
            targets=targets,
            phase0_result=phase0_result,
            selected_engines=selected_engines,
            max_depth=max_depth,
            max_urls=max_urls,
            timeout_seconds=timeout_seconds,
        )

        message = {
            "trace_id": trace_id,
            "correlation_id": scan_id,
            "payload": phase1_payload.model_dump(),
        }

        await broker.publish(
            Topic.TASK_SCAN_PHASE1,
            json.dumps(message).encode("utf-8"),
        )

        logger.info(
            f"[Phase1] Command sent to {Topic.TASK_SCAN_PHASE1.value} "
            f"(scan_id={scan_id}, engines={selected_engines}, max_depth={max_depth})"
        )

    async def process_phase0_result(
        self, payload: Phase0CompletedPayload
    ) -> dict[str, Any]:
        """處理 Phase0 結果

        Args:
            payload: Phase0 完成載荷

        Returns:
            處理後的 Phase0 數據
        """
        # 使用正確的 Phase0CompletedPayload 欄位
        summary = payload.summary
        fingerprints = payload.fingerprints
        assets = payload.assets
        recommendations = payload.recommendations
        
        # 計算技術數量
        tech_count = 0
        tech_list: list[str] = []
        if fingerprints:
            if fingerprints.framework:
                tech_list.extend(fingerprints.framework.keys())
                tech_count += len(fingerprints.framework)
            if fingerprints.language:
                tech_list.extend(fingerprints.language.keys())
                tech_count += len(fingerprints.language)
        
        processed = {
            "scan_id": payload.scan_id,
            "status": payload.status,
            "execution_time": payload.execution_time,
            # 從 summary 獲取統計
            "urls_found": summary.urls_found,
            "forms_found": summary.forms_found,
            "apis_found": summary.apis_found,
            # 從 fingerprints 獲取技術棧
            "discovered_technologies": tech_list,
            "tech_count": tech_count,
            # 從 assets 獲取資產
            "assets": assets,
            "asset_count": len(assets),
            # 從 recommendations 獲取風險評估
            "high_risk": recommendations.get("high_risk", False),
            "needs_js_engine": recommendations.get("needs_js_engine", False),
            "needs_form_testing": recommendations.get("needs_form_testing", False),
            "needs_api_testing": recommendations.get("needs_api_testing", False),
            # WAF 檢測
            "waf_detected": fingerprints.waf_detected if fingerprints else False,
            "waf_vendor": fingerprints.waf_vendor if fingerprints else None,
            # 遺留欄位 (為了兼容性)
            "sensitive_count": 0,  # Phase0 不再直接返回敏感資料數量
            "endpoint_count": summary.urls_found,
            "basic_endpoints": [],  # 已由 assets 替代
            "sensitive_data_found": [],  # 已由 assets 替代
            "risk_level": "high" if recommendations.get("high_risk", False) else "low",
        }

        logger.info(
            f"[Phase0] Result processed - "
            f"Technologies: {processed['tech_count']}, "
            f"URLs: {processed['urls_found']}, "
            f"Forms: {processed['forms_found']}, "
            f"APIs: {processed['apis_found']}, "
            f"Assets: {processed['asset_count']}, "
            f"Risk: {processed['risk_level']}"
        )

        return processed
