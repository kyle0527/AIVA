"""
掃描編排器 - 統一管理掃描流程的核心邏輯
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.aiva_common.schemas import (
    Asset,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
)
from services.aiva_common.utils import get_logger, new_id

from .authentication_manager import AuthenticationManager
from .core_crawling_engine.http_client_hi import HiHttpClient
from .core_crawling_engine.static_content_parser import StaticContentParser
from .core_crawling_engine.url_queue_manager import UrlQueueManager
from .fingerprint_manager import FingerprintCollector
from .header_configuration import HeaderConfiguration

if TYPE_CHECKING:
    from .scan_context import ScanContext  # type: ignore[attr-defined]

logger = get_logger(__name__)


class ScanOrchestrator:
    """統一的掃描編排器 - 協調所有掃描組件"""

    def __init__(self):
        self.static_parser = StaticContentParser()
        self.fingerprint_collector = FingerprintCollector()

    async def execute_scan(self, request: ScanStartPayload) -> ScanCompletedPayload:
        """執行完整的掃描流程"""
        logger.debug(f"Starting scan for request: {request.scan_id}")

        # 創建掃描上下文
        context = ScanContext(request)

        # 初始化組件
        auth_manager = AuthenticationManager(request.authentication)
        header_config = HeaderConfiguration(request.custom_headers)
        url_queue = UrlQueueManager([str(t) for t in request.targets])
        http_client = HiHttpClient(auth_manager, header_config)

        # 執行掃描
        await self._perform_crawling(context, url_queue, http_client)

        # 設置最終指紋
        context.set_fingerprints(self.fingerprint_collector.get_final_fingerprints())

        # 構建並返回結果
        result = self._build_scan_result(context)

        logger.debug(
            f"Scan completed for {request.scan_id}: "
            f"{context.urls_found} URLs, {context.forms_found} forms, "
            f"duration: {context.scan_duration}s"
        )

        return result

    async def _perform_crawling(
        self,
        context: ScanContext,
        url_queue: UrlQueueManager,
        http_client: HiHttpClient,
    ) -> None:
        """執行爬蟲掃描過程"""
        while url_queue.has_next():
            url = url_queue.next()
            response = await http_client.get(url)

            if response is None:
                continue

            # 更新統計
            context.increment_urls_found()

            # 創建資產
            asset = Asset(
                asset_id=new_id("asset"), type="URL", value=url, has_form=False
            )
            context.add_asset(asset)

            # 收集指紋信息
            await self.fingerprint_collector.process_response(response)

            # 靜態內容解析
            parsed_assets, forms_count = self.static_parser.extract(url, response)
            for parsed_asset in parsed_assets:
                context.add_asset(parsed_asset)
            context.add_forms_found(forms_count)

    def _build_scan_result(self, context: ScanContext) -> ScanCompletedPayload:
        """構建掃描結果"""
        summary = Summary(
            urls_found=context.urls_found,
            forms_found=context.forms_found,
            apis_found=0,  # 目前尚未實現API發現
            scan_duration_seconds=context.scan_duration,
        )

        return ScanCompletedPayload(
            scan_id=context.request.scan_id,
            status="completed",
            summary=summary,
            assets=context.assets,
            fingerprints=context.fingerprints,
        )

    def reset(self) -> None:
        """重置編排器狀態"""
        self.fingerprint_collector.reset()
