"""
掃描編排器 - 統一管理掃描流程的核心邏輯
"""



from typing import TYPE_CHECKING, Any

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
from .dynamic_engine.dynamic_content_extractor import (
    DynamicContentExtractor,
    ExtractionConfig,
)
from .dynamic_engine.headless_browser_pool import HeadlessBrowserPool, PoolConfig
from .fingerprint_manager import FingerprintCollector
from .header_configuration import HeaderConfiguration
from .info_gatherer.javascript_source_analyzer import JavaScriptSourceAnalyzer
from .info_gatherer.sensitive_info_detector import SensitiveInfoDetector
from .scan_context import ScanContext
from .strategy_controller import StrategyController, StrategyParameters

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ScanOrchestrator:
    """
    統一的掃描編排器 - 協調所有掃描組件

    改進點:
    - 整合動態和靜態爬蟲引擎
    - 根據策略自動選擇合適的引擎
    - 整合敏感信息檢測和 JavaScript 分析
    - 應用配置中心和策略控制器的設定
    - 使用增強的 HTTP 客戶端和 URL 隊列管理

    特性:
    - 支援多種掃描策略（FAST/DEEP/AGGRESSIVE 等）
    - 動態渲染頁面處理（JavaScript 執行）
    - 智能速率限制和重試
    - 實時資訊收集和分析
    - 詳細的進度追蹤
    """

    def __init__(self):
        """初始化掃描編排器"""
        self.static_parser = StaticContentParser()
        self.fingerprint_collector = FingerprintCollector()
        self.sensitive_detector = SensitiveInfoDetector()
        self.js_analyzer = JavaScriptSourceAnalyzer()

        # 動態引擎組件（延遲初始化）
        self.browser_pool: HeadlessBrowserPool | None = None
        self.dynamic_extractor: DynamicContentExtractor | None = None

        logger.info("ScanOrchestrator initialized")

    async def execute_scan(self, request: ScanStartPayload) -> ScanCompletedPayload:
        """
        執行完整的掃描流程

        Args:
            request: 掃描請求數據

        Returns:
            掃描完成結果
        """
        logger.info(f"Starting scan for request: {request.scan_id}")

        # 創建掃描上下文
        context = ScanContext(request)

        # 初始化策略
        strategy_controller = StrategyController(request.strategy)
        strategy_params = strategy_controller.get_parameters()

        logger.info(
            f"Scan strategy: {request.strategy}, "
            f"dynamic_scan: {strategy_params.enable_dynamic_scan}"
        )

        # 初始化組件
        auth_manager = AuthenticationManager(request.authentication)
        header_config = HeaderConfiguration(request.custom_headers)

        # 創建增強的 HTTP 客戶端
        http_client = HiHttpClient(
            auth_manager,
            header_config,
            requests_per_second=strategy_params.requests_per_second,
            per_host_rps=strategy_params.requests_per_second / 2,
            timeout=strategy_params.request_timeout,
            pool_size=strategy_params.connection_pool_size,
        )

        # 創建 URL 隊列管理器
        url_queue = UrlQueueManager(
            [str(t) for t in request.targets], max_depth=strategy_params.max_depth
        )

        # 如果策略啟用動態掃描,初始化瀏覽器池
        if strategy_params.enable_dynamic_scan:
            await self._init_dynamic_engine(strategy_params)

        try:
            # 執行爬蟲掃描
            await self._perform_crawling(
                context,
                url_queue,
                http_client,
                strategy_params,
            )

            # 設置最終指紋
            fingerprints = self.fingerprint_collector.get_final_fingerprints()
            if fingerprints:
                context.set_fingerprints(fingerprints)

            # 構建並返回結果
            result = self._build_scan_result(context)

            logger.info(
                f"Scan completed for {request.scan_id}: "
                f"{context.urls_found} URLs, {context.forms_found} forms, "
                f"duration: {context.scan_duration}s"
            )

            return result

        finally:
            # 清理資源
            await http_client.close()
            if self.browser_pool:
                await self.browser_pool.shutdown()

    async def _init_dynamic_engine(self, strategy_params: StrategyParameters) -> None:
        """
        初始化動態掃描引擎

        Args:
            strategy_params: 策略參數
        """
        logger.info("Initializing dynamic scan engine...")

        # 配置瀏覽器池
        pool_config = PoolConfig(
            min_instances=1,
            max_instances=strategy_params.browser_pool_size,
            headless=True,
            timeout_ms=int(strategy_params.page_load_timeout * 1000),
        )

        self.browser_pool = HeadlessBrowserPool(pool_config)
        await self.browser_pool.initialize()

        # 配置內容提取器
        extraction_config = ExtractionConfig(
            extract_forms=True,
            extract_links=True,
            extract_ajax=True,
            extract_api_calls=True,
            wait_for_network_idle=True,
        )

        self.dynamic_extractor = DynamicContentExtractor(extraction_config)

        logger.info("Dynamic scan engine initialized successfully")

    async def _perform_crawling(
        self,
        context: ScanContext,
        url_queue: UrlQueueManager,
        http_client: HiHttpClient,
        strategy_params: Any,
    ) -> None:
        """
        執行爬蟲掃描過程

        Args:
            context: 掃描上下文
            url_queue: URL 隊列管理器
            http_client: HTTP 客戶端
            strategy_params: 策略參數
        """
        pages_processed = 0
        max_pages = strategy_params.max_pages

        while url_queue.has_next() and pages_processed < max_pages:
            url = url_queue.next()

            logger.debug(f"Processing URL: {url}")

            # 根據策略選擇爬蟲引擎
            if strategy_params.enable_dynamic_scan and self.browser_pool:
                await self._process_url_dynamic(
                    url, context, url_queue, strategy_params
                )
            else:
                await self._process_url_static(
                    url, context, url_queue, http_client, strategy_params
                )

            pages_processed += 1
            context.increment_pages_crawled()

            # 定期報告進度
            if pages_processed % 10 == 0:
                stats = context.get_statistics()
                logger.info(f"Progress: {pages_processed} pages, {stats}")

    async def _process_url_static(
        self,
        url: str,
        context: ScanContext,
        url_queue: UrlQueueManager,
        http_client: HiHttpClient,
        strategy_params: Any,
    ) -> None:
        """
        使用靜態引擎處理 URL

        Args:
            url: 要處理的 URL
            context: 掃描上下文
            url_queue: URL 隊列
            http_client: HTTP 客戶端
            strategy_params: 策略參數
        """
        response = await http_client.get(url)

        if response is None:
            context.add_error("http_error", f"Failed to fetch {url}", url)
            return

        # 更新統計
        context.increment_urls_found()

        # 創建資產
        asset = Asset(asset_id=new_id("asset"), type="URL", value=url, has_form=False)
        context.add_asset(asset)

        # 收集指紋信息
        await self.fingerprint_collector.process_response(response)

        # 靜態內容解析
        parsed_assets, forms_count = self.static_parser.extract(url, response)
        for parsed_asset in parsed_assets:
            context.add_asset(parsed_asset)
        context.add_forms_found(forms_count)

        # 敏感信息檢測
        detection_result = self.sensitive_detector.detect_in_html(response.text, url)
        if detection_result.matches:
            logger.warning(
                f"Found {len(detection_result.matches)} sensitive info items in {url}"
            )
            # 可以將這些結果添加到上下文或發送到核心模組

        # JavaScript 源碼分析
        if response.headers.get("content-type", "").startswith("text/html"):
            analysis_result = self.js_analyzer.analyze(response.text, url)
            if analysis_result.sinks or analysis_result.patterns:
                total_findings = len(analysis_result.sinks) + len(
                    analysis_result.patterns
                )
                logger.info(f"Found {total_findings} JS findings in {url}")

    async def _process_url_dynamic(
        self,
        url: str,
        context: ScanContext,
        url_queue: UrlQueueManager,
        strategy_params: Any,
    ) -> None:
        """
        使用動態引擎處理 URL

        Args:
            url: 要處理的 URL
            context: 掃描上下文
            url_queue: URL 隊列
            strategy_params: 策略參數
        """
        if not self.browser_pool or not self.dynamic_extractor:
            logger.warning("Dynamic engine not initialized, falling back to static")
            return

        try:
            # 從瀏覽器池獲取頁面
            async with self.browser_pool.get_page() as page:
                # 提取動態內容
                dynamic_contents = await self.dynamic_extractor.extract_from_url(
                    url, page=page
                )

                logger.info(
                    f"Extracted {len(dynamic_contents)} dynamic contents from {url}"
                )

                # 處理提取的內容
                for content in dynamic_contents:
                    # 創建資產
                    asset = Asset(
                        asset_id=new_id("asset"),
                        type=content.content_type.value,
                        value=content.url,
                        has_form=content.content_type.value == "form",
                    )
                    context.add_asset(asset)

                    # 添加到 URL 隊列（如果是連結）
                    if content.content_type.value == "link":
                        url_queue.add(content.url, parent_url=url, depth=1)

                # 更新統計
                context.increment_urls_found()
                forms_found = sum(
                    1 for c in dynamic_contents if c.content_type.value == "form"
                )
                context.add_forms_found(forms_found)

                # 動態頁面也進行敏感信息檢測
                # 從 page.content() 獲取渲染後的 HTML 並分析
                # rendered_html = await page.content()
                # 可以進一步分析敏感信息和 JavaScript

        except Exception as e:
            logger.error(f"Dynamic processing failed for {url}: {e}")
            context.add_error("dynamic_error", str(e), url)

    def _build_scan_result(self, context: ScanContext) -> ScanCompletedPayload:
        """
        構建掃描結果

        Args:
            context: 掃描上下文

        Returns:
            掃描完成載荷
        """
        summary = Summary(
            urls_found=context.urls_found,
            forms_found=context.forms_found,
            apis_found=context.apis_found,
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
        logger.info("ScanOrchestrator reset")
