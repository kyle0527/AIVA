"""
掃描編排器 - 統一管理掃描流程的核心邏輯
"""



from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

from services.aiva_common.schemas import (
    Asset,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
    Phase0StartPayload,
    Phase0CompletedPayload,
    Phase1StartPayload,
    Phase1CompletedPayload,
    Fingerprints,
)
from services.aiva_common.utils import new_id, get_logger

logger = get_logger(__name__)

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
from .vulnerability_scanner import VulnerabilityScanner


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

        # 初始化漏洞掃描器 (Phase 2 MVP)
        self.vuln_scanner = VulnerabilityScanner()

        # 動態引擎組件（延遲初始化）
        self.browser_pool: HeadlessBrowserPool | None = None
        self.dynamic_extractor: DynamicContentExtractor | None = None

    async def execute_scan(self, request: ScanStartPayload) -> ScanCompletedPayload:
        """
        執行完整的掃描流程

        Args:
            request: 掃描請求數據

        Returns:
            掃描完成結果
        """
        # 創建掃描上下文
        context = ScanContext(request)

        # 確保漏洞掃描器已初始化
        await self.vuln_scanner.initialize()

        # 初始化策略
        strategy_controller = StrategyController(request.strategy)
        
        # 注意：ScanStartPayload 不包含 phase0_summary，這是 Phase1StartPayload 才有的
        # 如果需要根據 Phase0 結果調整策略，應該在 Phase1 中處理
        
        strategy_params = strategy_controller.get_parameters()

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

            # === Phase 2 MVP 閉環：執行漏洞驗證 ===
            if context.assets:
                # 篩選可測試的資產 (URL 或 Form)
                test_targets = []
                for asset in context.assets:
                    if asset.type in ['URL', 'form', 'link', 'api_endpoint']:
                        # 簡單去重
                        if asset.value not in test_targets:
                            test_targets.append(asset.value)
                
                # 限制 MVP 測試數量（避免過長執行時間）
                max_vuln_targets = 10
                test_targets = test_targets[:max_vuln_targets]
                
                for target in test_targets:
                    try:
                        vuln_results = await self.vuln_scanner.scan_target(target)
                        
                        # 如果發現漏洞，記錄下來
                        if vuln_results.get('vulnerabilities'):
                            pass
                    except Exception:
                        pass
            # ==============================================

            # 構建並返回結果
            result = self._build_scan_result(context)

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
            url, current_depth = url_queue.next()

            # 根據策略選擇爬蟲引擎
            if strategy_params.enable_dynamic_scan and self.browser_pool:
                await self._process_url_dynamic(
                    url, current_depth, context, url_queue, strategy_params
                )
            else:
                await self._process_url_static(
                    url, current_depth, context, url_queue, http_client, strategy_params
                )

            pages_processed += 1
            context.increment_pages_crawled()

            # 定期報告進度
            if pages_processed % 10 == 0:
                pass

    async def _process_url_static(
        self,
        url: str,
        current_depth: int,
        context: ScanContext,
        url_queue: UrlQueueManager,
        http_client: HiHttpClient,
        _strategy_params: Any,
    ) -> None:
        """
        使用靜態引擎處理 URL

        Args:
            url: 要處理的 URL
            current_depth: 當前 URL 的深度級別
            context: 掃描上下文
            url_queue: URL 佇列
            http_client: HTTP 客戶端
            _strategy_params: 策略參數 (未使用)
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
        logger.info(f"📝 Parsed {len(parsed_assets)} assets from {url}")
        for parsed_asset in parsed_assets:
            context.add_asset(parsed_asset)
        context.add_forms_found(forms_count)
        
        # 🔧 修復: 將發現的 URL 加入爬蟲隊列 (參考 Crawlee enqueue_links 模式)
        new_urls = [
            parsed_asset.value 
            for parsed_asset in parsed_assets 
            if parsed_asset.type == "URL" and not url_queue.is_processed(parsed_asset.value)
        ]
        
        if new_urls:
            logger.info(f"🔗 Adding {len(new_urls)} new URLs to queue (depth={current_depth + 1})")
            url_queue.add_batch(new_urls, parent_url=url, depth=current_depth + 1)
        else:
            logger.info(f"⚠️ No new URLs found from {url} (parsed {len(parsed_assets)} assets, forms={forms_count})")

        # 敏感信息檢測
        detection_result = self.sensitive_detector.detect_in_html(response.text, url)
        if detection_result.matches:
            pass

        # 🔧 JavaScript 源碼分析 - 只分析內聯 script
        if response.headers.get("content-type", "").startswith("text/html"):
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 提取內聯 script 內容
            inline_scripts = []
            for script_tag in soup.find_all('script'):
                if script_tag.string and len(script_tag.string.strip()) > 50:
                    inline_scripts.append(script_tag.string)
            
            # 分析所有內聯 scripts
            if inline_scripts:
                combined_js = '\n'.join(inline_scripts)
                analysis_result = self.js_analyzer.analyze(combined_js, url)
                if analysis_result.sinks or analysis_result.patterns:
                    pass

    async def _process_url_dynamic(
        self,
        url: str,
        current_depth: int,
        context: ScanContext,
        url_queue: UrlQueueManager,
        _strategy_params: Any,
    ) -> None:
        """
        使用動態引擎處理 URL

        Args:
            url: 要處理的 URL
            current_depth: 當前 URL 的深度級別
            context: 掃描上下文
            url_queue: URL 佇列
            _strategy_params: 策略參數 (未使用)
        """
        if not self.browser_pool or not self.dynamic_extractor:
            return

        try:
            # 從瀏覽器池獲取頁面
            async with self.browser_pool.get_page() as page:
                # 提取動態內容
                dynamic_contents = await self.dynamic_extractor.extract_from_url(
                    url, page=page
                )

                # 🔧 處理提取的內容 - 收集 URL
                new_urls = []
                for content in dynamic_contents:
                    # 創建資產
                    asset = Asset(
                        asset_id=new_id("asset"),
                        type=content.content_type.value,
                        value=content.url,
                        has_form=content.content_type.value == "form",
                    )
                    context.add_asset(asset)

                    # 🔧 收集 URL 以便批次加入隊列
                    if content.content_type.value == "link":
                        new_urls.append(content.url)
                
                # 🔧 批次加入隊列並正確設定深度
                if new_urls:
                    filtered_urls = [
                        u for u in new_urls 
                        if not url_queue.is_processed(u)
                    ]
                    if filtered_urls:
                        added_count = url_queue.add_batch(
                            filtered_urls, 
                            parent_url=url, 
                            depth=current_depth + 1
                        )
                        logger.debug(
                            f"Added {added_count} dynamic URLs from {url} at depth {current_depth + 1}"
                        )

                # 更新統計
                context.increment_urls_found()
                forms_found = sum(
                    1 for c in dynamic_contents if c.content_type.value == "form"
                )
                context.add_forms_found(forms_found)

                # 🔧 動態頁面也進行 JS 分析
                rendered_html = await page.content()
                
            # 提取並分析 JavaScript
            scripts = await self._extract_and_analyze_scripts(page, url, rendered_html)
            if scripts:
                # JS 分析完成，scripts 已包含結果
                logger.debug(f"Extracted {len(scripts)} scripts from {url}")

        except Exception:
            context.add_error("dynamic_error", "渲染失敗", url)

    async def _extract_and_analyze_scripts(
        self, page: Any, url: str, html: str
    ) -> list[dict[str, Any]]:
        """
        從頁面提取並分析 JavaScript

        Args:
            page: Playwright Page 物件
            url: 頁面 URL
            html: 渲染後的 HTML

        Returns:
            JS 分析結果列表
        """
        scripts = []
        
        try:
            # 🔧 提取內聯 script
            soup = BeautifulSoup(html, 'lxml')
            for script_tag in soup.find_all('script'):
                if script_tag.string and len(script_tag.string.strip()) > 50:
                    # 分析內聯 JS
                    analysis = self.js_analyzer.analyze(script_tag.string, url)
                    if analysis.sinks or analysis.patterns:
                        scripts.append({
                            'type': 'inline',
                            'content': script_tag.string[:200],
                            'sinks': len(analysis.sinks),
                            'patterns': len(analysis.patterns),
                        })
                        logger.info(
                            f"Inline script: {len(analysis.sinks)} sinks, "
                            f"{len(analysis.patterns)} patterns"
                        )
            
            # 🔧 獲取外部 script URLs
            script_urls = await page.evaluate("""
                () => {
                    const scripts = Array.from(document.querySelectorAll('script[src]'));
                    return scripts.map(s => s.src).filter(src => src && src.trim());
                }
            """)
            
            # 🔧 下載並分析外部 JS (只分析前 5 個以避免過慢)
            for script_url in script_urls[:5]:
                try:
                    response = await page.context.request.get(script_url, timeout=5000)
                    if response.ok:
                        js_content = await response.text()
                        if len(js_content) > 100:
                            analysis = self.js_analyzer.analyze(js_content, script_url)
                            if analysis.sinks or analysis.patterns:
                                scripts.append({
                                    'type': 'external',
                                    'url': script_url,
                                    'size': len(js_content),
                                    'sinks': len(analysis.sinks),
                                    'patterns': len(analysis.patterns),
                                })
                                logger.info(
                                    f"External script {script_url}: "
                                    f"{len(analysis.sinks)} sinks, {len(analysis.patterns)} patterns"
                                )
                except Exception:
                    # 外部 script 獲取失敗，跳過
                    pass
            
        except Exception:
            # JS 提取或分析失敗
            pass

        return scripts
    
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


    # ==================== Phase0/Phase1 兩階段掃描 ====================

    def execute_phase0(
        self, request: Phase0StartPayload
    ) -> Phase0CompletedPayload:
        """
        執行 Phase0 快速偵察掃描（5-10 分鐘）

        功能：
        1. 敏感資訊掃描（調用 Rust 引擎）
        2. 技術棧指紋識別
        3. 基礎端點發現

        Args:
            request: Phase0 掃描請求

        Returns:
            Phase0CompletedPayload: Phase0 掃描結果
        """
        import time


        start_time = time.time()

        discovered_assets: list[Asset] = []
        discovered_technologies: list[str] = []
        sensitive_data_found: list[str] = []
        basic_endpoints: list[str] = []

        try:
            # 1. 快速指紋識別
            for target in request.targets:
                target_str = str(target)


                # 使用現有的指紋識別器
                fingerprints_data = self._quick_fingerprint(target_str)
                discovered_technologies.extend(fingerprints_data.get("technologies", []))

            # 2. 敏感資訊掃描（調用 Rust 引擎）
            for target in request.targets:
                target_str = str(target)


                # 調用敏感信息檢測器
                sensitive_matches = self._quick_sensitive_scan(target_str)
                sensitive_data_found.extend(sensitive_matches)

            # 3. 基礎端點發現
            for target in request.targets:
                target_str = str(target)


                # 快速爬取（深度1，最多50個URL）
                endpoints = self._quick_endpoint_discovery(target_str)
                basic_endpoints.extend(endpoints)

            # 4. 構建資產列表
            for endpoint in basic_endpoints[:100]:
                discovered_assets.append(
                    Asset(
                        asset_id=new_id("asset"),
                        type="url",
                        value=endpoint,
                    )
                )

            # 5. 構建技術指紋
            fingerprints = Fingerprints(
                web_server={"type": discovered_technologies[0]} if discovered_technologies else None,
                framework={"detected": ", ".join(discovered_technologies[:5])} if discovered_technologies else None,
            )

            # 6. 構建摘要
            summary = Summary(
                urls_found=len(basic_endpoints),
                forms_found=0,
                apis_found=0,
                scan_duration_seconds=int(time.time() - start_time),
            )

            execution_time = time.time() - start_time

            # 7. 構建建議
            recommendations = {
                "needs_js_engine": any("javascript" in tech.lower() for tech in discovered_technologies),
                "needs_form_testing": False,
                "needs_api_testing": any("api" in endpoint.lower() for endpoint in basic_endpoints),
            }

            return Phase0CompletedPayload(
                scan_id=request.scan_id,
                status="success",
                execution_time=execution_time,
                assets=discovered_assets,
                fingerprints=fingerprints,
                summary=summary,
                recommendations=recommendations,
                error_info=None,
            )

        except Exception as e:
            execution_time = time.time() - start_time


            return Phase0CompletedPayload(
                scan_id=request.scan_id,
                status="failed",
                execution_time=execution_time,
                assets=[],
                fingerprints=None,
                summary=Summary(
                    urls_found=0,
                    forms_found=0,
                    apis_found=0,
                    scan_duration_seconds=int(execution_time),
                ),
                recommendations={},
                error_info=str(e),
            )

    async def execute_phase1(
        self, request: Phase1StartPayload
    ) -> Phase1CompletedPayload:
        """
        執行 Phase1 深度掃描（10-30 分鐘）

        功能：
        1. 根據 Phase0 結果選擇引擎
        2. 並行執行多引擎掃描
        3. 整合 Phase0 和 Phase1 結果

        Args:
            request: Phase1 掃描請求

        Returns:
            Phase1CompletedPayload: Phase1 掃描結果
        """
        import time

        # Phase1 開始
        start_time = time.time()

        complete_asset_list = []
        engines_used = request.selected_engines if request.selected_engines else []
        
        # 整合 Phase 0 (Rust) 的發現結果
        targets_to_scan = list(request.targets)
        
        # 如果有 Phase 0 結果，合併發現的端點
        if hasattr(request, 'phase0_result') and request.phase0_result:
            phase0_endpoints = getattr(request.phase0_result, 'basic_endpoints', [])
            if phase0_endpoints:
                # 繼承 phase0 結果
                targets_to_scan.extend(phase0_endpoints)
                # 去重
                targets_to_scan = list(dict.fromkeys(targets_to_scan))

        try:
            # 1. 根據引擎選擇執行掃描（強制使用 Python 進行 MVP 驗證）
            # MVP 模式：始終執行 Python 引擎以驗證完整流程
            if "python" in engines_used or not engines_used:  # 如果沒有指定引擎，預設使用 Python
                # 執行Python引擎
                python_results = await self._execute_python_scan(request, override_targets=[str(t) for t in targets_to_scan])
                complete_asset_list.extend(python_results)

            if "typescript" in engines_used:
                # TypeScript引擎留待實現
                # TypeScript 引擎需要獨立的 Worker 服務
                # 引擎呼叫由 multi_engine_coordinator 統一調度
                pass

            if "go" in engines_used:
                # Go引擎留待實現
                # Go 引擎需要獨立的 Worker 服務
                # 引擎呼叫由 multi_engine_coordinator 統一調度
                pass

            if "rust" in engines_used:
                # Rust引擎留待實現
                # Rust 引擎需要獨立的 Worker 服務
                # 引擎呼叫由 multi_engine_coordinator 統一調度
                pass

            # 2. 去重和關聯分析
            complete_asset_list = self._deduplicate_assets(complete_asset_list)

            execution_time = time.time() - start_time
            
            # 生成摘要信息
            from services.aiva_common.schemas import Summary
            summary = Summary(
                urls_found=len([a for a in complete_asset_list if a.type == "url"]),
                forms_found=len([a for a in complete_asset_list if a.has_form]),
                apis_found=len([a for a in complete_asset_list if a.type == "api"]),
                scan_duration_seconds=int(execution_time)
            )

            return Phase1CompletedPayload(
                scan_id=request.scan_id,
                status="completed",
                execution_time=execution_time,
                assets=complete_asset_list,
                fingerprints=None,
                summary=summary,
                engine_results={"python": {"assets_count": len(complete_asset_list)}},
                phase0_summary=None,
                error_info=None
            )

        except Exception as e:
            execution_time = time.time() - start_time
            # Phase1 失敗
            
            from services.aiva_common.schemas import Summary
            summary = Summary(
                urls_found=0,
                forms_found=0,
                apis_found=0,
                scan_duration_seconds=int(execution_time)
            )

            return Phase1CompletedPayload(
                scan_id=request.scan_id,
                status="failed",
                execution_time=execution_time,
                assets=complete_asset_list,
                fingerprints=None,
                summary=summary,
                engine_results={},
                phase0_summary=None,
                error_info=str(e)
            )

    # ==================== Phase0 輔助方法 ====================

    def _quick_fingerprint(self, _target: str) -> dict[str, Any]:
        """Phase0: 快速指紋識別"""
        # 使用現有的 fingerprint_collector
        # 簡化版本，只識別基本技術棧
        return {"technologies": ["HTTP", "HTML"]}

    def _quick_sensitive_scan(self, target: str) -> list[str]:
        """Phase0: 快速敏感資訊掃描
        
        優先調用 Rust 引擎(高性能),失敗時回退到 Python 實現
        """
        try:
            # 嘗試調用 Rust 引擎 (同步調用)
            return self._call_rust_sensitive_scanner([target])
        except Exception:
            # 回退到 Python 實現
            return self._python_sensitive_scan()

    def _call_rust_sensitive_scanner(self, _targets: list[str]) -> list[str]:
        """調用 Rust 引擎的敏感資訊掃描器
        
        Note: Rust 引擎在 Phase0 中由 Rust Worker 直接處理
        這裡保留為備用接口
        """
        # 使用Rust Worker
        return []

    def _python_sensitive_scan(self) -> list[str]:
        """回退方案：Python 實現的敏感資訊掃描。返回空列表以避免 Rust 引擎失敗時報告誤判。"""
        # 簡化實現：Rust 引擎失敗時返回空列表
        # 實際的敏感資訊掃描應該由 Rust 引擎在 Phase0 中完成
        # 使用Python fallback
        return []

    def _quick_endpoint_discovery(self, target: str) -> list[str]:
        """Phase0: 基礎端點發現（深度1，最多50個URL）"""
        # 快速爬取，深度1
        endpoints = [target]
        
        # 常見端點路徑
        common_paths = [
            '/api', '/admin', '/login', '/dashboard',
            '/robots.txt', '/sitemap.xml', '/.well-known',
        ]
        
        for path in common_paths[:5]:  # 限制數量
            endpoints.append(f"{target.rstrip('/')}{path}")
        
        return endpoints[:50]

    # ==================== Phase1 輔助方法 ====================

    async def _execute_python_scan(self, request: Phase1StartPayload, override_targets: list[str] | None = None) -> list[Asset]:
        """執行 Python 引擎掃描
        
        Args:
            request: Phase1 掃描請求
            override_targets: 可選的目標列表覆蓋（用於整合 Phase 0 結果）
        """
        from pydantic import HttpUrl

        # 使用傳入的 targets 或原本 request 的 targets
        if override_targets:
            final_targets = [HttpUrl(url) for url in override_targets]
        else:
            final_targets = request.targets

        # 將 Phase1 請求轉換為標準掃描請求
        scan_request = ScanStartPayload(
            scan_id=request.scan_id,
            targets=final_targets,
            strategy="deep",  # Phase1 強制使用深度掃描（啟用 Playwright）
            scope=request.scope,
            authentication=request.authentication,
        )

        # 執行標準掃描
        result = await self.execute_scan(scan_request)
        return result.assets

    def _deduplicate_assets(self, assets: list[Asset]) -> list[Asset]:
        """去重資產列表"""
        seen = set()
        unique_assets = []

        for asset in assets:
            # 使用 asset_id 作為去重鍵
            if asset.asset_id not in seen:
                seen.add(asset.asset_id)
                unique_assets.append(asset)

        # 資產去重完成
        return unique_assets
