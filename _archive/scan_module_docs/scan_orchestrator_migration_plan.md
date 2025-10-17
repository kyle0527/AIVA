# scan_orchestrator.py 遷移方案

## 目標
移除本地 dynamic_engine 調用，改為通過 RabbitMQ 與 aiva_scan_node 通信

## 修改步驟

### 1. 移除 dynamic_engine 導入

```python
# 移除這些導入
# from .dynamic_engine.dynamic_content_extractor import DynamicContentExtractor
# from .dynamic_engine.headless_browser_pool import HeadlessBrowserPool

# 添加 RabbitMQ 發布者
from services.aiva_common.mq import MQPublisher
```

### 2. 修改 ScanOrchestrator 初始化

```python
class ScanOrchestrator:
    def __init__(self):
        """初始化掃描編排器"""
        self.static_parser = StaticContentParser()
        self.fingerprint_collector = FingerprintCollector()
        self.sensitive_detector = SensitiveInfoDetector()
        self.js_analyzer = JavaScriptSourceAnalyzer()

        # 移除動態引擎組件
        # self.browser_pool: HeadlessBrowserPool | None = None
        # self.dynamic_extractor: DynamicContentExtractor | None = None
        
        # 添加 MQ 發布者
        self.mq_publisher = MQPublisher()
        
        # 動態掃描結果暫存
        self.dynamic_scan_results: Dict[str, Any] = {}

        logger.info("ScanOrchestrator initialized with RabbitMQ integration")
```

### 3. 移除 _init_dynamic_engine 方法

```python
# 完全移除此方法
# async def _init_dynamic_engine(self, strategy_params: StrategyParameters) -> None:
```

### 4. 重寫 _process_url_dynamic 方法

```python
async def _process_url_dynamic(
    self,
    url: str,
    context: ScanContext,
    url_queue: UrlQueueManager,
    strategy_params: Any,
) -> None:
    """
    發送動態掃描任務到 aiva_scan_node

    Args:
        url: 要處理的 URL
        context: 掃描上下文
        url_queue: URL 隊列
        strategy_params: 策略參數
    """
    logger.info(f"Dispatching dynamic scan task for: {url}")
    
    try:
        # 構建動態掃描任務
        dynamic_task = {
            "task_id": new_id("dynamic_task"),
            "scan_id": context.scan_id,
            "url": url,
            "extraction_config": {
                "extract_forms": True,
                "extract_links": True,
                "extract_ajax": True,
                "extract_api_calls": True,
                "extract_websockets": True,
                "extract_js_variables": strategy_params.enable_js_analysis,
                "extract_event_listeners": True,
                "wait_for_network_idle": True,
                "network_idle_timeout_ms": int(strategy_params.page_load_timeout * 1000)
            },
            "interaction_config": {
                "click_buttons": strategy_params.enable_interaction,
                "fill_forms": strategy_params.enable_interaction,
                "scroll_pages": True,
                "hover_elements": False,
                "trigger_events": strategy_params.enable_interaction,
                "wait_time_ms": 1000,
                "max_interactions": 10
            },
            "timeout_ms": int(strategy_params.page_load_timeout * 1000)
        }
        
        # 發送到 aiva_scan_node 隊列
        await self.mq_publisher.publish(
            queue_name="task.scan.dynamic.enhanced",
            message=dynamic_task,
            routing_key="dynamic.scan"
        )
        
        logger.info(f"Dynamic scan task dispatched: {dynamic_task['task_id']}")
        
        # 記錄發送的任務（用於後續結果匹配）
        context.add_pending_dynamic_task(dynamic_task['task_id'], url)
        
    except Exception as e:
        logger.error(f"Failed to dispatch dynamic scan task for {url}: {e}")
        context.add_error("dynamic_dispatch_error", str(e), url)
```

### 5. 修改 execute_scan 方法

```python
async def execute_scan(self, request: ScanStartPayload) -> ScanCompletedPayload:
    """執行掃描任務"""
    logger.info(f"Starting scan: {request.scan_id}")
    
    # 創建上下文
    context = ScanContext(
        scan_id=request.scan_id,
        targets=request.targets,
        strategy=request.strategy
    )
    
    # 加載策略參數
    strategy_params = StrategyController(request.strategy).get_parameters()
    
    # 創建 HTTP 客戶端
    auth_manager = AuthenticationManager(request.authentication)
    header_config = HeaderConfiguration(request.custom_headers)
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

    # 移除瀏覽器池初始化
    # if strategy_params.enable_dynamic_scan:
    #     await self._init_dynamic_engine(strategy_params)
    
    # 設置動態掃描結果監聽器（如果啟用）
    if strategy_params.enable_dynamic_scan:
        await self._setup_dynamic_scan_listener(context.scan_id)

    try:
        # 執行爬蟲掃描
        await self._perform_crawling(
            context,
            url_queue,
            http_client,
            strategy_params,
        )
        
        # 等待動態掃描結果（如果有的話）
        if strategy_params.enable_dynamic_scan:
            await self._wait_for_dynamic_results(context, strategy_params.dynamic_timeout)
        
        # 構建最終結果
        return self._build_scan_result(context)

    except Exception as e:
        logger.error(f"Scan execution failed: {e}", exc_info=True)
        context.add_error("scan_execution", str(e))
        return self._build_scan_result(context)

    finally:
        # 清理資源
        await http_client.close()
        
        # 移除瀏覽器池清理
        # if self.browser_pool:
        #     await self.browser_pool.cleanup()
        #     self.browser_pool = None
        
        # 清理動態掃描監聽器
        if strategy_params.enable_dynamic_scan:
            await self._cleanup_dynamic_scan_listener(context.scan_id)
```

### 6. 新增動態掃描結果處理方法

```python
async def _setup_dynamic_scan_listener(self, scan_id: str) -> None:
    """設置動態掃描結果監聽器"""
    try:
        # 監聽動態掃描結果隊列
        result_queue = f"results.scan.dynamic.{scan_id}"
        
        async def handle_dynamic_result(message):
            """處理動態掃描結果"""
            try:
                result = json.loads(message.body)
                task_id = result.get('task_id')
                
                logger.info(f"Received dynamic scan result: {task_id}")
                self.dynamic_scan_results[task_id] = result
                
            except Exception as e:
                logger.error(f"Failed to process dynamic scan result: {e}")
        
        await self.mq_publisher.setup_consumer(
            queue_name=result_queue,
            callback=handle_dynamic_result
        )
        
        logger.info(f"Dynamic scan listener setup for scan: {scan_id}")
        
    except Exception as e:
        logger.error(f"Failed to setup dynamic scan listener: {e}")

async def _wait_for_dynamic_results(self, context: ScanContext, timeout: int) -> None:
    """等待動態掃描結果"""
    pending_tasks = context.get_pending_dynamic_tasks()
    if not pending_tasks:
        return
    
    logger.info(f"Waiting for {len(pending_tasks)} dynamic scan results...")
    
    start_time = time.time()
    completed_tasks = set()
    
    while len(completed_tasks) < len(pending_tasks) and (time.time() - start_time) < timeout:
        for task_id in pending_tasks:
            if task_id in self.dynamic_scan_results and task_id not in completed_tasks:
                # 處理動態掃描結果
                result = self.dynamic_scan_results[task_id]
                await self._process_dynamic_scan_result(context, result)
                completed_tasks.add(task_id)
        
        await asyncio.sleep(0.5)  # 短暫等待
    
    logger.info(f"Dynamic scan completed: {len(completed_tasks)}/{len(pending_tasks)} tasks")

async def _process_dynamic_scan_result(self, context: ScanContext, result: dict) -> None:
    """處理單個動態掃描結果"""
    try:
        # 提取動態內容
        contents = result.get('contents', [])
        for content in contents:
            # 根據內容類型創建對應的資產
            if content['content_type'] == 'form':
                asset = Asset(
                    asset_id=new_id("asset"),
                    type="FORM",
                    value=content['url'],
                    has_form=True,
                    parameters=content.get('attributes', {}).get('parameters', [])
                )
                context.add_asset(asset)
            
            elif content['content_type'] == 'link':
                asset = Asset(
                    asset_id=new_id("asset"),
                    type="URL",
                    value=content['url'],
                    has_form=False
                )
                context.add_asset(asset)
            
            elif content['content_type'] == 'api_endpoint':
                asset = Asset(
                    asset_id=new_id("asset"),
                    type="API",
                    value=content['url'],
                    has_form=False,
                    metadata={'extraction_method': content.get('extraction_method')}
                )
                context.add_asset(asset)
        
        # 更新統計
        context.increment_urls_found(len(contents))
        
        # 記錄網路請求
        network_requests = result.get('network_requests', [])
        for request in network_requests[:10]:  # 限制數量
            context.add_network_activity(request['url'], request['method'])
        
        logger.info(f"Processed dynamic scan result: {len(contents)} contents, {len(network_requests)} network requests")
        
    except Exception as e:
        logger.error(f"Failed to process dynamic scan result: {e}")

async def _cleanup_dynamic_scan_listener(self, scan_id: str) -> None:
    """清理動態掃描監聽器"""
    try:
        result_queue = f"results.scan.dynamic.{scan_id}"
        await self.mq_publisher.cleanup_consumer(result_queue)
        
        # 清理結果緩存
        scan_results = [k for k in self.dynamic_scan_results.keys() if scan_id in k]
        for key in scan_results:
            del self.dynamic_scan_results[key]
            
        logger.info(f"Dynamic scan listener cleaned up for scan: {scan_id}")
        
    except Exception as e:
        logger.error(f"Failed to cleanup dynamic scan listener: {e}")
```

### 7. 修改 ScanContext 類

```python
# 在 ScanContext 類中添加動態掃描支援
class ScanContext:
    def __init__(self, scan_id: str, targets: list, strategy: str):
        # ... 現有初始化代碼 ...
        self._pending_dynamic_tasks: Dict[str, str] = {}  # task_id -> url
        self._network_activity: List[Dict[str, str]] = []
    
    def add_pending_dynamic_task(self, task_id: str, url: str) -> None:
        """添加待處理的動態掃描任務"""
        self._pending_dynamic_tasks[task_id] = url
    
    def get_pending_dynamic_tasks(self) -> List[str]:
        """獲取待處理的動態掃描任務ID"""
        return list(self._pending_dynamic_tasks.keys())
    
    def add_network_activity(self, url: str, method: str) -> None:
        """添加網路活動記錄"""
        self._network_activity.append({
            'url': url,
            'method': method,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_network_activity(self) -> List[Dict[str, str]]:
        """獲取網路活動記錄"""
        return self._network_activity.copy()
```

## 配置更新

### strategy_controller.py 添加新參數

```python
@dataclass 
class StrategyParameters:
    # ... 現有參數 ...
    
    # 動態掃描相關
    enable_js_analysis: bool = True
    enable_interaction: bool = True 
    dynamic_timeout: int = 300  # 動態掃描超時（秒）
    
    # 新增參數
    max_dynamic_tasks_per_scan: int = 50
    dynamic_scan_priority: str = "normal"  # normal, high, low
```

## 部署配置

### RabbitMQ 隊列配置

```yaml
# rabbitmq_setup.yml
queues:
  - name: "task.scan.dynamic.enhanced"
    durable: true
    arguments:
      x-message-ttl: 300000  # 5分鐘超時
      x-max-length: 1000     # 最大隊列長度
  
  - name: "results.scan.dynamic.*"
    pattern: true
    durable: true
    auto_delete: true
    arguments:
      x-expires: 600000  # 10分鐘後自動刪除
```

## 測試策略

### 單元測試

```python
# test_scan_orchestrator_integration.py
import pytest
from unittest.mock import AsyncMock, patch

class TestScanOrchestratorMQIntegration:
    
    @pytest.mark.asyncio
    async def test_dynamic_scan_dispatch(self):
        """測試動態掃描任務分發"""
        orchestrator = ScanOrchestrator()
        orchestrator.mq_publisher = AsyncMock()
        
        context = ScanContext("scan123", ["http://example.com"], "DEEP")
        url_queue = AsyncMock()
        strategy_params = AsyncMock()
        strategy_params.enable_js_analysis = True
        strategy_params.enable_interaction = True
        
        await orchestrator._process_url_dynamic(
            "http://example.com", 
            context, 
            url_queue, 
            strategy_params
        )
        
        # 驗證 MQ 消息發送
        orchestrator.mq_publisher.publish.assert_called_once()
        call_args = orchestrator.mq_publisher.publish.call_args
        assert call_args[1]['queue_name'] == "task.scan.dynamic.enhanced"
        
    @pytest.mark.asyncio
    async def test_dynamic_result_processing(self):
        """測試動態掃描結果處理"""
        orchestrator = ScanOrchestrator()
        context = ScanContext("scan123", ["http://example.com"], "DEEP")
        
        # 模擬動態掃描結果
        mock_result = {
            "task_id": "task123",
            "contents": [
                {
                    "content_type": "form",
                    "url": "http://example.com/login",
                    "attributes": {"parameters": ["username", "password"]}
                }
            ],
            "network_requests": [
                {"url": "http://api.example.com/data", "method": "GET"}
            ]
        }
        
        await orchestrator._process_dynamic_scan_result(context, mock_result)
        
        # 驗證資產創建
        assets = context.get_assets()
        assert len(assets) == 1
        assert assets[0].type == "FORM"
        assert assets[0].has_form == True
```

## 監控與日誌

### 新增監控指標

```python
# 在 scan_orchestrator.py 中添加監控
class ScanOrchestrator:
    def __init__(self):
        # ... 現有代碼 ...
        self.metrics = {
            'dynamic_tasks_sent': 0,
            'dynamic_results_received': 0,
            'dynamic_scan_timeouts': 0,
            'mq_publish_errors': 0
        }
    
    async def _process_url_dynamic(self, ...):
        try:
            # ... 發送邏輯 ...
            self.metrics['dynamic_tasks_sent'] += 1
            logger.info("Dynamic task sent", extra={
                'metric': 'dynamic_tasks_sent',
                'scan_id': context.scan_id,
                'url': url
            })
        except Exception as e:
            self.metrics['mq_publish_errors'] += 1
            logger.error("MQ publish failed", extra={
                'metric': 'mq_publish_errors', 
                'error': str(e)
            })
```

這個遷移方案確保了：
1. **零停機遷移**: 可以漸進式部署
2. **向後相容**: 保留靜態掃描功能
3. **錯誤處理**: 完善的異常處理和超時機制
4. **監控支援**: 詳細的指標和日誌記錄
5. **測試覆蓋**: 完整的單元測試和整合測試