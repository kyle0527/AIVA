# Dynamic Engine 遷移計劃

## 目標

將 Python `dynamic_engine` 的功能完全遷移到 `aiva_scan_node`，實現統一的動態掃描架構。

## 階段 1: 功能對照與遷移準備

### 現有 Python 功能對照

- `HeadlessBrowserPool` -> `aiva_scan_node/browser-pool.ts`
- `DynamicContentExtractor` -> `aiva_scan_node/content-extractor.ts`
- `JsInteractionSimulator` -> `aiva_scan_node/interaction-simulator.ts`

### RabbitMQ 隊列設計

```typescript
interface DynamicScanTask {
  task_id: string;
  scan_id: string;
  url: string;
  extraction_config: {
    extract_forms: boolean;
    extract_links: boolean;
    extract_ajax: boolean;
    extract_api_calls: boolean;
    wait_for_network_idle: boolean;
  };
  interaction_config: {
    click_buttons: boolean;
    fill_forms: boolean;
    scroll_pages: boolean;
    wait_time_ms: number;
  };
}
```

## 階段 2: 增強 aiva_scan_node 功能

### 網路請求攔截

```typescript
// 在現有 scan-service.ts 中添加
async setupNetworkInterception(page: Page): Promise<NetworkRequest[]> {
  const requests: NetworkRequest[] = [];

  page.on('request', request => {
    requests.push({
      url: request.url(),
      method: request.method(),
      headers: request.headers(),
      postData: request.postData()
    });
  });

  return requests;
}
```

### 用戶互動模擬

```typescript
async simulateUserInteractions(page: Page): Promise<InteractionResult[]> {
  const results: InteractionResult[] = [];

  // 點擊所有按鈕
  const buttons = await page.$$('button, input[type="submit"]');
  for (const button of buttons) {
    try {
      await button.click();
      await page.waitForTimeout(1000);
      results.push({ type: 'click', success: true });
    } catch (error) {
      results.push({ type: 'click', success: false, error: error.message });
    }
  }

  return results;
}
```

### DOM 變動監聽

```typescript
async setupDOMObserver(page: Page): Promise<DOMChange[]> {
  const changes: DOMChange[] = [];

  await page.evaluateOnNewDocument(() => {
    const observer = new MutationObserver(mutations => {
      mutations.forEach(mutation => {
        window.domChanges = window.domChanges || [];
        window.domChanges.push({
          type: mutation.type,
          target: mutation.target.nodeName,
          timestamp: Date.now()
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true
    });
  });

  return changes;
}
```

## 階段 3: 修改 scan_orchestrator.py

### 移除本地 dynamic_engine 調用

```python
# 移除這些導入
from .dynamic_engine.dynamic_content_extractor import DynamicContentExtractor
from .dynamic_engine.headless_browser_pool import HeadlessBrowserPool

# 修改 _process_url_dynamic 方法
async def _process_url_dynamic(self, url: str, context: ScanContext,
                               url_queue: UrlQueueManager,
                               strategy_params: Any) -> None:
    """改為發送任務到 RabbitMQ"""

    task = {
        "task_id": new_id("dynamic_task"),
        "scan_id": context.scan_id,
        "url": url,
        "extraction_config": {
            "extract_forms": True,
            "extract_links": True,
            "extract_ajax": True,
            "extract_api_calls": True,
            "wait_for_network_idle": True
        }
    }

    # 發送到專用的動態掃描隊列
    await self.mq_publisher.publish("task.scan.dynamic", task)
    logger.info(f"Dynamic scan task sent to queue: {url}")
```

## 階段 4: 棄用標記

在所有 `dynamic_engine` 文件頂部添加棄用警告：

```python
# DEPRECATED: This module will be removed in v2.0.0
# Use aiva_scan_node for dynamic scanning instead
import warnings
warnings.warn("dynamic_engine is deprecated, use aiva_scan_node",
              DeprecationWarning, stacklevel=2)
```

## 時程估算

- 階段 1-2: 2 週
- 階段 3: 1 週
- 階段 4: 1 週
- 測試與優化: 1 週

**總計**: 5 週完成遷移
