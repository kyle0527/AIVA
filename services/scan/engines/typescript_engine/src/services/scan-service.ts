/**
 * Scan Service - Playwright 掃描核心邏輯
 * 整合動態內容、SPA 路由、AJAX 攔截、WebSocket 檢測
 */

import { Browser, Page, BrowserContext } from 'playwright-core';
import { logger } from '../utils/logger.js';
import { NetworkInterceptor } from './network-interceptor.service.js';

interface ScanTask {
  scan_id: string;
  target_url: string;
  max_depth: number;
  max_pages: number;
  enable_javascript: boolean;
}

interface Asset {
  type: string;
  value: string;
  metadata: Record<string, any>;
}

interface ScanResult {
  scan_id: string;
  assets: Asset[];
  vulnerabilities: any[];
  metadata: {
    pages_scanned: number;
    duration_seconds: number;
    start_time: string;
    end_time: string;
    spa_detected: boolean;
    websockets_found: number;
    ajax_requests_found: number;
  };
}

export class ScanService {
  private readonly browser: Browser;
  private readonly networkInterceptor: NetworkInterceptor;

  constructor(browser: Browser) {
    this.browser = browser;
    this.networkInterceptor = new NetworkInterceptor();
  }

  /**
   * 正規化 URL：移除 hash 和尾隨斜線，確保一致性
   */
  private normalizeUrl(url: string): string {
    try {
      const parsed = new URL(url);
      parsed.hash = ''; // 移除 hash (#section)
      // 移除尾隨斜線（但保留根路徑的斜線）
      let normalized = parsed.href;
      if (normalized.endsWith('/') && parsed.pathname !== '/') {
        normalized = normalized.slice(0, -1);
      }
      return normalized;
    } catch {
      return url; // 無效 URL 直接返回原值
    }
  }

  async scan(task: ScanTask): Promise<ScanResult> {
    const startTime = new Date();
    const assets: Asset[] = [];
    const visited = new Set<string>();
    const queue: { url: string; depth: number }[] = [
      { url: task.target_url, depth: 0 },
    ];

    // 掃描超時設定：10 分鐘
    const MAX_SCAN_TIME_MS = 10 * 60 * 1000;
    const scanTimeout = Date.now() + MAX_SCAN_TIME_MS;

    let context: BrowserContext | null = null;
    let page: Page | null = null;
    let spaDetected = false;
    const webSocketEndpoints = new Set<string>();

    try {
      context = await this.browser.newContext({
        viewport: { width: 1920, height: 1080 },
        userAgent:
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 AIVA-Scanner/1.0',
      });

      page = await context.newPage();

      // 啟動網路攔截
      await this.networkInterceptor.startInterception(page);

      // 監聽 WebSocket 連接
      this.setupWebSocketMonitoring(page, webSocketEndpoints);

      // 監聽 SPA 路由變化 (設置監聽器)
      await this.setupSpaMonitoring(page);

      while (queue.length > 0 && 
             assets.length < task.max_pages &&
             Date.now() < scanTimeout) {
        const { url, depth } = queue.shift()!;
        
        // 正規化 URL 以避免重複爬取
        const normalizedUrl = this.normalizeUrl(url);

        if (visited.has(normalizedUrl) || depth > task.max_depth) {
          continue;
        }

        visited.add(normalizedUrl);

        try {
          logger.info({ scan_id: task.scan_id, url, depth }, '🕷️  掃描頁面');

          // 訪問頁面
          const response = await page.goto(url, {
            waitUntil: 'networkidle',
            timeout: 30000,
          });

          if (!response) {
            logger.warn({ url }, '⚠️  無法訪問頁面');
            continue;
          }

          // 等待動態內容載入
          await page.waitForTimeout(1000);

          // 提取頁面資訊
          const pageAssets = await this.extractAssets(page, url);
          assets.push(...pageAssets);

          // 檢查 SPA 框架
          const spaInfo = await this.detectSpaFramework(page);
          if (spaInfo.isSpa) {
            spaDetected = true;
            logger.info({ spa: spaInfo.framework, url }, '🎯 偵測到 SPA 應用');
            
            // 提取 SPA 路由
            const routes = await this.extractSpaRoutes(page, spaInfo.framework);
            for (const route of routes) {
              assets.push({
                type: 'spa_route',
                value: route,
                metadata: {
                  framework: spaInfo.framework,
                  base_url: url,
                },
              });
            }
          }

          // 提取連結
          if (depth < task.max_depth) {
            const links = await this.extractLinks(page, url);
            for (const link of links) {
              const normalizedLink = this.normalizeUrl(link);
              if (!visited.has(normalizedLink)) {
                queue.push({ url: normalizedLink, depth: depth + 1 });
              }
            }
          }

          // 等待一下避免過快
          await page.waitForTimeout(500);
        } catch (error: any) {
          logger.error({ url, error: error.message }, '❌ 掃描頁面失敗');
          continue; // 跳過當前頁面，繼續處理下一個
        }
      }

      // 檢查是否超時
      if (Date.now() >= scanTimeout) {
        logger.warn({ 
          scan_id: task.scan_id, 
          duration_ms: Date.now() - startTime.getTime(),
          pages_scanned: visited.size 
        }, '⏱️ 掃描達到最大時間限制');
      }

      // 停止攔截並提取網路請求資產
      const networkAssets = this.extractNetworkAssets();
      assets.push(...networkAssets);

      // 提取 WebSocket 資產
      for (const wsUrl of webSocketEndpoints) {
        assets.push({
          type: 'websocket',
          value: wsUrl,
          metadata: { discovered_at: new Date().toISOString() },
        });
      }

      const endTime = new Date();
      const ajaxRequests = this.networkInterceptor.getAjaxRequests();

      return {
        scan_id: task.scan_id,
        assets,
        vulnerabilities: [],
        metadata: {
          pages_scanned: visited.size,
          duration_seconds: (endTime.getTime() - startTime.getTime()) / 1000,
          start_time: startTime.toISOString(),
          end_time: endTime.toISOString(),
          spa_detected: spaDetected,
          websockets_found: webSocketEndpoints.size,
          ajax_requests_found: ajaxRequests.length,
        },
      };
    } finally {
      if (page) await page.close();
      if (context) await context.close();
    }
  }

  private async extractAssets(page: Page, url: string): Promise<Asset[]> {
    const assets: Asset[] = [];

    // 提取表單
    const forms = await page.locator('form').all();
    for (let i = 0; i < forms.length; i++) {
      const form = forms[i];
      const action = await form.getAttribute('action');
      const method = await form.getAttribute('method');

      assets.push({
        type: 'form',
        value: `form_${i}`,
        metadata: {
          url,
          action: action || '',
          method: (method || 'GET').toUpperCase(),
        },
      });
    }

    // 提取輸入框
    const inputs = await page
      .locator('input[type="text"], input[type="password"], textarea')
      .all();
    for (const input of inputs) {
      const name = await input.getAttribute('name');
      const type = await input.getAttribute('type');

      if (name) {
        assets.push({
          type: 'input',
          value: name,
          metadata: {
            url,
            input_type: type || 'text',
          },
        });
      }
    }

    return assets;
  }

  private async extractLinks(page: Page, baseUrl: string): Promise<string[]> {
    try {
      const links = await page.locator('a[href]').all();
      const urls: string[] = [];

      for (const link of links) {
        const href = await link.getAttribute('href');
        if (href) {
          try {
            const absoluteUrl = new URL(href, baseUrl);
            // 只保留同域名的連結
            if (absoluteUrl.origin === new URL(baseUrl).origin) {
              urls.push(absoluteUrl.href);
            }
          } catch {
            // 忽略無效的 URL
          }
        }
      }

      return [...new Set(urls)]; // 去重
    } catch (error: any) {
      logger.warn({ error: error.message, baseUrl }, '❌ Failed to extract links');
      return []; // 返回空數組而非崩潰
    }
  }

  /**
   * 設定 WebSocket 監聽
   */
  private setupWebSocketMonitoring(
    page: Page,
    webSocketEndpoints: Set<string>
  ): void {
    page.on('websocket', (ws: any) => {
      const wsUrl = ws.url();
      webSocketEndpoints.add(wsUrl);
      logger.info({ url: wsUrl }, '🔌 WebSocket 連接');

      ws.on('framereceived', (event: any) => {
        logger.debug({ payload: event.payload }, '📥 WebSocket 接收');
      });

      ws.on('framesent', (event: any) => {
        logger.debug({ payload: event.payload }, '📤 WebSocket 發送');
      });

      ws.on('close', () => {
        logger.debug({ url: wsUrl }, '🔌 WebSocket 關閉');
      });
    });
  }

  /**
   * 設定 SPA 監聽 (History API)
   */
  private async setupSpaMonitoring(page: Page): Promise<string[]> {
    const routes: string[] = [];

    // 監聽 History API 變化
    await page.exposeFunction('__aivaHistoryChange', (url: string) => {
      routes.push(url);
      logger.info({ route: url }, '🛤️  SPA 路由變化');
    });

    // 注入監聽腳本
    await page.addInitScript(() => {
      const originalPushState = history.pushState;
      const originalReplaceState = history.replaceState;

      history.pushState = function (...args) {
        originalPushState.apply(this, args);
        (globalThis as any).__aivaHistoryChange(globalThis.location.href);
      };

      history.replaceState = function (...args) {
        originalReplaceState.apply(this, args);
        (globalThis as any).__aivaHistoryChange(globalThis.location.href);
      };

      globalThis.addEventListener('popstate', () => {
        (globalThis as any).__aivaHistoryChange(globalThis.location.href);
      });
    });

    return routes;
  }

  /**
   * 檢測 SPA 框架
   */
  private async detectSpaFramework(page: Page): Promise<{
    isSpa: boolean;
    framework: string | null;
  }> {
    const result = await page.evaluate(() => {
      // 檢測 React
      if (
        (globalThis as any).__REACT_DEVTOOLS_GLOBAL_HOOK__ ||
        document.querySelector('[data-reactroot], [data-reactid]')
      ) {
        return { isSpa: true, framework: 'React' };
      }

      // 檢測 Vue
      if (
        (globalThis as any).__VUE__ ||
        (globalThis as any).__VUE_DEVTOOLS_GLOBAL_HOOK__ ||
        document.querySelector('[data-v-]')
      ) {
        return { isSpa: true, framework: 'Vue' };
      }

      // 檢測 Angular
      if (
        (globalThis as any).ng ||
        (globalThis as any).getAllAngularRootElements ||
        document.querySelector('[ng-app], [ng-version]')
      ) {
        return { isSpa: true, framework: 'Angular' };
      }

      // 檢測 Svelte
      if (document.querySelector('[class*="svelte-"]')) {
        return { isSpa: true, framework: 'Svelte' };
      }

      // 檢測通用 SPA 特徵
      const hasHistoryApi = !!(globalThis.history?.pushState);
      const hasSingleRootDiv =
        document.querySelectorAll('body > div').length === 1;

      if (hasHistoryApi && hasSingleRootDiv) {
        return { isSpa: true, framework: 'Unknown' };
      }

      return { isSpa: false, framework: null };
    });

    return result;
  }

  /**
   * 提取 SPA 路由
   */
  private async extractSpaRoutes(
    page: Page,
    framework: string | null
  ): Promise<string[]> {
    const routes = await page.evaluate(() => {
      const links = Array.from(document.querySelectorAll('a[href]'));
      return links
        .map((a) => (a as HTMLAnchorElement).href)
        .filter((href) => {
          try {
            const url = new URL(href);
            return url.origin === globalThis.location.origin && url.hash !== '';
          } catch {
            return false;
          }
        });
    });

    logger.info(
      { framework, count: routes.length },
      '🗺️  提取 SPA 路由'
    );

    return [...new Set(routes)];
  }

  /**
   * 從網路攔截器提取資產
   */
  private extractNetworkAssets(): Asset[] {
    const assets: Asset[] = [];

    // 提取 API 端點
    const apiRequests = this.networkInterceptor.getApiRequests();
    for (const req of apiRequests) {
      assets.push({
        type: 'api',
        value: req.url,
        metadata: {
          method: req.method,
          headers: req.headers,
          post_data: req.post_data,
          response_status: req.response_status,
        },
      });
    }

    // 提取 AJAX 請求
    const ajaxRequests = this.networkInterceptor.getAjaxRequests();
    for (const req of ajaxRequests) {
      assets.push({
        type: 'ajax',
        value: req.url,
        metadata: {
          method: req.method,
          headers: req.headers,
        },
      });
    }

    // 分析請求模式
    const patterns = this.networkInterceptor.analyzeRequestPatterns();
    logger.info(
      {
        domains: patterns.unique_domains.length,
        apis: patterns.potential_apis.length,
        methods: patterns.request_methods,
      },
      '📊 網路請求分析'
    );

    return assets;
  }
}
