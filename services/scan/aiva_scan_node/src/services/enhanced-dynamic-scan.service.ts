/**
 * Enhanced Dynamic Scan Service
 * 增強版動態掃描服務 - 整合所有功能模組
 */

import { Browser, Page, BrowserContext } from 'playwright-core';
import { logger } from '../utils/logger';
import { NetworkInterceptor } from './network-interceptor.service';
import { InteractionSimulator } from './interaction-simulator.service';
import { EnhancedContentExtractor } from './enhanced-content-extractor.service';
import { 
  DynamicScanTask, 
  DynamicScanResult, 
  DynamicContent,
  InteractionResult
} from '../interfaces/dynamic-scan.interfaces';

export class EnhancedDynamicScanService {
  private browser: Browser;

  constructor(browser: Browser) {
    this.browser = browser;
  }

  /**
   * 執行增強版動態掃描
   */
  async executeDynamicScan(task: DynamicScanTask): Promise<DynamicScanResult> {
    const startTime = Date.now();
    let context: BrowserContext | null = null;
    let page: Page | null = null;

    logger.info({
      task_id: task.task_id,
      scan_id: task.scan_id,
      url: task.url
    }, '🚀 Starting enhanced dynamic scan');

    try {
      // 創建瀏覽器上下文
      context = await this.browser.newContext({
        viewport: { width: 1920, height: 1080 },
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 AIVA-Enhanced-Scanner/2.0',
        ignoreHTTPSErrors: true,
        // 啟用 JavaScript
        javaScriptEnabled: true
      });

      page = await context.newPage();

      // 初始化服務
      const networkInterceptor = new NetworkInterceptor();
      const contentExtractor = new EnhancedContentExtractor(page, task.extraction_config);
      const interactionSimulator = new InteractionSimulator(page, task.interaction_config);

      // 開始網路攔截
      await networkInterceptor.startInterception(page);

      // 導航到目標頁面
      logger.debug({ url: task.url }, '🌐 Navigating to target URL');
      
      const response = await page.goto(task.url, {
        waitUntil: 'networkidle',
        timeout: task.timeout_ms || 30000
      });

      if (!response) {
        throw new Error('Failed to load page');
      }

      logger.info({
        status: response.status(),
        url: task.url
      }, '✅ Page loaded successfully');

      // 等待額外的網路活動結束
      if (task.extraction_config.wait_for_network_idle) {
        await page.waitForTimeout(task.extraction_config.network_idle_timeout_ms);
      }

      // 階段 1: 初始內容提取
      logger.debug('📋 Phase 1: Initial content extraction');
      const initialContents = await contentExtractor.extractAll(task.url);

      // 階段 2: 用戶互動模擬
      logger.debug('🎭 Phase 2: User interaction simulation');
      const interactions: InteractionResult[] = [];
      
      if (this.shouldSimulateInteractions(task)) {
        const interactionResults = await interactionSimulator.executeAll();
        interactions.push(...interactionResults);
        
        // 互動後再次提取內容
        logger.debug('📋 Phase 2.5: Post-interaction content extraction');
        const postInteractionContents = await contentExtractor.extractAll(task.url);
        initialContents.push(...postInteractionContents);
      }

      // 階段 3: 收集網路請求
      logger.debug('🌐 Phase 3: Network request collection');
      const networkRequests = networkInterceptor.stopInterception();

      // 階段 4: JavaScript 錯誤收集
      const jsErrors = await this.collectJavaScriptErrors(page);

      // 構建結果
      const result: DynamicScanResult = {
        task_id: task.task_id,
        scan_id: task.scan_id,
        url: task.url,
        status: 'completed',
        contents: this.deduplicateContents(initialContents),
        interactions: interactions,
        network_requests: networkRequests,
        dom_changes: this.extractDOMChangesFromInteractions(interactions),
        metadata: {
          total_interactions: interactions.length,
          total_network_requests: networkRequests.length,
          total_dom_changes: this.countDOMChanges(interactions),
          scan_duration_ms: Date.now() - startTime,
          javascript_errors: jsErrors
        }
      };

      logger.info({
        task_id: task.task_id,
        contents_found: result.contents.length,
        interactions_performed: result.interactions.length,
        network_requests_captured: result.network_requests.length,
        duration_ms: result.metadata.scan_duration_ms
      }, '🎉 Enhanced dynamic scan completed successfully');

      return result;

    } catch (error: any) {
      logger.error({
        task_id: task.task_id,
        error: error.message,
        stack: error.stack
      }, '❌ Enhanced dynamic scan failed');

      return {
        task_id: task.task_id,
        scan_id: task.scan_id,
        url: task.url,
        status: 'failed',
        contents: [],
        interactions: [],
        network_requests: [],
        dom_changes: [],
        metadata: {
          total_interactions: 0,
          total_network_requests: 0,
          total_dom_changes: 0,
          scan_duration_ms: Date.now() - startTime,
          javascript_errors: []
        },
        error_message: error.message
      };

    } finally {
      // 清理資源
      try {
        if (page) await page.close();
        if (context) await context.close();
      } catch (cleanupError: any) {
        logger.warn({
          error: cleanupError.message
        }, 'Failed to cleanup browser resources');
      }
    }
  }

  /**
   * 判斷是否應該進行互動模擬
   */
  private shouldSimulateInteractions(task: DynamicScanTask): boolean {
    const config = task.interaction_config;
    return config.click_buttons || 
           config.fill_forms || 
           config.scroll_pages || 
           config.hover_elements || 
           config.trigger_events;
  }

  /**
   * 收集 JavaScript 錯誤
   */
  private async collectJavaScriptErrors(page: Page): Promise<string[]> {
    const errors: string[] = [];
    
    try {
      // 獲取控制台錯誤
      const consoleErrors = await page.evaluate(() => {
        return (window as any).jsErrors || [];
      });
      
      errors.push(...consoleErrors);
    } catch (error: any) {
      logger.debug('Could not collect JS errors');
    }

    return errors;
  }

  /**
   * 從互動結果中提取 DOM 變更
   */
  private extractDOMChangesFromInteractions(interactions: InteractionResult[]) {
    const allChanges = [];
    
    for (const interaction of interactions) {
      allChanges.push(...interaction.dom_changes);
    }
    
    return allChanges;
  }

  /**
   * 計算 DOM 變更總數
   */
  private countDOMChanges(interactions: InteractionResult[]): number {
    return interactions.reduce((total, interaction) => {
      return total + interaction.dom_changes.length;
    }, 0);
  }

  /**
   * 去除重複的內容
   */
  private deduplicateContents(contents: DynamicContent[]): DynamicContent[] {
    const seen = new Set<string>();
    const deduplicated: DynamicContent[] = [];

    for (const content of contents) {
      const key = `${content.content_type}:${content.url}:${content.text_content}`;
      
      if (!seen.has(key)) {
        seen.add(key);
        deduplicated.push(content);
      }
    }

    return deduplicated;
  }

  /**
   * 獲取掃描器統計資訊
   */
  async getServiceStats() {
    return {
      browser_version: await this.browser.version(),
      contexts_count: this.browser.contexts().length,
      service_status: 'active'
    };
  }

  /**
   * 健康檢查
   */
  async healthCheck(): Promise<boolean> {
    try {
      const context = await this.browser.newContext();
      const page = await context.newPage();
      await page.goto('data:text/html,<h1>Health Check</h1>');
      await page.close();
      await context.close();
      return true;
    } catch (error) {
      logger.error('Health check failed', error);
      return false;
    }
  }
}