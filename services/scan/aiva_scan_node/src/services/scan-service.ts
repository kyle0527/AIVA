/**
 * Scan Service - Playwright 掃描核心邏輯
 */

import { Browser, Page, BrowserContext } from 'playwright-core';
import { logger } from '../utils/logger';

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
  };
}

export class ScanService {
  private browser: Browser;

  constructor(browser: Browser) {
    this.browser = browser;
  }

  async scan(task: ScanTask): Promise<ScanResult> {
    const startTime = new Date();
    const assets: Asset[] = [];
    const visited = new Set<string>();
    const queue: { url: string; depth: number }[] = [
      { url: task.target_url, depth: 0 },
    ];

    let context: BrowserContext | null = null;
    let page: Page | null = null;

    try {
      context = await this.browser.newContext({
        viewport: { width: 1920, height: 1080 },
        userAgent:
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 AIVA-Scanner/1.0',
      });

      page = await context.newPage();

      while (queue.length > 0 && assets.length < task.max_pages) {
        const { url, depth } = queue.shift()!;

        if (visited.has(url) || depth > task.max_depth) {
          continue;
        }

        visited.add(url);

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

          // 提取頁面資訊
          const pageAssets = await this.extractAssets(page, url);
          assets.push(...pageAssets);

          // 提取連結
          if (depth < task.max_depth) {
            const links = await this.extractLinks(page, url);
            for (const link of links) {
              if (!visited.has(link)) {
                queue.push({ url: link, depth: depth + 1 });
              }
            }
          }

          // 等待一下避免過快
          await page.waitForTimeout(500);
        } catch (error: any) {
          logger.error({ url, error: error.message }, '❌ 掃描頁面失敗');
        }
      }

      const endTime = new Date();

      return {
        scan_id: task.scan_id,
        assets,
        vulnerabilities: [],
        metadata: {
          pages_scanned: visited.size,
          duration_seconds: (endTime.getTime() - startTime.getTime()) / 1000,
          start_time: startTime.toISOString(),
          end_time: endTime.toISOString(),
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
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
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

    // 提取 API 端點 (從 XHR 請求)
    const apiCalls: string[] = [];
    page.on('request', (request) => {
      const reqUrl = request.url();
      if (
        request.resourceType() === 'xhr' ||
        request.resourceType() === 'fetch'
      ) {
        apiCalls.push(reqUrl);
      }
    });

    for (const api of apiCalls) {
      assets.push({
        type: 'api',
        value: api,
        metadata: { url },
      });
    }

    return assets;
  }

  private async extractLinks(page: Page, baseUrl: string): Promise<string[]> {
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
  }
}
