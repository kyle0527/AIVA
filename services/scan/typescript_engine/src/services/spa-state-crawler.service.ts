// SPA State Crawler
// OWASP A01:2021 - Broken Access Control
//
// 遞歸爬取 SPA 的所有路由狀態和 API 調用

import { Page, Browser, chromium, Route } from 'playwright';
import { injectable } from 'inversify';

export interface SPAState {
  url: string;
  title: string;
  forms: FormInfo[];
  apiCalls: APICallInfo[];
  localStorage: Record<string, string>;
  cookies: CookieInfo[];
  depth: number;
}

export interface FormInfo {
  action: string;
  method: string;
  inputs: InputInfo[];
}

export interface InputInfo {
  name: string;
  type: string;
  value?: string;
}

export interface APICallInfo {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: string;
  response?: {
    status: number;
    body?: string;
  };
}

export interface CookieInfo {
  name: string;
  value: string;
  domain: string;
  path: string;
  httpOnly: boolean;
  secure: boolean;
  sameSite: string;
}

@injectable()
export class SPAStateCrawler {
  private visitedURLs: Set<string> = new Set();
  private discoveredStates: SPAState[] = [];
  private apiCalls: APICallInfo[] = [];

  /**
   * 爬取 SPA 的所有狀態
   */
  async crawlStates(
    baseURL: string,
    maxDepth: number = 5,
    timeout: number = 60000
  ): Promise<SPAState[]> {
    const browser = await chromium.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });

    try {
      const page = await browser.newPage();

      // 設置 API 攔截
      await this.setupAPIInterceptor(page);

      // 開始爬取
      await this.crawlRecursive(page, baseURL, 0, maxDepth);

      return this.discoveredStates;
    } finally {
      await browser.close();
    }
  }

  /**
   * 遞歸爬取
   */
  private async crawlRecursive(
    page: Page,
    url: string,
    depth: number,
    maxDepth: number
  ): Promise<void> {
    // 檢查深度限制
    if (depth >= maxDepth) {
      return;
    }

    // 檢查是否已訪問
    const normalizedURL = this.normalizeURL(url);
    if (this.visitedURLs.has(normalizedURL)) {
      return;
    }

    this.visitedURLs.add(normalizedURL);

    try {
      // 訪問頁面
      await page.goto(url, {
        waitUntil: 'networkidle',
        timeout: 30000,
      });

      // 等待頁面穩定
      await page.waitForTimeout(2000);

      // 提取當前狀態
      const state = await this.extractState(page, depth);
      this.discoveredStates.push(state);

      // 查找所有可交互元素
      const interactableElements = await this.findInteractableElements(page);

      // 遞歸交互
      for (const element of interactableElements) {
        try {
          // 記錄當前 URL
          const beforeURL = page.url();

          // 執行交互
          await element.click({ timeout: 5000 });

          // 等待狀態更新
          await page.waitForTimeout(1000);

          // 檢查 URL 是否變化
          const afterURL = page.url();
          if (afterURL !== beforeURL) {
            // 遞歸爬取新狀態
            await this.crawlRecursive(page, afterURL, depth + 1, maxDepth);

            // 返回原始頁面
            await page.goBack({ waitUntil: 'networkidle', timeout: 10000 });
          }
        } catch (error) {
          // 元素不可點擊或導航失敗，繼續下一個
          console.warn(`Interaction failed: ${error}`);
        }
      }
    } catch (error) {
      console.error(`Failed to crawl ${url}:`, error);
    }
  }

  /**
   * 提取當前頁面狀態
   */
  private async extractState(page: Page, depth: number): Promise<SPAState> {
    // 提取表單信息
    const forms = await page.$$eval('form', (forms) =>
      forms.map((f) => ({
        action: f.action,
        method: f.method || 'GET',
        inputs: Array.from(f.querySelectorAll('input, select, textarea')).map(
          (input: any) => ({
            name: input.name,
            type: input.type,
            value: input.value,
          })
        ),
      }))
    );

    // 提取 localStorage
    const localStorage = await page.evaluate(() => {
      const storage: Record<string, string> = {};
      for (let i = 0; i < window.localStorage.length; i++) {
        const key = window.localStorage.key(i);
        if (key) {
          storage[key] = window.localStorage.getItem(key) || '';
        }
      }
      return storage;
    });

    // 提取 cookies
    const cookies = await page.context().cookies();
    const cookieInfo: CookieInfo[] = cookies.map((c) => ({
      name: c.name,
      value: c.value,
      domain: c.domain,
      path: c.path,
      httpOnly: c.httpOnly || false,
      secure: c.secure || false,
      sameSite: c.sameSite || 'None',
    }));

    return {
      url: page.url(),
      title: await page.title(),
      forms: forms,
      apiCalls: [...this.apiCalls], // 複製 API 調用記錄
      localStorage: localStorage,
      cookies: cookieInfo,
      depth: depth,
    };
  }

  /**
   * 查找所有可交互元素
   */
  private async findInteractableElements(page: Page) {
    return await page.$$(
      'a, button, [onclick], [role="button"], [role="link"], input[type="submit"], input[type="button"]'
    );
  }

  /**
   * 設置 API 攔截器
   */
  private async setupAPIInterceptor(page: Page): Promise<void> {
    await page.route('**/*', async (route: Route) => {
      const request = route.request();

      // 只記錄 API 調用（JSON, XHR, Fetch）
      if (
        request.resourceType() === 'xhr' ||
        request.resourceType() === 'fetch'
      ) {
        const apiCall: APICallInfo = {
          url: request.url(),
          method: request.method(),
          headers: request.headers(),
          body: request.postData() || undefined,
        };

        // 繼續請求並記錄響應
        const response = await route.fetch();
        apiCall.response = {
          status: response.status(),
          body: await response.text().catch(() => undefined),
        };

        this.apiCalls.push(apiCall);
      }

      // 繼續正常請求
      await route.continue();
    });
  }

  /**
   * 規範化 URL（移除 hash 和查詢參數）
   */
  private normalizeURL(url: string): string {
    try {
      const urlObj = new URL(url);
      return `${urlObj.origin}${urlObj.pathname}`;
    } catch {
      return url;
    }
  }

  /**
   * 生成狀態轉換圖
   */
  generateStateGraph(): Map<string, string[]> {
    const graph = new Map<string, string[]>();

    this.discoveredStates.forEach((state) => {
      const transitions: string[] = [];

      // 從表單提取可能的轉換
      state.forms.forEach((form) => {
        if (form.action) {
          transitions.push(form.action);
        }
      });

      // 從 API 調用提取轉換
      state.apiCalls.forEach((api) => {
        transitions.push(api.url);
      });

      graph.set(state.url, transitions);
    });

    return graph;
  }

  /**
   * 分析 API 端點
   */
  analyzeAPIEndpoints(): {
    uniqueEndpoints: string[];
    methodDistribution: Record<string, number>;
    authEndpoints: string[];
  } {
    const uniqueEndpoints = new Set<string>();
    const methodDistribution: Record<string, number> = {};
    const authEndpoints: string[] = [];

    this.apiCalls.forEach((call) => {
      uniqueEndpoints.add(call.url);

      // 統計方法分佈
      methodDistribution[call.method] =
        (methodDistribution[call.method] || 0) + 1;

      // 檢測認證端點
      if (
        call.url.includes('auth') ||
        call.url.includes('login') ||
        call.url.includes('token') ||
        call.headers['authorization']
      ) {
        authEndpoints.push(call.url);
      }
    });

    return {
      uniqueEndpoints: Array.from(uniqueEndpoints),
      methodDistribution,
      authEndpoints,
    };
  }

  /**
   * 檢測潛在的訪問控制問題
   */
  detectAccessControlIssues(): {
    unprotectedAdminPages: string[];
    exposedAPIKeys: string[];
    weakCookies: CookieInfo[];
  } {
    const unprotectedAdminPages: string[] = [];
    const exposedAPIKeys: string[] = [];
    const weakCookies: CookieInfo[] = [];

    this.discoveredStates.forEach((state) => {
      // 檢測未保護的管理頁面
      if (
        state.url.includes('/admin') ||
        state.url.includes('/dashboard') ||
        state.url.includes('/manage')
      ) {
        // 檢查是否需要認證
        const hasAuth = state.cookies.some(
          (c) => c.name.includes('token') || c.name.includes('session')
        );
        if (!hasAuth) {
          unprotectedAdminPages.push(state.url);
        }
      }

      // 檢測暴露的 API 密鑰
      Object.entries(state.localStorage).forEach(([key, value]) => {
        if (
          key.includes('api') ||
          key.includes('key') ||
          key.includes('token')
        ) {
          exposedAPIKeys.push(`${key}: ${value}`);
        }
      });

      // 檢測弱 Cookie 配置
      state.cookies.forEach((cookie) => {
        if (!cookie.httpOnly || !cookie.secure) {
          weakCookies.push(cookie);
        }
      });
    });

    return {
      unprotectedAdminPages,
      exposedAPIKeys,
      weakCookies,
    };
  }
}
