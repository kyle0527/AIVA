/**
 * SPA Route Security Analyzer
 * OWASP A01:2021 - Broken Access Control
 * 
 * 功能：
 * - SPA 路由發現
 * - 路由參數注入檢測
 * - 客戶端路由繞過檢測
 * - 未授權訪問檢測
 */

import { Page } from 'playwright';

export interface SPARouteFinding {
  type: 'SPA_ROUTE_TRAVERSAL' | 'SPA_UNAUTH_ACCESS' | 'SPA_CLIENT_SIDE_BYPASS';
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  url: string;
  route: string;
  evidence: string;
  remediation: string;
}

/**
 * SPA 路由安全分析器
 */
export class SPARouteSecurityAnalyzer {
  private page: Page;
  private findings: SPARouteFinding[] = [];
  private discoveredRoutes: Set<string> = new Set();

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * 執行 SPA 路由安全分析
   */
  async analyze(targetUrl: string): Promise<SPARouteFinding[]> {
    this.findings = [];
    this.discoveredRoutes.clear();

    await this.page.goto(targetUrl, { waitUntil: 'networkidle' });

    // 1. 檢測 SPA 框架
    const spaFramework = await this.detectSPAFramework();
    if (!spaFramework) {
      return this.findings;
    }

    // 2. 發現所有路由
    await this.discoverRoutes();

    // 3. 測試路由遍歷
    await this.testRouteTraversal(targetUrl);

    // 4. 測試客戶端繞過
    await this.testClientSideBypass(targetUrl);

    // 5. 測試未授權訪問
    await this.testUnauthorizedAccess(targetUrl);

    return this.findings;
  }

  /**
   * 檢測 SPA 框架
   */
  private async detectSPAFramework(): Promise<string | null> {
    return await this.page.evaluate(() => {
      if ((window as any).React || document.querySelector('[data-reactroot]')) {
        return 'React';
      }
      if ((window as any).Vue || document.querySelector('[data-v-]')) {
        return 'Vue';
      }
      if ((window as any).angular) {
        return 'Angular';
      }
      if ((window as any).__NEXT_DATA__) {
        return 'Next.js';
      }
      if ((window as any).Nuxt) {
        return 'Nuxt.js';
      }
      if (window.history.pushState) {
        return 'Generic SPA';
      }
      return null;
    });
  }

  /**
   * 發現所有路由
   */
  private async discoverRoutes(): Promise<void> {
    // 從頁面鏈接發現路由
    const links = await this.page.evaluate(() => {
      const anchors = Array.from(document.querySelectorAll('a[href]'));
      return anchors
        .map(a => (a as HTMLAnchorElement).href)
        .filter(href => {
          try {
            const url = new URL(href);
            return url.origin === window.location.origin;
          } catch {
            return false;
          }
        });
    });

    links.forEach(link => {
      const url = new URL(link);
      this.discoveredRoutes.add(url.pathname);
    });

    // 嘗試從 JavaScript 中提取路由定義
    const jsRoutes = await this.page.evaluate(() => {
      const routes: string[] = [];
      const scripts = Array.from(document.querySelectorAll('script'));
      
      scripts.forEach(script => {
        const content = script.textContent || '';
        
        // React Router patterns
        const reactMatches = content.match(/path:\s*["']([^"']+)["']/g);
        if (reactMatches) {
          reactMatches.forEach(match => {
            const route = match.match(/["']([^"']+)["']/)?.[1];
            if (route) routes.push(route);
          });
        }
        
        // Vue Router patterns
        const vueMatches = content.match(/path:\s*['"]([^'"]+)['"]/g);
        if (vueMatches) {
          vueMatches.forEach(match => {
            const route = match.match(/['"]([^'"]+)['"]/)?.[1];
            if (route) routes.push(route);
          });
        }
      });
      
      return routes;
    });

    jsRoutes.forEach(route => this.discoveredRoutes.add(route));
  }

  /**
   * 測試路由遍歷攻擊
   */
  private async testRouteTraversal(baseUrl: string): Promise<void> {
    const traversalPayloads = [
      '/../admin',
      '/..%2fadmin',
      '/%2e%2e/admin',
      '/user/../../admin',
      '/profile/../admin/users',
      '/.//admin',
      '/./admin',
      '//admin',
    ];

    for (const payload of traversalPayloads) {
      try {
        const testUrl = `${baseUrl}${payload}`;
        await this.page.goto(testUrl, { timeout: 5000, waitUntil: 'domcontentloaded' });
        
        // 檢查是否訪問到敏感頁面
        const content = await this.page.content();
        const title = await this.page.title();
        
        if (
          content.includes('admin') ||
          content.includes('dashboard') ||
          title.toLowerCase().includes('admin')
        ) {
          this.findings.push({
            type: 'SPA_ROUTE_TRAVERSAL',
            severity: 'High',
            url: baseUrl,
            route: payload,
            evidence: `Route traversal successful: ${testUrl} -> ${title}`,
            remediation: 'Implement proper route normalization and validation in the SPA router.',
          });
        }
      } catch (e) {
        // Navigation errors are expected
      }
    }
  }

  /**
   * 測試客戶端繞過
   */
  private async testClientSideBypass(baseUrl: string): Promise<void> {
    // 嘗試直接操作路由狀態
    const bypassResult = await this.page.evaluate(() => {
      const results: any[] = [];
      
      // 嘗試直接推送受保護的路由
      const protectedRoutes = ['/admin', '/dashboard', '/settings', '/users'];
      
      protectedRoutes.forEach(route => {
        try {
          window.history.pushState({}, '', route);
          results.push({
            route,
            success: true,
            currentPath: window.location.pathname,
          });
        } catch (e) {
          results.push({
            route,
            success: false,
            error: String(e),
          });
        }
      });
      
      return results;
    });

    for (const result of bypassResult) {
      if (result.success) {
        // 等待路由變化
        await this.page.waitForTimeout(1000);
        
        // 檢查是否真的訪問到了
        const content = await this.page.content();
        if (!content.includes('login') && !content.includes('unauthorized')) {
          this.findings.push({
            type: 'SPA_CLIENT_SIDE_BYPASS',
            severity: 'Critical',
            url: baseUrl,
            route: result.route,
            evidence: `Client-side route protection bypassed: ${result.route}`,
            remediation: 'Implement server-side authorization checks. Never rely solely on client-side routing for access control.',
          });
        }
      }
    }
  }

  /**
   * 測試未授權訪問
   */
  private async testUnauthorizedAccess(baseUrl: string): Promise<void> {
    // 清除所有存儲（模擬未登入狀態）
    await this.page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
      document.cookie.split(';').forEach(c => {
        document.cookie = c.replace(/^ +/, '').replace(/=.*/, `=;expires=${new Date().toUTCString()};path=/`);
      });
    });

    // 測試訪問受保護的路由
    const protectedRoutes = [
      '/admin',
      '/dashboard',
      '/profile',
      '/settings',
      '/users',
      '/api/admin',
    ];

    for (const route of protectedRoutes) {
      try {
        const testUrl = new URL(route, baseUrl).toString();
        const response = await this.page.goto(testUrl, { 
          timeout: 5000, 
          waitUntil: 'domcontentloaded' 
        });
        
        // 檢查響應狀態
        if (response && response.status() === 200) {
          const content = await this.page.content();
          
          // 如果沒有被重定向到登入頁面，可能存在問題
          if (
            !content.includes('login') &&
            !content.includes('signin') &&
            !content.includes('unauthorized')
          ) {
            this.findings.push({
              type: 'SPA_UNAUTH_ACCESS',
              severity: 'Critical',
              url: baseUrl,
              route,
              evidence: `Unauthorized access to protected route: ${testUrl}`,
              remediation: 'Implement proper authentication checks on both client and server side.',
            });
          }
        }
      } catch (e) {
        // Navigation errors are expected for some routes
      }
    }
  }

  getFindings(): SPARouteFinding[] {
    return this.findings;
  }
}
