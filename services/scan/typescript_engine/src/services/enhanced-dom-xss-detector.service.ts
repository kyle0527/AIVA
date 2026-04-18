// Enhanced DOM XSS Detector
// OWASP A03:2021 - Injection
//
// 通過攔截 JavaScript sink 函數檢測 DOM-based XSS

import { Page, Browser, chromium } from 'playwright';
import { injectable } from 'inversify';

export interface XSSResult {
  vulnerable: boolean;
  sinks: XSSSinkDetection[];
  marker: string;
  testURL: string;
}

export interface XSSSinkDetection {
  sink: string;
  payload: any[];
  timestamp: number;
  stackTrace?: string;
}

@injectable()
export class DOMXSSDetector {
  private dangerousSinks: string[] = [
    'document.write',
    'document.writeln',
    'eval',
    'setTimeout',
    'setInterval',
    'Function',
    'innerHTML',
    'outerHTML',
    'insertAdjacentHTML',
    'location.href',
    'location.assign',
    'location.replace',
    'window.open',
    'execScript',
  ];

  /**
   * 在頁面中攔截所有危險的 sink 函數
   */
  async interceptSinks(page: Page, marker: string): Promise<void> {
    await page.addInitScript(
      ([sinks, marker]: [string[], string]) => {
        // 存儲檢測結果
        (window as any).__xss_found = [];

        sinks.forEach((sink: string) => {
          const parts = sink.split('.');

          if (parts.length === 2) {
            const obj = parts[0];
            const method = parts[1];

            try {
              // 獲取原始對象和方法
              const target = obj === 'window' ? window : (window as any)[obj];
              if (!target || !target[method]) {
                return;
              }

              const original = target[method];

              // 覆蓋方法
              if (typeof original === 'function') {
                target[method] = function (...args: any[]) {
                  // 檢查參數是否包含標記
                  const argsStr = args.map((arg) => String(arg)).join('');
                  if (argsStr.includes(marker)) {
                    console.error(`[XSS] Detected in ${sink}:`, args);

                    // 記錄檢測結果
                    (window as any).__xss_found.push({
                      sink: sink,
                      payload: args,
                      timestamp: Date.now(),
                      stackTrace: new Error().stack,
                    });
                  }

                  // 調用原始方法
                  return original.apply(this, args);
                };
              }
            } catch (e) {
              console.warn(`Failed to intercept ${sink}:`, e);
            }
          } else if (parts.length === 1) {
            // 處理全局函數如 eval, setTimeout
            const method = parts[0];
            try {
              const original = (window as any)[method];
              if (typeof original === 'function') {
                (window as any)[method] = function (...args: any[]) {
                  const argsStr = args.map((arg) => String(arg)).join('');
                  if (argsStr.includes(marker)) {
                    console.error(`[XSS] Detected in ${method}:`, args);
                    (window as any).__xss_found.push({
                      sink: method,
                      payload: args,
                      timestamp: Date.now(),
                      stackTrace: new Error().stack,
                    });
                  }
                  return original.apply(this, args);
                };
              }
            } catch (e) {
              console.warn(`Failed to intercept ${method}:`, e);
            }
          }
        });

        // 攔截 innerHTML setter
        try {
          const elementProto = Element.prototype;
          const innerHTMLDescriptor = Object.getOwnPropertyDescriptor(
            elementProto,
            'innerHTML'
          );

          if (innerHTMLDescriptor && innerHTMLDescriptor.set) {
            const originalSetter = innerHTMLDescriptor.set;

            Object.defineProperty(elementProto, 'innerHTML', {
              set: function (value: string) {
                if (value && value.includes(marker)) {
                  console.error('[XSS] Detected in innerHTML:', value);
                  (window as any).__xss_found.push({
                    sink: 'innerHTML',
                    payload: [value],
                    timestamp: Date.now(),
                    stackTrace: new Error().stack,
                  });
                }
                return originalSetter!.call(this, value);
              },
              get: innerHTMLDescriptor.get,
              configurable: true,
            });
          }
        } catch (e) {
          console.warn('Failed to intercept innerHTML:', e);
        }
      },
      [this.dangerousSinks, marker] as any

    );
  }

  /**
   * 測試 DOM XSS 漏洞
   */
  async testDOMXSS(
    url: string,
    injectionPoint: string,
    browser?: Browser
  ): Promise<XSSResult> {
    const marker = `xss_test_${Date.now()}_${Math.random()
      .toString(36)
      .substring(7)}`;
    const testURL = url.replace(injectionPoint, encodeURIComponent(marker));

    const browserToUse = browser || (await chromium.launch({ headless: true }));
    const page = await browserToUse.newPage();

    try {
      // 設置攔截
      await this.interceptSinks(page, marker);

      // 訪問測試 URL
      await page.goto(testURL, {
        waitUntil: 'networkidle',
        timeout: 30000,
      });

      // 等待異步腳本執行
      await page.waitForTimeout(3000);

      // 獲取檢測結果
      const xssFound = await page.evaluate(() => (window as any).__xss_found);

      return {
        vulnerable: xssFound && xssFound.length > 0,
        sinks: xssFound || [],
        marker: marker,
        testURL: testURL,
      };
    } catch (error) {
      throw new Error(`DOM XSS test failed: ${error}`);
    } finally {
      await page.close();
      if (!browser) {
        await browserToUse.close();
      }
    }
  }

  /**
   * 批量測試多個注入點
   */
  async testMultipleInjectionPoints(
    url: string,
    injectionPoints: string[]
  ): Promise<Map<string, XSSResult>> {
    const browser = await chromium.launch({ headless: true });
    const results = new Map<string, XSSResult>();

    try {
      for (const point of injectionPoints) {
        try {
          const result = await this.testDOMXSS(url, point, browser);
          results.set(point, result);
        } catch (error) {
          console.error(
            `Failed to test injection point ${point}:`,
            error
          );
        }
      }
    } finally {
      await browser.close();
    }

    return results;
  }

  /**
   * 自動發現潛在的注入點
   */
  async discoverInjectionPoints(url: string): Promise<string[]> {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const injectionPoints: string[] = [];

    try {
      await page.goto(url, { waitUntil: 'networkidle' });

      // 分析 URL 參數
      const urlObj = new URL(url);
      urlObj.searchParams.forEach((value, key) => {
        if (value) {
          injectionPoints.push(value);
        }
      });

      // 分析頁面中的用戶輸入
      const inputs = await page.evaluate(() => {
        const results: string[] = [];

        // 檢查 URL hash
        if (window.location.hash) {
          results.push(window.location.hash);
        }

        // 檢查 document.referrer
        if (document.referrer) {
          results.push(document.referrer);
        }

        // 檢查 localStorage/sessionStorage
        try {
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key) {
              const value = localStorage.getItem(key);
              if (value) {
                results.push(value);
              }
            }
          }
        } catch (e) {
          // Ignore storage access errors
        }

        return results;
      });

      injectionPoints.push(...inputs);
    } finally {
      await page.close();
      await browser.close();
    }

    return injectionPoints;
  }

  /**
   * 生成 XSS payload
   */
  generatePayloads(): string[] {
    return [
      '<script>alert(1)</script>',
      '<img src=x onerror=alert(1)>',
      '<svg onload=alert(1)>',
      'javascript:alert(1)',
      '" onclick="alert(1)',
      "' onclick='alert(1)",
      '<iframe src=javascript:alert(1)>',
      '<body onload=alert(1)>',
      '<input onfocus=alert(1) autofocus>',
      '<select onfocus=alert(1) autofocus>',
      '<textarea onfocus=alert(1) autofocus>',
      '<keygen onfocus=alert(1) autofocus>',
      '<video><source onerror="alert(1)">',
      '<audio src=x onerror=alert(1)>',
      '<details open ontoggle=alert(1)>',
      '<marquee onstart=alert(1)>',
    ];
  }
}
