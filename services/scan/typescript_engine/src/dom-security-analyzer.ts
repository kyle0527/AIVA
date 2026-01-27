/**
 * Enhanced DOM Security Analyzer v2.0
 * OWASP A03:2021 - Injection (DOM XSS)
 * 
 * 功能：
 * - Source-to-Sink 追蹤
 * - PostMessage 安全檢測
 * - DOM Clobbering 檢測
 * - WebSocket 安全分析
 * - SPA 路由分析
 */

import { Page, Browser } from 'playwright';

/**
 * DOM XSS Source（數據來源）
 */
const DOM_SOURCES = [
  'location.href',
  'location.search',
  'location.hash',
  'document.URL',
  'document.documentURI',
  'document.referrer',
  'window.name',
  'document.cookie',
  'localStorage',
  'sessionStorage',
];

/**
 * DOM XSS Sink（危險函數）
 */
const DOM_SINKS = [
  'eval',
  'Function',
  'setTimeout',
  'setInterval',
  'document.write',
  'document.writeln',
  'innerHTML',
  'outerHTML',
  'insertAdjacentHTML',
  'location.href',
  'location.assign',
  'location.replace',
  'open',
  'execScript',
];

/**
 * DOM 安全發現
 */
export interface DOMSecurityFinding {
  type: 'DOM_XSS' | 'POSTMESSAGE' | 'DOM_CLOBBERING' | 'WEBSOCKET' | 'SPA_ROUTE';
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  confidence: 'High' | 'Medium' | 'Low';
  url: string;
  evidence: string;
  source?: string;
  sink?: string;
  payload?: string;
  remediation: string;
}

/**
 * 增強的 DOM 安全分析器
 */
export class DOMSecurityAnalyzer {
  private page: Page;
  private findings: DOMSecurityFinding[] = [];

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * 執行完整的 DOM 安全分析
   */
  async analyze(targetUrl: string): Promise<DOMSecurityFinding[]> {
    this.findings = [];

    await this.page.goto(targetUrl, { waitUntil: 'networkidle' });

    // 1. Source-to-Sink 追蹤
    await this.analyzeSourceToSink();

    // 2. PostMessage 安全
    await this.analyzePostMessage();

    // 3. DOM Clobbering
    await this.analyzeDOMClobbering();

    // 4. WebSocket 安全
    await this.analyzeWebSocket();

    // 5. SPA 路由安全
    await this.analyzeSPARoutes();

    return this.findings;
  }

  /**
   * Source-to-Sink 數據流追蹤
   */
  private async analyzeSourceToSink(): Promise<void> {
    const sourceToSinkCode = `
      (function() {
        const results = [];
        
        // 追蹤所有危險的數據流
        const sources = ${JSON.stringify(DOM_SOURCES)};
        const sinks = ${JSON.stringify(DOM_SINKS)};
        
        // Hook 所有 Sink 函數
        sinks.forEach(sink => {
          const parts = sink.split('.');
          let obj = window;
          let prop = sink;
          
          if (parts.length > 1) {
            for (let i = 0; i < parts.length - 1; i++) {
              if (obj[parts[i]]) {
                obj = obj[parts[i]];
              }
            }
            prop = parts[parts.length - 1];
          }
          
          // 保存原始函數
          const original = obj[prop];
          if (typeof original === 'function' || prop === 'innerHTML' || prop === 'outerHTML') {
            Object.defineProperty(obj, prop, {
              set: function(value) {
                // 檢查是否來自 Source
                const trace = new Error().stack;
                sources.forEach(source => {
                  if (trace.includes(source) || String(value).includes('location.') || String(value).includes('document.')) {
                    results.push({
                      source: source,
                      sink: sink,
                      value: String(value).substring(0, 200),
                      trace: trace.substring(0, 500)
                    });
                  }
                });
                
                // 執行原始操作
                if (typeof original === 'function') {
                  original.call(this, value);
                } else if (prop === 'innerHTML' || prop === 'outerHTML') {
                  this['_' + prop] = value;
                }
              },
              get: function() {
                if (typeof original === 'function') {
                  return original;
                }
                return this['_' + prop];
              }
            });
          }
        });
        
        // 等待一段時間讓頁面執行
        setTimeout(() => {
          window.__domSecurityResults = results;
        }, 2000);
        
        return 'Source-to-Sink tracking enabled';
      })();
    `;

    try {
      await this.page.evaluate(sourceToSinkCode);
      await this.page.waitForTimeout(3000);

      // 獲取結果
      const results = await this.page.evaluate(() => {
        return (window as any).__domSecurityResults || [];
      });

      // 測試各種 DOM XSS payloads
      const testPayloads = [
        { param: 'test', value: '<img src=x onerror=alert(1)>' },
        { param: 'q', value: 'javascript:alert(1)' },
        { param: 'callback', value: 'alert' },
        { param: 'redirect', value: 'javascript:alert(document.domain)' },
      ];

      for (const payload of testPayloads) {
        const testUrl = `${this.page.url()}?${payload.param}=${encodeURIComponent(payload.value)}`;
        
        try {
          await this.page.goto(testUrl, { timeout: 5000, waitUntil: 'domcontentloaded' });
          
          // 檢查是否觸發了 alert
          const alertTriggered = await this.page.evaluate(() => {
            return (window as any).__xssTriggered === true;
          });

          if (alertTriggered || results.length > 0) {
            this.findings.push({
              type: 'DOM_XSS',
              severity: 'Critical',
              confidence: 'High',
              url: testUrl,
              evidence: `DOM XSS detected via parameter ${payload.param}`,
              source: `URL parameter: ${payload.param}`,
              sink: results[0]?.sink || 'Unknown',
              payload: payload.value,
              remediation: 'Sanitize user input before inserting into DOM. Use textContent instead of innerHTML.',
            });
          }
        } catch (e) {
          // Navigation timeout is expected
        }
      }
    } catch (error) {
      console.error('Source-to-Sink analysis error:', error);
    }
  }

  /**
   * PostMessage 安全分析
   */
  private async analyzePostMessage(): Promise<void> {
    const postMessageCode = `
      (function() {
        const postMessageFindings = [];
        
        // Hook window.addEventListener
        const originalAddEventListener = window.addEventListener;
        window.addEventListener = function(type, listener, options) {
          if (type === 'message') {
            postMessageFindings.push({
              listenerCode: listener.toString().substring(0, 500),
              hasOriginCheck: listener.toString().includes('origin') || listener.toString().includes('event.origin'),
            });
          }
          return originalAddEventListener.call(this, type, listener, options);
        };
        
        // 檢查現有的 message listeners
        window.__postMessageFindings = postMessageFindings;
        
        return 'PostMessage monitoring enabled';
      })();
    `;

    await this.page.evaluate(postMessageCode);
    await this.page.waitForTimeout(1000);

    const findings = await this.page.evaluate(() => {
      return (window as any).__postMessageFindings || [];
    });

    // 測試 PostMessage 漏洞
    for (const finding of findings) {
      if (!finding.hasOriginCheck) {
        this.findings.push({
          type: 'POSTMESSAGE',
          severity: 'High',
          confidence: 'Medium',
          url: this.page.url(),
          evidence: `PostMessage listener without origin validation: ${finding.listenerCode}`,
          remediation: 'Always validate event.origin in postMessage listeners.',
        });
      }
    }

    // 嘗試發送惡意 postMessage
    try {
      await this.page.evaluate(() => {
        window.postMessage({ type: 'xss', data: '<img src=x onerror=alert(1)>' }, '*');
      });
    } catch (e) {
      // Expected
    }
  }

  /**
   * DOM Clobbering 檢測
   */
  private async analyzeDOMClobbering(): Promise<void> {
    const clobberingCode = `
      (function() {
        const clobberingFindings = [];
        
        // 檢查常見的 DOM Clobbering 目標
        const targets = [
          'defaultAvatar',
          'config',
          'isAdmin',
          'debug',
          'apiEndpoint',
          'csrf_token',
        ];
        
        targets.forEach(target => {
          // 測試 HTML 注入是否可以覆蓋變量
          const testElement = document.createElement('div');
          testElement.innerHTML = '<form id="' + target + '" name="' + target + '"><input name="value" value="hacked"></form>';
          document.body.appendChild(testElement);
          
          // 檢查是否可以訪問
          if (window[target] && window[target].value) {
            clobberingFindings.push({
              target: target,
              value: window[target].value,
              type: typeof window[target]
            });
          }
          
          testElement.remove();
        });
        
        return clobberingFindings;
      })();
    `;

    try {
      const findings = await this.page.evaluate(clobberingCode);

      if (findings && Array.isArray(findings) && findings.length > 0) {
        this.findings.push({
          type: 'DOM_CLOBBERING',
          severity: 'Medium',
          confidence: 'High',
          url: this.page.url(),
          evidence: `DOM Clobbering possible on: ${findings.map((f: any) => f.target).join(', ')}`,
          remediation: 'Use Object.hasOwnProperty() checks and avoid relying on global variables.',
        });
      }
    } catch (error) {
      console.error('DOM Clobbering analysis error:', error);
    }
  }

  /**
   * WebSocket 安全分析
   */
  private async analyzeWebSocket(): Promise<void> {
    const wsCode = `
      (function() {
        const wsFindings = [];
        
        // Hook WebSocket constructor
        const OriginalWebSocket = window.WebSocket;
        window.WebSocket = function(url, protocols) {
          wsFindings.push({
            url: url,
            protocols: protocols,
            isSecure: url.startsWith('wss://'),
          });
          
          const ws = new OriginalWebSocket(url, protocols);
          
          // Hook message handler
          const originalOnMessage = ws.onmessage;
          ws.onmessage = function(event) {
            wsFindings.push({
              type: 'message_received',
              data: String(event.data).substring(0, 200),
            });
            if (originalOnMessage) {
              originalOnMessage.call(this, event);
            }
          };
          
          return ws;
        };
        
        window.__wsFindings = wsFindings;
        return 'WebSocket monitoring enabled';
      })();
    `;

    await this.page.evaluate(wsCode);
    await this.page.waitForTimeout(2000);

    const findings = await this.page.evaluate(() => {
      return (window as any).__wsFindings || [];
    });

    for (const finding of findings) {
      if (finding.url && !finding.isSecure) {
        this.findings.push({
          type: 'WEBSOCKET',
          severity: 'Medium',
          confidence: 'High',
          url: this.page.url(),
          evidence: `Insecure WebSocket connection: ${finding.url}`,
          remediation: 'Use wss:// instead of ws:// for encrypted WebSocket connections.',
        });
      }
    }
  }

  /**
   * SPA 路由安全分析
   */
  private async analyzeSPARoutes(): Promise<void> {
    // 檢查是否是 SPA
    const isSPA = await this.page.evaluate(() => {
      return !!(
        window.history.pushState ||
        (window as any).angular ||
        (window as any).React ||
        (window as any).Vue ||
        (window as any).__NEXT_DATA__
      );
    });

    if (!isSPA) {
      return;
    }

    // 收集所有內部鏈接
    const links = await this.page.evaluate(() => {
      const anchors = Array.from(document.querySelectorAll('a[href]'));
      return anchors
        .map(a => (a as HTMLAnchorElement).href)
        .filter(href => href.startsWith(window.location.origin));
    });

    // 測試路由參數注入
    const routePatterns = [
      '/../admin',
      '/..%2fadmin',
      '/%2e%2e/admin',
      '/user/../../admin',
    ];

    for (const pattern of routePatterns) {
      try {
        const testUrl = `${this.page.url()}${pattern}`;
        await this.page.goto(testUrl, { timeout: 3000, waitUntil: 'domcontentloaded' });
        
        // 檢查是否訪問到了受保護的路由
        const content = await this.page.content();
        if (content.includes('admin') || content.includes('dashboard')) {
          this.findings.push({
            type: 'SPA_ROUTE',
            severity: 'High',
            confidence: 'Medium',
            url: testUrl,
            evidence: `Potential route traversal: ${pattern}`,
            remediation: 'Implement proper route validation and access control in SPA router.',
          });
        }
      } catch (e) {
        // Expected for invalid routes
      }
    }
  }

  /**
   * 獲取所有發現
   */
  getFindings(): DOMSecurityFinding[] {
    return this.findings;
  }
}
