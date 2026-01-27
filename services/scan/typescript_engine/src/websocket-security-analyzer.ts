/**
 * WebSocket Security Analyzer
 * OWASP A05:2021 - Security Misconfiguration
 * 
 * 功能：
 * - WebSocket 連接安全檢測
 * - 消息加密驗證
 * - Origin 驗證
 * - 敏感數據洩露檢測
 */

import { Page } from 'playwright';

export interface WebSocketFinding {
  type: 'WEBSOCKET_INSECURE' | 'WEBSOCKET_NO_ORIGIN_CHECK' | 'WEBSOCKET_DATA_LEAK';
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  url: string;
  wsUrl: string;
  evidence: string;
  remediation: string;
}

/**
 * WebSocket 安全分析器
 */
export class WebSocketSecurityAnalyzer {
  private page: Page;
  private findings: WebSocketFinding[] = [];

  constructor(page: Page) {
    this.page = page;
  }

  /**
   * 執行 WebSocket 安全分析
   */
  async analyze(targetUrl: string): Promise<WebSocketFinding[]> {
    this.findings = [];

    // 注入 WebSocket 監控代碼
    await this.page.goto(targetUrl, { waitUntil: 'networkidle' });

    await this.injectWebSocketMonitor();
    await this.page.waitForTimeout(3000);

    // 獲取監控結果
    const wsConnections = await this.page.evaluate(() => {
      return (window as any).__wsSecurityMonitor || [];
    });

    // 分析每個 WebSocket 連接
    for (const conn of wsConnections) {
      this.analyzeConnection(conn, targetUrl);
    }

    return this.findings;
  }

  /**
   * 注入 WebSocket 監控代碼
   */
  private async injectWebSocketMonitor(): Promise<void> {
    await this.page.evaluate(() => {
      const wsConnections: any[] = [];
      (window as any).__wsSecurityMonitor = wsConnections;

      // Hook WebSocket constructor
      const OriginalWebSocket = window.WebSocket;
      
      (window as any).WebSocket = function(url: string, protocols?: string | string[]) {
        const conn: any = {
          url,
          protocols,
          isSecure: url.startsWith('wss://'),
          messages: [],
          errors: [],
          timestamp: new Date().toISOString(),
        };

        wsConnections.push(conn);

        // 創建真實的 WebSocket 連接
        const ws = new OriginalWebSocket(url, protocols);

        // Hook onmessage
        const originalOnMessage = ws.onmessage;
        ws.onmessage = function(event) {
          conn.messages.push({
            data: String(event.data).substring(0, 500),
            timestamp: new Date().toISOString(),
          });
          
          if (originalOnMessage) {
            originalOnMessage.call(this, event);
          }
        };

        // Hook onerror
        const originalOnError = ws.onerror;
        ws.onerror = function(event) {
          conn.errors.push({
            error: String(event),
            timestamp: new Date().toISOString(),
          });
          
          if (originalOnError) {
            originalOnError.call(this, event);
          }
        };

        return ws;
      };

      // 保留原型鏈
      (window as any).WebSocket.prototype = OriginalWebSocket.prototype;
    });
  }

  /**
   * 分析單個 WebSocket 連接
   */
  private analyzeConnection(conn: any, targetUrl: string): void {
    // 1. 檢查是否使用加密連接
    if (!conn.isSecure) {
      this.findings.push({
        type: 'WEBSOCKET_INSECURE',
        severity: 'High',
        url: targetUrl,
        wsUrl: conn.url,
        evidence: `Insecure WebSocket connection detected: ${conn.url}`,
        remediation: 'Use wss:// instead of ws:// for encrypted WebSocket connections.',
      });
    }

    // 2. 檢查消息中的敏感數據
    for (const msg of conn.messages) {
      if (this.containsSensitiveData(msg.data)) {
        this.findings.push({
          type: 'WEBSOCKET_DATA_LEAK',
          severity: 'Medium',
          url: targetUrl,
          wsUrl: conn.url,
          evidence: `Sensitive data detected in WebSocket message: ${msg.data.substring(0, 100)}...`,
          remediation: 'Encrypt sensitive data before sending over WebSocket.',
        });
        break; // 只報告一次
      }
    }

    // 3. 檢查 Origin（需要在服務器端驗證，這裡只是提示）
    if (conn.url && !conn.url.includes(new URL(targetUrl).hostname)) {
      this.findings.push({
        type: 'WEBSOCKET_NO_ORIGIN_CHECK',
        severity: 'Medium',
        url: targetUrl,
        wsUrl: conn.url,
        evidence: `WebSocket connecting to different origin: ${conn.url}`,
        remediation: 'Verify Origin header on server-side to prevent CSRF attacks.',
      });
    }
  }

  /**
   * 檢查是否包含敏感數據
   */
  private containsSensitiveData(data: string): boolean {
    const sensitivePatterns = [
      /password/i,
      /token/i,
      /api[_-]?key/i,
      /secret/i,
      /credit[_-]?card/i,
      /ssn/i,
      /\b\d{3}-\d{2}-\d{4}\b/, // SSN pattern
      /\b\d{16}\b/, // Credit card pattern
      /Bearer\s+[A-Za-z0-9\-._~+\/]+=*/i, // JWT token
    ];

    return sensitivePatterns.some(pattern => pattern.test(data));
  }

  getFindings(): WebSocketFinding[] {
    return this.findings;
  }
}
