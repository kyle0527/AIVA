/**
 * Network Interceptor Service
 * 網路請求攔截與分析服務
 */

import { Page } from 'playwright-core';
import { logger } from '../utils/logger.js';
import { NetworkRequest } from '../interfaces/dynamic-scan.interfaces.js';

export class NetworkInterceptor {
  private requests: NetworkRequest[] = [];
  private isActive: boolean = false;
  private page: Page | null = null;
  
  // 儲存監聽器引用以便後續移除
  private requestHandler: ((request: any) => void) | null = null;
  private responseHandler: ((response: any) => void) | null = null;
  private failureHandler: ((request: any) => void) | null = null;
  
  constructor() {}

  /**
   * 開始攔截網路請求
   */
  async startInterception(page: Page): Promise<void> {
    // 如果已經在攔截，先清理舊的監聽器
    if (this.isActive && this.page) {
      this.removeListeners();
    }

    this.page = page;
    this.requests = [];
    this.isActive = true;

    // 建立監聽器函數
    this.requestHandler = (request: any) => {
      const networkRequest: NetworkRequest = {
        url: request.url(),
        method: request.method(),
        headers: request.headers(),
        post_data: request.postData(),
        timestamp: Date.now()
      };
      
      this.requests.push(networkRequest);
      
      logger.debug({
        url: request.url(),
        method: request.method(),
        resource_type: request.resourceType()
      }, '📡 Network Request Intercepted');
    };

    this.responseHandler = (response: any) => {
      const request = this.requests.find(req => req.url === response.url());
      if (request) {
        request.response_status = response.status();
        request.response_headers = response.headers();
      }
      
      logger.debug({
        url: response.url(),
        status: response.status()
      }, '📨 Network Response Intercepted');
    };

    this.failureHandler = (request: any) => {
      logger.warn({
        url: request.url(),
        failure: request.failure()?.errorText
      }, '❌ Network Request Failed');
    };

    // 註冊監聽器
    page.on('request', this.requestHandler);
    page.on('response', this.responseHandler);
    page.on('requestfailed', this.failureHandler);

    logger.info('🕸️  Network interception started');
  }

  /**
   * 移除事件監聽器
   */
  private removeListeners(): void {
    if (!this.page) return;

    if (this.requestHandler) {
      this.page.off('request', this.requestHandler);
    }
    if (this.responseHandler) {
      this.page.off('response', this.responseHandler);
    }
    if (this.failureHandler) {
      this.page.off('requestfailed', this.failureHandler);
    }

    this.requestHandler = null;
    this.responseHandler = null;
    this.failureHandler = null;
  }

  /**
   * 停止攔截
   */
  stopInterception(): NetworkRequest[] {
    this.isActive = false;
    
    // 移除事件監聽器
    this.removeListeners();
    
    const capturedRequests = [...this.requests];
    this.requests = [];
    this.page = null;
    
    logger.info({
      total_requests: capturedRequests.length
    }, '🛑 Network interception stopped');
    
    return capturedRequests;
  }

  /**
   * 獲取當前攔截到的請求
   */
  getRequests(): NetworkRequest[] {
    return [...this.requests];
  }

  /**
   * 過濾 API 請求
   */
  getApiRequests(): NetworkRequest[] {
    return this.requests.filter(req => 
      req.method !== 'GET' || 
      req.url.includes('/api/') ||
      req.url.includes('.json') ||
      this.isLikelyApiEndpoint(req.url)
    );
  }

  /**
   * 獲取 AJAX 請求
   */
  getAjaxRequests(): NetworkRequest[] {
    return this.requests.filter(req => 
      req.headers['x-requested-with'] === 'XMLHttpRequest' ||
      req.headers['content-type']?.includes('application/json') ||
      req.headers['accept']?.includes('application/json')
    );
  }

  /**
   * 獲取 WebSocket 連接
   */
  getWebSocketRequests(): NetworkRequest[] {
    return this.requests.filter(req => 
      req.headers['upgrade']?.toLowerCase() === 'websocket' ||
      req.url.startsWith('ws://') ||
      req.url.startsWith('wss://')
    );
  }

  /**
   * 判斷是否為可能的 API 端點
   */
  private isLikelyApiEndpoint(url: string): boolean {
    const apiPatterns = [
      /\/api\//,
      /\/v\d+\//,
      /\/rest\//,
      /\/graphql/,
      /\.json$/,
      /\/ajax\//,
      /\/rpc\//
    ];

    return apiPatterns.some(pattern => pattern.test(url));
  }

  /**
   * 分析網路請求模式
   */
  analyzeRequestPatterns(): {
    unique_domains: string[];
    request_methods: Record<string, number>;
    content_types: Record<string, number>;
    potential_apis: string[];
  } {
    const domains = new Set<string>();
    const methods: Record<string, number> = {};
    const contentTypes: Record<string, number> = {};
    const potentialApis: string[] = [];

    for (const request of this.requests) {
      try {
        const url = new URL(request.url);
        domains.add(url.hostname);

        // 統計方法
        methods[request.method] = (methods[request.method] || 0) + 1;

        // 統計內容類型
        const contentType = request.headers['content-type'];
        if (contentType) {
          contentTypes[contentType] = (contentTypes[contentType] || 0) + 1;
        }

        // 識別潛在的 API
        if (this.isLikelyApiEndpoint(request.url)) {
          potentialApis.push(request.url);
        }
      } catch (error) {
        logger.warn({ url: request.url }, 'Invalid URL in request analysis');
      }
    }

    return {
      unique_domains: Array.from(domains),
      request_methods: methods,
      content_types: contentTypes,
      potential_apis: [...new Set(potentialApis)]
    };
  }

  /**
   * 清空請求記錄
   */
  clear(): void {
    this.requests = [];
  }
}