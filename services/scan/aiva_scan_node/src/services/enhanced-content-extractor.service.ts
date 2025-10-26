/**
 * Enhanced Dynamic Content Extractor
 * 增強版動態內容提取器
 */

import { Page } from 'playwright-core';
import { logger } from '../utils/logger';
import { 
  DynamicContent,
  ExtractionConfig,
  JSVariable,
  EventListener 
} from '../interfaces/dynamic-scan.interfaces';

export class EnhancedContentExtractor {
  private page: Page;
  private config: ExtractionConfig;
  private extractedContents: DynamicContent[] = [];

  constructor(page: Page, config: ExtractionConfig) {
    this.page = page;
    this.config = config;
  }

  /**
   * 從頁面提取所有配置的動態內容
   */
  async extractAll(url: string): Promise<DynamicContent[]> {
    this.extractedContents = [];

    logger.info({ url }, '🔍 Starting enhanced content extraction');

    try {
      if (this.config.extract_forms) {
        await this.extractForms(url);
      }

      if (this.config.extract_links) {
        await this.extractDynamicLinks(url);
      }

      if (this.config.extract_ajax) {
        await this.extractAjaxEndpoints(url);
      }

      if (this.config.extract_api_calls) {
        await this.extractApiCalls(url);
      }

      if (this.config.extract_websockets) {
        await this.extractWebSockets(url);
      }

      if (this.config.extract_js_variables) {
        await this.extractJSVariables(url);
      }

      if (this.config.extract_event_listeners) {
        await this.extractEventListeners(url);
      }

      logger.info({
        url,
        total_contents: this.extractedContents.length
      }, '✅ Content extraction completed');

    } catch (error: any) {
      logger.error({
        url,
        error: error.message
      }, '❌ Content extraction failed');
    }

    return [...this.extractedContents];
  }

  /**
   * 提取表單資訊
   */
  private async extractForms(sourceUrl: string): Promise<void> {
    try {
      const forms = await this.page.locator('form').all();
      
      for (let i = 0; i < forms.length; i++) {
        const form = forms[i];
        const formData = await form.evaluate((el: HTMLFormElement, index: number) => {
          const inputs = Array.from(el.querySelectorAll('input, textarea, select'));
          const parameters = inputs.map(input => ({
            name: (input as HTMLInputElement).name || (input as HTMLInputElement).id || '',
            type: (input as HTMLInputElement).type || input.tagName.toLowerCase(),
            required: (input as HTMLInputElement).required || false
          })).filter(p => p.name);

          return {
            id: `form_${index}`,
            action: el.action || window.location.href,
            method: (el.method || 'GET').toUpperCase(),
            enctype: el.enctype || 'application/x-www-form-urlencoded',
            parameters: parameters,
            element_count: inputs.length
          };
        }, i);

        this.extractedContents.push({
          content_id: `form_${i}`,
          content_type: 'form',
          url: formData.action,
          source_url: sourceUrl,
          attributes: {
            method: formData.method,
            enctype: formData.enctype,
            parameters: formData.parameters,
            element_count: formData.element_count
          },
          confidence: 0.9,
          extraction_method: 'dom_analysis'
        });
      }

      logger.debug({ count: forms.length }, 'Forms extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract forms');
    }
  }

  /**
   * 提取動態生成的連結
   */
  private async extractDynamicLinks(sourceUrl: string): Promise<void> {
    try {
      // 等待動態內容載入
      if (this.config.wait_for_network_idle) {
        await this.page.waitForLoadState('networkidle', { 
          timeout: this.config.network_idle_timeout_ms 
        });
      }

      const links = await this.page.evaluate(() => {
        const linkElements = document.querySelectorAll('a[href]');
        return Array.from(linkElements).map((link, index) => {
          const href = (link as HTMLAnchorElement).href;
          const text = link.textContent?.trim() || '';
          const isJavaScriptGenerated = link.hasAttribute('data-dynamic') || 
                                       href.startsWith('javascript:') ||
                                       (link as HTMLAnchorElement).onclick !== null;
          
          return {
            id: `link_${index}`,
            href: href,
            text: text,
            is_dynamic: isJavaScriptGenerated,
            has_onclick: (link as HTMLAnchorElement).onclick !== null
          };
        }).filter(link => link.href && !link.href.startsWith('mailto:'));
      });

      for (const link of links) {
        this.extractedContents.push({
          content_id: link.id,
          content_type: 'link',
          url: link.href,
          source_url: sourceUrl,
          text_content: link.text,
          attributes: {
            is_dynamic: link.is_dynamic,
            has_onclick: link.has_onclick
          },
          confidence: link.is_dynamic ? 0.8 : 0.6,
          extraction_method: 'dom_analysis'
        });
      }

      logger.debug({ count: links.length }, 'Links extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract links');
    }
  }

  /**
   * 提取 AJAX 端點
   */
  private async extractAjaxEndpoints(sourceUrl: string): Promise<void> {
    try {
      const ajaxEndpoints = await this.page.evaluate(() => {
        const endpoints: string[] = [];
        
        // 檢查 jQuery AJAX 調用
        if ((window as any).$) {
          const originalAjax = (window as any).$.ajax;
          (window as any).$.ajax = function(settings: any) {
            if (settings.url) {
              endpoints.push(settings.url);
            }
            return originalAjax.apply(this, arguments);
          };
        }

        // 檢查 fetch 調用
        const originalFetch = window.fetch;
        window.fetch = function(...args: Parameters<typeof fetch>) {
          const [url] = args;
          endpoints.push(typeof url === 'string' ? url : url.toString());
          return originalFetch.apply(this, args);
        };

        // 檢查 XMLHttpRequest
        const originalOpen = XMLHttpRequest.prototype.open;
        XMLHttpRequest.prototype.open = function(method: string, url: string | URL, async: boolean = true, username?: string | null, password?: string | null) {
          endpoints.push(typeof url === 'string' ? url : url.toString());
          return originalOpen.call(this, method, url, async, username, password);
        };

        return endpoints;
      });

      for (let i = 0; i < ajaxEndpoints.length; i++) {
        this.extractedContents.push({
          content_id: `ajax_${i}`,
          content_type: 'api_endpoint',
          url: ajaxEndpoints[i],
          source_url: sourceUrl,
          attributes: {
            extraction_source: 'ajax_intercept'
          },
          confidence: 0.7,
          extraction_method: 'javascript_intercept'
        });
      }

      logger.debug({ count: ajaxEndpoints.length }, 'AJAX endpoints extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract AJAX endpoints');
    }
  }

  /**
   * 提取 API 調用
   */
  private async extractApiCalls(sourceUrl: string): Promise<void> {
    try {
      const apiCalls = await this.page.evaluate(() => {
        const scripts = document.querySelectorAll('script');
        const apiPatterns = [
          /fetch\s*\(\s*['"`]([^'"`]+)['"`]/g,
          /\$\.ajax\s*\(\s*\{[^}]*url\s*:\s*['"`]([^'"`]+)['"`]/g,
          /axios\.\w+\s*\(\s*['"`]([^'"`]+)['"`]/g,
          /\.get\s*\(\s*['"`]([^'"`]+)['"`]/g,
          /\.post\s*\(\s*['"`]([^'"`]+)['"`]/g
        ];

        const foundApis: string[] = [];
        
        for (const script of Array.from(scripts)) {
          const content = script.textContent || '';
          for (const pattern of apiPatterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
              foundApis.push(match[1]);
            }
          }
        }

        return [...new Set(foundApis)]; // 去重
      });

      for (let i = 0; i < apiCalls.length; i++) {
        this.extractedContents.push({
          content_id: `api_${i}`,
          content_type: 'api_endpoint',
          url: apiCalls[i],
          source_url: sourceUrl,
          attributes: {
            extraction_source: 'static_analysis'
          },
          confidence: 0.8,
          extraction_method: 'script_analysis'
        });
      }

      logger.debug({ count: apiCalls.length }, 'API calls extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract API calls');
    }
  }

  /**
   * 提取 WebSocket 連接
   */
  private async extractWebSockets(sourceUrl: string): Promise<void> {
    try {
      const webSockets = await this.page.evaluate(() => {
        const sockets: string[] = [];
        const scripts = document.querySelectorAll('script');
        
        const wsPatterns = [
          /new\s+WebSocket\s*\(\s*['"`]([^'"`]+)['"`]/g,
          /ws:\/\/[^\s'"`]+/g,
          /wss:\/\/[^\s'"`]+/g
        ];

        for (const script of Array.from(scripts)) {
          const content = script.textContent || '';
          for (const pattern of wsPatterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
              sockets.push(match[1] || match[0]);
            }
          }
        }

        return [...new Set(sockets)];
      });

      for (let i = 0; i < webSockets.length; i++) {
        this.extractedContents.push({
          content_id: `ws_${i}`,
          content_type: 'websocket',
          url: webSockets[i],
          source_url: sourceUrl,
          attributes: {
            protocol: webSockets[i].startsWith('wss://') ? 'secure' : 'insecure'
          },
          confidence: 0.9,
          extraction_method: 'script_analysis'
        });
      }

      logger.debug({ count: webSockets.length }, 'WebSockets extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract WebSockets');
    }
  }

  /**
   * 提取 JavaScript 變量
   */
  private async extractJSVariables(sourceUrl: string): Promise<void> {
    try {
      const jsVariables = await this.page.evaluate(() => {
        const variables: JSVariable[] = [];
        
        // 獲取全域變數
        for (const key of Object.keys(window)) {
          if (!key.startsWith('webkit') && 
              !key.startsWith('chrome') && 
              typeof (window as any)[key] !== 'function') {
            
            const value = (window as any)[key];
            variables.push({
              name: key,
              value: typeof value === 'object' ? '[Object]' : String(value),
              type: typeof value,
              scope: 'global'
            });
          }
        }

        // 從 script 標籤分析變量聲明
        const scripts = document.querySelectorAll('script');
        for (const script of Array.from(scripts)) {
          const content = script.textContent || '';
          const varPatterns = [
            /var\s+(\w+)\s*=\s*['"`]([^'"`]*)['"`]/g,
            /let\s+(\w+)\s*=\s*['"`]([^'"`]*)['"`]/g,
            /const\s+(\w+)\s*=\s*['"`]([^'"`]*)['"`]/g
          ];

          for (const pattern of varPatterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
              variables.push({
                name: match[1],
                value: match[2],
                type: 'string',
                scope: 'local'
              });
            }
          }
        }

        return variables.slice(0, 50); // 限制數量
      });

      for (let i = 0; i < jsVariables.length; i++) {
        this.extractedContents.push({
          content_id: `jsvar_${i}`,
          content_type: 'js_variable',
          url: sourceUrl,
          source_url: sourceUrl,
          text_content: `${jsVariables[i].name} = ${jsVariables[i].value}`,
          attributes: {
            variable_name: jsVariables[i].name,
            variable_type: jsVariables[i].type,
            scope: jsVariables[i].scope
          },
          confidence: 0.6,
          extraction_method: 'runtime_analysis'
        });
      }

      logger.debug({ count: jsVariables.length }, 'JS variables extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract JS variables');
    }
  }

  /**
   * 提取事件監聽器
   */
  private async extractEventListeners(sourceUrl: string): Promise<void> {
    try {
      const eventListeners = await this.page.evaluate(() => {
        const listeners: EventListener[] = [];
        const elements = document.querySelectorAll('*');
        
        for (const [index, element] of Array.from(elements).entries()) {
          // 檢查內聯事件處理器
          const onclickHandler = element.getAttribute('onclick');
          if (onclickHandler) {
            listeners.push({
              event_type: 'click',
              target_selector: element.tagName.toLowerCase() + `[${index}]`,
              handler_code: onclickHandler
            });
          }

          // 檢查其他內聯事件
          const eventAttrs = ['onmouseover', 'onmouseout', 'onload', 'onerror', 'onchange'];
          for (const attr of eventAttrs) {
            const handler = element.getAttribute(attr);
            if (handler) {
              listeners.push({
                event_type: attr.substring(2), // 移除 'on' 前綴
                target_selector: element.tagName.toLowerCase() + `[${index}]`,
                handler_code: handler
              });
            }
          }
        }

        return listeners.slice(0, 30); // 限制數量
      });

      for (let i = 0; i < eventListeners.length; i++) {
        this.extractedContents.push({
          content_id: `event_${i}`,
          content_type: 'event_listener',
          url: sourceUrl,
          source_url: sourceUrl,
          text_content: eventListeners[i].handler_code,
          attributes: {
            event_type: eventListeners[i].event_type,
            target_selector: eventListeners[i].target_selector
          },
          confidence: 0.7,
          extraction_method: 'dom_analysis'
        });
      }

      logger.debug({ count: eventListeners.length }, 'Event listeners extracted');
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to extract event listeners');
    }
  }

  /**
   * 獲取提取統計
   */
  getExtractionStats(): Record<string, number> {
    const stats: Record<string, number> = {};
    
    for (const content of this.extractedContents) {
      const type = content.content_type;
      stats[type] = (stats[type] || 0) + 1;
    }

    return stats;
  }

  /**
   * 清空提取結果
   */
  clear(): void {
    this.extractedContents = [];
  }
}