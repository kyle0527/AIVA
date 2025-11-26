/**
 * User Interaction Simulator Service
 * 用戶互動模擬服務
 */

import { Page } from 'playwright-core';
import { logger } from '../utils/logger';
import { 
  InteractionConfig, 
  InteractionResult, 
  DOMChange 
} from '../interfaces/dynamic-scan.interfaces';

export class InteractionSimulator {
  private page: Page;
  private config: InteractionConfig;
  
  constructor(page: Page, config: InteractionConfig) {
    this.page = page;
    this.config = config;
  }

  /**
   * 執行所有配置的互動
   */
  async executeAll(): Promise<InteractionResult[]> {
    const results: InteractionResult[] = [];

    logger.info('🎭 Starting user interaction simulation');

    // 設置 DOM 變更監聽器
    await this.setupDOMObserver();

    try {
      if (this.config.click_buttons) {
        const clickResults = await this.simulateButtonClicks();
        results.push(...clickResults);
      }

      if (this.config.fill_forms) {
        const formResults = await this.simulateFormFilling();
        results.push(...formResults);
      }

      if (this.config.scroll_pages) {
        const scrollResults = await this.simulateScrolling();
        results.push(...scrollResults);
      }

      if (this.config.hover_elements) {
        const hoverResults = await this.simulateHovering();
        results.push(...hoverResults);
      }

      if (this.config.trigger_events) {
        const eventResults = await this.simulateKeyboardEvents();
        results.push(...eventResults);
      }

      logger.info({
        total_interactions: results.length,
        successful: results.filter(r => r.success).length
      }, '✅ User interaction simulation completed');

    } catch (error: any) {
      logger.error({ error: error.message }, '❌ Interaction simulation failed');
    }

    return results;
  }

  /**
   * 設置 DOM 變更觀察器
   */
  private async setupDOMObserver(): Promise<void> {
    await this.page.addInitScript(() => {
      if (typeof window !== 'undefined') {
        (window as any).domChanges = [];
        
        const observer = new MutationObserver((mutations) => {
          for (const mutation of mutations) {
            const change: any = {
              type: mutation.type as 'childList' | 'attributes' | 'subtree',
              target_node: mutation.target.nodeName,
              timestamp: Date.now()
            };

            if (mutation.type === 'childList') {
              change.added_nodes = Array.from(mutation.addedNodes).map(n => n.nodeName);
              change.removed_nodes = Array.from(mutation.removedNodes).map(n => n.nodeName);
            } else if (mutation.type === 'attributes') {
              change.attribute_name = mutation.attributeName;
              change.old_value = mutation.oldValue;
            }

            (window as any).domChanges.push(change);
          }
        });

        observer.observe(document.body || document.documentElement, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeOldValue: true
        });
      }
    });
  }

  /**
   * 模擬按鈕點擊
   */
  private async simulateButtonClicks(): Promise<InteractionResult[]> {
    const results: InteractionResult[] = [];

    try {
      const buttons = await this.page.locator('button, input[type="submit"], input[type="button"], [role="button"]').all();
      
      for (let i = 0; i < Math.min(buttons.length, this.config.max_interactions); i++) {
        const button = buttons[i];
        
        try {
          // 檢查按鈕是否可見且可點擊
          const isVisible = await button.isVisible();
          const isEnabled = await button.isEnabled();
          
          if (!isVisible || !isEnabled) continue;

          const selector = await this.getElementSelector(button);
          
          logger.debug({ selector }, '👆 Clicking button');
          
          await button.click({ timeout: this.config.wait_time_ms });
          
          // 智能等待：網路閒置或 DOM 穩定，最多等待 2 秒
          await Promise.race([
            this.page.waitForLoadState('networkidle', { timeout: 2000 }),
            this.page.waitForTimeout(500) // 最短等待
          ]).catch(() => {}); // 超時也沒關係，繼續執行
          
          const afterDOMChanges = await this.captureDOMChanges();
          
          results.push({
            interaction_type: 'click',
            target_selector: selector,
            success: true,
            dom_changes: afterDOMChanges,
            network_requests: [], // 將由 NetworkInterceptor 提供
            timestamp: Date.now()
          });

        } catch (error: any) {
          const selector = await this.getElementSelector(button).catch(() => 'unknown');
          
          results.push({
            interaction_type: 'click',
            target_selector: selector,
            success: false,
            error_message: error.message,
            dom_changes: [],
            network_requests: [],
            timestamp: Date.now()
          });
        }
      }
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to simulate button clicks');
    }

    return results;
  }

  /**
   * 模擬表單填寫
   */
  private async simulateFormFilling(): Promise<InteractionResult[]> {
    const results: InteractionResult[] = [];

    try {
      const inputs = await this.page.locator('input[type="text"], input[type="email"], input[type="password"], textarea').all();
      
      for (const input of inputs) {
        try {
          const isVisible = await input.isVisible();
          if (!isVisible) continue;

          const type = await input.getAttribute('type') || 'text';
          const name = await input.getAttribute('name') || '';
          const selector = await this.getElementSelector(input);
          
          let testValue = this.getTestValueForInput(type, name);
          
          logger.debug({ selector, type, value: testValue }, '✏️  Filling input');
          
          await input.clear();
          await input.fill(testValue);
          await input.blur(); // 觸發 blur 事件
          
          const domChanges = await this.captureDOMChanges();
          
          results.push({
            interaction_type: 'input',
            target_selector: selector,
            success: true,
            dom_changes: domChanges,
            network_requests: [],
            timestamp: Date.now()
          });

        } catch (error: any) {
          const selector = await this.getElementSelector(input).catch(() => 'unknown');
          
          results.push({
            interaction_type: 'input',
            target_selector: selector,
            success: false,
            error_message: error.message,
            dom_changes: [],
            network_requests: [],
            timestamp: Date.now()
          });
        }
      }
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to simulate form filling');
    }

    return results;
  }

  /**
   * 模擬滾動操作
   */
  private async simulateScrolling(): Promise<InteractionResult[]> {
    const results: InteractionResult[] = [];

    try {
      logger.debug('📜 Simulating page scrolling');
      
      // 向下滾動
      await this.page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight / 2);
      });
      await this.page.waitForTimeout(1000);
      
      // 滾動到底部
      await this.page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight);
      });
      await this.page.waitForTimeout(1000);
      
      // 滾回頂部
      await this.page.evaluate(() => {
        window.scrollTo(0, 0);
      });
      await this.page.waitForTimeout(500);
      
      const domChanges = await this.captureDOMChanges();
      
      results.push({
        interaction_type: 'scroll',
        target_selector: 'body',
        success: true,
        dom_changes: domChanges,
        network_requests: [],
        timestamp: Date.now()
      });

    } catch (error: any) {
      results.push({
        interaction_type: 'scroll',
        target_selector: 'body',
        success: false,
        error_message: error.message,
        dom_changes: [],
        network_requests: [],
        timestamp: Date.now()
      });
    }

    return results;
  }

  /**
   * 模擬懸停操作
   */
  private async simulateHovering(): Promise<InteractionResult[]> {
    const results: InteractionResult[] = [];

    try {
      const hoverTargets = await this.page.locator('a, button, [onmouseover], [onmouseenter]').all();
      
      for (let i = 0; i < Math.min(hoverTargets.length, 5); i++) {
        const element = hoverTargets[i];
        
        try {
          const isVisible = await element.isVisible();
          if (!isVisible) continue;

          const selector = await this.getElementSelector(element);
          
          logger.debug({ selector }, '🖱️  Hovering element');
          
          await element.hover();
          await this.page.waitForTimeout(500);
          
          const domChanges = await this.captureDOMChanges();
          
          results.push({
            interaction_type: 'hover',
            target_selector: selector,
            success: true,
            dom_changes: domChanges,
            network_requests: [],
            timestamp: Date.now()
          });

        } catch (error: any) {
          const selector = await this.getElementSelector(element).catch(() => 'unknown');
          
          results.push({
            interaction_type: 'hover',
            target_selector: selector,
            success: false,
            error_message: error.message,
            dom_changes: [],
            network_requests: [],
            timestamp: Date.now()
          });
        }
      }
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to simulate hovering');
    }

    return results;
  }

  /**
   * 模擬鍵盤事件
   */
  private async simulateKeyboardEvents(): Promise<InteractionResult[]> {
    const results: InteractionResult[] = [];

    try {
      // 模擬常見的鍵盤操作
      const keyEvents = ['Tab', 'Enter', 'Escape'];
      
      for (const key of keyEvents) {
        try {
          logger.debug({ key }, '⌨️  Simulating keyboard event');
          
          await this.page.keyboard.press(key);
          await this.page.waitForTimeout(300);
          
          const domChanges = await this.captureDOMChanges();
          
          results.push({
            interaction_type: 'keyboard',
            target_selector: key,
            success: true,
            dom_changes: domChanges,
            network_requests: [],
            timestamp: Date.now()
          });

        } catch (error: any) {
          results.push({
            interaction_type: 'keyboard',
            target_selector: key,
            success: false,
            error_message: error.message,
            dom_changes: [],
            network_requests: [],
            timestamp: Date.now()
          });
        }
      }
    } catch (error: any) {
      logger.error({ error: error.message }, 'Failed to simulate keyboard events');
    }

    return results;
  }

  /**
   * 獲取元素選擇器
   */
  private async getElementSelector(element: any): Promise<string> {
    try {
      const id = await element.getAttribute('id');
      if (id) return `#${id}`;
      
      const className = await element.getAttribute('class');
      if (className) {
        const firstClass = className.split(' ')[0];
        return `.${firstClass}`;
      }
      
      const tagName = await element.evaluate((el: Element) => el.tagName.toLowerCase());
      return tagName;
    } catch {
      return 'unknown';
    }
  }

  /**
   * 根據輸入類型生成測試值
   */
  private getTestValueForInput(type: string, name: string): string {
    const nameLower = name.toLowerCase();
    
    switch (type) {
      case 'email':
        return 'test@example.com';
      case 'password':
        return 'TestPass123';
      case 'tel':
        return '+1-555-0123';
      case 'url':
        return 'https://example.com';
      case 'number':
        return '123';
      default:
        if (nameLower.includes('name')) return 'Test User';
        if (nameLower.includes('search') || nameLower.includes('query')) return 'test query';
        if (nameLower.includes('message') || nameLower.includes('comment')) return 'Test message content';
        return 'test input';
    }
  }



  /**
   * 捕獲 DOM 變更
   */
  private async captureDOMChanges(): Promise<DOMChange[]> {
    try {
      const changes = await this.page.evaluate(() => {
        return (window as any).domChanges || [];
      });
      
      // 清空已捕獲的變更
      await this.page.evaluate(() => {
        (window as any).domChanges = [];
      });
      
      return changes;
    } catch {
      return [];
    }
  }
}