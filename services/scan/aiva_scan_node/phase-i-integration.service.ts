/**
 * Phase I 整合服務 - TypeScript 掃描器
 * 
 * 負責整合客戶端授權繞過檢測與動態掃描能力
 */

import { NetworkInterceptor } from './src/services/network-interceptor.service';

export interface PhaseIConfig {
  enableJSAnalysis: boolean;
  enableDOMManipulation: boolean;
  enableNetworkInterception: boolean;
  maxScriptAnalysis: number;
}

import { FindingPayload } from '../../../schemas/aiva_schemas';

// 定義 client-side auth 專用的發現類型
type ClientSideAuthType = 'localStorage_auth' | 'sessionStorage_auth' | 'hardcoded_admin' | 'client_validation' | 'hidden_elements';

// 為了向後兼容，保留原始接口但基於標準 FindingPayload
interface ClientSideAuthFinding {
  type: ClientSideAuthType;
  severity: 'high' | 'medium' | 'low';
  description: string;
  evidence: {
    scriptUrl?: string;
    lineNumber?: number;
    codeSnippet?: string;
    domElement?: string;
  };
  recommendations: string[];
}

// 輔助函數：將 ClientSideAuthFinding 轉換為標準 FindingPayload
function toStandardFinding(
  finding: ClientSideAuthFinding, 
  taskId: string, 
  scanId: string, 
  url: string
): FindingPayload {
  return {
    finding_id: `client-auth-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    task_id: taskId,
    scan_id: scanId,
    status: 'confirmed',
    vulnerability: {
      name: finding.type as any, // 映射到適當的 VulnerabilityType
      severity: finding.severity as any,
      confidence: 'high' as any,
      description: finding.description,
      cwe: finding.type === 'client_validation' ? 'CWE-20' : 
           finding.type === 'localStorage_auth' ? 'CWE-922' :
           finding.type === 'sessionStorage_auth' ? 'CWE-922' :
           finding.type === 'hardcoded_admin' ? 'CWE-798' :
           finding.type === 'hidden_elements' ? 'CWE-602' : undefined
    },
    target: {
      url: url,
      method: 'GET',
      headers: {},
      params: {},
      body: null
    },
    strategy: 'client_side_auth_bypass',
    evidence: {
      payload: finding.evidence.codeSnippet || null,
      request: finding.evidence.scriptUrl || null,
      response: finding.evidence.domElement || null,
      proof: `${finding.type} detected at line ${finding.evidence.lineNumber || 'unknown'}`
    },
    recommendation: {
      fix: finding.recommendations.join('; '),
      priority: finding.severity === 'high' ? 'high' : 
                finding.severity === 'medium' ? 'medium' : 'low',
      remediation_steps: finding.recommendations,
      references: []
    },
    metadata: {
      detection_method: 'client_side_analysis',
      auth_type: finding.type
    },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };
}

export class PhaseIIntegrationService {
  private networkInterceptor: NetworkInterceptor;
  private browser: any; // Browser instance will be passed or created

  constructor(private config: PhaseIConfig) {
    this.networkInterceptor = new NetworkInterceptor();
  }

  /**
   * 執行客戶端授權繞過檢測 (增強版本)
   */
  async analyzeClientSideAuthBypass(url: string, browser?: any): Promise<ClientSideAuthFinding[]> {
    const findings: ClientSideAuthFinding[] = [];

    try {
      // 1. 創建頁面實例
      if (!browser) {
        throw new Error('Browser instance is required');
      }
      
      const page = await browser.newPage();
      
      // 2. 啟用網路攔截來捕獲所有 API 呼叫
      if (this.config.enableNetworkInterception) {
        await this.networkInterceptor.startInterception(page);
      }

      // 3. 導航到目標頁面
      await page.goto(url, { waitUntil: 'networkidle' });

      // 4. 提取並分析所有 JavaScript
      if (this.config.enableJSAnalysis) {
        const jsFindings = await this.analyzeJavaScriptSources(page);
        findings.push(...jsFindings);
      }

      // 5. 分析 DOM 操作和隱藏元素
      if (this.config.enableDOMManipulation) {
        const domFindings = await this.analyzeDOMManipulation(page);
        findings.push(...domFindings);
      }

      // 6. 模擬用戶交互來觸發動態授權檢查
      const interactionFindings = await this.analyzeInteractionAuthBypass(page);
      findings.push(...interactionFindings);

      await page.close();

    } catch (error) {
      console.error(`Phase I analysis error for ${url}:`, error);
    }

    return findings;
  }

  /**
   * JavaScript 源代碼分析 (瀏覽器環境)
   */
  private async analyzeJavaScriptSources(page: any): Promise<ClientSideAuthFinding[]> {
    const findings: ClientSideAuthFinding[] = [];

    // 在頁面上下文中執行分析腳本
    const analysisResult = await page.evaluate(() => {
      const results: any[] = [];

      // 檢查腳本中的授權繞過模式
      function checkScriptPatterns() {
        const storageChecks = [
          'localStorage.getItem', 'sessionStorage.getItem',
          'localStorage.token', 'localStorage.auth',
          'user.role', 'isAdmin', 'hasPermission'
        ];

        const scripts = Array.from(document.scripts);
        for (const [index, script] of scripts.entries()) {
          if (script.src || !script.textContent) continue;
          
          const content = script.textContent;
          if (content.length < 10) continue;

          // 檢查存儲模式和管理員權限
          checkStoragePatterns(content, index, storageChecks, results);
          checkAdminPatterns(content, index, results);
        }
      }

      function checkStoragePatterns(content: string, index: number, patterns: string[], results: any[]) {
        for (const pattern of patterns) {
          if (content.includes(pattern)) {
            results.push({
              type: pattern.includes('localStorage') ? 'localStorage_auth' : 'sessionStorage_auth',
              severity: 'medium',
              description: `Client-side authorization check using ${pattern}`,
              evidence: {
                scriptUrl: `inline-script-${index}`,
                codeSnippet: content.substring(
                  Math.max(0, content.indexOf(pattern) - 50),
                  content.indexOf(pattern) + 100
                )
              }
            });
          }
        }
      }

      function checkAdminPatterns(content: string, index: number, results: any[]) {
        const adminPatterns = [
          /role\s*===?\s*["']admin["']/gi,
          /user\.type\s*===?\s*["']admin["']/gi,
          /isAdmin\s*===?\s*true/gi,
          /permissions\.includes\s*\(\s*["']admin["']\s*\)/gi
        ];

        for (const pattern of adminPatterns) {
          const matches = content.match(pattern);
          if (matches) {
            for (const match of matches) {
              results.push({
                type: 'hardcoded_admin',
                severity: 'high',
                description: 'Hardcoded admin role check detected',
                evidence: {
                  scriptUrl: `inline-script-${index}`,
                  codeSnippet: match
                }
              });
            }
          }
        }
      }

      checkScriptPatterns();
      return results;
    });

    // 轉換結果格式
    for (const result of analysisResult) {
      findings.push({
        ...result,
        recommendations: this.generateRecommendations(result.type)
      });
    }

    return findings;
  }

  /**
   * DOM 操作分析
   */
  private async analyzeDOMManipulation(page: any): Promise<ClientSideAuthFinding[]> {
    const findings: ClientSideAuthFinding[] = [];

    const domAnalysis = await page.evaluate(() => {
      const results: any[] = [];

      // 檢查隱藏的管理元素
      const hiddenElements = document.querySelectorAll('[style*="display:none"], [hidden], .admin-only, .hidden');
      for (const element of Array.from(hiddenElements)) {
        const text = element.textContent || '';
        if (text.toLowerCase().includes('admin') || 
            text.toLowerCase().includes('管理') ||
            element.classList.contains('admin')) {
          
          results.push({
            type: 'hidden_elements',
            severity: 'medium',
            description: 'Hidden administrative element detected',
            evidence: {
              domElement: element.outerHTML.substring(0, 200)
            }
          });
        }
      }

      // 檢查僅客戶端的表單驗證
      const forms = document.querySelectorAll('form');
      for (const form of Array.from(forms)) {
        const onsubmit = form.getAttribute('onsubmit') || '';
        if (onsubmit.includes('validate') || onsubmit.includes('check')) {
          results.push({
            type: 'client_validation',
            severity: 'medium',
            description: 'Client-side only form validation detected',
            evidence: {
              domElement: form.outerHTML.substring(0, 200)
            }
          });
        }
      }

      return results;
    });

    for (const result of domAnalysis) {
      findings.push({
        ...result,
        recommendations: this.generateRecommendations(result.type)
      });
    }

    return findings;
  }

  /**
   * 交互式授權繞過分析
   */
  private async analyzeInteractionAuthBypass(page: any): Promise<ClientSideAuthFinding[]> {
    const findings: ClientSideAuthFinding[] = [];

    try {
      // 1. 嘗試修改 localStorage 中的角色信息
      const localStorageTest = await page.evaluate(() => {
        const originalRoles: string[] = [];
        
        // 保存原始值
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key && (key.includes('role') || key.includes('auth') || key.includes('user'))) {
            originalRoles.push(`${key}:${localStorage.getItem(key)}`);
          }
        }

        // 嘗試設置管理員權限
        localStorage.setItem('role', 'admin');
        localStorage.setItem('isAdmin', 'true');
        localStorage.setItem('userType', 'administrator');

        return originalRoles;
      });

      // 2. 重新加載頁面檢查是否生效
      await page.reload({ waitUntil: 'networkidle' });

      // 3. 檢查頁面是否顯示管理功能
      const adminElementsVisible = await page.evaluate(() => {
        const adminElements = document.querySelectorAll('.admin, [class*="admin"], #admin');
        return adminElements.length > 0 && 
               Array.from(adminElements).some(el => (el as HTMLElement).offsetParent !== null);
      });

      if (adminElementsVisible) {
        findings.push({
          type: 'localStorage_auth',
          severity: 'high',
          description: 'Authorization bypass via localStorage manipulation successful',
          evidence: {
            codeSnippet: 'localStorage.setItem("role", "admin") enabled admin features'
          },
          recommendations: this.generateRecommendations('localStorage_auth')
        });
      }

      // 4. 恢復原始值
      await page.evaluate((roles: string[]) => {
        localStorage.clear();
        for (const roleEntry of roles) {
          const [key, value] = roleEntry.split(':');
          localStorage.setItem(key, value);
        }
      }, localStorageTest);

    } catch (error) {
      console.error('Interaction analysis error:', error);
    }

    return findings;
  }

  /**
   * 生成修復建議
   */
  private generateRecommendations(type: ClientSideAuthFinding['type']): string[] {
    const recommendationMap: Record<ClientSideAuthFinding['type'], string[]> = {
      localStorage_auth: [
        '將所有授權檢查移至服務端',
        '使用 HTTP-Only Cookies 存儲身份令牌',
        '實施 JWT 與服務端驗證',
        '避免在客戶端存儲角色信息'
      ],
      sessionStorage_auth: [
        '將授權邏輯移至服務端',
        '使用安全的 Session 管理',
        '實施請求級別的權限驗證'
      ],
      hardcoded_admin: [
        '移除硬編碼的角色檢查',
        '所有權限檢查必須在服務端進行',
        '使用動態權限配置系統'
      ],
      client_validation: [
        '在服務端重複所有驗證邏輯',
        '客戶端驗證僅用於用戶體驗改善',
        '實施服務端資料驗證和清理'
      ],
      hidden_elements: [
        '移除隱藏的敏感功能',
        '基於服務端權限動態渲染內容',
        '實施適當的訪問控制'
      ]
    };

    return recommendationMap[type] || ['實施適當的服務端授權檢查'];
  }

  /**
   * 分析客戶端認證繞過並返回標準 FindingPayload 格式
   * 遵循單一事實來源原則
   */
  async analyzeClientSideAuthBypassStandard(
    url: string, 
    taskId: string, 
    scanId: string, 
    browser?: any
  ): Promise<FindingPayload[]> {
    const legacyFindings = await this.analyzeClientSideAuthBypass(url, browser);
    return legacyFindings.map(finding => toStandardFinding(finding, taskId, scanId, url));
  }
}