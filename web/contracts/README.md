# AIVA Web 合約使用指南

## 📋 概覽

本指南說明如何在 AIVA Web 前端使用標準化資料合約，確保前後端資料一致性。

## 🎯 合約文件位置

- **TypeScript 合約定義**: `web/contracts/aiva-web-contracts.ts`
- **後端合約來源**: `services/aiva_common/schemas/`
- **生成的類型定義**: `cli_generated/schemas.ts`

## 🔄 使用方式

### 1. 在 TypeScript 項目中使用

```typescript
import { 
  APIResponse, 
  Finding, 
  ScanRequest, 
  UserInfo 
} from './contracts/aiva-web-contracts';

// API 響應處理
async function handleApiResponse<T>(response: Response): Promise<APIResponse<T>> {
  const data: APIResponse<T> = await response.json();
  
  if (!data.success) {
    throw new Error(data.message);
  }
  
  return data;
}

// 掃描請求
function createScanRequest(targetUrl: string): ScanRequest {
  return {
    target_url: targetUrl,
    scan_type: 'full',
    scan_scope: {
      include_subdomains: true,
      max_depth: 3,
      exclude_paths: ['/admin', '/private'],
      include_paths: []
    }
  };
}
```

### 2. 在 JavaScript 中使用

```javascript
/**
 * @typedef {import('./contracts/aiva-web-contracts').APIResponse} APIResponse
 * @typedef {import('./contracts/aiva-web-contracts').Finding} Finding
 */

class AIVAClient {
  async makeRequest(endpoint, data = null) {
    const response = await fetch(`${this.apiBase}${endpoint}`, {
      method: data ? 'POST' : 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.token}`
      },
      body: data ? JSON.stringify(data) : null
    });
    
    /** @type {APIResponse} */
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.message);
    }
    
    return result;
  }
}
```

## 📊 主要合約類別

### 🔐 認證相關

```typescript
// 登入請求
const loginRequest: LoginRequest = {
  username: 'admin',
  password: 'password'
};

// 令牌響應
const tokenResponse: TokenResponse = {
  access_token: 'jwt_token_here',
  token_type: 'bearer',
  expires_in: 86400,
  user: userInfo,
  issued_at: '2025-11-01T00:00:00Z'
};
```

### 🔍 掃描相關

```typescript
// 掃描請求
const scanRequest: ScanRequest = {
  target_url: 'https://example.com',
  scan_type: 'full',
  scan_scope: {
    include_subdomains: true,
    max_depth: 5,
    exclude_paths: ['/admin'],
    include_paths: []
  },
  authentication: {
    type: 'bearer',
    credentials: { token: 'api_token' }
  }
};

// 掃描狀態
const scanStatus: ScanStatus = {
  scan_id: 'scan_123',
  status: 'running',
  progress: 45,
  current_phase: 'vulnerability_scanning',
  started_at: '2025-11-01T10:00:00Z',
  findings_count: 12,
  assets_discovered: 25
};
```

### 🛡️ 漏洞發現

```typescript
// 漏洞發現
const finding: Finding = {
  finding_id: 'finding_456',
  vulnerability: {
    name: 'Cross-Site Scripting (XSS)',
    description: 'Reflected XSS vulnerability',
    severity: 'high',
    confidence: 'certain',
    vulnerability_type: 'XSS',
    cwe_id: 'CWE-79'
  },
  target: {
    url: 'https://example.com/search',
    method: 'GET',
    parameters: { q: 'search_term' },
    headers: { 'User-Agent': 'AIVA Scanner' }
  },
  evidence: {
    request: 'GET /search?q=<script>alert(1)</script>',
    response: 'HTTP/1.1 200 OK...',
    proof_of_concept: 'Payload executed successfully'
  },
  impact: {
    description: 'Allows execution of arbitrary JavaScript',
    affected_users: 'All users of the search function',
    business_impact: 'Medium - potential data theft',
    technical_impact: 'Session hijacking, defacement',
    exploitability: 'Easy to exploit'
  },
  recommendation: {
    short_term: ['Input validation', 'Output encoding'],
    long_term: ['CSP implementation', 'Security training'],
    references: ['https://owasp.org/www-community/attacks/xss/']
  },
  created_at: '2025-11-01T10:30:00Z',
  updated_at: '2025-11-01T10:30:00Z'
};
```

## 🎨 UI 組件整合

### React 組件範例

```typescript
import React from 'react';
import { Finding, Severity } from '../contracts/aiva-web-contracts';

interface FindingCardProps {
  finding: Finding;
  onViewDetails: (findingId: string) => void;
}

const FindingCard: React.FC<FindingCardProps> = ({ finding, onViewDetails }) => {
  const getSeverityColor = (severity: Severity): string => {
    const colors = {
      'critical': 'text-red-600',
      'high': 'text-orange-600', 
      'medium': 'text-yellow-600',
      'low': 'text-blue-600',
      'info': 'text-gray-600'
    };
    return colors[severity] || 'text-gray-600';
  };

  return (
    <div className="border rounded-lg p-4 shadow-sm">
      <div className="flex justify-between items-start">
        <h3 className="font-semibold">{finding.vulnerability.name}</h3>
        <span className={`px-2 py-1 rounded text-sm ${getSeverityColor(finding.vulnerability.severity)}`}>
          {finding.vulnerability.severity.toUpperCase()}
        </span>
      </div>
      <p className="text-gray-600 mt-2">{finding.vulnerability.description}</p>
      <div className="mt-4 flex justify-between items-center">
        <span className="text-sm text-gray-500">
          {finding.target.method} {finding.target.url}
        </span>
        <button 
          onClick={() => onViewDetails(finding.finding_id)}
          className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
        >
          查看詳情
        </button>
      </div>
    </div>
  );
};
```

### Vue.js 組件範例

```typescript
<template>
  <div class="dashboard-stats">
    <div class="stat-card" v-for="stat in stats" :key="stat.label">
      <h3>{{ stat.value }}</h3>
      <p>{{ stat.label }}</p>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted } from 'vue';
import { DashboardStats, APIResponse } from '../contracts/aiva-web-contracts';

export default defineComponent({
  name: 'DashboardOverview',
  setup() {
    const dashboardStats = ref<DashboardStats | null>(null);
    
    const stats = computed(() => {
      if (!dashboardStats.value) return [];
      
      return [
        { label: '總掃描次數', value: dashboardStats.value.total_scans },
        { label: '活躍掃描', value: dashboardStats.value.active_scans },
        { label: '發現漏洞', value: dashboardStats.value.total_findings },
        { label: '嚴重漏洞', value: dashboardStats.value.critical_findings }
      ];
    });
    
    const loadStats = async () => {
      const response: APIResponse<DashboardStats> = await apiClient.get('/dashboard/stats');
      dashboardStats.value = response.data;
    };
    
    onMounted(loadStats);
    
    return { stats };
  }
});
</script>
```

## 📡 API 整合模式

### 統一 API 客戶端

```typescript
import { APIResponse } from './contracts/aiva-web-contracts';

class AIVAApiClient {
  private baseUrl: string;
  private token: string | null = null;
  
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }
  
  setToken(token: string) {
    this.token = token;
  }
  
  private async makeRequest<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(this.token && { 'Authorization': `Bearer ${this.token}` }),
      ...options.headers
    };
    
    const response = await fetch(url, {
      ...options,
      headers
    });
    
    const result: APIResponse<T> = await response.json();
    
    if (!result.success) {
      throw new APIError(result.message, response.status);
    }
    
    return result;
  }
  
  async get<T>(endpoint: string): Promise<APIResponse<T>> {
    return this.makeRequest<T>(endpoint);
  }
  
  async post<T>(endpoint: string, data: any): Promise<APIResponse<T>> {
    return this.makeRequest<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }
}

class APIError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = 'APIError';
  }
}
```

## 🔧 開發工具

### 類型檢查

```bash
# TypeScript 編譯檢查
npx tsc --noEmit

# ESLint 檢查
npx eslint src/**/*.ts

# Prettier 格式化
npx prettier --write src/**/*.ts
```

### 測試範例

```typescript
import { describe, it, expect } from 'vitest';
import { Finding, Severity } from '../contracts/aiva-web-contracts';

describe('Finding Contract', () => {
  it('should create valid finding object', () => {
    const finding: Finding = {
      finding_id: 'test_123',
      vulnerability: {
        name: 'Test Vulnerability',
        description: 'Test description',
        severity: 'high' as Severity,
        confidence: 'certain',
        vulnerability_type: 'XSS'
      },
      // ... 其他必要屬性
    };
    
    expect(finding.finding_id).toBe('test_123');
    expect(finding.vulnerability.severity).toBe('high');
  });
});
```

## 🎯 最佳實踐

### 1. 類型安全
- 始終使用 TypeScript 合約定義
- 啟用嚴格模式類型檢查
- 使用泛型確保 API 響應類型安全

### 2. 錯誤處理
```typescript
try {
  const response = await apiClient.get<Finding[]>('/findings');
  setFindings(response.data || []);
} catch (error) {
  if (error instanceof APIError) {
    setError(`API 錯誤: ${error.message}`);
  } else {
    setError('未知錯誤occurred');
  }
}
```

### 3. 數據驗證
```typescript
import { z } from 'zod';

const FindingSchema = z.object({
  finding_id: z.string(),
  vulnerability: z.object({
    name: z.string(),
    severity: z.enum(['critical', 'high', 'medium', 'low', 'info'])
  })
});

function validateFinding(data: unknown): Finding {
  return FindingSchema.parse(data) as Finding;
}
```

## 🔄 同步更新

當後端合約更新時：

1. **自動生成** (推薦)
   ```bash
   # 使用 aiva-contracts-tooling 生成最新合約
   aiva-contracts gen-ts --json schemas/aiva_schemas.json --out web/contracts/generated.d.ts
   ```

2. **手動同步**
   - 檢查 `services/aiva_common/schemas/` 更新
   - 更新 `web/contracts/aiva-web-contracts.ts`
   - 運行測試確保相容性

## 📚 延伸閱讀

- **[AIVA 合約開發指南](../../guides/AIVA_合約開發指南.md)** - 完整的合約系統開發指導
- **[合約架構設計](../../guides/AIVA_合約開發指南.md#合約架構設計)** - 深入了解合約分層結構
- **[性能優化策略](../../guides/AIVA_合約開發指南.md#合約優化策略)** - 提升合約使用效率

---

**維護**: Web 開發團隊  
**更新頻率**: 隨合約變更  
**版本**: 1.1.0