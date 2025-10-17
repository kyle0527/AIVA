# Security Testing Framework - Complete Implementation

## 🎯 Overview

本框架提供完整的安全測試功能，包括:

### ✅ 已實現功能

#### 1. **權限提升與越權測試** (Python)
位置: `services/function/function_idor/aiva_func_idor/privilege_escalation_tester.py`

- ✅ **水平越權** (Horizontal Privilege Escalation)
  - 同級別用戶間的資源訪問測試
  - 自動檢測資料洩露
  - 相似度分析
  
- ✅ **垂直越權** (Vertical Privilege Escalation)
  - 低權限用戶訪問高權限資源
  - 管理功能檢測
  - 權限提升路徑分析
  
- ✅ **資源枚舉** (Resource Enumeration)
  - 可預測 ID 掃描
  - 並發批量測試
  - 枚舉模式識別

#### 2. **認證安全測試** (Python)
位置: `services/function/function_authn_go/internal/auth_cors_tester/auth_cors_tester.py`

- ✅ 弱密碼策略測試
- ✅ 暴力破解防護測試
- ✅ JWT 安全性測試
- ✅ Session Fixation 測試
- ✅ Token 驗證測試

#### 3. **CORS 安全測試** (Python)
位置: `services/function/function_authn_go/internal/auth_cors_tester/auth_cors_tester.py`

- ✅ Null Origin 測試
- ✅ Wildcard + Credentials 測試
- ✅ Reflected Origin 測試
- ✅ Subdomain 繞過測試
- ✅ Credentials 洩露測試

#### 4. **改進的依賴分析** (Go)
位置: `services/function/function_sca_go/internal/analyzer/`

- ✅ `dependency_analyzer.go` - 多語言依賴解析
  - Node.js (package.json, package-lock.json)
  - Python (requirements.txt, Pipfile, pyproject.toml)
  - Go (go.mod, go.sum)
  - Rust (Cargo.toml, Cargo.lock)
  - PHP (composer.json)
  - Ruby (Gemfile)
  - Java/Maven (pom.xml)
  - .NET (.csproj)
  
- ✅ `enhanced_analyzer.go` - 增強型分析器
  - 並發漏洞掃描 (worker pool)
  - 漏洞快取機制
  - 嚴重性過濾
  - 深度掃描支持
  - 統計報告生成

- ✅ `vulndb/osv.go` - OSV 漏洞資料庫整合
  - OSV API 整合
  - CVSS 評分解析
  - 嚴重性自動判斷

## 📋 使用方法

### 1. 權限提升測試

```python
import asyncio
from services.function.function_idor.aiva_func_idor.privilege_escalation_tester import (
    PrivilegeEscalationTester,
    TestUser
)

async def test_privilege_escalation():
    # 創建測試用戶
    attacker = TestUser(
        user_id="123",
        username="alice",
        role="user",
        token="attacker_token"
    )
    
    victim = TestUser(
        user_id="456",
        username="bob",
        role="user",
        token="victim_token"
    )
    
    admin = TestUser(
        user_id="789",
        username="admin",
        role="admin",
        token="admin_token"
    )
    
    # 開始測試
    async with PrivilegeEscalationTester("https://target.com") as tester:
        # 測試水平越權
        h_finding = await tester.test_horizontal_escalation(
            attacker=attacker,
            victim=victim,
            target_url="https://target.com/api/user/profile?user_id=456"
        )
        print(f"水平越權: {'發現漏洞' if h_finding.vulnerable else '安全'}")
        
        # 測試垂直越權
        v_finding = await tester.test_vertical_escalation(
            low_priv_user=attacker,
            high_priv_user=admin,
            admin_url="https://target.com/admin/dashboard"
        )
        print(f"垂直越權: {'發現漏洞' if v_finding.vulnerable else '安全'}")
        
        # 測試資源枚舉
        enum_finding = await tester.test_resource_enumeration(
            user=attacker,
            base_url="https://target.com/api/user/profile",
            id_param="user_id",
            id_range=(1, 100)
        )
        print(f"資源枚舉: {enum_finding.evidence['accessible_count']} 個可訪問")
        
        # 生成報告
        tester.generate_report("idor_test_report.json")

asyncio.run(test_privilege_escalation())
```

### 2. 認證與 CORS 測試

```python
import asyncio
from services.function.function_authn_go.internal.auth_cors_tester.auth_cors_tester import (
    AuthenticationTester,
    CORSTester
)

async def test_auth_and_cors():
    target = "https://target.com"
    
    # 認證測試
    async with AuthenticationTester(target) as auth_tester:
        # 弱密碼測試
        await auth_tester.test_weak_password_policy(
            register_url=f"{target}/api/register"
        )
        
        # 暴力破解防護測試
        await auth_tester.test_brute_force_protection(
            login_url=f"{target}/api/login",
            username="test_user",
            max_attempts=20
        )
        
        # Session Fixation 測試
        await auth_tester.test_session_fixation(
            login_url=f"{target}/api/login",
            username="test_user",
            password="correct_password"
        )
        
        # JWT 安全測試
        await auth_tester.test_jwt_security(
            token="eyJhbGc...",
            api_url=f"{target}/api/protected"
        )
        
        auth_tester.generate_report("auth_test_report.json")
    
    # CORS 測試
    async with CORSTester(target) as cors_tester:
        # Null Origin 測試
        await cors_tester.test_null_origin(f"{target}/api/data")
        
        # Wildcard + Credentials 測試
        await cors_tester.test_wildcard_with_credentials(f"{target}/api/data")
        
        # Reflected Origin 測試
        await cors_tester.test_reflected_origin(
            f"{target}/api/data",
            test_origins=[
                "https://evil.com",
                "https://attacker.com",
                "http://localhost:8000"
            ]
        )
        
        cors_tester.generate_report("cors_test_report.json")

asyncio.run(test_auth_and_cors())
```

### 3. Go 依賴分析

```go
package main

import (
    "context"
    "log"
    "time"
    
    "go.uber.org/zap"
    "github.com/kyle0527/aiva/services/function/function_sca_go/internal/analyzer"
    "github.com/kyle0527/aiva/services/function/function_sca_go/internal/vulndb"
)

func main() {
    // 創建 logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    // 配置
    config := &analyzer.SCAConfig{
        SupportedLangs:  []string{"nodejs", "python", "go", "rust"},
        EnableDeepScan:  true,
        VulnSeverityMin: "MEDIUM",
        CacheResults:    true,
    }
    
    // 創建漏洞資料庫
    vulnDB := vulndb.NewOSVDatabase(logger)
    defer vulnDB.Close()
    
    // 創建增強型分析器
    scanner := analyzer.NewEnhancedSCAAnalyzer(logger, config, vulnDB)
    
    // 掃描專案
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
    defer cancel()
    
    result, err := scanner.ScanProject(ctx, "/path/to/project")
    if err != nil {
        log.Fatalf("Scan failed: %v", err)
    }
    
    // 輸出結果
    log.Printf("Total dependencies: %d", result.Statistics.TotalDeps)
    log.Printf("Vulnerable dependencies: %d", result.Statistics.VulnerableDeps)
    log.Printf("Total vulnerabilities: %d", result.Statistics.TotalVulns)
    
    // 導出報告
    if err := scanner.ExportJSON(result, "sca_report.json"); err != nil {
        log.Fatalf("Export failed: %v", err)
    }
}
```

## 🔧 配置說明

### 依賴分析器配置

```go
type SCAConfig struct {
    SupportedLangs   []string // 支援的語言: nodejs, python, go, rust, java, dotnet, php, ruby
    EnableDeepScan   bool     // 啟用深度掃描 (解析 lock 文件)
    VulnSeverityMin  string   // 最小漏洞嚴重性: LOW, MEDIUM, HIGH, CRITICAL
    CacheResults     bool     // 快取查詢結果
    SkipDirs         []string // 跳過的目錄
}
```

### 測試配置

```python
# 水平越權測試
await tester.test_horizontal_escalation(
    attacker=TestUser(...),
    victim=TestUser(...),
    target_url="...",
    method="GET"  # 支援 GET, POST, PUT, DELETE
)

# 資源枚舉測試
await tester.test_resource_enumeration(
    user=TestUser(...),
    base_url="...",
    id_param="user_id",  # 要測試的參數名
    id_range=(1, 1000),  # ID 範圍
    method="GET"
)
```

## 📊 報告格式

所有測試都會生成統一的 JSON 格式報告:

```json
{
  "summary": {
    "total_tests": 10,
    "vulnerable_tests": 3,
    "by_severity": {
      "CRITICAL": 1,
      "HIGH": 2,
      "MEDIUM": 3,
      "LOW": 2,
      "INFO": 2
    },
    "by_type": {
      "horizontal": 1,
      "vertical": 1,
      "enumeration": 1
    }
  },
  "findings": [
    {
      "test_id": "h_esc_123_456",
      "escalation_type": "horizontal",
      "severity": "HIGH",
      "vulnerable": true,
      "url": "https://target.com/api/user/profile",
      "description": "水平越權測試...",
      "evidence": {
        "attacker_status": 200,
        "leaked_fields": ["email", "phone"]
      },
      "impact": "攻擊者能夠訪問其他用戶資料...",
      "remediation": "1. 實施嚴格的身份驗證\n2. 使用 UUID...",
      "cvss_score": 7.5
    }
  ]
}
```

## 🎖️ 改進重點

### Go 依賴分析器改進 (根據您的要求)

1. ✅ **錯誤處理與日誌**
   - 在 `AnalyzeProject` 返回前記錄錯誤
   - 累積跳過的文件列表
   - 詳細的進度日誌

2. ✅ **程式結構**
   - 修正 `.csproj` 檔案判斷邏輯
   - 抽取重複的解析邏輯

3. ✅ **設計合理性**
   - `SkipDirs` 可配置
   - 未實現語言添加警告日誌
   - 支援策略模式擴展

4. ✅ **命名與輸出**
   - 漏洞資訊整合回 `Dependencies`
   - 統一使用 `analyzer.Vulnerability`
   - 並發安全的資料更新

5. ✅ **擴展性**
   - `SupportedLangs` 過濾功能
   - `EnableDeepScan` 實現
   - `VulnSeverityMin` 過濾
   - `CacheResults` 快取機制

### Enhanced Analyzer 改進

1. ✅ **錯誤處理**
   - Context 超時檢查與錯誤返回
   - 失敗計數統計
   - 部分結果標記

2. ✅ **程式流程**
   - 清晰的三階段掃描
   - Worker pool 並發控制
   - 結果回寫到原始列表

3. ✅ **漏洞快取**
   - 線程安全的快取實現
   - 基於 (Language, Name, Version) 的快取鍵

4. ✅ **統計與報告**
   - 完整的統計資訊
   - 按嚴重性分組
   - 按語言分組

## 🚀 進階功能

### 自定義測試案例

```python
from services.function.function_idor.aiva_func_idor.privilege_escalation_tester import (
    IDORTestCase,
    EscalationType,
    ResourceType
)

# 創建自定義測試案例
test_case = IDORTestCase(
    test_id="custom_test_001",
    escalation_type=EscalationType.HORIZONTAL,
    resource_type=ResourceType.USER_DATA,
    url="https://target.com/api/user/orders",
    method="GET",
    params={"user_id": "123"},
    attacker=attacker_user,
    victim=victim_user,
    description="測試用戶訂單越權訪問"
)

# 執行測試
finding = await tester.execute_test_case(test_case)
```

### 批量測試

```python
# 批量測試多個端點
endpoints = [
    "https://target.com/api/user/profile",
    "https://target.com/api/user/orders",
    "https://target.com/api/user/payments",
    "https://target.com/api/user/settings"
]

for endpoint in endpoints:
    await tester.test_horizontal_escalation(
        attacker=attacker,
        victim=victim,
        target_url=endpoint
    )
```

## 📝 注意事項

1. **合法授權**: 僅在獲得明確授權的系統上使用
2. **速率限制**: 注意測試頻率，避免觸發 DDoS 防護
3. **數據保護**: 測試數據可能包含敏感資訊，妥善保存報告
4. **環境隔離**: 建議在測試環境進行，避免影響生產系統

## 🔗 相關文件

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CWE-639: Authorization Bypass](https://cwe.mitre.org/data/definitions/639.html)
- [CWE-863: Incorrect Authorization](https://cwe.mitre.org/data/definitions/863.html)

## 📧 支援

如有問題或建議，請參考項目文檔或聯繫開發團隊。

---

**版本**: 1.0.0  
**最後更新**: 2025-01-17  
**作者**: AIVA Security Team
