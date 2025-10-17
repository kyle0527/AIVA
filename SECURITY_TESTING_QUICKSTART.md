# 🚀 安全測試框架快速入門

## 立即開始

### 1. 安裝依賴

```bash
# Python 依賴
pip install aiohttp

# Go 依賴 (SCA 分析器)
cd services/function/function_sca_go
go mod download
```

### 2. 配置測試目標

編輯 `security_test_config.json`:

```json
{
  "target_url": "http://localhost:3000",
  "test_users": [
    {
      "user_id": "123",
      "username": "alice",
      "role": "user",
      "token": "your_token_here"
    }
  ]
}
```

### 3. 運行測試

```bash
# 運行所有測試
python run_security_tests.py

# 只運行 IDOR 測試
python run_security_tests.py --only-idor

# 只運行認證測試
python run_security_tests.py --only-auth

# 只運行 CORS 測試
python run_security_tests.py --only-cors

# 詳細輸出
python run_security_tests.py --verbose
```

### 4. 查看報告

測試完成後,報告會保存在 `reports/` 目錄:

- `idor_test_report.json` - IDOR 測試報告
- `auth_test_report.json` - 認證測試報告
- `cors_test_report.json` - CORS 測試報告
- `comprehensive_security_report.json` - 綜合報告

## 實際案例

### 案例 1: 測試 OWASP Juice Shop

```bash
# 1. 啟動 Juice Shop
docker run -p 3000:3000 bkimminich/juice-shop

# 2. 配置測試
cat > juice_shop_config.json << EOF
{
  "target_url": "http://localhost:3000",
  "test_users": [
    {
      "user_id": "1",
      "username": "admin@juice-sh.op",
      "role": "admin",
      "token": "Bearer eyJhbGc..."
    },
    {
      "user_id": "2",
      "username": "jim@juice-sh.op",
      "role": "user",
      "token": "Bearer eyJhbGc..."
    }
  ],
  "horizontal_test_endpoints": [
    "/rest/user/whoami",
    "/api/Users/1",
    "/api/BasketItems"
  ],
  "vertical_test_endpoints": [
    "/rest/admin/application-version",
    "/api/Users"
  ]
}
EOF

# 3. 運行測試
python run_security_tests.py --config juice_shop_config.json
```

### 案例 2: 測試自己的 API

```python
# test_my_api.py
import asyncio
from services.function.function_idor.aiva_func_idor.privilege_escalation_tester import (
    PrivilegeEscalationTester,
    TestUser
)

async def main():
    # 定義測試用戶
    user1 = TestUser(
        user_id="user-001",
        username="alice",
        role="user",
        token="alice_token_here"
    )
    
    user2 = TestUser(
        user_id="user-002",
        username="bob",
        role="user",
        token="bob_token_here"
    )
    
    # 測試水平越權
    async with PrivilegeEscalationTester("https://myapi.com") as tester:
        finding = await tester.test_horizontal_escalation(
            attacker=user1,
            victim=user2,
            target_url="https://myapi.com/api/profile?user_id=user-002"
        )
        
        if finding.vulnerable:
            print(f"❌ 發現漏洞!")
            print(f"CVSS: {finding.cvss_score}")
            print(f"影響: {finding.impact}")
            print(f"修復建議: {finding.remediation}")
        else:
            print("✅ 安全")
        
        tester.generate_report("my_api_report.json")

asyncio.run(main())
```

### 案例 3: Go 依賴分析

```go
// analyze_deps.go
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
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    config := &analyzer.SCAConfig{
        SupportedLangs:  []string{"nodejs", "python", "go"},
        EnableDeepScan:  true,
        VulnSeverityMin: "HIGH",
        CacheResults:    true,
    }
    
    vulnDB := vulndb.NewOSVDatabase(logger)
    defer vulnDB.Close()
    
    scanner := analyzer.NewEnhancedSCAAnalyzer(logger, config, vulnDB)
    
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
    defer cancel()
    
    result, err := scanner.ScanProject(ctx, "./")
    if err != nil {
        log.Fatalf("Scan failed: %v", err)
    }
    
    log.Printf("Total dependencies: %d", result.Statistics.TotalDeps)
    log.Printf("Vulnerable dependencies: %d", result.Statistics.VulnerableDeps)
    log.Printf("Total vulnerabilities: %d", result.Statistics.TotalVulns)
    
    for severity, count := range result.Statistics.SeverityBreakdown {
        log.Printf("  %s: %d", severity, count)
    }
}
```

## 常見問題

### Q: 測試會影響生產環境嗎?

A: 建議僅在測試環境運行。如果必須在生產環境測試,請:
- 降低並發數
- 使用測試帳號
- 在非高峰時段進行
- 提前通知相關人員

### Q: 如何獲取測試用的 Token?

A: 有幾種方式:
1. 從瀏覽器開發者工具複製
2. 通過 API 登錄獲取
3. 使用測試帳號自動登錄

```python
# 自動登錄獲取 token
import aiohttp

async def get_token(url, username, password):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}/api/login",
            json={"username": username, "password": password}
        ) as resp:
            data = await resp.json()
            return data.get("token")
```

### Q: 如何解讀測試結果?

A: 查看報告中的關鍵指標:

- **CVSS Score**: 9.0-10.0 (Critical), 7.0-8.9 (High), 4.0-6.9 (Medium), 0.1-3.9 (Low)
- **Vulnerable**: true = 發現漏洞, false = 安全
- **Evidence**: 具體的證據和測試數據
- **Remediation**: 修復建議

### Q: 發現漏洞後該怎麼辦?

A: 標準流程:

1. **記錄**: 詳細記錄漏洞詳情
2. **評估**: 評估影響範圍和嚴重性
3. **通知**: 通知相關開發團隊
4. **修復**: 按照建議進行修復
5. **驗證**: 再次測試確認修復
6. **歸檔**: 記錄到漏洞管理系統

### Q: 可以整合到 CI/CD 嗎?

A: 可以! 示例 GitHub Actions:

```yaml
name: Security Tests

on: [push, pull_request]

jobs:
  security-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install aiohttp
      
      - name: Run security tests
        run: python run_security_tests.py --config ci_config.json
      
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: security-reports
          path: reports/
```

## 進階技巧

### 1. 自定義測試腳本

```python
# custom_test.py
import asyncio
from services.function.function_idor.aiva_func_idor.privilege_escalation_tester import (
    PrivilegeEscalationTester,
    TestUser,
    IDORFinding
)

class CustomTester(PrivilegeEscalationTester):
    async def test_custom_vulnerability(self, user: TestUser) -> IDORFinding:
        # 實現自定義測試邏輯
        response = await self._make_request(
            url=f"{self.target_url}/api/custom",
            method="POST",
            user=user
        )
        
        vulnerable = self._check_custom_condition(response)
        
        return IDORFinding(
            test_id="custom_001",
            # ...其他欄位
        )
```

### 2. 批量測試多個目標

```python
# batch_test.py
import asyncio
from run_security_tests import UnifiedSecurityTester

async def test_multiple_targets():
    targets = [
        "https://api1.example.com",
        "https://api2.example.com",
        "https://api3.example.com"
    ]
    
    for target in targets:
        config = load_config("config.json")
        config["target_url"] = target
        
        tester = UnifiedSecurityTester(config)
        await tester.run_all_tests()

asyncio.run(test_multiple_targets())
```

### 3. 監控與告警

```python
# monitor.py
import asyncio
import smtplib
from email.mime.text import MIMEText

async def monitor_and_alert():
    tester = UnifiedSecurityTester(config)
    results = await tester.run_all_tests()
    
    critical_findings = [
        f for f in tester._get_critical_findings()
        if f["cvss_score"] >= 9.0
    ]
    
    if critical_findings:
        send_alert_email(critical_findings)

def send_alert_email(findings):
    msg = MIMEText(f"發現 {len(findings)} 個嚴重漏洞!")
    msg['Subject'] = '🚨 安全告警'
    msg['From'] = 'security@example.com'
    msg['To'] = 'devops@example.com'
    
    # 發送郵件...
```

## 支援與資源

- 📖 完整文檔: [SECURITY_TESTING_FRAMEWORK_README.md](SECURITY_TESTING_FRAMEWORK_README.md)
- 🐛 問題回報: GitHub Issues
- 💬 討論: GitHub Discussions
- 📧 聯繫: security@example.com

## 貢獻

歡迎貢獻代碼、報告問題或提出建議！

---

**開始測試,保護您的應用! 🛡️**
