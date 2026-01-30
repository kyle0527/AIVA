# function_authn_go - 認證繞過與漏洞檢測

> **版本**: v1.3.0 | **狀態**: ✅ 完成 | **語言**: Go + Python Wrapper | **更新**: 2026-01-20

## 🎯 模組概述

Go語言實現的高性能認證安全測試引擎，專注於Web應用認證機制的漏洞檢測。

### 核心功能

✅ **弱密碼檢測** - 測試常見弱密碼組合  
✅ **默認憑證測試** - 檢測未修改的默認帳號密碼  
✅ **密碼噴灑攻擊** - 單密碼多帳號測試（避免帳號鎖定）  
✅ **JWT 令牌分析** - JWT 簽名算法漏洞、密鑰爆破  
✅ **Session 劫持測試** - Session 固定、會話令牌可預測性  
✅ **OAuth/SSO 漏洞** - OAuth 2.0 流程繞過、CSRF攻擊  

## 📐 架構設計

```
function_authn_go/
├── cmd/
│   └── worker/
│       └── main.go          # Go 主程序入口
├── internal/
│   ├── detector/
│   │   ├── weak_password.go
│   │   ├── jwt_analyzer.go
│   │   └── session_hijack.go
│   └── wordlist/
│       └── passwords.go
├── bin/
│   └── authn-worker.exe     # 編譯後的二進制文件
├── authn_wrapper.py         # Python 包裝器（CLI 調用入口）
├── authn_manager.py         # 高級管理接口
└── README.md
```

## 🚀 快速開始

### 1. 編譯 Go 引擎

```bash
cd services/features/function_authn_go

# Windows
go build -o bin/authn-worker.exe cmd/worker/main.go

# Linux/Mac
go build -o bin/authn-worker cmd/worker/main.go
```

### 2. Python 調用

```python
from services.features.function_authn_go.authn_wrapper import scan_authentication

# 弱密碼測試
result = scan_authentication(
    target="https://example.com/login",
    options={
        "username": "admin",
        "test_types": ["weak_password", "default_credentials"]
    }
)

print(f"發現 {len(result['vulnerabilities'])} 個漏洞")
```

### 3. CLI 直接調用

```bash
# external_executor 調用示例
python aiva_external_executor.py --lang go --module authn_go --target https://example.com/login
```

## 🔧 功能詳解

### 弱密碼檢測

測試常見弱密碼組合：
- `admin/admin123`
- `root/toor`
- `test/test123`
- Top 1000 常見密碼

```python
result = scan_authentication(
    target="https://example.com/login",
    options={
        "test_types": ["weak_password"],
        "wordlist": "rockyou.txt"  # 可選自定義字典
    }
)
```

### JWT 令牌分析

檢測 JWT 安全問題：
- **算法混淆攻擊** (alg=none, RS256→HS256)
- **弱密鑰爆破** (常見密鑰字典攻擊)
- **過期時間驗證繞過**
- **敏感信息洩露** (JWT payload 分析)

```python
result = scan_authentication(
    target="https://api.example.com",
    options={
        "test_types": ["jwt_analysis"],
        "jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
)
```

### Session 劫持測試

- **Session Fixation** - 會話固定攻擊
- **Predictable Session ID** - 可預測的會話令牌
- **Session Token in URL** - URL中的敏感令牌
- **Missing HTTPOnly/Secure Flags** - Cookie 安全標誌缺失

```python
result = scan_authentication(
    target="https://example.com",
    options={
        "test_types": ["session_hijack"],
        "cookies": {"PHPSESSID": "abc123def456"}
    }
)
```

## 📊 輸出格式

```json
{
  "success": true,
  "target": "https://example.com/login",
  "test_types": ["weak_password", "jwt_analysis"],
  "vulnerabilities": [
    {
      "type": "weak_password",
      "severity": "high",
      "username": "admin",
      "password": "admin123",
      "evidence": "Successfully logged in with default credentials",
      "recommendation": "Enforce strong password policy and change default credentials"
    },
    {
      "type": "jwt_algorithm_confusion",
      "severity": "critical",
      "detail": "JWT accepts 'none' algorithm",
      "recommendation": "Reject JWT tokens with 'none' algorithm"
    }
  ],
  "summary": {
    "total_tests": 15,
    "vulnerabilities_found": 2,
    "execution_time": "2.3s"
  }
}
```

## 🔒 安全模式

所有測試都在**安全模式**下運行：
- ✅ 僅檢測漏洞，不執行破壞性操作
- ✅ 速率限制（避免觸發 WAF/IDS）
- ✅ 自動停止（檢測到帳號鎖定時）

```python
# 安全模式（默認啟用）
result = scan_authentication(
    target="https://example.com",
    options={
        "safe_mode": True,  # 默認值
        "rate_limit": 10    # 每秒最多10個請求
    }
)
```

## 🎯 適用場景

✅ **Bug Bounty 計劃** - 符合各大平台測試範圍  
✅ **滲透測試** - 認證機制安全評估  
✅ **DevSecOps** - CI/CD 流程中的自動化安全測試  
✅ **Red Team** - 初始訪問階段的憑證獲取  

## 📈 性能特點

- ⚡ **高性能**: Go 併發處理，比 Python 快 10x
- 🎯 **智能速率控制**: 自動適應目標響應速度
- 🛡️ **防護繞過**: 內建 User-Agent 輪換、Proxy 支持
- 📊 **詳細報告**: 每個測試的完整證據和建議

## 🔗 相關資源

- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [JWT Security Best Practices](https://datatracker.ietf.org/doc/html/rfc8725)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)

## 📝 更新日誌

### v1.3.0 (2026-01-20)
- ✅ 移除不必要的 worker.py 包裝層
- ✅ 直接使用 Go 引擎，無 Python 回退
- ✅ 完善 JWT 令牌分析功能
- ✅ 添加 OAuth/SSO 漏洞檢測

### v1.2.0 (2025-12-17)
- ✅ 創建 BUILD_GUIDE.md
- ✅ authn_wrapper.py 完成
- ✅ 架構符合 SIMPLE_ARCHITECTURE.md

---

**維護者**: AIVA Team | **授權**: MIT License
