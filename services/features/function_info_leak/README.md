# 敏感資訊洩漏檢測模組

**Version:** 2.0.0  
**Author:** AIVA Security Team  
**License:** MIT

## 📖 概述

這是一個專業級的敏感資訊檢測模組,用於識別 HTTP 響應、HTML 內容、JavaScript 代碼等中的敏感資訊洩漏問題。

### 🎯 核心功能

- ✅ **50+ 種檢測模式**: 涵蓋 AWS、GCP、Azure、GitHub、GitLab、Slack、Stripe 等主流平台
- ✅ **熵值分析**: 自動識別高隨機性密鑰字符串
- ✅ **智能誤報過濾**: 上下文感知、白名單過濾、變數名識別
- ✅ **信心度評分**: 0.0-1.0 浮點數評分,基於熵值和上下文
- ✅ **風險評分機制**: 0-100 總分,4級風險等級 (LOW/MEDIUM/HIGH/CRITICAL)
- ✅ **SARIF 格式支持**: 符合 SARIF v2.1.0 靜態分析標準
- ✅ **多格式輸出**: Text、JSON、SARIF 三種格式
- ✅ **批次掃描**: 支持大規模批次處理
- ✅ **統計分析**: 趨勢分析、Top 10 問題類型

## 🚀 快速開始

### 基本使用

```python
from sensitive_info_detector import SensitiveInfoDetector

# 創建檢測器
detector = SensitiveInfoDetector()

# 掃描 HTML 內容
html_result = detector.detect_in_html(html_content, url="https://example.com")

# 掃描 HTTP 響應
response_result = detector.detect_in_response(
    response_body, 
    headers=response_headers,
    url="https://api.example.com"
)

# 輸出報告
print(detector.format_report(html_result, format='text'))
```

### 進階配置

```python
from sensitive_info_detector import SensitiveInfoDetector, AlertSeverity

detector = SensitiveInfoDetector(
    min_severity=AlertSeverity.MEDIUM,     # 只顯示 MEDIUM 以上
    enable_entropy_check=True,              # 啟用熵值檢測
    entropy_threshold=4.8,                  # 熵值閾值 (越高越嚴格)
    min_confidence=0.5                      # 最小信心度 50%
)
```

### 導出報告

```python
# 導出 JSON 格式
detector.export_report(result, "report.json", format="json")

# 導出 SARIF 格式 (可導入 VS Code、GitHub 等工具)
detector.export_report(result, "report.sarif", format="sarif")

# 導出純文本格式
detector.export_report(result, "report.txt", format="text")
```

### 批次掃描

```python
from sensitive_info_detector import batch_scan

targets = [
    (html_content1, "https://example.com/page1"),
    (html_content2, "https://example.com/page2"),
    (json_response, "https://api.example.com/data")
]

results = batch_scan(targets, output_dir="scan_reports")

# 查看統計資訊
stats = detector.get_statistics(results)
print(f"Total matches: {stats['total_matches']}")
print(f"Average risk score: {stats['average_risk_score']}")
```

### 快速掃描

```python
from sensitive_info_detector import quick_scan

# 自動識別內容類型
result = quick_scan(content, content_type="html", url="https://example.com")
result = quick_scan(json_data, content_type="json", url="https://api.example.com")
```

## 📊 檢測類型

### 認證與密鑰 (20+ 種)

| 類型 | 範例 | 嚴重性 |
|------|------|--------|
| AWS Access Key | `AKIAIOSFODNN7EXAMPLE` | CRITICAL |
| AWS Secret Key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | CRITICAL |
| GitHub Token | `ghp_wWPw5k4aXcaT4fNP0UcnZwJG8bjPVM` | HIGH |
| GitLab Token | `glpat-xxxxxxxxxxxxxxxxxxxx` | HIGH |
| Stripe Secret Key | `sk_live_51H8xyzABCDEFGHIJKLMNOP` | CRITICAL |
| Slack Token | `xoxb-1234567890-1234567890-xxxxxx` | HIGH |
| JWT Token | `eyJhbGciOiJIUzI1NiIsInR5cCI6...` | HIGH |
| Private Key | `-----BEGIN RSA PRIVATE KEY-----` | CRITICAL |

### 資料庫連線字串 (5種)

- MongoDB: `mongodb://user:pass@host:port/db`
- PostgreSQL: `postgresql://user:pass@host:port/db`
- MySQL: `mysql://user:pass@host:port/db`
- Redis: `redis://user:pass@host:port`
- Generic: `DB_CONNECTION` patterns

### 個人識別資訊 (PII, 6種)

- Email addresses
- Credit card numbers
- Social Security Numbers (SSN)
- Phone numbers (國際格式)
- Passport numbers
- Driver's license numbers

### 第三方服務 (10+ 種)

- Twilio, SendGrid, Mailgun (郵件/簡訊服務)
- NPM, PyPI (套件管理)
- Heroku, Docker (部署平台)
- PayPal, Square (支付平台)

### 內部資訊

- Internal file paths
- Stack traces
- SQL queries
- Debug information

## 🔬 技術特性

### 熵值分析

模組使用香農熵 (Shannon Entropy) 來識別高隨機性字符串:

```python
entropy = calculate_entropy("sk_live_51H8xyzABCDEFGHIJKLMNOP")
# 結果: 4.97 (高熵值,可能是密鑰)

entropy = calculate_entropy("password123")
# 結果: 3.01 (低熵值,可能是範例)
```

**熵值範圍**: 0-8 (理論最大值)  
**建議閾值**: 4.5 (可自訂)

### 誤報過濾

智能識別以下情況並降低信心度或忽略:

- 白名單關鍵字: `example`, `sample`, `test`, `demo`, `placeholder`
- 變數名: `api_key`, `token`, `password` (單純變數名)
- 佔位符: `{API_KEY}`, `<TOKEN>`, `${SECRET}`
- 範例代碼: 包含 `example:`, `e.g.`, `範例:` 的註釋

### 信心度評分

```python
confidence = base_confidence * (entropy / 8.0 + 0.5)
```

- **1.0**: 高信心 (確定是敏感資訊)
- **0.7-0.9**: 中高信心 (很可能是敏感資訊)
- **0.5-0.7**: 中等信心 (需要人工確認)
- **0.3-0.5**: 低信心 (可能是誤報)
- **< 0.3**: 過濾掉 (預設閾值)

### 風險評分計算

```python
severity_weights = {
    CRITICAL: 25,
    HIGH: 10,
    MEDIUM: 5,
    LOW: 2,
    INFO: 1
}

total_score = sum(weight * confidence for each match)
# 最高分: 100
```

**風險等級判定**:
- `CRITICAL`: critical_count > 0 或 score >= 75
- `HIGH`: high_count > 0 或 score >= 50
- `MEDIUM`: medium_count > 0 或 score >= 25
- `LOW`: 其他情況

## 📄 輸出格式

### 1. Text Format

```
================================================================================
敏感資訊檢測報告 - AIVA Security Scanner v2.0
================================================================================
掃描時間: 2026-01-28T01:30:00Z
目標URL: https://api.example.com

風險評分:
  總分: 75.00/100
  風險等級: CRITICAL

問題統計:
  🔴 CRITICAL: 2
  🟠 HIGH:     3
  🟡 MEDIUM:   1
  🟢 LOW:      0
  ℹ️  INFO:     0

發現問題數量: 6
================================================================================

🔴 [1] CRITICAL - aws_access_key
  位置: response_body
  行號: 15
  列號: 22
  值: AKIAIOSFODNN7EXAMPLE
  上下文: "aws_key": "AKIAIOSFODNN7EXAMPLE", "region": "us-east-1"
  信心度: 100.00%
  熵值: 4.523
  說明: AWS Access Key ID detected
  建議: Revoke this key immediately and rotate credentials via IAM console
  Hash: a3f5c8d9e2b1...
```

### 2. JSON Format

```json
{
  "url": "https://api.example.com",
  "scan_time": "2026-01-28T01:30:00Z",
  "total_matches": 6,
  "risk_score": {
    "total": 75.00,
    "level": "CRITICAL",
    "critical": 2,
    "high": 3,
    "medium": 1,
    "low": 0,
    "info": 0
  },
  "matches": [
    {
      "type": "aws_access_key",
      "location": "response_body",
      "severity": "critical",
      "confidence": 1.0,
      "value": "AKIAIOSFODNN7EXAMPLE",
      "context": "\"aws_key\": \"AKIAIOSFODNN7EXAMPLE\", \"region\"...",
      "line": 15,
      "column": 22,
      "description": "AWS Access Key ID detected",
      "recommendation": "Revoke this key immediately...",
      "entropy": 4.523,
      "hash": "a3f5c8d9e2b1..."
    }
  ]
}
```

### 3. SARIF Format

符合 [SARIF v2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html) 標準,可導入:

- Visual Studio Code
- GitHub Advanced Security
- Azure DevOps
- SonarQube
- 其他 SAST 工具

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "AIVA Sensitive Info Detector",
        "version": "2.0.0",
        "informationUri": "https://github.com/kyle0527/AIVA",
        "rules": [...]
      }
    },
    "results": [...]
  }]
}
```

## 🔧 配置選項

### 初始化參數

```python
SensitiveInfoDetector(
    min_severity: AlertSeverity = AlertSeverity.INFO,
    enable_entropy_check: bool = True,
    entropy_threshold: float = 4.5,
    min_confidence: float = 0.3,
    custom_patterns: Optional[Dict] = None
)
```

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `min_severity` | AlertSeverity | INFO | 最小嚴重性閾值 |
| `enable_entropy_check` | bool | True | 啟用熵值檢測 |
| `entropy_threshold` | float | 4.5 | 熵值閾值 (0-8) |
| `min_confidence` | float | 0.3 | 最小信心度閾值 (0-1) |
| `custom_patterns` | Dict | None | 自訂檢測規則 |

### 自訂檢測規則

```python
custom_patterns = {
    SensitiveInfoType.API_KEY: {
        "pattern": re.compile(r'my-custom-key-pattern'),
        "severity": AlertSeverity.HIGH,
        "description": "Custom API key pattern",
        "recommendation": "Rotate the key"
    }
}

detector = SensitiveInfoDetector(custom_patterns=custom_patterns)
```

## 📈 統計分析

```python
# 批次掃描多個目標
results = batch_scan(targets, output_dir="reports")

# 生成統計資訊
stats = detector.get_statistics(results)

print(stats)
# {
#   "total_scans": 10,
#   "total_matches": 45,
#   "average_risk_score": 62.5,
#   "severity_distribution": {
#     "critical": 5,
#     "high": 12,
#     "medium": 15,
#     "low": 10,
#     "info": 3
#   },
#   "top_issues": {
#     "aws_access_key": 8,
#     "jwt_token": 6,
#     "api_key": 5,
#     ...
#   },
#   "scanned_urls": [...]
# }
```

## 🧪 測試

```bash
# 語法檢查
python -m py_compile sensitive_info_detector.py

# 運行內建測試
python sensitive_info_detector.py

# 單元測試 (如果有)
pytest tests/test_sensitive_info_detector.py
```

## 📚 相關資源

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [SARIF Specification](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
- [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request!

### 添加新的檢測模式

1. 在 `SensitiveInfoType` Enum 中添加新類型
2. 在 `_build_patterns()` 方法中添加正則表達式
3. 更新文檔

### 改進誤報過濾

1. 更新 `FALSE_POSITIVE_KEYWORDS` 集合
2. 優化 `_is_false_positive()` 方法
3. 調整信心度計算邏輯

## 📝 版本歷史

### v2.0.0 (2026-01-28)
- ✅ 完全重寫,從 11 種擴展到 50+ 種檢測模式
- ✅ 新增熵值分析系統
- ✅ 新增智能誤報過濾
- ✅ 新增信心度評分
- ✅ 新增風險評分機制
- ✅ 新增 SARIF 格式支持
- ✅ 新增批次掃描功能
- ✅ 新增統計分析功能
- ✅ 性能優化 (正則預編譯、去重機制)

### v1.0.0 (2026-01-18)
- 初始版本,基礎檢測功能

## 📧 聯絡方式

- GitHub: [kyle0527/AIVA](https://github.com/kyle0527/AIVA)
- Issues: [GitHub Issues](https://github.com/kyle0527/AIVA/issues)

## 📄 授權

MIT License - 詳見 LICENSE 文件

---

**Powered by AIVA Security Team** 🛡️
