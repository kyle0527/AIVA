# function_info_leak - 敏感資訊洩漏檢測模組

> **版本**: v3.0.0 | **狀態**: ✅ 模組完成 | **語言**: Python

## 🎯 模組概述

專業級敏感資訊檢測模組，負責識別 HTTP 回應、HTML 內容、標頭等之中的敏感資訊洩漏（API Key、Token、私鑰等），涵蓋 50+ 種偵測模式，並支援香農熵 (Shannon Entropy) 分析。

### 功能清單

| 功能 | 說明 |
|------|------|
| 50+ 種偵測模式 | AWS/GCP/Azure/GitHub/Stripe 等主流平台之金鑰與 Token 辨識 |
| 熵值分析 | 利用香農熵識別未符合特定模式的高隨機性密鑰 |
| 智能誤報過濾 | 上下文感知、白名單過濾、變數名識別 |
| 多格式輸出 | 支援匯出為 Text / JSON / SARIF 格式報告 |

## 📐 架構設計

```
function_info_leak/
├── __init__.py                 # 模組入口匯出
└── sensitive_info_detector.py  # 全部實作（SensitiveInfoDetector）
```

> 此模組為單一檔案實作，所有邏輯集中於 `sensitive_info_detector.py`，不依賴外部執行檔。

## 🚀 執行方式

### 作為 Python 模組匯入

```python
from services.features.function_info_leak import SensitiveInfoDetector

detector = SensitiveInfoDetector()

# 掃描完整 HTTP 回應內容
result = detector.detect_in_response(response_body="...", headers={}, url="https://example.com")

# 輸出 JSON 報告
print(detector.format_report(result, format="json"))
```

## 🔧 內部 API 參考

| 類別 / 方法 | 說明 |
|------|------|
| `SensitiveInfoDetector.detect_in_response(...)` | 掃描完整 HTTP 回應（主要入口） |
| `SensitiveInfoDetector.detect_in_html(...)` | 掃描 HTML 內容 |
| `SensitiveInfoDetector.detect_in_headers(...)` | 掃描 HTTP 標頭 |
| `SensitiveInfoDetector.format_report(result, format)` | 格式化輸出（text / json / sarif） |
| `SensitiveInfoDetector.export_report(...)` | 輸出報告至檔案 |

## 🔒 偵測類型與風險等級

- 🔴 **CRITICAL**: 包含 AWS Access/Secret Key, RSA/EC 私鑰, Stripe Secret Key 等。
- 🟠 **HIGH**: 包含 GitHub/GitLab Token, JWT Token, 資料庫連線字串等。
- 🟡 **MEDIUM**: 包含 PII（Email/SSN/電話）及 高熵字串 (熵值 > 4.5)。
- 🟢 **LOW**: 低風險資訊。

## 注意事項
- 偵測結果物件中敏感值通常僅保留部分或遮罩，避免在日誌中二次洩漏。
- 若出現 `example` / `test` 等字眼，引擎會自動降低信心度。
