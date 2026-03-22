# function_info_leak - 敏感資訊洩漏檢測模組

> **版本**: v2.0.0 | **狀態**: ✅ 完成 | **語言**: Python | **能力登錄**: ⬜ 待登錄（對應 `secret_detection`）

## 模組概述

專業級敏感資訊檢測模組，識別 HTTP 回應、HTML 內容、標頭等中的敏感資訊洩漏，涵蓋 50+ 種偵測類型。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 50+ 種偵測模式 | ✅ 完成 | AWS/GCP/Azure/GitHub/Stripe 等主流平台 |
| 熵值分析 | ✅ 完成 | 香農熵識別高隨機性密鑰 |
| 智能誤報過濾 | ✅ 完成 | 上下文感知、白名單、變數名識別 |
| 信心度評分 | ✅ 完成 | 0.0–1.0 浮點數評分 |
| 風險評分機制 | ✅ 完成 | 0–100 總分，4 級風險等級 |
| SARIF 格式輸出 | ✅ 完成 | 符合 SARIF v2.1.0 標準 |
| 多格式輸出 | ✅ 完成 | Text / JSON / SARIF |
| HTTP 標頭掃描 | ✅ 完成 | 獨立標頭分析方法 |

## 架構

```
function_info_leak/
└── sensitive_info_detector.py  # 全部實作（SensitiveInfoDetector）
```

> 此模組為單一檔案實作，所有邏輯集中於 `sensitive_info_detector.py`。

## 執行方式

### 透過 AIVA 執行器（推薦）

```bash
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func SensitiveInfoDetector.detect_in_response \
    --target https://example.com
```

### 直接使用

```python
from services.features.function_info_leak.sensitive_info_detector import SensitiveInfoDetector

detector = SensitiveInfoDetector()

# 掃描完整 HTTP 回應（主要入口）
result = detector.detect_in_response(response_body, headers=headers, url="https://example.com")

# 掃描 HTML 內容
result = detector.detect_in_html(html_content, url="https://example.com")

# 掃描 HTTP 標頭
result = detector.detect_in_headers(headers, url="https://example.com")

# 輸出報告
print(detector.format_report(result, format="text"))
detector.export_report(result, "report.sarif", format="sarif")
```

## 偵測類型（50+）

| 類型 | 嚴重度 | 範例 |
|------|:------:|------|
| AWS Access Key | 🔴 CRITICAL | `AKIAIOSFODNN7EXAMPLE` |
| AWS Secret Key | 🔴 CRITICAL | `wJalrXUtnFEMI/...` |
| RSA / EC 私鑰 | 🔴 CRITICAL | `-----BEGIN RSA PRIVATE KEY-----` |
| Stripe Secret Key | 🔴 CRITICAL | `sk_live_51H8xyz...` |
| GitHub Token | 🟠 HIGH | `ghp_wWPw5k4aXcaT4fN...` |
| GitLab Token | 🟠 HIGH | `glpat-xxxxxxxxxxxx` |
| JWT Token | 🟠 HIGH | `eyJhbGciOiJIUzI1NiIs...` |
| 資料庫連線字串 | 🟠 HIGH | `mongodb://user:pass@host` |
| PII（Email/SSN/電話） | 🟡 MEDIUM | 個人識別資訊 |
| 高熵字串 | 🟡 MEDIUM | 熵值 > 4.5 的任意字串 |

## 可調用方法（公開 API）

| 方法 | 說明 |
|------|------|
| `detect_in_response(response_body, headers, url)` | 掃描完整 HTTP 回應（主要入口） |
| `detect_in_html(html_content, url)` | 掃描 HTML 內容 |
| `detect_in_headers(headers, url)` | 掃描 HTTP 標頭 |
| `format_report(result, format)` | 格式化輸出（text / json / sarif） |
| `export_report(result, output_file, format)` | 輸出報告至檔案 |
| `get_statistics(results)` | 多次掃描統計分析 |
| `calculate_entropy(data)` | 計算字串香農熵 |

## 風險評分

```
CRITICAL: critical_count > 0 或 score ≥ 75
HIGH:     high_count > 0 或 score ≥ 50
MEDIUM:   medium_count > 0 或 score ≥ 25
LOW:      其他
```

## 注意事項

- 僅限授權安全測試使用
- 偵測結果中敏感值會自動雜湊（`hash` 欄位），不以明文儲存
- 誤報過濾：白名單關鍵字（`example`/`sample`/`test`/`demo`）自動降低信心度
