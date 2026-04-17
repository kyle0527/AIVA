# function_idor - 不安全的直接物件參考（IDOR）檢測模組

> **版本**: v3.0.0 | **狀態**: ✅ 完成 | **語言**: Python

## 🎯 模組概述

IDOR 綜合檢測模組，負責測試水平權限提升（A 使用者存取 B 使用者的資料）與垂直權限提升（低權限存取高權限專屬資源）。本模組具備從 URL 中自動萃取物件 ID 的能力。

### 功能清單

| 功能 | 說明 |
|------|------|
| 水平 IDOR 偵測 | 跨使用者資源存取測試 (A/B 兩個不同 Token) |
| 垂直 IDOR 偵測 | 低權限帳號→高權限路徑存取測試 |
| 資源 ID 萃取 | 自動識別 URL 中的 ID 並生成變體 Payload |
| 敏感度評分 | 1–5 分判斷回應內容的敏感度，用以調整漏洞嚴重度 |
| 智慧偵測 | SmartIDORDetector 自動編排與發送測試請求 |

## 📐 架構設計

```
function_idor/
├── __init__.py                 # 模組入口匯出
├── detector/
│   └── idor_detector.py        # 主入口 (IDORDetector)
├── engine/
│   └── idor_engine.py          # 核心引擎 (IDOREngine)
├── config/
│   └── idor_config.py          # 設定檔 (IdorConfig)
├── smart_idor_detector.py      # 智慧偵測邏輯 (SmartIDORDetector)
├── resource_id_extractor.py    # ID 萃取邏輯 (ResourceIdExtractor)
└── testers/
    ├── cross_user_tester.py    # 水平測試 (CrossUserTester)
    └── vertical_escalation_tester.py  # 垂直測試 (VerticalEscalationTester)
```

## 🚀 執行方式

### 透過 Python 模組匯入

需要傳遞至少兩個不同權限層級或不同使用者的 HTTP Header 給掃描器才能進行對比。

```python
from services.features.function_idor import IDORDetector

detector = IDORDetector()
task = {
    "url": "https://example.com/api/user/123",
    "test_type": "horizontal",
    "auth_a": {"Authorization": "Bearer UserA_Token"},
    "auth_b": {"Authorization": "Bearer UserB_Token"}
}

findings = await detector.analyze(task)
```

## 🔧 內部 API 參考

| 類別 / 方法 | 說明 |
|------|------|
| `IDORDetector.analyze(task)` | 主要掃描入口，分派水平或垂直測試 |
| `SmartIDORDetector.detect_vulnerabilities(task)` | 智慧化判斷與參數萃取 |
| `IDOREngine.test_horizontal(url, user_a_hdr, user_b_hdr)` | 水平 IDOR 測試邏輯 |
| `IDOREngine.test_vertical(url, low_auth_hdr)` | 垂直 IDOR 測試邏輯 |
| `IDOREngine.extract_ids_from_url(url)` | 透過 Regex 取出 URL 候選 ID |
| `IDOREngine.generate_variants(raw, count)` | 對找出的 ID 生成相似的測試 ID 變體 |
| `CrossUserTester.test_horizontal_idor(...)` | 執行並對比 A/B Token 的回應結果差異 |
| `VerticalEscalationTester.test_vertical_escalation(...)` | 驗證高權限資源是否能被低權限 Token 存取 |
| `ResourceIdExtractor.extract_from_url(url)` | 工具類，從 URL 萃取候選 ID |

## 🔒 注意事項

- 本模組是雙向對比測試，**必須**提供兩組不同使用者的認證標頭（`user_a_hdr`、`user_b_hdr`）才能進行水平測試。
- 垂直測試需要提供低權限認證標頭與高權限的路徑。
- 僅限授權滲透測試使用。
