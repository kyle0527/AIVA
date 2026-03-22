# function_idor - 不安全的直接物件參考（IDOR）檢測模組

> **版本**: v2.0.0 | **狀態**: ✅ 完成 | **語言**: Python | **能力登錄**: ⬜ 待登錄（對應 `idor`）

## 模組概述

IDOR 綜合檢測模組，測試水平權限提升（跨使用者存取）與垂直權限提升（低權限存取高權限資源）。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 水平 IDOR 偵測 | ✅ 完成 | 跨使用者資源存取測試 |
| 垂直 IDOR 偵測 | ✅ 完成 | 低權限→高權限存取測試 |
| 資源 ID 萃取 | ✅ 完成 | URL 中的 ID 自動識別與變體生成 |
| 敏感度評分 | ✅ 完成 | 1–5 分回應敏感度評估 |
| 智慧偵測 | ✅ 完成 | SmartIDORDetector 自動編排 |

## 架構

```
function_idor/
├── detector/
│   └── idor_detector.py        # 主入口（IDORDetector）
├── engine/
│   └── idor_engine.py          # 核心引擎（IDOREngine）
├── smart_idor_detector.py      # 智慧偵測（SmartIDORDetector）
├── resource_id_extractor.py    # ID 萃取（ResourceIdExtractor）
└── testers/
    ├── cross_user_tester.py    # 水平測試（CrossUserTester）
    └── vertical_escalation_tester.py  # 垂直測試（VerticalEscalationTester）
```

## 執行方式

### 透過 AIVA 執行器（推薦）

```bash
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func IDORDetector.analyze --target https://example.com/api/user/123
```

### 直接使用

```python
from services.features.function_idor.detector.idor_detector import IDORDetector
from services.features.function_idor.config.idor_config import IdorConfig

config = IdorConfig()
detector = IDORDetector(config)
findings = await detector.analyze(task)
```

## 可調用方法（公開 API）

| 類別 | 方法 | 說明 |
|------|------|------|
| `IDORDetector` | `analyze(task)` | 主要掃描入口 |
| `SmartIDORDetector` | `detect_vulnerabilities(task)` | 智慧偵測入口 |
| `IDOREngine` | `test_horizontal(url, user_a_hdr, user_b_hdr)` | 水平 IDOR 測試 |
| `IDOREngine` | `test_vertical(url, low_auth_hdr)` | 垂直 IDOR 測試 |
| `IDOREngine` | `extract_ids_from_url(url)` | URL ID 萃取 |
| `IDOREngine` | `generate_variants(raw, count)` | ID 變體生成 |
| `CrossUserTester` | `test_horizontal_idor(url, resource_id, user_a_auth, user_b_auth, method)` | 跨使用者存取測試 |
| `VerticalEscalationTester` | `test_vertical_escalation(url, test_privilege, required_privilege, auth_headers, method)` | 垂直提權測試 |
| `ResourceIdExtractor` | `extract_from_url(url)` | 從 URL 萃取候選 ID |

## 注意事項

- 需提供兩組不同使用者的認證標頭（`user_a_hdr`、`user_b_hdr`）進行水平測試
- 垂直測試需提供低權限認證標頭
- 僅限授權滲透測試使用
