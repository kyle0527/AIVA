# function_ssrf - 伺服器端請求偽造（SSRF）檢測模組

> **版本**: v2.0.0 | **狀態**: ✅ 完成 | **語言**: Python | **能力登錄**: ✅ 已登錄 (`ssrf_comprehensive`)

## 模組概述

SSRF 綜合檢測模組，涵蓋內網存取、雲端 Metadata 端點、File Protocol、DNS Rebinding 等攻擊向量，並整合 OAST 外帶回調驗證。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 基礎 SSRF 偵測 | ✅ 完成 | SSRFDetector 主入口 |
| 內網存取檢測 | ✅ 完成 | 10.x / 172.x / 192.168.x 偵測 |
| 雲端 Metadata | ✅ 完成 | AWS / GCP / Azure Metadata 端點 |
| File Protocol | ✅ 完成 | `file:///etc/passwd` 等 |
| DNS Rebinding | ✅ 完成 | 向量生成 + 驗證 |
| Blind SSRF（OAST） | ✅ 完成 | 外帶回調 OastDispatcher |
| 參數語意分析 | ✅ 完成 | 自動識別可能觸發 SSRF 的參數 |
| 內部 IP 識別 | ✅ 完成 | InternalAddressDetector |

## 架構

```
function_ssrf/
├── detector/
│   └── ssrf_detector.py            # 主入口（SSRFDetector）
├── engine/
│   └── ssrf_engine.py              # 核心引擎（SSRFEngine）
├── smart_ssrf_detector.py          # 智慧偵測（SmartSSRFDetector）
├── dns_rebinding_detector.py       # DNS Rebinding（DnsRebindingDetector）
├── oast_dispatcher.py              # OAST 回調（OastDispatcher）
├── param_semantics_analyzer.py     # 參數語意分析（ParamSemanticsAnalyzer）
├── internal_address_detector.py    # 內部 IP 識別（InternalAddressDetector）
└── result_publisher.py             # 結果發布（SsrfResultPublisher）
```

## 執行方式

### 透過 AIVA 執行器（推薦）

```bash
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func SSRFDetector.analyze --target https://example.com
```

### 直接使用

```python
from services.features.function_ssrf.detector.ssrf_detector import SSRFDetector
from services.features.function_ssrf.config.ssrf_config import SsrfConfig

config = SsrfConfig()
detector = SSRFDetector(config)
findings = await detector.analyze("https://example.com/fetch?url=")
```

## 可調用方法（公開 API）

| 類別 | 方法 | 說明 |
|------|------|------|
| `SSRFDetector` | `analyze(target_url)` | 主要掃描入口 |
| `SSRFEngine` | `check_internal_access(url)` | 內網存取檢測 |
| `SSRFEngine` | `check_cloud_metadata()` | 雲端 Metadata 端點測試 |
| `SSRFEngine` | `check_file_protocol(url)` | File Protocol 利用 |
| `SSRFEngine` | `run()` | 執行所有檢測 |
| `SmartSSRFDetector` | `detect_vulnerabilities(task)` | 智慧偵測入口 |
| `DnsRebindingDetector` | `generate_vectors(target_internal_ip, attacker_ip)` | DNS Rebinding 向量生成 |
| `OastDispatcher` | `register(task)` | 登錄 Blind SSRF 回調 |
| `ParamSemanticsAnalyzer` | `analyze(task)` | 參數語意分析 |

## 注意事項

- 僅限授權滲透測試使用
- 雲端 Metadata 測試僅限有存取授權的環境
- Blind SSRF（OAST）需要外部回調伺服器
