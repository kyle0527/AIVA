# function_ssrf - 伺服器端請求偽造（SSRF）檢測模組

> **版本**: v3.0.0 | **狀態**: ✅ 完成 | **語言**: Python

## 🎯 模組概述

SSRF 綜合檢測模組，涵蓋內網存取、雲端 Metadata 端點、File Protocol、DNS Rebinding 等攻擊向量，並整合 OAST 外帶回調驗證。

### 功能清單

| 功能 | 說明 |
|------|------|
| 基礎 SSRF 偵測 | SSRFDetector 主入口 |
| 內網存取檢測 | 10.x / 172.x / 192.168.x 內網 IP 探測 |
| 雲端 Metadata | 探測 AWS / GCP / Azure 的 Metadata 端點 |
| File Protocol | 本地檔案讀取，如 `file:///etc/passwd` |
| DNS Rebinding | 向量生成與驗證邏輯 |
| Blind SSRF（OAST） | 外帶回調派發與驗證 |
| 參數語意分析 | 自動識別名稱類似 url, path, redirect 等可能觸發 SSRF 的參數 |

## 📐 架構設計

```
function_ssrf/
├── __init__.py                     # 模組入口匯出
├── detector/
│   └── ssrf_detector.py            # 主掃描入口 (SSRFDetector)
├── engine/
│   └── ssrf_engine.py              # 核心掃描引擎 (SSRFEngine)
├── config/
│   └── ssrf_config.py              # 設定檔 (SsrfConfig)
├── smart_ssrf_detector.py          # 智慧偵測控制邏輯
├── dns_rebinding_detector.py       # DNS Rebinding (DnsRebindingDetector)
├── oast_dispatcher.py              # OAST 回調 (OastDispatcher)
├── param_semantics_analyzer.py     # 參數語意分析 (ParamSemanticsAnalyzer)
├── internal_address_detector.py    # 內部 IP 識別
└── result_publisher.py             # 結果發布
```

## 🚀 執行方式

### 透過 Python 模組匯入

```python
import asyncio
from services.features.function_ssrf import SSRFDetector

async def main():
    detector = SSRFDetector()
    findings = await detector.analyze(
        target="https://example.com/fetch?url=",
        test_type="comprehensive",
        task_id="123",
        scan_id="456"
    )
    for finding in findings:
        print(finding)

asyncio.run(main())
```

## 🔧 內部 API 參考

| 類別 / 方法 | 說明 |
|------|------|
| `SSRFDetector.analyze(...)` | 主要掃描對外接口 |
| `SSRFEngine.check_internal_access(url)` | 內網存取檢測 |
| `SSRFEngine.check_cloud_metadata()` | 雲端 Metadata 端點測試 |
| `SSRFEngine.check_file_protocol(url)` | File Protocol 測試 |
| `SmartSSRFDetector.detect_vulnerabilities(task)` | 智慧型參數與目標分析後執行探測 |
| `DnsRebindingDetector.generate_vectors(...)` | 產生 DNS Rebinding 解析 Payload |
| `OastDispatcher.register(task)` | 登錄與追蹤 Blind SSRF 的回調請求 |
| `ParamSemanticsAnalyzer.analyze(task)` | 從 HTTP 請求中提取 SSRF 高風險參數 |

## 🔒 注意事項

- 僅限授權滲透測試使用。
- 雲端 Metadata 測試如果成功，可能會導致雲端環境憑證外洩，請確保在授權範圍內。
- OAST 驗證依賴外部網路連線，在高度封閉的網路環境中 Blind SSRF 測試會失效。
