# function_xss - 跨站腳本（XSS）檢測模組

> **版本**: v3.0.0 | **狀態**: ✅ 完成 | **語言**: Python

## 🎯 模組概述

XSS 綜合檢測模組，支援 Reflected、Stored、DOM-based、Blind XSS 四種類型，並整合外部工具（如 Dalfox）與跨語言引擎。此模組已經全面改版並支援獨立 CLI 驅動。

### 功能清單

| 功能 | 說明 |
|------|------|
| Reflected XSS | Payload 反射驗證 + HTML 轉義排除 (TraditionalXssDetector) |
| Stored XSS | 提交 + 回讀雙階段偵測 (StoredXssDetector) |
| DOM-based XSS | source-to-sink 靜態/動態特徵分析 (DomXssDetector) |
| Blind XSS | OAST 回調監聽 (BlindXssListenerValidator) |
| Payload 生成 | 50+ Payload 變體產生器，含 WAF 繞過變體 |
| 獨立 CLI 執行 | `__main__.py` 提供獨立命令列介面 |

## 📐 架構設計

```
function_xss/
├── __init__.py                     # 模組入口匯出
├── __main__.py                     # 獨立 CLI 執行入口
├── __main___sync.py                # 同步版 CLI 入口
├── scanner.py                      # 統一掃描入口 (XssScanner)
├── traditional_detector.py         # Reflected XSS 偵測
├── stored_detector.py              # Stored XSS 偵測
├── dom_xss_detector.py             # DOM XSS 偵測
├── blind_xss_listener_validator.py # Blind XSS 監聽器
├── payload_generator.py            # Payload 生成器
├── command_handler.py              # (過渡期) 舊版 CommandHandler 實作
├── task_queue.py                   # 任務佇列 (XssTaskQueue)
├── result_publisher.py             # 結果發布 (XssResultPublisher)
├── payloads.json                   # 外部 Payload 字典
├── hackingtool_config.py           # 外部工具 (如 Dalfox) 配置
├── external_tools/                 # 外部開源工具適配器
│   ├── __init__.py
│   └── dalfox_adapter.py           # Dalfox 整合適配器
├── engines/
│   └── hackingtool_engine.py       # 跨語言引擎介面
└── integration_tools/
    └── xss_tools.py                # 綜合管理介面
```

## 🚀 執行方式

### 透過獨立 CLI 執行 (推薦)

透過 CLI 可以直接以 JSON 格式取得漏洞掃描結果，非常適合腳本或外部整合：

```bash
# 執行綜合 XSS 測試
python -m services.features.function_xss --url "https://example.com" --type "comprehensive"
```

### 透過 Python 模組匯入

```python
from services.features.function_xss.scanner import XssScanner

scanner = XssScanner()

# 支援的類型: reflected, stored, dom, comprehensive
result = await scanner.scan(
    target_url="https://example.com",
    scan_type="comprehensive"
)

print(result)
```

## 🔧 內部 API 參考

| 類別 / 方法 | 說明 |
|------|------|
| `XssScanner.scan(...)` | 統一掃描入口（主要對外 API） |
| `TraditionalXssDetector.execute(payloads)` | 送出 Payload 並比對回傳 HTML 是否直接反射且未經編碼 |
| `StoredXssDetector.execute(payloads)` | 在某處提交 Payload 並在另一處讀取驗證是否儲存型 XSS |
| `DomXssDetector.analyze()` | 解析 JavaScript 判斷是否有 `innerHTML` 或 `eval` 接觸到使用者輸入 |
| `BlindXssListenerValidator.provision_payload(task)` | 取得 OAST 回調 URL 並埋入 Payload，並啟動輪詢監聽 |
| `XssPayloadGenerator.generate_all_payloads()` | 根據目標特徵產生全部組合的 XSS Payload |
| `DalfoxAdapter.run_scan(target)` | 呼叫外部的 Dalfox 工具進行高強度 XSS 測試 |

## 🔒 注意事項

- 預設模式只會進行非破壞性探測（例如 `<script>console.log(1)</script>`）。
- Blind XSS 需要外部回調伺服器（OAST / Interactsh）。
- Dalfox 整合需另行安裝。
- 僅限授權滲透測試使用。
