# function_xss - 跨站腳本（XSS）檢測模組

> **版本**: v2.0.0 | **狀態**: ✅ 完成 | **語言**: Python | **能力登錄**: ✅ 已登錄 (`xss_multi_context`)

## 模組概述

XSS 綜合檢測模組，支援 Reflected、Stored、DOM-based、Blind XSS 四種類型，並整合外部工具（Dalfox）與跨語言引擎。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| Reflected XSS | ✅ 完成 | Payload 反射驗證 + HTML 轉義排除 |
| Stored XSS | ✅ 完成 | 提交 + 回讀雙階段偵測 |
| DOM-based XSS | ✅ 完成 | source-to-sink 分析 |
| Blind XSS | ✅ 完成 | OAST 回調監聽（BlindXssListenerValidator） |
| Payload 生成 | ✅ 完成 | 50+ Payload，含 WAF 繞過變體 |
| 外部工具整合 | ✅ 完成 | Dalfox（需另行安裝） |
| 跨語言引擎 | ✅ 完成 | CrossLanguageXSSEngine（Go/Rust 工具支援） |

## 架構

```
function_xss/
├── scanner.py                      # 統一掃描入口（XssScanner）
├── traditional_detector.py         # Reflected XSS（TraditionalXssDetector）
├── stored_detector.py              # Stored XSS（StoredXssDetector）
├── dom_xss_detector.py             # DOM XSS（DomXssDetector）
├── blind_xss_listener_validator.py # Blind XSS（BlindXssListenerValidator）
├── payload_generator.py            # Payload 生成（XssPayloadGenerator）
├── command_handler.py              # 指令處理（XSSCommandHandler）
├── task_queue.py                   # 任務佇列（XssTaskQueue）
├── result_publisher.py             # 結果發布（XssResultPublisher）
├── hackingtool_config.py           # 外部工具配置
├── engines/
│   └── hackingtool_engine.py       # 跨語言引擎（CrossLanguageXSSEngine）
└── integration_tools/
    └── xss_tools.py                # XSSManager（綜合入口）
```

## 執行方式

### 透過 AIVA 執行器（推薦）

```bash
# 綜合掃描（自動選擇掃描類型）
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func XssScanner.scan --target https://example.com

# 指定掃描類型
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func XSSManager.comprehensive_scan --target https://example.com
```

### 直接使用

```python
from services.features.function_xss.scanner import XssScanner

scanner = XssScanner()
result = await scanner.scan(
    target_url="https://example.com",
    scan_type="comprehensive",   # reflected / stored / dom / comprehensive
)
```

## 可調用方法（公開 API）

| 類別 | 方法 | 說明 |
|------|------|------|
| `XssScanner` | `scan(target_url, scan_type, options)` | 統一掃描入口（推薦） |
| `TraditionalXssDetector` | `execute(payloads)` | Reflected XSS 偵測 |
| `StoredXssDetector` | `execute(payloads)` | Stored XSS 偵測 |
| `DomXssDetector` | `analyze()` | DOM XSS 分析 |
| `BlindXssListenerValidator` | `provision_payload(task)` | Blind XSS Payload 配置 |
| `XssPayloadGenerator` | `generate_all_payloads()` | 產生全部 Payload |
| `XSSManager` | `comprehensive_scan(target_url, options)` | 整合所有工具的綜合掃描 |

## 注意事項

- 僅限授權滲透測試使用
- Blind XSS 需要外部回調伺服器（OAST）
- Dalfox 整合需另行安裝：`DalfoxIntegration.install_dalfox()`
