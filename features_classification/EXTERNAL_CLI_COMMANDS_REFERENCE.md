# AIVA 外部模組 CLI 指令參考手冊

> 生成時間: 2026-01-20 16:14:45
> 資料來源: classification_data.json
> 總流程數: 212

## 模組總覽

| 語言 | 流程數 | 模組數 |
|------|--------|--------|
| PYTHON | 207 | 7 |
| RUST | 1 | 1 |
| GO | 4 | 1 |

---

## PYTHON 流程

### function_bizlogic

**類型**: business_logic

**描述**: 業務邏輯漏洞

**用途**: [業務邏輯-競態] 檢測競態條件漏洞，同時發送多個請求測試鎖機制

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 5 | comprehensive_scan → RaceConditionScanner | 2 | `python aiva_external_executor.py --lang python --flow 5 --target <URL>` |
| 6 | comprehensive_scan → PriceManipulationScanner | 2 | `python aiva_external_executor.py --lang python --flow 6 --target <URL>` |
| 7 | comprehensive_scan → WorkflowBypassScanner | 2 | `python aiva_external_executor.py --lang python --flow 7 --target <URL>` |
| 8 | main → run_price_test | 2 | `python aiva_external_executor.py --lang python --flow 8 --target <URL>` |
| 9 | main → run_race_test | 2 | `python aiva_external_executor.py --lang python --flow 9 --target <URL>` |
| 10 | main → run_workflow_test | 2 | `python aiva_external_executor.py --lang python --flow 10 --target <URL>` |
| 11 | main → mk_finding_dict → create_bizlogic_finding | 3 | `python aiva_external_executor.py --lang python --flow 11 --target <URL>` |

### function_idor

**類型**: access_control

**描述**: IDOR 漏洞檢測

**用途**: [IDOR檢測引擎] 核心越權檢測邏輯，嘗試不同用戶訪問相同資源

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 12 | analyze → IDOREngine | 2 | `python aiva_external_executor.py --lang python --flow 12 --target <URL>` |
| 13 | run → EnhancedIDORWorker | 2 | `python aiva_external_executor.py --lang python --flow 13 --target <URL>` |
| 14 | run → _topic | 2 | `python aiva_external_executor.py --lang python --flow 14 --target <URL>` |
| 15 | run → IdorConfig | 2 | `python aiva_external_executor.py --lang python --flow 15 --target <URL>` |
| 16 | run → IDORDetector | 2 | `python aiva_external_executor.py --lang python --flow 16 --target <URL>` |
| 17 | run → ResourceIdExtractor | 2 | `python aiva_external_executor.py --lang python --flow 17 --target <URL>` |
| 18 | process_task → ResourceIdExtractor | 2 | `python aiva_external_executor.py --lang python --flow 18 --target <URL>` |
| 19 | process_task → EnhancedIDORWorker | 2 | `python aiva_external_executor.py --lang python --flow 19 --target <URL>` |
| 20 | detect_vulnerabilities → IDORDetectionContext | 2 | `python aiva_external_executor.py --lang python --flow 20 --target <URL>` |
| 21 | __init__ → IdorConfig | 2 | `python aiva_external_executor.py --lang python --flow 21 --target <URL>` |

*... 還有 3 個流程*

### function_postex

**類型**: unknown

**描述**: Function Postex

**用途**: [function_postex] 安全檢測功能

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 25 | __init__ → PostExDetector | 2 | `python aiva_external_executor.py --lang python --flow 25 --target <URL>` |
| 26 | run → _process_task | 2 | `python aiva_external_executor.py --lang python --flow 26 --target <URL>` |
| 27 | scan_full → PostExResult | 2 | `python aiva_external_executor.py --lang python --flow 27 --target <URL>` |
| 28 | main → LateralMovementTester | 2 | `python aiva_external_executor.py --lang python --flow 28 --target <URL>` |
| 29 | main → PersistenceChecker | 2 | `python aiva_external_executor.py --lang python --flow 29 --target <URL>` |
| 30 | main → PrivilegeEscalator | 2 | `python aiva_external_executor.py --lang python --flow 30 --target <URL>` |
| 31 | __init__ → PrivilegeEscalationEngine | 2 | `python aiva_external_executor.py --lang python --flow 31 --target <URL>` |
| 32 | __init__ → LateralMovementEngine | 2 | `python aiva_external_executor.py --lang python --flow 32 --target <URL>` |
| 33 | __init__ → PersistenceEngine | 2 | `python aiva_external_executor.py --lang python --flow 33 --target <URL>` |
| 34 | scan_target → PostExManager | 2 | `python aiva_external_executor.py --lang python --flow 34 --target <URL>` |

### function_sqli

**類型**: injection

**描述**: SQL 注入檢測

**用途**: [SQLi賞金模式] 針對高價值目標的深度測試，使用高級繞過技術

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 35 | __init__ → BountyHunterManager | 2 | `python aiva_external_executor.py --lang python --flow 35 --target <URL>` |
| 36 | detect → PayloadWrapperEncoder | 2 | `python aiva_external_executor.py --lang python --flow 36 --target <URL>` |
| 37 | __init__ → BountyHunterScanner | 2 | `python aiva_external_executor.py --lang python --flow 37 --target <URL>` |
| 38 | _consume_queue → _execute_task | 2 | `python aiva_external_executor.py --lang python --flow 38 --target <URL>` |
| 39 | _convert_to_detection_result → DetectionResult | 2 | `python aiva_external_executor.py --lang python --flow 39 --target <URL>` |
| 40 | detect → PayloadWrapperEncoder | 2 | `python aiva_external_executor.py --lang python --flow 40 --target <URL>` |
| 41 | process_task → SqliOrchestrator | 2 | `python aiva_external_executor.py --lang python --flow 41 --target <URL>` |
| 42 | process_task → SqliContext | 2 | `python aiva_external_executor.py --lang python --flow 42 --target <URL>` |
| 43 | _create_detection_result → DetectionResult | 2 | `python aiva_external_executor.py --lang python --flow 43 --target <URL>` |
| 44 | __init__ → SqliEngineConfig | 2 | `python aiva_external_executor.py --lang python --flow 44 --target <URL>` |

*... 還有 20 個流程*

### function_ssrf

**類型**: ssrf

**描述**: SSRF 漏洞檢測

**用途**: [SSRF通用] 服務器端請求偽造檢測，測試後端 URL 請求功能

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 65 | _build_internal_finding → _severity_from_summary | 2 | `python aiva_external_executor.py --lang python --flow 65 --target <URL>` |
| 66 | register → OastProbe | 2 | `python aiva_external_executor.py --lang python --flow 66 --target <URL>` |
| 67 | analyze → SSRFEngine | 2 | `python aiva_external_executor.py --lang python --flow 67 --target <URL>` |
| 68 | fetch_events → OastEvent | 2 | `python aiva_external_executor.py --lang python --flow 68 --target <URL>` |
| 69 | __init__ → OastDispatcher | 2 | `python aiva_external_executor.py --lang python --flow 69 --target <URL>` |
| 70 | __init__ → InternalAddressDetector | 2 | `python aiva_external_executor.py --lang python --flow 70 --target <URL>` |
| 71 | __init__ → ParamSemanticsAnalyzer | 2 | `python aiva_external_executor.py --lang python --flow 71 --target <URL>` |
| 72 | run → SsrfResultPublisher | 2 | `python aiva_external_executor.py --lang python --flow 72 --target <URL>` |
| 73 | run → ParamSemanticsAnalyzer | 2 | `python aiva_external_executor.py --lang python --flow 73 --target <URL>` |
| 74 | run → InternalAddressDetector | 2 | `python aiva_external_executor.py --lang python --flow 74 --target <URL>` |

*... 還有 20 個流程*

### function_web_scanner

**類型**: unknown

**描述**: Function Web Scanner

**用途**: [function_web_scanner] 安全檢測功能

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 95 | comprehensive_scan → WebTarget | 2 | `python aiva_external_executor.py --lang python --flow 95 --target <URL>` |
| 96 | comprehensive_scan → ScanResult | 2 | `python aiva_external_executor.py --lang python --flow 96 --target <URL>` |
| 97 | __init__ → SubdomainEnumerator | 2 | `python aiva_external_executor.py --lang python --flow 97 --target <URL>` |
| 98 | __init__ → DirectoryScanner | 2 | `python aiva_external_executor.py --lang python --flow 98 --target <URL>` |
| 99 | __init__ → VulnerabilityScanner | 2 | `python aiva_external_executor.py --lang python --flow 99 --target <URL>` |
| 100 | __init__ → TechnologyDetector | 2 | `python aiva_external_executor.py --lang python --flow 100 --target <URL>` |
| 101 | __init__ → WebAttackManager | 2 | `python aiva_external_executor.py --lang python --flow 101 --target <URL>` |
| 102 | __init__ → WebAttackCLI | 2 | `python aiva_external_executor.py --lang python --flow 102 --target <URL>` |
| 103 | register_capability → WebAttackCapability | 2 | `python aiva_external_executor.py --lang python --flow 103 --target <URL>` |
| 104 | scan_target → WebScannerManager | 2 | `python aiva_external_executor.py --lang python --flow 104 --target <URL>` |

### function_xss

**類型**: injection

**描述**: XSS 漏洞檢測

**用途**: XSS testing from bruteforcer to getUrl

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 105 | bruteforcer → getUrl | 2 | `python aiva_external_executor.py --lang python --flow 105 --target <URL>` |
| 106 | bruteforcer → getParams | 2 | `python aiva_external_executor.py --lang python --flow 106 --target <URL>` |
| 107 | bruteforcer → requester → converter | 3 | `python aiva_external_executor.py --lang python --flow 107 --target <URL>` |
| 108 | _test_dom_payloads → XSSVulnerability | 2 | `python aiva_external_executor.py --lang python --flow 108 --target <URL>` |
| 109 | _detect_language_environments → checker → fillHoles | 3 | `python aiva_external_executor.py --lang python --flow 109 --target <URL>` |
| 110 | _detect_language_environments → LanguageEnvironment | 2 | `python aiva_external_executor.py --lang python --flow 110 --target <URL>` |
| 111 | run → XssResultPublisher | 2 | `python aiva_external_executor.py --lang python --flow 111 --target <URL>` |
| 112 | run → XssTaskQueue | 2 | `python aiva_external_executor.py --lang python --flow 112 --target <URL>` |
| 113 | __init__ → get_xss_tools_config | 2 | `python aiva_external_executor.py --lang python --flow 113 --target <URL>` |
| 114 | _consume_queue → _execute_task → process_task → XssPayloadGenerator | 4 | `python aiva_external_executor.py --lang python --flow 114 --target <URL>` |

*... 還有 97 個流程*

---

## RUST 流程

### rust_engine

**類型**: unknown

**描述**: Rust Engine

**用途**: [rust_engine] 安全檢測功能

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 212 | main → PathsConfig::new | 2 | `python aiva_external_executor.py --lang rust --func main` |

---

## GO 流程

### function_authn_go

**類型**: authentication

**描述**: 身份驗證

**用途**: [認證通用] 身份驗證漏洞檢測，測試登錄、會話、權限

| Flow ID | 流程路徑 | 長度 | CLI 指令 |
|---------|----------|------|----------|
| 1 | scan_authentication → _find_go_binary | 2 | `python aiva_external_executor.py --lang go --func scan_authentication` |
| 2 | get_engine_info → _find_go_binary | 2 | `python aiva_external_executor.py --lang go --func get_engine_info` |
| 3 | scan_target → AuthnManager | 2 | `python aiva_external_executor.py --lang go --func scan_target` |
| 4 | _check_go_availability → _find_go_binary | 2 | `python aiva_external_executor.py --lang go --func _check_go_availability` |

---

## 使用指南

### 基本用法

```bash
# 列出所有可用能力
python aiva_external_executor.py --list

# 列出特定語言
python aiva_external_executor.py --list --lang python

# 執行 Python 流程（dry-run）
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run

# 實際執行
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000
```

### 按攻擊類型分類

- **access_control**: 13 個流程，涵蓋 function_idor
- **authentication**: 4 個流程，涵蓋 function_authn_go
- **business_logic**: 7 個流程，涵蓋 function_bizlogic
- **injection**: 137 個流程，涵蓋 function_sqli, function_xss
- **ssrf**: 30 個流程，涵蓋 function_ssrf
- **unknown**: 21 個流程，涵蓋 function_postex, function_web_scanner, rust_engine
