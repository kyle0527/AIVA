# AIVA 能力對應補全清單

> 建立日期：2026-03-21
> 目的：記錄所有功能模組的可呼叫能力，以及它們與 `CAPABILITY_CONFIGS` 的對應關係，供本地執行時補全。

---

## 核心問題

`aiva_common/enums/capabilities.py` 中的 `CAPABILITY_CONFIGS` 共有 **14 筆紀錄，但全部 `module / entrypoint / class_name` 均為 `None`**。
這代表 AI 的 RAG 系統與 Decision Engine 無法透過枚舉自動找到對應的程式碼入口。

```
總枚舉能力數：117
有 CAPABILITY_CONFIGS 紀錄：14（全為 null）
無任何紀錄：103
```

**需要做的事：** 補全 `CAPABILITY_CONFIGS`，為每個 capability 填入對應的 `module`、`class_name`、`entrypoint`、`method`。

---

## 各模組能力清單與建議對應

### 1. `function_sqli` ── 76 個公開方法

**建議對應的 capability enums：**

| Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `sql_injection` | `SmartDetectionManager` | `scan_target(task)` | `services.features.function_sqli.smart_detection_manager` |
| `sql_injection_blind` | `BooleanDetectionEngine` | `detect(target_url, params, method)` | `services.features.function_sqli.engines.boolean_detection_engine` |
| `sql_injection_time_based` | `TimeDetectionEngine` | `detect(target_url, params, method)` | `services.features.function_sqli.engines.time_detection_engine` |
| `sql_injection_union` | `UnionDetectionEngine` | `detect(target_url, params, method)` | `services.features.function_sqli.engines.union_detection_engine` |

**其他可用但尚未有枚舉的能力：**
- `ErrorDetectionEngine.detect()` ── error-based SQLi
- `OOBDetectionEngine.detect()` ── Out-of-Band SQLi
- `HackingToolDetectionEngine.detect()` ── 外部工具整合（sqlmap 等）
- `SQLInjectionManager.comprehensive_scan(target_url, options)` ── 綜合掃描入口
- `NoSQLInjectionScanner.scan_target(target)` ── NoSQL 注入
- `BlindSQLInjectionScanner.scan_blind_injection(target)` ── Blind SQLi
- `HackingToolSQLManager.get_tool_recommendations(target_type)` ── 工具推薦
- `BackendDbFingerprinter.fingerprint(response)` ── 資料庫指紋識別

---

### 2. `function_xss` ── 45 個公開方法

**建議對應的 capability enums：**

| Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `xss_reflected` | `TraditionalXssDetector` | `execute(payloads)` | `services.features.function_xss.traditional_detector` |
| `xss_stored` | `StoredXssDetector` | `execute(payloads)` | `services.features.function_xss.stored_detector` |
| `xss_dom` | `DomXssDetector` | `analyze()` | `services.features.function_xss.dom_xss_detector` |

**其他可用但尚未有枚舉的能力：**
- `XssScanner.scan(target_url, scan_type, options)` ── 統一掃描入口（推薦作為主要 entrypoint）
- `BlindXssListenerValidator.provision_payload(task)` ── Blind XSS
- `CrossLanguageXSSEngine.detect(target_url, mode, ...)` ── 跨語言引擎（整合 Dalfox 等）
- `XSSManager.comprehensive_scan(target_url, options)` ── 綜合掃描
- `XssPayloadGenerator.generate_all_payloads()` ── Payload 生成

---

### 3. `function_ssrf` ── 26 個公開方法

**建議對應的 capability enums：**

| Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `ssrf_basic` | `SSRFDetector` | `analyze(target_url)` | `services.features.function_ssrf.detector.ssrf_detector` |
| `ssrf_blind` | `OastDispatcher` | `register(task)` + `fetch_events(token)` | `services.features.function_ssrf.oast_dispatcher` |

**其他可用但尚未有枚舉的能力：**
- `SSRFEngine.check_internal_access(url)` ── 內網存取檢測
- `SSRFEngine.check_cloud_metadata()` ── 雲端 Metadata 端點（AWS/GCP/Azure）
- `SSRFEngine.check_file_protocol(url)` ── File protocol 利用
- `DnsRebindingDetector.generate_vectors(...)` ── DNS Rebinding 攻擊向量
- `ParamSemanticsAnalyzer.analyze(task)` ── 參數語意分析
- `SmartSSRFDetector.detect_vulnerabilities(task)` ── 智慧偵測入口

---

### 4. `function_idor` ── 18 個公開方法

**尚未對應任何 capability enum（模組整體未登錄）**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `idor` | `IDORDetector` | `analyze(task)` | `services.features.function_idor.detector.idor_detector` |
| *(新增)* `idor_horizontal` | `CrossUserTester` | `test_horizontal_idor(url, resource_id, ...)` | `services.features.function_idor.testers.cross_user_tester` |
| *(新增)* `idor_vertical` | `VerticalEscalationTester` | `test_vertical_escalation(url, ...)` | `services.features.function_idor.testers.vertical_escalation_tester` |

**其他可用能力：**
- `SmartIDORDetector.detect_vulnerabilities(task)` ── 智慧偵測入口
- `ResourceIdExtractor.extract_from_url(url)` ── 資源 ID 萃取
- `IDOREngine.test_horizontal(url, ...)` / `test_vertical(url, ...)` ── 底層引擎

---

### 5. `function_postex` ── 26 個公開方法

**尚未對應任何 capability enum**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `privilege_escalation_linux` | `PrivilegeEscalator` | `check_suid_binaries()` / `check_sudo_misconfiguration()` / `check_kernel_exploits()` / `run_full_assessment()` | `services.features.function_postex.engines.privilege_engine` |
| `privilege_escalation_windows` | `PrivilegeEscalator` | `check_writable_services()` / `run_full_assessment()` | `services.features.function_postex.engines.privilege_engine` |
| `lateral_movement` | `LateralMovementTester` | `run_full_assessment()` | `services.features.function_postex.engines.lateral_engine` |
| `pass_the_hash` | `LateralMovementTester` | `simulate_pass_the_hash()` | `services.features.function_postex.engines.lateral_engine` |
| `persistence_install` | `PersistenceChecker` | `run_full_assessment()` | `services.features.function_postex.engines.persistence_engine` |
| *(統一入口)* | `PostExDetector` | `analyze(test_type, target, ...)` | `services.features.function_postex.detector.postex_detector` |

**`PostExDetector.analyze()` 的 `test_type` 參數值：**
`privilege_escalation` / `lateral_movement` / `persistence`

---

### 6. `function_exploit` ── 16 個公開方法

**尚未對應任何 capability enum**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| *(新增)* `exploit_execute` | `AttackExecutor` | `execute_plan_with_ai_analysis(plan, target, ai_analysis_results)` | `services.features.function_exploit.executor.attack_executor` |
| `data_exfiltration` | `AttackExecutor` | `execute_plan(plan, target)` | `services.features.function_exploit.executor.attack_executor` |
| *(新增)* `payload_generate` | `PayloadGenerator` | `generate_with_target_analysis(vuln_type, target_info, ...)` | `services.features.function_exploit.generators.payload_generator` |

**其他可用能力：**
- `ExploitManager.execute_exploit(exploit_id, target, _parameters)` ── 執行已登錄的 exploit
- `AttackValidator.validate_result(attack_type, result)` ── 攻擊結果驗證
- `BizLogicAttackExecutor.execute_attack(attack_type, target_url, parameters)` ── 業務邏輯攻擊執行

---

### 7. `function_web_scanner` ── 27 個公開方法

**尚未對應任何 capability enum**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `port_scan_tcp` | `PortScanner` | `scan(host, ports)` | `services.features.function_web_scanner.scanners.port_scanner` |
| `port_scan_full` | `PortScanner` | `scan(host, ports=None)` | `services.features.function_web_scanner.scanners.port_scanner` |
| `technology_detection` | `TechDetector` | `detect(url)` | `services.features.function_web_scanner.scanners.tech_detector` |
| `subdomain_enumeration` | `SubdomainScanner` | `scan(domain, mode)` | `services.features.function_web_scanner.scanners.subdomain_scanner` |
| `directory_bruteforce` | `DirectoryBruteforcer` | `scan(base_url, extensions)` | `services.features.function_web_scanner.scanners.directory_bruteforcer` |
| `web_crawler` | `WebCrawler` | `crawl(start_url)` | `services.features.function_web_scanner.scanners.web_crawler` |
| `vulnerability_scan` | `WebAttackManager` | `comprehensive_scan(target_url, options)` | `services.features.function_web_scanner.integration_tools.web_tools` |

**注意：** `web_tools.py` 中有 4 個 `NotImplementedError`（`SubdomainEnumerator`, `DirectoryScanner`, `VulnerabilityScanner`, `TechnologyDetector`），
應優先使用 `scanners/` 目錄下的具體實作，而非 `integration_tools/` 中的包裝類別。

---

### 8. `function_bizlogic` ── 22 個公開方法

**尚未對應任何 capability enum**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| *(新增)* `race_condition` | `RaceConditionScanner` | `run_all_tests(test_endpoints)` | `services.features.function_bizlogic.race_condition_scanner` |
| *(新增)* `price_manipulation` | `PriceManipulationScanner` | `run_all_tests(endpoint)` | `services.features.function_bizlogic.price_manipulation_scanner` |
| *(新增)* `workflow_bypass` | `WorkflowBypassScanner` | `run_all_tests()` | `services.features.function_bizlogic.workflow_bypass_scanner` |
| *(統一入口)* `rest_api_abuse` | `BizLogicManager` | `comprehensive_scan(target_url, options)` | `services.features.function_bizlogic.integration_tools.bizlogic_tools` |

---

### 9. `function_forensic` ── 9 個公開方法（排除 legacy）

**尚未對應任何 capability enum**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `memory_analysis` | `ForensicManager` | `analyze_memory_dump(evidence_id)` | `services.features.function_forensic.manager` |
| `disk_image` | `ForensicManager` | `analyze_disk_image(evidence_id, deep_scan)` | `services.features.function_forensic.manager` |
| `timeline_analysis` | `ForensicManager` | `generate_timeline(evidence_id)` | `services.features.function_forensic.manager` |
| *(新增)* `evidence_acquire` | `ForensicManager` | `acquire_evidence(case_id, source_path, evidence_type, acquired_by)` | `services.features.function_forensic.manager` |

---

### 10. `function_info_leak` ── 10 個公開方法

**尚未對應任何 capability enum（這是覆蓋最廣的模組之一）**

| 建議 Capability ID | 對應類別 | 對應方法 | 模組路徑 |
|---|---|---|---|
| `secret_detection` | `SensitiveInfoDetector` | `detect_in_response(response_body, headers, url)` | `services.features.function_info_leak.sensitive_info_detector` |
| *(新增)* `info_leak_html` | `SensitiveInfoDetector` | `detect_in_html(html_content, url)` | `services.features.function_info_leak.sensitive_info_detector` |
| *(新增)* `info_leak_headers` | `SensitiveInfoDetector` | `detect_in_headers(headers, url)` | `services.features.function_info_leak.sensitive_info_detector` |

**偵測涵蓋類型（50+ 種）：**
API Keys、AWS/GCP/Azure 憑證、GitHub/GitLab tokens、JWT tokens、
RSA/EC 私鑰、資料庫連線字串、PII（Email/電話/SSN/護照）、信用卡號、
高熵字串（entropy analysis）等。

---

## 特殊模組（非 Python）

### `function_authn_go`
- **語言：** Go
- **執行方式：** 編譯後 binary `function_authn_go/bin/authn-worker`
- **目前狀態：** Python wrapper 存在但 Go 主體未完成
- **建議：** 完成 Go 實作後，透過 `subprocess` 呼叫

### `function_crypto`
- **語言：** Rust
- **執行方式：** Rust CLI binary `crypto-scanner`，直接 subprocess 呼叫
- **目前狀態：** Rust 核心存在，Python binding 缺失
- **建議：** 直接對應 `Analysis.secret_detection` 或 `Analysis.sast_scan`

---

## CAPABILITY_CONFIGS 補全範本

以下為 `aiva_common/enums/capabilities.py` 中 `CAPABILITY_CONFIGS` 需要補全的格式：

```python
CAPABILITY_CONFIGS = {
    # ── SQL Injection ──
    "sql_injection": {
        "module": "services.features.function_sqli.smart_detection_manager",
        "class_name": "SmartDetectionManager",
        "method": "scan_target",
        "entrypoint": "services/features/function_sqli/smart_detection_manager.py",
        "required_params": ["target"],
        "optional_params": ["config"],
    },
    "sql_injection_blind": {
        "module": "services.features.function_sqli.engines.boolean_detection_engine",
        "class_name": "BooleanDetectionEngine",
        "method": "detect",
        "entrypoint": "services/features/function_sqli/engines/boolean_detection_engine.py",
        "required_params": ["target_url", "params", "method"],
        "optional_params": [],
    },
    "sql_injection_time_based": {
        "module": "services.features.function_sqli.engines.time_detection_engine",
        "class_name": "TimeDetectionEngine",
        "method": "detect",
        "entrypoint": "services/features/function_sqli/engines/time_detection_engine.py",
        "required_params": ["target_url", "params", "method"],
        "optional_params": [],
    },
    "sql_injection_union": {
        "module": "services.features.function_sqli.engines.union_detection_engine",
        "class_name": "UnionDetectionEngine",
        "method": "detect",
        "entrypoint": "services/features/function_sqli/engines/union_detection_engine.py",
        "required_params": ["target_url", "params", "method"],
        "optional_params": [],
    },

    # ── XSS ──
    "xss_reflected": {
        "module": "services.features.function_xss.traditional_detector",
        "class_name": "TraditionalXssDetector",
        "method": "execute",
        "entrypoint": "services/features/function_xss/traditional_detector.py",
        "required_params": ["payloads"],
        "optional_params": [],
    },
    "xss_stored": {
        "module": "services.features.function_xss.stored_detector",
        "class_name": "StoredXssDetector",
        "method": "execute",
        "entrypoint": "services/features/function_xss/stored_detector.py",
        "required_params": ["payloads"],
        "optional_params": [],
    },
    "xss_dom": {
        "module": "services.features.function_xss.dom_xss_detector",
        "class_name": "DomXssDetector",
        "method": "analyze",
        "entrypoint": "services/features/function_xss/dom_xss_detector.py",
        "required_params": [],
        "optional_params": [],
    },

    # ── SSRF ──
    "ssrf_basic": {
        "module": "services.features.function_ssrf.detector.ssrf_detector",
        "class_name": "SSRFDetector",
        "method": "analyze",
        "entrypoint": "services/features/function_ssrf/detector/ssrf_detector.py",
        "required_params": ["target_url"],
        "optional_params": [],
    },
    "ssrf_blind": {
        "module": "services.features.function_ssrf.oast_dispatcher",
        "class_name": "OastDispatcher",
        "method": "register",
        "entrypoint": "services/features/function_ssrf/oast_dispatcher.py",
        "required_params": ["task"],
        "optional_params": [],
    },

    # ── IDOR ──
    "idor": {
        "module": "services.features.function_idor.detector.idor_detector",
        "class_name": "IDORDetector",
        "method": "analyze",
        "entrypoint": "services/features/function_idor/detector/idor_detector.py",
        "required_params": ["task"],
        "optional_params": [],
    },

    # ── Port Scan ──
    "port_scan_tcp": {
        "module": "services.features.function_web_scanner.scanners.port_scanner",
        "class_name": "PortScanner",
        "method": "scan",
        "entrypoint": "services/features/function_web_scanner/scanners/port_scanner.py",
        "required_params": ["host"],
        "optional_params": ["ports"],
    },
    "port_scan_full": {
        "module": "services.features.function_web_scanner.scanners.port_scanner",
        "class_name": "PortScanner",
        "method": "scan",
        "entrypoint": "services/features/function_web_scanner/scanners/port_scanner.py",
        "required_params": ["host"],
        "optional_params": [],
    },

    # ── Scan ──
    "vulnerability_scan": {
        "module": "services.features.function_web_scanner.integration_tools.web_tools",
        "class_name": "WebAttackManager",
        "method": "comprehensive_scan",
        "entrypoint": "services/features/function_web_scanner/integration_tools/web_tools.py",
        "required_params": ["target_url"],
        "optional_params": ["options"],
    },
    "subdomain_enumeration": {
        "module": "services.features.function_web_scanner.scanners.subdomain_scanner",
        "class_name": "SubdomainScanner",
        "method": "scan",
        "entrypoint": "services/features/function_web_scanner/scanners/subdomain_scanner.py",
        "required_params": ["domain"],
        "optional_params": ["mode"],
    },
    "directory_bruteforce": {
        "module": "services.features.function_web_scanner.scanners.directory_bruteforcer",
        "class_name": "DirectoryBruteforcer",
        "method": "scan",
        "entrypoint": "services/features/function_web_scanner/scanners/directory_bruteforcer.py",
        "required_params": ["base_url"],
        "optional_params": ["extensions"],
    },
    "web_crawler": {
        "module": "services.features.function_web_scanner.scanners.web_crawler",
        "class_name": "WebCrawler",
        "method": "crawl",
        "entrypoint": "services/features/function_web_scanner/scanners/web_crawler.py",
        "required_params": ["start_url"],
        "optional_params": [],
    },
    "technology_detection": {
        "module": "services.features.function_web_scanner.scanners.tech_detector",
        "class_name": "TechDetector",
        "method": "detect",
        "entrypoint": "services/features/function_web_scanner/scanners/tech_detector.py",
        "required_params": ["url"],
        "optional_params": [],
    },

    # ── Recon ──
    "whois_lookup": {
        "module": None,  # 尚無對應模組，需新增或外部工具
        "class_name": None,
        "method": None,
        "entrypoint": None,
        "required_params": ["domain"],
        "optional_params": [],
    },
    "dns_lookup": {
        "module": None,  # 尚無對應模組
        "class_name": None,
        "method": None,
        "entrypoint": None,
        "required_params": ["domain"],
        "optional_params": [],
    },

    # ── Analysis ──
    "sast_scan": {
        "module": None,  # function_crypto (Rust) 或外部工具
        "class_name": None,
        "method": None,
        "entrypoint": "crypto-scanner",  # Rust binary
        "required_params": ["target_path"],
        "optional_params": [],
    },
    "secret_detection": {
        "module": "services.features.function_info_leak.sensitive_info_detector",
        "class_name": "SensitiveInfoDetector",
        "method": "detect_in_response",
        "entrypoint": "services/features/function_info_leak/sensitive_info_detector.py",
        "required_params": ["response_body", "headers", "url"],
        "optional_params": [],
    },

    # ── Forensic ──
    "memory_analysis": {
        "module": "services.features.function_forensic.manager",
        "class_name": "ForensicManager",
        "method": "analyze_memory_dump",
        "entrypoint": "services/features/function_forensic/manager.py",
        "required_params": ["evidence_id"],
        "optional_params": [],
    },
    "steganography_detect": {
        "module": "services.features.function_steganography.manager",
        "class_name": "SteganographyManager",
        "method": "detect_hidden_data",
        "entrypoint": "services/features/function_steganography/manager.py",
        "required_params": ["file_path"],
        "optional_params": [],
    },

    # ── Exploit ──
    "privilege_escalation_linux": {
        "module": "services.features.function_postex.engines.privilege_engine",
        "class_name": "PrivilegeEscalator",
        "method": "run_full_assessment",
        "entrypoint": "services/features/function_postex/engines/privilege_engine.py",
        "required_params": [],
        "optional_params": [],
    },
    "privilege_escalation_windows": {
        "module": "services.features.function_postex.engines.privilege_engine",
        "class_name": "PrivilegeEscalator",
        "method": "run_full_assessment",
        "entrypoint": "services/features/function_postex/engines/privilege_engine.py",
        "required_params": [],
        "optional_params": [],
    },
    "lateral_movement": {
        "module": "services.features.function_postex.engines.lateral_engine",
        "class_name": "LateralMovementTester",
        "method": "run_full_assessment",
        "entrypoint": "services/features/function_postex/engines/lateral_engine.py",
        "required_params": [],
        "optional_params": [],
    },
    "persistence_install": {
        "module": "services.features.function_postex.engines.persistence_engine",
        "class_name": "PersistenceChecker",
        "method": "run_full_assessment",
        "entrypoint": "services/features/function_postex/engines/persistence_engine.py",
        "required_params": [],
        "optional_params": [],
    },
}
```

---

## 摘要統計

| 類別 | 枚舉總數 | 已有對應模組 | 缺少對應模組 |
|---|:---:|:---:|:---:|
| Attack | 40 | 7（sqli×4, xss×3, ssrf×2, idor×1） | 33 |
| Scan | 19 | 6（port×2, vuln, subdomain, dir, crawler, tech） | 13 |
| Recon | 16 | 0 | 16 |
| Analysis | 14 | 2（secret_detection, sast_scan） | 12 |
| Forensic | 12 | 2（memory_analysis, steganography_detect） | 10 |
| Exploit | 11 | 4（privesc×2, lateral, persistence） | 7 |
| Report | 5 | 0（需要整合 reporting 模組） | 5 |
| **合計** | **117** | **~22** | **~95** |

---

## 待處理工作清單

### 高優先（有現成模組，直接填入 CAPABILITY_CONFIGS）
- [ ] `function_idor` → `idor` enum 補全
- [ ] `function_postex` → `privilege_escalation_*`, `lateral_movement`, `persistence_install` 補全
- [ ] `function_web_scanner` → port scan, tech detection, subdomain, dir, crawler 補全
- [ ] `function_info_leak` → `secret_detection` 補全
- [ ] `function_bizlogic` → 需新增 `race_condition`, `price_manipulation`, `workflow_bypass` 三個新 enum

### 中優先（模組存在但需確認介面）
- [ ] `function_exploit` → 新增 `exploit_execute`, `payload_generate` enum
- [ ] `function_forensic` → `memory_analysis`, `disk_image`, `timeline_analysis` 補全

### 低優先（模組不完整或依賴外部工具）
- [ ] `whois_lookup`, `dns_lookup` → 需新增小型工具模組或整合 `dnspython` / `python-whois`
- [ ] `sast_scan` → 接通 Rust `crypto-scanner` binary
- [ ] `function_authn_go` → 完成 Go 主體後接入

### 需新增到 AttackCapability enum
- [ ] `race_condition`
- [ ] `price_manipulation`
- [ ] `workflow_bypass`
- [ ] `nosql_injection`（function_sqli 已有 NoSQLInjectionScanner）
- [ ] `blind_xss`（function_xss 已有 BlindXssListenerValidator）
