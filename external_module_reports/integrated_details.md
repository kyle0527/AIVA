# 外部模組詳細流程信息

生成時間: 2026-01-13 23:04:16

---

## FEATURES

### function_xss

**描述**: XSS 漏洞檢測

**攻擊類型**: injection

**流程數**: 109

**語言**: Unknown

**流程範例**:

1. CrossLanguageXSSEngine._detect_language_environments → checker → fillHoles
2. CrossLanguageXSSEngine._detect_language_environments → LanguageEnvironment
3. bruteforcer → getUrl

---

### function_ssrf

**描述**: SSRF 漏洞檢測

**攻擊類型**: ssrf

**流程數**: 35

**語言**: Unknown

**流程範例**:

1. SsrfWorkerService.process_task → SsrfResultPublisher
2. SsrfWorkerService.process_task → _execute_task → process_task ... (+1 更多)
3. SsrfWorkerService.process_task → _execute_task → process_task ... (+1 更多)

---

### function_sqli

**描述**: SQL 注入檢測

**攻擊類型**: injection

**流程數**: 32

**語言**: Unknown

**流程範例**:

1. _consume_queue → _execute_task
2. SQLInjectionManager.__init__ → SqlmapIntegration
3. SQLInjectionManager.__init__ → CustomSQLInjectionScanner

---

### function_idor

**描述**: IDOR 漏洞檢測

**攻擊類型**: access_control

**流程數**: 19

**語言**: Unknown

**流程範例**:

1. IdorWorkerService.__init__ → IdorWorker
2. SmartIDORDetector.__init__ → IdorConfig
3. EnhancedIDORWorker.run → ResourceIdExtractor

---

### function_bizlogic

**描述**: 業務邏輯漏洞

**攻擊類型**: business_logic

**流程數**: 8

**語言**: Unknown

**流程範例**:

1. BizLogicManager.comprehensive_scan → RaceConditionScanner
2. BizLogicManager.comprehensive_scan → PriceManipulationScanner
3. BizLogicManager.comprehensive_scan → WorkflowBypassScanner

---

### function_authn_go

**描述**: 身份驗證

**攻擊類型**: authentication

**流程數**: 4

**語言**: Go

**流程範例**:

1. main → DialBroker
2. main → TopicFromEnv
3. main → DefaultConfig

---

### function_crypto

**描述**: 加密相關漏洞

**攻擊類型**: cryptographic

**流程數**: 4

**語言**: Rust

**流程範例**:

1. main → scan_javascript
2. main → analyze_tls
3. main → analyze_cookies

---

## SCAN

### rust_engine

**描述**: Rust 分析引擎

**攻擊類型**: language_engine

**流程數**: 4

**語言**: Rust

**流程範例**:

1. scan_single_target → EndpointDiscoverer::new
2. scan_single_target → JsAnalyzer::new
3. scan_single_target → SensitiveInfoScanner::with_mode

---

### typescript_engine

**描述**: TypeScript 分析引擎

**攻擊類型**: language_engine

**流程數**: 3

**語言**: Typescript

**流程範例**:

1. ScanService.extractNetworkAssets → NetworkInterceptor.getApiRequests
2. ScanService.extractNetworkAssets → NetworkInterceptor.getAjaxRequests
3. ScanService.extractNetworkAssets → NetworkInterceptor.analyzeRequestPatterns

---

### scan_engine

**描述**: 掃描引擎

**攻擊類型**: scanner

**流程數**: 0

**語言**: 

---

