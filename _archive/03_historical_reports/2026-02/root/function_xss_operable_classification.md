# function_xss 可操作功能分類報告

> 生成時間: 2026-01-21  
> 分析模組: function_xss  
> 可操作流程總數: 101

---

## 📊 分類概覽

| 功能類別 | 流程數 | 佔比 |
|---------|--------|------|
| 掃描管理與協調 | 3 | 3.0% |
| DOM XSS 檢測 | 2 | 2.0% |
| 反射型 XSS 檢測 | 2 | 2.0% |
| 存儲型 XSS 檢測 | 2 | 2.0% |
| 盲測 XSS 檢測 | 2 | 2.0% |
| XSS 檢測引擎 | 10 | 9.9% |
| 外部工具整合 | 2 | 2.0% |
| Payload 生成 | 6 | 5.9% |
| 回調監聽 | 4 | 4.0% |
| 任務調度 | 5 | 5.0% |
| 結果發佈與遙測 | 5 | 5.0% |
| 配置管理 | 5 | 5.0% |
| 掃描執行 | 5 | 5.0% |
| JS/前端分析 | 2 | 2.0% |
| 數據處理 | 7 | 6.9% |
| 日誌記錄 | 3 | 3.0% |
| 工具函數 | 7 | 6.9% |
| 其他工具 | 29 | 28.7% |

---

## 🎯 核心功能分類


### 掃描管理與協調 (3 個流程)

**功能說明**: 負責整體掃描流程的協調、命令處理和結果管理。這是 AI 調用的主要入口點。

**主要入口點**:

- `XSSCommandHandler` (2 個流程)
- `XSSManager` (1 個流程)

**流程示例**:

1. **aiva_features_xss_6**
   - 入口: `XSSCommandHandler.__init__`
   - 出口: `self.logger.info`
   - 路徑長度: 3 步
   - 用途: XSS testing from XSSCommandHandler.__init__ to self.logger.info

2. **aiva_features_xss_7**
   - 入口: `XSSCommandHandler.handle_command`
   - 出口: `self.logger.error`
   - 路徑長度: 11 步
   - 用途: XSS testing from XSSCommandHandler.handle_command to self.logger.error

3. **aiva_features_xss_112**
   - 入口: `XSSManager.__init__`
   - 出口: `BlindXSSDetector`
   - 路徑長度: 6 步
   - 用途: XSS testing from XSSManager.__init__ to BlindXSSDetector


### DOM XSS 檢測 (2 個流程)

**功能說明**: 檢測 DOM-based XSS 漏洞，分析 JavaScript 代碼中的客戶端注入點。

**主要入口點**:

- `DomXssDetector` (1 個流程)
- `DOMXSSDetector` (1 個流程)

**流程示例**:

1. **aiva_features_xss_10**
   - 入口: `DomXssDetector.analyze`
   - 出口: `window.strip`
   - 路徑長度: 6 步
   - 用途: XSS testing from DomXssDetector.analyze to window.strip

2. **aiva_features_xss_99**
   - 入口: `DOMXSSDetector.scan_dom_xss`
   - 出口: `vulnerabilities.extend`
   - 路徑長度: 7 步
   - 用途: XSS testing from DOMXSSDetector.scan_dom_xss to vulnerabilities.extend


### 反射型 XSS 檢測 (2 個流程)

**功能說明**: 檢測反射型 XSS 漏洞，測試輸入是否直接反射到響應中。

**主要入口點**:

- `TraditionalXssDetector` (2 個流程)

**流程示例**:

1. **aiva_features_xss_32**
   - 入口: `TraditionalXssDetector.__init__`
   - 出口: `max`
   - 路徑長度: 2 步
   - 用途: XSS testing from TraditionalXssDetector.__init__ to max

2. **aiva_features_xss_33**
   - 入口: `TraditionalXssDetector.execute`
   - 出口: `client.aclose`
   - 路徑長度: 6 步
   - 用途: XSS testing from TraditionalXssDetector.execute to client.aclose


### 存儲型 XSS 檢測 (2 個流程)

**功能說明**: 檢測存儲型 XSS 漏洞，測試 payload 是否被持久化存儲並在後續請求中觸發。

**主要入口點**:

- `StoredXssDetector` (1 個流程)
- `StoredXSSDetector` (1 個流程)

**流程示例**:

1. **aiva_features_xss_22**
   - 入口: `StoredXssDetector.execute`
   - 出口: `client.aclose`
   - 路徑長度: 6 步
   - 用途: XSS testing from StoredXssDetector.execute to client.aclose

2. **aiva_features_xss_102**
   - 入口: `StoredXSSDetector.scan_stored_xss`
   - 出口: `vulnerabilities.extend`
   - 路徑長度: 4 步
   - 用途: XSS testing from StoredXSSDetector.scan_stored_xss to vulnerabilities.extend


### 盲測 XSS 檢測 (2 個流程)

**功能說明**: 使用 OAST（Out-of-Band Application Security Testing）技術檢測盲測 XSS。

**主要入口點**:

- `BlindXSSDetector` (2 個流程)

**流程示例**:

1. **aiva_features_xss_105**
   - 入口: `BlindXSSDetector.__init__`
   - 出口: `self._generate_blind_payloads`
   - 路徑長度: 2 步
   - 用途: XSS testing from BlindXSSDetector.__init__ to self._generate_blind_payloads

2. **aiva_features_xss_106**
   - 入口: `BlindXSSDetector.scan_blind_xss`
   - 出口: `vulnerabilities.append`
   - 路徑長度: 4 步
   - 用途: XSS testing from BlindXSSDetector.scan_blind_xss to vulnerabilities.append


### XSS 檢測引擎 (10 個流程)

**功能說明**: 多語言 XSS 檢測引擎，支持多種檢測工具的統一調用。

**主要入口點**:

- `CrossLanguageXSSEngine` (1 個流程)
- `detect_xss` (1 個流程)
- `DalfoxIntegration` (1 個流程)
- `XSSManager` (1 個流程)
- `wafDetector` (1 個流程)

**流程示例**:

1. **aiva_features_xss_74**
   - 入口: `CrossLanguageXSSEngine.detect`
   - 出口: `self.logger.warning`
   - 路徑長度: 4 步
   - 用途: XSS testing from CrossLanguageXSSEngine.detect to self.logger.warning

2. **aiva_features_xss_92**
   - 入口: `detect_xss`
   - 出口: `get_xss_engine`
   - 路徑長度: 2 步
   - 用途: XSS testing from detect_xss to get_xss_engine

3. **aiva_features_xss_95**
   - 入口: `DalfoxIntegration.scan_target`
   - 出口: `self._parse_dalfox_output`
   - 路徑長度: 9 步
   - 用途: XSS testing from DalfoxIntegration.scan_target to self._parse_dalfox_output


### 外部工具整合 (2 個流程)

**功能說明**: 整合 Dalfox、XSStrike 等外部 XSS 檢測工具。

**主要入口點**:

- `DalfoxIntegration` (2 個流程)

**流程示例**:

1. **aiva_features_xss_93**
   - 入口: `DalfoxIntegration.__init__`
   - 出口: `self._find_dalfox_path`
   - 路徑長度: 2 步
   - 用途: XSS testing from DalfoxIntegration.__init__ to self._find_dalfox_path

2. **aiva_features_xss_94**
   - 入口: `DalfoxIntegration.install_dalfox`
   - 出口: `process.communicate`
   - 路徑長度: 2 步
   - 用途: XSS testing from DalfoxIntegration.install_dalfox to process.communicate


### Payload 生成 (6 個流程)

**功能說明**: 生成各種類型的 XSS payload，支持上下文感知的 payload 生成。

**主要入口點**:

- `XSSPayloadGenerator` (2 個流程)
- `BlindXssListenerValidator` (1 個流程)
- `XssPayloadGenerator` (1 個流程)
- `payloadsList` (1 個流程)
- `generator` (1 個流程)

**流程示例**:

1. **aiva_features_xss_4**
   - 入口: `BlindXssListenerValidator.provision_payload`
   - 出口: `self._store.register_probe`
   - 路徑長度: 2 步
   - 用途: XSS testing from BlindXssListenerValidator.provision_payload to self._store.register_probe

2. **aiva_features_xss_16**
   - 入口: `XssPayloadGenerator.generate`
   - 出口: `ordered.setdefault`
   - 路徑長度: 5 步
   - 用途: XSS testing from XssPayloadGenerator.generate to ordered.setdefault

3. **aiva_features_xss_97**
   - 入口: `XSSPayloadGenerator.__init__`
   - 出口: `self._load_context_specific_payloads`
   - 路徑長度: 3 步
   - 用途: XSS testing from XSSPayloadGenerator.__init__ to self._load_context_specific_payloads


### 回調監聽 (4 個流程)

**功能說明**: 管理 OAST 回調，監聽盲測 XSS 的觸發事件。

**主要入口點**:

- `OastHttpCallbackStore` (2 個流程)
- `BlindXssListenerValidator` (2 個流程)

**流程示例**:

1. **aiva_features_xss_1**
   - 入口: `OastHttpCallbackStore.register_probe`
   - 出口: `isinstance`
   - 路徑長度: 11 步
   - 用途: XSS testing from OastHttpCallbackStore.register_probe to isinstance

2. **aiva_features_xss_2**
   - 入口: `OastHttpCallbackStore.fetch_events`
   - 出口: `events.append`
   - 路徑長度: 9 步
   - 用途: XSS testing from OastHttpCallbackStore.fetch_events to events.append

3. **aiva_features_xss_3**
   - 入口: `BlindXssListenerValidator.__init__`
   - 出口: `OastHttpCallbackStore`
   - 路徑長度: 3 步
   - 用途: XSS testing from BlindXssListenerValidator.__init__ to OastHttpCallbackStore


### 任務調度 (5 個流程)

**功能說明**: 管理 XSS 檢測任務的佇列和執行。

**主要入口點**:

- `XssTaskQueue` (4 個流程)
- `XssWorkerService` (1 個流程)

**流程示例**:

1. **aiva_features_xss_27**
   - 入口: `XssTaskQueue.__init__`
   - 出口: `itertools.count`
   - 路徑長度: 2 步
   - 用途: XSS testing from XssTaskQueue.__init__ to itertools.count

2. **aiva_features_xss_28**
   - 入口: `XssTaskQueue.put`
   - 出口: `self._condition.notify_all`
   - 路徑長度: 9 步
   - 用途: XSS testing from XssTaskQueue.put to self._condition.notify_all

3. **aiva_features_xss_29**
   - 入口: `XssTaskQueue.get`
   - 出口: `self._entries.pop`
   - 路徑長度: 10 步
   - 用途: XSS testing from XssTaskQueue.get to self._entries.pop


### 結果發佈與遙測 (5 個流程)

**功能說明**: 發佈檢測結果，收集遙測數據。

**主要入口點**:

- `XssResultPublisher` (4 個流程)
- `XssExecutionTelemetry` (1 個流程)

**流程示例**:

1. **aiva_features_xss_17**
   - 入口: `XssResultPublisher.__init__`
   - 出口: `new_id`
   - 路徑長度: 2 步
   - 用途: XSS testing from XssResultPublisher.__init__ to new_id

2. **aiva_features_xss_18**
   - 入口: `XssResultPublisher.publish_status`
   - 出口: `self._publish`
   - 路徑長度: 4 步
   - 用途: XSS testing from XssResultPublisher.publish_status to self._publish

3. **aiva_features_xss_19**
   - 入口: `XssResultPublisher.publish_finding`
   - 出口: `self._publish`
   - 路徑長度: 3 步
   - 用途: XSS testing from XssResultPublisher.publish_finding to self._publish


### 配置管理 (5 個流程)

**功能說明**: 管理檢測工具的配置和優先級。

**主要入口點**:

- `HackingToolXSSConfig` (4 個流程)
- `setup_logger` (1 個流程)

**流程示例**:

1. **aiva_features_xss_11**
   - 入口: `HackingToolXSSConfig.__init__`
   - 出口: `self._calculate_priority_order`
   - 路徑長度: 3 步
   - 用途: XSS testing from HackingToolXSSConfig.__init__ to self._calculate_priority_order

2. **aiva_features_xss_13**
   - 入口: `HackingToolXSSConfig.validate_tool_requirements`
   - 出口: `self.get_tool_config`
   - 路徑長度: 2 步
   - 用途: XSS testing from HackingToolXSSConfig.validate_tool_requirements to self.get_tool_config

3. **aiva_features_xss_14**
   - 入口: `HackingToolXSSConfig.export_config`
   - 出口: `print`
   - 路徑長度: 3 步
   - 用途: XSS testing from HackingToolXSSConfig.export_config to print


### 掃描執行 (5 個流程)

**功能說明**: 執行具體的掃描操作。

**主要入口點**:

- `run` (1 個流程)
- `run_reflected_test` (1 個流程)
- `run_dom_test` (1 個流程)
- `run_stored_test` (1 個流程)
- `run_xss_test` (1 個流程)

**流程示例**:

1. **aiva_features_xss_40**
   - 入口: `run`
   - 出口: `queue.close`
   - 路徑長度: 9 步
   - 用途: XSS testing from run to queue.close

2. **aiva_features_xss_58**
   - 入口: `run_reflected_test`
   - 出口: `detector.execute`
   - 路徑長度: 6 步
   - 用途: XSS testing from run_reflected_test to detector.execute

3. **aiva_features_xss_59**
   - 入口: `run_dom_test`
   - 出口: `detector.analyze`
   - 路徑長度: 4 步
   - 用途: XSS testing from run_dom_test to detector.analyze


### JS/前端分析 (2 個流程)

**功能說明**: 分析 JavaScript 代碼和前端資源。

**主要入口點**:

- `jsContexter` (1 個流程)
- `retireJs` (1 個流程)

**流程示例**:

1. **aiva_features_xss_132**
   - 入口: `jsContexter`
   - 出口: `script.split`
   - 路徑長度: 2 步
   - 用途: XSS testing from jsContexter to script.split

2. **aiva_features_xss_174**
   - 入口: `retireJs`
   - 出口: `main_scanner`
   - 路徑長度: 6 步
   - 用途: XSS testing from retireJs to main_scanner


### 數據處理 (7 個流程)

**功能說明**: 處理和解析檢測過程中的數據。

**主要入口點**:

- `checker` (1 個流程)
- `filterChecker` (1 個流程)
- `htmlParser` (1 個流程)
- `extractHeaders` (1 個流程)
- `extractScripts` (1 個流程)

**流程示例**:

1. **aiva_features_xss_126**
   - 入口: `checker`
   - 出口: `efficiencies.append`
   - 路徑長度: 11 步
   - 用途: XSS testing from checker to efficiencies.append

2. **aiva_features_xss_128**
   - 入口: `filterChecker`
   - 出口: `efficiencies.extend`
   - 路徑長度: 11 步
   - 用途: XSS testing from filterChecker to efficiencies.extend

3. **aiva_features_xss_131**
   - 入口: `htmlParser`
   - 出口: `occurence.start`
   - 路徑長度: 11 步
   - 用途: XSS testing from htmlParser to occurence.start


### 日誌記錄 (3 個流程)

**功能說明**: 記錄檢測過程的日誌信息。

**主要入口點**:

- `log_red_line` (1 個流程)
- `log_no_format` (1 個流程)
- `log_debug_json` (1 個流程)

**流程示例**:

1. **aiva_features_xss_139**
   - 入口: `log_red_line`
   - 出口: `_switch_to_default_loggers`
   - 路徑長度: 4 步
   - 用途: XSS testing from log_red_line to _switch_to_default_loggers

2. **aiva_features_xss_140**
   - 入口: `log_no_format`
   - 出口: `_switch_to_default_loggers`
   - 路徑長度: 4 步
   - 用途: XSS testing from log_no_format to _switch_to_default_loggers

3. **aiva_features_xss_141**
   - 入口: `log_debug_json`
   - 出口: `self.debug`
   - 路徑長度: 4 步
   - 用途: XSS testing from log_debug_json to self.debug


### 工具函數 (7 個流程)

**功能說明**: 提供各種輔助功能的工具函數。

**主要入口點**:

- `photon` (1 個流程)
- `prompt` (1 個流程)
- `requester` (1 個流程)
- `updater` (1 個流程)
- `converter` (1 個流程)

**流程示例**:

1. **aiva_features_xss_143**
   - 入口: `photon`
   - 出口: `threadpool.submit`
   - 路徑長度: 6 步
   - 用途: XSS testing from photon to threadpool.submit

2. **aiva_features_xss_144**
   - 入口: `prompt`
   - 出口: `tmpfile.seek`
   - 路徑長度: 6 步
   - 用途: XSS testing from prompt to tmpfile.seek

3. **aiva_features_xss_145**
   - 入口: `requester`
   - 出口: `requests.post`
   - 路徑長度: 8 步
   - 用途: XSS testing from requester to requests.post


### 其他工具 (29 個流程)

**主要入口點**:

- `CrossLanguageXSSEngine` (5 個流程)
- `process_task` (1 個流程)
- `main` (1 個流程)
- `get_xss_engine` (1 個流程)
- `get_user_agent` (1 個流程)

**流程示例**:

1. **aiva_features_xss_43**
   - 入口: `process_task`
   - 出口: `_process_detections`
   - 路徑長度: 11 步
   - 用途: XSS testing from process_task to _process_detections

2. **aiva_features_xss_61**
   - 入口: `main`
   - 出口: `print`
   - 路徑長度: 10 步
   - 用途: XSS testing from main to print

3. **aiva_features_xss_63**
   - 入口: `CrossLanguageXSSEngine.__init__`
   - 出口: `Path`
   - 路徑長度: 4 步
   - 用途: XSS testing from CrossLanguageXSSEngine.__init__ to Path


---

## 🚀 AI 調用建議

### 主要入口點

根據不同的檢測需求，AI 可以選擇以下入口點：

#### 1. 綜合掃描
```python
# 使用 XSSManager 進行綜合掃描
from services.features.function_xss.integration_tools.xss_tools import XSSManager

manager = XSSManager()
result = manager.comprehensive_scan(
    target_url="https://example.com/search?q=test",
    options={
        "scan_type": "all",  # 執行所有類型的檢測
        "timeout": 30,
        "max_payloads": 100
    }
)
```

#### 2. 特定類型檢測
```python
# DOM XSS 檢測
from services.features.function_xss.dom_xss_detector import DOMXSSDetector

detector = DOMXSSDetector()
result = detector.scan_dom_xss(target_url, javascript_code)

# 存儲型 XSS 檢測
from services.features.function_xss.stored_xss_detector import StoredXSSDetector

detector = StoredXSSDetector()
result = detector.scan_stored_xss(target_url, injection_points)

# 盲測 XSS 檢測
from services.features.function_xss.blind_xss_detector import BlindXSSDetector

detector = BlindXSSDetector()
result = detector.scan_blind_xss(target_url, callback_url)
```

#### 3. 外部工具整合
```python
# 使用 Dalfox
from services.features.function_xss.dalfox_integration import DalfoxIntegration

dalfox = DalfoxIntegration()
result = dalfox.scan_target(target_url)
```

### 調用流程建議

1. **快速掃描**: 使用 `detect_xss()` 統一入口
2. **深度掃描**: 使用 `XSSManager.comprehensive_scan()`
3. **特定檢測**: 根據需求選擇對應的 Detector
4. **批量掃描**: 使用 `XssTaskQueue` 進行任務調度

---

## 📈 可操作性分析

### 可操作流程特徵

✅ **可操作的流程具有以下特徵**:
- 公開函數（非 `_` 開頭）
- 不依賴 Web Context（request/session 對象）
- 參數為基本類型或可序列化類型
- 有明確的輸入輸出定義

### 不可操作原因統計

| 原因 | 數量 |
|------|------|
| 私有函數（`_` 開頭）| 64 |
| Web Context 依賴 | 9 |

---

## 📋 附錄：完整流程清單

### 按類別列出所有可操作流程


#### 掃描管理與協調

1. `aiva_features_xss_6`: XSSCommandHandler.__init__ → self.logger.info (3 步)
2. `aiva_features_xss_7`: XSSCommandHandler.handle_command → self.logger.error (11 步)
3. `aiva_features_xss_112`: XSSManager.__init__ → BlindXSSDetector (6 步)

#### DOM XSS 檢測

1. `aiva_features_xss_10`: DomXssDetector.analyze → window.strip (6 步)
2. `aiva_features_xss_99`: DOMXSSDetector.scan_dom_xss → vulnerabilities.extend (7 步)

#### 反射型 XSS 檢測

1. `aiva_features_xss_32`: TraditionalXssDetector.__init__ → max (2 步)
2. `aiva_features_xss_33`: TraditionalXssDetector.execute → client.aclose (6 步)

#### 存儲型 XSS 檢測

1. `aiva_features_xss_22`: StoredXssDetector.execute → client.aclose (6 步)
2. `aiva_features_xss_102`: StoredXSSDetector.scan_stored_xss → vulnerabilities.extend (4 步)

#### 盲測 XSS 檢測

1. `aiva_features_xss_105`: BlindXSSDetector.__init__ → self._generate_blind_payloads (2 步)
2. `aiva_features_xss_106`: BlindXSSDetector.scan_blind_xss → vulnerabilities.append (4 步)

#### XSS 檢測引擎

1. `aiva_features_xss_74`: CrossLanguageXSSEngine.detect → self.logger.warning (4 步)
2. `aiva_features_xss_92`: detect_xss → get_xss_engine (2 步)
3. `aiva_features_xss_95`: DalfoxIntegration.scan_target → self._parse_dalfox_output (9 步)
4. `aiva_features_xss_114`: XSSManager.comprehensive_scan → self.scan_results.extend (11 步)
5. `aiva_features_xss_160`: wafDetector → bestMatch.extend (4 步)
6. `aiva_features_xss_164`: scan → detected.append (3 步)
7. `aiva_features_xss_170`: scan_uri → scan (2 步)
8. `aiva_features_xss_171`: scan_filename → scan (2 步)
9. `aiva_features_xss_172`: scan_file_content → _scanhash (4 步)
10. `aiva_features_xss_173`: main_scanner → vulnerabilities.add (8 步)

#### 外部工具整合

1. `aiva_features_xss_93`: DalfoxIntegration.__init__ → self._find_dalfox_path (2 步)
2. `aiva_features_xss_94`: DalfoxIntegration.install_dalfox → process.communicate (2 步)

#### Payload 生成

1. `aiva_features_xss_4`: BlindXssListenerValidator.provision_payload → self._store.register_probe (2 步)
2. `aiva_features_xss_16`: XssPayloadGenerator.generate → ordered.setdefault (5 步)
3. `aiva_features_xss_97`: XSSPayloadGenerator.__init__ → self._load_context_specific_payloads (3 步)
4. `aiva_features_xss_98`: XSSPayloadGenerator.generate_payloads → payloads.extend (4 步)
5. `aiva_features_xss_120`: payloadsList → f.read (9 步)
6. `aiva_features_xss_130`: generator → set (11 步)

#### 回調監聽

1. `aiva_features_xss_1`: OastHttpCallbackStore.register_probe → isinstance (11 步)
2. `aiva_features_xss_2`: OastHttpCallbackStore.fetch_events → events.append (9 步)
3. `aiva_features_xss_3`: BlindXssListenerValidator.__init__ → OastHttpCallbackStore (3 步)
4. `aiva_features_xss_5`: BlindXssListenerValidator.collect_events → self._store.fetch_events (2 步)

#### 任務調度

1. `aiva_features_xss_27`: XssTaskQueue.__init__ → itertools.count (2 步)
2. `aiva_features_xss_28`: XssTaskQueue.put → self._condition.notify_all (9 步)
3. `aiva_features_xss_29`: XssTaskQueue.get → self._entries.pop (10 步)
4. `aiva_features_xss_30`: XssTaskQueue.close → self._condition.notify_all (2 步)
5. `aiva_features_xss_57`: XssWorkerService.process_task → ValueError (3 步)

#### 結果發佈與遙測

1. `aiva_features_xss_17`: XssResultPublisher.__init__ → new_id (2 步)
2. `aiva_features_xss_18`: XssResultPublisher.publish_status → self._publish (4 步)
3. `aiva_features_xss_19`: XssResultPublisher.publish_finding → self._publish (3 步)
4. `aiva_features_xss_20`: XssResultPublisher.publish_error → self.publish_status (2 步)
5. `aiva_features_xss_39`: XssExecutionTelemetry → field (2 步)

#### 配置管理

1. `aiva_features_xss_11`: HackingToolXSSConfig.__init__ → self._calculate_priority_order (3 步)
2. `aiva_features_xss_13`: HackingToolXSSConfig.validate_tool_requirements → self.get_tool_config (2 步)
3. `aiva_features_xss_14`: HackingToolXSSConfig.export_config → print (3 步)
4. `aiva_features_xss_15`: HackingToolXSSConfig.get_execution_plan → execution_plan.append (4 步)
5. `aiva_features_xss_142`: setup_logger → file_handler.setLevel (11 步)

#### 掃描執行

1. `aiva_features_xss_40`: run → queue.close (9 步)
2. `aiva_features_xss_58`: run_reflected_test → detector.execute (6 步)
3. `aiva_features_xss_59`: run_dom_test → detector.analyze (4 步)
4. `aiva_features_xss_60`: run_stored_test → detector.execute (6 步)
5. `aiva_features_xss_62`: run_xss_test → findings.append (4 步)

#### JS/前端分析

1. `aiva_features_xss_132`: jsContexter → script.split (2 步)
2. `aiva_features_xss_174`: retireJs → main_scanner (6 步)

#### 數據處理

1. `aiva_features_xss_126`: checker → efficiencies.append (11 步)
2. `aiva_features_xss_128`: filterChecker → efficiencies.extend (11 步)
3. `aiva_features_xss_131`: htmlParser → occurence.start (11 步)
4. `aiva_features_xss_150`: extractHeaders → headers.replace (2 步)
5. `aiva_features_xss_151`: extractScripts → scripts.append (2 步)
6. `aiva_features_xss_157`: js_extractor → scripts.append (2 步)
7. `aiva_features_xss_168`: check → result.get (2 步)

#### 日誌記錄

1. `aiva_features_xss_139`: log_red_line → _switch_to_default_loggers (4 步)
2. `aiva_features_xss_140`: log_no_format → _switch_to_default_loggers (4 步)
3. `aiva_features_xss_141`: log_debug_json → self.debug (4 步)

#### 工具函數

1. `aiva_features_xss_143`: photon → threadpool.submit (6 步)
2. `aiva_features_xss_144`: prompt → tmpfile.seek (6 步)
3. `aiva_features_xss_145`: requester → requests.post (8 步)
4. `aiva_features_xss_146`: updater → get (2 步)
5. `aiva_features_xss_147`: converter → data.split (2 步)
6. `aiva_features_xss_155`: writer → savefile.close (4 步)
7. `aiva_features_xss_156`: reader → open (2 步)

#### 其他工具

1. `aiva_features_xss_43`: process_task → _process_detections (11 步)
2. `aiva_features_xss_61`: main → print (10 步)
3. `aiva_features_xss_63`: CrossLanguageXSSEngine.__init__ → Path (4 步)
4. `aiva_features_xss_64`: CrossLanguageXSSEngine.initialize → self.logger.error (6 步)
5. `aiva_features_xss_88`: CrossLanguageXSSEngine.get_available_tools → available_tools.append (2 步)
6. `aiva_features_xss_89`: CrossLanguageXSSEngine.cleanup → self.logger.error (4 步)
7. `aiva_features_xss_90`: CrossLanguageXSSEngine.__del__ → self.cleanup (2 步)
8. `aiva_features_xss_91`: get_xss_engine → _xss_engine_instance.initialize (3 步)
9. `aiva_features_xss_117`: get_user_agent → print (4 步)
10. `aiva_features_xss_118`: dorkFind → f.write (9 步)
11. `aiva_features_xss_119`: entryy → sleep (11 步)
12. `aiva_features_xss_121`: pylds → print (4 步)
13. `aiva_features_xss_122`: islem → print (11 步)
14. `aiva_features_xss_123`: Menu → print (11 步)
15. `aiva_features_xss_124`: proxy_lister → file.writelines (11 步)
16. `aiva_features_xss_125`: xssFind → print (11 步)
17. `aiva_features_xss_127`: dom → highlighted.append (11 步)
18. `aiva_features_xss_129`: fuzzer → encoding (10 步)
19. `aiva_features_xss_148`: closest → abs (3 步)
20. `aiva_features_xss_149`: fillHoles → filled.extend (3 步)
21. `aiva_features_xss_152`: flattenParams → flatted.append (2 步)
22. `aiva_features_xss_153`: genGen → vectors.append (5 步)
23. `aiva_features_xss_154`: getParams → each.append (5 步)
24. `aiva_features_xss_158`: equalize → array.append (2 步)
25. `aiva_features_xss_159`: escaped → match.group (2 步)
26. `aiva_features_xss_161`: zetanize → d (6 步)
27. `aiva_features_xss_162`: bruteforcer → encoding (8 步)
28. `aiva_features_xss_163`: crawl → requester (8 步)
29. `aiva_features_xss_165`: singleFuzz → fuzzer (8 步)


---

## 🔍 使用建議

### 選擇合適的檢測方法

1. **不確定漏洞類型**: 使用 `XSSManager.comprehensive_scan()` 進行全面掃描
2. **已知是 DOM XSS**: 直接使用 `DOMXSSDetector`
3. **需要檢測存儲型**: 使用 `StoredXSSDetector`
4. **目標有 WAF 保護**: 考慮使用盲測方法 `BlindXSSDetector`
5. **快速驗證**: 使用 `detect_xss()` 快速檢測

### 性能優化建議

- 對於大批量掃描，使用 `XssTaskQueue` 進行任務調度
- 設置合理的 timeout 避免長時間等待
- 根據目標特性選擇合適的 payload 數量
- 使用外部工具（Dalfox）可以提高檢測效率

---

*本報告由 AIVA 自動生成*
