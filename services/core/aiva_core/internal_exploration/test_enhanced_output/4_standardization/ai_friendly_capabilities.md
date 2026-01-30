# AIVA 模組化能力文檔（AI友善版本）

生成時間: 2026-01-24T17:18:14.562348

## 文檔說明

此文檔按照以下優先級組織：
1. **模組分類**（最高優先級）
2. **相同起終點能力集中**
3. **內部/外部能力區分**
4. **AI可讀描述欄位**

## 認知核心模組

**模組統計**:
- 總能力數量: 1
- 內部能力: 1
- 外部能力: 0

### photon → threadpool.submit

**AI描述欄位 📋**:
- **能力概要**: photon到threadpool.submit的處理能力
- **使用時機**: 當需要對photon進行深度分析並生成threadpool.submit結果時
- **預期結果**: 獲得部分AI輔助的threadpool.submit結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 585: 認知：Threadpool.Submit

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行認知：Threadpool.Submit相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 585
```
```bash
aiva_internal_executor.py --flow 585 --dry-run
```

---

## 內探模組

**模組統計**:
- 總能力數量: 67
- 內部能力: 67
- 外部能力: 0

### BackendDbFingerprinter.analyze_response_characteristics → self._extract_error_signatures

**AI描述欄位 📋**:
- **能力概要**: BackendDbFingerprinter.analyze_response_characteristics到self._extract_error_signatures的處理能力
- **使用時機**: 當需要對BackendDbFingerprinter.analyze_response_characteristics進行深度分析並生成self._extract_error_signatures結果時
- **預期結果**: 獲得基於程式邏輯的self._extract_error_signatures結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 153: 內探：Self. Extract Error Signatures

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Self. Extract Error Signatures相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 153
```
```bash
aiva_internal_executor.py --flow 153 --dry-run
```

---

### BooleanDetectionEngine._analyze_boolean_responses → abs

**AI描述欄位 📋**:
- **能力概要**: BooleanDetectionEngine._analyze_boolean_responses到abs的處理能力
- **使用時機**: 當需要對BooleanDetectionEngine._analyze_boolean_responses進行深度分析並生成abs結果時
- **預期結果**: 獲得基於程式邏輯的abs結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 219: 內探：Abs

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Abs相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 219
```
```bash
aiva_internal_executor.py --flow 219 --dry-run
```

---

### BountyHunterScanner._analyze_bounty_response → BountyVulnerability

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner._analyze_bounty_response到BountyVulnerability的處理能力
- **使用時機**: 當需要對BountyHunterScanner._analyze_bounty_response進行深度分析並生成BountyVulnerability結果時
- **預期結果**: 獲得部分AI輔助的BountyVulnerability結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 250: 內探：Bountyvulnerability

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行內探：Bountyvulnerability相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 250
```
```bash
aiva_internal_executor.py --flow 250 --dry-run
```

---

### CustomSQLInjectionScanner._analyze_response → abs

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner._analyze_response到abs的處理能力
- **使用時機**: 當需要快速從CustomSQLInjectionScanner._analyze_response獲取abs的基礎信息時
- **預期結果**: 獲得部分AI輔助的abs結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 277: 內探：Abs

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供內探：Abs的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 277
```
```bash
aiva_internal_executor.py --flow 277 --dry-run
```

---

### DOMXSSDetector._analyze_javascript → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: DOMXSSDetector._analyze_javascript到vulnerabilities.append的處理能力
- **使用時機**: 當需要對DOMXSSDetector._analyze_javascript進行深度分析並生成vulnerabilities.append結果時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 542: 內探：Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 542
```
```bash
aiva_internal_executor.py --flow 542 --dry-run
```

---

### DOMXSSDetector.scan_dom_xss → vulnerabilities.extend

**AI描述欄位 📋**:
- **能力概要**: DOMXSSDetector.scan_dom_xss到vulnerabilities.extend的處理能力
- **使用時機**: 當需要對DOMXSSDetector.scan_dom_xss進行深度分析並生成vulnerabilities.extend結果時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 541: 內探：Vulnerabilities.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Vulnerabilities.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 541
```
```bash
aiva_internal_executor.py --flow 541 --dry-run
```

---

### DnsRebindingDetector.verify_internal_access → client.aclose

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector.verify_internal_access到client.aclose的處理能力
- **使用時機**: 當需要對DnsRebindingDetector.verify_internal_access進行深度分析並生成client.aclose結果時
- **預期結果**: 獲得基於程式邏輯的client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 302: 內探：Client.Aclose

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Client.Aclose相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 302
```
```bash
aiva_internal_executor.py --flow 302 --dry-run
```

---

### DomXssDetector.analyze → window.strip

**AI描述欄位 📋**:
- **能力概要**: DomXssDetector.analyze到window.strip的處理能力
- **使用時機**: 當需要對DomXssDetector.analyze進行深度分析並生成window.strip結果時
- **預期結果**: 獲得基於程式邏輯的window.strip結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 452: 內探：Window.Strip

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Window.Strip相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 452
```
```bash
aiva_internal_executor.py --flow 452 --dry-run
```

---

### EnhancedPrivilegeAnalyzer.analyze_system_permissions → datetime.now

**AI描述欄位 📋**:
- **能力概要**: EnhancedPrivilegeAnalyzer.analyze_system_permissions到datetime.now的處理能力
- **使用時機**: 當需要對EnhancedPrivilegeAnalyzer.analyze_system_permissions進行深度分析並生成datetime.now結果時
- **預期結果**: 獲得基於程式邏輯的datetime.now結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 134: 內探：Datetime.Now

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Datetime.Now相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 134
```
```bash
aiva_internal_executor.py --flow 134 --dry-run
```

---

### ErrorDetectionEngine.detect → results.append

**AI描述欄位 📋**:
- **能力概要**: ErrorDetectionEngine.detect到results.append的處理能力
- **使用時機**: 當需要對ErrorDetectionEngine.detect進行深度分析並生成results.append結果時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 221: 內探：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 221
```
```bash
aiva_internal_executor.py --flow 221 --dry-run
```

---

### IDORDetector.analyze → engine.close

**AI描述欄位 📋**:
- **能力概要**: IDORDetector.analyze到engine.close的處理能力
- **使用時機**: 當需要對IDORDetector.analyze進行深度分析並生成engine.close結果時
- **預期結果**: 獲得基於程式邏輯的engine.close結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 89: 內探：Engine.Close

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Engine.Close相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 89
```
```bash
aiva_internal_executor.py --flow 89 --dry-run
```

---

### InternalAddressDetection → field

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetection到field的處理能力
- **使用時機**: 當需要快速從InternalAddressDetection獲取field的基礎信息時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 304: 內探：Field

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 304
```
```bash
aiva_internal_executor.py --flow 304 --dry-run
```

---

### InternalAddressDetector → field

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector到field的處理能力
- **使用時機**: 當需要對InternalAddressDetector進行深度分析並生成field結果時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 305: 內探：Field

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 305
```
```bash
aiva_internal_executor.py --flow 305 --dry-run
```

---

### InternalAddressDetector._generate_evidence → evidence.append

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._generate_evidence到evidence.append的處理能力
- **使用時機**: 當需要對InternalAddressDetector._generate_evidence進行深度分析並生成evidence.append結果時
- **預期結果**: 獲得基於程式邏輯的evidence.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 313: 內探：Evidence.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Evidence.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 313
```
```bash
aiva_internal_executor.py --flow 313 --dry-run
```

---

### InternalAddressDetector._is_metadata_response → response.lower

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._is_metadata_response到response.lower的處理能力
- **使用時機**: 當需要快速從InternalAddressDetector._is_metadata_response獲取response.lower的基礎信息時
- **預期結果**: 獲得基於程式邏輯的response.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 310: 內探：Response.Lower

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Response.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 310
```
```bash
aiva_internal_executor.py --flow 310 --dry-run
```

---

### InternalAddressDetector._is_protocol_supported → response.lower

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._is_protocol_supported到response.lower的處理能力
- **使用時機**: 當需要快速從InternalAddressDetector._is_protocol_supported獲取response.lower的基礎信息時
- **預期結果**: 獲得基於程式邏輯的response.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 312: 內探：Response.Lower

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Response.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 312
```
```bash
aiva_internal_executor.py --flow 312 --dry-run
```

---

### InternalAddressDetector._is_service_response → response.lower

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._is_service_response到response.lower的處理能力
- **使用時機**: 當需要快速從InternalAddressDetector._is_service_response獲取response.lower的基礎信息時
- **預期結果**: 獲得基於程式邏輯的response.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 311: 內探：Response.Lower

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Response.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 311
```
```bash
aiva_internal_executor.py --flow 311 --dry-run
```

---

### InternalAddressDetector._is_successful_response → response.lower

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._is_successful_response到response.lower的處理能力
- **使用時機**: 當需要快速從InternalAddressDetector._is_successful_response獲取response.lower的基礎信息時
- **預期結果**: 獲得基於程式邏輯的response.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 309: 內探：Response.Lower

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Response.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 309
```
```bash
aiva_internal_executor.py --flow 309 --dry-run
```

---

### InternalAddressDetector._test_internal_services → detected_services.append

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._test_internal_services到detected_services.append的處理能力
- **使用時機**: 當需要對InternalAddressDetector._test_internal_services進行深度分析並生成detected_services.append結果時
- **預期結果**: 獲得基於程式邏輯的detected_services.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 307: 內探：Detected Services.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Detected Services.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 307
```
```bash
aiva_internal_executor.py --flow 307 --dry-run
```

---

### InternalAddressDetector._test_protocol_support → supported_protocols.append

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector._test_protocol_support到supported_protocols.append的處理能力
- **使用時機**: 當需要快速從InternalAddressDetector._test_protocol_support獲取supported_protocols.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的supported_protocols.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 308: 內探：Supported Protocols.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Supported Protocols.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 308
```
```bash
aiva_internal_executor.py --flow 308 --dry-run
```

---

### InternalAddressDetector.analyze → indicators.append

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector.analyze到indicators.append的處理能力
- **使用時機**: 當需要對InternalAddressDetector.analyze進行深度分析並生成indicators.append結果時
- **預期結果**: 獲得基於程式邏輯的indicators.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 306: 內探：Indicators.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Indicators.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 306
```
```bash
aiva_internal_executor.py --flow 306 --dry-run
```

---

### InternalAddressDetector.is_internal_address → ipaddress.ip_address

**AI描述欄位 📋**:
- **能力概要**: InternalAddressDetector.is_internal_address到ipaddress.ip_address的處理能力
- **使用時機**: 當需要快速從InternalAddressDetector.is_internal_address獲取ipaddress.ip_address的基礎信息時
- **預期結果**: 獲得基於程式邏輯的ipaddress.ip_address結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 314: 內探：Ipaddress.Ip Address

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Ipaddress.Ip Address的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 314
```
```bash
aiva_internal_executor.py --flow 314 --dry-run
```

---

### ParamSemanticsAnalyzer._add_cross_protocol_vectors → plan.vectors.append

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._add_cross_protocol_vectors到plan.vectors.append的處理能力
- **使用時機**: 當需要對ParamSemanticsAnalyzer._add_cross_protocol_vectors進行深度分析並生成plan.vectors.append結果時
- **預期結果**: 獲得部分AI輔助的plan.vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 327: 內探：Plan.Vectors.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行內探：Plan.Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 327
```
```bash
aiva_internal_executor.py --flow 327 --dry-run
```

---

### ParamSemanticsAnalyzer._add_file_vectors → plan.vectors.append

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._add_file_vectors到plan.vectors.append的處理能力
- **使用時機**: 當需要快速從ParamSemanticsAnalyzer._add_file_vectors獲取plan.vectors.append的基礎信息時
- **預期結果**: 獲得部分AI輔助的plan.vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 325: 內探：Plan.Vectors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供內探：Plan.Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 325
```
```bash
aiva_internal_executor.py --flow 325 --dry-run
```

---

### ParamSemanticsAnalyzer._add_oast_vector → plan.vectors.append

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._add_oast_vector到plan.vectors.append的處理能力
- **使用時機**: 當需要快速從ParamSemanticsAnalyzer._add_oast_vector獲取plan.vectors.append的基礎信息時
- **預期結果**: 獲得部分AI輔助的plan.vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 329: 內探：Plan.Vectors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供內探：Plan.Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 329
```
```bash
aiva_internal_executor.py --flow 329 --dry-run
```

---

### ParamSemanticsAnalyzer._add_protocol_vectors → plan.vectors.append

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._add_protocol_vectors到plan.vectors.append的處理能力
- **使用時機**: 當需要快速從ParamSemanticsAnalyzer._add_protocol_vectors獲取plan.vectors.append的基礎信息時
- **預期結果**: 獲得部分AI輔助的plan.vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 326: 內探：Plan.Vectors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供內探：Plan.Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 326
```
```bash
aiva_internal_executor.py --flow 326 --dry-run
```

---

### ParamSemanticsAnalyzer._add_semantic_vectors → self._add_protocol_vectors

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._add_semantic_vectors到self._add_protocol_vectors的處理能力
- **使用時機**: 當需要對ParamSemanticsAnalyzer._add_semantic_vectors進行深度分析並生成self._add_protocol_vectors結果時
- **預期結果**: 獲得部分AI輔助的self._add_protocol_vectors結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 324: 內探：Self. Add Protocol Vectors

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行內探：Self. Add Protocol Vectors相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 324
```
```bash
aiva_internal_executor.py --flow 324 --dry-run
```

---

### ParamSemanticsAnalyzer._add_standard_vectors → plan.vectors.append

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._add_standard_vectors到plan.vectors.append的處理能力
- **使用時機**: 當需要快速從ParamSemanticsAnalyzer._add_standard_vectors獲取plan.vectors.append的基礎信息時
- **預期結果**: 獲得部分AI輔助的plan.vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 323: 內探：Plan.Vectors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供內探：Plan.Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 323
```
```bash
aiva_internal_executor.py --flow 323 --dry-run
```

---

### ParamSemanticsAnalyzer._build_payloads → seen.add

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._build_payloads到seen.add的處理能力
- **使用時機**: 當需要對ParamSemanticsAnalyzer._build_payloads進行深度分析並生成seen.add結果時
- **預期結果**: 獲得基於程式邏輯的seen.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 330: 內探：Seen.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Seen.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 330
```
```bash
aiva_internal_executor.py --flow 330 --dry-run
```

---

### ParamSemanticsAnalyzer._get_advanced_payloads → advanced.extend

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._get_advanced_payloads到advanced.extend的處理能力
- **使用時機**: 當需要對ParamSemanticsAnalyzer._get_advanced_payloads進行深度分析並生成advanced.extend結果時
- **預期結果**: 獲得基於程式邏輯的advanced.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 322: 內探：Advanced.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Advanced.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 322
```
```bash
aiva_internal_executor.py --flow 322 --dry-run
```

---

### ParamSemanticsAnalyzer._get_base_payloads → payloads.extend

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._get_base_payloads到payloads.extend的處理能力
- **使用時機**: 當需要快速從ParamSemanticsAnalyzer._get_base_payloads獲取payloads.extend的基礎信息時
- **預期結果**: 獲得基於程式邏輯的payloads.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 321: 內探：Payloads.Extend

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Payloads.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 321
```
```bash
aiva_internal_executor.py --flow 321 --dry-run
```

---

### ParamSemanticsAnalyzer._get_selected_protocols → headers.get

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._get_selected_protocols到headers.get的處理能力
- **使用時機**: 當需要快速從ParamSemanticsAnalyzer._get_selected_protocols獲取headers.get的基礎信息時
- **預期結果**: 獲得基於程式邏輯的headers.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 328: 內探：Headers.Get

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Headers.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 328
```
```bash
aiva_internal_executor.py --flow 328 --dry-run
```

---

### ParamSemanticsAnalyzer._should_enable_oast → payload_sources.extend

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer._should_enable_oast到payload_sources.extend的處理能力
- **使用時機**: 當需要對ParamSemanticsAnalyzer._should_enable_oast進行深度分析並生成payload_sources.extend結果時
- **預期結果**: 獲得基於程式邏輯的payload_sources.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 331: 內探：Payload Sources.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Payload Sources.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 331
```
```bash
aiva_internal_executor.py --flow 331 --dry-run
```

---

### ParamSemanticsAnalyzer.analyze → self._add_oast_vector

**AI描述欄位 📋**:
- **能力概要**: ParamSemanticsAnalyzer.analyze到self._add_oast_vector的處理能力
- **使用時機**: 當需要對ParamSemanticsAnalyzer.analyze進行深度分析並生成self._add_oast_vector結果時
- **預期結果**: 獲得部分AI輔助的self._add_oast_vector結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 320: 內探：Self. Add Oast Vector

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行內探：Self. Add Oast Vector相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 320
```
```bash
aiva_internal_executor.py --flow 320 --dry-run
```

---

### PassiveAnalyzer.__init__ → self._setup_patterns

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer.__init__到self._setup_patterns的處理能力
- **使用時機**: 當需要快速從PassiveAnalyzer.__init__獲取self._setup_patterns的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._setup_patterns結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 627: 內探：Self. Setup Patterns

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Self. Setup Patterns的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 627
```
```bash
aiva_internal_executor.py --flow 627 --dry-run
```

---

### PassiveAnalyzer._analyze_cookies → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._analyze_cookies到findings.append的處理能力
- **使用時機**: 當需要對PassiveAnalyzer._analyze_cookies進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 635: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 635
```
```bash
aiva_internal_executor.py --flow 635 --dry-run
```

---

### PassiveAnalyzer._analyze_request → findings.extend

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._analyze_request到findings.extend的處理能力
- **使用時機**: 當需要對PassiveAnalyzer._analyze_request進行深度分析並生成findings.extend結果時
- **預期結果**: 獲得基於程式邏輯的findings.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 629: 內探：Findings.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 629
```
```bash
aiva_internal_executor.py --flow 629 --dry-run
```

---

### PassiveAnalyzer._analyze_response → findings.extend

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._analyze_response到findings.extend的處理能力
- **使用時機**: 當需要對PassiveAnalyzer._analyze_response進行深度分析並生成findings.extend結果時
- **預期結果**: 獲得基於程式邏輯的findings.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 630: 內探：Findings.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 630
```
```bash
aiva_internal_executor.py --flow 630 --dry-run
```

---

### PassiveAnalyzer._analyze_set_cookie → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._analyze_set_cookie到findings.append的處理能力
- **使用時機**: 當需要對PassiveAnalyzer._analyze_set_cookie進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 636: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 636
```
```bash
aiva_internal_executor.py --flow 636 --dry-run
```

---

### PassiveAnalyzer._check_error_disclosure → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._check_error_disclosure到findings.append的處理能力
- **使用時機**: 當需要對PassiveAnalyzer._check_error_disclosure進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 637: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 637
```
```bash
aiva_internal_executor.py --flow 637 --dry-run
```

---

### PassiveAnalyzer._check_security_headers → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._check_security_headers到findings.append的處理能力
- **使用時機**: 當需要對PassiveAnalyzer._check_security_headers進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 634: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 634
```
```bash
aiva_internal_executor.py --flow 634 --dry-run
```

---

### PassiveAnalyzer._check_sensitive_data_in_body → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._check_sensitive_data_in_body到findings.append的處理能力
- **使用時機**: 當需要快速從PassiveAnalyzer._check_sensitive_data_in_body獲取findings.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 633: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 633
```
```bash
aiva_internal_executor.py --flow 633 --dry-run
```

---

### PassiveAnalyzer._check_sensitive_data_in_url → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._check_sensitive_data_in_url到findings.append的處理能力
- **使用時機**: 當需要快速從PassiveAnalyzer._check_sensitive_data_in_url獲取findings.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 631: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 631
```
```bash
aiva_internal_executor.py --flow 631 --dry-run
```

---

### PassiveAnalyzer._check_sensitive_params → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer._check_sensitive_params到findings.append的處理能力
- **使用時機**: 當需要快速從PassiveAnalyzer._check_sensitive_params獲取findings.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 632: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 632
```
```bash
aiva_internal_executor.py --flow 632 --dry-run
```

---

### PassiveAnalyzer.analyze_har → findings.append

**AI描述欄位 📋**:
- **能力概要**: PassiveAnalyzer.analyze_har到findings.append的處理能力
- **使用時機**: 當需要對PassiveAnalyzer.analyze_har進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 628: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 628
```
```bash
aiva_internal_executor.py --flow 628 --dry-run
```

---

### PostExDetector.analyze → findings.append

**AI描述欄位 📋**:
- **能力概要**: PostExDetector.analyze到findings.append的處理能力
- **使用時機**: 當需要對PostExDetector.analyze進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 104: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 104
```
```bash
aiva_internal_executor.py --flow 104 --dry-run
```

---

### PostExManager.scan → self._generate_summary

**AI描述欄位 📋**:
- **能力概要**: PostExManager.scan到self._generate_summary的處理能力
- **使用時機**: 當需要對PostExManager.scan進行深度分析並生成self._generate_summary結果時
- **預期結果**: 獲得基於程式邏輯的self._generate_summary結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 101: 內探：Self. Generate Summary

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Self. Generate Summary相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 101
```
```bash
aiva_internal_executor.py --flow 101 --dry-run
```

---

### SSRFDetector.analyze → engine.close

**AI描述欄位 📋**:
- **能力概要**: SSRFDetector.analyze到engine.close的處理能力
- **使用時機**: 當需要對SSRFDetector.analyze進行深度分析並生成engine.close結果時
- **預期結果**: 獲得基於程式邏輯的engine.close結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 364: 內探：Engine.Close

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Engine.Close相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 364
```
```bash
aiva_internal_executor.py --flow 364 --dry-run
```

---

### SSRFEngine._is_internal_ip → ipaddress.ip_address

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine._is_internal_ip到ipaddress.ip_address的處理能力
- **使用時機**: 當需要快速從SSRFEngine._is_internal_ip獲取ipaddress.ip_address的基礎信息時
- **預期結果**: 獲得基於程式邏輯的ipaddress.ip_address結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 368: 內探：Ipaddress.Ip Address

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Ipaddress.Ip Address的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 368
```
```bash
aiva_internal_executor.py --flow 368 --dry-run
```

---

### SSRFEngine.check_internal_access → issues.append

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine.check_internal_access到issues.append的處理能力
- **使用時機**: 當需要對SSRFEngine.check_internal_access進行深度分析並生成issues.append結果時
- **預期結果**: 獲得基於程式邏輯的issues.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 369: 內探：Issues.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Issues.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 369
```
```bash
aiva_internal_executor.py --flow 369 --dry-run
```

---

### SmartSSRFDetector._test_vector → context.add_finding

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._test_vector到context.add_finding的處理能力
- **使用時機**: 當需要對SmartSSRFDetector._test_vector進行深度分析並生成context.add_finding結果時
- **預期結果**: 獲得部分AI輔助的context.add_finding結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 344: 內探：Context.Add Finding

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行內探：Context.Add Finding相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 344
```
```bash
aiva_internal_executor.py --flow 344 --dry-run
```

---

### SmartSSRFDetector._verify_internal_service_access → self._verify_service_content

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._verify_internal_service_access到self._verify_service_content的處理能力
- **使用時機**: 當需要快速從SmartSSRFDetector._verify_internal_service_access獲取self._verify_service_content的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._verify_service_content結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 349: 內探：Self. Verify Service Content

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Self. Verify Service Content的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 349
```
```bash
aiva_internal_executor.py --flow 349 --dry-run
```

---

### SmartSSRFDetector.detect_vulnerabilities → context.add_error

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector.detect_vulnerabilities到context.add_error的處理能力
- **使用時機**: 當需要對SmartSSRFDetector.detect_vulnerabilities進行深度分析並生成context.add_error結果時
- **預期結果**: 獲得基於程式邏輯的context.add_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 341: 內探：Context.Add Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Context.Add Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 341
```
```bash
aiva_internal_executor.py --flow 341 --dry-run
```

---

### SsrfWorkerService.__init__ → ParamSemanticsAnalyzer

**AI描述欄位 📋**:
- **能力概要**: SsrfWorkerService.__init__到ParamSemanticsAnalyzer的處理能力
- **使用時機**: 當需要對SsrfWorkerService.__init__進行深度分析並生成ParamSemanticsAnalyzer結果時
- **預期結果**: 獲得基於程式邏輯的ParamSemanticsAnalyzer結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 362: 內探：Paramsemanticsanalyzer

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Paramsemanticsanalyzer相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 362
```
```bash
aiva_internal_executor.py --flow 362 --dry-run
```

---

### TechDetector._analyze_cookies → technologies.add

**AI描述欄位 📋**:
- **能力概要**: TechDetector._analyze_cookies到technologies.add的處理能力
- **使用時機**: 當需要快速從TechDetector._analyze_cookies獲取technologies.add的基礎信息時
- **預期結果**: 獲得基於程式邏輯的technologies.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 438: 內探：Technologies.Add

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Technologies.Add的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 438
```
```bash
aiva_internal_executor.py --flow 438 --dry-run
```

---

### TechDetector._analyze_headers → technologies.add

**AI描述欄位 📋**:
- **能力概要**: TechDetector._analyze_headers到technologies.add的處理能力
- **使用時機**: 當需要對TechDetector._analyze_headers進行深度分析並生成technologies.add結果時
- **預期結果**: 獲得基於程式邏輯的technologies.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 436: 內探：Technologies.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Technologies.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 436
```
```bash
aiva_internal_executor.py --flow 436 --dry-run
```

---

### TechDetector._analyze_html → technologies.add

**AI描述欄位 📋**:
- **能力概要**: TechDetector._analyze_html到technologies.add的處理能力
- **使用時機**: 當需要對TechDetector._analyze_html進行深度分析並生成technologies.add結果時
- **預期結果**: 獲得基於程式邏輯的technologies.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 437: 內探：Technologies.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Technologies.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 437
```
```bash
aiva_internal_executor.py --flow 437 --dry-run
```

---

### TechDetector._analyze_meta_tags → technologies.add

**AI描述欄位 📋**:
- **能力概要**: TechDetector._analyze_meta_tags到technologies.add的處理能力
- **使用時機**: 當需要對TechDetector._analyze_meta_tags進行深度分析並生成technologies.add結果時
- **預期結果**: 獲得基於程式邏輯的technologies.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 439: 內探：Technologies.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Technologies.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 439
```
```bash
aiva_internal_executor.py --flow 439 --dry-run
```

---

### TechDetector._analyze_scripts → technologies.add

**AI描述欄位 📋**:
- **能力概要**: TechDetector._analyze_scripts到technologies.add的處理能力
- **使用時機**: 當需要對TechDetector._analyze_scripts進行深度分析並生成technologies.add結果時
- **預期結果**: 獲得基於程式邏輯的technologies.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 440: 內探：Technologies.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Technologies.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 440
```
```bash
aiva_internal_executor.py --flow 440 --dry-run
```

---

### XXEDetector.test_with_soap → findings.append

**AI描述欄位 📋**:
- **能力概要**: XXEDetector.test_with_soap到findings.append的處理能力
- **使用時機**: 當需要對XXEDetector.test_with_soap進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 640: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 640
```
```bash
aiva_internal_executor.py --flow 640 --dry-run
```

---

### XXEDetector.test_xxe → findings.append

**AI描述欄位 📋**:
- **能力概要**: XXEDetector.test_xxe到findings.append的處理能力
- **使用時機**: 當需要對XXEDetector.test_xxe進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 639: 內探：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 639
```
```bash
aiva_internal_executor.py --flow 639 --dry-run
```

---

### _analyze_detection_with_dom → dom_engine.analyze

**AI描述欄位 📋**:
- **能力概要**: _analyze_detection_with_dom到dom_engine.analyze的處理能力
- **使用時機**: 當需要快速從_analyze_detection_with_dom獲取dom_engine.analyze的基礎信息時
- **預期結果**: 獲得基於程式邏輯的dom_engine.analyze結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 490: 內探：Dom Engine.Analyze

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Dom Engine.Analyze的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 490
```
```bash
aiva_internal_executor.py --flow 490 --dry-run
```

---

### _build_internal_finding → FindingEvidence

**AI描述欄位 📋**:
- **能力概要**: _build_internal_finding到FindingEvidence的處理能力
- **使用時機**: 當需要快速從_build_internal_finding獲取FindingEvidence的基礎信息時
- **預期結果**: 獲得基於程式邏輯的FindingEvidence結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 358: 內探：Findingevidence

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供內探：Findingevidence的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 358
```
```bash
aiva_internal_executor.py --flow 358 --dry-run
```

---

### _process_detections → stats_collector.record_payload_test

**AI描述欄位 📋**:
- **能力概要**: _process_detections到stats_collector.record_payload_test的處理能力
- **使用時機**: 當需要對_process_detections進行深度分析並生成stats_collector.record_payload_test結果時
- **預期結果**: 獲得基於程式邏輯的stats_collector.record_payload_test結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 489: 內探：Stats Collector.Record Payload Test

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Stats Collector.Record Payload Test相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 489
```
```bash
aiva_internal_executor.py --flow 489 --dry-run
```

---

### _process_task → broker.publish

**AI描述欄位 📋**:
- **能力概要**: _process_task到broker.publish的處理能力
- **使用時機**: 當需要對_process_task進行深度分析並生成broker.publish結果時
- **預期結果**: 獲得基於程式邏輯的broker.publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 150: 內探：Broker.Publish

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Broker.Publish相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 150
```
```bash
aiva_internal_executor.py --flow 150 --dry-run
```

---

### process_task → RuntimeError

**AI描述欄位 📋**:
- **能力概要**: process_task到RuntimeError的處理能力
- **使用時機**: 當需要對process_task進行深度分析並生成RuntimeError結果時
- **預期結果**: 獲得基於程式邏輯的RuntimeError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 355: 內探：Runtimeerror

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Runtimeerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 355
```
```bash
aiva_internal_executor.py --flow 355 --dry-run
```

---

### run_dom_test → detector.analyze

**AI描述欄位 📋**:
- **能力概要**: run_dom_test到detector.analyze的處理能力
- **使用時機**: 當需要對run_dom_test進行深度分析並生成detector.analyze結果時
- **預期結果**: 獲得基於程式邏輯的detector.analyze結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 501: 內探：Detector.Analyze

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行內探：Detector.Analyze相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 501
```
```bash
aiva_internal_executor.py --flow 501 --dry-run
```

---

## 任務規劃模組

**模組統計**:
- 總能力數量: 72
- 內部能力: 41
- 外部能力: 31

### AnalysisPlan → field

**AI描述欄位 📋**:
- **能力概要**: AnalysisPlan到field的處理能力
- **使用時機**: 當外部系統需要簡單的AnalysisPlan到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 319: 規劃：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 319
```
```bash
aiva_internal_executor.py --flow 319 --dry-run
```

---

### BizLogicManager.scan → task.metadata.get

**AI描述欄位 📋**:
- **能力概要**: BizLogicManager.scan到task.metadata.get的處理能力
- **使用時機**: 當外部系統需要簡單的BizLogicManager.scan到task.metadata.get轉換時
- **預期結果**: 獲得基於程式邏輯的task.metadata.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 63: 規劃：Task.Metadata.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Task.Metadata.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 63
```
```bash
aiva_internal_executor.py --flow 63 --dry-run
```

---

### BountyHunterManager.hunt_vulnerabilities → self.scanner.session.close

**AI描述欄位 📋**:
- **能力概要**: BountyHunterManager.hunt_vulnerabilities到self.scanner.session.close的處理能力
- **使用時機**: 當需要對BountyHunterManager.hunt_vulnerabilities進行深度分析並生成self.scanner.session.close結果時
- **預期結果**: 獲得部分AI輔助的self.scanner.session.close結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 256: 規劃：Self.Scanner.Session.Close

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Self.Scanner.Session.Close相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 256
```
```bash
aiva_internal_executor.py --flow 256 --dry-run
```

---

### CrossLanguageXSSEngine._execute_go_tool → ValueError

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._execute_go_tool到ValueError的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._execute_go_tool進行深度分析並生成ValueError結果時
- **預期結果**: 獲得基於程式邏輯的ValueError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 520: 規劃：Valueerror

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Valueerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 520
```
```bash
aiva_internal_executor.py --flow 520 --dry-run
```

---

### CrossLanguageXSSEngine._execute_parallel_detection → detection_results.append

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._execute_parallel_detection到detection_results.append的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._execute_parallel_detection進行深度分析並生成detection_results.append結果時
- **預期結果**: 獲得基於程式邏輯的detection_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 518: 規劃：Detection Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Detection Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 518
```
```bash
aiva_internal_executor.py --flow 518 --dry-run
```

---

### CrossLanguageXSSEngine._execute_python_tool → run_pattern.format

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._execute_python_tool到run_pattern.format的處理能力
- **使用時機**: 當外部系統需要簡單的CrossLanguageXSSEngine._execute_python_tool到run_pattern.format轉換時
- **預期結果**: 獲得基於程式邏輯的run_pattern.format結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 522: 規劃：Run Pattern.Format

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Run Pattern.Format的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 522
```
```bash
aiva_internal_executor.py --flow 522 --dry-run
```

---

### CrossLanguageXSSEngine._execute_ruby_tool → ValueError

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._execute_ruby_tool到ValueError的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._execute_ruby_tool進行深度分析並生成ValueError結果時
- **預期結果**: 獲得基於程式邏輯的ValueError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 521: 規劃：Valueerror

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Valueerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 521
```
```bash
aiva_internal_executor.py --flow 521 --dry-run
```

---

### CrossLanguageXSSEngine._execute_rust_tool → ValueError

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._execute_rust_tool到ValueError的處理能力
- **使用時機**: 當外部系統需要簡單的CrossLanguageXSSEngine._execute_rust_tool到ValueError轉換時
- **預期結果**: 獲得基於程式邏輯的ValueError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 523: 規劃：Valueerror

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Valueerror的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 523
```
```bash
aiva_internal_executor.py --flow 523 --dry-run
```

---

### CrossLanguageXSSEngine._execute_tool_detection → self._parse_tool_output

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._execute_tool_detection到self._parse_tool_output的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._execute_tool_detection進行深度分析並生成self._parse_tool_output結果時
- **預期結果**: 獲得基於程式邏輯的self._parse_tool_output結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 519: 規劃：Self. Parse Tool Output

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Parse Tool Output相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 519
```
```bash
aiva_internal_executor.py --flow 519 --dry-run
```

---

### CrossLanguageXSSEngine._get_available_execution_plans → self.logger.warning

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._get_available_execution_plans到self.logger.warning的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._get_available_execution_plans進行深度分析並生成self.logger.warning結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.warning結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 517: 規劃：Self.Logger.Warning

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self.Logger.Warning相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 517
```
```bash
aiva_internal_executor.py --flow 517 --dry-run
```

---

### CrossLanguageXSSEngine.detect → self.logger.warning

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine.detect到self.logger.warning的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine.detect進行深度分析並生成self.logger.warning結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.warning結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 516: 規劃：Self.Logger.Warning

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self.Logger.Warning相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 516
```
```bash
aiva_internal_executor.py --flow 516 --dry-run
```

---

### DirectoryScanner.scan_directories → tasks.append

**AI描述欄位 📋**:
- **能力概要**: DirectoryScanner.scan_directories到tasks.append的處理能力
- **使用時機**: 當需要對DirectoryScanner.scan_directories進行深度分析並生成tasks.append結果時
- **預期結果**: 獲得部分AI輔助的tasks.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 385: 規劃：Tasks.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Tasks.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 385
```
```bash
aiva_internal_executor.py --flow 385 --dry-run
```

---

### EnhancedIDORWorker._execute_task → broker.publish

**AI描述欄位 📋**:
- **能力概要**: EnhancedIDORWorker._execute_task到broker.publish的處理能力
- **使用時機**: 當需要對EnhancedIDORWorker._execute_task進行深度分析並生成broker.publish結果時
- **預期結果**: 獲得基於程式邏輯的broker.publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 70: 規劃：Broker.Publish

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Broker.Publish相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 70
```
```bash
aiva_internal_executor.py --flow 70 --dry-run
```

---

### EnhancedIDORWorker.process_task → stats_collector.set_module_specific

**AI描述欄位 📋**:
- **能力概要**: EnhancedIDORWorker.process_task到stats_collector.set_module_specific的處理能力
- **使用時機**: 當需要對EnhancedIDORWorker.process_task進行深度分析並生成stats_collector.set_module_specific結果時
- **預期結果**: 獲得基於程式邏輯的stats_collector.set_module_specific結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 71: 規劃：Stats Collector.Set Module Specific

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Stats Collector.Set Module Specific相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 71
```
```bash
aiva_internal_executor.py --flow 71 --dry-run
```

---

### HackingToolDetectionEngine._execute_tool → RuntimeError

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._execute_tool到RuntimeError的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolDetectionEngine._execute_tool的RuntimeError服務時
- **預期結果**: 獲得基於程式邏輯的RuntimeError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 229: 規劃：Runtimeerror

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Runtimeerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 229
```
```bash
aiva_internal_executor.py --flow 229 --dry-run
```

---

### HackingToolDetectionEngine._run_tool_detection → self._parse_tool_output

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._run_tool_detection到self._parse_tool_output的處理能力
- **使用時機**: 當需要對HackingToolDetectionEngine._run_tool_detection進行深度分析並生成self._parse_tool_output結果時
- **預期結果**: 獲得基於程式邏輯的self._parse_tool_output結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 228: 規劃：Self. Parse Tool Output

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Parse Tool Output相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 228
```
```bash
aiva_internal_executor.py --flow 228 --dry-run
```

---

### HackingToolDetectionEngine.detect → results.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine.detect到results.append的處理能力
- **使用時機**: 當需要對HackingToolDetectionEngine.detect進行深度分析並生成results.append結果時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 227: 規劃：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 227
```
```bash
aiva_internal_executor.py --flow 227 --dry-run
```

---

### HackingToolXSSConfig.get_execution_plan → execution_plan.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolXSSConfig.get_execution_plan到execution_plan.append的處理能力
- **使用時機**: 當需要對HackingToolXSSConfig.get_execution_plan進行深度分析並生成execution_plan.append結果時
- **預期結果**: 獲得基於程式邏輯的execution_plan.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 457: 規劃：Execution Plan.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Execution Plan.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 457
```
```bash
aiva_internal_executor.py --flow 457 --dry-run
```

---

### PersistenceChecker.check_scheduled_tasks → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceChecker.check_scheduled_tasks到self.test_results.append的處理能力
- **使用時機**: 當需要對PersistenceChecker.check_scheduled_tasks進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 130: 規劃：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 130
```
```bash
aiva_internal_executor.py --flow 130 --dry-run
```

---

### RaceConditionScanner.test_concurrent_requests → findings.append

**AI描述欄位 📋**:
- **能力概要**: RaceConditionScanner.test_concurrent_requests到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供RaceConditionScanner.test_concurrent_requests的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 45: 規劃：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 45
```
```bash
aiva_internal_executor.py --flow 45 --dry-run
```

---

### SQLInjectionBountyCapability.execute → self.manager.generate_bounty_report

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionBountyCapability.execute到self.manager.generate_bounty_report的處理能力
- **使用時機**: 當需要對SQLInjectionBountyCapability.execute進行深度分析並生成self.manager.generate_bounty_report結果時
- **預期結果**: 獲得基於程式邏輯的self.manager.generate_bounty_report結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 264: 規劃：Self.Manager.Generate Bounty Report

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self.Manager.Generate Bounty Report相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 264
```
```bash
aiva_internal_executor.py --flow 264 --dry-run
```

---

### SQLInjectionManager.comprehensive_scan → self.custom_scanner.scan_target

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionManager.comprehensive_scan到self.custom_scanner.scan_target的處理能力
- **使用時機**: 當需要對SQLInjectionManager.comprehensive_scan進行深度分析並生成self.custom_scanner.scan_target結果時
- **預期結果**: 獲得基於程式邏輯的self.custom_scanner.scan_target結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 285: 規劃：Self.Custom Scanner.Scan Target

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self.Custom Scanner.Scan Target相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 285
```
```bash
aiva_internal_executor.py --flow 285 --dry-run
```

---

### SSRFEngine.run → issues.extend

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine.run到issues.extend的處理能力
- **使用時機**: 當需要為外部用戶提供SSRFEngine.run的issues.extend服務時
- **預期結果**: 獲得基於程式邏輯的issues.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 372: 規劃：Issues.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Issues.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 372
```
```bash
aiva_internal_executor.py --flow 372 --dry-run
```

---

### SmartIDORDetector._execute_horizontal_testing → smart_manager.update_progress

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector._execute_horizontal_testing到smart_manager.update_progress的處理能力
- **使用時機**: 當需要快速從SmartIDORDetector._execute_horizontal_testing獲取smart_manager.update_progress的基礎信息時
- **預期結果**: 獲得基於程式邏輯的smart_manager.update_progress結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 84: 規劃：Smart Manager.Update Progress

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Smart Manager.Update Progress的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 84
```
```bash
aiva_internal_executor.py --flow 84 --dry-run
```

---

### SmartIDORDetector._execute_vertical_testing → smart_manager.update_progress

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector._execute_vertical_testing到smart_manager.update_progress的處理能力
- **使用時機**: 當需要快速從SmartIDORDetector._execute_vertical_testing獲取smart_manager.update_progress的基礎信息時
- **預期結果**: 獲得基於程式邏輯的smart_manager.update_progress結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 85: 規劃：Smart Manager.Update Progress

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Smart Manager.Update Progress的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 85
```
```bash
aiva_internal_executor.py --flow 85 --dry-run
```

---

### SmartSSRFDetector._execute_detection → context.increment_attempts

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._execute_detection到context.increment_attempts的處理能力
- **使用時機**: 當需要對SmartSSRFDetector._execute_detection進行深度分析並生成context.increment_attempts結果時
- **預期結果**: 獲得部分AI輔助的context.increment_attempts結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 343: 規劃：Context.Increment Attempts

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Context.Increment Attempts相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 343
```
```bash
aiva_internal_executor.py --flow 343 --dry-run
```

---

### SmartSSRFDetector._execute_http_request → urlunparse

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._execute_http_request到urlunparse的處理能力
- **使用時機**: 當外部系統需要簡單的SmartSSRFDetector._execute_http_request到urlunparse轉換時
- **預期結果**: 獲得基於程式邏輯的urlunparse結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 348: 規劃：Urlunparse

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Urlunparse的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 348
```
```bash
aiva_internal_executor.py --flow 348 --dry-run
```

---

### SqliDetector._execute_parallel_detection → engine.detect

**AI描述欄位 📋**:
- **能力概要**: SqliDetector._execute_parallel_detection到engine.detect的處理能力
- **使用時機**: 當外部系統需要簡單的SqliDetector._execute_parallel_detection到engine.detect轉換時
- **預期結果**: 獲得基於程式邏輯的engine.detect結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 213: 規劃：Engine.Detect

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Engine.Detect的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 213
```
```bash
aiva_internal_executor.py --flow 213 --dry-run
```

---

### SqliDetector.detect_sqli → self._execute_parallel_detection

**AI描述欄位 📋**:
- **能力概要**: SqliDetector.detect_sqli到self._execute_parallel_detection的處理能力
- **使用時機**: 當需要對SqliDetector.detect_sqli進行深度分析並生成self._execute_parallel_detection結果時
- **預期結果**: 獲得基於程式邏輯的self._execute_parallel_detection結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 212: 規劃：Self. Execute Parallel Detection

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Execute Parallel Detection相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 212
```
```bash
aiva_internal_executor.py --flow 212 --dry-run
```

---

### SqliOrchestrator.execute_detection → context.telemetry.add_error

**AI描述欄位 📋**:
- **能力概要**: SqliOrchestrator.execute_detection到context.telemetry.add_error的處理能力
- **使用時機**: 當需要對SqliOrchestrator.execute_detection進行深度分析並生成context.telemetry.add_error結果時
- **預期結果**: 獲得部分AI輔助的context.telemetry.add_error結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 202: 規劃：Context.Telemetry.Add Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Context.Telemetry.Add Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 202
```
```bash
aiva_internal_executor.py --flow 202 --dry-run
```

---

### SqliResultBinderPublisher.publish_status → self._publish

**AI描述欄位 📋**:
- **能力概要**: SqliResultBinderPublisher.publish_status到self._publish的處理能力
- **使用時機**: 當需要快速從SqliResultBinderPublisher.publish_status獲取self._publish的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 187: 規劃：Self. Publish

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self. Publish的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 187
```
```bash
aiva_internal_executor.py --flow 187 --dry-run
```

---

### SqliTaskQueue.close → self._queue.put

**AI描述欄位 📋**:
- **能力概要**: SqliTaskQueue.close到self._queue.put的處理能力
- **使用時機**: 當需要快速從SqliTaskQueue.close獲取self._queue.put的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._queue.put結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 193: 規劃：Self. Queue.Put

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self. Queue.Put的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 193
```
```bash
aiva_internal_executor.py --flow 193 --dry-run
```

---

### SqliTaskQueue.get → self._queue.task_done

**AI描述欄位 📋**:
- **能力概要**: SqliTaskQueue.get到self._queue.task_done的處理能力
- **使用時機**: 當需要快速從SqliTaskQueue.get獲取self._queue.task_done的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._queue.task_done結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 192: 規劃：Self. Queue.Task Done

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self. Queue.Task Done的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 192
```
```bash
aiva_internal_executor.py --flow 192 --dry-run
```

---

### SqliTaskQueue.put → self._queue.put

**AI描述欄位 📋**:
- **能力概要**: SqliTaskQueue.put到self._queue.put的處理能力
- **使用時機**: 當需要快速從SqliTaskQueue.put獲取self._queue.put的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._queue.put結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 191: 規劃：Self. Queue.Put

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self. Queue.Put的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 191
```
```bash
aiva_internal_executor.py --flow 191 --dry-run
```

---

### SqliWorkerService.process_task → stats_collector.set_module_specific

**AI描述欄位 📋**:
- **能力概要**: SqliWorkerService.process_task到stats_collector.set_module_specific的處理能力
- **使用時機**: 當需要對SqliWorkerService.process_task進行深度分析並生成stats_collector.set_module_specific結果時
- **預期結果**: 獲得部分AI輔助的stats_collector.set_module_specific結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 204: 規劃：Stats Collector.Set Module Specific

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Stats Collector.Set Module Specific相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 204
```
```bash
aiva_internal_executor.py --flow 204 --dry-run
```

---

### SqliWorkerService.process_task_dict → f.model_dump

**AI描述欄位 📋**:
- **能力概要**: SqliWorkerService.process_task_dict到f.model_dump的處理能力
- **使用時機**: 當需要對SqliWorkerService.process_task_dict進行深度分析並生成f.model_dump結果時
- **預期結果**: 獲得部分AI輔助的f.model_dump結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 205: 規劃：F.Model Dump

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：F.Model Dump相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 205
```
```bash
aiva_internal_executor.py --flow 205 --dry-run
```

---

### SsrfResultPublisher.publish_status → self._publish

**AI描述欄位 📋**:
- **能力概要**: SsrfResultPublisher.publish_status到self._publish的處理能力
- **使用時機**: 當需要對SsrfResultPublisher.publish_status進行深度分析並生成self._publish結果時
- **預期結果**: 獲得基於程式邏輯的self._publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 333: 規劃：Self. Publish

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Publish相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 333
```
```bash
aiva_internal_executor.py --flow 333 --dry-run
```

---

### SsrfWorkerService.process_task → _execute_task

**AI描述欄位 📋**:
- **能力概要**: SsrfWorkerService.process_task到_execute_task的處理能力
- **使用時機**: 當需要為外部用戶提供SsrfWorkerService.process_task的_execute_task服務時
- **預期結果**: 獲得基於程式邏輯的_execute_task結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 363: 規劃： Execute Task

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃： Execute Task相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 363
```
```bash
aiva_internal_executor.py --flow 363 --dry-run
```

---

### StoredXssDetector.execute → client.aclose

**AI描述欄位 📋**:
- **能力概要**: StoredXssDetector.execute到client.aclose的處理能力
- **使用時機**: 當需要對StoredXssDetector.execute進行深度分析並生成client.aclose結果時
- **預期結果**: 獲得基於程式邏輯的client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 464: 規劃：Client.Aclose

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Client.Aclose相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 464
```
```bash
aiva_internal_executor.py --flow 464 --dry-run
```

---

### TaskDependency → Field

**AI描述欄位 📋**:
- **能力概要**: TaskDependency到Field的處理能力
- **使用時機**: 當需要為外部用戶提供TaskDependency的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 18: 規劃：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 18
```
```bash
aiva_internal_executor.py --flow 18 --dry-run
```

---

### TaskExecution → Field

**AI描述欄位 📋**:
- **能力概要**: TaskExecution到Field的處理能力
- **使用時機**: 當需要為外部用戶提供TaskExecution的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 19: 規劃：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 19
```
```bash
aiva_internal_executor.py --flow 19 --dry-run
```

---

### TaskExecution.validate_task_id → AIVAError

**AI描述欄位 📋**:
- **能力概要**: TaskExecution.validate_task_id到AIVAError的處理能力
- **使用時機**: 當外部系統需要簡單的TaskExecution.validate_task_id到AIVAError轉換時
- **預期結果**: 獲得基於程式邏輯的AIVAError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 20: 規劃：Aivaerror

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Aivaerror的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 20
```
```bash
aiva_internal_executor.py --flow 20 --dry-run
```

---

### TaskQueue → Field

**AI描述欄位 📋**:
- **能力概要**: TaskQueue到Field的處理能力
- **使用時機**: 當需要為外部用戶提供TaskQueue的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 21: 規劃：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 21
```
```bash
aiva_internal_executor.py --flow 21 --dry-run
```

---

### TestTask → Field

**AI描述欄位 📋**:
- **能力概要**: TestTask到Field的處理能力
- **使用時機**: 當需要為外部用戶提供TestTask的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 32: 規劃：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 32
```
```bash
aiva_internal_executor.py --flow 32 --dry-run
```

---

### TraditionalXssDetector.execute → client.aclose

**AI描述欄位 📋**:
- **能力概要**: TraditionalXssDetector.execute到client.aclose的處理能力
- **使用時機**: 當需要對TraditionalXssDetector.execute進行深度分析並生成client.aclose結果時
- **預期結果**: 獲得基於程式邏輯的client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 475: 規劃：Client.Aclose

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Client.Aclose相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 475
```
```bash
aiva_internal_executor.py --flow 475 --dry-run
```

---

### WebAttackCapability._execute_comprehensive_scan → self.manager.comprehensive_scan

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability._execute_comprehensive_scan到self.manager.comprehensive_scan的處理能力
- **使用時機**: 當需要對WebAttackCapability._execute_comprehensive_scan進行深度分析並生成self.manager.comprehensive_scan結果時
- **預期結果**: 獲得基於程式邏輯的self.manager.comprehensive_scan結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 412: 規劃：Self.Manager.Comprehensive Scan

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self.Manager.Comprehensive Scan相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 412
```
```bash
aiva_internal_executor.py --flow 412 --dry-run
```

---

### WebAttackCapability._execute_directory_scan → self.manager.directory_scanner.scan_directories

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability._execute_directory_scan到self.manager.directory_scanner.scan_directories的處理能力
- **使用時機**: 當需要對WebAttackCapability._execute_directory_scan進行深度分析並生成self.manager.directory_scanner.scan_directories結果時
- **預期結果**: 獲得部分AI輔助的self.manager.directory_scanner.scan_directories結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 414: 規劃：Self.Manager.Directory Scanner.Scan Directories

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Self.Manager.Directory Scanner.Scan Directories相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 414
```
```bash
aiva_internal_executor.py --flow 414 --dry-run
```

---

### WebAttackCapability._execute_interactive → self.cli.run_interactive

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability._execute_interactive到self.cli.run_interactive的處理能力
- **使用時機**: 當需要快速從WebAttackCapability._execute_interactive獲取self.cli.run_interactive的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.cli.run_interactive結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 417: 規劃：Self.Cli.Run Interactive

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self.Cli.Run Interactive的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 417
```
```bash
aiva_internal_executor.py --flow 417 --dry-run
```

---

### WebAttackCapability._execute_subdomain_scan → self.manager.subdomain_enumerator.enumerate_subdomains

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability._execute_subdomain_scan到self.manager.subdomain_enumerator.enumerate_subdomains的處理能力
- **使用時機**: 當需要快速從WebAttackCapability._execute_subdomain_scan獲取self.manager.subdomain_enumerator.enumerate_subdomains的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.manager.subdomain_enumerator.enumerate_subdomains結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 413: 規劃：Self.Manager.Subdomain Enumerator.Enumerate Subdomains

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self.Manager.Subdomain Enumerator.Enumerate Subdomains的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 413
```
```bash
aiva_internal_executor.py --flow 413 --dry-run
```

---

### WebAttackCapability._execute_technology_detection → self.manager.technology_detector.detect_technologies

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability._execute_technology_detection到self.manager.technology_detector.detect_technologies的處理能力
- **使用時機**: 當需要快速從WebAttackCapability._execute_technology_detection獲取self.manager.technology_detector.detect_technologies的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.manager.technology_detector.detect_technologies結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 416: 規劃：Self.Manager.Technology Detector.Detect Technologies

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self.Manager.Technology Detector.Detect Technologies的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 416
```
```bash
aiva_internal_executor.py --flow 416 --dry-run
```

---

### WebAttackCapability._execute_vulnerability_scan → self.manager.vulnerability_scanner.scan_vulnerabilities

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability._execute_vulnerability_scan到self.manager.vulnerability_scanner.scan_vulnerabilities的處理能力
- **使用時機**: 當需要對WebAttackCapability._execute_vulnerability_scan進行深度分析並生成self.manager.vulnerability_scanner.scan_vulnerabilities結果時
- **預期結果**: 獲得部分AI輔助的self.manager.vulnerability_scanner.scan_vulnerabilities結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 415: 規劃：Self.Manager.Vulnerability Scanner.Scan Vulnerabilities

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃：Self.Manager.Vulnerability Scanner.Scan Vulnerabilities相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 415
```
```bash
aiva_internal_executor.py --flow 415 --dry-run
```

---

### WebAttackCapability.execute → command_handlers.get

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability.execute到command_handlers.get的處理能力
- **使用時機**: 當外部系統需要簡單的WebAttackCapability.execute到command_handlers.get轉換時
- **預期結果**: 獲得基於程式邏輯的command_handlers.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 411: 規劃：Command Handlers.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Command Handlers.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 411
```
```bash
aiva_internal_executor.py --flow 411 --dry-run
```

---

### WebAttackManager.comprehensive_scan → progress.remove_task

**AI描述欄位 📋**:
- **能力概要**: WebAttackManager.comprehensive_scan到progress.remove_task的處理能力
- **使用時機**: 當需要對WebAttackManager.comprehensive_scan進行深度分析並生成progress.remove_task結果時
- **預期結果**: 獲得基於程式邏輯的progress.remove_task結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 398: 規劃：Progress.Remove Task

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Progress.Remove Task相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 398
```
```bash
aiva_internal_executor.py --flow 398 --dry-run
```

---

### XssResultPublisher.publish_status → self._publish

**AI描述欄位 📋**:
- **能力概要**: XssResultPublisher.publish_status到self._publish的處理能力
- **使用時機**: 當需要對XssResultPublisher.publish_status進行深度分析並生成self._publish結果時
- **預期結果**: 獲得基於程式邏輯的self._publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 460: 規劃：Self. Publish

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Publish相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 460
```
```bash
aiva_internal_executor.py --flow 460 --dry-run
```

---

### XssTaskQueue.__init__ → itertools.count

**AI描述欄位 📋**:
- **能力概要**: XssTaskQueue.__init__到itertools.count的處理能力
- **使用時機**: 當外部系統需要簡單的XssTaskQueue.__init__到itertools.count轉換時
- **預期結果**: 獲得基於程式邏輯的itertools.count結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 469: 規劃：Itertools.Count

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Itertools.Count的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 469
```
```bash
aiva_internal_executor.py --flow 469 --dry-run
```

---

### XssTaskQueue._discard_invalid_locked → heapq.heappop

**AI描述欄位 📋**:
- **能力概要**: XssTaskQueue._discard_invalid_locked到heapq.heappop的處理能力
- **使用時機**: 當外部系統需要簡單的XssTaskQueue._discard_invalid_locked到heapq.heappop轉換時
- **預期結果**: 獲得基於程式邏輯的heapq.heappop結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 473: 規劃：Heapq.Heappop

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Heapq.Heappop的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 473
```
```bash
aiva_internal_executor.py --flow 473 --dry-run
```

---

### XssTaskQueue.close → self._condition.notify_all

**AI描述欄位 📋**:
- **能力概要**: XssTaskQueue.close到self._condition.notify_all的處理能力
- **使用時機**: 當需要快速從XssTaskQueue.close獲取self._condition.notify_all的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._condition.notify_all結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 472: 規劃：Self. Condition.Notify All

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Self. Condition.Notify All的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 472
```
```bash
aiva_internal_executor.py --flow 472 --dry-run
```

---

### XssTaskQueue.get → self._entries.pop

**AI描述欄位 📋**:
- **能力概要**: XssTaskQueue.get到self._entries.pop的處理能力
- **使用時機**: 當需要對XssTaskQueue.get進行深度分析並生成self._entries.pop結果時
- **預期結果**: 獲得基於程式邏輯的self._entries.pop結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 471: 規劃：Self. Entries.Pop

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Entries.Pop相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 471
```
```bash
aiva_internal_executor.py --flow 471 --dry-run
```

---

### XssTaskQueue.put → self._condition.notify_all

**AI描述欄位 📋**:
- **能力概要**: XssTaskQueue.put到self._condition.notify_all的處理能力
- **使用時機**: 當需要對XssTaskQueue.put進行深度分析並生成self._condition.notify_all結果時
- **預期結果**: 獲得基於程式邏輯的self._condition.notify_all結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 470: 規劃：Self. Condition.Notify All

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Self. Condition.Notify All相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 470
```
```bash
aiva_internal_executor.py --flow 470 --dry-run
```

---

### XssWorkerService.process_task → ValueError

**AI描述欄位 📋**:
- **能力概要**: XssWorkerService.process_task到ValueError的處理能力
- **使用時機**: 當外部系統需要簡單的XssWorkerService.process_task到ValueError轉換時
- **預期結果**: 獲得基於程式邏輯的ValueError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 499: 規劃：Valueerror

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃：Valueerror的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 499
```
```bash
aiva_internal_executor.py --flow 499 --dry-run
```

---

### _consume_queue → _execute_task

**AI描述欄位 📋**:
- **能力概要**: _consume_queue到_execute_task的處理能力
- **使用時機**: 當外部系統需要簡單的_consume_queue到_execute_task轉換時
- **預期結果**: 獲得基於程式邏輯的_execute_task結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 207: 規劃： Execute Task

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供規劃： Execute Task的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 207
```
```bash
aiva_internal_executor.py --flow 207 --dry-run
```

---

### _execute_stored_xss → findings.append

**AI描述欄位 📋**:
- **能力概要**: _execute_stored_xss到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供_execute_stored_xss的findings.append服務時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 491: 規劃：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 491
```
```bash
aiva_internal_executor.py --flow 491 --dry-run
```

---

### _execute_task → publisher.publish_error

**AI描述欄位 📋**:
- **能力概要**: _execute_task到publisher.publish_error的處理能力
- **使用時機**: 當需要為外部用戶提供_execute_task的publisher.publish_error服務時
- **預期結果**: 獲得基於程式邏輯的publisher.publish_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 208: 規劃：Publisher.Publish Error

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Publisher.Publish Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 208
```
```bash
aiva_internal_executor.py --flow 208 --dry-run
```

---

### _execute_task → publisher.publish_status

**AI描述欄位 📋**:
- **能力概要**: _execute_task到publisher.publish_status的處理能力
- **使用時機**: 當需要為外部用戶提供_execute_task的publisher.publish_status服務時
- **預期結果**: 獲得基於程式邏輯的publisher.publish_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 354: 規劃：Publisher.Publish Status

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Publisher.Publish Status相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 354
```
```bash
aiva_internal_executor.py --flow 354 --dry-run
```

---

### _execute_traditional_detection → _handle_detection_errors

**AI描述欄位 📋**:
- **能力概要**: _execute_traditional_detection到_handle_detection_errors的處理能力
- **使用時機**: 當需要為外部用戶提供_execute_traditional_detection的_handle_detection_errors服務時
- **預期結果**: 獲得基於程式邏輯的_handle_detection_errors結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 487: 規劃： Handle Detection Errors

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃： Handle Detection Errors相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 487
```
```bash
aiva_internal_executor.py --flow 487 --dry-run
```

---

### process_task → _process_detections

**AI描述欄位 📋**:
- **能力概要**: process_task到_process_detections的處理能力
- **使用時機**: 當需要為外部用戶提供process_task的_process_detections服務時
- **預期結果**: 獲得基於程式邏輯的_process_detections結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 485: 規劃： Process Detections

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃： Process Detections相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 485
```
```bash
aiva_internal_executor.py --flow 485 --dry-run
```

---

### process_task → context.statistics_collector.get_summary

**AI描述欄位 📋**:
- **能力概要**: process_task到context.statistics_collector.get_summary的處理能力
- **使用時機**: 當需要為外部用戶提供process_task的context.statistics_collector.get_summary服務時
- **預期結果**: 獲得基於程式邏輯的context.statistics_collector.get_summary結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 209: 規劃：Context.Statistics Collector.Get Summary

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Context.Statistics Collector.Get Summary相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 209
```
```bash
aiva_internal_executor.py --flow 209 --dry-run
```

---

### process_task → worker.process_task

**AI描述欄位 📋**:
- **能力概要**: process_task到worker.process_task的處理能力
- **使用時機**: 當需要為外部用戶提供process_task的worker.process_task服務時
- **預期結果**: 獲得基於程式邏輯的worker.process_task結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 74: 規劃：Worker.Process Task

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Worker.Process Task相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 74
```
```bash
aiva_internal_executor.py --flow 74 --dry-run
```

---

### run → _process_task

**AI描述欄位 📋**:
- **能力概要**: run到_process_task的處理能力
- **使用時機**: 當需要為外部用戶提供run的_process_task服務時
- **預期結果**: 獲得部分AI輔助的_process_task結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 149: 規劃： Process Task

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行規劃： Process Task相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 149
```
```bash
aiva_internal_executor.py --flow 149 --dry-run
```

---

### run → queue.close

**AI描述欄位 📋**:
- **能力概要**: run到queue.close的處理能力
- **使用時機**: 當需要為外部系統提供完整的run到queue.close解決方案時
- **預期結果**: 獲得基於程式邏輯的queue.close結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 206: 規劃：Queue.Close

- **範圍**: external
- **複雜度**: complex
- **AI等級**: none
- **用途**: 進行規劃：Queue.Close的高級處理，涉及複雜的邏輯判斷和多系統協調

**使用命令**:
```bash
aiva_internal_executor.py --flow 206
```
```bash
aiva_internal_executor.py --flow 206 --dry-run
```

---

### run_reflected_test → detector.execute

**AI描述欄位 📋**:
- **能力概要**: run_reflected_test到detector.execute的處理能力
- **使用時機**: 當需要為外部用戶提供run_reflected_test的detector.execute服務時
- **預期結果**: 獲得基於程式邏輯的detector.execute結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 500: 規劃：Detector.Execute

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Detector.Execute相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 500
```
```bash
aiva_internal_executor.py --flow 500 --dry-run
```

---

### run_stored_test → detector.execute

**AI描述欄位 📋**:
- **能力概要**: run_stored_test到detector.execute的處理能力
- **使用時機**: 當需要為外部用戶提供run_stored_test的detector.execute服務時
- **預期結果**: 獲得基於程式邏輯的detector.execute結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 502: 規劃：Detector.Execute

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行規劃：Detector.Execute相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 502
```
```bash
aiva_internal_executor.py --flow 502 --dry-run
```

---

## 核心能力模組

**模組統計**:
- 總能力數量: 49
- 內部能力: 24
- 外部能力: 25

### AuthnManager.__init__ → self._check_go_engine_availability

**AI描述欄位 📋**:
- **能力概要**: AuthnManager.__init__到self._check_go_engine_availability的處理能力
- **使用時機**: 當需要快速從AuthnManager.__init__獲取self._check_go_engine_availability的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._check_go_engine_availability結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 1: 核心：Self. Check Go Engine Availability

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Self. Check Go Engine Availability的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 1
```
```bash
aiva_internal_executor.py --flow 1 --dry-run
```

---

### AuthnManager._create_finding → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: AuthnManager._create_finding到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供AuthnManager._create_finding的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 6: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 6
```
```bash
aiva_internal_executor.py --flow 6 --dry-run
```

---

### BlindXSSDetector.scan_blind_xss → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector.scan_blind_xss到vulnerabilities.append的處理能力
- **使用時機**: 當需要對BlindXSSDetector.scan_blind_xss進行深度分析並生成vulnerabilities.append結果時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 548: 核心：Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 548
```
```bash
aiva_internal_executor.py --flow 548 --dry-run
```

---

### BooleanDetectionEngine._build_detection_result → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: BooleanDetectionEngine._build_detection_result到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供BooleanDetectionEngine._build_detection_result的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 220: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 220
```
```bash
aiva_internal_executor.py --flow 220 --dry-run
```

---

### BountyHunterScanner._verify_vulnerability → response.text

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner._verify_vulnerability到response.text的處理能力
- **使用時機**: 當需要對BountyHunterScanner._verify_vulnerability進行深度分析並生成response.text結果時
- **預期結果**: 獲得部分AI輔助的response.text結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 252: 核心：Response.Text

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Response.Text相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 252
```
```bash
aiva_internal_executor.py --flow 252 --dry-run
```

---

### CrossLanguageXSSEngine._check_dalfox_availability → self.logger.warning

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_dalfox_availability到self.logger.warning的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_dalfox_availability進行深度分析並生成self.logger.warning結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.warning結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 508: 核心：Self.Logger.Warning

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Self.Logger.Warning相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 508
```
```bash
aiva_internal_executor.py --flow 508 --dry-run
```

---

### CrossLanguageXSSEngine._check_xspear_availability → self.logger.warning

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_xspear_availability到self.logger.warning的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_xspear_availability進行深度分析並生成self.logger.warning結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.warning結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 509: 核心：Self.Logger.Warning

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Self.Logger.Warning相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 509
```
```bash
aiva_internal_executor.py --flow 509 --dry-run
```

---

### CrossLanguageXSSEngine._check_xsser_availability → self.logger.warning

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_xsser_availability到self.logger.warning的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_xsser_availability進行深度分析並生成self.logger.warning結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.warning結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 510: 核心：Self.Logger.Warning

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Self.Logger.Warning相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 510
```
```bash
aiva_internal_executor.py --flow 510 --dry-run
```

---

### CrossLanguageXSSEngine._validate_tool_availability → available_tools.append

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._validate_tool_availability到available_tools.append的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._validate_tool_availability進行深度分析並生成available_tools.append結果時
- **預期結果**: 獲得基於程式邏輯的available_tools.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 507: 核心：Available Tools.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Available Tools.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 507
```
```bash
aiva_internal_executor.py --flow 507 --dry-run
```

---

### CrossLanguageXSSEngine.initialize → self.logger.error

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine.initialize到self.logger.error的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine.initialize進行深度分析並生成self.logger.error結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 506: 核心：Self.Logger.Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Self.Logger.Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 506
```
```bash
aiva_internal_executor.py --flow 506 --dry-run
```

---

### DOMXSSDetector._test_dom_payloads → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: DOMXSSDetector._test_dom_payloads到vulnerabilities.append的處理能力
- **使用時機**: 當需要為外部用戶提供DOMXSSDetector._test_dom_payloads的vulnerabilities.append服務時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 543: 核心：Vulnerabilities.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 543
```
```bash
aiva_internal_executor.py --flow 543 --dry-run
```

---

### DalfoxIntegration._parse_dalfox_output → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: DalfoxIntegration._parse_dalfox_output到vulnerabilities.append的處理能力
- **使用時機**: 當外部系統需要簡單的DalfoxIntegration._parse_dalfox_output到vulnerabilities.append轉換時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 538: 核心：Vulnerabilities.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Vulnerabilities.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 538
```
```bash
aiva_internal_executor.py --flow 538 --dry-run
```

---

### ErrorDetectionEngine._build_detection_result → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: ErrorDetectionEngine._build_detection_result到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供ErrorDetectionEngine._build_detection_result的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 222: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 222
```
```bash
aiva_internal_executor.py --flow 222 --dry-run
```

---

### HackingToolDetectionEngine._check_tool_availability → tool_name.lower

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._check_tool_availability到tool_name.lower的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolDetectionEngine._check_tool_availability到tool_name.lower轉換時
- **預期結果**: 獲得基於程式邏輯的tool_name.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 225: 核心：Tool Name.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Tool Name.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 225
```
```bash
aiva_internal_executor.py --flow 225 --dry-run
```

---

### HackingToolDetectionEngine._create_detection_result → DetectionResult

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._create_detection_result到DetectionResult的處理能力
- **使用時機**: 當需要對HackingToolDetectionEngine._create_detection_result進行深度分析並生成DetectionResult結果時
- **預期結果**: 獲得基於程式邏輯的DetectionResult結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 231: 核心：Detectionresult

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Detectionresult相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 231
```
```bash
aiva_internal_executor.py --flow 231 --dry-run
```

---

### HackingToolDetectionEngine._validate_tools_availability → available_tools.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._validate_tools_availability到available_tools.append的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolDetectionEngine._validate_tools_availability到available_tools.append轉換時
- **預期結果**: 獲得基於程式邏輯的available_tools.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 224: 核心：Available Tools.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Available Tools.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 224
```
```bash
aiva_internal_executor.py --flow 224 --dry-run
```

---

### HackingToolDetectionEngine.get_tool_status → self.integrator.check_tool_availability

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine.get_tool_status到self.integrator.check_tool_availability的處理能力
- **使用時機**: 當需要快速從HackingToolDetectionEngine.get_tool_status獲取self.integrator.check_tool_availability的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.integrator.check_tool_availability結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 232: 核心：Self.Integrator.Check Tool Availability

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Self.Integrator.Check Tool Availability的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 232
```
```bash
aiva_internal_executor.py --flow 232 --dry-run
```

---

### HackingToolDetectionEngine.initialize → RuntimeError

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine.initialize到RuntimeError的處理能力
- **使用時機**: 當需要對HackingToolDetectionEngine.initialize進行深度分析並生成RuntimeError結果時
- **預期結果**: 獲得基於程式邏輯的RuntimeError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 226: 核心：Runtimeerror

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Runtimeerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 226
```
```bash
aiva_internal_executor.py --flow 226 --dry-run
```

---

### HackingToolSQLIntegrator.check_tool_availability → Path

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLIntegrator.check_tool_availability到Path的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolSQLIntegrator.check_tool_availability到Path轉換時
- **預期結果**: 獲得基於程式邏輯的Path結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 160: 核心：Path

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Path的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 160
```
```bash
aiva_internal_executor.py --flow 160 --dry-run
```

---

### HackingToolSQLIntegrator.generate_capability_records → records.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLIntegrator.generate_capability_records到records.append的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolSQLIntegrator.generate_capability_records到records.append轉換時
- **預期結果**: 獲得基於程式邏輯的records.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 163: 核心：Records.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Records.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 163
```
```bash
aiva_internal_executor.py --flow 163 --dry-run
```

---

### IDORDetector._to_finding → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: IDORDetector._to_finding到FindingTarget的處理能力
- **使用時機**: 當需要對IDORDetector._to_finding進行深度分析並生成FindingTarget結果時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 92: 核心：Findingtarget

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 92
```
```bash
aiva_internal_executor.py --flow 92 --dry-run
```

---

### OOBDetectionEngine._build_detection_result → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: OOBDetectionEngine._build_detection_result到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供OOBDetectionEngine._build_detection_result的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 236: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 236
```
```bash
aiva_internal_executor.py --flow 236 --dry-run
```

---

### PostExDetector._mk_finding → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: PostExDetector._mk_finding到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供PostExDetector._mk_finding的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 105: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 105
```
```bash
aiva_internal_executor.py --flow 105 --dry-run
```

---

### SQLInjectionBountyCapability.__init__ → BountyHunterManager

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionBountyCapability.__init__到BountyHunterManager的處理能力
- **使用時機**: 當外部系統需要簡單的SQLInjectionBountyCapability.__init__到BountyHunterManager轉換時
- **預期結果**: 獲得基於程式邏輯的BountyHunterManager結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 263: 核心：Bountyhuntermanager

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Bountyhuntermanager的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 263
```
```bash
aiva_internal_executor.py --flow 263 --dry-run
```

---

### SQLInjectionBountyCapability.cleanup → self.manager.target_queue.clear

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionBountyCapability.cleanup到self.manager.target_queue.clear的處理能力
- **使用時機**: 當需要對SQLInjectionBountyCapability.cleanup進行深度分析並生成self.manager.target_queue.clear結果時
- **預期結果**: 獲得部分AI輔助的self.manager.target_queue.clear結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 265: 核心：Self.Manager.Target Queue.Clear

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self.Manager.Target Queue.Clear相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 265
```
```bash
aiva_internal_executor.py --flow 265 --dry-run
```

---

### SSRFDetector._issue_to_finding → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: SSRFDetector._issue_to_finding到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供SSRFDetector._issue_to_finding的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 365: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 365
```
```bash
aiva_internal_executor.py --flow 365 --dry-run
```

---

### SmartIDORDetector._test_horizontal_access → context.add_error

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector._test_horizontal_access到context.add_error的處理能力
- **使用時機**: 當需要對SmartIDORDetector._test_horizontal_access進行深度分析並生成context.add_error結果時
- **預期結果**: 獲得基於程式邏輯的context.add_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 86: 核心：Context.Add Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Context.Add Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 86
```
```bash
aiva_internal_executor.py --flow 86 --dry-run
```

---

### SmartIDORDetector._test_vertical_access → context.add_error

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector._test_vertical_access到context.add_error的處理能力
- **使用時機**: 當需要對SmartIDORDetector._test_vertical_access進行深度分析並生成context.add_error結果時
- **預期結果**: 獲得基於程式邏輯的context.add_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 87: 核心：Context.Add Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Context.Add Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 87
```
```bash
aiva_internal_executor.py --flow 87 --dry-run
```

---

### StoredXSSDetector._check_stored_execution → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: StoredXSSDetector._check_stored_execution到vulnerabilities.append的處理能力
- **使用時機**: 當需要為外部用戶提供StoredXSSDetector._check_stored_execution的vulnerabilities.append服務時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 546: 核心：Vulnerabilities.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 546
```
```bash
aiva_internal_executor.py --flow 546 --dry-run
```

---

### TimeDetectionEngine._build_detection_result → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: TimeDetectionEngine._build_detection_result到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供TimeDetectionEngine._build_detection_result的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 240: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 240
```
```bash
aiva_internal_executor.py --flow 240 --dry-run
```

---

### UnionDetectionEngine._build_detection_result → FindingTarget

**AI描述欄位 📋**:
- **能力概要**: UnionDetectionEngine._build_detection_result到FindingTarget的處理能力
- **使用時機**: 當需要為外部用戶提供UnionDetectionEngine._build_detection_result的FindingTarget服務時
- **預期結果**: 獲得基於程式邏輯的FindingTarget結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 245: 核心：Findingtarget

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingtarget相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 245
```
```bash
aiva_internal_executor.py --flow 245 --dry-run
```

---

### VulnerabilityCorrelation → Field

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityCorrelation到Field的處理能力
- **使用時機**: 當需要為外部用戶提供VulnerabilityCorrelation的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 25: 核心：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 25
```
```bash
aiva_internal_executor.py --flow 25 --dry-run
```

---

### VulnerabilityScanner._scan_clickjacking → self.vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityScanner._scan_clickjacking到self.vulnerabilities.append的處理能力
- **使用時機**: 當需要對VulnerabilityScanner._scan_clickjacking進行深度分析並生成self.vulnerabilities.append結果時
- **預期結果**: 獲得部分AI輔助的self.vulnerabilities.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 392: 核心：Self.Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self.Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 392
```
```bash
aiva_internal_executor.py --flow 392 --dry-run
```

---

### VulnerabilityScanner._scan_directory_traversal → self.vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityScanner._scan_directory_traversal到self.vulnerabilities.append的處理能力
- **使用時機**: 當需要對VulnerabilityScanner._scan_directory_traversal進行深度分析並生成self.vulnerabilities.append結果時
- **預期結果**: 獲得部分AI輔助的self.vulnerabilities.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 390: 核心：Self.Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self.Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 390
```
```bash
aiva_internal_executor.py --flow 390 --dry-run
```

---

### VulnerabilityScanner._scan_security_headers → self.vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityScanner._scan_security_headers到self.vulnerabilities.append的處理能力
- **使用時機**: 當需要對VulnerabilityScanner._scan_security_headers進行深度分析並生成self.vulnerabilities.append結果時
- **預期結果**: 獲得部分AI輔助的self.vulnerabilities.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 391: 核心：Self.Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self.Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 391
```
```bash
aiva_internal_executor.py --flow 391 --dry-run
```

---

### VulnerabilityScanner._scan_sql_injection → self.vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityScanner._scan_sql_injection到self.vulnerabilities.append的處理能力
- **使用時機**: 當需要對VulnerabilityScanner._scan_sql_injection進行深度分析並生成self.vulnerabilities.append結果時
- **預期結果**: 獲得部分AI輔助的self.vulnerabilities.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 389: 核心：Self.Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self.Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 389
```
```bash
aiva_internal_executor.py --flow 389 --dry-run
```

---

### VulnerabilityScanner._scan_xss → self.vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityScanner._scan_xss到self.vulnerabilities.append的處理能力
- **使用時機**: 當需要對VulnerabilityScanner._scan_xss進行深度分析並生成self.vulnerabilities.append結果時
- **預期結果**: 獲得部分AI輔助的self.vulnerabilities.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 388: 核心：Self.Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self.Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 388
```
```bash
aiva_internal_executor.py --flow 388 --dry-run
```

---

### VulnerabilityScanner.scan_vulnerabilities → self._scan_clickjacking

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityScanner.scan_vulnerabilities到self._scan_clickjacking的處理能力
- **使用時機**: 當需要對VulnerabilityScanner.scan_vulnerabilities進行深度分析並生成self._scan_clickjacking結果時
- **預期結果**: 獲得部分AI輔助的self._scan_clickjacking結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 387: 核心：Self. Scan Clickjacking

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Self. Scan Clickjacking相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 387
```
```bash
aiva_internal_executor.py --flow 387 --dry-run
```

---

### VulnerabilityTestStrategy → Field

**AI描述欄位 📋**:
- **能力概要**: VulnerabilityTestStrategy到Field的處理能力
- **使用時機**: 當需要為外部用戶提供VulnerabilityTestStrategy的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 34: 核心：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 34
```
```bash
aiva_internal_executor.py --flow 34 --dry-run
```

---

### WebAttackCLI._vulnerability_scan → table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._vulnerability_scan到table.add_row的處理能力
- **使用時機**: 當需要對WebAttackCLI._vulnerability_scan進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得部分AI輔助的table.add_row結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 404: 核心：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 404
```
```bash
aiva_internal_executor.py --flow 404 --dry-run
```

---

### WebAttackCLI.run_interactive → self._export_results

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI.run_interactive到self._export_results的處理能力
- **使用時機**: 當需要對WebAttackCLI.run_interactive進行深度分析並生成self._export_results結果時
- **預期結果**: 獲得基於程式邏輯的self._export_results結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 400: 核心：Self. Export Results

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Self. Export Results相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 400
```
```bash
aiva_internal_executor.py --flow 400 --dry-run
```

---

### WebAttackCapability.__init__ → WebAttackCLI

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability.__init__到WebAttackCLI的處理能力
- **使用時機**: 當外部系統需要簡單的WebAttackCapability.__init__到WebAttackCLI轉換時
- **預期結果**: 獲得基於程式邏輯的WebAttackCLI結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 409: 核心：Webattackcli

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Webattackcli的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 409
```
```bash
aiva_internal_executor.py --flow 409 --dry-run
```

---

### WebAttackCapability.cleanup → self.manager.scan_results.clear

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability.cleanup到self.manager.scan_results.clear的處理能力
- **使用時機**: 當需要快速從WebAttackCapability.cleanup獲取self.manager.scan_results.clear的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.manager.scan_results.clear結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 418: 核心：Self.Manager.Scan Results.Clear

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Self.Manager.Scan Results.Clear的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 418
```
```bash
aiva_internal_executor.py --flow 418 --dry-run
```

---

### WebAttackCapability.initialize → __import__

**AI描述欄位 📋**:
- **能力概要**: WebAttackCapability.initialize到__import__的處理能力
- **使用時機**: 當外部系統需要簡單的WebAttackCapability.initialize到__import__轉換時
- **預期結果**: 獲得基於程式邏輯的__import__結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 410: 核心：  Import  

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：  Import  的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 410
```
```bash
aiva_internal_executor.py --flow 410 --dry-run
```

---

### WebAttackManager.__init__ → TechnologyDetector

**AI描述欄位 📋**:
- **能力概要**: WebAttackManager.__init__到TechnologyDetector的處理能力
- **使用時機**: 當需要為外部用戶提供WebAttackManager.__init__的TechnologyDetector服務時
- **預期結果**: 獲得部分AI輔助的TechnologyDetector結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 397: 核心：Technologydetector

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行核心：Technologydetector相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 397
```
```bash
aiva_internal_executor.py --flow 397 --dry-run
```

---

### _check_go_availability → _find_go_binary

**AI描述欄位 📋**:
- **能力概要**: _check_go_availability到_find_go_binary的處理能力
- **使用時機**: 當外部系統需要簡單的_check_go_availability到_find_go_binary轉換時
- **預期結果**: 獲得基於程式邏輯的_find_go_binary結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 11: 核心： Find Go Binary

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心： Find Go Binary的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 11
```
```bash
aiva_internal_executor.py --flow 11 --dry-run
```

---

### _collect_blind_callbacks → stats_collector.record_vulnerability

**AI描述欄位 📋**:
- **能力概要**: _collect_blind_callbacks到stats_collector.record_vulnerability的處理能力
- **使用時機**: 當需要為外部用戶提供_collect_blind_callbacks的stats_collector.record_vulnerability服務時
- **預期結果**: 獲得基於程式邏輯的stats_collector.record_vulnerability結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 492: 核心：Stats Collector.Record Vulnerability

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Stats Collector.Record Vulnerability相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 492
```
```bash
aiva_internal_executor.py --flow 492 --dry-run
```

---

### create_bizlogic_finding → FindingEvidence

**AI描述欄位 📋**:
- **能力概要**: create_bizlogic_finding到FindingEvidence的處理能力
- **使用時機**: 當需要為外部用戶提供create_bizlogic_finding的FindingEvidence服務時
- **預期結果**: 獲得基於程式邏輯的FindingEvidence結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 35: 核心：Findingevidence

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行核心：Findingevidence相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 35
```
```bash
aiva_internal_executor.py --flow 35 --dry-run
```

---

### register_capability → RealCapabilityRegistry.register_capability

**AI描述欄位 📋**:
- **能力概要**: register_capability到RealCapabilityRegistry.register_capability的處理能力
- **使用時機**: 當外部系統需要簡單的register_capability到RealCapabilityRegistry.register_capability轉換時
- **預期結果**: 獲得基於程式邏輯的RealCapabilityRegistry.register_capability結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 419: 核心：Realcapabilityregistry.Register Capability

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供核心：Realcapabilityregistry.Register Capability的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 419
```
```bash
aiva_internal_executor.py --flow 419 --dry-run
```

---

## 服務骨幹模組

**模組統計**:
- 總能力數量: 446
- 內部能力: 168
- 外部能力: 278

### AssetAnalysis → Field

**AI描述欄位 📋**:
- **能力概要**: AssetAnalysis到Field的處理能力
- **使用時機**: 當需要為外部用戶提供AssetAnalysis的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 26: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 26
```
```bash
aiva_internal_executor.py --flow 26 --dry-run
```

---

### AttackPath → Field

**AI描述欄位 📋**:
- **能力概要**: AttackPath到Field的處理能力
- **使用時機**: 當需要為外部用戶提供AttackPath的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 17: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 17
```
```bash
aiva_internal_executor.py --flow 17 --dry-run
```

---

### AttackPathNode → Field

**AI描述欄位 📋**:
- **能力概要**: AttackPathNode到Field的處理能力
- **使用時機**: 當需要為外部用戶提供AttackPathNode的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 16: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 16
```
```bash
aiva_internal_executor.py --flow 16 --dry-run
```

---

### AttackSurfaceAnalysis → Field

**AI描述欄位 📋**:
- **能力概要**: AttackSurfaceAnalysis到Field的處理能力
- **使用時機**: 當需要為外部用戶提供AttackSurfaceAnalysis的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 31: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 31
```
```bash
aiva_internal_executor.py --flow 31 --dry-run
```

---

### AuthnManager._convert_go_finding_to_payload → severity_map.get

**AI描述欄位 📋**:
- **能力概要**: AuthnManager._convert_go_finding_to_payload到severity_map.get的處理能力
- **使用時機**: 當外部系統需要簡單的AuthnManager._convert_go_finding_to_payload到severity_map.get轉換時
- **預期結果**: 獲得基於程式邏輯的severity_map.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 7: 服務：Severity Map.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Severity Map.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 7
```
```bash
aiva_internal_executor.py --flow 7 --dry-run
```

---

### AuthnManager._find_go_binary → Path

**AI描述欄位 📋**:
- **能力概要**: AuthnManager._find_go_binary到Path的處理能力
- **使用時機**: 當外部系統需要簡單的AuthnManager._find_go_binary到Path轉換時
- **預期結果**: 獲得基於程式邏輯的Path結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 2: 服務：Path

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Path的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 2
```
```bash
aiva_internal_executor.py --flow 2 --dry-run
```

---

### AuthnManager._generate_summary → severity_counts.get

**AI描述欄位 📋**:
- **能力概要**: AuthnManager._generate_summary到severity_counts.get的處理能力
- **使用時機**: 當外部系統需要簡單的AuthnManager._generate_summary到severity_counts.get轉換時
- **預期結果**: 獲得基於程式邏輯的severity_counts.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 8: 服務：Severity Counts.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Severity Counts.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 8
```
```bash
aiva_internal_executor.py --flow 8 --dry-run
```

---

### AuthnManager._python_fallback → findings.append

**AI描述欄位 📋**:
- **能力概要**: AuthnManager._python_fallback到findings.append的處理能力
- **使用時機**: 當需要對AuthnManager._python_fallback進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 5: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 5
```
```bash
aiva_internal_executor.py --flow 5 --dry-run
```

---

### AuthnManager._run_go_engine → findings.append

**AI描述欄位 📋**:
- **能力概要**: AuthnManager._run_go_engine到findings.append的處理能力
- **使用時機**: 當需要對AuthnManager._run_go_engine進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 4: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 4
```
```bash
aiva_internal_executor.py --flow 4 --dry-run
```

---

### AuthnManager.scan → self._generate_summary

**AI描述欄位 📋**:
- **能力概要**: AuthnManager.scan到self._generate_summary的處理能力
- **使用時機**: 當需要為外部用戶提供AuthnManager.scan的self._generate_summary服務時
- **預期結果**: 獲得基於程式邏輯的self._generate_summary結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 3: 服務：Self. Generate Summary

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Generate Summary相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 3
```
```bash
aiva_internal_executor.py --flow 3 --dry-run
```

---

### BackendDbFingerprinter._contains_sql_keywords → found_keywords.append

**AI描述欄位 📋**:
- **能力概要**: BackendDbFingerprinter._contains_sql_keywords到found_keywords.append的處理能力
- **使用時機**: 當外部系統需要簡單的BackendDbFingerprinter._contains_sql_keywords到found_keywords.append轉換時
- **預期結果**: 獲得基於程式邏輯的found_keywords.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 154: 服務：Found Keywords.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Found Keywords.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 154
```
```bash
aiva_internal_executor.py --flow 154 --dry-run
```

---

### BackendDbFingerprinter._extract_error_signatures → error_signatures.extend

**AI描述欄位 📋**:
- **能力概要**: BackendDbFingerprinter._extract_error_signatures到error_signatures.extend的處理能力
- **使用時機**: 當外部系統需要簡單的BackendDbFingerprinter._extract_error_signatures到error_signatures.extend轉換時
- **預期結果**: 獲得基於程式邏輯的error_signatures.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 155: 服務：Error Signatures.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Error Signatures.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 155
```
```bash
aiva_internal_executor.py --flow 155 --dry-run
```

---

### BackendDbFingerprinter._extract_version → match.group

**AI描述欄位 📋**:
- **能力概要**: BackendDbFingerprinter._extract_version到match.group的處理能力
- **使用時機**: 當需要快速從BackendDbFingerprinter._extract_version獲取match.group的基礎信息時
- **預期結果**: 獲得基於程式邏輯的match.group結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 152: 服務：Match.Group

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Match.Group的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 152
```
```bash
aiva_internal_executor.py --flow 152 --dry-run
```

---

### BackendDbFingerprinter.fingerprint → self._extract_version

**AI描述欄位 📋**:
- **能力概要**: BackendDbFingerprinter.fingerprint到self._extract_version的處理能力
- **使用時機**: 當需要快速從BackendDbFingerprinter.fingerprint獲取self._extract_version的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._extract_version結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 151: 服務：Self. Extract Version

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Extract Version的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 151
```
```bash
aiva_internal_executor.py --flow 151 --dry-run
```

---

### BizLogicManager.__init__ → self.logger.info

**AI描述欄位 📋**:
- **能力概要**: BizLogicManager.__init__到self.logger.info的處理能力
- **使用時機**: 當需要快速從BizLogicManager.__init__獲取self.logger.info的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.logger.info結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 61: 服務：Self.Logger.Info

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Logger.Info的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 61
```
```bash
aiva_internal_executor.py --flow 61 --dry-run
```

---

### BizLogicManager._wrap_price_test → self.logger.error

**AI描述欄位 📋**:
- **能力概要**: BizLogicManager._wrap_price_test到self.logger.error的處理能力
- **使用時機**: 當需要快速從BizLogicManager._wrap_price_test獲取self.logger.error的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.logger.error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 65: 服務：Self.Logger.Error

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Logger.Error的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 65
```
```bash
aiva_internal_executor.py --flow 65 --dry-run
```

---

### BizLogicManager._wrap_race_condition_test → self.logger.error

**AI描述欄位 📋**:
- **能力概要**: BizLogicManager._wrap_race_condition_test到self.logger.error的處理能力
- **使用時機**: 當需要快速從BizLogicManager._wrap_race_condition_test獲取self.logger.error的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.logger.error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 64: 服務：Self.Logger.Error

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Logger.Error的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 64
```
```bash
aiva_internal_executor.py --flow 64 --dry-run
```

---

### BizLogicManager._wrap_workflow_test → self.logger.error

**AI描述欄位 📋**:
- **能力概要**: BizLogicManager._wrap_workflow_test到self.logger.error的處理能力
- **使用時機**: 當需要快速從BizLogicManager._wrap_workflow_test獲取self.logger.error的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.logger.error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 66: 服務：Self.Logger.Error

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Logger.Error的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 66
```
```bash
aiva_internal_executor.py --flow 66 --dry-run
```

---

### BizLogicManager.comprehensive_scan → options.get

**AI描述欄位 📋**:
- **能力概要**: BizLogicManager.comprehensive_scan到options.get的處理能力
- **使用時機**: 當需要為外部用戶提供BizLogicManager.comprehensive_scan的options.get服務時
- **預期結果**: 獲得部分AI輔助的options.get結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 62: 服務：Options.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Options.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 62
```
```bash
aiva_internal_executor.py --flow 62 --dry-run
```

---

### BlindSQLInjectionScanner._test_boolean_blind_injection → results.append

**AI描述欄位 📋**:
- **能力概要**: BlindSQLInjectionScanner._test_boolean_blind_injection到results.append的處理能力
- **使用時機**: 當需要對BlindSQLInjectionScanner._test_boolean_blind_injection進行深度分析並生成results.append結果時
- **預期結果**: 獲得部分AI輔助的results.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 283: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 283
```
```bash
aiva_internal_executor.py --flow 283 --dry-run
```

---

### BlindSQLInjectionScanner._test_time_blind_injection → results.append

**AI描述欄位 📋**:
- **能力概要**: BlindSQLInjectionScanner._test_time_blind_injection到results.append的處理能力
- **使用時機**: 當需要對BlindSQLInjectionScanner._test_time_blind_injection進行深度分析並生成results.append結果時
- **預期結果**: 獲得部分AI輔助的results.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 282: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 282
```
```bash
aiva_internal_executor.py --flow 282 --dry-run
```

---

### BlindSQLInjectionScanner.scan_blind_injection → results.extend

**AI描述欄位 📋**:
- **能力概要**: BlindSQLInjectionScanner.scan_blind_injection到results.extend的處理能力
- **使用時機**: 當需要對BlindSQLInjectionScanner.scan_blind_injection進行深度分析並生成results.extend結果時
- **預期結果**: 獲得部分AI輔助的results.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 281: 服務：Results.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 281
```
```bash
aiva_internal_executor.py --flow 281 --dry-run
```

---

### BlindXSSDetector.__init__ → self._generate_blind_payloads

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector.__init__到self._generate_blind_payloads的處理能力
- **使用時機**: 當需要快速從BlindXSSDetector.__init__獲取self._generate_blind_payloads的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._generate_blind_payloads結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 547: 服務：Self. Generate Blind Payloads

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Generate Blind Payloads的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 547
```
```bash
aiva_internal_executor.py --flow 547 --dry-run
```

---

### BlindXSSDetector._submit_blind_payloads → method

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector._submit_blind_payloads到method的處理能力
- **使用時機**: 當外部系統需要簡單的BlindXSSDetector._submit_blind_payloads到method轉換時
- **預期結果**: 獲得基於程式邏輯的method結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 549: 服務：Method

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Method的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 549
```
```bash
aiva_internal_executor.py --flow 549 --dry-run
```

---

### BlindXSSDetector._submit_via_forms → session.get

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector._submit_via_forms到session.get的處理能力
- **使用時機**: 當需要為外部用戶提供BlindXSSDetector._submit_via_forms的session.get服務時
- **預期結果**: 獲得基於程式邏輯的session.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 550: 服務：Session.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Session.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 550
```
```bash
aiva_internal_executor.py --flow 550 --dry-run
```

---

### BlindXSSDetector._submit_via_headers → session.get

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector._submit_via_headers到session.get的處理能力
- **使用時機**: 當需要為外部用戶提供BlindXSSDetector._submit_via_headers的session.get服務時
- **預期結果**: 獲得基於程式邏輯的session.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 552: 服務：Session.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Session.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 552
```
```bash
aiva_internal_executor.py --flow 552 --dry-run
```

---

### BlindXSSDetector._submit_via_parameters → session.get

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector._submit_via_parameters到session.get的處理能力
- **使用時機**: 當外部系統需要簡單的BlindXSSDetector._submit_via_parameters到session.get轉換時
- **預期結果**: 獲得基於程式邏輯的session.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 551: 服務：Session.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Session.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 551
```
```bash
aiva_internal_executor.py --flow 551 --dry-run
```

---

### BlindXSSDetector._submit_via_user_agent → session.get

**AI描述欄位 📋**:
- **能力概要**: BlindXSSDetector._submit_via_user_agent到session.get的處理能力
- **使用時機**: 當外部系統需要簡單的BlindXSSDetector._submit_via_user_agent到session.get轉換時
- **預期結果**: 獲得基於程式邏輯的session.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 553: 服務：Session.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Session.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 553
```
```bash
aiva_internal_executor.py --flow 553 --dry-run
```

---

### BlindXssListenerValidator.__init__ → OastHttpCallbackStore

**AI描述欄位 📋**:
- **能力概要**: BlindXssListenerValidator.__init__到OastHttpCallbackStore的處理能力
- **使用時機**: 當外部系統需要簡單的BlindXssListenerValidator.__init__到OastHttpCallbackStore轉換時
- **預期結果**: 獲得基於程式邏輯的OastHttpCallbackStore結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 449: 服務：Oasthttpcallbackstore

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Oasthttpcallbackstore的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 449
```
```bash
aiva_internal_executor.py --flow 449 --dry-run
```

---

### BlindXssListenerValidator.collect_events → self._store.fetch_events

**AI描述欄位 📋**:
- **能力概要**: BlindXssListenerValidator.collect_events到self._store.fetch_events的處理能力
- **使用時機**: 當需要快速從BlindXssListenerValidator.collect_events獲取self._store.fetch_events的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._store.fetch_events結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 451: 服務：Self. Store.Fetch Events

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Store.Fetch Events的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 451
```
```bash
aiva_internal_executor.py --flow 451 --dry-run
```

---

### BlindXssListenerValidator.provision_payload → self._store.register_probe

**AI描述欄位 📋**:
- **能力概要**: BlindXssListenerValidator.provision_payload到self._store.register_probe的處理能力
- **使用時機**: 當需要快速從BlindXssListenerValidator.provision_payload獲取self._store.register_probe的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._store.register_probe結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 450: 服務：Self. Store.Register Probe

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Store.Register Probe的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 450
```
```bash
aiva_internal_executor.py --flow 450 --dry-run
```

---

### BooleanDetectionEngine._get_baseline_response → encoder.encode

**AI描述欄位 📋**:
- **能力概要**: BooleanDetectionEngine._get_baseline_response到encoder.encode的處理能力
- **使用時機**: 當外部系統需要簡單的BooleanDetectionEngine._get_baseline_response到encoder.encode轉換時
- **預期結果**: 獲得基於程式邏輯的encoder.encode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 217: 服務：Encoder.Encode

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Encoder.Encode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 217
```
```bash
aiva_internal_executor.py --flow 217 --dry-run
```

---

### BooleanDetectionEngine._send_payload_request → encoder.encode

**AI描述欄位 📋**:
- **能力概要**: BooleanDetectionEngine._send_payload_request到encoder.encode的處理能力
- **使用時機**: 當外部系統需要簡單的BooleanDetectionEngine._send_payload_request到encoder.encode轉換時
- **預期結果**: 獲得基於程式邏輯的encoder.encode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 218: 服務：Encoder.Encode

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Encoder.Encode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 218
```
```bash
aiva_internal_executor.py --flow 218 --dry-run
```

---

### BooleanDetectionEngine.detect → results.append

**AI描述欄位 📋**:
- **能力概要**: BooleanDetectionEngine.detect到results.append的處理能力
- **使用時機**: 當需要對BooleanDetectionEngine.detect進行深度分析並生成results.append結果時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 216: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 216
```
```bash
aiva_internal_executor.py --flow 216 --dry-run
```

---

### BountyHunterCLI.__init__ → BountyHunterManager

**AI描述欄位 📋**:
- **能力概要**: BountyHunterCLI.__init__到BountyHunterManager的處理能力
- **使用時機**: 當外部系統需要簡單的BountyHunterCLI.__init__到BountyHunterManager轉換時
- **預期結果**: 獲得基於程式邏輯的BountyHunterManager結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 257: 服務：Bountyhuntermanager

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Bountyhuntermanager的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 257
```
```bash
aiva_internal_executor.py --flow 257 --dry-run
```

---

### BountyHunterCLI._add_targets → self.manager.add_high_value_target

**AI描述欄位 📋**:
- **能力概要**: BountyHunterCLI._add_targets到self.manager.add_high_value_target的處理能力
- **使用時機**: 當需要對BountyHunterCLI._add_targets進行深度分析並生成self.manager.add_high_value_target結果時
- **預期結果**: 獲得基於程式邏輯的self.manager.add_high_value_target結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 259: 服務：Self.Manager.Add High Value Target

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Manager.Add High Value Target相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 259
```
```bash
aiva_internal_executor.py --flow 259 --dry-run
```

---

### BountyHunterCLI._generate_report → f.write

**AI描述欄位 📋**:
- **能力概要**: BountyHunterCLI._generate_report到f.write的處理能力
- **使用時機**: 當需要對BountyHunterCLI._generate_report進行深度分析並生成f.write結果時
- **預期結果**: 獲得基於程式邏輯的f.write結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 262: 服務：F.Write

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：F.Write相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 262
```
```bash
aiva_internal_executor.py --flow 262 --dry-run
```

---

### BountyHunterCLI._show_vulnerabilities → table.add_row

**AI描述欄位 📋**:
- **能力概要**: BountyHunterCLI._show_vulnerabilities到table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供BountyHunterCLI._show_vulnerabilities的table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 261: 服務：Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 261
```
```bash
aiva_internal_executor.py --flow 261 --dry-run
```

---

### BountyHunterCLI._start_hunting → table.add_row

**AI描述欄位 📋**:
- **能力概要**: BountyHunterCLI._start_hunting到table.add_row的處理能力
- **使用時機**: 當需要對BountyHunterCLI._start_hunting進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 260: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 260
```
```bash
aiva_internal_executor.py --flow 260 --dry-run
```

---

### BountyHunterCLI.run → self._show_statistics

**AI描述欄位 📋**:
- **能力概要**: BountyHunterCLI.run到self._show_statistics的處理能力
- **使用時機**: 當需要對BountyHunterCLI.run進行深度分析並生成self._show_statistics結果時
- **預期結果**: 獲得基於程式邏輯的self._show_statistics結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 258: 服務：Self. Show Statistics

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Show Statistics相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 258
```
```bash
aiva_internal_executor.py --flow 258 --dry-run
```

---

### BountyHunterManager.__init__ → BountyHunterScanner

**AI描述欄位 📋**:
- **能力概要**: BountyHunterManager.__init__到BountyHunterScanner的處理能力
- **使用時機**: 當外部系統需要簡單的BountyHunterManager.__init__到BountyHunterScanner轉換時
- **預期結果**: 獲得部分AI輔助的BountyHunterScanner結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 254: 服務：Bountyhunterscanner

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Bountyhunterscanner的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 254
```
```bash
aiva_internal_executor.py --flow 254 --dry-run
```

---

### BountyHunterManager.add_high_value_target → self.target_queue.append

**AI描述欄位 📋**:
- **能力概要**: BountyHunterManager.add_high_value_target到self.target_queue.append的處理能力
- **使用時機**: 當需要對BountyHunterManager.add_high_value_target進行深度分析並生成self.target_queue.append結果時
- **預期結果**: 獲得基於程式邏輯的self.target_queue.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 255: 服務：Self.Target Queue.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Target Queue.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 255
```
```bash
aiva_internal_executor.py --flow 255 --dry-run
```

---

### BountyHunterScanner.__init__ → self._load_fp_filters

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner.__init__到self._load_fp_filters的處理能力
- **使用時機**: 當需要對BountyHunterScanner.__init__進行深度分析並生成self._load_fp_filters結果時
- **預期結果**: 獲得部分AI輔助的self._load_fp_filters結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 246: 服務：Self. Load Fp Filters

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self. Load Fp Filters相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 246
```
```bash
aiva_internal_executor.py --flow 246 --dry-run
```

---

### BountyHunterScanner._get_baseline_response → response.text

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner._get_baseline_response到response.text的處理能力
- **使用時機**: 當需要對BountyHunterScanner._get_baseline_response進行深度分析並生成response.text結果時
- **預期結果**: 獲得部分AI輔助的response.text結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 253: 服務：Response.Text

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Response.Text相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 253
```
```bash
aiva_internal_executor.py --flow 253 --dry-run
```

---

### BountyHunterScanner._is_false_positive → content.lower

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner._is_false_positive到content.lower的處理能力
- **使用時機**: 當外部系統需要簡單的BountyHunterScanner._is_false_positive到content.lower轉換時
- **預期結果**: 獲得部分AI輔助的content.lower結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 251: 服務：Content.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Content.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 251
```
```bash
aiva_internal_executor.py --flow 251 --dry-run
```

---

### BountyHunterScanner._test_payload_type → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner._test_payload_type到vulnerabilities.append的處理能力
- **使用時機**: 當需要對BountyHunterScanner._test_payload_type進行深度分析並生成vulnerabilities.append結果時
- **預期結果**: 獲得部分AI輔助的vulnerabilities.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 248: 服務：Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 248
```
```bash
aiva_internal_executor.py --flow 248 --dry-run
```

---

### BountyHunterScanner._test_single_payload → self._analyze_bounty_response

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner._test_single_payload到self._analyze_bounty_response的處理能力
- **使用時機**: 當需要對BountyHunterScanner._test_single_payload進行深度分析並生成self._analyze_bounty_response結果時
- **預期結果**: 獲得部分AI輔助的self._analyze_bounty_response結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 249: 服務：Self. Analyze Bounty Response

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self. Analyze Bounty Response相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 249
```
```bash
aiva_internal_executor.py --flow 249 --dry-run
```

---

### BountyHunterScanner.scan_high_value_target → vulnerabilities.extend

**AI描述欄位 📋**:
- **能力概要**: BountyHunterScanner.scan_high_value_target到vulnerabilities.extend的處理能力
- **使用時機**: 當需要對BountyHunterScanner.scan_high_value_target進行深度分析並生成vulnerabilities.extend結果時
- **預期結果**: 獲得部分AI輔助的vulnerabilities.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 247: 服務：Vulnerabilities.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vulnerabilities.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 247
```
```bash
aiva_internal_executor.py --flow 247 --dry-run
```

---

### CrossLanguageXSSEngine.__del__ → self.cleanup

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine.__del__到self.cleanup的處理能力
- **使用時機**: 當需要快速從CrossLanguageXSSEngine.__del__獲取self.cleanup的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.cleanup結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 532: 服務：Self.Cleanup

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Cleanup的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 532
```
```bash
aiva_internal_executor.py --flow 532 --dry-run
```

---

### CrossLanguageXSSEngine.__init__ → Path

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine.__init__到Path的處理能力
- **使用時機**: 當需要為外部用戶提供CrossLanguageXSSEngine.__init__的Path服務時
- **預期結果**: 獲得基於程式邏輯的Path結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 505: 服務：Path

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Path相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 505
```
```bash
aiva_internal_executor.py --flow 505 --dry-run
```

---

### CrossLanguageXSSEngine._check_go_environment → shutil.which

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_go_environment到shutil.which的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_go_environment進行深度分析並生成shutil.which結果時
- **預期結果**: 獲得基於程式邏輯的shutil.which結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 512: 服務：Shutil.Which

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Shutil.Which相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 512
```
```bash
aiva_internal_executor.py --flow 512 --dry-run
```

---

### CrossLanguageXSSEngine._check_python_environment → shutil.which

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_python_environment到shutil.which的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_python_environment進行深度分析並生成shutil.which結果時
- **預期結果**: 獲得基於程式邏輯的shutil.which結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 514: 服務：Shutil.Which

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Shutil.Which相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 514
```
```bash
aiva_internal_executor.py --flow 514 --dry-run
```

---

### CrossLanguageXSSEngine._check_ruby_environment → shutil.which

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_ruby_environment到shutil.which的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_ruby_environment進行深度分析並生成shutil.which結果時
- **預期結果**: 獲得基於程式邏輯的shutil.which結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 513: 服務：Shutil.Which

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Shutil.Which相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 513
```
```bash
aiva_internal_executor.py --flow 513 --dry-run
```

---

### CrossLanguageXSSEngine._check_rust_environment → shutil.which

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._check_rust_environment到shutil.which的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._check_rust_environment進行深度分析並生成shutil.which結果時
- **預期結果**: 獲得基於程式邏輯的shutil.which結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 515: 服務：Shutil.Which

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Shutil.Which相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 515
```
```bash
aiva_internal_executor.py --flow 515 --dry-run
```

---

### CrossLanguageXSSEngine._create_result_from_json → item.get

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._create_result_from_json到item.get的處理能力
- **使用時機**: 當需要為外部用戶提供CrossLanguageXSSEngine._create_result_from_json的item.get服務時
- **預期結果**: 獲得基於程式邏輯的item.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 525: 服務：Item.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Item.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 525
```
```bash
aiva_internal_executor.py --flow 525 --dry-run
```

---

### CrossLanguageXSSEngine._detect_language_environments → LanguageEnvironment

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._detect_language_environments到LanguageEnvironment的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine._detect_language_environments進行深度分析並生成LanguageEnvironment結果時
- **預期結果**: 獲得基於程式邏輯的LanguageEnvironment結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 511: 服務：Languageenvironment

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Languageenvironment相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 511
```
```bash
aiva_internal_executor.py --flow 511 --dry-run
```

---

### CrossLanguageXSSEngine._is_language_available → self.language_environments.get

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._is_language_available到self.language_environments.get的處理能力
- **使用時機**: 當需要快速從CrossLanguageXSSEngine._is_language_available獲取self.language_environments.get的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.language_environments.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 529: 服務：Self.Language Environments.Get

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Language Environments.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 529
```
```bash
aiva_internal_executor.py --flow 529 --dry-run
```

---

### CrossLanguageXSSEngine._parse_regex_output → self._process_regex_matches

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._parse_regex_output到self._process_regex_matches的處理能力
- **使用時機**: 當需要快速從CrossLanguageXSSEngine._parse_regex_output獲取self._process_regex_matches的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._process_regex_matches結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 526: 服務：Self. Process Regex Matches

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Process Regex Matches的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 526
```
```bash
aiva_internal_executor.py --flow 526 --dry-run
```

---

### CrossLanguageXSSEngine._parse_tool_output → self._parse_regex_output

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._parse_tool_output到self._parse_regex_output的處理能力
- **使用時機**: 當需要快速從CrossLanguageXSSEngine._parse_tool_output獲取self._parse_regex_output的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._parse_regex_output結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 524: 服務：Self. Parse Regex Output

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Parse Regex Output的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 524
```
```bash
aiva_internal_executor.py --flow 524 --dry-run
```

---

### CrossLanguageXSSEngine._process_regex_matches → float

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._process_regex_matches到float的處理能力
- **使用時機**: 當外部系統需要簡單的CrossLanguageXSSEngine._process_regex_matches到float轉換時
- **預期結果**: 獲得基於程式邏輯的float結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 527: 服務：Float

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Float的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 527
```
```bash
aiva_internal_executor.py --flow 527 --dry-run
```

---

### CrossLanguageXSSEngine._run_command → process.wait

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine._run_command到process.wait的處理能力
- **使用時機**: 當外部系統需要簡單的CrossLanguageXSSEngine._run_command到process.wait轉換時
- **預期結果**: 獲得基於程式邏輯的process.wait結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 528: 服務：Process.Wait

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Process.Wait的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 528
```
```bash
aiva_internal_executor.py --flow 528 --dry-run
```

---

### CrossLanguageXSSEngine.cleanup → self.logger.error

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine.cleanup到self.logger.error的處理能力
- **使用時機**: 當需要對CrossLanguageXSSEngine.cleanup進行深度分析並生成self.logger.error結果時
- **預期結果**: 獲得基於程式邏輯的self.logger.error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 531: 服務：Self.Logger.Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Logger.Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 531
```
```bash
aiva_internal_executor.py --flow 531 --dry-run
```

---

### CrossLanguageXSSEngine.get_available_tools → available_tools.append

**AI描述欄位 📋**:
- **能力概要**: CrossLanguageXSSEngine.get_available_tools到available_tools.append的處理能力
- **使用時機**: 當外部系統需要簡單的CrossLanguageXSSEngine.get_available_tools到available_tools.append轉換時
- **預期結果**: 獲得基於程式邏輯的available_tools.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 530: 服務：Available Tools.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Available Tools.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 530
```
```bash
aiva_internal_executor.py --flow 530 --dry-run
```

---

### CustomSQLInjectionScanner.__init__ → self._load_payloads

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner.__init__到self._load_payloads的處理能力
- **使用時機**: 當需要快速從CustomSQLInjectionScanner.__init__獲取self._load_payloads的基礎信息時
- **預期結果**: 獲得部分AI輔助的self._load_payloads結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 271: 服務：Self. Load Payloads

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Self. Load Payloads的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 271
```
```bash
aiva_internal_executor.py --flow 271 --dry-run
```

---

### CustomSQLInjectionScanner._get_baseline_response → response.text

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner._get_baseline_response到response.text的處理能力
- **使用時機**: 當需要對CustomSQLInjectionScanner._get_baseline_response進行深度分析並生成response.text結果時
- **預期結果**: 獲得部分AI輔助的response.text結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 275: 服務：Response.Text

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Response.Text相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 275
```
```bash
aiva_internal_executor.py --flow 275 --dry-run
```

---

### CustomSQLInjectionScanner._test_injection_type → results.append

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner._test_injection_type到results.append的處理能力
- **使用時機**: 當需要對CustomSQLInjectionScanner._test_injection_type進行深度分析並生成results.append結果時
- **預期結果**: 獲得部分AI輔助的results.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 274: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 274
```
```bash
aiva_internal_executor.py --flow 274 --dry-run
```

---

### CustomSQLInjectionScanner._test_payload → session.post

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner._test_payload到session.post的處理能力
- **使用時機**: 當需要為外部用戶提供CustomSQLInjectionScanner._test_payload的session.post服務時
- **預期結果**: 獲得部分AI輔助的session.post結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 276: 服務：Session.Post

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Session.Post相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 276
```
```bash
aiva_internal_executor.py --flow 276 --dry-run
```

---

### CustomSQLInjectionScanner.close → self.session.close

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner.close到self.session.close的處理能力
- **使用時機**: 當需要快速從CustomSQLInjectionScanner.close獲取self.session.close的基礎信息時
- **預期結果**: 獲得部分AI輔助的self.session.close結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 272: 服務：Self.Session.Close

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Self.Session.Close的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 272
```
```bash
aiva_internal_executor.py --flow 272 --dry-run
```

---

### CustomSQLInjectionScanner.scan_target → results.extend

**AI描述欄位 📋**:
- **能力概要**: CustomSQLInjectionScanner.scan_target到results.extend的處理能力
- **使用時機**: 當需要對CustomSQLInjectionScanner.scan_target進行深度分析並生成results.extend結果時
- **預期結果**: 獲得部分AI輔助的results.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 273: 服務：Results.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 273
```
```bash
aiva_internal_executor.py --flow 273 --dry-run
```

---

### DalfoxIntegration.__init__ → self._find_dalfox_path

**AI描述欄位 📋**:
- **能力概要**: DalfoxIntegration.__init__到self._find_dalfox_path的處理能力
- **使用時機**: 當需要快速從DalfoxIntegration.__init__獲取self._find_dalfox_path的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._find_dalfox_path結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 535: 服務：Self. Find Dalfox Path

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Find Dalfox Path的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 535
```
```bash
aiva_internal_executor.py --flow 535 --dry-run
```

---

### DalfoxIntegration.install_dalfox → process.communicate

**AI描述欄位 📋**:
- **能力概要**: DalfoxIntegration.install_dalfox到process.communicate的處理能力
- **使用時機**: 當外部系統需要簡單的DalfoxIntegration.install_dalfox到process.communicate轉換時
- **預期結果**: 獲得基於程式邏輯的process.communicate結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 536: 服務：Process.Communicate

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Process.Communicate的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 536
```
```bash
aiva_internal_executor.py --flow 536 --dry-run
```

---

### DalfoxIntegration.scan_target → self._parse_dalfox_output

**AI描述欄位 📋**:
- **能力概要**: DalfoxIntegration.scan_target到self._parse_dalfox_output的處理能力
- **使用時機**: 當需要為外部用戶提供DalfoxIntegration.scan_target的self._parse_dalfox_output服務時
- **預期結果**: 獲得基於程式邏輯的self._parse_dalfox_output結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 537: 服務：Self. Parse Dalfox Output

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Parse Dalfox Output相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 537
```
```bash
aiva_internal_executor.py --flow 537 --dry-run
```

---

### DeserializationDetector._check_deserialization_error → response_text.lower

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector._check_deserialization_error到response_text.lower的處理能力
- **使用時機**: 當外部系統需要簡單的DeserializationDetector._check_deserialization_error到response_text.lower轉換時
- **預期結果**: 獲得基於程式邏輯的response_text.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 620: 服務：Response Text.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Response Text.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 620
```
```bash
aiva_internal_executor.py --flow 620 --dry-run
```

---

### DeserializationDetector._create_python_pickle_payload → pickle.dumps

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector._create_python_pickle_payload到pickle.dumps的處理能力
- **使用時機**: 當外部系統需要簡單的DeserializationDetector._create_python_pickle_payload到pickle.dumps轉換時
- **預期結果**: 獲得基於程式邏輯的pickle.dumps結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 624: 服務：Pickle.Dumps

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Pickle.Dumps的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 624
```
```bash
aiva_internal_executor.py --flow 624 --dry-run
```

---

### DeserializationDetector._measure_baseline → times.append

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector._measure_baseline到times.append的處理能力
- **使用時機**: 當需要為外部用戶提供DeserializationDetector._measure_baseline的times.append服務時
- **預期結果**: 獲得基於程式邏輯的times.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 625: 服務：Times.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Times.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 625
```
```bash
aiva_internal_executor.py --flow 625 --dry-run
```

---

### DeserializationDetector._measure_response_time → requests.get

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector._measure_response_time到requests.get的處理能力
- **使用時機**: 當外部系統需要簡單的DeserializationDetector._measure_response_time到requests.get轉換時
- **預期結果**: 獲得基於程式邏輯的requests.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 619: 服務：Requests.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Requests.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 619
```
```bash
aiva_internal_executor.py --flow 619 --dry-run
```

---

### DeserializationDetector._test_single_payload → requests.get

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector._test_single_payload到requests.get的處理能力
- **使用時機**: 當需要為外部用戶提供DeserializationDetector._test_single_payload的requests.get服務時
- **預期結果**: 獲得基於程式邏輯的requests.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 626: 服務：Requests.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Requests.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 626
```
```bash
aiva_internal_executor.py --flow 626 --dry-run
```

---

### DeserializationDetector.generate_detection_payloads → self._create_jsonpickle_payload

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector.generate_detection_payloads到self._create_jsonpickle_payload的處理能力
- **使用時機**: 當需要對DeserializationDetector.generate_detection_payloads進行深度分析並生成self._create_jsonpickle_payload結果時
- **預期結果**: 獲得基於程式邏輯的self._create_jsonpickle_payload結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 623: 服務：Self. Create Jsonpickle Payload

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Create Jsonpickle Payload相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 623
```
```bash
aiva_internal_executor.py --flow 623 --dry-run
```

---

### DeserializationDetector.generate_java_payload_with_ysoserial → print

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector.generate_java_payload_with_ysoserial到print的處理能力
- **使用時機**: 當外部系統需要簡單的DeserializationDetector.generate_java_payload_with_ysoserial到print轉換時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 622: 服務：Print

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Print的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 622
```
```bash
aiva_internal_executor.py --flow 622 --dry-run
```

---

### DeserializationDetector.generate_payloads → self._generate_jsonnet_payload

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector.generate_payloads到self._generate_jsonnet_payload的處理能力
- **使用時機**: 當需要對DeserializationDetector.generate_payloads進行深度分析並生成self._generate_jsonnet_payload結果時
- **預期結果**: 獲得基於程式邏輯的self._generate_jsonnet_payload結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 617: 服務：Self. Generate Jsonnet Payload

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Generate Jsonnet Payload相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 617
```
```bash
aiva_internal_executor.py --flow 617 --dry-run
```

---

### DeserializationDetector.test_cookie_deserialization → findings.append

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector.test_cookie_deserialization到findings.append的處理能力
- **使用時機**: 當需要對DeserializationDetector.test_cookie_deserialization進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 621: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 621
```
```bash
aiva_internal_executor.py --flow 621 --dry-run
```

---

### DeserializationDetector.test_deserialization → findings.append

**AI描述欄位 📋**:
- **能力概要**: DeserializationDetector.test_deserialization到findings.append的處理能力
- **使用時機**: 當需要對DeserializationDetector.test_deserialization進行深度分析並生成findings.append結果時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 618: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 618
```
```bash
aiva_internal_executor.py --flow 618 --dry-run
```

---

### DirectoryBruteforcer.__init__ → self.session.headers.update

**AI描述欄位 📋**:
- **能力概要**: DirectoryBruteforcer.__init__到self.session.headers.update的處理能力
- **使用時機**: 當需要對DirectoryBruteforcer.__init__進行深度分析並生成self.session.headers.update結果時
- **預期結果**: 獲得基於程式邏輯的self.session.headers.update結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 420: 服務：Self.Session.Headers.Update

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Session.Headers.Update相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 420
```
```bash
aiva_internal_executor.py --flow 420 --dry-run
```

---

### DirectoryBruteforcer._determine_severity → response.url.lower

**AI描述欄位 📋**:
- **能力概要**: DirectoryBruteforcer._determine_severity到response.url.lower的處理能力
- **使用時機**: 當外部系統需要簡單的DirectoryBruteforcer._determine_severity到response.url.lower轉換時
- **預期結果**: 獲得基於程式邏輯的response.url.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 423: 服務：Response.Url.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Response.Url.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 423
```
```bash
aiva_internal_executor.py --flow 423 --dry-run
```

---

### DirectoryBruteforcer._test_url → self._determine_severity

**AI描述欄位 📋**:
- **能力概要**: DirectoryBruteforcer._test_url到self._determine_severity的處理能力
- **使用時機**: 當需要快速從DirectoryBruteforcer._test_url獲取self._determine_severity的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._determine_severity結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 422: 服務：Self. Determine Severity

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Determine Severity的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 422
```
```bash
aiva_internal_executor.py --flow 422 --dry-run
```

---

### DirectoryBruteforcer.scan → results.append

**AI描述欄位 📋**:
- **能力概要**: DirectoryBruteforcer.scan到results.append的處理能力
- **使用時機**: 當需要為外部用戶提供DirectoryBruteforcer.scan的results.append服務時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 421: 服務：Results.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 421
```
```bash
aiva_internal_executor.py --flow 421 --dry-run
```

---

### DirectoryScanner._check_path → self.found_directories.append

**AI描述欄位 📋**:
- **能力概要**: DirectoryScanner._check_path到self.found_directories.append的處理能力
- **使用時機**: 當需要對DirectoryScanner._check_path進行深度分析並生成self.found_directories.append結果時
- **預期結果**: 獲得部分AI輔助的self.found_directories.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 386: 服務：Self.Found Directories.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self.Found Directories.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 386
```
```bash
aiva_internal_executor.py --flow 386 --dry-run
```

---

### DnsRebindingDetector._generate_rbndr_domain → second_ip.split

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector._generate_rbndr_domain到second_ip.split的處理能力
- **使用時機**: 當外部系統需要簡單的DnsRebindingDetector._generate_rbndr_domain到second_ip.split轉換時
- **預期結果**: 獲得基於程式邏輯的second_ip.split結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 299: 服務：Second Ip.Split

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Second Ip.Split的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 299
```
```bash
aiva_internal_executor.py --flow 299 --dry-run
```

---

### DnsRebindingDetector._generate_rebind_it_domain → ip_to_hex

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector._generate_rebind_it_domain到ip_to_hex的處理能力
- **使用時機**: 當外部系統需要簡單的DnsRebindingDetector._generate_rebind_it_domain到ip_to_hex轉換時
- **預期結果**: 獲得基於程式邏輯的ip_to_hex結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 298: 服務：Ip To Hex

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Ip To Hex的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 298
```
```bash
aiva_internal_executor.py --flow 298 --dry-run
```

---

### DnsRebindingDetector._resolve_domain → socket.getaddrinfo

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector._resolve_domain到socket.getaddrinfo的處理能力
- **使用時機**: 當需要為外部用戶提供DnsRebindingDetector._resolve_domain的socket.getaddrinfo服務時
- **預期結果**: 獲得基於程式邏輯的socket.getaddrinfo結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 301: 服務：Socket.Getaddrinfo

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Socket.Getaddrinfo相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 301
```
```bash
aiva_internal_executor.py --flow 301 --dry-run
```

---

### DnsRebindingDetector.generate_payloads → payloads.append

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector.generate_payloads到payloads.append的處理能力
- **使用時機**: 當需要對DnsRebindingDetector.generate_payloads進行深度分析並生成payloads.append結果時
- **預期結果**: 獲得部分AI輔助的payloads.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 303: 服務：Payloads.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Payloads.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 303
```
```bash
aiva_internal_executor.py --flow 303 --dry-run
```

---

### DnsRebindingDetector.generate_vectors → vectors.append

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector.generate_vectors到vectors.append的處理能力
- **使用時機**: 當需要對DnsRebindingDetector.generate_vectors進行深度分析並生成vectors.append結果時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 297: 服務：Vectors.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 297
```
```bash
aiva_internal_executor.py --flow 297 --dry-run
```

---

### DnsRebindingDetector.test_rebinding → client.aclose

**AI描述欄位 📋**:
- **能力概要**: DnsRebindingDetector.test_rebinding到client.aclose的處理能力
- **使用時機**: 當需要對DnsRebindingDetector.test_rebinding進行深度分析並生成client.aclose結果時
- **預期結果**: 獲得基於程式邏輯的client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 300: 服務：Client.Aclose

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Client.Aclose相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 300
```
```bash
aiva_internal_executor.py --flow 300 --dry-run
```

---

### EncodedPayload.build_request_dump → lines.append

**AI描述欄位 📋**:
- **能力概要**: EncodedPayload.build_request_dump到lines.append的處理能力
- **使用時機**: 當需要為外部用戶提供EncodedPayload.build_request_dump的lines.append服務時
- **預期結果**: 獲得基於程式邏輯的lines.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 183: 服務：Lines.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Lines.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 183
```
```bash
aiva_internal_executor.py --flow 183 --dry-run
```

---

### EnhancedIDORWorker.__init__ → SmartIDORDetector

**AI描述欄位 📋**:
- **能力概要**: EnhancedIDORWorker.__init__到SmartIDORDetector的處理能力
- **使用時機**: 當外部系統需要簡單的EnhancedIDORWorker.__init__到SmartIDORDetector轉換時
- **預期結果**: 獲得基於程式邏輯的SmartIDORDetector結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 68: 服務：Smartidordetector

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Smartidordetector的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 68
```
```bash
aiva_internal_executor.py --flow 68 --dry-run
```

---

### EnhancedIDORWorker._convert_to_finding_payloads → findings.append

**AI描述欄位 📋**:
- **能力概要**: EnhancedIDORWorker._convert_to_finding_payloads到findings.append的處理能力
- **使用時機**: 當外部系統需要簡單的EnhancedIDORWorker._convert_to_finding_payloads到findings.append轉換時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 72: 服務：Findings.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 72
```
```bash
aiva_internal_executor.py --flow 72 --dry-run
```

---

### EnhancedIDORWorker.run → self._execute_task

**AI描述欄位 📋**:
- **能力概要**: EnhancedIDORWorker.run到self._execute_task的處理能力
- **使用時機**: 當需要為外部用戶提供EnhancedIDORWorker.run的self._execute_task服務時
- **預期結果**: 獲得基於程式邏輯的self._execute_task結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 69: 服務：Self. Execute Task

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Execute Task相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 69
```
```bash
aiva_internal_executor.py --flow 69 --dry-run
```

---

### EnhancedIdorTelemetry → field

**AI描述欄位 📋**:
- **能力概要**: EnhancedIdorTelemetry到field的處理能力
- **使用時機**: 當外部系統需要簡單的EnhancedIdorTelemetry到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 67: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 67
```
```bash
aiva_internal_executor.py --flow 67 --dry-run
```

---

### GeneralTestStrategy → Field

**AI描述欄位 📋**:
- **能力概要**: GeneralTestStrategy到Field的處理能力
- **使用時機**: 當需要為外部用戶提供GeneralTestStrategy的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 22: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 22
```
```bash
aiva_internal_executor.py --flow 22 --dry-run
```

---

### HackingToolDetectionEngine.__init__ → new_id

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine.__init__到new_id的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolDetectionEngine.__init__到new_id轉換時
- **預期結果**: 獲得基於程式邏輯的new_id結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 223: 服務：New Id

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：New Id的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 223
```
```bash
aiva_internal_executor.py --flow 223 --dry-run
```

---

### HackingToolDetectionEngine._convert_to_detection_result → DetectionResult

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._convert_to_detection_result到DetectionResult的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolDetectionEngine._convert_to_detection_result到DetectionResult轉換時
- **預期結果**: 獲得基於程式邏輯的DetectionResult結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 234: 服務：Detectionresult

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Detectionresult的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 234
```
```bash
aiva_internal_executor.py --flow 234 --dry-run
```

---

### HackingToolDetectionEngine._parse_tool_output → results.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine._parse_tool_output到results.append的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolDetectionEngine._parse_tool_output的results.append服務時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 230: 服務：Results.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 230
```
```bash
aiva_internal_executor.py --flow 230 --dry-run
```

---

### HackingToolDetectionEngine.install_missing_tools → self.integrator.install_tool

**AI描述欄位 📋**:
- **能力概要**: HackingToolDetectionEngine.install_missing_tools到self.integrator.install_tool的處理能力
- **使用時機**: 當需要快速從HackingToolDetectionEngine.install_missing_tools獲取self.integrator.install_tool的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.integrator.install_tool結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 233: 服務：Self.Integrator.Install Tool

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Integrator.Install Tool的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 233
```
```bash
aiva_internal_executor.py --flow 233 --dry-run
```

---

### HackingToolSQLCLI.generate_report → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.generate_report到print的處理能力
- **使用時機**: 當需要對HackingToolSQLCLI.generate_report進行深度分析並生成print結果時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 179: 服務：Print

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 179
```
```bash
aiva_internal_executor.py --flow 179 --dry-run
```

---

### HackingToolSQLCLI.get_recommendations → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.get_recommendations到print的處理能力
- **使用時機**: 當需要對HackingToolSQLCLI.get_recommendations進行深度分析並生成print結果時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 181: 服務：Print

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 181
```
```bash
aiva_internal_executor.py --flow 181 --dry-run
```

---

### HackingToolSQLCLI.install_all_tools → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.install_all_tools到print的處理能力
- **使用時機**: 當需要對HackingToolSQLCLI.install_all_tools進行深度分析並生成print結果時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 177: 服務：Print

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 177
```
```bash
aiva_internal_executor.py --flow 177 --dry-run
```

---

### HackingToolSQLCLI.install_tool → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.install_tool到print的處理能力
- **使用時機**: 當需要對HackingToolSQLCLI.install_tool進行深度分析並生成print結果時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 176: 服務：Print

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 176
```
```bash
aiva_internal_executor.py --flow 176 --dry-run
```

---

### HackingToolSQLCLI.list_tools → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.list_tools到print的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolSQLCLI.list_tools的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 180: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 180
```
```bash
aiva_internal_executor.py --flow 180 --dry-run
```

---

### HackingToolSQLCLI.show_status → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.show_status到print的處理能力
- **使用時機**: 當需要對HackingToolSQLCLI.show_status進行深度分析並生成print結果時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 175: 服務：Print

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 175
```
```bash
aiva_internal_executor.py --flow 175 --dry-run
```

---

### HackingToolSQLCLI.test_tool → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLCLI.test_tool到print的處理能力
- **使用時機**: 當需要對HackingToolSQLCLI.test_tool進行深度分析並生成print結果時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 178: 服務：Print

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 178
```
```bash
aiva_internal_executor.py --flow 178 --dry-run
```

---

### HackingToolSQLConfig → field

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLConfig到field的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolSQLConfig的field服務時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 159: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 159
```
```bash
aiva_internal_executor.py --flow 159 --dry-run
```

---

### HackingToolSQLIntegrator.get_available_tools → available.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLIntegrator.get_available_tools到available.append的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolSQLIntegrator.get_available_tools到available.append轉換時
- **預期結果**: 獲得基於程式邏輯的available.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 161: 服務：Available.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Available.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 161
```
```bash
aiva_internal_executor.py --flow 161 --dry-run
```

---

### HackingToolSQLIntegrator.get_enabled_tools → enabled.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLIntegrator.get_enabled_tools到enabled.append的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolSQLIntegrator.get_enabled_tools到enabled.append轉換時
- **預期結果**: 獲得基於程式邏輯的enabled.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 162: 服務：Enabled.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Enabled.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 162
```
```bash
aiva_internal_executor.py --flow 162 --dry-run
```

---

### HackingToolSQLIntegrator.install_tool → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLIntegrator.install_tool到print的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolSQLIntegrator.install_tool的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 164: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 164
```
```bash
aiva_internal_executor.py --flow 164 --dry-run
```

---

### HackingToolSQLIntegrator.run_tool → APIResponse

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLIntegrator.run_tool到APIResponse的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolSQLIntegrator.run_tool的APIResponse服務時
- **預期結果**: 獲得基於程式邏輯的APIResponse結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 165: 服務：Apiresponse

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Apiresponse相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 165
```
```bash
aiva_internal_executor.py --flow 165 --dry-run
```

---

### HackingToolSQLManager.__init__ → self.tools_dir.mkdir

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.__init__到self.tools_dir.mkdir的處理能力
- **使用時機**: 當需要對HackingToolSQLManager.__init__進行深度分析並生成self.tools_dir.mkdir結果時
- **預期結果**: 獲得基於程式邏輯的self.tools_dir.mkdir結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 166: 服務：Self.Tools Dir.Mkdir

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Tools Dir.Mkdir相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 166
```
```bash
aiva_internal_executor.py --flow 166 --dry-run
```

---

### HackingToolSQLManager._check_tool_status → self._test_tool_executable

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager._check_tool_status到self._test_tool_executable的處理能力
- **使用時機**: 當需要對HackingToolSQLManager._check_tool_status進行深度分析並生成self._test_tool_executable結果時
- **預期結果**: 獲得基於程式邏輯的self._test_tool_executable結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 168: 服務：Self. Test Tool Executable

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Test Tool Executable相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 168
```
```bash
aiva_internal_executor.py --flow 168 --dry-run
```

---

### HackingToolSQLManager.check_all_tools_status → self._check_tool_status

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.check_all_tools_status到self._check_tool_status的處理能力
- **使用時機**: 當需要快速從HackingToolSQLManager.check_all_tools_status獲取self._check_tool_status的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._check_tool_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 167: 服務：Self. Check Tool Status

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Check Tool Status的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 167
```
```bash
aiva_internal_executor.py --flow 167 --dry-run
```

---

### HackingToolSQLManager.generate_status_report → self.check_all_tools_status

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.generate_status_report到self.check_all_tools_status的處理能力
- **使用時機**: 當需要快速從HackingToolSQLManager.generate_status_report獲取self.check_all_tools_status的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.check_all_tools_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 174: 服務：Self.Check All Tools Status

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Check All Tools Status的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 174
```
```bash
aiva_internal_executor.py --flow 174 --dry-run
```

---

### HackingToolSQLManager.get_installation_script → script_lines.append

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.get_installation_script到script_lines.append的處理能力
- **使用時機**: 當需要為外部用戶提供HackingToolSQLManager.get_installation_script的script_lines.append服務時
- **預期結果**: 獲得基於程式邏輯的script_lines.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 173: 服務：Script Lines.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Script Lines.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 173
```
```bash
aiva_internal_executor.py --flow 173 --dry-run
```

---

### HackingToolSQLManager.get_tool_recommendations → recommendations.sort

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.get_tool_recommendations到recommendations.sort的處理能力
- **使用時機**: 當需要對HackingToolSQLManager.get_tool_recommendations進行深度分析並生成recommendations.sort結果時
- **預期結果**: 獲得基於程式邏輯的recommendations.sort結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 172: 服務：Recommendations.Sort

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Recommendations.Sort相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 172
```
```bash
aiva_internal_executor.py --flow 172 --dry-run
```

---

### HackingToolSQLManager.install_all_tools → self.install_tool

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.install_all_tools到self.install_tool的處理能力
- **使用時機**: 當需要快速從HackingToolSQLManager.install_all_tools獲取self.install_tool的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.install_tool結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 170: 服務：Self.Install Tool

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Install Tool的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 170
```
```bash
aiva_internal_executor.py --flow 170 --dry-run
```

---

### HackingToolSQLManager.install_tool → self._check_tool_status

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.install_tool到self._check_tool_status的處理能力
- **使用時機**: 當需要對HackingToolSQLManager.install_tool進行深度分析並生成self._check_tool_status結果時
- **預期結果**: 獲得基於程式邏輯的self._check_tool_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 169: 服務：Self. Check Tool Status

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Check Tool Status相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 169
```
```bash
aiva_internal_executor.py --flow 169 --dry-run
```

---

### HackingToolSQLManager.uninstall_tool → shutil.rmtree

**AI描述欄位 📋**:
- **能力概要**: HackingToolSQLManager.uninstall_tool到shutil.rmtree的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolSQLManager.uninstall_tool到shutil.rmtree轉換時
- **預期結果**: 獲得基於程式邏輯的shutil.rmtree結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 171: 服務：Shutil.Rmtree

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Shutil.Rmtree的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 171
```
```bash
aiva_internal_executor.py --flow 171 --dry-run
```

---

### HackingToolXSSConfig.__init__ → self._calculate_priority_order

**AI描述欄位 📋**:
- **能力概要**: HackingToolXSSConfig.__init__到self._calculate_priority_order的處理能力
- **使用時機**: 當需要快速從HackingToolXSSConfig.__init__獲取self._calculate_priority_order的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._calculate_priority_order結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 453: 服務：Self. Calculate Priority Order

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Calculate Priority Order的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 453
```
```bash
aiva_internal_executor.py --flow 453 --dry-run
```

---

### HackingToolXSSConfig._calculate_priority_order → sorted

**AI描述欄位 📋**:
- **能力概要**: HackingToolXSSConfig._calculate_priority_order到sorted的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolXSSConfig._calculate_priority_order到sorted轉換時
- **預期結果**: 獲得基於程式邏輯的sorted結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 454: 服務：Sorted

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Sorted的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 454
```
```bash
aiva_internal_executor.py --flow 454 --dry-run
```

---

### HackingToolXSSConfig.export_config → print

**AI描述欄位 📋**:
- **能力概要**: HackingToolXSSConfig.export_config到print的處理能力
- **使用時機**: 當外部系統需要簡單的HackingToolXSSConfig.export_config到print轉換時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 456: 服務：Print

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Print的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 456
```
```bash
aiva_internal_executor.py --flow 456 --dry-run
```

---

### HackingToolXSSConfig.validate_tool_requirements → self.get_tool_config

**AI描述欄位 📋**:
- **能力概要**: HackingToolXSSConfig.validate_tool_requirements到self.get_tool_config的處理能力
- **使用時機**: 當需要快速從HackingToolXSSConfig.validate_tool_requirements獲取self.get_tool_config的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.get_tool_config結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 455: 服務：Self.Get Tool Config

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Get Tool Config的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 455
```
```bash
aiva_internal_executor.py --flow 455 --dry-run
```

---

### IDORDetectionContext → field

**AI描述欄位 📋**:
- **能力概要**: IDORDetectionContext到field的處理能力
- **使用時機**: 當需要為外部用戶提供IDORDetectionContext的field服務時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 78: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 78
```
```bash
aiva_internal_executor.py --flow 78 --dry-run
```

---

### IDORDetectionContext.add_error → self.errors.append

**AI描述欄位 📋**:
- **能力概要**: IDORDetectionContext.add_error到self.errors.append的處理能力
- **使用時機**: 當需要快速從IDORDetectionContext.add_error獲取self.errors.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.errors.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 80: 服務：Self.Errors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Errors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 80
```
```bash
aiva_internal_executor.py --flow 80 --dry-run
```

---

### IDORDetectionContext.add_finding → self.findings.append

**AI描述欄位 📋**:
- **能力概要**: IDORDetectionContext.add_finding到self.findings.append的處理能力
- **使用時機**: 當需要快速從IDORDetectionContext.add_finding獲取self.findings.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 79: 服務：Self.Findings.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 79
```
```bash
aiva_internal_executor.py --flow 79 --dry-run
```

---

### IDORDetector._perform_horizontal_tests → findings.append

**AI描述欄位 📋**:
- **能力概要**: IDORDetector._perform_horizontal_tests到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供IDORDetector._perform_horizontal_tests的findings.append服務時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 90: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 90
```
```bash
aiva_internal_executor.py --flow 90 --dry-run
```

---

### IDORDetector._perform_vertical_tests → findings.append

**AI描述欄位 📋**:
- **能力概要**: IDORDetector._perform_vertical_tests到findings.append的處理能力
- **使用時機**: 當外部系統需要簡單的IDORDetector._perform_vertical_tests到findings.append轉換時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 91: 服務：Findings.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 91
```
```bash
aiva_internal_executor.py --flow 91 --dry-run
```

---

### IDOREngine._calculate_sensitivity → max

**AI描述欄位 📋**:
- **能力概要**: IDOREngine._calculate_sensitivity到max的處理能力
- **使用時機**: 當外部系統需要簡單的IDOREngine._calculate_sensitivity到max轉換時
- **預期結果**: 獲得基於程式邏輯的max結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 99: 服務：Max

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Max的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 99
```
```bash
aiva_internal_executor.py --flow 99 --dry-run
```

---

### IDOREngine._has_shared_access → user_b_text.lower

**AI描述欄位 📋**:
- **能力概要**: IDOREngine._has_shared_access到user_b_text.lower的處理能力
- **使用時機**: 當需要為外部用戶提供IDOREngine._has_shared_access的user_b_text.lower服務時
- **預期結果**: 獲得基於程式邏輯的user_b_text.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 98: 服務：User B Text.Lower

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：User B Text.Lower相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 98
```
```bash
aiva_internal_executor.py --flow 98 --dry-run
```

---

### IDOREngine._is_public_resource → url.lower

**AI描述欄位 📋**:
- **能力概要**: IDOREngine._is_public_resource到url.lower的處理能力
- **使用時機**: 當外部系統需要簡單的IDOREngine._is_public_resource到url.lower轉換時
- **預期結果**: 獲得基於程式邏輯的url.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 97: 服務：Url.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Url.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 97
```
```bash
aiva_internal_executor.py --flow 97 --dry-run
```

---

### IDOREngine.close → self.client.aclose

**AI描述欄位 📋**:
- **能力概要**: IDOREngine.close到self.client.aclose的處理能力
- **使用時機**: 當需要快速從IDOREngine.close獲取self.client.aclose的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 93: 服務：Self.Client.Aclose

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Client.Aclose的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 93
```
```bash
aiva_internal_executor.py --flow 93 --dry-run
```

---

### IDOREngine.extract_ids_from_url → ids.append

**AI描述欄位 📋**:
- **能力概要**: IDOREngine.extract_ids_from_url到ids.append的處理能力
- **使用時機**: 當外部系統需要簡單的IDOREngine.extract_ids_from_url到ids.append轉換時
- **預期結果**: 獲得基於程式邏輯的ids.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 94: 服務：Ids.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Ids.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 94
```
```bash
aiva_internal_executor.py --flow 94 --dry-run
```

---

### IDOREngine.test_horizontal → self._calculate_sensitivity

**AI描述欄位 📋**:
- **能力概要**: IDOREngine.test_horizontal到self._calculate_sensitivity的處理能力
- **使用時機**: 當需要對IDOREngine.test_horizontal進行深度分析並生成self._calculate_sensitivity結果時
- **預期結果**: 獲得基於程式邏輯的self._calculate_sensitivity結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 95: 服務：Self. Calculate Sensitivity

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Calculate Sensitivity相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 95
```
```bash
aiva_internal_executor.py --flow 95 --dry-run
```

---

### IDOREngine.test_vertical → self._calculate_sensitivity

**AI描述欄位 📋**:
- **能力概要**: IDOREngine.test_vertical到self._calculate_sensitivity的處理能力
- **使用時機**: 當需要快速從IDOREngine.test_vertical獲取self._calculate_sensitivity的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._calculate_sensitivity結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 96: 服務：Self. Calculate Sensitivity

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Calculate Sensitivity的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 96
```
```bash
aiva_internal_executor.py --flow 96 --dry-run
```

---

### IdorCandidate → Field

**AI描述欄位 📋**:
- **能力概要**: IdorCandidate到Field的處理能力
- **使用時機**: 當需要為外部用戶提供IdorCandidate的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 30: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 30
```
```bash
aiva_internal_executor.py --flow 30 --dry-run
```

---

### IdorConfig → Field

**AI描述欄位 📋**:
- **能力概要**: IdorConfig到Field的處理能力
- **使用時機**: 當外部系統需要簡單的IdorConfig到Field轉換時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 88: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 88
```
```bash
aiva_internal_executor.py --flow 88 --dry-run
```

---

### LateralMovementEngine._check_rdp_access → vectors.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine._check_rdp_access到vectors.append的處理能力
- **使用時機**: 當需要對LateralMovementEngine._check_rdp_access進行深度分析並生成vectors.append結果時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 119: 服務：Vectors.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: advanced
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 119
```
```bash
aiva_internal_executor.py --flow 119 --dry-run
```

---

### LateralMovementEngine._check_smb_access → vectors.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine._check_smb_access到vectors.append的處理能力
- **使用時機**: 當需要對LateralMovementEngine._check_smb_access進行深度分析並生成vectors.append結果時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 117: 服務：Vectors.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: advanced
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 117
```
```bash
aiva_internal_executor.py --flow 117 --dry-run
```

---

### LateralMovementEngine._check_ssh_access → vectors.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine._check_ssh_access到vectors.append的處理能力
- **使用時機**: 當需要對LateralMovementEngine._check_ssh_access進行深度分析並生成vectors.append結果時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 118: 服務：Vectors.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: advanced
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 118
```
```bash
aiva_internal_executor.py --flow 118 --dry-run
```

---

### LateralMovementEngine._check_winrm_access → vectors.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine._check_winrm_access到vectors.append的處理能力
- **使用時機**: 當需要對LateralMovementEngine._check_winrm_access進行深度分析並生成vectors.append結果時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 120: 服務：Vectors.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: advanced
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 120
```
```bash
aiva_internal_executor.py --flow 120 --dry-run
```

---

### LateralMovementEngine._discover_hosts → alive_hosts.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine._discover_hosts到alive_hosts.append的處理能力
- **使用時機**: 當外部系統需要簡單的LateralMovementEngine._discover_hosts到alive_hosts.append轉換時
- **預期結果**: 獲得基於程式邏輯的alive_hosts.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 115: 服務：Alive Hosts.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Alive Hosts.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 115
```
```bash
aiva_internal_executor.py --flow 115 --dry-run
```

---

### LateralMovementEngine._is_host_alive → sock.close

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine._is_host_alive到sock.close的處理能力
- **使用時機**: 當需要為外部用戶提供LateralMovementEngine._is_host_alive的sock.close服務時
- **預期結果**: 獲得部分AI輔助的sock.close結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 116: 服務：Sock.Close

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Sock.Close相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 116
```
```bash
aiva_internal_executor.py --flow 116 --dry-run
```

---

### LateralMovementEngine.scan_network → vectors.extend

**AI描述欄位 📋**:
- **能力概要**: LateralMovementEngine.scan_network到vectors.extend的處理能力
- **使用時機**: 當需要對LateralMovementEngine.scan_network進行深度分析並生成vectors.extend結果時
- **預期結果**: 獲得部分AI輔助的vectors.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 114: 服務：Vectors.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: advanced
- **用途**: 執行服務：Vectors.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 114
```
```bash
aiva_internal_executor.py --flow 114 --dry-run
```

---

### LateralMovementTester.enumerate_services → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementTester.enumerate_services到self.test_results.append的處理能力
- **使用時機**: 當需要快速從LateralMovementTester.enumerate_services獲取self.test_results.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 109: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 109
```
```bash
aiva_internal_executor.py --flow 109 --dry-run
```

---

### LateralMovementTester.scan_network → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementTester.scan_network到self.test_results.append的處理能力
- **使用時機**: 當需要對LateralMovementTester.scan_network進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得部分AI輔助的self.test_results.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 108: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 108
```
```bash
aiva_internal_executor.py --flow 108 --dry-run
```

---

### LateralMovementTester.simulate_pass_the_hash → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementTester.simulate_pass_the_hash到self.test_results.append的處理能力
- **使用時機**: 當需要快速從LateralMovementTester.simulate_pass_the_hash獲取self.test_results.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 111: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 111
```
```bash
aiva_internal_executor.py --flow 111 --dry-run
```

---

### LateralMovementTester.test_credential_reuse → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementTester.test_credential_reuse到self.test_results.append的處理能力
- **使用時機**: 當需要快速從LateralMovementTester.test_credential_reuse獲取self.test_results.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 110: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 110
```
```bash
aiva_internal_executor.py --flow 110 --dry-run
```

---

### LateralMovementTester.test_remote_access → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: LateralMovementTester.test_remote_access到self.test_results.append的處理能力
- **使用時機**: 當需要快速從LateralMovementTester.test_remote_access獲取self.test_results.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 112: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 112
```
```bash
aiva_internal_executor.py --flow 112 --dry-run
```

---

### Menu → print

**AI描述欄位 📋**:
- **能力概要**: Menu到print的處理能力
- **使用時機**: 當需要為外部用戶提供Menu的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 565: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 565
```
```bash
aiva_internal_executor.py --flow 565 --dry-run
```

---

### ModuleStatus → Field

**AI描述欄位 📋**:
- **能力概要**: ModuleStatus到Field的處理能力
- **使用時機**: 當需要為外部用戶提供ModuleStatus的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 23: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 23
```
```bash
aiva_internal_executor.py --flow 23 --dry-run
```

---

### NetworkError.__str__ → parts.append

**AI描述欄位 📋**:
- **能力概要**: NetworkError.__str__到parts.append的處理能力
- **使用時機**: 當需要為外部用戶提供NetworkError.__str__的parts.append服務時
- **預期結果**: 獲得部分AI輔助的parts.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 158: 服務：Parts.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Parts.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 158
```
```bash
aiva_internal_executor.py --flow 158 --dry-run
```

---

### NoSQLInjectionScanner.__init__ → self._load_nosql_payloads

**AI描述欄位 📋**:
- **能力概要**: NoSQLInjectionScanner.__init__到self._load_nosql_payloads的處理能力
- **使用時機**: 當需要快速從NoSQLInjectionScanner.__init__獲取self._load_nosql_payloads的基礎信息時
- **預期結果**: 獲得部分AI輔助的self._load_nosql_payloads結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 278: 服務：Self. Load Nosql Payloads

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Self. Load Nosql Payloads的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 278
```
```bash
aiva_internal_executor.py --flow 278 --dry-run
```

---

### NoSQLInjectionScanner._test_nosql_payload → response.text

**AI描述欄位 📋**:
- **能力概要**: NoSQLInjectionScanner._test_nosql_payload到response.text的處理能力
- **使用時機**: 當需要對NoSQLInjectionScanner._test_nosql_payload進行深度分析並生成response.text結果時
- **預期結果**: 獲得部分AI輔助的response.text結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 280: 服務：Response.Text

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Response.Text相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 280
```
```bash
aiva_internal_executor.py --flow 280 --dry-run
```

---

### NoSQLInjectionScanner.scan_target → results.append

**AI描述欄位 📋**:
- **能力概要**: NoSQLInjectionScanner.scan_target到results.append的處理能力
- **使用時機**: 當需要對NoSQLInjectionScanner.scan_target進行深度分析並生成results.append結果時
- **預期結果**: 獲得部分AI輔助的results.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 279: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 279
```
```bash
aiva_internal_executor.py --flow 279 --dry-run
```

---

### OOBDetectionEngine.detect → results.append

**AI描述欄位 📋**:
- **能力概要**: OOBDetectionEngine.detect到results.append的處理能力
- **使用時機**: 當需要為外部用戶提供OOBDetectionEngine.detect的results.append服務時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 235: 服務：Results.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 235
```
```bash
aiva_internal_executor.py --flow 235 --dry-run
```

---

### OastDispatcher._resolve_token → normalized.split

**AI描述欄位 📋**:
- **能力概要**: OastDispatcher._resolve_token到normalized.split的處理能力
- **使用時機**: 當外部系統需要簡單的OastDispatcher._resolve_token到normalized.split轉換時
- **預期結果**: 獲得基於程式邏輯的normalized.split結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 318: 服務：Normalized.Split

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Normalized.Split的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 318
```
```bash
aiva_internal_executor.py --flow 318 --dry-run
```

---

### OastDispatcher.close → self._client.aclose

**AI描述欄位 📋**:
- **能力概要**: OastDispatcher.close到self._client.aclose的處理能力
- **使用時機**: 當需要快速從OastDispatcher.close獲取self._client.aclose的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 317: 服務：Self. Client.Aclose

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Client.Aclose的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 317
```
```bash
aiva_internal_executor.py --flow 317 --dry-run
```

---

### OastDispatcher.fetch_events → events.append

**AI描述欄位 📋**:
- **能力概要**: OastDispatcher.fetch_events到events.append的處理能力
- **使用時機**: 當需要對OastDispatcher.fetch_events進行深度分析並生成events.append結果時
- **預期結果**: 獲得基於程式邏輯的events.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 316: 服務：Events.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Events.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 316
```
```bash
aiva_internal_executor.py --flow 316 --dry-run
```

---

### OastDispatcher.register → OastProbe

**AI描述欄位 📋**:
- **能力概要**: OastDispatcher.register到OastProbe的處理能力
- **使用時機**: 當需要為外部用戶提供OastDispatcher.register的OastProbe服務時
- **預期結果**: 獲得基於程式邏輯的OastProbe結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 315: 服務：Oastprobe

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Oastprobe相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 315
```
```bash
aiva_internal_executor.py --flow 315 --dry-run
```

---

### OastHttpCallbackStore.fetch_events → events.append

**AI描述欄位 📋**:
- **能力概要**: OastHttpCallbackStore.fetch_events到events.append的處理能力
- **使用時機**: 當需要對OastHttpCallbackStore.fetch_events進行深度分析並生成events.append結果時
- **預期結果**: 獲得基於程式邏輯的events.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 448: 服務：Events.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Events.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 448
```
```bash
aiva_internal_executor.py --flow 448 --dry-run
```

---

### OastHttpCallbackStore.register_probe → isinstance

**AI描述欄位 📋**:
- **能力概要**: OastHttpCallbackStore.register_probe到isinstance的處理能力
- **使用時機**: 當需要為外部用戶提供OastHttpCallbackStore.register_probe的isinstance服務時
- **預期結果**: 獲得基於程式邏輯的isinstance結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 447: 服務：Isinstance

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Isinstance相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 447
```
```bash
aiva_internal_executor.py --flow 447 --dry-run
```

---

### PayloadWrapperEncoder._inject_query → urlencode

**AI描述欄位 📋**:
- **能力概要**: PayloadWrapperEncoder._inject_query到urlencode的處理能力
- **使用時機**: 當外部系統需要簡單的PayloadWrapperEncoder._inject_query到urlencode轉換時
- **預期結果**: 獲得基於程式邏輯的urlencode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 185: 服務：Urlencode

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Urlencode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 185
```
```bash
aiva_internal_executor.py --flow 185 --dry-run
```

---

### PayloadWrapperEncoder.encode → request_kwargs.items

**AI描述欄位 📋**:
- **能力概要**: PayloadWrapperEncoder.encode到request_kwargs.items的處理能力
- **使用時機**: 當需要為外部用戶提供PayloadWrapperEncoder.encode的request_kwargs.items服務時
- **預期結果**: 獲得基於程式邏輯的request_kwargs.items結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 184: 服務：Request Kwargs.Items

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Request Kwargs.Items相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 184
```
```bash
aiva_internal_executor.py --flow 184 --dry-run
```

---

### PersistenceChecker.__init__ → platform.system

**AI描述欄位 📋**:
- **能力概要**: PersistenceChecker.__init__到platform.system的處理能力
- **使用時機**: 當外部系統需要簡單的PersistenceChecker.__init__到platform.system轉換時
- **預期結果**: 獲得基於程式邏輯的platform.system結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 128: 服務：Platform.System

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Platform.System的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 128
```
```bash
aiva_internal_executor.py --flow 128 --dry-run
```

---

### PersistenceChecker.check_cron_jobs → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceChecker.check_cron_jobs到self.test_results.append的處理能力
- **使用時機**: 當需要對PersistenceChecker.check_cron_jobs進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 133: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 133
```
```bash
aiva_internal_executor.py --flow 133 --dry-run
```

---

### PersistenceChecker.check_registry_persistence → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceChecker.check_registry_persistence到self.test_results.append的處理能力
- **使用時機**: 當需要對PersistenceChecker.check_registry_persistence進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 132: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 132
```
```bash
aiva_internal_executor.py --flow 132 --dry-run
```

---

### PersistenceChecker.check_services → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceChecker.check_services到self.test_results.append的處理能力
- **使用時機**: 當需要對PersistenceChecker.check_services進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 131: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 131
```
```bash
aiva_internal_executor.py --flow 131 --dry-run
```

---

### PersistenceChecker.check_startup_items → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceChecker.check_startup_items到self.test_results.append的處理能力
- **使用時機**: 當需要對PersistenceChecker.check_startup_items進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 129: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 129
```
```bash
aiva_internal_executor.py --flow 129 --dry-run
```

---

### PersistenceEngine._check_cron_persistence → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine._check_cron_persistence到vectors.append的處理能力
- **使用時機**: 當外部系統需要簡單的PersistenceEngine._check_cron_persistence到vectors.append轉換時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 123: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 123
```
```bash
aiva_internal_executor.py --flow 123 --dry-run
```

---

### PersistenceEngine._check_ld_preload_persistence → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine._check_ld_preload_persistence到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供PersistenceEngine._check_ld_preload_persistence的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 127: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 127
```
```bash
aiva_internal_executor.py --flow 127 --dry-run
```

---

### PersistenceEngine._check_linux_persistence → vectors.extend

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine._check_linux_persistence到vectors.extend的處理能力
- **使用時機**: 當需要為外部用戶提供PersistenceEngine._check_linux_persistence的vectors.extend服務時
- **預期結果**: 獲得部分AI輔助的vectors.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 122: 服務：Vectors.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 122
```
```bash
aiva_internal_executor.py --flow 122 --dry-run
```

---

### PersistenceEngine._check_shell_rc_persistence → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine._check_shell_rc_persistence到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供PersistenceEngine._check_shell_rc_persistence的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 125: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 125
```
```bash
aiva_internal_executor.py --flow 125 --dry-run
```

---

### PersistenceEngine._check_ssh_persistence → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine._check_ssh_persistence到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供PersistenceEngine._check_ssh_persistence的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 126: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 126
```
```bash
aiva_internal_executor.py --flow 126 --dry-run
```

---

### PersistenceEngine._check_systemd_persistence → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine._check_systemd_persistence到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供PersistenceEngine._check_systemd_persistence的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 124: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 124
```
```bash
aiva_internal_executor.py --flow 124 --dry-run
```

---

### PersistenceEngine.scan → vectors.extend

**AI描述欄位 📋**:
- **能力概要**: PersistenceEngine.scan到vectors.extend的處理能力
- **使用時機**: 當需要為外部用戶提供PersistenceEngine.scan的vectors.extend服務時
- **預期結果**: 獲得部分AI輔助的vectors.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 121: 服務：Vectors.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 121
```
```bash
aiva_internal_executor.py --flow 121 --dry-run
```

---

### PortScanner._grab_banner → sock.settimeout

**AI描述欄位 📋**:
- **能力概要**: PortScanner._grab_banner到sock.settimeout的處理能力
- **使用時機**: 當外部系統需要簡單的PortScanner._grab_banner到sock.settimeout轉換時
- **預期結果**: 獲得部分AI輔助的sock.settimeout結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 426: 服務：Sock.Settimeout

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Sock.Settimeout的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 426
```
```bash
aiva_internal_executor.py --flow 426 --dry-run
```

---

### PortScanner._scan_port → sock.close

**AI描述欄位 📋**:
- **能力概要**: PortScanner._scan_port到sock.close的處理能力
- **使用時機**: 當需要對PortScanner._scan_port進行深度分析並生成sock.close結果時
- **預期結果**: 獲得部分AI輔助的sock.close結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 425: 服務：Sock.Close

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Sock.Close相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 425
```
```bash
aiva_internal_executor.py --flow 425 --dry-run
```

---

### PortScanner.scan → results.append

**AI描述欄位 📋**:
- **能力概要**: PortScanner.scan到results.append的處理能力
- **使用時機**: 當需要為外部用戶提供PortScanner.scan的results.append服務時
- **預期結果**: 獲得部分AI輔助的results.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 424: 服務：Results.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 424
```
```bash
aiva_internal_executor.py --flow 424 --dry-run
```

---

### PostExDetector.__init__ → PersistenceEngine

**AI描述欄位 📋**:
- **能力概要**: PostExDetector.__init__到PersistenceEngine的處理能力
- **使用時機**: 當需要為外部用戶提供PostExDetector.__init__的PersistenceEngine服務時
- **預期結果**: 獲得基於程式邏輯的PersistenceEngine結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 106: 服務：Persistenceengine

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Persistenceengine相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 106
```
```bash
aiva_internal_executor.py --flow 106 --dry-run
```

---

### PostExDetector.scan_full → PostExResult

**AI描述欄位 📋**:
- **能力概要**: PostExDetector.scan_full到PostExResult的處理能力
- **使用時機**: 當需要對PostExDetector.scan_full進行深度分析並生成PostExResult結果時
- **預期結果**: 獲得部分AI輔助的PostExResult結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 107: 服務：Postexresult

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Postexresult相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 107
```
```bash
aiva_internal_executor.py --flow 107 --dry-run
```

---

### PostExManager.__init__ → PostExDetector

**AI描述欄位 📋**:
- **能力概要**: PostExManager.__init__到PostExDetector的處理能力
- **使用時機**: 當外部系統需要簡單的PostExManager.__init__到PostExDetector轉換時
- **預期結果**: 獲得基於程式邏輯的PostExDetector結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 100: 服務：Postexdetector

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Postexdetector的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 100
```
```bash
aiva_internal_executor.py --flow 100 --dry-run
```

---

### PostExManager._generate_summary → severity_counts.get

**AI描述欄位 📋**:
- **能力概要**: PostExManager._generate_summary到severity_counts.get的處理能力
- **使用時機**: 當外部系統需要簡單的PostExManager._generate_summary到severity_counts.get轉換時
- **預期結果**: 獲得基於程式邏輯的severity_counts.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 102: 服務：Severity Counts.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Severity Counts.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 102
```
```bash
aiva_internal_executor.py --flow 102 --dry-run
```

---

### PriceManipulationScanner._detect_business_limits → response_data.get

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner._detect_business_limits到response_data.get的處理能力
- **使用時機**: 當需要為外部用戶提供PriceManipulationScanner._detect_business_limits的response_data.get服務時
- **預期結果**: 獲得部分AI輔助的response_data.get結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 39: 服務：Response Data.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Response Data.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 39
```
```bash
aiva_internal_executor.py --flow 39 --dry-run
```

---

### PriceManipulationScanner._verify_actual_price_change → response_data.get

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner._verify_actual_price_change到response_data.get的處理能力
- **使用時機**: 當需要為外部用戶提供PriceManipulationScanner._verify_actual_price_change的response_data.get服務時
- **預期結果**: 獲得部分AI輔助的response_data.get結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 36: 服務：Response Data.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Response Data.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 36
```
```bash
aiva_internal_executor.py --flow 36 --dry-run
```

---

### PriceManipulationScanner._verify_transaction_completed → check_data.get

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner._verify_transaction_completed到check_data.get的處理能力
- **使用時機**: 當需要為外部用戶提供PriceManipulationScanner._verify_transaction_completed的check_data.get服務時
- **預期結果**: 獲得部分AI輔助的check_data.get結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 37: 服務：Check Data.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Check Data.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 37
```
```bash
aiva_internal_executor.py --flow 37 --dry-run
```

---

### PriceManipulationScanner._verify_user_privilege → permission_matrix.get

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner._verify_user_privilege到permission_matrix.get的處理能力
- **使用時機**: 當外部系統需要簡單的PriceManipulationScanner._verify_user_privilege到permission_matrix.get轉換時
- **預期結果**: 獲得部分AI輔助的permission_matrix.get結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 38: 服務：Permission Matrix.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Permission Matrix.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 38
```
```bash
aiva_internal_executor.py --flow 38 --dry-run
```

---

### PriceManipulationScanner.run_all_tests → all_findings.extend

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner.run_all_tests到all_findings.extend的處理能力
- **使用時機**: 當外部系統需要簡單的PriceManipulationScanner.run_all_tests到all_findings.extend轉換時
- **預期結果**: 獲得部分AI輔助的all_findings.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 44: 服務：All Findings.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：All Findings.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 44
```
```bash
aiva_internal_executor.py --flow 44 --dry-run
```

---

### PriceManipulationScanner.test_negative_price → findings.append

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner.test_negative_price到findings.append的處理能力
- **使用時機**: 當需要對PriceManipulationScanner.test_negative_price進行深度分析並生成findings.append結果時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 40: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 40
```
```bash
aiva_internal_executor.py --flow 40 --dry-run
```

---

### PriceManipulationScanner.test_overflow_price → findings.append

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner.test_overflow_price到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供PriceManipulationScanner.test_overflow_price的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 43: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 43
```
```bash
aiva_internal_executor.py --flow 43 --dry-run
```

---

### PriceManipulationScanner.test_price_tampering → findings.append

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner.test_price_tampering到findings.append的處理能力
- **使用時機**: 當需要對PriceManipulationScanner.test_price_tampering進行深度分析並生成findings.append結果時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 42: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 42
```
```bash
aiva_internal_executor.py --flow 42 --dry-run
```

---

### PriceManipulationScanner.test_zero_price → findings.append

**AI描述欄位 📋**:
- **能力概要**: PriceManipulationScanner.test_zero_price到findings.append的處理能力
- **使用時機**: 當需要對PriceManipulationScanner.test_zero_price進行深度分析並生成findings.append結果時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 41: 服務：Findings.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 41
```
```bash
aiva_internal_executor.py --flow 41 --dry-run
```

---

### PrivilegeEscalationEngine._check_cron_jobs → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_cron_jobs到vectors.append的處理能力
- **使用時機**: 當外部系統需要簡單的PrivilegeEscalationEngine._check_cron_jobs到vectors.append轉換時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 146: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 146
```
```bash
aiva_internal_executor.py --flow 146 --dry-run
```

---

### PrivilegeEscalationEngine._check_docker_socket → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_docker_socket到vectors.append的處理能力
- **使用時機**: 當外部系統需要簡單的PrivilegeEscalationEngine._check_docker_socket到vectors.append轉換時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 147: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 147
```
```bash
aiva_internal_executor.py --flow 147 --dry-run
```

---

### PrivilegeEscalationEngine._check_kernel_version → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_kernel_version到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供PrivilegeEscalationEngine._check_kernel_version的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 148: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 148
```
```bash
aiva_internal_executor.py --flow 148 --dry-run
```

---

### PrivilegeEscalationEngine._check_linux_privesc → vectors.extend

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_linux_privesc到vectors.extend的處理能力
- **使用時機**: 當需要為外部用戶提供PrivilegeEscalationEngine._check_linux_privesc的vectors.extend服務時
- **預期結果**: 獲得部分AI輔助的vectors.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 142: 服務：Vectors.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 142
```
```bash
aiva_internal_executor.py --flow 142 --dry-run
```

---

### PrivilegeEscalationEngine._check_sudo_config → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_sudo_config到vectors.append的處理能力
- **使用時機**: 當外部系統需要簡單的PrivilegeEscalationEngine._check_sudo_config到vectors.append轉換時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 144: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 144
```
```bash
aiva_internal_executor.py --flow 144 --dry-run
```

---

### PrivilegeEscalationEngine._check_suid_binaries → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_suid_binaries到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供PrivilegeEscalationEngine._check_suid_binaries的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 143: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 143
```
```bash
aiva_internal_executor.py --flow 143 --dry-run
```

---

### PrivilegeEscalationEngine._check_writable_paths → vectors.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine._check_writable_paths到vectors.append的處理能力
- **使用時機**: 當外部系統需要簡單的PrivilegeEscalationEngine._check_writable_paths到vectors.append轉換時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 145: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Vectors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 145
```
```bash
aiva_internal_executor.py --flow 145 --dry-run
```

---

### PrivilegeEscalationEngine.scan → vectors.extend

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalationEngine.scan到vectors.extend的處理能力
- **使用時機**: 當需要為外部用戶提供PrivilegeEscalationEngine.scan的vectors.extend服務時
- **預期結果**: 獲得部分AI輔助的vectors.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 141: 服務：Vectors.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 141
```
```bash
aiva_internal_executor.py --flow 141 --dry-run
```

---

### PrivilegeEscalator.check_kernel_exploits → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalator.check_kernel_exploits到self.test_results.append的處理能力
- **使用時機**: 當需要對PrivilegeEscalator.check_kernel_exploits進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 137: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 137
```
```bash
aiva_internal_executor.py --flow 137 --dry-run
```

---

### PrivilegeEscalator.check_sudo_misconfiguration → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalator.check_sudo_misconfiguration到self.test_results.append的處理能力
- **使用時機**: 當需要快速從PrivilegeEscalator.check_sudo_misconfiguration獲取self.test_results.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 136: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 136
```
```bash
aiva_internal_executor.py --flow 136 --dry-run
```

---

### PrivilegeEscalator.check_suid_binaries → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalator.check_suid_binaries到self.test_results.append的處理能力
- **使用時機**: 當需要對PrivilegeEscalator.check_suid_binaries進行深度分析並生成self.test_results.append結果時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 135: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Test Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 135
```
```bash
aiva_internal_executor.py --flow 135 --dry-run
```

---

### PrivilegeEscalator.check_writable_services → self.test_results.append

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalator.check_writable_services到self.test_results.append的處理能力
- **使用時機**: 當需要快速從PrivilegeEscalator.check_writable_services獲取self.test_results.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 138: 服務：Self.Test Results.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 138
```
```bash
aiva_internal_executor.py --flow 138 --dry-run
```

---

### PrivilegeEscalator.clear_results → self.test_results.clear

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalator.clear_results到self.test_results.clear的處理能力
- **使用時機**: 當需要快速從PrivilegeEscalator.clear_results獲取self.test_results.clear的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.test_results.clear結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 140: 服務：Self.Test Results.Clear

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Test Results.Clear的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 140
```
```bash
aiva_internal_executor.py --flow 140 --dry-run
```

---

### PrivilegeEscalator.run_full_assessment → platform.system

**AI描述欄位 📋**:
- **能力概要**: PrivilegeEscalator.run_full_assessment到platform.system的處理能力
- **使用時機**: 當外部系統需要簡單的PrivilegeEscalator.run_full_assessment到platform.system轉換時
- **預期結果**: 獲得基於程式邏輯的platform.system結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 139: 服務：Platform.System

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Platform.System的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 139
```
```bash
aiva_internal_executor.py --flow 139 --dry-run
```

---

### RaceConditionScanner.run_all_tests → all_findings.extend

**AI描述欄位 📋**:
- **能力概要**: RaceConditionScanner.run_all_tests到all_findings.extend的處理能力
- **使用時機**: 當外部系統需要簡單的RaceConditionScanner.run_all_tests到all_findings.extend轉換時
- **預期結果**: 獲得部分AI輔助的all_findings.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 49: 服務：All Findings.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：All Findings.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 49
```
```bash
aiva_internal_executor.py --flow 49 --dry-run
```

---

### RaceConditionScanner.test_balance_manipulation → findings.append

**AI描述欄位 📋**:
- **能力概要**: RaceConditionScanner.test_balance_manipulation到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供RaceConditionScanner.test_balance_manipulation的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 46: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 46
```
```bash
aiva_internal_executor.py --flow 46 --dry-run
```

---

### RaceConditionScanner.test_coupon_reuse → findings.append

**AI描述欄位 📋**:
- **能力概要**: RaceConditionScanner.test_coupon_reuse到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供RaceConditionScanner.test_coupon_reuse的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 47: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 47
```
```bash
aiva_internal_executor.py --flow 47 --dry-run
```

---

### RaceConditionScanner.test_inventory_depletion → findings.append

**AI描述欄位 📋**:
- **能力概要**: RaceConditionScanner.test_inventory_depletion到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供RaceConditionScanner.test_inventory_depletion的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 48: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 48
```
```bash
aiva_internal_executor.py --flow 48 --dry-run
```

---

### ResourceIdExtractor.extract_from_url → ids.append

**AI描述欄位 📋**:
- **能力概要**: ResourceIdExtractor.extract_from_url到ids.append的處理能力
- **使用時機**: 當需要為外部用戶提供ResourceIdExtractor.extract_from_url的ids.append服務時
- **預期結果**: 獲得基於程式邏輯的ids.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 75: 服務：Ids.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Ids.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 75
```
```bash
aiva_internal_executor.py --flow 75 --dry-run
```

---

### ResourceIdExtractor.generate_test_ids → test_ids.extend

**AI描述欄位 📋**:
- **能力概要**: ResourceIdExtractor.generate_test_ids到test_ids.extend的處理能力
- **使用時機**: 當需要為外部用戶提供ResourceIdExtractor.generate_test_ids的test_ids.extend服務時
- **預期結果**: 獲得基於程式邏輯的test_ids.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 76: 服務：Test Ids.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Test Ids.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 76
```
```bash
aiva_internal_executor.py --flow 76 --dry-run
```

---

### ResourceIdExtractor.replace_id_in_url → urlunparse

**AI描述欄位 📋**:
- **能力概要**: ResourceIdExtractor.replace_id_in_url到urlunparse的處理能力
- **使用時機**: 當需要為外部用戶提供ResourceIdExtractor.replace_id_in_url的urlunparse服務時
- **預期結果**: 獲得基於程式邏輯的urlunparse結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 77: 服務：Urlunparse

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Urlunparse相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 77
```
```bash
aiva_internal_executor.py --flow 77 --dry-run
```

---

### RiskAssessment → Field

**AI描述欄位 📋**:
- **能力概要**: RiskAssessment到Field的處理能力
- **使用時機**: 當需要為外部用戶提供RiskAssessment的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 15: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 15
```
```bash
aiva_internal_executor.py --flow 15 --dry-run
```

---

### RiskFactor → Field

**AI描述欄位 📋**:
- **能力概要**: RiskFactor到Field的處理能力
- **使用時機**: 當需要為外部用戶提供RiskFactor的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 14: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 14
```
```bash
aiva_internal_executor.py --flow 14 --dry-run
```

---

### SQLInjectionCLI._blind_injection_scan → table.add_row

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._blind_injection_scan到table.add_row的處理能力
- **使用時機**: 當需要對SQLInjectionCLI._blind_injection_scan進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得部分AI輔助的table.add_row結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 293: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 293
```
```bash
aiva_internal_executor.py --flow 293 --dry-run
```

---

### SQLInjectionCLI._comprehensive_scan → self._display_scan_results

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._comprehensive_scan到self._display_scan_results的處理能力
- **使用時機**: 當需要快速從SQLInjectionCLI._comprehensive_scan獲取self._display_scan_results的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._display_scan_results結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 289: 服務：Self. Display Scan Results

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Display Scan Results的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 289
```
```bash
aiva_internal_executor.py --flow 289 --dry-run
```

---

### SQLInjectionCLI._custom_payload_test → content.lower

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._custom_payload_test到content.lower的處理能力
- **使用時機**: 當需要為外部用戶提供SQLInjectionCLI._custom_payload_test的content.lower服務時
- **預期結果**: 獲得基於程式邏輯的content.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 291: 服務：Content.Lower

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Content.Lower相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 291
```
```bash
aiva_internal_executor.py --flow 291 --dry-run
```

---

### SQLInjectionCLI._display_scan_results → method_table.add_row

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._display_scan_results到method_table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供SQLInjectionCLI._display_scan_results的method_table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的method_table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 296: 服務：Method Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Method Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 296
```
```bash
aiva_internal_executor.py --flow 296 --dry-run
```

---

### SQLInjectionCLI._export_report → open

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._export_report到open的處理能力
- **使用時機**: 當需要對SQLInjectionCLI._export_report進行深度分析並生成open結果時
- **預期結果**: 獲得基於程式邏輯的open結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 295: 服務：Open

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Open相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 295
```
```bash
aiva_internal_executor.py --flow 295 --dry-run
```

---

### SQLInjectionCLI._nosql_scan → table.add_row

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._nosql_scan到table.add_row的處理能力
- **使用時機**: 當需要對SQLInjectionCLI._nosql_scan進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得部分AI輔助的table.add_row結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 292: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 292
```
```bash
aiva_internal_executor.py --flow 292 --dry-run
```

---

### SQLInjectionCLI._show_scan_history → table.add_row

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._show_scan_history到table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供SQLInjectionCLI._show_scan_history的table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 294: 服務：Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 294
```
```bash
aiva_internal_executor.py --flow 294 --dry-run
```

---

### SQLInjectionCLI._sqlmap_scan → table.add_row

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI._sqlmap_scan到table.add_row的處理能力
- **使用時機**: 當需要對SQLInjectionCLI._sqlmap_scan進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 290: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 290
```
```bash
aiva_internal_executor.py --flow 290 --dry-run
```

---

### SQLInjectionCLI.run_interactive → self._export_report

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI.run_interactive到self._export_report的處理能力
- **使用時機**: 當需要對SQLInjectionCLI.run_interactive進行深度分析並生成self._export_report結果時
- **預期結果**: 獲得基於程式邏輯的self._export_report結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 288: 服務：Self. Export Report

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Export Report相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 288
```
```bash
aiva_internal_executor.py --flow 288 --dry-run
```

---

### SQLInjectionCLI.show_main_menu → table.add_row

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionCLI.show_main_menu到table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供SQLInjectionCLI.show_main_menu的table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 287: 服務：Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 287
```
```bash
aiva_internal_executor.py --flow 287 --dry-run
```

---

### SQLInjectionManager.__init__ → BlindSQLInjectionScanner

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionManager.__init__到BlindSQLInjectionScanner的處理能力
- **使用時機**: 當需要為外部用戶提供SQLInjectionManager.__init__的BlindSQLInjectionScanner服務時
- **預期結果**: 獲得部分AI輔助的BlindSQLInjectionScanner結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 284: 服務：Blindsqlinjectionscanner

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Blindsqlinjectionscanner相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 284
```
```bash
aiva_internal_executor.py --flow 284 --dry-run
```

---

### SQLInjectionManager._parse_target → urllib.parse.urlparse

**AI描述欄位 📋**:
- **能力概要**: SQLInjectionManager._parse_target到urllib.parse.urlparse的處理能力
- **使用時機**: 當外部系統需要簡單的SQLInjectionManager._parse_target到urllib.parse.urlparse轉換時
- **預期結果**: 獲得基於程式邏輯的urllib.parse.urlparse結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 286: 服務：Urllib.Parse.Urlparse

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Urllib.Parse.Urlparse的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 286
```
```bash
aiva_internal_executor.py --flow 286 --dry-run
```

---

### SQLTarget → field

**AI描述欄位 📋**:
- **能力概要**: SQLTarget到field的處理能力
- **使用時機**: 當需要為外部用戶提供SQLTarget的field服務時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 266: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 266
```
```bash
aiva_internal_executor.py --flow 266 --dry-run
```

---

### SSRFDetectionContext → field

**AI描述欄位 📋**:
- **能力概要**: SSRFDetectionContext到field的處理能力
- **使用時機**: 當需要為外部用戶提供SSRFDetectionContext的field服務時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 337: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 337
```
```bash
aiva_internal_executor.py --flow 337 --dry-run
```

---

### SSRFDetectionContext.add_error → self.errors.append

**AI描述欄位 📋**:
- **能力概要**: SSRFDetectionContext.add_error到self.errors.append的處理能力
- **使用時機**: 當需要快速從SSRFDetectionContext.add_error獲取self.errors.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.errors.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 339: 服務：Self.Errors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Errors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 339
```
```bash
aiva_internal_executor.py --flow 339 --dry-run
```

---

### SSRFDetectionContext.add_finding → self.findings.append

**AI描述欄位 📋**:
- **能力概要**: SSRFDetectionContext.add_finding到self.findings.append的處理能力
- **使用時機**: 當需要快速從SSRFDetectionContext.add_finding獲取self.findings.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 338: 服務：Self.Findings.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Findings.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 338
```
```bash
aiva_internal_executor.py --flow 338 --dry-run
```

---

### SSRFEngine._resolve_ips → ips.append

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine._resolve_ips到ips.append的處理能力
- **使用時機**: 當外部系統需要簡單的SSRFEngine._resolve_ips到ips.append轉換時
- **預期結果**: 獲得基於程式邏輯的ips.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 367: 服務：Ips.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Ips.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 367
```
```bash
aiva_internal_executor.py --flow 367 --dry-run
```

---

### SSRFEngine.check_cloud_metadata → issues.append

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine.check_cloud_metadata到issues.append的處理能力
- **使用時機**: 當需要對SSRFEngine.check_cloud_metadata進行深度分析並生成issues.append結果時
- **預期結果**: 獲得基於程式邏輯的issues.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 370: 服務：Issues.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Issues.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 370
```
```bash
aiva_internal_executor.py --flow 370 --dry-run
```

---

### SSRFEngine.check_file_protocol → issues.append

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine.check_file_protocol到issues.append的處理能力
- **使用時機**: 當外部系統需要簡單的SSRFEngine.check_file_protocol到issues.append轉換時
- **預期結果**: 獲得基於程式邏輯的issues.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 371: 服務：Issues.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Issues.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 371
```
```bash
aiva_internal_executor.py --flow 371 --dry-run
```

---

### SSRFEngine.close → self.client.aclose

**AI描述欄位 📋**:
- **能力概要**: SSRFEngine.close到self.client.aclose的處理能力
- **使用時機**: 當需要快速從SSRFEngine.close獲取self.client.aclose的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.client.aclose結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 366: 服務：Self.Client.Aclose

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Client.Aclose的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 366
```
```bash
aiva_internal_executor.py --flow 366 --dry-run
```

---

### SmartIDORDetector.__init__ → IdorConfig

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector.__init__到IdorConfig的處理能力
- **使用時機**: 當外部系統需要簡單的SmartIDORDetector.__init__到IdorConfig轉換時
- **預期結果**: 獲得基於程式邏輯的IdorConfig結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 81: 服務：Idorconfig

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Idorconfig的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 81
```
```bash
aiva_internal_executor.py --flow 81 --dry-run
```

---

### SmartIDORDetector._extract_resource_ids → context.add_error

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector._extract_resource_ids到context.add_error的處理能力
- **使用時機**: 當需要為外部用戶提供SmartIDORDetector._extract_resource_ids的context.add_error服務時
- **預期結果**: 獲得基於程式邏輯的context.add_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 83: 服務：Context.Add Error

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Context.Add Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 83
```
```bash
aiva_internal_executor.py --flow 83 --dry-run
```

---

### SmartIDORDetector.detect_vulnerabilities → context.add_error

**AI描述欄位 📋**:
- **能力概要**: SmartIDORDetector.detect_vulnerabilities到context.add_error的處理能力
- **使用時機**: 當需要對SmartIDORDetector.detect_vulnerabilities進行深度分析並生成context.add_error結果時
- **預期結果**: 獲得基於程式邏輯的context.add_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 82: 服務：Context.Add Error

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Context.Add Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 82
```
```bash
aiva_internal_executor.py --flow 82 --dry-run
```

---

### SmartSSRFDetector.__init__ → SsrfConfig

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector.__init__到SsrfConfig的處理能力
- **使用時機**: 當外部系統需要簡單的SmartSSRFDetector.__init__到SsrfConfig轉換時
- **預期結果**: 獲得基於程式邏輯的SsrfConfig結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 340: 服務：Ssrfconfig

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Ssrfconfig的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 340
```
```bash
aiva_internal_executor.py --flow 340 --dry-run
```

---

### SmartSSRFDetector._extract_token → domain.split

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._extract_token到domain.split的處理能力
- **使用時機**: 當外部系統需要簡單的SmartSSRFDetector._extract_token到domain.split轉換時
- **預期結果**: 獲得基於程式邏輯的domain.split結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 351: 服務：Domain.Split

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Domain.Split的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 351
```
```bash
aiva_internal_executor.py --flow 351 --dry-run
```

---

### SmartSSRFDetector._issue_request → self._process_parameter_injection

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._issue_request到self._process_parameter_injection的處理能力
- **使用時機**: 當需要快速從SmartSSRFDetector._issue_request獲取self._process_parameter_injection的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._process_parameter_injection結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 346: 服務：Self. Process Parameter Injection

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Process Parameter Injection的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 346
```
```bash
aiva_internal_executor.py --flow 346 --dry-run
```

---

### SmartSSRFDetector._prioritize_vectors → other_vectors.append

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._prioritize_vectors到other_vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供SmartSSRFDetector._prioritize_vectors的other_vectors.append服務時
- **預期結果**: 獲得部分AI輔助的other_vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 342: 服務：Other Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Other Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 342
```
```bash
aiva_internal_executor.py --flow 342 --dry-run
```

---

### SmartSSRFDetector._process_parameter_injection → handler

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._process_parameter_injection到handler的處理能力
- **使用時機**: 當外部系統需要簡單的SmartSSRFDetector._process_parameter_injection到handler轉換時
- **預期結果**: 獲得基於程式邏輯的handler結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 347: 服務：Handler

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Handler的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 347
```
```bash
aiva_internal_executor.py --flow 347 --dry-run
```

---

### SmartSSRFDetector._resolve_payload → payload.replace

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._resolve_payload到payload.replace的處理能力
- **使用時機**: 當外部系統需要簡單的SmartSSRFDetector._resolve_payload到payload.replace轉換時
- **預期結果**: 獲得基於程式邏輯的payload.replace結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 345: 服務：Payload.Replace

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Payload.Replace的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 345
```
```bash
aiva_internal_executor.py --flow 345 --dry-run
```

---

### SmartSSRFDetector._verify_service_content → response.text.lower

**AI描述欄位 📋**:
- **能力概要**: SmartSSRFDetector._verify_service_content到response.text.lower的處理能力
- **使用時機**: 當外部系統需要簡單的SmartSSRFDetector._verify_service_content到response.text.lower轉換時
- **預期結果**: 獲得基於程式邏輯的response.text.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 350: 服務：Response.Text.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Response.Text.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 350
```
```bash
aiva_internal_executor.py --flow 350 --dry-run
```

---

### SqliCandidate → Field

**AI描述欄位 📋**:
- **能力概要**: SqliCandidate到Field的處理能力
- **使用時機**: 當需要為外部用戶提供SqliCandidate的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 28: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 28
```
```bash
aiva_internal_executor.py --flow 28 --dry-run
```

---

### SqliConfig.validate → ValueError

**AI描述欄位 📋**:
- **能力概要**: SqliConfig.validate到ValueError的處理能力
- **使用時機**: 當需要為外部用戶提供SqliConfig.validate的ValueError服務時
- **預期結果**: 獲得基於程式邏輯的ValueError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 156: 服務：Valueerror

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Valueerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 156
```
```bash
aiva_internal_executor.py --flow 156 --dry-run
```

---

### SqliContext → field

**AI描述欄位 📋**:
- **能力概要**: SqliContext到field的處理能力
- **使用時機**: 當外部系統需要簡單的SqliContext到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 199: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 199
```
```bash
aiva_internal_executor.py --flow 199 --dry-run
```

---

### SqliDetector.__init__ → self.engines.append

**AI描述欄位 📋**:
- **能力概要**: SqliDetector.__init__到self.engines.append的處理能力
- **使用時機**: 當需要快速從SqliDetector.__init__獲取self.engines.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.engines.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 210: 服務：Self.Engines.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Engines.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 210
```
```bash
aiva_internal_executor.py --flow 210 --dry-run
```

---

### SqliDetector._deduplicate_and_normalize → merged.append

**AI描述欄位 📋**:
- **能力概要**: SqliDetector._deduplicate_and_normalize到merged.append的處理能力
- **使用時機**: 當需要為外部用戶提供SqliDetector._deduplicate_and_normalize的merged.append服務時
- **預期結果**: 獲得基於程式邏輯的merged.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 215: 服務：Merged.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Merged.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 215
```
```bash
aiva_internal_executor.py --flow 215 --dry-run
```

---

### SqliDetector._process_and_merge_results → flat_results.extend

**AI描述欄位 📋**:
- **能力概要**: SqliDetector._process_and_merge_results到flat_results.extend的處理能力
- **使用時機**: 當外部系統需要簡單的SqliDetector._process_and_merge_results到flat_results.extend轉換時
- **預期結果**: 獲得基於程式邏輯的flat_results.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 214: 服務：Flat Results.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Flat Results.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 214
```
```bash
aiva_internal_executor.py --flow 214 --dry-run
```

---

### SqliDetector._try_import_engine → cls

**AI描述欄位 📋**:
- **能力概要**: SqliDetector._try_import_engine到cls的處理能力
- **使用時機**: 當需要為外部用戶提供SqliDetector._try_import_engine的cls服務時
- **預期結果**: 獲得基於程式邏輯的cls結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 211: 服務：Cls

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Cls相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 211
```
```bash
aiva_internal_executor.py --flow 211 --dry-run
```

---

### SqliError.__str__ → parts.append

**AI描述欄位 📋**:
- **能力概要**: SqliError.__str__到parts.append的處理能力
- **使用時機**: 當需要為外部用戶提供SqliError.__str__的parts.append服務時
- **預期結果**: 獲得基於程式邏輯的parts.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 157: 服務：Parts.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Parts.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 157
```
```bash
aiva_internal_executor.py --flow 157 --dry-run
```

---

### SqliExecutionTelemetry → field

**AI描述欄位 📋**:
- **能力概要**: SqliExecutionTelemetry到field的處理能力
- **使用時機**: 當外部系統需要簡單的SqliExecutionTelemetry到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 194: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 194
```
```bash
aiva_internal_executor.py --flow 194 --dry-run
```

---

### SqliExecutionTelemetry.add_engine → self.record_engine_execution

**AI描述欄位 📋**:
- **能力概要**: SqliExecutionTelemetry.add_engine到self.record_engine_execution的處理能力
- **使用時機**: 當需要快速從SqliExecutionTelemetry.add_engine獲取self.record_engine_execution的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.record_engine_execution結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 197: 服務：Self.Record Engine Execution

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Record Engine Execution的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 197
```
```bash
aiva_internal_executor.py --flow 197 --dry-run
```

---

### SqliExecutionTelemetry.add_error → self.record_error

**AI描述欄位 📋**:
- **能力概要**: SqliExecutionTelemetry.add_error到self.record_error的處理能力
- **使用時機**: 當需要快速從SqliExecutionTelemetry.add_error獲取self.record_error的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.record_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 198: 服務：Self.Record Error

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Record Error的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 198
```
```bash
aiva_internal_executor.py --flow 198 --dry-run
```

---

### SqliExecutionTelemetry.record_engine_execution → self.engines_run.append

**AI描述欄位 📋**:
- **能力概要**: SqliExecutionTelemetry.record_engine_execution到self.engines_run.append的處理能力
- **使用時機**: 當需要快速從SqliExecutionTelemetry.record_engine_execution獲取self.engines_run.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.engines_run.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 195: 服務：Self.Engines Run.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Engines Run.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 195
```
```bash
aiva_internal_executor.py --flow 195 --dry-run
```

---

### SqliExecutionTelemetry.record_error → self.errors.append

**AI描述欄位 📋**:
- **能力概要**: SqliExecutionTelemetry.record_error到self.errors.append的處理能力
- **使用時機**: 當需要快速從SqliExecutionTelemetry.record_error獲取self.errors.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.errors.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 196: 服務：Self.Errors.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Errors.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 196
```
```bash
aiva_internal_executor.py --flow 196 --dry-run
```

---

### SqliOrchestrator.__init__ → self._setup_default_engines

**AI描述欄位 📋**:
- **能力概要**: SqliOrchestrator.__init__到self._setup_default_engines的處理能力
- **使用時機**: 當需要對SqliOrchestrator.__init__進行深度分析並生成self._setup_default_engines結果時
- **預期結果**: 獲得部分AI輔助的self._setup_default_engines結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 200: 服務：Self. Setup Default Engines

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self. Setup Default Engines相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 200
```
```bash
aiva_internal_executor.py --flow 200 --dry-run
```

---

### SqliOrchestrator._setup_default_engines → self.register_engine

**AI描述欄位 📋**:
- **能力概要**: SqliOrchestrator._setup_default_engines到self.register_engine的處理能力
- **使用時機**: 當需要對SqliOrchestrator._setup_default_engines進行深度分析並生成self.register_engine結果時
- **預期結果**: 獲得部分AI輔助的self.register_engine結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 201: 服務：Self.Register Engine

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self.Register Engine相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 201
```
```bash
aiva_internal_executor.py --flow 201 --dry-run
```

---

### SqliResultBinderPublisher.__init__ → uuid.uuid4

**AI描述欄位 📋**:
- **能力概要**: SqliResultBinderPublisher.__init__到uuid.uuid4的處理能力
- **使用時機**: 當外部系統需要簡單的SqliResultBinderPublisher.__init__到uuid.uuid4轉換時
- **預期結果**: 獲得基於程式邏輯的uuid.uuid4結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 186: 服務：Uuid.Uuid4

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Uuid.Uuid4的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 186
```
```bash
aiva_internal_executor.py --flow 186 --dry-run
```

---

### SqliResultBinderPublisher._publish → self._broker.publish

**AI描述欄位 📋**:
- **能力概要**: SqliResultBinderPublisher._publish到self._broker.publish的處理能力
- **使用時機**: 當需要快速從SqliResultBinderPublisher._publish獲取self._broker.publish的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._broker.publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 190: 服務：Self. Broker.Publish

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Broker.Publish的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 190
```
```bash
aiva_internal_executor.py --flow 190 --dry-run
```

---

### SqliResultBinderPublisher.publish_error → self.publish_status

**AI描述欄位 📋**:
- **能力概要**: SqliResultBinderPublisher.publish_error到self.publish_status的處理能力
- **使用時機**: 當需要快速從SqliResultBinderPublisher.publish_error獲取self.publish_status的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.publish_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 188: 服務：Self.Publish Status

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Publish Status的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 188
```
```bash
aiva_internal_executor.py --flow 188 --dry-run
```

---

### SqliResultBinderPublisher.publish_finding → self._publish

**AI描述欄位 📋**:
- **能力概要**: SqliResultBinderPublisher.publish_finding到self._publish的處理能力
- **使用時機**: 當需要快速從SqliResultBinderPublisher.publish_finding獲取self._publish的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 189: 服務：Self. Publish

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Publish的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 189
```
```bash
aiva_internal_executor.py --flow 189 --dry-run
```

---

### SqliWorkerService.__init__ → SqliEngineConfig

**AI描述欄位 📋**:
- **能力概要**: SqliWorkerService.__init__到SqliEngineConfig的處理能力
- **使用時機**: 當需要為外部用戶提供SqliWorkerService.__init__的SqliEngineConfig服務時
- **預期結果**: 獲得部分AI輔助的SqliEngineConfig結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 203: 服務：Sqliengineconfig

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Sqliengineconfig相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 203
```
```bash
aiva_internal_executor.py --flow 203 --dry-run
```

---

### SqlmapIntegration.__init__ → self._find_sqlmap_path

**AI描述欄位 📋**:
- **能力概要**: SqlmapIntegration.__init__到self._find_sqlmap_path的處理能力
- **使用時機**: 當需要快速從SqlmapIntegration.__init__獲取self._find_sqlmap_path的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._find_sqlmap_path結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 267: 服務：Self. Find Sqlmap Path

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Find Sqlmap Path的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 267
```
```bash
aiva_internal_executor.py --flow 267 --dry-run
```

---

### SqlmapIntegration._parse_sqlmap_output → results.append

**AI描述欄位 📋**:
- **能力概要**: SqlmapIntegration._parse_sqlmap_output到results.append的處理能力
- **使用時機**: 當需要為外部用戶提供SqlmapIntegration._parse_sqlmap_output的results.append服務時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 270: 服務：Results.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 270
```
```bash
aiva_internal_executor.py --flow 270 --dry-run
```

---

### SqlmapIntegration.install_sqlmap → process.communicate

**AI描述欄位 📋**:
- **能力概要**: SqlmapIntegration.install_sqlmap到process.communicate的處理能力
- **使用時機**: 當外部系統需要簡單的SqlmapIntegration.install_sqlmap到process.communicate轉換時
- **預期結果**: 獲得基於程式邏輯的process.communicate結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 268: 服務：Process.Communicate

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Process.Communicate的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 268
```
```bash
aiva_internal_executor.py --flow 268 --dry-run
```

---

### SqlmapIntegration.scan_target → cmd.extend

**AI描述欄位 📋**:
- **能力概要**: SqlmapIntegration.scan_target到cmd.extend的處理能力
- **使用時機**: 當需要為外部用戶提供SqlmapIntegration.scan_target的cmd.extend服務時
- **預期結果**: 獲得基於程式邏輯的cmd.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 269: 服務：Cmd.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Cmd.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 269
```
```bash
aiva_internal_executor.py --flow 269 --dry-run
```

---

### SsrfCandidate → Field

**AI描述欄位 📋**:
- **能力概要**: SsrfCandidate到Field的處理能力
- **使用時機**: 當需要為外部用戶提供SsrfCandidate的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 29: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 29
```
```bash
aiva_internal_executor.py --flow 29 --dry-run
```

---

### SsrfResultPublisher.__init__ → new_id

**AI描述欄位 📋**:
- **能力概要**: SsrfResultPublisher.__init__到new_id的處理能力
- **使用時機**: 當外部系統需要簡單的SsrfResultPublisher.__init__到new_id轉換時
- **預期結果**: 獲得基於程式邏輯的new_id結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 332: 服務：New Id

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：New Id的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 332
```
```bash
aiva_internal_executor.py --flow 332 --dry-run
```

---

### SsrfResultPublisher._publish → self._broker.publish

**AI描述欄位 📋**:
- **能力概要**: SsrfResultPublisher._publish到self._broker.publish的處理能力
- **使用時機**: 當需要對SsrfResultPublisher._publish進行深度分析並生成self._broker.publish結果時
- **預期結果**: 獲得部分AI輔助的self._broker.publish結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 336: 服務：Self. Broker.Publish

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self. Broker.Publish相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 336
```
```bash
aiva_internal_executor.py --flow 336 --dry-run
```

---

### SsrfResultPublisher.publish_error → self.publish_status

**AI描述欄位 📋**:
- **能力概要**: SsrfResultPublisher.publish_error到self.publish_status的處理能力
- **使用時機**: 當需要快速從SsrfResultPublisher.publish_error獲取self.publish_status的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.publish_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 335: 服務：Self.Publish Status

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Publish Status的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 335
```
```bash
aiva_internal_executor.py --flow 335 --dry-run
```

---

### SsrfResultPublisher.publish_finding → self._publish

**AI描述欄位 📋**:
- **能力概要**: SsrfResultPublisher.publish_finding到self._publish的處理能力
- **使用時機**: 當需要快速從SsrfResultPublisher.publish_finding獲取self._publish的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 334: 服務：Self. Publish

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Publish的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 334
```
```bash
aiva_internal_executor.py --flow 334 --dry-run
```

---

### SsrfTelemetry → field

**AI描述欄位 📋**:
- **能力概要**: SsrfTelemetry到field的處理能力
- **使用時機**: 當外部系統需要簡單的SsrfTelemetry到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 352: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 352
```
```bash
aiva_internal_executor.py --flow 352 --dry-run
```

---

### StoredXSSDetector._submit_payloads → session.get

**AI描述欄位 📋**:
- **能力概要**: StoredXSSDetector._submit_payloads到session.get的處理能力
- **使用時機**: 當需要為外部用戶提供StoredXSSDetector._submit_payloads的session.get服務時
- **預期結果**: 獲得基於程式邏輯的session.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 545: 服務：Session.Get

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Session.Get相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 545
```
```bash
aiva_internal_executor.py --flow 545 --dry-run
```

---

### StoredXSSDetector.scan_stored_xss → vulnerabilities.extend

**AI描述欄位 📋**:
- **能力概要**: StoredXSSDetector.scan_stored_xss到vulnerabilities.extend的處理能力
- **使用時機**: 當需要對StoredXSSDetector.scan_stored_xss進行深度分析並生成vulnerabilities.extend結果時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 544: 服務：Vulnerabilities.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Vulnerabilities.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 544
```
```bash
aiva_internal_executor.py --flow 544 --dry-run
```

---

### StoredXssDetector._inject_query → urlencode

**AI描述欄位 📋**:
- **能力概要**: StoredXssDetector._inject_query到urlencode的處理能力
- **使用時機**: 當外部系統需要簡單的StoredXssDetector._inject_query到urlencode轉換時
- **預期結果**: 獲得基於程式邏輯的urlencode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 467: 服務：Urlencode

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Urlencode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 467
```
```bash
aiva_internal_executor.py --flow 467 --dry-run
```

---

### StoredXssDetector._submit_payload → payload.encode

**AI描述欄位 📋**:
- **能力概要**: StoredXssDetector._submit_payload到payload.encode的處理能力
- **使用時機**: 當需要快速從StoredXssDetector._submit_payload獲取payload.encode的基礎信息時
- **預期結果**: 獲得基於程式邏輯的payload.encode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 465: 服務：Payload.Encode

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Payload.Encode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 465
```
```bash
aiva_internal_executor.py --flow 465 --dry-run
```

---

### StoredXssDetector._verify_persistence → html.escape

**AI描述欄位 📋**:
- **能力概要**: StoredXssDetector._verify_persistence到html.escape的處理能力
- **使用時機**: 當外部系統需要簡單的StoredXssDetector._verify_persistence到html.escape轉換時
- **預期結果**: 獲得基於程式邏輯的html.escape結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 466: 服務：Html.Escape

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Html.Escape的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 466
```
```bash
aiva_internal_executor.py --flow 466 --dry-run
```

---

### StrategyGenerationConfig → Field

**AI描述欄位 📋**:
- **能力概要**: StrategyGenerationConfig到Field的處理能力
- **使用時機**: 當需要為外部用戶提供StrategyGenerationConfig的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 33: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 33
```
```bash
aiva_internal_executor.py --flow 33 --dry-run
```

---

### SubdomainEnumerator.__init__ → set

**AI描述欄位 📋**:
- **能力概要**: SubdomainEnumerator.__init__到set的處理能力
- **使用時機**: 當外部系統需要簡單的SubdomainEnumerator.__init__到set轉換時
- **預期結果**: 獲得基於程式邏輯的set結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 380: 服務：Set

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Set的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 380
```
```bash
aiva_internal_executor.py --flow 380 --dry-run
```

---

### SubdomainEnumerator._enumerate_common_subdomains → self.found_subdomains.add

**AI描述欄位 📋**:
- **能力概要**: SubdomainEnumerator._enumerate_common_subdomains到self.found_subdomains.add的處理能力
- **使用時機**: 當需要快速從SubdomainEnumerator._enumerate_common_subdomains獲取self.found_subdomains.add的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.found_subdomains.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 384: 服務：Self.Found Subdomains.Add

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Found Subdomains.Add的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 384
```
```bash
aiva_internal_executor.py --flow 384 --dry-run
```

---

### SubdomainEnumerator._enumerate_crt_sh → self.found_subdomains.add

**AI描述欄位 📋**:
- **能力概要**: SubdomainEnumerator._enumerate_crt_sh到self.found_subdomains.add的處理能力
- **使用時機**: 當需要對SubdomainEnumerator._enumerate_crt_sh進行深度分析並生成self.found_subdomains.add結果時
- **預期結果**: 獲得基於程式邏輯的self.found_subdomains.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 382: 服務：Self.Found Subdomains.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Found Subdomains.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 382
```
```bash
aiva_internal_executor.py --flow 382 --dry-run
```

---

### SubdomainEnumerator._enumerate_dns_brute → self.found_subdomains.add

**AI描述欄位 📋**:
- **能力概要**: SubdomainEnumerator._enumerate_dns_brute到self.found_subdomains.add的處理能力
- **使用時機**: 當需要快速從SubdomainEnumerator._enumerate_dns_brute獲取self.found_subdomains.add的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.found_subdomains.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 383: 服務：Self.Found Subdomains.Add

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Found Subdomains.Add的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 383
```
```bash
aiva_internal_executor.py --flow 383 --dry-run
```

---

### SubdomainEnumerator.enumerate_subdomains → self._enumerate_common_subdomains

**AI描述欄位 📋**:
- **能力概要**: SubdomainEnumerator.enumerate_subdomains到self._enumerate_common_subdomains的處理能力
- **使用時機**: 當需要對SubdomainEnumerator.enumerate_subdomains進行深度分析並生成self._enumerate_common_subdomains結果時
- **預期結果**: 獲得基於程式邏輯的self._enumerate_common_subdomains結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 381: 服務：Self. Enumerate Common Subdomains

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Enumerate Common Subdomains相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 381
```
```bash
aiva_internal_executor.py --flow 381 --dry-run
```

---

### SubdomainScanner.__init__ → dns.resolver.Resolver

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner.__init__到dns.resolver.Resolver的處理能力
- **使用時機**: 當需要對SubdomainScanner.__init__進行深度分析並生成dns.resolver.Resolver結果時
- **預期結果**: 獲得部分AI輔助的dns.resolver.Resolver結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 427: 服務：Dns.Resolver.Resolver

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Dns.Resolver.Resolver相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 427
```
```bash
aiva_internal_executor.py --flow 427 --dry-run
```

---

### SubdomainScanner._bruteforce_discovery → subdomains.add

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner._bruteforce_discovery到subdomains.add的處理能力
- **使用時機**: 當需要對SubdomainScanner._bruteforce_discovery進行深度分析並生成subdomains.add結果時
- **預期結果**: 獲得部分AI輔助的subdomains.add結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 431: 服務：Subdomains.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Subdomains.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 431
```
```bash
aiva_internal_executor.py --flow 431 --dry-run
```

---

### SubdomainScanner._dns_zone_transfer → subdomains.add

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner._dns_zone_transfer到subdomains.add的處理能力
- **使用時機**: 當需要對SubdomainScanner._dns_zone_transfer進行深度分析並生成subdomains.add結果時
- **預期結果**: 獲得部分AI輔助的subdomains.add結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 432: 服務：Subdomains.Add

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Subdomains.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 432
```
```bash
aiva_internal_executor.py --flow 432 --dry-run
```

---

### SubdomainScanner._passive_discovery → subdomains.update

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner._passive_discovery到subdomains.update的處理能力
- **使用時機**: 當需要為外部用戶提供SubdomainScanner._passive_discovery的subdomains.update服務時
- **預期結果**: 獲得部分AI輔助的subdomains.update結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 429: 服務：Subdomains.Update

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Subdomains.Update相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 429
```
```bash
aiva_internal_executor.py --flow 429 --dry-run
```

---

### SubdomainScanner._resolve_domain → self.resolver.resolve

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner._resolve_domain到self.resolver.resolve的處理能力
- **使用時機**: 當需要快速從SubdomainScanner._resolve_domain獲取self.resolver.resolve的基礎信息時
- **預期結果**: 獲得部分AI輔助的self.resolver.resolve結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 433: 服務：Self.Resolver.Resolve

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Self.Resolver.Resolve的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 433
```
```bash
aiva_internal_executor.py --flow 433 --dry-run
```

---

### SubdomainScanner._search_crtsh → subdomains.add

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner._search_crtsh到subdomains.add的處理能力
- **使用時機**: 當需要為外部用戶提供SubdomainScanner._search_crtsh的subdomains.add服務時
- **預期結果**: 獲得部分AI輔助的subdomains.add結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 430: 服務：Subdomains.Add

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Subdomains.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 430
```
```bash
aiva_internal_executor.py --flow 430 --dry-run
```

---

### SubdomainScanner.scan → subdomains.update

**AI描述欄位 📋**:
- **能力概要**: SubdomainScanner.scan到subdomains.update的處理能力
- **使用時機**: 當需要為外部用戶提供SubdomainScanner.scan的subdomains.update服務時
- **預期結果**: 獲得部分AI輔助的subdomains.update結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 428: 服務：Subdomains.Update

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Subdomains.Update相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 428
```
```bash
aiva_internal_executor.py --flow 428 --dry-run
```

---

### SystemOrchestration → Field

**AI描述欄位 📋**:
- **能力概要**: SystemOrchestration到Field的處理能力
- **使用時機**: 當需要為外部用戶提供SystemOrchestration的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 24: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 24
```
```bash
aiva_internal_executor.py --flow 24 --dry-run
```

---

### TechDetector.__init__ → self._load_fingerprints

**AI描述欄位 📋**:
- **能力概要**: TechDetector.__init__到self._load_fingerprints的處理能力
- **使用時機**: 當需要對TechDetector.__init__進行深度分析並生成self._load_fingerprints結果時
- **預期結果**: 獲得基於程式邏輯的self._load_fingerprints結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 434: 服務：Self. Load Fingerprints

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Load Fingerprints相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 434
```
```bash
aiva_internal_executor.py --flow 434 --dry-run
```

---

### TechDetector.detect → technologies.update

**AI描述欄位 📋**:
- **能力概要**: TechDetector.detect到technologies.update的處理能力
- **使用時機**: 當需要對TechDetector.detect進行深度分析並生成technologies.update結果時
- **預期結果**: 獲得基於程式邏輯的technologies.update結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 435: 服務：Technologies.Update

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Technologies.Update相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 435
```
```bash
aiva_internal_executor.py --flow 435 --dry-run
```

---

### TechnologyDetector._detect_css_frameworks → self.technologies.append

**AI描述欄位 📋**:
- **能力概要**: TechnologyDetector._detect_css_frameworks到self.technologies.append的處理能力
- **使用時機**: 當需要快速從TechnologyDetector._detect_css_frameworks獲取self.technologies.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.technologies.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 396: 服務：Self.Technologies.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Technologies.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 396
```
```bash
aiva_internal_executor.py --flow 396 --dry-run
```

---

### TechnologyDetector._detect_frameworks → self.technologies.append

**AI描述欄位 📋**:
- **能力概要**: TechnologyDetector._detect_frameworks到self.technologies.append的處理能力
- **使用時機**: 當需要快速從TechnologyDetector._detect_frameworks獲取self.technologies.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.technologies.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 394: 服務：Self.Technologies.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Technologies.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 394
```
```bash
aiva_internal_executor.py --flow 394 --dry-run
```

---

### TechnologyDetector._detect_js_libraries → self.technologies.append

**AI描述欄位 📋**:
- **能力概要**: TechnologyDetector._detect_js_libraries到self.technologies.append的處理能力
- **使用時機**: 當需要快速從TechnologyDetector._detect_js_libraries獲取self.technologies.append的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.technologies.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 395: 服務：Self.Technologies.Append

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Technologies.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 395
```
```bash
aiva_internal_executor.py --flow 395 --dry-run
```

---

### TechnologyDetector.detect_technologies → self._detect_css_frameworks

**AI描述欄位 📋**:
- **能力概要**: TechnologyDetector.detect_technologies到self._detect_css_frameworks的處理能力
- **使用時機**: 當需要對TechnologyDetector.detect_technologies進行深度分析並生成self._detect_css_frameworks結果時
- **預期結果**: 獲得基於程式邏輯的self._detect_css_frameworks結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 393: 服務：Self. Detect Css Frameworks

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Detect Css Frameworks相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 393
```
```bash
aiva_internal_executor.py --flow 393 --dry-run
```

---

### TimeDetectionEngine._measure_baseline_times → times.append

**AI描述欄位 📋**:
- **能力概要**: TimeDetectionEngine._measure_baseline_times到times.append的處理能力
- **使用時機**: 當需要為外部用戶提供TimeDetectionEngine._measure_baseline_times的times.append服務時
- **預期結果**: 獲得基於程式邏輯的times.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 238: 服務：Times.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Times.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 238
```
```bash
aiva_internal_executor.py --flow 238 --dry-run
```

---

### TimeDetectionEngine._measure_payload_time → client.request

**AI描述欄位 📋**:
- **能力概要**: TimeDetectionEngine._measure_payload_time到client.request的處理能力
- **使用時機**: 當外部系統需要簡單的TimeDetectionEngine._measure_payload_time到client.request轉換時
- **預期結果**: 獲得基於程式邏輯的client.request結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 239: 服務：Client.Request

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Client.Request的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 239
```
```bash
aiva_internal_executor.py --flow 239 --dry-run
```

---

### TimeDetectionEngine.detect → results.append

**AI描述欄位 📋**:
- **能力概要**: TimeDetectionEngine.detect到results.append的處理能力
- **使用時機**: 當需要對TimeDetectionEngine.detect進行深度分析並生成results.append結果時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 237: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 237
```
```bash
aiva_internal_executor.py --flow 237 --dry-run
```

---

### TraditionalXssDetector.__init__ → max

**AI描述欄位 📋**:
- **能力概要**: TraditionalXssDetector.__init__到max的處理能力
- **使用時機**: 當外部系統需要簡單的TraditionalXssDetector.__init__到max轉換時
- **預期結果**: 獲得基於程式邏輯的max結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 474: 服務：Max

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Max的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 474
```
```bash
aiva_internal_executor.py --flow 474 --dry-run
```

---

### TraditionalXssDetector._build_request_parts → target.body.encode

**AI描述欄位 📋**:
- **能力概要**: TraditionalXssDetector._build_request_parts到target.body.encode的處理能力
- **使用時機**: 當需要為外部用戶提供TraditionalXssDetector._build_request_parts的target.body.encode服務時
- **預期結果**: 獲得基於程式邏輯的target.body.encode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 476: 服務：Target.Body.Encode

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Target.Body.Encode相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 476
```
```bash
aiva_internal_executor.py --flow 476 --dry-run
```

---

### UnionDetectionEngine._check_content_change → sorted

**AI描述欄位 📋**:
- **能力概要**: UnionDetectionEngine._check_content_change到sorted的處理能力
- **使用時機**: 當需要為外部用戶提供UnionDetectionEngine._check_content_change的sorted服務時
- **預期結果**: 獲得基於程式邏輯的sorted結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 244: 服務：Sorted

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Sorted相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 244
```
```bash
aiva_internal_executor.py --flow 244 --dry-run
```

---

### UnionDetectionEngine._check_union_success → content.lower

**AI描述欄位 📋**:
- **能力概要**: UnionDetectionEngine._check_union_success到content.lower的處理能力
- **使用時機**: 當外部系統需要簡單的UnionDetectionEngine._check_union_success到content.lower轉換時
- **預期結果**: 獲得基於程式邏輯的content.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 243: 服務：Content.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Content.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 243
```
```bash
aiva_internal_executor.py --flow 243 --dry-run
```

---

### UnionDetectionEngine._get_baseline_response → encoder.encode

**AI描述欄位 📋**:
- **能力概要**: UnionDetectionEngine._get_baseline_response到encoder.encode的處理能力
- **使用時機**: 當外部系統需要簡單的UnionDetectionEngine._get_baseline_response到encoder.encode轉換時
- **預期結果**: 獲得基於程式邏輯的encoder.encode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 242: 服務：Encoder.Encode

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Encoder.Encode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 242
```
```bash
aiva_internal_executor.py --flow 242 --dry-run
```

---

### UnionDetectionEngine.detect → results.append

**AI描述欄位 📋**:
- **能力概要**: UnionDetectionEngine.detect到results.append的處理能力
- **使用時機**: 當需要對UnionDetectionEngine.detect進行深度分析並生成results.append結果時
- **預期結果**: 獲得基於程式邏輯的results.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 241: 服務：Results.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Results.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 241
```
```bash
aiva_internal_executor.py --flow 241 --dry-run
```

---

### WebAttackCLI._comprehensive_scan → self._display_scan_results

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._comprehensive_scan到self._display_scan_results的處理能力
- **使用時機**: 當需要快速從WebAttackCLI._comprehensive_scan獲取self._display_scan_results的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._display_scan_results結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 401: 服務：Self. Display Scan Results

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Display Scan Results的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 401
```
```bash
aiva_internal_executor.py --flow 401 --dry-run
```

---

### WebAttackCLI._directory_scan → table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._directory_scan到table.add_row的處理能力
- **使用時機**: 當需要對WebAttackCLI._directory_scan進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得部分AI輔助的table.add_row結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 403: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 403
```
```bash
aiva_internal_executor.py --flow 403 --dry-run
```

---

### WebAttackCLI._display_scan_results → vuln_table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._display_scan_results到vuln_table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供WebAttackCLI._display_scan_results的vuln_table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的vuln_table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 408: 服務：Vuln Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Vuln Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 408
```
```bash
aiva_internal_executor.py --flow 408 --dry-run
```

---

### WebAttackCLI._export_results → open

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._export_results到open的處理能力
- **使用時機**: 當需要為外部用戶提供WebAttackCLI._export_results的open服務時
- **預期結果**: 獲得基於程式邏輯的open結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 407: 服務：Open

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Open相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 407
```
```bash
aiva_internal_executor.py --flow 407 --dry-run
```

---

### WebAttackCLI._show_scan_history → table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._show_scan_history到table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供WebAttackCLI._show_scan_history的table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 406: 服務：Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 406
```
```bash
aiva_internal_executor.py --flow 406 --dry-run
```

---

### WebAttackCLI._subdomain_enumeration → table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._subdomain_enumeration到table.add_row的處理能力
- **使用時機**: 當需要對WebAttackCLI._subdomain_enumeration進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 402: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 402
```
```bash
aiva_internal_executor.py --flow 402 --dry-run
```

---

### WebAttackCLI._technology_detection → table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI._technology_detection到table.add_row的處理能力
- **使用時機**: 當需要對WebAttackCLI._technology_detection進行深度分析並生成table.add_row結果時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 405: 服務：Table.Add Row

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 405
```
```bash
aiva_internal_executor.py --flow 405 --dry-run
```

---

### WebAttackCLI.show_main_menu → table.add_row

**AI描述欄位 📋**:
- **能力概要**: WebAttackCLI.show_main_menu到table.add_row的處理能力
- **使用時機**: 當需要為外部用戶提供WebAttackCLI.show_main_menu的table.add_row服務時
- **預期結果**: 獲得基於程式邏輯的table.add_row結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 399: 服務：Table.Add Row

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Table.Add Row相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 399
```
```bash
aiva_internal_executor.py --flow 399 --dry-run
```

---

### WebCrawler.__init__ → self.session.headers.update

**AI描述欄位 📋**:
- **能力概要**: WebCrawler.__init__到self.session.headers.update的處理能力
- **使用時機**: 當需要對WebCrawler.__init__進行深度分析並生成self.session.headers.update結果時
- **預期結果**: 獲得基於程式邏輯的self.session.headers.update結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 441: 服務：Self.Session.Headers.Update

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Session.Headers.Update相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 441
```
```bash
aiva_internal_executor.py --flow 441 --dry-run
```

---

### WebCrawler._crawl_page → self._extract_parameters

**AI描述欄位 📋**:
- **能力概要**: WebCrawler._crawl_page到self._extract_parameters的處理能力
- **使用時機**: 當需要對WebCrawler._crawl_page進行深度分析並生成self._extract_parameters結果時
- **預期結果**: 獲得基於程式邏輯的self._extract_parameters結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 443: 服務：Self. Extract Parameters

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self. Extract Parameters相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 443
```
```bash
aiva_internal_executor.py --flow 443 --dry-run
```

---

### WebCrawler._extract_forms → forms.append

**AI描述欄位 📋**:
- **能力概要**: WebCrawler._extract_forms到forms.append的處理能力
- **使用時機**: 當外部系統需要簡單的WebCrawler._extract_forms到forms.append轉換時
- **預期結果**: 獲得基於程式邏輯的forms.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 444: 服務：Forms.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Forms.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 444
```
```bash
aiva_internal_executor.py --flow 444 --dry-run
```

---

### WebCrawler._extract_links → links.append

**AI描述欄位 📋**:
- **能力概要**: WebCrawler._extract_links到links.append的處理能力
- **使用時機**: 當需要為外部用戶提供WebCrawler._extract_links的links.append服務時
- **預期結果**: 獲得基於程式邏輯的links.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 445: 服務：Links.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Links.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 445
```
```bash
aiva_internal_executor.py --flow 445 --dry-run
```

---

### WebCrawler._extract_parameters → parameters.add

**AI描述欄位 📋**:
- **能力概要**: WebCrawler._extract_parameters到parameters.add的處理能力
- **使用時機**: 當需要為外部用戶提供WebCrawler._extract_parameters的parameters.add服務時
- **預期結果**: 獲得基於程式邏輯的parameters.add結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 446: 服務：Parameters.Add

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Parameters.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 446
```
```bash
aiva_internal_executor.py --flow 446 --dry-run
```

---

### WebCrawler.crawl → to_visit.append

**AI描述欄位 📋**:
- **能力概要**: WebCrawler.crawl到to_visit.append的處理能力
- **使用時機**: 當需要對WebCrawler.crawl進行深度分析並生成to_visit.append結果時
- **預期結果**: 獲得基於程式邏輯的to_visit.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 442: 服務：To Visit.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：To Visit.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 442
```
```bash
aiva_internal_executor.py --flow 442 --dry-run
```

---

### WebScannerManager._detect_technologies_sync → technologies.append

**AI描述欄位 📋**:
- **能力概要**: WebScannerManager._detect_technologies_sync到technologies.append的處理能力
- **使用時機**: 當需要為外部用戶提供WebScannerManager._detect_technologies_sync的technologies.append服務時
- **預期結果**: 獲得部分AI輔助的technologies.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 376: 服務：Technologies.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Technologies.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 376
```
```bash
aiva_internal_executor.py --flow 376 --dry-run
```

---

### WebScannerManager._scan_directories_sync → directories.append

**AI描述欄位 📋**:
- **能力概要**: WebScannerManager._scan_directories_sync到directories.append的處理能力
- **使用時機**: 當需要為外部用戶提供WebScannerManager._scan_directories_sync的directories.append服務時
- **預期結果**: 獲得部分AI輔助的directories.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 375: 服務：Directories.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Directories.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 375
```
```bash
aiva_internal_executor.py --flow 375 --dry-run
```

---

### WebScannerManager._scan_subdomains_sync → subdomains.append

**AI描述欄位 📋**:
- **能力概要**: WebScannerManager._scan_subdomains_sync到subdomains.append的處理能力
- **使用時機**: 當需要為外部用戶提供WebScannerManager._scan_subdomains_sync的subdomains.append服務時
- **預期結果**: 獲得部分AI輔助的subdomains.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 374: 服務：Subdomains.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Subdomains.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 374
```
```bash
aiva_internal_executor.py --flow 374 --dry-run
```

---

### WebScannerManager.scan → findings.extend

**AI描述欄位 📋**:
- **能力概要**: WebScannerManager.scan到findings.extend的處理能力
- **使用時機**: 當需要對WebScannerManager.scan進行深度分析並生成findings.extend結果時
- **預期結果**: 獲得部分AI輔助的findings.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 373: 服務：Findings.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 373
```
```bash
aiva_internal_executor.py --flow 373 --dry-run
```

---

### WebTarget → field

**AI描述欄位 📋**:
- **能力概要**: WebTarget到field的處理能力
- **使用時機**: 當需要為外部用戶提供WebTarget的field服務時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 378: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 378
```
```bash
aiva_internal_executor.py --flow 378 --dry-run
```

---

### WebTarget.__post_init__ → urllib.parse.urlparse

**AI描述欄位 📋**:
- **能力概要**: WebTarget.__post_init__到urllib.parse.urlparse的處理能力
- **使用時機**: 當外部系統需要簡單的WebTarget.__post_init__到urllib.parse.urlparse轉換時
- **預期結果**: 獲得基於程式邏輯的urllib.parse.urlparse結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 379: 服務：Urllib.Parse.Urlparse

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Urllib.Parse.Urlparse的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 379
```
```bash
aiva_internal_executor.py --flow 379 --dry-run
```

---

### WorkflowBypassScanner.run_all_tests → all_findings.extend

**AI描述欄位 📋**:
- **能力概要**: WorkflowBypassScanner.run_all_tests到all_findings.extend的處理能力
- **使用時機**: 當外部系統需要簡單的WorkflowBypassScanner.run_all_tests到all_findings.extend轉換時
- **預期結果**: 獲得部分AI輔助的all_findings.extend結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 55: 服務：All Findings.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：All Findings.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 55
```
```bash
aiva_internal_executor.py --flow 55 --dry-run
```

---

### WorkflowBypassScanner.test_admin_access_bypass → findings.append

**AI描述欄位 📋**:
- **能力概要**: WorkflowBypassScanner.test_admin_access_bypass到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供WorkflowBypassScanner.test_admin_access_bypass的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 54: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 54
```
```bash
aiva_internal_executor.py --flow 54 --dry-run
```

---

### WorkflowBypassScanner.test_direct_checkout → findings.append

**AI描述欄位 📋**:
- **能力概要**: WorkflowBypassScanner.test_direct_checkout到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供WorkflowBypassScanner.test_direct_checkout的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 51: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 51
```
```bash
aiva_internal_executor.py --flow 51 --dry-run
```

---

### WorkflowBypassScanner.test_payment_bypass → findings.append

**AI描述欄位 📋**:
- **能力概要**: WorkflowBypassScanner.test_payment_bypass到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供WorkflowBypassScanner.test_payment_bypass的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 52: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 52
```
```bash
aiva_internal_executor.py --flow 52 --dry-run
```

---

### WorkflowBypassScanner.test_step_skipping → findings.append

**AI描述欄位 📋**:
- **能力概要**: WorkflowBypassScanner.test_step_skipping到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供WorkflowBypassScanner.test_step_skipping的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 50: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 50
```
```bash
aiva_internal_executor.py --flow 50 --dry-run
```

---

### WorkflowBypassScanner.test_verification_bypass → findings.append

**AI描述欄位 📋**:
- **能力概要**: WorkflowBypassScanner.test_verification_bypass到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供WorkflowBypassScanner.test_verification_bypass的findings.append服務時
- **預期結果**: 獲得部分AI輔助的findings.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 53: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 53
```
```bash
aiva_internal_executor.py --flow 53 --dry-run
```

---

### XSSManager.__init__ → BlindXSSDetector

**AI描述欄位 📋**:
- **能力概要**: XSSManager.__init__到BlindXSSDetector的處理能力
- **使用時機**: 當需要為外部用戶提供XSSManager.__init__的BlindXSSDetector服務時
- **預期結果**: 獲得基於程式邏輯的BlindXSSDetector結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 554: 服務：Blindxssdetector

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Blindxssdetector相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 554
```
```bash
aiva_internal_executor.py --flow 554 --dry-run
```

---

### XSSManager._custom_xss_scan → vulnerabilities.append

**AI描述欄位 📋**:
- **能力概要**: XSSManager._custom_xss_scan到vulnerabilities.append的處理能力
- **使用時機**: 當需要對XSSManager._custom_xss_scan進行深度分析並生成vulnerabilities.append結果時
- **預期結果**: 獲得基於程式邏輯的vulnerabilities.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 557: 服務：Vulnerabilities.Append

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Vulnerabilities.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 557
```
```bash
aiva_internal_executor.py --flow 557 --dry-run
```

---

### XSSManager._generate_summary → all_vulns.extend

**AI描述欄位 📋**:
- **能力概要**: XSSManager._generate_summary到all_vulns.extend的處理能力
- **使用時機**: 當外部系統需要簡單的XSSManager._generate_summary到all_vulns.extend轉換時
- **預期結果**: 獲得基於程式邏輯的all_vulns.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 558: 服務：All Vulns.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：All Vulns.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 558
```
```bash
aiva_internal_executor.py --flow 558 --dry-run
```

---

### XSSManager._parse_target → parameters.items

**AI描述欄位 📋**:
- **能力概要**: XSSManager._parse_target到parameters.items的處理能力
- **使用時機**: 當外部系統需要簡單的XSSManager._parse_target到parameters.items轉換時
- **預期結果**: 獲得基於程式邏輯的parameters.items結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 555: 服務：Parameters.Items

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Parameters.Items的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 555
```
```bash
aiva_internal_executor.py --flow 555 --dry-run
```

---

### XSSManager.comprehensive_scan → self.scan_results.extend

**AI描述欄位 📋**:
- **能力概要**: XSSManager.comprehensive_scan到self.scan_results.extend的處理能力
- **使用時機**: 當需要對XSSManager.comprehensive_scan進行深度分析並生成self.scan_results.extend結果時
- **預期結果**: 獲得基於程式邏輯的self.scan_results.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 556: 服務：Self.Scan Results.Extend

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Scan Results.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 556
```
```bash
aiva_internal_executor.py --flow 556 --dry-run
```

---

### XSSPayloadGenerator.__init__ → self._load_context_specific_payloads

**AI描述欄位 📋**:
- **能力概要**: XSSPayloadGenerator.__init__到self._load_context_specific_payloads的處理能力
- **使用時機**: 當需要快速從XSSPayloadGenerator.__init__獲取self._load_context_specific_payloads的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._load_context_specific_payloads結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 539: 服務：Self. Load Context Specific Payloads

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Load Context Specific Payloads的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 539
```
```bash
aiva_internal_executor.py --flow 539 --dry-run
```

---

### XSSPayloadGenerator.generate_payloads → payloads.extend

**AI描述欄位 📋**:
- **能力概要**: XSSPayloadGenerator.generate_payloads到payloads.extend的處理能力
- **使用時機**: 當需要為外部用戶提供XSSPayloadGenerator.generate_payloads的payloads.extend服務時
- **預期結果**: 獲得基於程式邏輯的payloads.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 540: 服務：Payloads.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Payloads.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 540
```
```bash
aiva_internal_executor.py --flow 540 --dry-run
```

---

### XXEDetector.__init__ → self._generate_payloads

**AI描述欄位 📋**:
- **能力概要**: XXEDetector.__init__到self._generate_payloads的處理能力
- **使用時機**: 當需要快速從XXEDetector.__init__獲取self._generate_payloads的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._generate_payloads結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 638: 服務：Self. Generate Payloads

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Generate Payloads的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 638
```
```bash
aiva_internal_executor.py --flow 638 --dry-run
```

---

### XssCandidate → Field

**AI描述欄位 📋**:
- **能力概要**: XssCandidate到Field的處理能力
- **使用時機**: 當需要為外部用戶提供XssCandidate的Field服務時
- **預期結果**: 獲得基於程式邏輯的Field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 27: 服務：Field

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Field相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 27
```
```bash
aiva_internal_executor.py --flow 27 --dry-run
```

---

### XssExecutionTelemetry → field

**AI描述欄位 📋**:
- **能力概要**: XssExecutionTelemetry到field的處理能力
- **使用時機**: 當外部系統需要簡單的XssExecutionTelemetry到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 481: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 481
```
```bash
aiva_internal_executor.py --flow 481 --dry-run
```

---

### XssPayloadGenerator.generate → ordered.setdefault

**AI描述欄位 📋**:
- **能力概要**: XssPayloadGenerator.generate到ordered.setdefault的處理能力
- **使用時機**: 當需要為外部用戶提供XssPayloadGenerator.generate的ordered.setdefault服務時
- **預期結果**: 獲得基於程式邏輯的ordered.setdefault結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 458: 服務：Ordered.Setdefault

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Ordered.Setdefault相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 458
```
```bash
aiva_internal_executor.py --flow 458 --dry-run
```

---

### XssResultPublisher.__init__ → new_id

**AI描述欄位 📋**:
- **能力概要**: XssResultPublisher.__init__到new_id的處理能力
- **使用時機**: 當外部系統需要簡單的XssResultPublisher.__init__到new_id轉換時
- **預期結果**: 獲得基於程式邏輯的new_id結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 459: 服務：New Id

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：New Id的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 459
```
```bash
aiva_internal_executor.py --flow 459 --dry-run
```

---

### XssResultPublisher._publish → self._broker.publish

**AI描述欄位 📋**:
- **能力概要**: XssResultPublisher._publish到self._broker.publish的處理能力
- **使用時機**: 當需要對XssResultPublisher._publish進行深度分析並生成self._broker.publish結果時
- **預期結果**: 獲得部分AI輔助的self._broker.publish結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 463: 服務：Self. Broker.Publish

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Self. Broker.Publish相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 463
```
```bash
aiva_internal_executor.py --flow 463 --dry-run
```

---

### XssResultPublisher.publish_error → self.publish_status

**AI描述欄位 📋**:
- **能力概要**: XssResultPublisher.publish_error到self.publish_status的處理能力
- **使用時機**: 當需要快速從XssResultPublisher.publish_error獲取self.publish_status的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self.publish_status結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 462: 服務：Self.Publish Status

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self.Publish Status的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 462
```
```bash
aiva_internal_executor.py --flow 462 --dry-run
```

---

### XssResultPublisher.publish_finding → self._publish

**AI描述欄位 📋**:
- **能力概要**: XssResultPublisher.publish_finding到self._publish的處理能力
- **使用時機**: 當需要快速從XssResultPublisher.publish_finding獲取self._publish的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._publish結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 461: 服務：Self. Publish

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Publish的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 461
```
```bash
aiva_internal_executor.py --flow 461 --dry-run
```

---

### _QueueEntry → field

**AI描述欄位 📋**:
- **能力概要**: _QueueEntry到field的處理能力
- **使用時機**: 當外部系統需要簡單的_QueueEntry到field轉換時
- **預期結果**: 獲得基於程式邏輯的field結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 468: 服務：Field

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Field的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 468
```
```bash
aiva_internal_executor.py --flow 468 --dry-run
```

---

### _build_blind_finding → FindingEvidence

**AI描述欄位 📋**:
- **能力概要**: _build_blind_finding到FindingEvidence的處理能力
- **使用時機**: 當外部系統需要簡單的_build_blind_finding到FindingEvidence轉換時
- **預期結果**: 獲得基於程式邏輯的FindingEvidence結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 496: 服務：Findingevidence

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Findingevidence的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 496
```
```bash
aiva_internal_executor.py --flow 496 --dry-run
```

---

### _build_finding → FindingEvidence

**AI描述欄位 📋**:
- **能力概要**: _build_finding到FindingEvidence的處理能力
- **使用時機**: 當外部系統需要簡單的_build_finding到FindingEvidence轉換時
- **預期結果**: 獲得基於程式邏輯的FindingEvidence結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 495: 服務：Findingevidence

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Findingevidence的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 495
```
```bash
aiva_internal_executor.py --flow 495 --dry-run
```

---

### _build_oast_finding → FindingEvidence

**AI描述欄位 📋**:
- **能力概要**: _build_oast_finding到FindingEvidence的處理能力
- **使用時機**: 當外部系統需要簡單的_build_oast_finding到FindingEvidence轉換時
- **預期結果**: 獲得基於程式邏輯的FindingEvidence結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 359: 服務：Findingevidence

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Findingevidence的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 359
```
```bash
aiva_internal_executor.py --flow 359 --dry-run
```

---

### _build_payloads → generator.generate

**AI描述欄位 📋**:
- **能力概要**: _build_payloads到generator.generate的處理能力
- **使用時機**: 當需要為外部用戶提供_build_payloads的generator.generate服務時
- **預期結果**: 獲得基於程式邏輯的generator.generate結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 494: 服務：Generator.Generate

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Generator.Generate相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 494
```
```bash
aiva_internal_executor.py --flow 494 --dry-run
```

---

### _detect_waf_interference → response_text.lower

**AI描述欄位 📋**:
- **能力概要**: _detect_waf_interference到response_text.lower的處理能力
- **使用時機**: 當外部系統需要簡單的_detect_waf_interference到response_text.lower轉換時
- **預期結果**: 獲得基於程式邏輯的response_text.lower結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 480: 服務：Response Text.Lower

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Response Text.Lower的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 480
```
```bash
aiva_internal_executor.py --flow 480 --dry-run
```

---

### _finalize_statistics → stats_collector.finalize

**AI描述欄位 📋**:
- **能力概要**: _finalize_statistics到stats_collector.finalize的處理能力
- **使用時機**: 當需要為外部用戶提供_finalize_statistics的stats_collector.finalize服務時
- **預期結果**: 獲得基於程式邏輯的stats_collector.finalize結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 493: 服務：Stats Collector.Finalize

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Stats Collector.Finalize相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 493
```
```bash
aiva_internal_executor.py --flow 493 --dry-run
```

---

### _find_go_binary → Path

**AI描述欄位 📋**:
- **能力概要**: _find_go_binary到Path的處理能力
- **使用時機**: 當外部系統需要簡單的_find_go_binary到Path轉換時
- **預期結果**: 獲得基於程式邏輯的Path結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 10: 服務：Path

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Path的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 10
```
```bash
aiva_internal_executor.py --flow 10 --dry-run
```

---

### _format_request → lines.append

**AI描述欄位 📋**:
- **能力概要**: _format_request到lines.append的處理能力
- **使用時機**: 當需要為外部用戶提供_format_request的lines.append服務時
- **預期結果**: 獲得基於程式邏輯的lines.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 360: 服務：Lines.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Lines.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 360
```
```bash
aiva_internal_executor.py --flow 360 --dry-run
```

---

### _format_response → lines.append

**AI描述欄位 📋**:
- **能力概要**: _format_response到lines.append的處理能力
- **使用時機**: 當需要為外部用戶提供_format_response的lines.append服務時
- **預期結果**: 獲得基於程式邏輯的lines.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 361: 服務：Lines.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Lines.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 361
```
```bash
aiva_internal_executor.py --flow 361 --dry-run
```

---

### _get_level_and_log → self.info

**AI描述欄位 📋**:
- **能力概要**: _get_level_and_log到self.info的處理能力
- **使用時機**: 當需要對_get_level_and_log進行深度分析並生成self.info結果時
- **預期結果**: 獲得基於程式邏輯的self.info結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 580: 服務：Self.Info

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Info相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 580
```
```bash
aiva_internal_executor.py --flow 580 --dry-run
```

---

### _good → self._log

**AI描述欄位 📋**:
- **能力概要**: _good到self._log的處理能力
- **使用時機**: 當需要快速從_good獲取self._log的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._log結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 577: 服務：Self. Log

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Log的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 577
```
```bash
aiva_internal_executor.py --flow 577 --dry-run
```

---

### _handle_detection_errors → error.to_detail

**AI描述欄位 📋**:
- **能力概要**: _handle_detection_errors到error.to_detail的處理能力
- **使用時機**: 當需要為外部用戶提供_handle_detection_errors的error.to_detail服務時
- **預期結果**: 獲得基於程式邏輯的error.to_detail結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 488: 服務：Error.To Detail

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Error.To Detail相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 488
```
```bash
aiva_internal_executor.py --flow 488 --dry-run
```

---

### _inject_query → urlencode

**AI描述欄位 📋**:
- **能力概要**: _inject_query到urlencode的處理能力
- **使用時機**: 當外部系統需要簡單的_inject_query到urlencode轉換時
- **預期結果**: 獲得基於程式邏輯的urlencode結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 477: 服務：Urlencode

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Urlencode的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 477
```
```bash
aiva_internal_executor.py --flow 477 --dry-run
```

---

### _is_at_or_above → _to_comparable

**AI描述欄位 📋**:
- **能力概要**: _is_at_or_above到_to_comparable的處理能力
- **使用時機**: 當外部系統需要簡單的_is_at_or_above到_to_comparable轉換時
- **預期結果**: 獲得基於程式邏輯的_to_comparable結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 611: 服務： To Comparable

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務： To Comparable的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 611
```
```bash
aiva_internal_executor.py --flow 611 --dry-run
```

---

### _issue_request → client.request

**AI描述欄位 📋**:
- **能力概要**: _issue_request到client.request的處理能力
- **使用時機**: 當需要為外部用戶提供_issue_request的client.request服務時
- **預期結果**: 獲得基於程式邏輯的client.request結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 357: 服務：Client.Request

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Client.Request相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 357
```
```bash
aiva_internal_executor.py --flow 357 --dry-run
```

---

### _payload_in_response → unescape

**AI描述欄位 📋**:
- **能力概要**: _payload_in_response到unescape的處理能力
- **使用時機**: 當外部系統需要簡單的_payload_in_response到unescape轉換時
- **預期結果**: 獲得基於程式邏輯的unescape結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 478: 服務：Unescape

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Unescape的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 478
```
```bash
aiva_internal_executor.py --flow 478 --dry-run
```

---

### _replacement_match → ar.group

**AI描述欄位 📋**:
- **能力概要**: _replacement_match到ar.group的處理能力
- **使用時機**: 當外部系統需要簡單的_replacement_match到ar.group轉換時
- **預期結果**: 獲得基於程式邏輯的ar.group結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 609: 服務：Ar.Group

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Ar.Group的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 609
```
```bash
aiva_internal_executor.py --flow 609 --dry-run
```

---

### _resolve_payload → payload.replace

**AI描述欄位 📋**:
- **能力概要**: _resolve_payload到payload.replace的處理能力
- **使用時機**: 當外部系統需要簡單的_resolve_payload到payload.replace轉換時
- **預期結果**: 獲得基於程式邏輯的payload.replace結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 356: 服務：Payload.Replace

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Payload.Replace的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 356
```
```bash
aiva_internal_executor.py --flow 356 --dry-run
```

---

### _run → self._log

**AI描述欄位 📋**:
- **能力概要**: _run到self._log的處理能力
- **使用時機**: 當需要快速從_run獲取self._log的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._log結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 576: 服務：Self. Log

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Log的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 576
```
```bash
aiva_internal_executor.py --flow 576 --dry-run
```

---

### _setup_blind_xss → stats_collector.record_error

**AI描述欄位 📋**:
- **能力概要**: _setup_blind_xss到stats_collector.record_error的處理能力
- **使用時機**: 當需要為外部用戶提供_setup_blind_xss的stats_collector.record_error服務時
- **預期結果**: 獲得基於程式邏輯的stats_collector.record_error結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 486: 服務：Stats Collector.Record Error

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Stats Collector.Record Error相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 486
```
```bash
aiva_internal_executor.py --flow 486 --dry-run
```

---

### _simple_match → deJSON

**AI描述欄位 📋**:
- **能力概要**: _simple_match到deJSON的處理能力
- **使用時機**: 當外部系統需要簡單的_simple_match到deJSON轉換時
- **預期結果**: 獲得基於程式邏輯的deJSON結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 608: 服務：Dejson

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Dejson的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 608
```
```bash
aiva_internal_executor.py --flow 608 --dry-run
```

---

### _switch_to_default_loggers → self.addHandler

**AI描述欄位 📋**:
- **能力概要**: _switch_to_default_loggers到self.addHandler的處理能力
- **使用時機**: 當需要對_switch_to_default_loggers進行深度分析並生成self.addHandler結果時
- **預期結果**: 獲得基於程式邏輯的self.addHandler結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 579: 服務：Self.Addhandler

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Addhandler相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 579
```
```bash
aiva_internal_executor.py --flow 579 --dry-run
```

---

### _switch_to_no_format_loggers → self.addHandler

**AI描述欄位 📋**:
- **能力概要**: _switch_to_no_format_loggers到self.addHandler的處理能力
- **使用時機**: 當需要對_switch_to_no_format_loggers進行深度分析並生成self.addHandler結果時
- **預期結果**: 獲得基於程式邏輯的self.addHandler結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 578: 服務：Self.Addhandler

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Addhandler相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 578
```
```bash
aiva_internal_executor.py --flow 578 --dry-run
```

---

### _verify_execution_context → match.group

**AI描述欄位 📋**:
- **能力概要**: _verify_execution_context到match.group的處理能力
- **使用時機**: 當需要為外部用戶提供_verify_execution_context的match.group服務時
- **預期結果**: 獲得基於程式邏輯的match.group結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 479: 服務：Match.Group

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Match.Group相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 479
```
```bash
aiva_internal_executor.py --flow 479 --dry-run
```

---

### _vuln → self._log

**AI描述欄位 📋**:
- **能力概要**: _vuln到self._log的處理能力
- **使用時機**: 當需要快速從_vuln獲取self._log的基礎信息時
- **預期結果**: 獲得基於程式邏輯的self._log結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 575: 服務：Self. Log

- **範圍**: internal
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Self. Log的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 575
```
```bash
aiva_internal_executor.py --flow 575 --dry-run
```

---

### bruteforcer → encoding

**AI描述欄位 📋**:
- **能力概要**: bruteforcer到encoding的處理能力
- **使用時機**: 當需要為外部用戶提供bruteforcer的encoding服務時
- **預期結果**: 獲得基於程式邏輯的encoding結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 604: 服務：Encoding

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Encoding相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 604
```
```bash
aiva_internal_executor.py --flow 604 --dry-run
```

---

### check → result.get

**AI描述欄位 📋**:
- **能力概要**: check到result.get的處理能力
- **使用時機**: 當外部系統需要簡單的check到result.get轉換時
- **預期結果**: 獲得基於程式邏輯的result.get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 610: 服務：Result.Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Result.Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 610
```
```bash
aiva_internal_executor.py --flow 610 --dry-run
```

---

### checker → efficiencies.append

**AI描述欄位 📋**:
- **能力概要**: checker到efficiencies.append的處理能力
- **使用時機**: 當需要為外部用戶提供checker的efficiencies.append服務時
- **預期結果**: 獲得基於程式邏輯的efficiencies.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 568: 服務：Efficiencies.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Efficiencies.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 568
```
```bash
aiva_internal_executor.py --flow 568 --dry-run
```

---

### closest → abs

**AI描述欄位 📋**:
- **能力概要**: closest到abs的處理能力
- **使用時機**: 當外部系統需要簡單的closest到abs轉換時
- **預期結果**: 獲得基於程式邏輯的abs結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 590: 服務：Abs

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Abs的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 590
```
```bash
aiva_internal_executor.py --flow 590 --dry-run
```

---

### converter → data.split

**AI描述欄位 📋**:
- **能力概要**: converter到data.split的處理能力
- **使用時機**: 當外部系統需要簡單的converter到data.split轉換時
- **預期結果**: 獲得基於程式邏輯的data.split結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 589: 服務：Data.Split

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Data.Split的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 589
```
```bash
aiva_internal_executor.py --flow 589 --dry-run
```

---

### crawl → requester

**AI描述欄位 📋**:
- **能力概要**: crawl到requester的處理能力
- **使用時機**: 當需要為外部用戶提供crawl的requester服務時
- **預期結果**: 獲得基於程式邏輯的requester結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 605: 服務：Requester

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Requester相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 605
```
```bash
aiva_internal_executor.py --flow 605 --dry-run
```

---

### detect_xss → get_xss_engine

**AI描述欄位 📋**:
- **能力概要**: detect_xss到get_xss_engine的處理能力
- **使用時機**: 當外部系統需要簡單的detect_xss到get_xss_engine轉換時
- **預期結果**: 獲得基於程式邏輯的get_xss_engine結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 534: 服務：Get Xss Engine

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Get Xss Engine的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 534
```
```bash
aiva_internal_executor.py --flow 534 --dry-run
```

---

### dom → highlighted.append

**AI描述欄位 📋**:
- **能力概要**: dom到highlighted.append的處理能力
- **使用時機**: 當需要為外部用戶提供dom的highlighted.append服務時
- **預期結果**: 獲得基於程式邏輯的highlighted.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 569: 服務：Highlighted.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Highlighted.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 569
```
```bash
aiva_internal_executor.py --flow 569 --dry-run
```

---

### dorkFind → f.write

**AI描述欄位 📋**:
- **能力概要**: dorkFind到f.write的處理能力
- **使用時機**: 當需要為外部用戶提供dorkFind的f.write服務時
- **預期結果**: 獲得基於程式邏輯的f.write結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 560: 服務：F.Write

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：F.Write相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 560
```
```bash
aiva_internal_executor.py --flow 560 --dry-run
```

---

### entryy → sleep

**AI描述欄位 📋**:
- **能力概要**: entryy到sleep的處理能力
- **使用時機**: 當需要為外部用戶提供entryy的sleep服務時
- **預期結果**: 獲得基於程式邏輯的sleep結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 561: 服務：Sleep

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Sleep相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 561
```
```bash
aiva_internal_executor.py --flow 561 --dry-run
```

---

### equalize → array.append

**AI描述欄位 📋**:
- **能力概要**: equalize到array.append的處理能力
- **使用時機**: 當外部系統需要簡單的equalize到array.append轉換時
- **預期結果**: 獲得基於程式邏輯的array.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 600: 服務：Array.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Array.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 600
```
```bash
aiva_internal_executor.py --flow 600 --dry-run
```

---

### escaped → match.group

**AI描述欄位 📋**:
- **能力概要**: escaped到match.group的處理能力
- **使用時機**: 當外部系統需要簡單的escaped到match.group轉換時
- **預期結果**: 獲得基於程式邏輯的match.group結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 601: 服務：Match.Group

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Match.Group的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 601
```
```bash
aiva_internal_executor.py --flow 601 --dry-run
```

---

### extractHeaders → headers.replace

**AI描述欄位 📋**:
- **能力概要**: extractHeaders到headers.replace的處理能力
- **使用時機**: 當外部系統需要簡單的extractHeaders到headers.replace轉換時
- **預期結果**: 獲得基於程式邏輯的headers.replace結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 592: 服務：Headers.Replace

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Headers.Replace的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 592
```
```bash
aiva_internal_executor.py --flow 592 --dry-run
```

---

### extractScripts → scripts.append

**AI描述欄位 📋**:
- **能力概要**: extractScripts到scripts.append的處理能力
- **使用時機**: 當外部系統需要簡單的extractScripts到scripts.append轉換時
- **預期結果**: 獲得基於程式邏輯的scripts.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 593: 服務：Scripts.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Scripts.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 593
```
```bash
aiva_internal_executor.py --flow 593 --dry-run
```

---

### fillHoles → filled.extend

**AI描述欄位 📋**:
- **能力概要**: fillHoles到filled.extend的處理能力
- **使用時機**: 當外部系統需要簡單的fillHoles到filled.extend轉換時
- **預期結果**: 獲得基於程式邏輯的filled.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 591: 服務：Filled.Extend

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Filled.Extend的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 591
```
```bash
aiva_internal_executor.py --flow 591 --dry-run
```

---

### filterChecker → efficiencies.extend

**AI描述欄位 📋**:
- **能力概要**: filterChecker到efficiencies.extend的處理能力
- **使用時機**: 當需要為外部用戶提供filterChecker的efficiencies.extend服務時
- **預期結果**: 獲得基於程式邏輯的efficiencies.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 570: 服務：Efficiencies.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Efficiencies.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 570
```
```bash
aiva_internal_executor.py --flow 570 --dry-run
```

---

### flattenParams → flatted.append

**AI描述欄位 📋**:
- **能力概要**: flattenParams到flatted.append的處理能力
- **使用時機**: 當外部系統需要簡單的flattenParams到flatted.append轉換時
- **預期結果**: 獲得基於程式邏輯的flatted.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 594: 服務：Flatted.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Flatted.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 594
```
```bash
aiva_internal_executor.py --flow 594 --dry-run
```

---

### fuzzer → encoding

**AI描述欄位 📋**:
- **能力概要**: fuzzer到encoding的處理能力
- **使用時機**: 當需要為外部用戶提供fuzzer的encoding服務時
- **預期結果**: 獲得基於程式邏輯的encoding結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 571: 服務：Encoding

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Encoding相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 571
```
```bash
aiva_internal_executor.py --flow 571 --dry-run
```

---

### genGen → vectors.append

**AI描述欄位 📋**:
- **能力概要**: genGen到vectors.append的處理能力
- **使用時機**: 當需要為外部用戶提供genGen的vectors.append服務時
- **預期結果**: 獲得部分AI輔助的vectors.append結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 595: 服務：Vectors.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vectors.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 595
```
```bash
aiva_internal_executor.py --flow 595 --dry-run
```

---

### generator → set

**AI描述欄位 📋**:
- **能力概要**: generator到set的處理能力
- **使用時機**: 當需要為外部用戶提供generator的set服務時
- **預期結果**: 獲得基於程式邏輯的set結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 572: 服務：Set

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Set相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 572
```
```bash
aiva_internal_executor.py --flow 572 --dry-run
```

---

### getParams → each.append

**AI描述欄位 📋**:
- **能力概要**: getParams到each.append的處理能力
- **使用時機**: 當需要為外部用戶提供getParams的each.append服務時
- **預期結果**: 獲得基於程式邏輯的each.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 596: 服務：Each.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Each.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 596
```
```bash
aiva_internal_executor.py --flow 596 --dry-run
```

---

### get_engine_info → result.stdout.strip

**AI描述欄位 📋**:
- **能力概要**: get_engine_info到result.stdout.strip的處理能力
- **使用時機**: 當外部系統需要簡單的get_engine_info到result.stdout.strip轉換時
- **預期結果**: 獲得基於程式邏輯的result.stdout.strip結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 13: 服務：Result.Stdout.Strip

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Result.Stdout.Strip的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 13
```
```bash
aiva_internal_executor.py --flow 13 --dry-run
```

---

### get_user_agent → print

**AI描述欄位 📋**:
- **能力概要**: get_user_agent到print的處理能力
- **使用時機**: 當需要為外部用戶提供get_user_agent的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 559: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 559
```
```bash
aiva_internal_executor.py --flow 559 --dry-run
```

---

### get_xss_engine → _xss_engine_instance.initialize

**AI描述欄位 📋**:
- **能力概要**: get_xss_engine到_xss_engine_instance.initialize的處理能力
- **使用時機**: 當外部系統需要簡單的get_xss_engine到_xss_engine_instance.initialize轉換時
- **預期結果**: 獲得基於程式邏輯的_xss_engine_instance.initialize結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 533: 服務： Xss Engine Instance.Initialize

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務： Xss Engine Instance.Initialize的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 533
```
```bash
aiva_internal_executor.py --flow 533 --dry-run
```

---

### htmlParser → occurence.start

**AI描述欄位 📋**:
- **能力概要**: htmlParser到occurence.start的處理能力
- **使用時機**: 當需要為外部用戶提供htmlParser的occurence.start服務時
- **預期結果**: 獲得基於程式邏輯的occurence.start結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 573: 服務：Occurence.Start

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Occurence.Start相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 573
```
```bash
aiva_internal_executor.py --flow 573 --dry-run
```

---

### islem → print

**AI描述欄位 📋**:
- **能力概要**: islem到print的處理能力
- **使用時機**: 當需要為外部用戶提供islem的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 564: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 564
```
```bash
aiva_internal_executor.py --flow 564 --dry-run
```

---

### jsContexter → script.split

**AI描述欄位 📋**:
- **能力概要**: jsContexter到script.split的處理能力
- **使用時機**: 當外部系統需要簡單的jsContexter到script.split轉換時
- **預期結果**: 獲得基於程式邏輯的script.split結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 574: 服務：Script.Split

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Script.Split的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 574
```
```bash
aiva_internal_executor.py --flow 574 --dry-run
```

---

### js_extractor → scripts.append

**AI描述欄位 📋**:
- **能力概要**: js_extractor到scripts.append的處理能力
- **使用時機**: 當外部系統需要簡單的js_extractor到scripts.append轉換時
- **預期結果**: 獲得基於程式邏輯的scripts.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 599: 服務：Scripts.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Scripts.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 599
```
```bash
aiva_internal_executor.py --flow 599 --dry-run
```

---

### log_debug_json → self.debug

**AI描述欄位 📋**:
- **能力概要**: log_debug_json到self.debug的處理能力
- **使用時機**: 當需要對log_debug_json進行深度分析並生成self.debug結果時
- **預期結果**: 獲得基於程式邏輯的self.debug結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 583: 服務：Self.Debug

- **範圍**: internal
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Self.Debug相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 583
```
```bash
aiva_internal_executor.py --flow 583 --dry-run
```

---

### log_no_format → _switch_to_default_loggers

**AI描述欄位 📋**:
- **能力概要**: log_no_format到_switch_to_default_loggers的處理能力
- **使用時機**: 當需要為外部用戶提供log_no_format的_switch_to_default_loggers服務時
- **預期結果**: 獲得基於程式邏輯的_switch_to_default_loggers結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 582: 服務： Switch To Default Loggers

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務： Switch To Default Loggers相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 582
```
```bash
aiva_internal_executor.py --flow 582 --dry-run
```

---

### log_red_line → _switch_to_default_loggers

**AI描述欄位 📋**:
- **能力概要**: log_red_line到_switch_to_default_loggers的處理能力
- **使用時機**: 當需要為外部用戶提供log_red_line的_switch_to_default_loggers服務時
- **預期結果**: 獲得基於程式邏輯的_switch_to_default_loggers結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 581: 服務： Switch To Default Loggers

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務： Switch To Default Loggers相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 581
```
```bash
aiva_internal_executor.py --flow 581 --dry-run
```

---

### main → PathsConfig::new

**AI描述欄位 📋**:
- **能力概要**: main到PathsConfig::new的處理能力
- **使用時機**: 當外部系統需要簡單的main到PathsConfig::new轉換時
- **預期結果**: 獲得基於程式邏輯的PathsConfig::new結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 641: 服務：Pathsconfig::New

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Pathsconfig::New的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 641
```
```bash
aiva_internal_executor.py --flow 641 --dry-run
```

---

### main → cli.run

**AI描述欄位 📋**:
- **能力概要**: main到cli.run的處理能力
- **使用時機**: 當外部系統需要簡單的main到cli.run轉換時
- **預期結果**: 獲得基於程式邏輯的cli.run結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 182: 服務：Cli.Run

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Cli.Run的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 182
```
```bash
aiva_internal_executor.py --flow 182 --dry-run
```

---

### main → print

**AI描述欄位 📋**:
- **能力概要**: main到print的處理能力
- **使用時機**: 當需要為外部系統提供完整的main到print解決方案時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 113: 服務：Print

- **範圍**: external
- **複雜度**: complex
- **AI等級**: none
- **用途**: 進行服務：Print的高級處理，涉及複雜的邏輯判斷和多系統協調

**使用命令**:
```bash
aiva_internal_executor.py --flow 113
```
```bash
aiva_internal_executor.py --flow 113 --dry-run
```

---

### main → subparsers.add_parser

**AI描述欄位 📋**:
- **能力概要**: main到subparsers.add_parser的處理能力
- **使用時機**: 當需要為外部用戶提供main的subparsers.add_parser服務時
- **預期結果**: 獲得基於程式邏輯的subparsers.add_parser結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 60: 服務：Subparsers.Add Parser

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Subparsers.Add Parser相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 60
```
```bash
aiva_internal_executor.py --flow 60 --dry-run
```

---

### main_scanner → vulnerabilities.add

**AI描述欄位 📋**:
- **能力概要**: main_scanner到vulnerabilities.add的處理能力
- **使用時機**: 當需要為外部用戶提供main_scanner的vulnerabilities.add服務時
- **預期結果**: 獲得部分AI輔助的vulnerabilities.add結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 615: 服務：Vulnerabilities.Add

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Vulnerabilities.Add相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 615
```
```bash
aiva_internal_executor.py --flow 615 --dry-run
```

---

### mk_finding_dict → create_bizlogic_finding

**AI描述欄位 📋**:
- **能力概要**: mk_finding_dict到create_bizlogic_finding的處理能力
- **使用時機**: 當外部系統需要簡單的mk_finding_dict到create_bizlogic_finding轉換時
- **預期結果**: 獲得基於程式邏輯的create_bizlogic_finding結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 56: 服務：Create Bizlogic Finding

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Create Bizlogic Finding的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 56
```
```bash
aiva_internal_executor.py --flow 56 --dry-run
```

---

### payloadsList → f.read

**AI描述欄位 📋**:
- **能力概要**: payloadsList到f.read的處理能力
- **使用時機**: 當需要為外部用戶提供payloadsList的f.read服務時
- **預期結果**: 獲得基於程式邏輯的f.read結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 562: 服務：F.Read

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：F.Read相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 562
```
```bash
aiva_internal_executor.py --flow 562 --dry-run
```

---

### prompt → tmpfile.seek

**AI描述欄位 📋**:
- **能力概要**: prompt到tmpfile.seek的處理能力
- **使用時機**: 當需要為外部用戶提供prompt的tmpfile.seek服務時
- **預期結果**: 獲得基於程式邏輯的tmpfile.seek結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 586: 服務：Tmpfile.Seek

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Tmpfile.Seek相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 586
```
```bash
aiva_internal_executor.py --flow 586 --dry-run
```

---

### proxy_lister → file.writelines

**AI描述欄位 📋**:
- **能力概要**: proxy_lister到file.writelines的處理能力
- **使用時機**: 當需要為外部用戶提供proxy_lister的file.writelines服務時
- **預期結果**: 獲得基於程式邏輯的file.writelines結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 566: 服務：File.Writelines

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：File.Writelines相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 566
```
```bash
aiva_internal_executor.py --flow 566 --dry-run
```

---

### pylds → print

**AI描述欄位 📋**:
- **能力概要**: pylds到print的處理能力
- **使用時機**: 當需要為外部用戶提供pylds的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 563: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 563
```
```bash
aiva_internal_executor.py --flow 563 --dry-run
```

---

### reader → open

**AI描述欄位 📋**:
- **能力概要**: reader到open的處理能力
- **使用時機**: 當外部系統需要簡單的reader到open轉換時
- **預期結果**: 獲得基於程式邏輯的open結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 598: 服務：Open

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Open的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 598
```
```bash
aiva_internal_executor.py --flow 598 --dry-run
```

---

### requester → requests.post

**AI描述欄位 📋**:
- **能力概要**: requester到requests.post的處理能力
- **使用時機**: 當需要為外部用戶提供requester的requests.post服務時
- **預期結果**: 獲得基於程式邏輯的requests.post結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 587: 服務：Requests.Post

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Requests.Post相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 587
```
```bash
aiva_internal_executor.py --flow 587 --dry-run
```

---

### retireJs → main_scanner

**AI描述欄位 📋**:
- **能力概要**: retireJs到main_scanner的處理能力
- **使用時機**: 當需要為外部用戶提供retireJs的main_scanner服務時
- **預期結果**: 獲得部分AI輔助的main_scanner結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 616: 服務：Main Scanner

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Main Scanner相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 616
```
```bash
aiva_internal_executor.py --flow 616 --dry-run
```

---

### run → AivaMessage

**AI描述欄位 📋**:
- **能力概要**: run到AivaMessage的處理能力
- **使用時機**: 當需要為外部用戶提供run的AivaMessage服務時
- **預期結果**: 獲得部分AI輔助的AivaMessage結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 353: 服務：Aivamessage

- **範圍**: external
- **複雜度**: medium
- **AI等級**: basic
- **用途**: 執行服務：Aivamessage相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 353
```
```bash
aiva_internal_executor.py --flow 353 --dry-run
```

---

### run → new_id

**AI描述欄位 📋**:
- **能力概要**: run到new_id的處理能力
- **使用時機**: 當需要為外部用戶提供run的new_id服務時
- **預期結果**: 獲得基於程式邏輯的new_id結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 73: 服務：New Id

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：New Id相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 73
```
```bash
aiva_internal_executor.py --flow 73 --dry-run
```

---

### run_price_test → tester.run_all_tests

**AI描述欄位 📋**:
- **能力概要**: run_price_test到tester.run_all_tests的處理能力
- **使用時機**: 當外部系統需要簡單的run_price_test到tester.run_all_tests轉換時
- **預期結果**: 獲得基於程式邏輯的tester.run_all_tests結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 57: 服務：Tester.Run All Tests

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Tester.Run All Tests的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 57
```
```bash
aiva_internal_executor.py --flow 57 --dry-run
```

---

### run_race_test → tester.run_all_tests

**AI描述欄位 📋**:
- **能力概要**: run_race_test到tester.run_all_tests的處理能力
- **使用時機**: 當外部系統需要簡單的run_race_test到tester.run_all_tests轉換時
- **預期結果**: 獲得基於程式邏輯的tester.run_all_tests結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 58: 服務：Tester.Run All Tests

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Tester.Run All Tests的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 58
```
```bash
aiva_internal_executor.py --flow 58 --dry-run
```

---

### run_workflow_test → tester.run_all_tests

**AI描述欄位 📋**:
- **能力概要**: run_workflow_test到tester.run_all_tests的處理能力
- **使用時機**: 當需要為外部用戶提供run_workflow_test的tester.run_all_tests服務時
- **預期結果**: 獲得基於程式邏輯的tester.run_all_tests結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 59: 服務：Tester.Run All Tests

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Tester.Run All Tests相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 59
```
```bash
aiva_internal_executor.py --flow 59 --dry-run
```

---

### run_xss_test → findings.append

**AI描述欄位 📋**:
- **能力概要**: run_xss_test到findings.append的處理能力
- **使用時機**: 當需要為外部用戶提供run_xss_test的findings.append服務時
- **預期結果**: 獲得基於程式邏輯的findings.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 504: 服務：Findings.Append

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Findings.Append相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 504
```
```bash
aiva_internal_executor.py --flow 504 --dry-run
```

---

### scan → detected.append

**AI描述欄位 📋**:
- **能力概要**: scan到detected.append的處理能力
- **使用時機**: 當外部系統需要簡單的scan到detected.append轉換時
- **預期結果**: 獲得基於程式邏輯的detected.append結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 606: 服務：Detected.Append

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Detected.Append的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 606
```
```bash
aiva_internal_executor.py --flow 606 --dry-run
```

---

### scan_authentication → ValueError

**AI描述欄位 📋**:
- **能力概要**: scan_authentication到ValueError的處理能力
- **使用時機**: 當需要為外部用戶提供scan_authentication的ValueError服務時
- **預期結果**: 獲得基於程式邏輯的ValueError結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 12: 服務：Valueerror

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Valueerror相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 12
```
```bash
aiva_internal_executor.py --flow 12 --dry-run
```

---

### scan_file_content → _scanhash

**AI描述欄位 📋**:
- **能力概要**: scan_file_content到_scanhash的處理能力
- **使用時機**: 當需要為外部用戶提供scan_file_content的_scanhash服務時
- **預期結果**: 獲得基於程式邏輯的_scanhash結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 614: 服務： Scanhash

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務： Scanhash相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 614
```
```bash
aiva_internal_executor.py --flow 614 --dry-run
```

---

### scan_filename → scan

**AI描述欄位 📋**:
- **能力概要**: scan_filename到scan的處理能力
- **使用時機**: 當外部系統需要簡單的scan_filename到scan轉換時
- **預期結果**: 獲得基於程式邏輯的scan結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 613: 服務：Scan

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Scan的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 613
```
```bash
aiva_internal_executor.py --flow 613 --dry-run
```

---

### scan_target → AuthnManager

**AI描述欄位 📋**:
- **能力概要**: scan_target到AuthnManager的處理能力
- **使用時機**: 當外部系統需要簡單的scan_target到AuthnManager轉換時
- **預期結果**: 獲得基於程式邏輯的AuthnManager結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 9: 服務：Authnmanager

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Authnmanager的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 9
```
```bash
aiva_internal_executor.py --flow 9 --dry-run
```

---

### scan_target → PostExManager

**AI描述欄位 📋**:
- **能力概要**: scan_target到PostExManager的處理能力
- **使用時機**: 當外部系統需要簡單的scan_target到PostExManager轉換時
- **預期結果**: 獲得基於程式邏輯的PostExManager結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 103: 服務：Postexmanager

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Postexmanager的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 103
```
```bash
aiva_internal_executor.py --flow 103 --dry-run
```

---

### scan_target → WebScannerManager

**AI描述欄位 📋**:
- **能力概要**: scan_target到WebScannerManager的處理能力
- **使用時機**: 當外部系統需要簡單的scan_target到WebScannerManager轉換時
- **預期結果**: 獲得部分AI輔助的WebScannerManager結果，結合程式邏輯和智能處理
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 377: 服務：Webscannermanager

- **範圍**: external
- **複雜度**: simple
- **AI等級**: basic
- **用途**: 提供服務：Webscannermanager的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 377
```
```bash
aiva_internal_executor.py --flow 377 --dry-run
```

---

### scan_uri → scan

**AI描述欄位 📋**:
- **能力概要**: scan_uri到scan的處理能力
- **使用時機**: 當外部系統需要簡單的scan_uri到scan轉換時
- **預期結果**: 獲得基於程式邏輯的scan結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 612: 服務：Scan

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Scan的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 612
```
```bash
aiva_internal_executor.py --flow 612 --dry-run
```

---

### setup_logger → file_handler.setLevel

**AI描述欄位 📋**:
- **能力概要**: setup_logger到file_handler.setLevel的處理能力
- **使用時機**: 當需要為外部用戶提供setup_logger的file_handler.setLevel服務時
- **預期結果**: 獲得基於程式邏輯的file_handler.setLevel結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 584: 服務：File Handler.Setlevel

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：File Handler.Setlevel相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 584
```
```bash
aiva_internal_executor.py --flow 584 --dry-run
```

---

### singleFuzz → fuzzer

**AI描述欄位 📋**:
- **能力概要**: singleFuzz到fuzzer的處理能力
- **使用時機**: 當需要為外部用戶提供singleFuzz的fuzzer服務時
- **預期結果**: 獲得基於程式邏輯的fuzzer結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 607: 服務：Fuzzer

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Fuzzer相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 607
```
```bash
aiva_internal_executor.py --flow 607 --dry-run
```

---

### updater → get

**AI描述欄位 📋**:
- **能力概要**: updater到get的處理能力
- **使用時機**: 當外部系統需要簡單的updater到get轉換時
- **預期結果**: 獲得基於程式邏輯的get結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 588: 服務：Get

- **範圍**: external
- **複雜度**: simple
- **AI等級**: none
- **用途**: 提供服務：Get的基礎功能，適合快速執行簡單任務

**使用命令**:
```bash
aiva_internal_executor.py --flow 588
```
```bash
aiva_internal_executor.py --flow 588 --dry-run
```

---

### wafDetector → bestMatch.extend

**AI描述欄位 📋**:
- **能力概要**: wafDetector到bestMatch.extend的處理能力
- **使用時機**: 當需要為外部用戶提供wafDetector的bestMatch.extend服務時
- **預期結果**: 獲得基於程式邏輯的bestMatch.extend結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 602: 服務：Bestmatch.Extend

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Bestmatch.Extend相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 602
```
```bash
aiva_internal_executor.py --flow 602 --dry-run
```

---

### writer → savefile.close

**AI描述欄位 📋**:
- **能力概要**: writer到savefile.close的處理能力
- **使用時機**: 當需要為外部用戶提供writer的savefile.close服務時
- **預期結果**: 獲得基於程式邏輯的savefile.close結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 597: 服務：Savefile.Close

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Savefile.Close相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 597
```
```bash
aiva_internal_executor.py --flow 597 --dry-run
```

---

### xssFind → print

**AI描述欄位 📋**:
- **能力概要**: xssFind到print的處理能力
- **使用時機**: 當需要為外部用戶提供xssFind的print服務時
- **預期結果**: 獲得基於程式邏輯的print結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 567: 服務：Print

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：Print相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 567
```
```bash
aiva_internal_executor.py --flow 567 --dry-run
```

---

### zetanize → d

**AI描述欄位 📋**:
- **能力概要**: zetanize到d的處理能力
- **使用時機**: 當需要為外部用戶提供zetanize的d服務時
- **預期結果**: 獲得基於程式邏輯的d結果，確保準確性和一致性
- **路徑變體**: 1 種

**路徑選擇建議**:
- 單一路徑，直接使用

**具體能力**:

#### Flow 603: 服務：D

- **範圍**: external
- **複雜度**: medium
- **AI等級**: none
- **用途**: 執行服務：D相關的中等複雜度操作，包含多步驟處理流程

**使用命令**:
```bash
aiva_internal_executor.py --flow 603
```
```bash
aiva_internal_executor.py --flow 603 --dry-run
```

---

## AI使用指南

### 選擇策略

**step1**: 首先確定需要的模組範圍

**step2**: 識別起點和終點需求

**step3**: 根據複雜度和範圍選擇具體能力

**step4**: 參考AI描述欄位確認符合需求

### 模組優先級

- 認知核心模組：AI核心功能和決策
- 內探模組：系統自我分析和監控
- 任務規劃模組：規劃和執行管理
- 核心能力模組：基礎功能和工具
- 服務骨幹模組：基礎設施和支援

### AI閱讀提示

- 優先閱讀group_summary了解能力概況
- 查看ai_usage_context確定使用時機
- 根據selection_criteria選擇最適合的路徑變體
- 參考expected_outcome確認是否符合預期
