# AIVA 四大模組架構統一方案

**執行時間:** 2025-10-13  
**目標:** 在四大模組架構下統一命名、格式和 Schema

---

## 📊 當前四大模組架構

### 核心四大模組
1. **Core 模組** - 智慧分析與協調中心
2. **Scan 模組** - 資產發現與爬蟲引擎  
3. **Function 模組** - 漏洞檢測與測試
4. **Integration 模組** - 資料整合與報告生成

### 擴展智慧模組 (已實作)
- ✅ **ThreatIntel** - 威脅情報聚合 (歸屬 Core)
- ✅ **Remediation** - 自動化修復 (歸屬 Integration)
- ✅ **AuthZ** - 授權檢測 (歸屬 Function)
- ✅ **PostEx** - 後滲透測試 (歸屬 Function, 受限環境)

### 新建模組 (本次新增)
- 🆕 **BizLogic** - 業務邏輯漏洞測試 (歸屬 Function)

---

## 🔍 當前 Schema 狀況掃描

### ✅ 已存在的 Schema (aiva_common/schemas.py)

#### 基礎 Schema
- `ScanStartPayload` - 掃描啟動
- `ScanCompletedPayload` - 掃描完成
- `FindingPayload` - 漏洞發現
- `Asset` - 資產
- `Fingerprints` - 指紋
- `Summary` - 摘要

#### 智慧模組 Schema
- `ThreatIntelLookupPayload` - 威脅情報查詢
- `ThreatIntelResultPayload` - 威脅情報結果
- `AuthZCheckPayload` - 權限檢查
- `AuthZAnalysisPayload` - 權限分析
- `AuthZResultPayload` - 權限分析結果
- `RemediationGeneratePayload` - 修復方案生成
- `RemediationResultPayload` - 修復方案結果
- `PostExTestPayload` - 後滲透測試
- `PostExResultPayload` - 後滲透測試結果

#### 新增 Schema (已添加)
- ✅ `SensitiveMatch` - 敏感資料匹配
- ✅ `JavaScriptAnalysisResult` - JavaScript 分析結果

### ❌ 缺少的 Schema (需添加)

#### BizLogic 模組 Schema
- `BizLogicTestPayload` - 業務邏輯測試請求
- `BizLogicResultPayload` - 業務邏輯測試結果

---

## 🔧 需要統一的命名規範

### 1. Schema 欄位命名不一致問題

#### SensitiveMatch 欄位
**當前定義** (schemas.py line 535-547):
```python
class SensitiveMatch(BaseModel):
    match_id: str
    pattern_name: str      # ✅ 正確
    matched_text: str      # ✅ 正確
    context: str
    confidence: float
    line_number: int | None = None
    file_path: str | None = None
    url: str | None = None
    severity: Severity
```

**使用情況**:
- ✅ `sensitive_data_scanner.py` - 已修正使用 `pattern_name`, `matched_text`, `url`
- ✅ `scan_context.py` - 已修正使用 `url`

**狀態**: ✅ 統一完成

#### JavaScriptAnalysisResult 欄位
**當前定義** (schemas.py line 549-561):
```python
class JavaScriptAnalysisResult(BaseModel):
    analysis_id: str
    url: str                          # ✅ 使用 url
    source_size_bytes: int
    findings: list[str]
    apis_called: list[str]
    ajax_endpoints: list[str]
    suspicious_patterns: list[str]
    risk_score: float
    timestamp: datetime
```

**使用情況**:
- ❌ `javascript_analyzer.py` - 使用錯誤的欄位名 `file_url`, `size_bytes`, `dangerous_functions`, `external_resources`, `data_leaks`, `security_score`

**需要修正**: 
1. 決定統一欄位名稱
2. 修正 `javascript_analyzer.py` 或更新 Schema

### 2. Vulnerability 類型不一致

**當前架構**:
```python
# Vulnerability 是 BaseModel 而非 Enum
class Vulnerability(BaseModel):
    name: VulnerabilityType    # 引用 VulnerabilityType Enum
    cwe: str | None
    severity: Severity
    confidence: Confidence
```

**VulnerabilityType Enum** (已更新):
```python
class VulnerabilityType(str, Enum):
    XSS = "XSS"
    SQLI = "SQL Injection"
    SSRF = "SSRF"
    IDOR = "IDOR"
    BOLA = "BOLA"
    INFO_LEAK = "Information Leak"
    WEAK_AUTH = "Weak Authentication"
    # ✅ 新增 BizLogic 類型
    PRICE_MANIPULATION = "Price Manipulation"
    WORKFLOW_BYPASS = "Workflow Bypass"
    RACE_CONDITION = "Race Condition"
    FORCED_BROWSING = "Forced Browsing"
    STATE_MANIPULATION = "State Manipulation"
```

**狀態**: ✅ 已統一

### 3. Topic 枚舉統一

**當前定義** (enums.py):
```python
class Topic(str, Enum):
    # 基礎 Topics
    TASK_SCAN_START = "tasks.scan.start"
    TASK_FUNCTION_START = "tasks.function.start"  # ✅ 新增
    RESULTS_SCAN_COMPLETED = "results.scan.completed"
    RESULTS_FUNCTION_COMPLETED = "results.function.completed"  # ✅ 新增
    
    # 智慧模組 Topics (已存在)
    TASK_THREAT_INTEL_LOOKUP = "tasks.threat_intel.lookup"
    RESULTS_THREAT_INTEL = "results.threat_intel"
    TASK_AUTHZ_CHECK = "tasks.authz.check"
    RESULTS_AUTHZ = "results.authz"
    TASK_POSTEX_TEST = "tasks.postex.test"
    RESULTS_POSTEX = "results.postex"
    TASK_REMEDIATION_GENERATE = "tasks.remediation.generate"
    RESULTS_REMEDIATION = "results.remediation"
```

**狀態**: ✅ 已統一

### 4. ModuleName 枚舉統一

**當前定義**:
```python
class ModuleName(str, Enum):
    # 核心模組
    CORE = "CoreModule"
    SCAN = "ScanModule"
    FUNCTION = "FunctionModule"     # ✅ 新增
    INTEGRATION = "IntegrationModule"
    
    # Function 子模組
    FUNC_XSS = "FunctionXSS"
    FUNC_SQLI = "FunctionSQLI"
    FUNC_SSRF = "FunctionSSRF"
    FUNC_IDOR = "FunctionIDOR"
    
    # 智慧模組
    THREAT_INTEL = "ThreatIntelModule"
    AUTHZ = "AuthZModule"
    POSTEX = "PostExModule"
    REMEDIATION = "RemediationModule"
    BIZLOGIC = "BizLogicModule"     # ✅ 新增
```

**狀態**: ✅ 已統一

---

## 📝 需要修正的檔案清單

### 優先級 P0 - 立即修正

#### 1. BizLogic 模組 Finding 創建
**檔案**: 
- `services/bizlogic/price_manipulation_tester.py`
- `services/bizlogic/workflow_bypass_tester.py`
- `services/bizlogic/race_condition_tester.py`

**問題**:
- 直接使用 `FindingPayload()` 但參數不匹配
- 缺少 `task_id`, `scan_id`, `status` 必需參數
- 需要創建 `Vulnerability` 和 `FindingTarget` 對象

**解決方案**: 使用 `finding_helper.py` 輔助函數統一創建

#### 2. JavaScript Analyzer Schema 不匹配
**檔案**: `services/scan/aiva_scan/javascript_analyzer.py`

**問題**:
```python
# 使用的欄位名
result = JavaScriptAnalysisResult(
    file_url=file_url,           # ❌ 應為 url
    size_bytes=len(...),         # ❌ 應為 source_size_bytes
    dangerous_functions=[],       # ❌ Schema 中沒有此欄位
    external_resources=[],        # ❌ Schema 中沒有此欄位
    data_leaks=[],               # ❌ Schema 中沒有此欄位
    security_score=0             # ❌ 應為 risk_score
)
```

**解決方案選項**:
A. 更新 Schema 匹配使用 (推薦)
B. 更新 Analyzer 匹配 Schema

#### 3. Worker Refactored 構造函數不匹配
**檔案**: `services/scan/aiva_scan/worker_refactored.py`

**問題**:
```python
orchestrator = ScanOrchestrator(req)  # ❌ 構造函數不接受參數
scan_context = await orchestrator.execute_scan()  # ❌ 缺少 request 參數
```

**需要確認**: `ScanOrchestrator` 的正確使用方式

### 優先級 P1 - 重要修正

#### 4. 添加 BizLogic Schema
**檔案**: `services/aiva_common/schemas.py`

**需要添加**:
```python
class BizLogicTestPayload(BaseModel):
    """業務邏輯測試 Payload"""
    task_id: str
    scan_id: str
    test_type: str  # price_manipulation, workflow_bypass, race_condition
    target_urls: dict[str, str]  # 目標 URL 字典
    test_config: dict[str, Any] = Field(default_factory=dict)
    product_id: str | None = None
    workflow_steps: list[dict[str, str]] = Field(default_factory=list)

class BizLogicResultPayload(BaseModel):
    """業務邏輯測試結果 Payload"""
    task_id: str
    scan_id: str
    test_type: str
    status: str
    findings: list[FindingPayload]
    statistics: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

#### 5. 修正方法簽名缺少 task_id/scan_id
**檔案**: 所有 BizLogic 測試器

**需要修正的方法**:
- `test_negative_quantity()` ✅ 已修正
- `test_race_condition_pricing()` ❌ 需修正
- `test_coupon_reuse()` ❌ 需修正
- `test_price_tampering()` ❌ 需修正
- `test_step_skipping()` ❌ 需修正
- `test_forced_browsing()` ❌ 需修正
- `test_state_manipulation()` ❌ 需修正
- `test_inventory_race()` ❌ 需修正
- `test_balance_race()` ❌ 需修正

---

## 🎯 統一修正方案

### 方案 A: JavaScriptAnalysisResult Schema 擴展 (推薦)

**理由**: 
- `dangerous_functions`, `external_resources`, `data_leaks` 等資訊對安全分析很重要
- 當前 Schema 的 `findings` 是通用 `list[str]`,不夠結構化
- 保持 Analyzer 的豐富功能

**修正 Schema**:
```python
class JavaScriptAnalysisResult(BaseModel):
    """JavaScript 分析結果"""
    analysis_id: str
    url: str                                    # 統一使用 url
    source_size_bytes: int                      # 統一使用 source_size_bytes
    
    # 詳細分析結果
    dangerous_functions: list[str] = Field(default_factory=list)
    external_resources: list[str] = Field(default_factory=list)
    data_leaks: list[dict[str, str]] = Field(default_factory=list)
    
    # 通用欄位 (保持兼容)
    findings: list[str] = Field(default_factory=list)
    apis_called: list[str] = Field(default_factory=list)
    ajax_endpoints: list[str] = Field(default_factory=list)
    suspicious_patterns: list[str] = Field(default_factory=list)
    
    # 統一評分欄位
    risk_score: float = Field(ge=0.0, le=10.0, default=0.0)
    security_score: int = Field(ge=0, le=100, default=100)  # 新增,0-100分
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

### 方案 B: 最小變動方案

僅修正 `javascript_analyzer.py` 使用正確的欄位名:
- `file_url` → `url`
- `size_bytes` → `source_size_bytes`
- 將 `dangerous_functions` 等合併到 `findings`

---

## ✅ 執行順序

### Phase 1: Schema 統一 (30分鐘)

1. **決定 JavaScriptAnalysisResult 方案** - 推薦方案 A
2. **更新 schemas.py**:
   - 擴展 `JavaScriptAnalysisResult` 
   - 添加 `BizLogicTestPayload` 和 `BizLogicResultPayload`
3. **驗證 Schema 無錯誤**

### Phase 2: BizLogic 模組修正 (1小時)

1. **確保 finding_helper.py 正確**
2. **修正所有測試器方法簽名** - 添加 `task_id` 和 `scan_id` 參數
3. **替換所有 FindingPayload 創建** - 使用 `create_bizlogic_finding()`
4. **修正 worker.py** - 處理 `task_id` 和 `scan_id` 傳遞

### Phase 3: JavaScript Analyzer 修正 (30分鐘)

1. **更新 javascript_analyzer.py** - 使用正確的 Schema 欄位
2. **更新 scan_context.py** - 使用 `result.url`
3. **測試完整流程**

### Phase 4: 驗證測試 (30分鐘)

1. **運行靜態檢查**: `mypy services/`
2. **運行 Linter**: `ruff check services/`
3. **修正剩餘小問題**

---

## 📋 修正檢查清單

- [ ] Schema 更新完成
  - [ ] `JavaScriptAnalysisResult` 擴展
  - [ ] `BizLogicTestPayload` 添加
  - [ ] `BizLogicResultPayload` 添加
  
- [ ] BizLogic 模組修正
  - [x] `finding_helper.py` 創建
  - [x] `test_negative_quantity()` 簽名修正
  - [ ] 其他 8 個測試方法簽名修正
  - [ ] 所有 FindingPayload 創建替換
  - [ ] worker.py 參數傳遞
  
- [ ] JavaScript Analyzer 修正
  - [ ] Schema 欄位名統一
  - [ ] scan_context 調用修正
  
- [ ] 全面驗證
  - [ ] 無類型錯誤
  - [ ] 無導入錯誤
  - [ ] 格式統一

---

## 🎨 命名規範總結

### Schema 命名
- **Payload 後綴**: 用於消息隊列傳遞的數據結構
- **Result 後綴**: 用於測試/分析結果
- **Match 後綴**: 用於匹配/檢測結果

### 欄位命名
- **URL 欄位**: 統一使用 `url` (不使用 `file_url`, `target_url`, `affected_url`)
- **ID 欄位**: 使用 `_id` 後綴 (如 `task_id`, `scan_id`, `finding_id`)
- **大小欄位**: 使用 `_bytes` 後綴 (如 `source_size_bytes`)
- **評分欄位**: 
  - `risk_score`: 0.0-10.0 浮點數
  - `security_score`: 0-100 整數
  - `confidence`: 使用 Confidence Enum

### Topic 命名
- **任務**: `TASK_{MODULE}_{ACTION}` (如 `TASK_THREAT_INTEL_LOOKUP`)
- **結果**: `RESULTS_{MODULE}` (如 `RESULTS_THREAT_INTEL`)

### Module 命名
- **主模組**: `{Name}Module` (如 `CoreModule`, `ScanModule`)
- **Function 子模組**: `Function{Type}` (如 `FunctionXSS`, `FunctionSQLI`)
- **智慧模組**: `{Name}Module` (如 `ThreatIntelModule`, `BizLogicModule`)

---

## 📊 當前進度

- ✅ Enums 統一完成
- ✅ Topic 擴展完成
- ✅ ModuleName 擴展完成
- ✅ VulnerabilityType 擴展完成
- ✅ SensitiveMatch Schema 統一
- ⏳ JavaScriptAnalysisResult 待決定方案
- ⏳ BizLogic Schema 待添加
- ⏳ BizLogic 測試器待修正
- ⏳ JavaScript Analyzer 待修正

---

**下一步**: 選擇 JavaScriptAnalysisResult 方案並開始 Phase 1
