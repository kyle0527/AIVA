# 需要新增的檔案完成清單

## 已完成的檔案創建

### 1. BizLogic 模組 ✅

- [x] `services/bizlogic/__init__.py`
- [x] `services/bizlogic/price_manipulation_tester.py`
- [x] `services/bizlogic/workflow_bypass_tester.py`
- [x] `services/bizlogic/race_condition_tester.py`
- [x] `services/bizlogic/worker.py`

### 2. Core 模組擴展 ✅

- [x] `services/core/aiva_core/analysis/risk_assessment_engine.py`

### 3. Scan 模組擴展 ✅

- [x] `services/scan/aiva_scan/scan_context.py` - 已添加 `add_sensitive_match()` 和 `add_js_analysis_result()` 方法
- [x] `services/scan/aiva_scan/worker_refactored.py` - 使用 ScanOrchestrator 的新 Worker
- [x] `services/scan/aiva_scan/sensitive_data_scanner.py` - 敏感資料掃描器
- [x] `services/scan/aiva_scan/javascript_analyzer.py` - JavaScript 安全分析器

### 4. Dynamic Engine 擴展 ✅

- [x] `services/scan/aiva_scan/dynamic_engine/ajax_api_handler.py` - AJAX/API 處理器
- [x] `services/scan/aiva_scan/dynamic_engine/__init__.py` - 已更新導出

### 5. 通用模組擴展 ✅

- [x] `services/aiva_common/schemas.py` - 已添加 `SensitiveMatch` 和 `JavaScriptAnalysisResult`

### 6. 文檔 ✅

- [x] `INTEGRATION_GUIDE.md` - 完整的整合指南

---

## 已知問題總結（需要後續修復）

### Type Errors

#### 1. ScanContext 導入問題

**文件**: `services/scan/aiva_scan/scan_context.py`

**問題**:

- `SensitiveMatch` 未定義
- `JavaScriptAnalysisResult` 未定義

**修復方法**:

```python
from services.aiva_common.schemas import SensitiveMatch, JavaScriptAnalysisResult
```

#### 2. Worker Refactored 問題

**文件**: `services/scan/aiva_scan/worker_refactored.py`

**問題**:

- `ScanOrchestrator(req)` - 預期有 0 個位置引數
- `execute_scan()` - 參數 "request" 遺漏引數

**修復方法**:

需要檢查 `ScanOrchestrator` 的實際構造函數簽名並調整

#### 3. JavaScriptAnalysisResult Schema 不匹配

**文件**: `services/scan/aiva_scan/javascript_analyzer.py`

**問題**:

- 構造參數名稱不匹配 (`file_url` vs `url`, `size_bytes` vs `source_size_bytes`)
- 缺少必需參數 (`analysis_id`, `url`, `source_size_bytes`)
- 屬性訪問失敗 (所有屬性都不存在)

**修復方法**:

需要更新 `services/aiva_common/schemas.py` 中的 `JavaScriptAnalysisResult` 定義以匹配使用

#### 4. BizLogic 模組的 Finding 創建問題

**文件**:

- `services/bizlogic/price_manipulation_tester.py`
- `services/bizlogic/workflow_bypass_tester.py`
- `services/bizlogic/race_condition_tester.py`

**問題**:

- `FindingPayload` 不接受 `title`, `description` 等參數
- `Vulnerability` 枚舉缺少 BizLogic 相關類型
- `FindingTarget` 類型不匹配

**修復方法**:

1. 更新 `FindingPayload` schema 接受這些參數
2. 在 `Vulnerability` 枚舉中添加:
   - `PRICE_MANIPULATION`
   - `WORKFLOW_BYPASS`
   - `RACE_CONDITION`

#### 5. RiskAssessmentEngine 問題

**文件**: `services/core/aiva_core/analysis/risk_assessment_engine.py`

**問題**:

- `FindingPayload` 缺少 `cve_id`, `severity`, `metadata` 等屬性

**修復方法**:

需要擴展 `FindingPayload` schema

#### 6. SensitiveDataScanner 問題

**文件**: `services/scan/aiva_scan/sensitive_data_scanner.py`

**問題**:

- 可能存在 `SensitiveMatch` 參數不匹配（需要確認 schema 定義）

---

## 下一步工作

### Phase 1: Schema 修復（優先）

1. 更新 `services/aiva_common/schemas.py`:
   - 擴展 `FindingPayload` 接受新參數
   - 修正 `JavaScriptAnalysisResult` 定義
   - 確認 `SensitiveMatch` 參數正確

2. 更新 `services/aiva_common/enums.py`:
   - 添加 `Topic.TASK_BIZLOGIC`, `Topic.RESULTS_BIZLOGIC`
   - 添加 `Vulnerability.PRICE_MANIPULATION` 等

### Phase 2: 導入修復

1. 修復 `scan_context.py` 的導入
2. 檢查並修復所有模組的導入路徑

### Phase 3: 構造函數和方法簽名修復

1. 確認 `ScanOrchestrator` 構造函數
2. 修復 `worker_refactored.py` 的調用
3. 修復所有 `FindingPayload` 創建代碼

### Phase 4: 集成測試

1. 運行靜態類型檢查: `mypy services/`
2. 運行 linter: `ruff check services/`
3. 修復剩餘的小問題

### Phase 5: 功能測試

1. 單元測試每個新模組
2. 集成測試完整流程
3. 端到端測試真實場景

---

## 文件創建完成確認

所有計劃的新文件都已創建完成 ✅

現在可以開始修復階段，按照上述 Phase 1-5 的順序進行。
