# AIVA Schema 架構分析與統一規範

## 四大模組 Schema 現狀分析

### 1. `aiva_common/schemas.py` (公共模組)

**定位**: 跨模組共享的數據合約
**內容結構**:

- 訊息系統: `MessageHeader`, `AivaMessage`
- 掃描核心: `ScanStartPayload`, `ScanCompletedPayload`
- 功能任務: `FunctionTaskPayload`, `FunctionTaskTarget`
- 發現數據: `FindingPayload`, `FindingTarget`
- 情報系統: `IOCPayload`, `ThreatIntelPayload`
- 修復管理: `RemediationGeneratePayload`
- 業務邏輯: `BizLogicResultPayload`
- OAST 系統: `OastEvent`

**命名特徵**: 多數使用 `Payload` 後綴，遵循消息傳遞模式

### 2. `core/aiva_core/schemas.py` (核心模組)

**定位**: AI 引擎和策略分析的數據定義
**內容結構**:

- 資產分析: `AssetAnalysis`, `VulnerabilityCandidate`
- 測試任務: `TestTask`, `XssCandidate`, `SsrfCandidate`
- 學習系統: `LearningFeedback`

**命名特徵**: 多數不使用後綴，直接描述功能

### 3. `core/aiva_core/ai_ui_schemas.py` (AI/UI 介面)

**定位**: AI 代理和 UI 面板的專用數據格式
**內容結構**:

- AI 系統: `AIAgentQuery`, `AIAgentResponse`
- 工具執行: `ToolExecutionRequest`, `ToolExecutionResult`
- RAG 檢索: `RAGChunk`
- 工具結果: `CodeReadResult`, `CommandExecutionResult`

**命名特徵**: 使用 `Request`/`Response`/`Result` 後綴

### 4. `scan/aiva_scan/schemas.py` (掃描模組)

**定位**: 掃描引擎專用數據結構
**內容結構**:

- 敏感資訊: `SensitiveMatch`
- JavaScript 分析: `JavaScriptAnalysisResult`
- 網路請求: `NetworkRequest`
- 互動結果: `InteractionResult`

**命名特徵**: 使用 `Result`/`Match`/`Request` 後綴

### 5. Function 模組 Schemas

#### `function_xss/schemas.py`

- `XssDetectionResult`, `XssTelemetry`, `DomDetectionResult`

#### `function_sqli/schemas.py`

- `SqlInjectionTestVector`, `DetectionError`, `SqliTelemetry`

#### `function_ssrf/schemas.py`

- `SsrfTestVector`, `TaskExecutionResult`, `InternalAddressDetectionResult`

**命名特徵**: 功能型模組多使用 `Result`/`Vector`/`Telemetry` 後綴

### 6. TypeScript 介面 (aiva_scan_node)

**位置**: `aiva_scan_node/src/interfaces/dynamic-scan.interfaces.ts`
**內容結構**:

- `DynamicScanTask`, `DynamicScanResult`
- `ExtractionConfig`, `InteractionConfig`
- `DynamicContent`, `NetworkRequest`

## 問題識別

### 1. 命名不一致問題

- **Payload vs Result**: `aiva_common` 偏好 `Payload`，其他模組偏好 `Result`
- **缺少統一前綴**: 沒有模組識別前綴
- **TypeScript 與 Python 不對應**: TS 使用 PascalCase，Python 也使用 PascalCase 但缺乏一致性

### 2. 結構重複問題

- **敏感資訊檢測**: `aiva_common.SensitiveMatch` vs `aiva_scan.SensitiveMatch`
- **網路請求**: TypeScript `NetworkRequest` vs Python `NetworkRequest`
- **互動結果**: TypeScript vs Python 版本格式不同

### 3. 功能缺失問題

- **IDOR 模組**: 缺少專用的 `schemas.py` 文件
- **PostEx 模組**: 缺少專用的數據合約定義
- **Dynamic Scan**: TypeScript 與 Python 定義不完全對應

## 統一規範建議

### 1. 命名規範

```text
[ModulePrefix][FunctionName][DataType]

例如:
- CoreAssetAnalysis (Core 模組的資產分析)
- ScanNetworkRequest (Scan 模組的網路請求)
- FuncXssDetectionResult (Function XSS 模組的檢測結果)
- CommonFindingPayload (Common 模組的發現數據)
```

### 2. 數據類型後綴標準

- `Payload`: 消息隊列傳輸的數據包
- `Request`: API 請求數據
- `Response`: API 響應數據
- `Result`: 處理/分析結果
- `Config`: 配置數據
- `Task`: 任務定義
- `Event`: 事件數據
- `Telemetry`: 遙測數據

### 3. 字段命名標準

- 使用 snake_case (Python) 和 camelCase (TypeScript)
- 時間字段統一使用 `*_at` 後綴 (如 `created_at`, `updated_at`)
- ID 字段統一使用 `*_id` 後綴 (如 `task_id`, `scan_id`)
- 布林字段使用 `is_*` 或 `has_*` 前綴

### 4. 必須增加的 Schemas

#### IDOR 模組

```python
# services/function/function_idor/aiva_func_idor/schemas.py
class IdorTestVector(BaseModel)
class IdorDetectionResult(BaseModel)
class IdorTelemetry(FunctionTelemetry)
```

#### PostEx 模組

```python
# services/function/function_postex/schemas.py
class PostExTestVector(BaseModel)
class PostExDetectionResult(BaseModel)
class PostExTelemetry(FunctionTelemetry)
```

#### Dynamic Scan 統一接口

```python
# services/scan/aiva_scan/schemas.py
class DynamicScanTask(BaseModel)  # 對應 TypeScript 版本
class DynamicScanResult(BaseModel)  # 對應 TypeScript 版本
```
