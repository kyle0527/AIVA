# AIVA 數據合約完整文檔

**Data Contract Documentation**

版本：1.0.0  
最後更新：2025-10-13  
維護者：AIVA Development Team

---

## 📑 **目錄**

1. [概述](#概述)
2. [核心通用合約](#核心通用合約)
3. [功能模組專用合約](#功能模組專用合約)
4. [使用指南與最佳實踐](#使用指南與最佳實踐)
5. [版本控制與變更歷史](#版本控制與變更歷史)

---

## 🎯 **概述**

### 設計原則

AIVA 系統的所有數據合約遵循以下設計原則：

1. **統一性**：所有數據模型使用 Pydantic v2.12.0 BaseModel
2. **驗證性**：關鍵字段包含完整的 `field_validator`
3. **模組化**：通用合約在 `aiva_common`，專用合約在各模組
4. **可擴展性**：支持繼承和組合
5. **向後兼容**：提供別名以保持兼容性

### 架構層次

```
services/aiva_common/
├── schemas.py      # 通用數據合約（18個核心類 + 8個新增基礎類）
└── enums.py        # 枚舉類型（7個枚舉）

services/function_*/*/
└── schemas.py      # 各功能模組專用數據合約
    ├── function_sqli/aiva_func_sqli/schemas.py   # SQLi專用
    ├── function_xss/aiva_func_xss/schemas.py     # XSS專用
    ├── function_ssrf/aiva_func_ssrf/schemas.py   # SSRF專用
    └── function_idor/                            # IDOR已直接使用Pydantic

services/scan/aiva_scan/
└── schemas.py      # 掃描模組專用數據合約
```

---

## 📦 **核心通用合約**

### 1. 訊息協議

#### MessageHeader

**用途**：所有 MQ 訊息的標準標頭

**字段**：

| 字段 | 類型 | 必填 | 說明 | 驗證規則 |
|------|------|------|------|---------|
| `message_id` | `str` | ✅ | 訊息唯一標識符 | - |
| `trace_id` | `str` | ✅ | 追蹤 ID（用於分散式追蹤） | - |
| `correlation_id` | `str \| None` | ❌ | 關聯 ID（用於請求-響應模式） | - |
| `source_module` | `ModuleName` | ✅ | 來源模組 | 必須是有效的 ModuleName |
| `timestamp` | `datetime` | ✅ | 時間戳（UTC） | 自動生成 |
| `version` | `str` | ✅ | 協議版本 | 默認 "1.0" |

**示例**：

```python
from services.aiva_common.schemas import MessageHeader
from services.aiva_common.enums import ModuleName
from services.aiva_common.utils import new_id

header = MessageHeader(
    message_id=new_id("msg"),
    trace_id=new_id("trace"),
    correlation_id="scan_abc123",
    source_module=ModuleName.CORE,
)
```

#### AivaMessage

**用途**：統一的訊息封裝格式

**字段**：

| 字段 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `header` | `MessageHeader` | ✅ | 訊息標頭 |
| `topic` | `Topic` | ✅ | MQ 主題 |
| `schema_version` | `str` | ✅ | 數據結構版本 |
| `payload` | `dict[str, Any]` | ✅ | 業務數據載荷 |

**示例**：

```python
from services.aiva_common.schemas import AivaMessage, ScanStartPayload
from services.aiva_common.enums import Topic

scan_payload = ScanStartPayload(
    scan_id="scan_abc123",
    targets=["<https://example.com"],>
)

message = AivaMessage(
    header=header,
    topic=Topic.TASK_SCAN_START,
    payload=scan_payload.model_dump(),
)
```

---

### 2. 掃描相關合約

#### ScanStartPayload

**用途**：啟動新掃描任務

**字段**：

| 字段 | 類型 | 必填 | 默認值 | 驗證規則 |
|------|------|------|--------|---------|
| `scan_id` | `str` | ✅ | - | 必須以 `scan_` 開頭，長度≥10 |
| `targets` | `list[HttpUrl]` | ✅ | - | 1-100個有效 URL |
| `scope` | `ScanScope` | ❌ | `ScanScope()` | - |
| `authentication` | `Authentication` | ❌ | `Authentication()` | - |
| `strategy` | `str` | ❌ | `"deep"` | quick/normal/deep/full/custom |
| `rate_limit` | `RateLimit` | ❌ | `RateLimit()` | - |
| `custom_headers` | `dict[str, str]` | ❌ | `{}` | - |
| `x_forwarded_for` | `str \| None` | ❌ | `None` | - |

**驗證規則**：

- `scan_id`：必須以 "scan_" 開頭，長度不少於10字符
- `targets`：至少1個，最多100個有效的 HTTP/HTTPS URL
- `strategy`：必須是 "quick", "normal", "deep", "full", "custom" 之一

**示例**：

```python
from services.aiva_common.schemas import ScanStartPayload

payload = ScanStartPayload(
    scan_id="scan_abc123xyz",
    targets=[
        "<https://example.com",>
        "<https://api.example.com",>
    ],
    strategy="deep",
)
```

#### ScanCompletedPayload

**用途**：掃描完成報告

**字段**：

| 字段 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `scan_id` | `str` | ✅ | 掃描 ID |
| `status` | `str` | ✅ | 完成狀態 |
| `summary` | `Summary` | ✅ | 掃描摘要統計 |
| `assets` | `list[Asset]` | ❌ | 發現的資產 |
| `fingerprints` | `Fingerprints \| None` | ❌ | 指紋信息 |
| `error_info` | `str \| None` | ❌ | 錯誤信息 |

---

### 3. 功能任務合約

#### FunctionTaskPayload

**用途**：發送給功能模組（XSS, SQLi, SSRF, IDOR）的測試任務

**字段**：

| 字段 | 類型 | 必填 | 默認值 | 驗證規則 |
|------|------|------|--------|---------|
| `task_id` | `str` | ✅ | - | 必須以 `task_` 開頭 |
| `scan_id` | `str` | ✅ | - | 必須以 `scan_` 開頭 |
| `priority` | `int` | ❌ | `5` | 1-10之間 |
| `target` | `FunctionTaskTarget` | ✅ | - | - |
| `context` | `FunctionTaskContext` | ❌ | `FunctionTaskContext()` | - |
| `strategy` | `str` | ❌ | `"full"` | - |
| `custom_payloads` | `list[str] \| None` | ❌ | `None` | - |
| `test_config` | `FunctionTaskTestConfig` | ❌ | `FunctionTaskTestConfig()` | - |

**驗證規則**：

- `task_id`：必須以 "task_" 開頭
- `scan_id`：必須以 "scan_" 開頭
- `priority`：1-10之間的整數

**示例**：

```python
from services.aiva_common.schemas import (
    FunctionTaskPayload,
    FunctionTaskTarget,
)

task = FunctionTaskPayload(
    task_id="task_xss_001",
    scan_id="scan_abc123",
    priority=8,
    target=FunctionTaskTarget(
        url="<https://example.com/search",>
        parameter="q",
        method="GET",
        parameter_location="query",
    ),
)
```

#### FunctionTaskTarget

**用途**：定義測試目標的詳細信息

**字段**：

| 字段 | 類型 | 必填 | 默認值 |
|------|------|------|--------|
| `url` | `Any` | ✅ | - |
| `parameter` | `str \| None` | ❌ | `None` |
| `method` | `str` | ❌ | `"GET"` |
| `parameter_location` | `str` | ❌ | `"query"` |
| `headers` | `dict[str, str]` | ❌ | `{}` |
| `cookies` | `dict[str, str]` | ❌ | `{}` |
| `form_data` | `dict[str, Any]` | ❌ | `{}` |
| `json_data` | `dict[str, Any] \| None` | ❌ | `None` |
| `body` | `str \| None` | ❌ | `None` |

---

### 4. 漏洞報告合約

#### FindingPayload

**用途**：報告發現的漏洞

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `finding_id` | `str` | ✅ | 必須以 `finding_` 開頭 |
| `task_id` | `str` | ✅ | 必須以 `task_` 開頭 |
| `scan_id` | `str` | ✅ | 必須以 `scan_` 開頭 |
| `status` | `str` | ✅ | confirmed/potential/false_positive/needs_review |
| `vulnerability` | `Vulnerability` | ✅ | - |
| `target` | `FindingTarget` | ✅ | - |
| `strategy` | `str \| None` | ❌ | - |
| `evidence` | `FindingEvidence \| None` | ❌ | - |
| `impact` | `FindingImpact \| None` | ❌ | - |
| `recommendation` | `FindingRecommendation \| None` | ❌ | - |

**驗證規則**：

- 所有 ID 字段必須符合命名約定（前綴）
- `status` 必須是預定義值之一

**示例**：

```python
from services.aiva_common.schemas import (
    FindingPayload,
    Vulnerability,
    FindingTarget,
    FindingEvidence,
)
from services.aiva_common.enums import VulnerabilityType, Severity, Confidence

finding = FindingPayload(
    finding_id="finding_xss_001",
    task_id="task_xss_001",
    scan_id="scan_abc123",
    status="confirmed",
    vulnerability=Vulnerability(
        name=VulnerabilityType.XSS,
        cwe="CWE-79",
        severity=Severity.HIGH,
        confidence=Confidence.CERTAIN,
    ),
    target=FindingTarget(
        url="<https://example.com/search?q=test",>
        parameter="q",
        method="GET",
    ),
    evidence=FindingEvidence(
        payload="<script>alert(1)</script>",
        proof="Script reflected in response without encoding",
    ),
)
```

---

### 5. 新增通用基礎類

#### FunctionTelemetry

**用途**：功能模組遙測數據基礎類（可被繼承）

**字段**：

| 字段 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `payloads_sent` | `int` | `0` | 發送的 payload 數量 |
| `detections` | `int` | `0` | 檢測到的漏洞數量 |
| `attempts` | `int` | `0` | 嘗試次數 |
| `errors` | `list[str]` | `[]` | 錯誤列表 |
| `duration_seconds` | `float` | `0.0` | 執行時間（秒） |
| `timestamp` | `datetime` | 當前時間 | 時間戳 |

**方法**：

- `to_details(findings_count: int | None = None) -> dict[str, Any]`：轉換為詳細報告格式

**使用示例**：

```python
from services.aiva_common.schemas import FunctionTelemetry

telemetry = FunctionTelemetry()
telemetry.payloads_sent = 10
telemetry.detections = 2
telemetry.errors.append("Timeout on payload #5")

report = telemetry.to_details(findings_count=2)
# {'payloads_sent': 10, 'detections': 2, 'attempts': 0, 
#  'duration_seconds': 0.0, 'findings': 2, 'errors': [...]}
```

#### ExecutionError

**用途**：統一的執行錯誤記錄格式

**字段**：

| 字段 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `error_id` | `str` | ✅ | 錯誤唯一標識符 |
| `error_type` | `str` | ✅ | 錯誤類型 |
| `message` | `str` | ✅ | 錯誤消息 |
| `payload` | `str \| None` | ❌ | 相關的 payload |
| `vector` | `str \| None` | ❌ | 測試向量 |
| `timestamp` | `datetime` | ✅ | 時間戳 |
| `attempts` | `int` | ✅ | 嘗試次數 |

#### OastEvent

**用途**：OAST（Out-of-Band）事件記錄

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `event_id` | `str` | ✅ | - |
| `probe_token` | `str` | ✅ | - |
| `event_type` | `str` | ✅ | http/dns/smtp/ftp/ldap/other |
| `source_ip` | `str` | ✅ | - |
| `timestamp` | `datetime` | ✅ | 自動生成 |
| `protocol` | `str \| None` | ❌ | - |
| `raw_request` | `str \| None` | ❌ | - |
| `raw_data` | `dict[str, Any]` | ❌ | `{}` |

**驗證規則**：

- `event_type` 必須是預定義的事件類型之一

#### OastProbe

**用途**：OAST 探針配置

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `probe_id` | `str` | ✅ | - |
| `token` | `str` | ✅ | - |
| `callback_url` | `HttpUrl` | ✅ | 有效的 HTTP/HTTPS URL |
| `task_id` | `str` | ✅ | - |
| `scan_id` | `str` | ✅ | - |
| `created_at` | `datetime` | ✅ | 自動生成 |
| `expires_at` | `datetime \| None` | ❌ | - |
| `status` | `str` | ✅ | active/triggered/expired/cancelled |

#### ModuleStatus

**用途**：模組健康狀態報告

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `module` | `ModuleName` | ✅ | - |
| `status` | `str` | ✅ | running/stopped/error/initializing/degraded |
| `worker_id` | `str` | ✅ | - |
| `worker_count` | `int` | ❌ | 默認 1 |
| `queue_size` | `int` | ❌ | 默認 0 |
| `tasks_completed` | `int` | ❌ | 默認 0 |
| `tasks_failed` | `int` | ❌ | 默認 0 |
| `last_heartbeat` | `datetime` | ✅ | 自動生成 |
| `metrics` | `dict[str, Any]` | ❌ | `{}` |
| `uptime_seconds` | `float` | ❌ | 默認 0.0 |

---

## 🔧 **功能模組專用合約**

### SQLi 模組（services/function_sqli/aiva_func_sqli/schemas.py）

#### SqliDetectionResult

**用途**：SQLi 檢測結果

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `is_vulnerable` | `bool` | ✅ | - |
| `vulnerability` | `Vulnerability` | ✅ | - |
| `evidence` | `FindingEvidence` | ✅ | - |
| `impact` | `FindingImpact` | ✅ | - |
| `recommendation` | `FindingRecommendation` | ✅ | - |
| `target` | `FindingTarget` | ✅ | - |
| `detection_method` | `str` | ✅ | error/boolean/time/union/oob/stacked |
| `payload_used` | `str` | ✅ | - |
| `confidence_score` | `float` | ✅ | 0.0 - 1.0 |
| `db_fingerprint` | `str \| None` | ❌ | - |
| `response_time` | `float` | ❌ | 默認 0.0 |

**驗證規則**：

- `detection_method` 必須是有效的檢測方法
- `confidence_score` 在 0.0 到 1.0 之間

**示例**：

```python
from services.function_sqli.aiva_func_sqli.schemas import SqliDetectionResult
from services.aiva_common.schemas import Vulnerability, FindingEvidence
from services.aiva_common.enums import VulnerabilityType, Severity, Confidence

result = SqliDetectionResult(
    is_vulnerable=True,
    vulnerability=Vulnerability(
        name=VulnerabilityType.SQLI,
        cwe="CWE-89",
        severity=Severity.HIGH,
        confidence=Confidence.CERTAIN,
    ),
    evidence=FindingEvidence(
        payload="' OR '1'='1",
        proof="Database error message revealed",
    ),
    detection_method="error",
    payload_used="' OR '1'='1",
    confidence_score=0.95,
    db_fingerprint="MySQL 5.7",
)
```

#### SqliTelemetry

**用途**：SQLi 專用遙測（繼承自 FunctionTelemetry）

**新增字段**：

| 字段 | 類型 | 默認值 |
|------|------|--------|
| `engines_run` | `list[str]` | `[]` |
| `blind_detections` | `int` | `0` |
| `error_based_detections` | `int` | `0` |
| `union_based_detections` | `int` | `0` |
| `time_based_detections` | `int` | `0` |
| `oob_detections` | `int` | `0` |

**方法**：

- `record_engine_execution(engine_name: str)`
- `record_payload_sent()`
- `record_detection(method: str = "generic")`
- `record_error(error_message: str)`

#### SqliEngineConfig

**用途**：SQLi 檢測引擎配置

**字段**：

| 字段 | 類型 | 默認值 | 驗證規則 |
|------|------|--------|---------|
| `timeout_seconds` | `float` | `20.0` | 0 < x ≤ 300 |
| `max_retries` | `int` | `3` | 1 ≤ x ≤ 10 |
| `enable_error_detection` | `bool` | `True` | - |
| `enable_boolean_detection` | `bool` | `True` | - |
| `enable_time_detection` | `bool` | `True` | - |
| `enable_union_detection` | `bool` | `True` | - |
| `enable_oob_detection` | `bool` | `True` | - |
| `time_threshold_seconds` | `float` | `5.0` | 0 < x ≤ 30 |

**驗證規則**：

- `timeout_seconds` 必須 ≥ `time_threshold_seconds`

---

### XSS 模組（services/function_xss/aiva_func_xss/schemas.py）

#### XssDetectionResult

**用途**：XSS 檢測結果

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `payload` | `str` | ✅ | - |
| `request_url` | `str` | ✅ | - |
| `request_method` | `str` | ❌ | GET/POST/PUT/DELETE/等 |
| `response_status` | `int` | ✅ | 100-599 |
| `response_headers` | `dict[str, str]` | ❌ | `{}` |
| `response_text` | `str` | ❌ | `""` |
| `reflection_found` | `bool` | ❌ | `False` |
| `context` | `str \| None` | ❌ | - |
| `sink_type` | `str \| None` | ❌ | - |

#### XssTelemetry

**用途**：XSS 專用遙測（繼承自 FunctionTelemetry）

**新增字段**：

| 字段 | 類型 | 默認值 |
|------|------|--------|
| `reflections` | `int` | `0` |
| `dom_escalations` | `int` | `0` |
| `blind_callbacks` | `int` | `0` |
| `stored_xss_found` | `int` | `0` |
| `contexts_tested` | `list[str]` | `[]` |

**方法**：

- `record_reflection()`
- `record_dom_escalation()`
- `record_blind_callback()`
- `record_stored_xss()`
- `record_context(context: str)`

#### DomDetectionResult

**用途**：DOM XSS 檢測結果

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `vulnerable` | `bool` | ✅ | - |
| `sink_type` | `str` | ✅ | innerHTML/eval/document.write/等 |
| `source_type` | `str` | ✅ | location.hash/location.search/等 |
| `payload` | `str` | ✅ | - |
| `evidence` | `str` | ✅ | - |
| `confidence` | `float` | ❌ | 0.0 - 1.0 |

---

### SSRF 模組（services/function_ssrf/aiva_func_ssrf/schemas.py）

#### SsrfTestVector

**用途**：SSRF 測試向量

**字段**：

| 字段 | 類型 | 默認值 | 驗證規則 |
|------|------|--------|---------|
| `payload` | `str` | - | - |
| `vector_type` | `str` | - | internal/cloud_metadata/oast/cross_protocol/dns |
| `priority` | `int` | `5` | 1-10 |
| `requires_oast` | `bool` | `False` | - |
| `protocol` | `str` | `"http"` | http/https/ftp/gopher/file/等 |

#### AnalysisPlan

**用途**：SSRF 分析計劃

**字段**：

| 字段 | 類型 | 必填 | 驗證規則 |
|------|------|------|---------|
| `vectors` | `list[SsrfTestVector]` | ✅ | 1-1000 個向量 |
| `param_name` | `str \| None` | ❌ | - |
| `semantic_hints` | `list[str]` | ❌ | `[]` |
| `requires_oast` | `bool` | ❌ | `False` |
| `estimated_tests` | `int` | ❌ | 自動計算 |

#### SsrfTelemetry

**用途**：SSRF 專用遙測（繼承自 FunctionTelemetry）

**新增字段**：

| 字段 | 類型 | 默認值 |
|------|------|--------|
| `oast_callbacks` | `int` | `0` |
| `internal_access` | `int` | `0` |
| `cloud_metadata_access` | `int` | `0` |
| `dns_lookups` | `int` | `0` |
| `protocols_tested` | `list[str]` | `[]` |

---

### Scan 模組（services/scan/aiva_scan/schemas.py）

#### SensitiveMatch

**用途**：敏感信息匹配結果

**字段**：

| 字段 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `info_type` | `SensitiveInfoType` | ✅ | 敏感信息類型 |
| `value` | `str` | ✅ | 匹配的值 |
| `location` | `Location` | ✅ | 位置 |
| `context` | `str` | ✅ | 上下文 |
| `line_number` | `int \| None` | ❌ | 行號 |
| `severity` | `Severity` | ❌ | 嚴重程度 |
| `description` | `str` | ❌ | 描述（自動生成） |
| `recommendation` | `str` | ❌ | 建議（自動生成） |

**自動功能**：

- 如果 `description` 和 `recommendation` 為空，會根據 `info_type` 自動生成

#### JavaScriptAnalysisResult

**用途**：JavaScript 代碼分析結果

**字段**：

| 字段 | 類型 | 默認值 | 驗證規則 |
|------|------|--------|---------|
| `url` | `str` | - | - |
| `has_sensitive_data` | `bool` | `False` | - |
| `api_endpoints` | `list[str]` | `[]` | 最多 1000 項 |
| `dom_sinks` | `list[str]` | `[]` | 最多 1000 項 |
| `sensitive_functions` | `list[str]` | `[]` | - |
| `external_requests` | `list[str]` | `[]` | 最多 1000 項 |
| `cookies_accessed` | `list[str]` | `[]` | - |

---

## 📚 **使用指南與最佳實踐**

### 1. 導入約定

```python
# ✅ 推薦：從 aiva_common 導入通用合約
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    FindingPayload,
    FunctionTaskPayload,
)
from services.aiva_common.enums import ModuleName, Topic

# ✅ 推薦：從模組專用 schemas 導入專用合約
from services.function_sqli.aiva_func_sqli.schemas import (
    SqliDetectionResult,
    SqliTelemetry,
)

# ❌ 避免：不要直接從舊的 dataclass 文件導入
# from services.function_sqli.aiva_func_sqli.detection_models import DetectionResult
```

### 2. 創建和驗證數據

```python
from pydantic import ValidationError

# ✅ 正確：使用 Pydantic 的自動驗證
try:
    payload = ScanStartPayload(
        scan_id="scan_abc123xyz",
        targets=["<https://example.com"],>
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# ✅ 正確：使用 model_dump() 序列化
data_dict = payload.model_dump()

# ✅ 正確：使用 model_validate() 反序列化
payload_restored = ScanStartPayload.model_validate(data_dict)
```

### 3. 繼承和擴展

```python
# ✅ 推薦：繼承 FunctionTelemetry 創建專用遙測
from services.aiva_common.schemas import FunctionTelemetry

class CustomTelemetry(FunctionTelemetry):
    """自定義遙測，添加特定字段"""
    custom_metric: int = 0
    
    def record_custom_event(self) -> None:
        self.custom_metric += 1
```

### 4. JSON 序列化

```python
import json

# ✅ 推薦：使用 model_dump_json()
json_str = payload.model_dump_json()

# ✅ 推薦：使用 model_validate_json()
payload = ScanStartPayload.model_validate_json(json_str)

# ✅ 也可以：先 model_dump() 再 json.dumps()
data_dict = payload.model_dump()
json_str = json.dumps(data_dict)
```

### 5. 錯誤處理

```python
from pydantic import ValidationError

try:
    # 嘗試創建無效數據
    payload = ScanStartPayload(
        scan_id="invalid_id",  # 應該以 scan_ 開頭
        targets=[],  # 至少需要一個目標
    )
except ValidationError as e:
    # 獲取詳細的驗證錯誤
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Error: {error['msg']}")
        print(f"Type: {error['type']}")
```

---

## 📝 **版本控制與變更歷史**

### 版本 1.0.0（2025-10-13）

**新增**：

1. **通用基礎類**（services/aiva_common/schemas.py）
   - `FunctionTelemetry`：功能模組遙測基礎類
   - `ExecutionError`：統一錯誤記錄格式
   - `FunctionExecutionResult`：功能模組執行結果統一格式
   - `OastEvent`：OAST 事件記錄
   - `OastProbe`：OAST 探針配置
   - `ModuleStatus`：模組健康狀態報告

2. **枚舉擴展**（services/aiva_common/enums.py）
   - `TaskStatus`：任務狀態枚舉
   - `ScanStatus`：掃描狀態枚舉

3. **驗證規則增強**：
   - `ScanStartPayload`：scan_id 格式、targets 數量和有效性、strategy 值
   - `FindingPayload`：所有 ID 格式、status 值
   - `FunctionTaskPayload`：ID 格式、priority 範圍
   - `RateLimit`：burst ≥ requests_per_second

4. **模組專用 Schemas**：
   - **SQLi**：`SqliDetectionResult`, `SqliTelemetry`, `SqliEngineConfig`, `EncodedPayload`
   - **XSS**：`XssDetectionResult`, `XssTelemetry`, `DomDetectionResult`, `StoredXssResult`
   - **SSRF**：`SsrfTestVector`, `AnalysisPlan`, `SsrfTelemetry`
   - **Scan**：`SensitiveMatch`, `JavaScriptAnalysisResult`, `NetworkRequest`, `InteractionResult`

**改進**：

- 所有數據合約統一使用 Pydantic v2.12.0 BaseModel
- 為關鍵字段添加 `field_validator`
- 添加完整的類型提示
- 添加詳細的文檔字符串

**向後兼容**：

- 保留別名：`DetectionResult = SqliDetectionResult`
- 保留別名：`XssExecutionTelemetry = XssTelemetry`
- 保留別名：`SqliExecutionTelemetry = SqliTelemetry`

---

## 🔗 **相關文檔**

- [ARCHITECTURE_REPORT.md](./ARCHITECTURE_REPORT.md)：系統架構文檔
- [DATA_CONTRACT_ANALYSIS.md](./DATA_CONTRACT_ANALYSIS.md)：數據合約分析報告
- [QUICK_START.md](./QUICK_START.md)：快速開始指南

---

**文檔完成 - AIVA Data Contracts v1.0.0**
