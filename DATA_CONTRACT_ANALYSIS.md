# AIVA 數據合約完整分析報告

## Data Contract Analysis Report

生成時間：2025-10-13  
系統版本：1.0.0  
分析範圍：所有四大模組（Core, Scan, Function, Integration）

---

## 📋 **目錄**

1. [執行摘要](#執行摘要)
2. [現有數據合約分析](#現有數據合約分析)
3. [數據模型分類與統計](#數據模型分類與統計)
4. [問題與改進建議](#問題與改進建議)
5. [完善計劃](#完善計劃)
6. [實施路線圖](#實施路線圖)

---

## 🎯 **執行摘要**

### 分析結果概覽

| 項目 | 數量 | 狀態 |
|------|------|------|
| **官方 Pydantic 模型** | 18 | ✅ 完整 |
| **dataclass 模型** | 15+ | ⚠️ 需轉換 |
| **缺失的核心合約** | 8+ | ❌ 待新增 |
| **需要驗證規則** | 20+ | ⚠️ 待增強 |
| **模組專用 schemas** | 0/4 | ❌ 未實現 |

### 關鍵發現

1. **✅ 優勢**：
   - `aiva_common.schemas` 已全部使用 Pydantic v2.12.0 BaseModel
   - 核心訊息傳遞協議完整（MessageHeader, AivaMessage）
   - 漏洞報告結構完善（FindingPayload）

2. **⚠️ 需改進**：
   - 各功能模組仍大量使用 @dataclass（非 Pydantic）
   - 缺少統一的 Result、Error、Telemetry 基礎類
   - 驗證規則不完整（URL、ID、時間範圍等）

3. **❌ 缺失**：
   - OAST 事件數據合約
   - 模組狀態報告合約
   - 任務執行結果統一格式
   - 各功能模組專用 schemas 子模組

---

## 📊 **現有數據合約分析**

### 1. 核心通用合約（aiva_common.schemas）

#### ✅ **已完成 - 使用 Pydantic BaseModel**

```python
# 訊息協議
- MessageHeader          # MQ 訊息標頭
- AivaMessage            # 統一訊息封裝

# 掃描相關
- ScanStartPayload       # 掃描啟動
- ScanCompletedPayload   # 掃描完成
- ScanScope              # 掃描範圍
- Authentication         # 認證配置
- RateLimit              # 速率限制

# 功能任務
- FunctionTaskPayload    # 功能模組任務
- FunctionTaskTarget     # 測試目標
- FunctionTaskContext    # 上下文信息
- FunctionTaskTestConfig # 測試配置

# 漏洞報告
- FindingPayload         # 漏洞發現
- Vulnerability          # 漏洞類型
- FindingTarget          # 漏洞目標
- FindingEvidence        # 證據
- FindingImpact          # 影響
- FindingRecommendation  # 建議

# 反饋與狀態
- FeedbackEventPayload   # 反饋事件
- TaskUpdatePayload      # 任務更新
- HeartbeatPayload       # 心跳
- ConfigUpdatePayload    # 配置更新

# 資產與摘要
- Asset                  # 資產
- Summary                # 摘要
- Fingerprints           # 指紋
```

**驗證規則現狀**：

- ✅ RateLimit: non_negative validator
- ⚠️ 其他模型缺少驗證器

---

### 2. 枚舉類型（aiva_common.enums）

```python
✅ ModuleName           # 模組名稱（8 種）
✅ Topic                # MQ 主題（14 種）
✅ Severity             # 嚴重程度（5 種）
✅ Confidence           # 信心程度（3 種）
✅ VulnerabilityType    # 漏洞類型（7 種）
```

**狀態**: 完整，符合標準

---

### 3. SQLi 模組數據模型

#### ⚠️ **需轉換 - SQLi 模組使用 dataclass**

```python
# detection_models.py
@dataclass
class DetectionResult:          # ❌ 應為 Pydantic BaseModel
    is_vulnerable: bool
    vulnerability: Vulnerability
    evidence: FindingEvidence
    impact: FindingImpact
    recommendation: FindingRecommendation
    target: FindingTarget
    detection_method: str
    payload_used: str
    confidence_score: float

@dataclass
class DetectionError:           # ❌ 應為 Pydantic BaseModel
    payload: str
    vector: str
    message: str
    attempts: int
    engine_name: str

# telemetry.py
@dataclass
class SqliExecutionTelemetry:  # ❌ 應為 Pydantic BaseModel
    payloads_sent: int
    detections: int
    errors: list[str]
    engines_run: list[str]

# payload_wrapper_encoder.py
@dataclass
class EncodedPayload:           # ❌ 應為 Pydantic BaseModel
    url: str
    method: str
    payload: str
    request_kwargs: dict[str, Any]

# worker.py / worker_legacy.py
@dataclass
class SqliEngineConfig:         # ❌ 應為 Pydantic BaseModel
    timeout_seconds: float
    max_retries: int
    enable_error_detection: bool
    enable_boolean_detection: bool
    enable_time_detection: bool
    enable_union_detection: bool
    enable_oob_detection: bool

@dataclass
class SqliDetectionContext:     # ❌ 應為 Pydantic BaseModel
    task: FunctionTaskPayload
    findings: list[FindingPayload]
    telemetry: SqliExecutionTelemetry
    http_client: httpx.AsyncClient
```

---

### 4. XSS 模組數據模型

#### ⚠️ **需轉換 - XSS 模組使用 dataclass**

```python
# traditional_detector.py
@dataclass
class XssDetectionResult:       # ❌ 應為 Pydantic BaseModel
    payload: str
    request: httpx.Request
    response_status: int
    response_headers: dict[str, str]
    response_text: str

@dataclass
class XssExecutionError:        # ❌ 應為 Pydantic BaseModel
    payload: str
    vector: str
    message: str
    attempts: int

# worker.py
@dataclass
class XssExecutionTelemetry:   # ❌ 應為 Pydantic BaseModel
    payloads_sent: int
    reflections: int
    dom_escalations: int
    blind_callbacks: int
    errors: list[str]

@dataclass
class TaskExecutionResult:      # ❌ 應為 Pydantic BaseModel
    findings: list[FindingPayload]
    telemetry: XssExecutionTelemetry

# dom_xss_detector.py
@dataclass
class DomDetectionResult:       # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取

# stored_detector.py
@dataclass
class StoredXssResult:          # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取
```

---

### 5. SSRF 模組數據模型

#### ⚠️ **需轉換 - SSRF 模組使用 dataclass**

```python
# worker.py
@dataclass
class SsrfTelemetry:            # ❌ 應為 Pydantic BaseModel
    attempts: int
    findings: int
    oast_callbacks: int
    errors: list[str]

@dataclass
class TaskExecutionResult:      # ❌ 應為 Pydantic BaseModel
    findings: list[FindingPayload]
    telemetry: SsrfTelemetry

# param_semantics_analyzer.py
@dataclass
class SsrfTestVector:           # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取

@dataclass
class AnalysisPlan:             # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取

# 缺失：OastEvent 數據合約
```

---

### 6. IDOR 模組數據模型

#### ✅ **已轉換 - 使用 Pydantic BaseModel**

```python
# vertical_escalation_tester.py
class PrivilegeLevel(str, Enum):       # ✅ 標準 Enum
    GUEST = "guest"
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"

class VerticalTestResult(BaseModel):   # ✅ Pydantic BaseModel
    vulnerable: bool
    confidence: Confidence
    severity: Severity
    vulnerability_type: VulnerabilityType
    evidence: str | None
    description: str | None
    privilege_level: PrivilegeLevel
    status_code: int | None
    should_deny: bool
    actual_level: PrivilegeLevel | None
    attempted_level: PrivilegeLevel | None

# cross_user_tester.py
class CrossUserTestResult(BaseModel):  # ✅ Pydantic BaseModel
    vulnerable: bool
    confidence: Confidence
    severity: Severity
    vulnerability_type: VulnerabilityType
    evidence: str | None
    description: str | None
    test_status: str
    similarity_score: float
```

**狀態**: ✅ 已完成，符合標準

---

### 7. Scan 模組數據模型

#### ⚠️ **需轉換 - Scan 模組使用 dataclass**

```python
# sensitive_info_detector.py
class SensitiveInfoType(Enum):         # ✅ 標準 Enum
class Location(Enum):                  # ✅ 標準 Enum

@dataclass
class SensitiveMatch:                  # ❌ 應為 Pydantic BaseModel
    info_type: SensitiveInfoType
    value: str
    location: Location
    context: str
    line_number: int | None
    severity: Severity
    description: str
    recommendation: str

# javascript_source_analyzer.py
@dataclass
class AnalysisResult:                  # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取

# dynamic_content_extractor.py
@dataclass
class NetworkRequest:                  # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取

# js_interaction_simulator.py
@dataclass
class InteractionResult:               # ❌ 應為 Pydantic BaseModel
    # 字段信息未完整讀取
```

---

### 8. Integration 模組數據模型

#### ✅ **使用 SQLAlchemy ORM + Pydantic 轉換**

```python
# sql_result_database.py
class FindingRecord(Base):             # ✅ SQLAlchemy ORM
    """數據庫表模型"""
    finding_id: str
    task_id: str
    scan_id: str
    vulnerability_type: str
    severity: str
    confidence: str
    raw_data: str  # JSON 存儲完整 FindingPayload
    
    def to_finding_payload(self) -> FindingPayload:
        """轉換為 Pydantic 模型"""
        return FindingPayload.model_validate_json(self.raw_data)
```

**狀態**: ✅ 設計合理

---

## 📈 **數據模型分類與統計**

### 按模組分類

| 模組 | Pydantic ✅ | dataclass ⚠️ | 其他 | 完成度 |
|------|------------|--------------|------|---------|
| **aiva_common** | 18 | 0 | 5 Enum | 100% |
| **function_sqli** | 0 | 6+ | - | 0% |
| **function_xss** | 0 | 5+ | - | 0% |
| **function_ssrf** | 0 | 4+ | - | 0% |
| **function_idor** | 2 | 0 | 1 Enum | 100% |
| **scan** | 0 | 4+ | 2 Enum | 0% |
| **integration** | 1 | 0 | 1 ORM | 100% |
| **core** | 使用通用 | - | - | - |
| **總計** | 21 | 19+ | 9 | **52%** |

### 按功能分類

| 功能類別 | 模型數量 | 標準化程度 |
|---------|---------|-----------|
| **訊息傳遞** | 5 | ✅ 完整 |
| **任務管理** | 6 | ✅ 完整 |
| **漏洞報告** | 6 | ✅ 完整 |
| **檢測結果** | 8+ | ⚠️ 不一致 |
| **遙測數據** | 4+ | ⚠️ 不一致 |
| **測試配置** | 3+ | ⚠️ 不一致 |
| **錯誤處理** | 2+ | ⚠️ 不一致 |

---

## ⚠️ **問題與改進建議**

### 問題清單

#### 1. **數據模型不統一**

- **問題**: SQLi, XSS, SSRF 使用 dataclass；IDOR 使用 Pydantic
- **影響**:
  - 無法使用 Pydantic 的驗證功能
  - JSON 序列化不一致
  - 與 FastAPI 集成困難
- **建議**: 全部轉換為 Pydantic BaseModel

#### 2. **缺少統一的基礎類**

- **問題**: 每個模組自定義 DetectionResult, Telemetry
- **影響**: 代碼重複，維護困難
- **建議**: 在 aiva_common.schemas 定義通用基礎類

```python
# 建議新增
class FunctionExecutionResult(BaseModel):
    """功能模組執行結果基礎類"""
    findings: list[FindingPayload]
    telemetry: BaseTelemetry
    errors: list[ExecutionError]

class BaseTelemetry(BaseModel):
    """遙測數據基礎類"""
    payloads_sent: int
    detections: int
    errors: list[str]
    duration_seconds: float

class ExecutionError(BaseModel):
    """執行錯誤統一格式"""
    payload: str
    error_type: str
    message: str
    timestamp: datetime
```

#### 3. **驗證規則不完整**

- **問題**: 大部分模型沒有 field_validator
- **影響**:
  - 無效數據可能進入系統
  - 運行時錯誤增加
  - 安全風險
- **建議**: 添加完整的驗證規則

```python
# 示例
class FunctionTaskPayload(BaseModel):
    task_id: str
    scan_id: str
    priority: int = Field(ge=1, le=10)  # 範圍驗證
    
    @field_validator("task_id", "scan_id")
    def validate_id_format(cls, v: str) -> str:
        """驗證 ID 格式"""
        if not v.startswith(("task_", "scan_")):
            raise ValueError("Invalid ID format")
        return v
```

#### 4. **缺少 OAST 事件合約**

- **問題**: SSRF 模組使用 OAST，但沒有標準化事件格式
- **影響**: OAST 服務難以標準化
- **建議**: 定義 OastEvent, OastProbe 等合約

#### 5. **模組專用 schemas 未實現**

- **問題**: 各功能模組沒有專用的 schemas 子模組
- **影響**:
  - 無法清晰區分通用與專用合約
  - 維護困難
- **建議**: 創建模組專用 schemas

```text
services/function_sqli/aiva_func_sqli/
    schemas.py              # SQLi 專用數據合約
    __init__.py

services/function_xss/aiva_func_xss/
    schemas.py              # XSS 專用數據合約
    __init__.py
```

#### 6. **JSON 序列化不一致**

- **問題**: dataclass 需要手動實現 to_dict；Pydantic 有 model_dump
- **影響**: 代碼不一致
- **建議**: 統一使用 Pydantic model_dump()

#### 7. **缺少文檔**

- **問題**: 沒有數據合約文檔
- **影響**: 新開發者難以理解
- **建議**: 創建 DATA_CONTRACT.md

---

## 🔧 **完善計劃**

### Phase 1: 統一數據模型基礎（Week 1-2）

#### 1.1 擴展 aiva_common.schemas

```python
# 新增通用基礎類
class FunctionExecutionResult(BaseModel):
    """功能模組執行結果統一格式"""
    findings: list[FindingPayload]
    telemetry: FunctionTelemetry
    errors: list[ExecutionError] = Field(default_factory=list)
    duration_seconds: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class FunctionTelemetry(BaseModel):
    """功能模組遙測基礎類"""
    payloads_sent: int = 0
    detections: int = 0
    attempts: int = 0
    success_rate: float = 0.0
    errors: list[str] = Field(default_factory=list)

class ExecutionError(BaseModel):
    """執行錯誤統一格式"""
    error_id: str
    error_type: str
    message: str
    payload: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    severity: Severity = Severity.MEDIUM

class OastEvent(BaseModel):
    """OAST 事件數據合約"""
    event_id: str
    probe_token: str
    event_type: str  # "http", "dns", "smtp"
    source_ip: str
    timestamp: datetime
    raw_data: dict[str, Any] = Field(default_factory=dict)

class OastProbe(BaseModel):
    """OAST 探針數據合約"""
    probe_id: str
    token: str
    callback_url: HttpUrl
    task_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

class ModuleStatus(BaseModel):
    """模組狀態報告"""
    module: ModuleName
    status: str  # "running", "stopped", "error"
    worker_count: int
    queue_size: int
    last_heartbeat: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)

class TaskStatus(str, Enum):
    """任務狀態枚舉"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

#### 1.2 添加完整驗證規則

```python
class ScanStartPayload(BaseModel):
    scan_id: str
    targets: list[HttpUrl]
    
    @field_validator("scan_id")
    def validate_scan_id(cls, v: str) -> str:
        if not v.startswith("scan_"):
            raise ValueError("scan_id must start with 'scan_'")
        if len(v) < 10:
            raise ValueError("scan_id too short")
        return v
    
    @field_validator("targets")
    def validate_targets(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        if not v:
            raise ValueError("At least one target required")
        if len(v) > 100:
            raise ValueError("Too many targets (max 100)")
        return v

class RateLimit(BaseModel):
    requests_per_second: int = Field(ge=1, le=1000)
    burst: int = Field(ge=1, le=5000)
    
    @field_validator("burst")
    def burst_must_be_larger(cls, v: int, info) -> int:
        if "requests_per_second" in info.data:
            if v < info.data["requests_per_second"]:
                raise ValueError("burst must be >= requests_per_second")
        return v

class FindingPayload(BaseModel):
    finding_id: str
    task_id: str
    scan_id: str
    
    @field_validator("finding_id", "task_id", "scan_id")
    def validate_id_format(cls, v: str, info) -> str:
        field_name = info.field_name
        prefix_map = {
            "finding_id": "finding_",
            "task_id": "task_",
            "scan_id": "scan_",
        }
        prefix = prefix_map.get(field_name, "")
        if not v.startswith(prefix):
            raise ValueError(f"{field_name} must start with '{prefix}'")
        return v
```

### Phase 2: 創建模組專用 schemas（Week 2-3）

#### 2.1 SQLi 模組 schemas

創建 `services/function_sqli/aiva_func_sqli/schemas.py`:

```python
"""SQLi 模組專用數據合約"""
from __future__ import annotations

from pydantic import BaseModel, Field

from services.aiva_common.schemas import (
    FindingEvidence,
    FindingImpact,
    FindingRecommendation,
    FindingTarget,
    FunctionTelemetry,
    Vulnerability,
)


class SqliDetectionResult(BaseModel):
    """SQLi 檢測結果"""
    is_vulnerable: bool
    vulnerability: Vulnerability
    evidence: FindingEvidence
    impact: FindingImpact
    recommendation: FindingRecommendation
    target: FindingTarget
    detection_method: str  # "error", "boolean", "time", "union", "oob"
    payload_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    db_fingerprint: str | None = None
    response_time: float = 0.0


class SqliTelemetry(FunctionTelemetry):
    """SQLi 專用遙測"""
    engines_run: list[str] = Field(default_factory=list)
    blind_detections: int = 0
    error_based_detections: int = 0
    union_based_detections: int = 0


class SqliEngineConfig(BaseModel):
    """SQLi 引擎配置"""
    timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)
    enable_error_detection: bool = True
    enable_boolean_detection: bool = True
    enable_time_detection: bool = True
    enable_union_detection: bool = True
    enable_oob_detection: bool = True


class EncodedPayload(BaseModel):
    """編碼後的 Payload"""
    url: str
    method: str
    payload: str
    request_kwargs: dict[str, Any] = Field(default_factory=dict)
    
    def build_request_dump(self) -> str:
        """構建請求轉儲字符串"""
        lines = [f"{self.method} {self.url}"]
        # ... 實現邏輯
        return "\n".join(lines)
```

#### 2.2 XSS 模組 schemas

創建 `services/function_xss/aiva_func_xss/schemas.py`:

```python
"""XSS 模組專用數據合約"""
from __future__ import annotations

from pydantic import BaseModel, Field
import httpx

from services.aiva_common.schemas import FunctionTelemetry


class XssDetectionResult(BaseModel):
    """XSS 檢測結果"""
    payload: str
    request_url: str  # 替代 httpx.Request
    request_method: str
    response_status: int
    response_headers: dict[str, str]
    response_text: str
    reflection_found: bool
    context: str | None = None


class XssTelemetry(FunctionTelemetry):
    """XSS 專用遙測"""
    reflections: int = 0
    dom_escalations: int = 0
    blind_callbacks: int = 0
    stored_xss_found: int = 0


class DomDetectionResult(BaseModel):
    """DOM XSS 檢測結果"""
    vulnerable: bool
    sink_type: str  # "innerHTML", "eval", "document.write"
    source_type: str  # "location.hash", "location.search"
    payload: str
    evidence: str
```

#### 2.3 SSRF 模組 schemas

創建 `services/function_ssrf/aiva_func_ssrf/schemas.py`:

```python
"""SSRF 模組專用數據合約"""
from __future__ import annotations

from pydantic import BaseModel, Field

from services.aiva_common.schemas import FunctionTelemetry, OastEvent


class SsrfTestVector(BaseModel):
    """SSRF 測試向量"""
    payload: str
    vector_type: str  # "internal", "cloud_metadata", "oast"
    priority: int = Field(ge=1, le=10)
    requires_oast: bool = False


class AnalysisPlan(BaseModel):
    """SSRF 分析計劃"""
    vectors: list[SsrfTestVector]
    param_name: str | None
    semantic_hints: list[str] = Field(default_factory=list)


class SsrfTelemetry(FunctionTelemetry):
    """SSRF 專用遙測"""
    oast_callbacks: int = 0
    internal_access: int = 0
    cloud_metadata_access: int = 0
```

### Phase 3: 轉換現有代碼（Week 3-4）

#### 3.1 轉換策略

1. **保留向後兼容性**：創建別名
2. **漸進式遷移**：先新增 Pydantic 版本，再替換舊代碼
3. **測試覆蓋**：確保每個轉換都有測試

#### 3.2 轉換優先級

| 優先級 | 模組 | 原因 |
|--------|------|------|
| **P0** | function_sqli | 使用最頻繁 |
| **P0** | function_xss | 使用最頻繁 |
| **P1** | function_ssrf | 依賴 OAST |
| **P1** | scan | 核心模組 |
| **P2** | 其他 | - |

### Phase 4: 文檔與測試（Week 4-5）

#### 4.1 創建 DATA_CONTRACT.md

完整記錄：

- 所有數據合約的用途
- 字段說明與示例
- 驗證規則
- 版本歷史
- 使用指南

#### 4.2 添加單元測試

```python
# tests/test_schemas.py
def test_scan_start_payload_validation():
    # 有效數據
    valid_payload = ScanStartPayload(
        scan_id="scan_abc123",
        targets=["<https://example.com"]>
    )
    assert valid_payload.scan_id == "scan_abc123"
    
    # 無效 scan_id
    with pytest.raises(ValidationError):
        ScanStartPayload(
            scan_id="invalid",
            targets=["<https://example.com"]>
        )
```

---

## 📅 **實施路線圖**

### Week 1-2: 基礎設施

- [x] ✅ 分析現有數據合約
- [ ] 🔄 擴展 aiva_common.schemas（通用基礎類）
- [ ] 🔄 添加完整驗證規則
- [ ] 🔄 更新 enums.py（TaskStatus 等）

### Week 2-3: 模組專用 schemas

- [ ] 📝 創建 function_sqli/schemas.py
- [ ] 📝 創建 function_xss/schemas.py
- [ ] 📝 創建 function_ssrf/schemas.py
- [ ] 📝 創建 scan/schemas.py

### Week 3-4: 代碼轉換

- [ ] 🔄 轉換 SQLi 模組（6 個 dataclass）
- [ ] 🔄 轉換 XSS 模組（5 個 dataclass）
- [ ] 🔄 轉換 SSRF 模組（4 個 dataclass）
- [ ] 🔄 轉換 Scan 模組（4 個 dataclass）

### Week 4-5: 測試與文檔

- [ ] 📖 創建 DATA_CONTRACT.md
- [ ] ✅ 添加數據合約單元測試
- [ ] ✅ 添加集成測試
- [ ] 📊 生成 API 文檔

---

## 📝 **總結**

### 完成標準

1. **100% Pydantic**: 所有數據模型使用 Pydantic BaseModel
2. **完整驗證**: 所有關鍵字段有 field_validator
3. **模組化**: 每個功能模組有專用 schemas.py
4. **文檔完整**: DATA_CONTRACT.md 涵蓋所有合約
5. **測試覆蓋**: 所有數據合約有單元測試

### 預期收益

- ✅ **一致性**: 統一的數據格式
- ✅ **安全性**: 完整的輸入驗證
- ✅ **可維護性**: 清晰的模組結構
- ✅ **可擴展性**: 易於添加新功能
- ✅ **自動化**: FastAPI 自動生成文檔

---

## 報告完成 - 準備開始實施
