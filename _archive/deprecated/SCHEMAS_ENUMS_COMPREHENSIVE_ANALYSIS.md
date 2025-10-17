# AIVA Schemas & Enums 全面檢查報告

## 📋 執行時間
**日期**: 2025-10-16  
**檢查範圍**: 
- `services/aiva_common/schemas/` (13 個文件)
- `services/aiva_common/enums/` (4 個文件)
- 全系統類似腳本掃描

---

## 🎯 檢查目標

1. 根據現況和未來發展，確認 schemas 和 enums 是否需要新增或調整
2. 檢查整個系統有無類似的統計/遙測腳本
3. 評估是否需要統一化或標準化

---

## 📊 現狀分析

### 1. Schemas 文件清單

| 文件名 | 用途 | 狀態 |
|--------|------|------|
| `base.py` | 基礎模型（MessageHeader, Authentication 等） | ✅ 完整 |
| `messaging.py` | 訊息系統（AivaMessage, AIVARequest 等） | ✅ 完整 |
| `tasks.py` | 任務相關（掃描、功能任務等） | ✅ 完整 |
| `findings.py` | 漏洞發現（FindingPayload, Vulnerability 等） | ✅ 完整 |
| `telemetry.py` | 遙測監控（HeartbeatPayload, OastEvent 等） | ⚠️ 需擴展 |
| `ai.py` | AI 相關（訓練、RAG 等） | ✅ 完整 |
| `assets.py` | 資產管理（EASM 相關） | ✅ 完整 |
| `risk.py` | 風險評估（攻擊路徑等） | ✅ 完整 |
| `languages.py` | 多語言支援 | ✅ 完整 |
| `api_testing.py` | API 安全測試 | ✅ 完整 |
| `enhanced.py` | 增強版 Schema | ✅ 完整 |
| `system.py` | 系統編排 | ✅ 完整 |
| `references.py` | 參考資料（CVE, CWE 等） | ✅ 完整 |

### 2. Enums 文件清單

| 文件名 | 用途 | 狀態 |
|--------|------|------|
| `common.py` | 通用枚舉（Severity, Confidence 等） | ✅ 完整 |
| `modules.py` | 模組相關（ModuleName, Topic 等） | ✅ 完整 |
| `security.py` | 安全測試（VulnerabilityType 等） | ✅ 完整 |
| `assets.py` | 資產管理 | ✅ 完整 |

---

## 🔍 發現的類似腳本

### 統計/遙測相關實現

#### 1. Worker 統計模組（新建）
- **位置**: `services/function/common/worker_statistics.py`
- **用途**: 統一的 Worker 統計數據收集
- **特點**: 
  - 完整的 Pydantic 模型
  - 支持 OAST 回調、錯誤分類、Early Stopping
  - 模組特定擴展支持

#### 2. 各功能模組的遙測實現

| 模組 | 文件 | 實現類型 | 狀態 |
|------|------|---------|------|
| **SQLi** | `function_sqli/telemetry.py` | Dataclass | ⚠️ 應遷移 |
| **XSS** | `function_xss/worker.py` | Dataclass (內嵌) | ⚠️ 應遷移 |
| **SSRF** | `function_ssrf/worker.py` | Dataclass | ⚠️ 應遷移 |
| **SSRF (Enhanced)** | `function_ssrf/enhanced_worker.py` | Dataclass + StatisticsCollector | ✅ 現代化 |
| **IDOR (Enhanced)** | `function_idor/enhanced_worker.py` | Dataclass + StatisticsCollector | ✅ 現代化 |
| **PostEx** | `function_postex/schemas.py` | Pydantic (繼承 FunctionTelemetry) | ✅ 標準化 |

#### 3. Core 模組的指標收集器

| 組件 | 文件 | 實現 | 用途 |
|------|------|------|------|
| **MetricsCollector** | `core/optimized_core.py` | Class | 性能指標收集 |
| **FingerprintCollector** | `scan/fingerprint_manager.py` | Class | 指紋信息收集 |
| **ResultCollector** | `core/messaging/result_collector.py` | Class | 結果收集 |

---

## ⚠️ 發現的問題

### 1. Telemetry Schema 不完整

**當前狀態** (`telemetry.py`):
```python
class FunctionTelemetry(BaseModel):
    """功能模組遙測數據基礎類"""
    payloads_sent: int = 0
    detections: int = 0
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**問題**:
- ❌ 缺少 OAST 回調統計
- ❌ 缺少錯誤分類
- ❌ 缺少 Early Stopping 信息
- ❌ 缺少自適應行為追蹤
- ❌ 不支持模組特定擴展

### 2. 遙測實現不統一

**發現的模式**:
1. **Dataclass 模式** (舊): SQLi, XSS, SSRF basic
2. **Pydantic + 繼承模式** (中): PostEx
3. **Dataclass + StatisticsCollector 模式** (新): SSRF/IDOR enhanced
4. **獨立 MetricsCollector** (Core): 性能指標

**問題**:
- 不同模組使用不同的遙測模式
- 缺乏統一的接口和標準
- 難以聚合和分析跨模組數據

### 3. 錯誤處理缺乏標準化

**當前實現**:
- SQLi: `errors: list[str]` (簡單字符串)
- XSS: `errors: list[str]` (簡單字符串)
- worker_statistics: `ErrorRecord` + `ErrorCategory` 枚舉

**問題**:
- 錯誤信息格式不統一
- 缺少錯誤分類和嚴重程度
- 難以進行錯誤分析和告警

### 4. OAST 相關定義分散

**當前狀態**:
- `telemetry.py`: `OastEvent`, `OastProbe` (基礎定義)
- `worker_statistics.py`: `OastCallbackRecord` (詳細記錄)
- 各 Worker: 自行實現 OAST 統計

**問題**:
- OAST 相關定義分散在多個文件
- 缺少統一的 OAST 統計標準

---

## 💡 建議的改進方案

### 方案 A: 漸進式統一（推薦）

#### 階段 1: 擴展現有 Schema（立即可行）

**在 `telemetry.py` 中新增**:

```python
# 1. 新增增強版功能遙測基類
class EnhancedFunctionTelemetry(FunctionTelemetry):
    """增強版功能模組遙測數據"""
    
    # OAST 統計
    oast_probes_sent: int = 0
    oast_callbacks_received: int = 0
    oast_callback_details: list[dict[str, Any]] = Field(default_factory=list)
    
    # 錯誤統計
    error_count: int = 0
    errors_by_category: dict[str, int] = Field(default_factory=dict)
    error_details: list[dict[str, Any]] = Field(default_factory=list)
    
    # Early Stopping
    early_stopping_triggered: bool = False
    early_stopping_reason: str | None = None
    
    # 自適應行為
    adaptive_timeout_used: bool = False
    rate_limiting_applied: bool = False
    protection_detected: bool = False
    
    # 模組特定數據
    module_specific: dict[str, Any] = Field(default_factory=dict)

# 2. 新增錯誤記錄 Schema
class ErrorRecord(BaseModel):
    """錯誤記錄"""
    category: str  # 錯誤類別
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    severity: str = "error"  # "error", "warning", "critical"
    request_info: dict[str, Any] = Field(default_factory=dict)
    stack_trace: str | None = None

# 3. 新增 OAST 回調記錄
class OastCallbackDetail(BaseModel):
    """OAST 回調詳細記錄"""
    probe_token: str
    callback_type: str  # "dns", "http", "smtp"
    source_ip: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload_info: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
```

**優點**:
- ✅ 保持向後兼容（繼承現有 FunctionTelemetry）
- ✅ 立即可用，無需大規模重構
- ✅ 逐步遷移各 Worker

#### 階段 2: 新增 Enums（立即可行）

**在 `common.py` 中新增**:

```python
class ErrorCategory(str, Enum):
    """錯誤分類"""
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    PROTECTION = "protection"
    PARSING = "parsing"
    UNKNOWN = "unknown"

class StoppingReason(str, Enum):
    """Early Stopping 原因"""
    MAX_VULNERABILITIES = "max_vulnerabilities_reached"
    TIME_LIMIT = "time_limit_exceeded"
    PROTECTION_DETECTED = "protection_detected"
    ERROR_THRESHOLD = "error_threshold_exceeded"
    RATE_LIMITED = "rate_limited"
    NO_RESPONSE = "no_response_timeout"
```

**優點**:
- ✅ 標準化錯誤分類
- ✅ 提供類型安全
- ✅ 便於統計和分析

#### 階段 3: 統一 Worker 遙測（中期）

**遷移計劃**:

1. **Week 1**: SQLi Worker
   - 遷移 `SqliExecutionTelemetry` → 使用 `EnhancedFunctionTelemetry`
   - 集成 `StatisticsCollector`

2. **Week 2**: XSS Worker
   - 遷移 `XssExecutionTelemetry` → 使用 `EnhancedFunctionTelemetry`
   - 集成 `StatisticsCollector`

3. **Week 3**: 舊版 SSRF/IDOR Worker
   - 遷移到 Enhanced Worker 模式
   - 統一使用 `StatisticsCollector`

### 方案 B: 完全重構（長期）

創建獨立的 `worker_telemetry.py` 模組，整合所有 Worker 遙測邏輯：

```python
# services/aiva_common/schemas/worker_telemetry.py
"""
Worker 遙測統一模組
整合所有功能模組的遙測數據定義
"""

from .telemetry import FunctionTelemetry
from services.function.common.worker_statistics import (
    WorkerStatistics,
    StatisticsCollector,
    ErrorCategory,
    StoppingReason,
    ErrorRecord,
    OastCallbackRecord,
    EarlyStoppingRecord,
)

# 重新導出以便統一訪問
__all__ = [
    "FunctionTelemetry",
    "WorkerStatistics",
    "StatisticsCollector",
    "ErrorCategory",
    "StoppingReason",
    "ErrorRecord",
    "OastCallbackRecord",
    "EarlyStoppingRecord",
]
```

---

## 📈 優先級建議

### 高優先級（立即執行）

1. ✅ **在 `common.py` 新增錯誤和停止原因枚舉**
   - 工時: 1-2 小時
   - 影響: 所有 Worker
   - ROI: 95/100

2. ✅ **在 `telemetry.py` 新增 EnhancedFunctionTelemetry**
   - 工時: 2-3 小時
   - 影響: 提供現代化遙測基類
   - ROI: 90/100

3. ✅ **在 `__init__.py` 導出新增的類型**
   - 工時: 30 分鐘
   - 影響: 確保其他模組可訪問
   - ROI: 100/100

### 中優先級（1-2 週內）

4. **遷移 SQLi Worker 遙測**
   - 工時: 1-2 天
   - 影響: SQLi 模組現代化
   - ROI: 85/100

5. **遷移 XSS Worker 遙測**
   - 工時: 1-2 天
   - 影響: XSS 模組現代化
   - ROI: 85/100

6. **創建遙測標準化文檔**
   - 工時: 1 天
   - 影響: 開發團隊對齊
   - ROI: 80/100

### 低優先級（長期）

7. **統一 Core MetricsCollector**
   - 工時: 1-2 週
   - 影響: Core 模組優化
   - ROI: 70/100

8. **創建遙測可視化儀表板**
   - 工時: 2-3 週
   - 影響: 運維監控
   - ROI: 75/100

---

## 🎯 具體實施步驟

### Step 1: 新增 Enums（立即）

**文件**: `services/aiva_common/enums/common.py`

**添加位置**: 在現有 enums 之後

```python
class ErrorCategory(str, Enum):
    """錯誤分類 - 用於統計和分析"""
    NETWORK = "network"  # 網絡錯誤
    TIMEOUT = "timeout"  # 超時錯誤
    RATE_LIMIT = "rate_limit"  # 速率限制
    VALIDATION = "validation"  # 驗證錯誤
    PROTECTION = "protection"  # 保護機制檢測到
    PARSING = "parsing"  # 解析錯誤
    AUTHENTICATION = "authentication"  # 認證錯誤
    AUTHORIZATION = "authorization"  # 授權錯誤
    UNKNOWN = "unknown"  # 未知錯誤


class StoppingReason(str, Enum):
    """Early Stopping 原因 - 用於記錄檢測提前終止的原因"""
    MAX_VULNERABILITIES = "max_vulnerabilities_reached"  # 達到最大漏洞數
    TIME_LIMIT = "time_limit_exceeded"  # 超過時間限制
    PROTECTION_DETECTED = "protection_detected"  # 檢測到防護
    ERROR_THRESHOLD = "error_threshold_exceeded"  # 錯誤率過高
    RATE_LIMITED = "rate_limited"  # 被速率限制
    NO_RESPONSE = "no_response_timeout"  # 無響應超時
    MANUAL_STOP = "manual_stop"  # 手動停止
    RESOURCE_EXHAUSTED = "resource_exhausted"  # 資源耗盡
```

**更新**: `services/aiva_common/enums/__init__.py`

添加到導出列表：
```python
from .common import (
    # ... 現有導出 ...
    ErrorCategory,
    StoppingReason,
)

__all__ = [
    # ... 現有列表 ...
    "ErrorCategory",
    "StoppingReason",
]
```

### Step 2: 擴展 Telemetry Schema（立即）

**文件**: `services/aiva_common/schemas/telemetry.py`

**添加位置**: 在 `FunctionTelemetry` 之後

```python
class ErrorRecord(BaseModel):
    """錯誤記錄 - 提供結構化的錯誤信息"""
    
    category: str = Field(description="錯誤類別")
    message: str = Field(description="錯誤訊息")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="錯誤發生時間"
    )
    severity: str = Field(
        default="error",
        description="嚴重程度: error, warning, critical"
    )
    request_info: dict[str, Any] = Field(
        default_factory=dict,
        description="請求相關信息"
    )
    stack_trace: str | None = Field(
        default=None,
        description="堆棧追蹤（如果可用）"
    )
    
    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"error", "warning", "critical", "info"}
        if v not in allowed:
            raise ValueError(f"Invalid severity: {v}. Must be one of {allowed}")
        return v


class OastCallbackDetail(BaseModel):
    """OAST 回調詳細記錄"""
    
    probe_token: str = Field(description="探針 Token")
    callback_type: str = Field(description="回調類型: dns, http, smtp 等")
    source_ip: str = Field(description="來源 IP 地址")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="回調時間"
    )
    payload_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Payload 詳細信息"
    )
    success: bool = Field(default=True, description="是否成功")
    
    @field_validator("callback_type")
    @classmethod
    def validate_callback_type(cls, v: str) -> str:
        allowed = {"dns", "http", "https", "smtp", "ftp", "ldap", "other"}
        if v not in allowed:
            raise ValueError(
                f"Invalid callback_type: {v}. Must be one of {allowed}"
            )
        return v


class EnhancedFunctionTelemetry(FunctionTelemetry):
    """
    增強版功能模組遙測數據
    
    擴展基礎 FunctionTelemetry，添加現代化的統計功能：
    - OAST 回調追蹤
    - 結構化錯誤記錄
    - Early Stopping 檢測
    - 自適應行為追蹤
    - 模組特定擴展支持
    """
    
    # OAST 統計
    oast_probes_sent: int = Field(default=0, description="發送的 OAST 探針數量")
    oast_callbacks_received: int = Field(
        default=0,
        description="接收到的 OAST 回調數量"
    )
    oast_callback_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="OAST 回調詳細記錄列表"
    )
    
    # 增強錯誤統計
    error_count: int = Field(default=0, description="錯誤總數")
    errors_by_category: dict[str, int] = Field(
        default_factory=dict,
        description="按類別統計的錯誤數"
    )
    error_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="錯誤詳細記錄列表"
    )
    
    # Early Stopping
    early_stopping_triggered: bool = Field(
        default=False,
        description="是否觸發了 Early Stopping"
    )
    early_stopping_reason: str | None = Field(
        default=None,
        description="Early Stopping 原因"
    )
    early_stopping_details: dict[str, Any] | None = Field(
        default=None,
        description="Early Stopping 詳細信息"
    )
    
    # 自適應行為追蹤
    adaptive_timeout_used: bool = Field(
        default=False,
        description="是否使用了自適應超時"
    )
    timeout_adjustments: int = Field(
        default=0,
        description="超時調整次數"
    )
    rate_limiting_applied: bool = Field(
        default=False,
        description="是否應用了速率限制"
    )
    protection_detected: bool = Field(
        default=False,
        description="是否檢測到保護機制（WAF, Rate Limit 等）"
    )
    
    # 請求詳細統計
    total_requests: int = Field(default=0, description="總請求數")
    successful_requests: int = Field(default=0, description="成功請求數")
    failed_requests: int = Field(default=0, description="失敗請求數")
    timeout_requests: int = Field(default=0, description="超時請求數")
    rate_limited_requests: int = Field(default=0, description="被速率限制的請求數")
    
    # 檢測效率
    payloads_tested: int = Field(default=0, description="測試的 Payload 數量")
    payloads_succeeded: int = Field(default=0, description="成功的 Payload 數量")
    false_positives_filtered: int = Field(
        default=0,
        description="過濾的誤報數量"
    )
    
    # 模組特定數據（靈活擴展）
    module_specific: dict[str, Any] = Field(
        default_factory=dict,
        description="模組特定的統計數據"
    )
    
    def get_success_rate(self) -> float:
        """計算請求成功率"""
        if self.total_requests == 0:
            return 0.0
        return round(self.successful_requests / self.total_requests * 100, 2)
    
    def get_payload_success_rate(self) -> float:
        """計算 Payload 成功率"""
        if self.payloads_tested == 0:
            return 0.0
        return round(self.payloads_succeeded / self.payloads_tested * 100, 2)
    
    def get_oast_success_rate(self) -> float:
        """計算 OAST 回調成功率"""
        if self.oast_probes_sent == 0:
            return 0.0
        return round(self.oast_callbacks_received / self.oast_probes_sent * 100, 2)
    
    def get_error_rate(self) -> float:
        """計算錯誤率"""
        if self.total_requests == 0:
            return 0.0
        return round(self.error_count / self.total_requests * 100, 2)
```

**更新**: `services/aiva_common/schemas/__init__.py`

添加到導出列表：
```python
from .telemetry import (
    # ... 現有導出 ...
    ErrorRecord,
    OastCallbackDetail,
    EnhancedFunctionTelemetry,
)

__all__ = [
    # ... 現有列表 ...
    "ErrorRecord",
    "OastCallbackDetail",
    "EnhancedFunctionTelemetry",
]
```

---

## ✅ 驗收標準

### Enums 新增

- [ ] `ErrorCategory` 枚舉已添加到 `enums/common.py`
- [ ] `StoppingReason` 枚舉已添加到 `enums/common.py`
- [ ] 兩個枚舉已導出到 `enums/__init__.py`
- [ ] Pylance 檢查通過（無語法錯誤）
- [ ] Ruff 檢查通過（符合規範）

### Telemetry Schema 擴展

- [ ] `ErrorRecord` 已添加到 `schemas/telemetry.py`
- [ ] `OastCallbackDetail` 已添加到 `schemas/telemetry.py`
- [ ] `EnhancedFunctionTelemetry` 已添加到 `schemas/telemetry.py`
- [ ] 三個類已導出到 `schemas/__init__.py`
- [ ] Pydantic 驗證正常工作
- [ ] 向後兼容性測試通過

### 文檔和測試

- [ ] 更新 API 文檔
- [ ] 創建使用範例
- [ ] 編寫單元測試（可選，但推薦）

---

## 📊 預期收益

### 短期收益（1-2 週）

1. **標準化錯誤處理**
   - 統一的錯誤分類
   - 結構化的錯誤記錄
   - 便於錯誤分析和告警

2. **增強可觀測性**
   - 詳細的 OAST 回調追蹤
   - Early Stopping 檢測
   - 自適應行為可視化

3. **提高開發效率**
   - 統一的遙測接口
   - 減少重複代碼
   - 類型安全保證

### 中期收益（1-2 月）

4. **支持高級分析**
   - 跨模組數據聚合
   - 性能趨勢分析
   - 異常檢測

5. **優化運維監控**
   - 實時性能指標
   - 自動告警觸發
   - SLA 監控支持

6. **AI/ML 增強**
   - 統計數據用於模型訓練
   - 檢測策略優化
   - 自適應參數調整

### 長期收益（3-6 月）

7. **系統成熟度提升**
   - 企業級遙測能力
   - 符合可觀測性最佳實踐
   - 支持大規模部署

8. **技術債務減少**
   - 統一的架構模式
   - 易於維護和擴展
   - 降低學習曲線

---

## 🎯 總結

### 當前狀態
- ✅ 基礎 Schema 和 Enums 完整
- ⚠️ Telemetry 需要現代化擴展
- ⚠️ Worker 遙測實現不統一
- ⚠️ 錯誤處理缺乏標準化

### 建議行動
1. **立即執行**: 新增 Enums 和 Telemetry Schema（2-3 小時）
2. **短期計劃**: 遷移 SQLi/XSS Worker（1-2 週）
3. **長期規劃**: 統一所有 Worker 遙測（1-2 月）

### ROI 評估
- **實施成本**: 低（2-3 小時初始，1-2 週完整遷移）
- **技術收益**: 高（標準化、可觀測性、可維護性）
- **業務價值**: 高（更好的監控、更快的問題診斷）
- **總體 ROI**: **95/100** ⭐

---

**報告生成時間**: 2025-10-16  
**下一步**: 執行 Schema 和 Enums 擴展（高優先級項目）
