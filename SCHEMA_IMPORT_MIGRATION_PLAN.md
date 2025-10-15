# Schema 導入路徑遷移計劃

## 概述

根據新的模組化 schema 結構，需要更新所有使用舊 `schemas.py` 導入的文件。

## 新的 Schema 分佈

### 1. aiva_common/models.py (248 行)
**包含內容**:
- `MessageHeader`, `AivaMessage` - 消息系統
- `TokenInfo`, `UserInfo`, `AuthenticationPayload`, `RateLimitConfig` - 認證授權
- `CVSSv3Metrics` - CVSS v3.1 評分
- `SARIFLocation`, `SARIFRegion`, `SARIFResult`, `SARIFRun`, `SARIFReport` - SARIF v2.1.0
- `CVEReference`, `CWEReference`, `CAPECReference` - 標準引用

### 2. scan/models.py (338 行)
**包含內容**:
- `ScanScope`, `ScanStartPayload` - 掃描控制
- `Asset`, `TechnicalFingerprint` - 資產管理
- `Vulnerability`, `VulnerabilityDiscovery`, `VulnerabilityReportPayload` - 漏洞發現
- `EASMAsset`, `EASMScanPayload`, `EASMUpdatePayload` - EASM集成
- `AssetInventoryPayload`, `AssetLifecyclePayload` - 生命週期

### 3. function/models.py (368 行)
**包含內容**:
- `FunctionTaskPayload` - 任務管理
- `TestExecution`, `TestExecutionDetail` - 測試執行
- `ExploitPayload`, `ExploitResult`, `PostExTestPayload` - 漏洞利用
- `APISecurityTestPayload`, `APITestResult` - API安全
- `OastEvent`, `OastProbe`, `OastDetectedEvent` - OAST檢測
- `BizLogicTestPayload`, `BizLogicFindingPayload` - 業務邏輯
- `AuthZCheckPayload`, `AuthZTestResult` - 授權測試
- `SensitiveDataTestPayload`, `SensitiveMatch` - 敏感數據

### 4. integration/models.py (143 行)
**包含內容**:
- `ThreatIntelLookupPayload`, `ThreatIntelResultPayload` - 威脅情報
- `EnhancedIOCRecord` - IOC記錄
- `SIEMEventPayload`, `SIEMEvent` - SIEM集成
- `NotificationPayload` - 通知系統
- `WebhookPayload` - Webhook

### 5. core/models.py (522 行)
**包含內容**:
- `Target`, `FindingEvidence`, `FindingImpact`, `FindingRecommendation`, `FindingPayload` - 發現管理
- `EnhancedVulnerability`, `EnhancedFindingPayload`, `FeedbackEventPayload` - 增強發現
- `RiskFactor`, `RiskAssessmentContext`, `RiskAssessmentResult`, `EnhancedRiskAssessment` - 風險評估
- `RiskTrendAnalysis` - 風險趨勢
- `AttackPathNode`, `AttackPathEdge`, `AttackPathPayload`, `AttackPathRecommendation` - 攻擊路徑
- `EnhancedAttackPathNode`, `EnhancedAttackPath` - 增強攻擊路徑
- `VulnerabilityCorrelation`, `EnhancedVulnerabilityCorrelation` - 漏洞關聯
- `CodeLevelRootCause`, `SASTDASTCorrelation` - 根本原因分析
- `TaskUpdatePayload`, `TaskDependency`, `EnhancedTaskExecution`, `TaskQueue` - 任務管理
- `TestStrategy` - 測試策略
- `ModuleStatus`, `EnhancedModuleStatus`, `HeartbeatPayload` - 系統協調
- `ConfigUpdatePayload`, `SystemOrchestration` - 系統編排
- `RemediationGeneratePayload`, `RemediationResultPayload` - 修復建議

### 6. core/ai_models.py (391 行)
**包含內容**:
- `AIVerificationRequest`, `AIVerificationResult` - AI驗證
- `AITrainingStartPayload`, `AITrainingProgressPayload`, `AITrainingCompletedPayload` - AI訓練
- `AIExperienceCreatedEvent`, `AITraceCompletedEvent`, `AIModelUpdatedEvent` - AI事件
- `AIModelDeployCommand` - AI命令
- `AttackStep`, `AttackPlan`, `PlanExecutionMetrics`, `PlanExecutionResult` - 攻擊規劃
- `TraceRecord` - 追蹤記錄
- `ExperienceSample`, `SessionState` - 經驗學習
- `ModelTrainingConfig`, `ModelTrainingResult` - 模型管理
- `StandardScenario`, `ScenarioTestResult` - 場景測試
- `RAGKnowledgeUpdatePayload`, `RAGQueryPayload`, `RAGResponsePayload` - RAG知識庫
- `AIVARequest`, `AIVAResponse`, `AIVAEvent`, `AIVACommand` - AIVA核心

---

## 需要更新的文件清單

### Scan 模組文件

#### 1. `services/scan/aiva_scan/worker_refactored.py`
**當前導入**:
```python
from services.aiva_common.schemas import (
    Asset,
    ScanScope,
    Vulnerability,
    # ...
)
```

**需要改為**:
```python
from services.scan.models import (
    Asset,
    ScanScope,
    Vulnerability,
)
```

#### 2. `services/scan/aiva_scan/worker.py`
**當前導入**:
```python
from services.aiva_common.schemas import (
    Asset,
    ScanResult,
    # ...
)
```

**需要改為**:
```python
from services.scan.models import Asset
# ScanResult 可能在舊 schemas 中，需要檢查是否已遷移
```

#### 3. `services/scan/aiva_scan/sensitive_data_scanner.py`
**當前導入**:
```python
from services.aiva_common.schemas import SensitiveMatch
```

**需要改為**:
```python
from services.function.models import SensitiveMatch
```

#### 4. `services/scan/aiva_scan/scope_manager.py`
**當前導入**:
```python
from services.aiva_common.schemas import ScanScope
```

**需要改為**:
```python
from services.scan.models import ScanScope
```

#### 5. `services/scan/aiva_scan/scan_orchestrator.py`
**當前導入**:
```python
from services.aiva_common.schemas import (
    ScanScope,
    ScanStartPayload,
    # ...
)
```

**需要改為**:
```python
from services.scan.models import ScanScope, ScanStartPayload
```

#### 6. `services/scan/aiva_scan/scan_context.py`
**當前導入**:
```python
from services.aiva_common.schemas import (
    Asset,
    ScanScope,
    # ...
)
```

**需要改為**:
```python
from services.scan.models import Asset, ScanScope
```

#### 7. `services/scan/aiva_scan/javascript_analyzer.py`
**當前**: `from services.aiva_common.schemas import JavaScriptAnalysisResult`
**問題**: 這個類可能不在新 models 中，需要檢查

#### 8. `services/scan/aiva_scan/info_gatherer/passive_fingerprinter.py`
**當前**: `from services.aiva_common.schemas import Fingerprints`
**問題**: 這個類可能不在新 models 中，需要檢查

#### 9. `services/scan/aiva_scan/fingerprint_manager.py`
**當前**: `from services.aiva_common.schemas import Fingerprints`
**問題**: 這個類可能不在新 models 中，需要檢查

#### 10. `services/scan/aiva_scan/dynamic_engine/dynamic_content_extractor.py`
**當前**: `from services.aiva_common.schemas import Asset`
**需要改為**: `from services.scan.models import Asset`

#### 11. `services/scan/aiva_scan/dynamic_engine/ajax_api_handler.py`
**當前**: `from services.aiva_common.schemas import Asset`
**需要改為**: `from services.scan.models import Asset`

#### 12. `services/scan/aiva_scan/core_crawling_engine/static_content_parser.py`
**當前**: `from services.aiva_common.schemas import Asset`
**需要改為**: `from services.scan.models import Asset`

#### 13. `services/scan/aiva_scan/authentication_manager.py`
**當前**: `from services.aiva_common.schemas import Authentication`
**問題**: 這個類可能不在新 models 中，需要檢查

---

### Function 模組文件

#### 14. `services/function/function_xss/aiva_func_xss/task_queue.py`
**當前**: `from services.aiva_common.schemas import FunctionTaskPayload`
**需要改為**: `from services.function.models import FunctionTaskPayload`

#### 15. `services/function/function_xss/aiva_func_xss/worker.py`
**當前導入**:
```python
from services.aiva_common.schemas import (
    FunctionTaskPayload,
    ExploitResult,
    # ...
)
```

**需要改為**:
```python
from services.function.models import FunctionTaskPayload, ExploitResult
```

#### 16. `services/function/function_xss/aiva_func_xss/traditional_detector.py`
**當前**: `from services.aiva_common.schemas import FunctionTaskPayload`
**需要改為**: `from services.function.models import FunctionTaskPayload`

#### 17. `services/function/function_xss/aiva_func_xss/stored_detector.py`
**當前**: `from services.aiva_common.schemas import FunctionTaskPayload`
**需要改為**: `from services.function.models import FunctionTaskPayload`

#### 18. `services/function/function_xss/aiva_func_xss/schemas.py`
**當前導入**:
```python
from services.aiva_common.schemas import (
    FunctionTaskPayload,
    CVSSv3Metrics,
    # ...
)
```

**需要改為**:
```python
from services.function.models import FunctionTaskPayload
from services.aiva_common.models import CVSSv3Metrics
```

#### 19. `services/function/function_xss/aiva_func_xss/result_publisher.py`
**當前導入**: 需要檢查具體內容
**需要改為**: 根據實際使用的類決定

#### 20. `services/function/function_xss/aiva_func_xss/blind_xss_listener_validator.py`
**當前**: `from services.aiva_common.schemas import FunctionTaskPayload`
**需要改為**: `from services.function.models import FunctionTaskPayload`

---

#### 21-30. `services/function/function_ssrf/` 相關文件
同樣需要將 `FunctionTaskPayload`, `ExploitResult` 等改為從 `services.function.models` 導入

#### 31-40. `services/function/function_sqli/` 相關文件
同樣需要將 `FunctionTaskPayload`, `ExploitResult` 等改為從 `services.function.models` 導入

---

### Integration 模組文件

#### `services/integration/api_gateway/api_gateway/app.py`
**當前**:
```python
from services.aiva_common.schemas import AivaMessage, MessageHeader, ScanStartPayload
```

**需要改為**:
```python
from services.aiva_common.models import AivaMessage, MessageHeader
from services.scan.models import ScanStartPayload
```

---

## 遷移策略

### 階段 1: 創建兼容層 (推薦先執行)

在 `services/aiva_common/schemas.py` 中添加重新導出：

```python
"""
向後兼容層 - 重新導出所有 models
此文件將逐步棄用，請使用各模組的 models.py
"""

# 從 aiva_common.models 重新導出
from .models import (
    AivaMessage,
    AuthenticationPayload,
    CAPECReference,
    CVEReference,
    CVSSv3Metrics,
    CWEReference,
    MessageHeader,
    RateLimitConfig,
    SARIFLocation,
    SARIFRegion,
    SARIFReport,
    SARIFResult,
    SARIFRun,
    TokenInfo,
    UserInfo,
)

# 從 scan.models 重新導出
from services.scan.models import (
    Asset,
    AssetInventoryPayload,
    AssetLifecyclePayload,
    EASMAsset,
    EASMScanPayload,
    EASMUpdatePayload,
    ScanScope,
    ScanStartPayload,
    TechnicalFingerprint,
    Vulnerability,
    VulnerabilityDiscovery,
    VulnerabilityReportPayload,
)

# 從 function.models 重新導出
from services.function.models import (
    APISecurityTestPayload,
    APITestResult,
    AuthZCheckPayload,
    AuthZTestResult,
    BizLogicFindingPayload,
    BizLogicTestPayload,
    ExploitPayload,
    ExploitResult,
    FunctionTaskPayload,
    OastDetectedEvent,
    OastEvent,
    OastProbe,
    PostExTestPayload,
    SensitiveDataTestPayload,
    SensitiveMatch,
    TestExecution,
    TestExecutionDetail,
)

# 從 integration.models 重新導出
from services.integration.models import (
    EnhancedIOCRecord,
    NotificationPayload,
    SIEMEvent,
    SIEMEventPayload,
    ThreatIntelLookupPayload,
    ThreatIntelResultPayload,
    WebhookPayload,
)

# 從 core.models 重新導出
from services.core.models import (
    AttackPathEdge,
    AttackPathNode,
    AttackPathPayload,
    AttackPathRecommendation,
    CodeLevelRootCause,
    ConfigUpdatePayload,
    EnhancedAttackPath,
    EnhancedAttackPathNode,
    EnhancedFindingPayload,
    EnhancedModuleStatus,
    EnhancedRiskAssessment,
    EnhancedTaskExecution,
    EnhancedVulnerability,
    EnhancedVulnerabilityCorrelation,
    FeedbackEventPayload,
    FindingEvidence,
    FindingImpact,
    FindingPayload,
    FindingRecommendation,
    HeartbeatPayload,
    ModuleStatus,
    RemediationGeneratePayload,
    RemediationResultPayload,
    RiskAssessmentContext,
    RiskAssessmentResult,
    RiskFactor,
    RiskTrendAnalysis,
    SASTDASTCorrelation,
    SystemOrchestration,
    Target,
    TaskDependency,
    TaskQueue,
    TaskUpdatePayload,
    TestStrategy,
    VulnerabilityCorrelation,
)

# 從 core.ai_models 重新導出
from services.core.ai_models import (
    AIExperienceCreatedEvent,
    AIModelDeployCommand,
    AIModelUpdatedEvent,
    AITraceCompletedEvent,
    AITrainingCompletedPayload,
    AITrainingProgressPayload,
    AITrainingStartPayload,
    AIVACommand,
    AIVAEvent,
    AIVARequest,
    AIVAResponse,
    AIVerificationRequest,
    AIVerificationResult,
    AttackPlan,
    AttackStep,
    ExperienceSample,
    ModelTrainingConfig,
    ModelTrainingResult,
    PlanExecutionMetrics,
    PlanExecutionResult,
    RAGKnowledgeUpdatePayload,
    RAGQueryPayload,
    RAGResponsePayload,
    ScenarioTestResult,
    SessionState,
    StandardScenario,
    TraceRecord,
)

__all__ = [
    # ... 列出所有重新導出的類
]
```

**優點**:
- 現有代碼無需修改，立即可用
- 可以逐步遷移
- 降低風險

**缺點**:
- 需要維護兼容層
- 可能隱藏循環依賴問題

---

### 階段 2: 逐步遷移

1. **優先遷移 Scan 模組** (影響範圍較小)
2. **然後遷移 Function 模組** (可能影響多個子功能)
3. **最後遷移 Integration 和 Core** (影響範圍最大)

---

### 階段 3: 清理和驗證

1. 移除兼容層
2. 運行完整測試套件
3. 更新文檔

---

## 缺失的類別檢查

需要檢查以下類別是否在新 models 中:

- [ ] `JavaScriptAnalysisResult` (scan 使用)
- [ ] `Fingerprints` (scan 使用)
- [ ] `Authentication` (scan 使用)
- [ ] `ScanResult` (scan 使用)
- [ ] `BaseModel` (多處使用)
- [ ] `AssetInventory` (可能使用)
- [ ] `ConfigurationData` (可能使用)
- [ ] `IOCRecord` (可能使用)
- [ ] `RiskAssessment` (可能使用)
- [ ] `SystemStatus` (可能使用)
- [ ] `TargetInfo` (可能使用)
- [ ] `TechStackInfo` (可能使用)
- [ ] `ServiceInfo` (可能使用)
- [ ] `ScopeDefinition` (可能使用)
- [ ] `TestResult` (可能使用)
- [ ] `ThreatIndicator` (可能使用)
- [ ] `VulnerabilityFinding` (多處使用)

---

## 行動計劃

### 立即執行

1. ✅ 創建此遷移計劃文檔
2. ⏳ 檢查缺失的類別
3. ⏳ 創建兼容層 (schemas.py 重新導出)
4. ⏳ 測試兼容層是否正常工作

### 短期執行 (1-2 天)

5. ⏳ 逐步更新 Scan 模組文件
6. ⏳ 逐步更新 Function 模組文件
7. ⏳ 更新 Integration 模組文件

### 中期執行 (1 週)

8. ⏳ 更新所有測試文件
9. ⏳ 運行完整測試套件
10. ⏳ 修復發現的問題

### 長期執行 (可選)

11. ⏳ 移除兼容層
12. ⏳ 更新所有文檔
13. ⏳ 添加 deprecation warnings

---

## 測試檢查清單

- [ ] 所有 Scan 模組功能正常
- [ ] 所有 Function 模組功能正常
- [ ] Integration API Gateway 正常
- [ ] Core 模組功能正常
- [ ] 消息傳遞正常
- [ ] 數據序列化/反序列化正常
- [ ] 無循環依賴錯誤
- [ ] 無導入錯誤

---

**創建日期**: 2025-10-15
**狀態**: 待執行
**優先級**: 高
