# AIVA 模式重新分配進度追蹤

**開始時間**: 2025年10月15日
**目標**: 將2,411行的統一schema文件重新分配到四大模組

## 進度概覽

```
總進度: ████░░░░░░ 20% (1/5完成)
```

## 模組分配狀態

### ✅ 1. aiva_common (共享基礎層) - 已完成
- **文件**: `services/aiva_common/models.py`
- **行數**: 248行
- **狀態**: ✅ 完成
- **包含內容**:
  - ✅ MessageHeader, AivaMessage (核心消息協議)
  - ✅ Authentication, RateLimit (通用認證控制)
  - ✅ CVSSv3Metrics (CVSS v3.1 完整實現)
  - ✅ CVEReference, CWEReference, CAPECReference (官方標準)
  - ✅ SARIF 完整格式 (Location, Result, Rule, Tool, Run, Report)

### 🔄 2. scan (掃描發現模組) - 進行中
- **文件**: `services/scan/models.py`
- **預估行數**: ~500行
- **狀態**: ⏳ 待開始
- **規劃內容**:
  - ScanScope, ScanStartPayload, EnhancedScanScope, EnhancedScanRequest
  - Asset, AssetInventoryItem, TechnicalFingerprint, Fingerprints
  - Summary, ScanCompletedPayload
  - Vulnerability, VulnerabilityDiscovery, VulnerabilityLifecyclePayload
  - EASM相關: EASMDiscoveryPayload, DiscoveredAsset, EASMAsset

### ⏳ 3. function (功能測試模組) - 待開始
- **文件**: `services/function/models.py`
- **預估行數**: ~550行
- **狀態**: ⏳ 待開始
- **規劃內容**:
  - FunctionTaskTarget, FunctionTaskContext, FunctionTaskTestConfig, FunctionTaskPayload
  - FunctionTelemetry, FunctionExecutionResult, TestExecution
  - ExploitPayload, ExploitResult
  - PostExTestPayload, PostExResultPayload
  - API測試: APISchemaPayload, APITestCase, APISecurityTestPayload
  - OAST: OastEvent, OastProbe
  - 專項測試: BizLogicTestPayload, AuthZCheckPayload等

### ⏳ 4. integration (整合服務模組) - 待開始
- **文件**: `services/integration/models.py`
- **預估行數**: ~400行
- **狀態**: ⏳ 待開始
- **規劃內容**:
  - 威脅情報: ThreatIntelLookupPayload, ThreatIntelResultPayload, EnhancedIOCRecord
  - SIEM: SIEMEventPayload, SIEMEvent
  - 通知: NotificationPayload
  - Webhook: WebhookPayload

### ⏳ 5. core (核心業務模組) - 待開始
- **文件**: `services/core/models.py`
- **預估行數**: ~650行
- **狀態**: ⏳ 待開始
- **規劃內容**:
  - 風險評估: RiskFactor, RiskAssessmentContext, EnhancedRiskAssessment, RiskTrendAnalysis
  - 攻擊路徑: AttackPathNode, AttackPathEdge, EnhancedAttackPath
  - 漏洞關聯: VulnerabilityCorrelation, CodeLevelRootCause, SASTDASTCorrelation
  - 任務管理: TaskDependency, TaskUpdatePayload, EnhancedTaskExecution, TaskQueue
  - 測試策略: TestStrategy
  - 系統編排: ModuleStatus, EnhancedModuleStatus, SystemOrchestration
  - AI系統: 訓練、RAG、AIVA接口等
  - 發現和影響: Target, FindingEvidence, FindingImpact, FindingPayload

## 行數分配統計

| 模組 | 狀態 | 實際行數 | 預估行數 | 偏差 |
|------|------|----------|----------|------|
| aiva_common | ✅ | 248 | ~600 | -352 (更精簡) |
| scan | ⏳ | - | ~500 | - |
| function | ⏳ | - | ~550 | - |
| integration | ⏳ | - | ~400 | - |
| core | ⏳ | - | ~650 | - |
| **總計** | 20% | 248 | ~2700 | - |

## 下一步行動

1. ✅ 創建 `aiva_common/models.py` (248行)
2. 🔜 創建 `scan/models.py` - **下一個目標**
3. ⏳ 創建 `function/models.py`
4. ⏳ 創建 `integration/models.py`
5. ⏳ 創建 `core/models.py`
6. ⏳ 更新所有 `__init__.py` 的導入
7. ⏳ 全面測試和驗證
8. ⏳ 更新文檔

## 設計決策記錄

### aiva_common (已完成)
- **決策**: 只保留真正跨模組共享的基礎設施
- **理由**: 避免成為垃圾桶，保持職責清晰
- **結果**: 比預期精簡 (248行 vs 600行預估)
- **包含**:
  - 核心消息協議 (所有模組間通信必需)
  - 官方安全標準 (CVSS, SARIF, CVE/CWE - 100%合規)
  - 通用認證控制 (跨模組使用)

## 品質指標

- ✅ 類型安全: 使用 Pydantic BaseModel
- ✅ 文檔完整: 每個類都有docstring
- ✅ 標準合規: CVSS v3.1, SARIF v2.1.0 完整實現
- ✅ 無循環依賴: 只依賴 enums
- ✅ 清晰命名: 描述性類名和欄位名

## 備註
- 原始統一文件 `schemas.py`: 2,411行 (已備份2份)
- 備份文件: `schemas_master_backup_1.py`, `schemas_master_backup_2.py`
- 新架構更精簡、更專注於業務領域分離

---
**最後更新**: 2025年10月15日 - aiva_common 完成
