# AIVA 模式定義重新分配計劃

**執行日期**: 2025年10月15日
**參考主文件**: schemas.py (2,411行)
**目標**: 在四大模組架構下重新分配所有定義，避免單點過載，維持單一事實來源

## 四大模組架構
```
aiva_common/    - 共享基礎設施和標準
core/          - 核心業務邏輯和編排
scan/          - 掃描和發現
function/      - 功能測試和漏洞利用
integration/   - 外部服務整合
```

## 分配策略

### 1️⃣ **aiva_common** (共享基礎層)
**職責**: 提供跨模組共享的基礎設施和官方標準
**預估行數**: ~600行

#### 保留的定義:
```python
# === 核心消息協議 (所有模組都需要) ===
- MessageHeader
- AivaMessage
- Authentication
- RateLimit

# === 官方安全標準 (CVSS, SARIF, CVE/CWE - 100%合規) ===
- CVSSv3Metrics
- CVEReference
- CWEReference
- CAPECReference
- SARIFLocation
- SARIFResult
- SARIFRule
- SARIFTool
- SARIFRun
- SARIFReport

# === 通用枚舉和基礎類型 ===
- Severity, Confidence, TestStatus 等枚舉
- 通用的 metadata, timestamp 模式
```

---

### 2️⃣ **scan** (掃描發現模組)
**職責**: 資產發現、指紋識別、初步掃描、漏洞檢測
**預估行數**: ~500行

#### 分配的定義:
```python
# === 掃描配置和控制 ===
- ScanScope
- ScanStartPayload
- EnhancedScanScope
- EnhancedScanRequest

# === 資產和指紋 ===
- Asset
- AssetInventoryItem
- TechnicalFingerprint
- Fingerprints

# === 掃描結果 ===
- Summary
- ScanCompletedPayload

# === 漏洞發現 ===
- Vulnerability
- VulnerabilityDiscovery
- VulnerabilityStatus
- VulnerabilityLifecyclePayload
- VulnerabilityUpdatePayload

# === EASM (外部攻擊面) ===
- EASMDiscoveryPayload
- DiscoveredAsset
- EASMDiscoveryResult
- EASMAsset
```

---

### 3️⃣ **function** (功能測試模組)
**職責**: 主動測試、漏洞驗證、漏洞利用、POC執行
**預估行數**: ~550行

#### 分配的定義:
```python
# === 功能測試任務 ===
- FunctionTaskTarget
- FunctionTaskContext
- FunctionTaskTestConfig
- FunctionTaskPayload
- EnhancedFunctionTaskTarget

# === 測試執行 ===
- FunctionTelemetry
- FunctionExecutionResult
- TestExecution
- ExecutionError

# === 漏洞利用 ===
- ExploitPayload
- ExploitResult
- PostExTestPayload
- PostExResultPayload

# === 專項測試 ===
- BizLogicTestPayload
- BizLogicResultPayload
- AuthZCheckPayload
- AuthZAnalysisPayload
- AuthZResultPayload

# === API 安全測試 ===
- APISchemaPayload
- APITestCase
- APISecurityTestPayload

# === OAST (帶外測試) ===
- OastEvent
- OastProbe

# === 敏感數據檢測 ===
- SensitiveMatch
- JavaScriptAnalysisResult
```

---

### 4️⃣ **integration** (整合服務模組)
**職責**: 外部服務、威脅情報、SIEM、通知、Webhook
**預估行數**: ~400行

#### 分配的定義:
```python
# === 威脅情報整合 ===
- ThreatIntelLookupPayload
- ThreatIntelResultPayload
- EnhancedIOCRecord

# === SIEM 整合 ===
- SIEMEventPayload
- SIEMEvent

# === 通知系統 ===
- NotificationPayload

# === Webhook ===
- WebhookPayload

# === 第三方服務 ===
- 各種外部API整合模式
```

---

### 5️⃣ **core** (核心業務模組)
**職責**: 風險評估、任務編排、策略生成、系統協調、AI決策
**預估行數**: ~650行

#### 分配的定義:
```python
# === 風險評估 ===
- RiskFactor
- RiskAssessmentContext
- RiskAssessmentResult
- EnhancedRiskAssessment
- RiskTrendAnalysis

# === 攻擊路徑分析 ===
- AttackPathNode
- AttackPathEdge
- AttackPathPayload
- AttackPathRecommendation
- EnhancedAttackPathNode
- EnhancedAttackPath

# === 漏洞關聯 ===
- VulnerabilityCorrelation
- EnhancedVulnerabilityCorrelation
- CodeLevelRootCause
- SASTDASTCorrelation

# === 任務管理 ===
- TaskDependency
- TaskUpdatePayload
- EnhancedTaskExecution
- TaskQueue

# === 測試策略 ===
- TestStrategy

# === 系統編排 ===
- ModuleStatus
- EnhancedModuleStatus
- SystemOrchestration
- HeartbeatPayload
- ConfigUpdatePayload

# === AI 智能系統 ===
- AIVerificationRequest
- AIVerificationResult
- AITrainingStartPayload
- AITrainingProgressPayload
- AITrainingCompletedPayload
- AIExperienceCreatedEvent
- AITraceCompletedEvent
- AIModelUpdatedEvent
- AIModelDeployCommand
- AttackStep
- AttackPlan
- TraceRecord
- PlanExecutionMetrics
- PlanExecutionResult
- ExperienceSample
- SessionState
- ModelTrainingConfig
- ModelTrainingResult
- StandardScenario
- ScenarioTestResult

# === RAG 知識庫 ===
- RAGKnowledgeUpdatePayload
- RAGQueryPayload
- RAGResponsePayload

# === AIVA 統一接口 ===
- AIVARequest
- AIVAResponse
- AIVAEvent
- AIVACommand

# === 發現和影響 ===
- Target
- FindingEvidence
- FindingImpact
- FindingRecommendation
- FindingPayload
- EnhancedVulnerability
- EnhancedFindingPayload
- FeedbackEventPayload

# === 資產和漏洞生命週期 ===
- AssetLifecyclePayload
- RemediationGeneratePayload
- RemediationResultPayload
```

---

## 實施步驟

### Phase 1: 創建新的模組化文件
1. ✅ 已完成：創建主文件和備份
2. 🔄 進行中：創建各模組的分配文件
   - `aiva_common/models.py` - 共享基礎模型
   - `scan/models.py` - 掃描模型
   - `function/models.py` - 功能測試模型
   - `integration/models.py` - 整合服務模型
   - `core/models.py` - 核心業務模型

### Phase 2: 從主文件提取分配
- 根據上述分配策略，將定義從 `schemas.py` 複製到對應模組
- 保持代碼完整性和註釋

### Phase 3: 更新導入系統
- 更新各模組的 `__init__.py`
- 建立清晰的導入路徑
- 確保向後兼容

### Phase 4: 驗證和測試
- 檢查所有導入路徑
- 運行類型檢查
- 確保無循環依賴

### Phase 5: 文檔更新
- 更新架構文檔
- 創建導入指南
- 標記棄用的導入路徑

---

## 設計原則

### ✅ DO (應該做的)
1. **單一職責**: 每個模組只負責其業務領域的定義
2. **最小依賴**: 減少跨模組依賴，優先依賴 aiva_common
3. **清晰命名**: 使用描述性名稱，避免歧義
4. **完整文檔**: 每個定義都有清晰的文檔字符串
5. **類型安全**: 使用 Pydantic 確保運行時類型安全

### ❌ DON'T (不應該做的)
1. **循環依賴**: 避免模組間的循環導入
2. **重複定義**: 不在多個地方定義相同的模式
3. **過度耦合**: 不讓模組直接依賴其他業務模組
4. **隱式依賴**: 明確聲明所有導入
5. **破壞兼容**: 保持現有代碼的兼容性

---

## 預期收益

### 📈 **可維護性提升**
- 每個文件 < 700 行，易於理解和維護
- 清晰的模組邊界，降低認知負擔
- 專注於特定業務領域

### 🚀 **開發效率**
- 快速定位相關定義
- 減少合併衝突
- 並行開發不同模組

### 🔒 **代碼質量**
- 明確的依賴關係
- 更好的類型推導
- 更容易的單元測試

### 📚 **知識管理**
- 按業務領域組織
- 新成員快速上手
- 清晰的架構文檔

---

## 下一步行動
1. 創建 `aiva_common/models.py` - 提取共享基礎
2. 創建 `scan/models.py` - 提取掃描相關
3. 創建 `function/models.py` - 提取測試相關
4. 創建 `integration/models.py` - 提取整合相關
5. 創建 `core/models.py` - 提取核心業務
6. 更新所有 `__init__.py` 的導入鏈
7. 全面測試和驗證

開始執行！🚀
