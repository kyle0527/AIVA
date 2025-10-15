# AIVA Schema 快速參考指南

> **版本**: v2.0
> **最後更新**: 2025-10-14
> **Total Schemas**: 114 個

---

## 📋 四大模組 Schema 速查

### 🧠 Core AI 模組 (29 Schemas)

#### 訓練控制

```python
AITrainingStartPayload       # tasks.ai.training.start
AITrainingStopPayload        # tasks.ai.training.stop (新)
AITrainingProgressPayload    # results.ai.training.progress
AITrainingCompletedPayload   # results.ai.training.completed
AITrainingFailedPayload      # results.ai.training.failed (新)
```

#### 事件

```python
AIExperienceCreatedEvent     # events.ai.experience.created
AITraceCompletedEvent        # events.ai.trace.completed
AIModelUpdatedEvent          # events.ai.model.updated
AIScenarioLoadedEvent        # events.ai.scenario.loaded (新)
```

#### 核心組件

```python
AttackPlan                   # 攻擊計畫
AttackStep                   # 攻擊步驟
TraceRecord                  # 執行追蹤
ExperienceSample             # 經驗樣本
ModelTrainingConfig          # 訓練配置
StandardScenario             # 標準場景
```

#### RAG

```python
RAGQueryPayload              # tasks.rag.query
RAGResponsePayload           # results.rag.response
RAGKnowledgeUpdatePayload    # tasks.rag.knowledge.update
```

---

### 🔍 Scan 模組 (10 Schemas)

#### 生命週期

```python
ScanStartPayload             # tasks.scan.start
ScanProgressPayload          # results.scan.progress (新)
ScanCompletedPayload         # results.scan.completed
ScanFailedPayload            # results.scan.failed (新)
```

#### 事件

```python
ScanAssetDiscoveredEvent     # events.scan.asset.discovered (新)
```

#### 數據模型

```python
ScanScope                    # 掃描範圍
Asset                        # 資產信息
Fingerprints                 # 指紋信息
AssetLifecyclePayload        # 資產生命週期
```

---

### ⚙️ Function 模組 (11 Schemas)

#### 生命週期

```python
FunctionTaskPayload          # tasks.function.*
FunctionTaskProgressPayload  # results.function.progress (新)
FunctionTaskCompletedPayload # results.function.completed (新)
FunctionTaskFailedPayload    # results.function.failed (新)
```

#### 事件

```python
FeedbackEventPayload         # feedback.core.strategy
FunctionVulnFoundEvent       # events.function.vuln.found (新)
```

#### 配置

```python
FunctionTaskTarget           # 任務目標
FunctionTaskContext          # 任務上下文
FunctionTaskTestConfig       # 測試配置
FunctionTelemetry            # 遙測數據
```

---

### 🔗 Integration 模組 (44 Schemas)

#### 分析流程

```python
IntegrationAnalysisStartPayload      # tasks.integration.analysis.start (新)
IntegrationAnalysisProgressPayload   # results.integration.analysis.progress (新)
IntegrationAnalysisCompletedPayload  # results.integration.analysis.completed (新)
```

#### 報告生成

```python
IntegrationReportGenerateCommand     # commands.integration.report.generate (新)
IntegrationReportGeneratedEvent      # events.integration.report.generated (新)
```

#### 漏洞分析

```python
FindingPayload               # 漏洞發現
EnhancedFindingPayload       # 增強漏洞信息
Vulnerability                # 漏洞詳情
VulnerabilityCorrelation     # 漏洞相關性
```

#### 攻擊路徑

```python
AttackPathPayload            # 攻擊路徑
AttackPathNode               # 路徑節點
AttackPathEdge               # 路徑邊
AttackPathRecommendation     # 路徑建議
```

#### 風險評估

```python
RiskAssessmentContext        # 風險評估上下文
RiskAssessmentResult         # 風險評估結果
RiskTrendAnalysis            # 風險趨勢
```

#### 其他

```python
SARIFReport                  # SARIF 報告
CVSSv3Metrics                # CVSS 評分
ThreatIntelLookupPayload     # 威脅情報查詢
RemediationGeneratePayload   # 修復建議生成
```

---

## 🎯 命名規範速查

### Payload 命名

```
格式: <Module><Action>Payload

示例:
✅ ScanStartPayload
✅ FunctionTaskProgressPayload
✅ AITrainingCompletedPayload
✅ IntegrationAnalysisStartPayload
```

### Event 命名

```
格式: <Module><EventName>Event

示例:
✅ AIExperienceCreatedEvent
✅ ScanAssetDiscoveredEvent
✅ FunctionVulnFoundEvent
✅ IntegrationReportGeneratedEvent
```

### Command 命名

```
格式: <Module><CommandName>Command

示例:
✅ AIModelDeployCommand
✅ IntegrationReportGenerateCommand
```

### Request/Response 命名

```
格式: <Module><Action>Request / <Module><Action>Response
或: <Module><Action>Payload / <Module><Action>ResultPayload

示例:
✅ RAGQueryPayload / RAGResponsePayload
✅ AIVerificationRequest / AIVerificationResult
```

---

## 📡 Topic 路由速查

### Core AI Topics

```python
tasks.ai.training.start
tasks.ai.training.stop
tasks.ai.training.episode
results.ai.training.progress
results.ai.training.completed
results.ai.training.failed

events.ai.experience.created
events.ai.trace.completed
events.ai.model.updated
events.ai.scenario.loaded

commands.ai.model.deploy

tasks.rag.query
tasks.rag.knowledge.update
results.rag.response
```

### Scan Topics

```python
tasks.scan.start
results.scan.progress        # 新
results.scan.completed
results.scan.failed          # 新

events.scan.asset.discovered # 新
```

### Function Topics

```python
tasks.function.start
tasks.function.xss
tasks.function.sqli
tasks.function.ssrf
tasks.function.idor

results.function.progress    # 新
results.function.completed
results.function.failed      # 新

events.function.vuln.found   # 新
```

### Integration Topics

```python
tasks.integration.analysis.start           # 新
results.integration.analysis.progress      # 新
results.integration.analysis.completed     # 新

commands.integration.report.generate       # 新
events.integration.report.generated        # 新

findings.detected
```

### 通用 Topics

```python
log.results.all
status.task.update
feedback.core.strategy
module.heartbeat
command.task.cancel
config.global.update
```

---

## 🔄 典型消息流程

### 1. 掃描流程

```
Core → Scan:  tasks.scan.start (ScanStartPayload)
Scan → Core:  results.scan.progress (ScanProgressPayload) [新]
Scan → All:   events.scan.asset.discovered (ScanAssetDiscoveredEvent) [新]
Scan → Core:  results.scan.completed (ScanCompletedPayload)
或
Scan → Core:  results.scan.failed (ScanFailedPayload) [新]
```

### 2. 功能測試流程

```
Core → Function:  tasks.function.* (FunctionTaskPayload)
Function → Core:  results.function.progress (FunctionTaskProgressPayload) [新]
Function → All:   events.function.vuln.found (FunctionVulnFoundEvent) [新]
Function → Core:  results.function.completed (FunctionTaskCompletedPayload) [新]
或
Function → Core:  results.function.failed (FunctionTaskFailedPayload) [新]
```

### 3. AI 訓練流程

```
UI/Orchestrator → Core:  tasks.ai.training.start (AITrainingStartPayload)
Core → UI:               results.ai.training.progress (AITrainingProgressPayload)
Core → Storage:          events.ai.experience.created (AIExperienceCreatedEvent)
Core → Storage:          events.ai.trace.completed (AITraceCompletedEvent)
Core → UI:               results.ai.training.completed (AITrainingCompletedPayload)
或
UI/Orchestrator → Core:  tasks.ai.training.stop (AITrainingStopPayload) [新]
或
Core → UI:               results.ai.training.failed (AITrainingFailedPayload) [新]
```

### 4. 整合分析流程

```
Core → Integration:  tasks.integration.analysis.start (IntegrationAnalysisStartPayload) [新]
Integration → Core:  results.integration.analysis.progress (IntegrationAnalysisProgressPayload) [新]
Integration → Core:  results.integration.analysis.completed (IntegrationAnalysisCompletedPayload) [新]
```

### 5. 報告生成流程

```
UI → Integration:  commands.integration.report.generate (IntegrationReportGenerateCommand) [新]
Integration → UI:  events.integration.report.generated (IntegrationReportGeneratedEvent) [新]
```

---

## 💡 使用示例

### 發送掃描進度

```python
from aiva_common.schemas import ScanProgressPayload
from aiva_common.enums import Topic

progress = ScanProgressPayload(
    scan_id="scan_123",
    progress_percentage=45.5,
    current_target="https://example.com/api",
    assets_discovered=12,
    vulnerabilities_found=3,
    estimated_time_remaining_seconds=300
)

await broker.publish(
    topic=Topic.RESULTS_SCAN_PROGRESS,
    payload=progress
)
```

### 發送漏洞發現事件

```python
from aiva_common.schemas import FunctionVulnFoundEvent

event = FunctionVulnFoundEvent(
    task_id="task_456",
    scan_id="scan_123",
    vulnerability=vuln_obj,
    confidence=Confidence.FIRM,
    severity=Severity.HIGH,
    test_type="xss",
    evidence=evidence_obj
)

await broker.publish(
    topic=Topic.EVENT_FUNCTION_VULN_FOUND,
    payload=event
)
```

### 啟動整合分析

```python
from aiva_common.schemas import IntegrationAnalysisStartPayload

analysis = IntegrationAnalysisStartPayload(
    analysis_id="analysis_789",
    scan_id="scan_123",
    analysis_types=["correlation", "attack_path", "risk_assessment"],
    findings=[finding1, finding2, finding3],
    context={"environment": "production"}
)

await broker.publish(
    topic=Topic.TASK_INTEGRATION_ANALYSIS_START,
    payload=analysis
)
```

---

## 📊 統計總結

| 分類 | 數量 |
|------|------|
| **總 Schemas** | 114 |
| **Core AI** | 29 |
| **Scan** | 10 |
| **Function** | 11 |
| **Integration** | 44 |
| **Shared** | 20 |
| **總 Topics** | 50+ |

---

## 📚 相關文檔

- [完整報告](./SCHEMA_COMPLETION_REPORT.md)
- [統一計畫](./SCHEMA_UNIFICATION_PLAN.md)
- [通訊合約](./MODULE_COMMUNICATION_CONTRACTS.md)
- [架構圖集](./COMPLETE_ARCHITECTURE_DIAGRAMS.md)
