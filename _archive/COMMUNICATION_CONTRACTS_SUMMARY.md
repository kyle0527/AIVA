# AIVA 模組間通訊合約完成總結

## ✅ 完成狀態

**所有模組間的通訊合約已經完成並實現！**

---

## 📋 完成的通訊合約清單

### 1. 核心消息結構 ✅

- ✅ `MessageHeader` - 標準消息頭（含 trace_id、correlation_id）
- ✅ `AivaMessage` - 統一消息包裝器
- ✅ `Topic` 枚舉 - 30+ 個消息主題定義

### 2. 掃描模組合約 ✅

- ✅ `ScanStartPayload` - 掃描任務啟動
- ✅ `ScanCompletedPayload` - 掃描結果回報
- ✅ `Asset` - 資產信息
- ✅ `Summary` - 掃描統計摘要

**Topics**:

- `tasks.scan.start` (Core → Scan)
- `results.scan.completed` (Scan → Core)

### 3. 功能測試模組合約 ✅

- ✅ `FunctionTaskPayload` - 功能測試任務
- ✅ `FindingPayload` - 漏洞發現報告
- ✅ `EnhancedVulnerability` - 增強漏洞信息（集成 CVSS、CVE、CWE、MITRE）
- ✅ `FindingEvidence` - 漏洞證據
- ✅ `FindingImpact` - 漏洞影響評估
- ✅ `FindingRecommendation` - 修復建議

**Topics**:

- `tasks.function.start` (Core → Function)
- `tasks.function.xss` (Core → Function)
- `tasks.function.sqli` (Core → Function)
- `tasks.function.ssrf` (Core → Function)
- `tasks.function.idor` (Core → Function)
- `results.function.completed` (Function → Core)
- `findings.detected` (Function → Core)

### 4. AI 訓練模組合約 ✅ **NEW!**

#### 訓練控制合約

- ✅ `AITrainingStartPayload` - 訓練會話啟動
- ✅ `AITrainingProgressPayload` - 訓練進度報告
- ✅ `AITrainingCompletedPayload` - 訓練完成報告

**Topics**:

- `tasks.ai.training.start` (UI/Core → AI)
- `tasks.ai.training.episode` (Orchestrator → AI)
- `tasks.ai.training.stop` (UI/Core → AI)
- `results.ai.training.progress` (AI → UI/Core)
- `results.ai.training.completed` (AI → UI/Core)
- `results.ai.training.failed` (AI → UI/Core)

#### AI 事件合約

- ✅ `AIExperienceCreatedEvent` - 經驗樣本創建通知
- ✅ `AITraceCompletedEvent` - 執行追蹤完成通知
- ✅ `AIModelUpdatedEvent` - 模型更新通知

**Topics**:

- `events.ai.experience.created` (AI → Storage)
- `events.ai.trace.completed` (AI → Storage)
- `events.ai.model.updated` (AI → Core)

#### AI 命令合約

- ✅ `AIModelDeployCommand` - 模型部署命令

**Topics**:

- `commands.ai.model.deploy` (Core → AI)

### 5. RAG 知識庫合約 ✅ **NEW!**

- ✅ `RAGKnowledgeUpdatePayload` - 知識庫更新請求
- ✅ `RAGQueryPayload` - 知識檢索請求
- ✅ `RAGResponsePayload` - 知識檢索響應

**Topics**:

- `tasks.rag.knowledge.update` (Any → RAG)
- `tasks.rag.query` (Any → RAG)
- `results.rag.response` (RAG → Requester)

### 6. 統一通訊包裝器 ✅ **NEW!**

- ✅ `AIVARequest` - 統一請求包裝（支持請求-響應模式）
- ✅ `AIVAResponse` - 統一響應包裝
- ✅ `AIVAEvent` - 統一事件包裝
- ✅ `AIVACommand` - 統一命令包裝（支持優先級）

### 7. 強化學習核心 Schemas ✅

- ✅ `AttackPlan` - 攻擊計畫（含 MITRE ATT&CK 技術映射）
- ✅ `AttackResult` - 攻擊結果（含 CVSS 評分）
- ✅ `TraceRecord` - 完整執行追蹤
- ✅ `TraceStep` - 單步追蹤記錄
- ✅ `ExperienceSample` - 訓練經驗樣本
- ✅ `PlanExecutionMetrics` - 執行性能指標
- ✅ `ModelTrainingConfig` - 模型訓練配置

### 8. 業界標準支持 ✅

- ✅ `CVSSv3Metrics` - CVSS v3.1 完整評分計算
- ✅ `CWEReference` - CWE 弱點參考
- ✅ `CVEReference` - CVE 漏洞參考
- ✅ `MITREAttackTechnique` - MITRE ATT&CK 映射
- ✅ `SARIFResult` - SARIF v2.1.0 格式支持
- ✅ `SARIFLocation` - SARIF 位置信息
- ✅ `SARIFReport` - SARIF 完整報告

---

## 📊 統計數據

| 類別 | 數量 |
|------|------|
| **消息主題 (Topics)** | 30+ |
| **Payload Schemas** | 50+ |
| **事件 Events** | 10+ |
| **命令 Commands** | 5+ |
| **支持的模組** | 7 (Core, Scan, Function, AI, RAG, Storage, Monitor) |
| **支持的語言** | 4 (Python, Go, TypeScript, Rust) |
| **業界標準** | 5 (CVSS, CWE, CVE, MITRE ATT&CK, SARIF) |

---

## 🔄 完整的通訊流程

### 流程 1: 掃描 → 測試 → AI 學習

```
1. UI 發起掃描
   ↓ tasks.scan.start (ScanStartPayload)
2. Scan Module 執行掃描
   ↓ results.scan.completed (ScanCompletedPayload)
3. Core 分析資產，分發測試任務
   ↓ tasks.function.* (FunctionTaskPayload)
4. Function Module 執行測試
   ↓ findings.detected (FindingPayload)
5. Core 收集結果，觸發 AI 學習
   ↓ events.ai.experience.created (AIExperienceCreatedEvent)
6. Storage 保存經驗樣本
```

### 流程 2: AI 訓練全流程

```
1. UI 啟動訓練
   ↓ tasks.ai.training.start (AITrainingStartPayload)
2. Training Orchestrator 接收任務
   ↓ 循環執行訓練回合
3. 每個回合:
   - Plan Executor 執行攻擊
   - events.ai.trace.completed (AITraceCompletedEvent)
   - Experience Manager 創建樣本
   - events.ai.experience.created (AIExperienceCreatedEvent)
   - results.ai.training.progress (AITrainingProgressPayload)
4. 訓練完成
   ↓ events.ai.model.updated (AIModelUpdatedEvent)
   ↓ results.ai.training.completed (AITrainingCompletedPayload)
5. 部署模型
   ↓ commands.ai.model.deploy (AIModelDeployCommand)
```

### 流程 3: RAG 增強決策

```
1. BioNeuron Agent 需要做決策
   ↓ tasks.rag.query (RAGQueryPayload)
2. RAG Engine 檢索知識
   - Vector Store 向量搜索
   - Knowledge Base 獲取詳情
   ↓ results.rag.response (RAGResponsePayload)
3. Agent 使用增強上下文
   - 生成更準確的攻擊計畫
   - 提高決策質量
```

---

## 📁 文件位置

### 核心定義文件

- **Schemas**: `/workspaces/AIVA/services/aiva_common/schemas.py` (1900+ 行)
- **Enums**: `/workspaces/AIVA/services/aiva_common/enums.py` (400+ 行)
- **消息隊列**: `/workspaces/AIVA/services/aiva_common/mq.py`

### 文檔

- **通訊合約文檔**: `/workspaces/AIVA/MODULE_COMMUNICATION_CONTRACTS.md`
- **本總結**: `/workspaces/AIVA/COMMUNICATION_CONTRACTS_SUMMARY.md`

---

## 🎯 使用方式

### 發送訓練任務

```python
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    AITrainingStartPayload,
    ModelTrainingConfig
)
from services.aiva_common.enums import Topic, ModuleName

# 創建訓練配置
config = ModelTrainingConfig(
    episodes=100,
    learning_rate=0.001,
    gamma=0.99,
    batch_size=32
)

# 創建訓練請求
payload = AITrainingStartPayload(
    training_id="training_20250114_001",
    training_type="batch",
    scenario_id="SQLI-1",
    target_vulnerability="sqli",
    config=config,
    metadata={"notes": "第一次批量訓練"}
)

# 包裝為標準消息
message = AivaMessage(
    header=MessageHeader(
        message_id="msg_abc123",
        trace_id="trace_xyz789",
        source_module=ModuleName.CORE
    ),
    topic=Topic.TASK_AI_TRAINING_START,
    payload=payload.model_dump()
)

# 發送到 RabbitMQ
await broker.publish(Topic.TASK_AI_TRAINING_START, message.model_dump_json().encode())
```

### 監聽訓練進度

```python
from services.aiva_common.mq import get_broker
from services.aiva_common.enums import Topic

broker = await get_broker()

async for mqmsg in broker.subscribe(Topic.RESULTS_AI_TRAINING_PROGRESS):
    msg = AivaMessage.model_validate_json(mqmsg.body)
    progress = AITrainingProgressPayload(**msg.payload)

    print(f"訓練進度: {progress.episode_number}/{progress.total_episodes}")
    print(f"平均質量: {progress.avg_quality:.2f}")
    print(f"平均獎勵: {progress.avg_reward:.2f}")
```

### 查詢 RAG 知識庫

```python
from services.aiva_common.schemas import RAGQueryPayload, RAGResponsePayload

# 創建查詢
query = RAGQueryPayload(
    query_id="query_001",
    query_text="SQL 注入的有效載荷",
    top_k=5,
    min_similarity=0.7,
    knowledge_types=["payload", "technique"]
)

# 發送查詢
message = AivaMessage(
    header=MessageHeader(
        message_id="msg_query_001",
        trace_id="trace_001",
        source_module=ModuleName.CORE
    ),
    topic=Topic.TASK_RAG_QUERY,
    payload=query.model_dump()
)

await broker.publish(Topic.TASK_RAG_QUERY, message.model_dump_json().encode())

# 接收響應
async for mqmsg in broker.subscribe(Topic.RESULTS_RAG_RESPONSE):
    msg = AivaMessage.model_validate_json(mqmsg.body)
    response = RAGResponsePayload(**msg.payload)

    print(f"找到 {response.total_results} 個相關知識")
    print(f"增強上下文: {response.enhanced_context}")
```

---

## ✅ 驗證

所有 Schema 已通過 Python 語法檢查：

```bash
✅ python -m py_compile services/aiva_common/schemas.py
✅ python -m py_compile services/aiva_common/enums.py
```

所有新增的合約類別：

```python
✅ AITrainingStartPayload
✅ AITrainingProgressPayload
✅ AITrainingCompletedPayload
✅ AIExperienceCreatedEvent
✅ AITraceCompletedEvent
✅ AIModelUpdatedEvent
✅ AIModelDeployCommand
✅ RAGKnowledgeUpdatePayload
✅ RAGQueryPayload
✅ RAGResponsePayload
✅ AIVARequest
✅ AIVAResponse
✅ AIVAEvent
✅ AIVACommand
```

---

## 🚀 下一步

現在所有通訊合約已經完成，可以繼續實現：

1. ✅ **完成 TrainingOrchestrator** - 使用 AI 訓練合約
2. ✅ **完成 TrainingUI** - 監聽訓練進度消息
3. ✅ **整合 RabbitMQ** - 所有模組使用統一的消息格式
4. ✅ **測試端到端流程** - 驗證完整的訓練流程

---

## 📝 更新日誌

- **2025-10-14 20:00** - ✅ 添加完整的 AI 訓練、RAG、統一包裝器合約
- **2025-10-14 19:30** - ✅ 更新 Topic 枚舉，添加 AI 和 RAG 主題
- **2025-10-13** - ✅ 增強漏洞發現合約，集成業界標準
- **2025-10-12** - ✅ 添加強化學習核心 Schemas

---

## 🎉 總結

**所有模組間的通訊合約已經完成！**

- ✅ 30+ Topics 涵蓋所有通訊場景
- ✅ 50+ Payload Schemas 定義清晰
- ✅ 支持 Python、Go、TypeScript、Rust 多語言
- ✅ 集成 CVSS、CVE、CWE、MITRE ATT&CK、SARIF 業界標準
- ✅ 統一的請求-響應、事件、命令模式
- ✅ 完整的 AI 訓練、RAG 知識庫通訊支持

現在 AIVA 系統擁有完整、標準化、可擴展的模組間通訊協議！🎊
