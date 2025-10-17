# AIVA 跨模組通訊 CLI 參考文件

> **文件類型**: 系統架構分析 + CLI 參考指南  
> **生成時間**: 2025年10月16日 23:00  
> **資料來源**: 14 個 Mermaid 架構圖 + 程式碼分析  
> **目的**: 完整記錄所有跨模組通訊方式，作為 CLI 和整合開發參考

---

## 📋 目錄

1. [通訊架構概覽](#通訊架構概覽)
2. [訊息佇列系統](#訊息佇列系統)
3. [跨模組通訊方式分類](#跨模組通訊方式分類)
4. [Topic 路由表完整清單](#topic-路由表完整清單)
5. [各模組通訊接口](#各模組通訊接口)
6. [CLI 指令參考](#cli-指令參考)
7. [整合範例](#整合範例)

---

## 🏗️ 通訊架構概覽

### 系統架構層次

```
┌─────────────────────────────────────────────────────────┐
│                    🎨 前端層 (Frontend)                  │
│              Web UI (FastAPI + React)                   │
│              REST API (OpenAPI 3.0)                     │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────────┐
│                  🤖 核心層 (Core Layer)                  │
│  • AI Core Engine (生物神經網路)                         │
│  • Strategy Generator (策略生成器)                       │
│  • Task Manager (任務管理器)                             │
└───┬──────────────────────────────────────────────┬──────┘
    │                                              │
    │ RabbitMQ (TOPIC Exchange)                   │
    │                                              │
┌───▼──────────────────┐              ┌───────────▼───────┐
│  🔍 掃描層 (Scan)     │              │ ⚡ 檢測層 (Function) │
│  • Python Scanner    │◄────────────►│  • Python Funcs   │
│  • TypeScript Scanner│              │  • Go Funcs       │
│  • Rust Gatherer     │              │  • Rust Funcs     │
└──────────┬───────────┘              └────────┬──────────┘
           │                                   │
           │              RabbitMQ             │
           │                                   │
           └───────────────┬───────────────────┘
                           │
                ┌──────────▼──────────┐
                │ 🔗 整合層 (Integration) │
                │  • Report Generator │
                │  • Risk Analyzer    │
                │  • Result Aggregator│
                └─────────────────────┘
```

### 通訊技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **訊息佇列** | RabbitMQ | 模組間異步通訊 |
| **交換器類型** | TOPIC Exchange | 靈活的路由規則 |
| **訊息格式** | JSON (Pydantic) | 結構化資料傳輸 |
| **傳輸協定** | AMQP 0-9-1 | 可靠的訊息傳遞 |
| **持久化** | Persistent Messages | 訊息不丟失 |
| **確認機制** | Publisher Confirms | 發送確認 |
| **QoS** | Prefetch Count: 10 | 流量控制 |

---

## 📨 訊息佇列系統

### RabbitMQ 配置

#### 核心參數
```python
# 來源: services/aiva_common/mq.py

class RabbitBroker:
    """RabbitMQ 消息代理實現"""
    
    # 連接配置
    rabbitmq_url = "amqp://guest:guest@localhost:5672/"
    exchange_name = "aiva.tasks"  # 預設交換器名稱
    exchange_type = ExchangeType.TOPIC  # TOPIC 路由
    
    # QoS 配置
    prefetch_count = 10  # 預取訊息數量
    
    # 訊息配置
    delivery_mode = DeliveryMode.PERSISTENT  # 持久化
    publisher_confirms = True  # 發布者確認
```

#### 交換器架構
```
主交換器: aiva.tasks (TOPIC)
    │
    ├─ tasks.scan.*          → Scan Module
    ├─ tasks.function.*      → Function Modules
    ├─ tasks.ai.*            → AI Training
    ├─ tasks.rag.*           → RAG Knowledge
    ├─ tasks.integration.*   → Integration
    ├─ results.*             → 結果回傳
    ├─ events.*              → 事件通知
    ├─ commands.*            → 指令控制
    └─ status.*              → 狀態更新
```

### 訊息封裝結構

#### MQMessage (基礎封裝)
```python
@dataclass
class MQMessage:
    """訊息佇列基礎封裝"""
    body: bytes              # 訊息主體 (JSON 序列化)
    routing_key: str | None  # 路由鍵 (Topic)
```

#### AivaMessage (業務封裝)
```python
class AivaMessage(BaseModel):
    """AIVA 統一訊息格式"""
    message_id: str          # UUID 訊息 ID
    correlation_id: str      # 關聯 ID (追蹤用)
    timestamp: datetime      # 時間戳
    source_module: ModuleName  # 來源模組
    target_module: ModuleName  # 目標模組
    topic: Topic             # 訊息主題
    payload: dict            # 業務資料
    metadata: dict           # 額外元數據
```

---

## 🔀 跨模組通訊方式分類

根據 14 個 Mermaid 架構圖分析，AIVA 共有 **7 種**跨模組通訊方式：

### 1️⃣ 任務分發 (Task Distribution)

**模式**: Core → Function/Scan  
**方向**: 單向 (請求)  
**Topic 前綴**: `tasks.*`

```mermaid
sequenceDiagram
    Core->>MQ: tasks.function.sqli
    MQ->>Function: 接收任務
    Function->>Function: 執行檢測
```

**特點**:
- Core 作為任務調度中心
- 支援多種檢測類型
- 異步非阻塞執行

**適用場景**:
- SQL 注入檢測
- XSS 檢測
- SSRF 檢測
- IDOR 檢測
- 掃描任務啟動

---

### 2️⃣ 結果回傳 (Result Reporting)

**模式**: Function/Scan → Core/Integration  
**方向**: 單向 (回應)  
**Topic 前綴**: `results.*`

```mermaid
sequenceDiagram
    Function->>MQ: results.function.completed
    MQ->>Core: 接收結果
    Core->>Integration: 轉發整合
```

**特點**:
- 帶狀態的結果回報 (completed/progress/failed)
- 包含詳細檢測結果
- 支援部分結果上報

**適用場景**:
- 檢測完成通知
- 掃描進度更新
- 錯誤報告

---

### 3️⃣ 事件通知 (Event Broadcasting)

**模式**: Any → All (廣播)  
**方向**: 多向  
**Topic 前綴**: `events.*`

```mermaid
graph LR
    Scan -->|events.scan.asset.discovered| MQ
    Function -->|events.function.vuln.found| MQ
    AI -->|events.ai.model.updated| MQ
    MQ --> Core
    MQ --> Integration
```

**特點**:
- 發布-訂閱模式
- 多個模組可同時接收
- 不等待回應

**適用場景**:
- 資產發現通知
- 漏洞發現事件
- AI 模型更新
- 經驗學習記錄

---

### 4️⃣ 指令控制 (Command Control)

**模式**: Core → Any  
**方向**: 單向 (控制)  
**Topic 前綴**: `commands.*`

```mermaid
sequenceDiagram
    Core->>MQ: commands.task.cancel
    MQ->>Function: 取消執行
    Function->>MQ: 確認停止
```

**特點**:
- 即時控制指令
- 優先級高
- 需要確認機制

**適用場景**:
- 任務取消
- 全域配置更新
- AI 模型部署
- 報告生成指令

---

### 5️⃣ 狀態同步 (Status Synchronization)

**模式**: Any → Core  
**方向**: 單向 (狀態)  
**Topic 前綴**: `status.*`

```mermaid
sequenceDiagram
    loop 每 30 秒
        Function->>MQ: module.heartbeat
        Scan->>MQ: module.heartbeat
    end
    MQ->>Core: 健康檢查
```

**特點**:
- 週期性心跳
- 任務狀態更新
- 健康監控

**適用場景**:
- 模組心跳
- 任務進度更新
- 系統健康檢查

---

### 6️⃣ 反饋優化 (Feedback Loop)

**模式**: Integration → Core  
**方向**: 雙向 (學習)  
**Topic 前綴**: `feedback.*`

```mermaid
sequenceDiagram
    Integration->>MQ: feedback.core.strategy
    MQ->>Core: 策略優化建議
    Core->>AI: 更新模型
    AI->>MQ: events.ai.model.updated
```

**特點**:
- AI 學習迴路
- 策略動態調整
- 持續優化

**適用場景**:
- 檢測策略優化
- 誤報率調整
- 效能優化建議

---

### 7️⃣ 日誌聚合 (Log Aggregation)

**模式**: All → Log Collector  
**方向**: 多對一  
**Topic**: `log.results.all`

```mermaid
graph TB
    Core -->|log.results.all| Logger
    Scan -->|log.results.all| Logger
    Function -->|log.results.all| Logger
    Integration -->|log.results.all| Logger
```

**特點**:
- 統一日誌收集
- 便於審計追蹤
- 支援全文搜索

**適用場景**:
- 系統審計
- 錯誤追蹤
- 效能分析

---

## 📊 Topic 路由表完整清單

### 掃描相關 Topics (Scan)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.scan.start` | 任務 | Core → Scan | 開始掃描任務 |
| `results.scan.completed` | 結果 | Scan → Core | 掃描完成 |
| `results.scan.progress` | 結果 | Scan → Core | 掃描進度更新 |
| `results.scan.failed` | 結果 | Scan → Core | 掃描失敗 |
| `events.scan.asset.discovered` | 事件 | Scan → All | 資產發現通知 |

**使用範例**:
```python
# 啟動掃描
await broker.publish(
    topic=Topic.TASK_SCAN_START,
    body=json.dumps({
        "target_url": "https://example.com",
        "scan_type": "full",
        "depth": 3
    }).encode()
)
```

---

### 檢測功能 Topics (Function)

#### SQL 注入檢測
| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.function.sqli` | 任務 | Core → Function | SQL 注入檢測任務 |
| `results.function.completed` | 結果 | Function → Core | 檢測完成 |
| `results.function.progress` | 結果 | Function → Core | 檢測進度 |
| `results.function.failed` | 結果 | Function → Core | 檢測失敗 |
| `events.function.vuln.found` | 事件 | Function → All | 漏洞發現通知 |

**使用範例**:
```python
# 提交 SQLi 檢測任務
await broker.publish(
    topic=Topic.TASK_FUNCTION_SQLI,
    body=json.dumps({
        "target": "https://api.example.com/users",
        "parameter": "id",
        "method": "GET",
        "detection_engines": ["boolean", "time", "error"]
    }).encode()
)
```

#### XSS 檢測
| Topic | 說明 |
|-------|------|
| `tasks.function.xss` | XSS 檢測任務 |

**檢測類型**:
- Reflected XSS
- Stored XSS
- DOM-based XSS
- Blind XSS

#### SSRF 檢測
| Topic | 說明 |
|-------|------|
| `tasks.function.ssrf` | SSRF 檢測任務 |

**檢測方法**:
- 內部位址探測
- OAST 帶外測試
- 協議繞過檢測

#### IDOR 檢測
| Topic | 說明 |
|-------|------|
| `tasks.function.idor` | IDOR 檢測任務 |

⚠️ **注意**: 當前 Topic 命名不一致  
- 實際: `FUNCTION_IDOR_TASK`  
- 建議: `tasks.function.idor` (統一命名規範)

---

### AI 訓練 Topics (AI Training)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.ai.training.start` | 任務 | Core → AI | 開始訓練 |
| `tasks.ai.training.episode` | 任務 | Core → AI | 訓練單個回合 |
| `tasks.ai.training.stop` | 命令 | Core → AI | 停止訓練 |
| `results.ai.training.progress` | 結果 | AI → Core | 訓練進度 |
| `results.ai.training.completed` | 結果 | AI → Core | 訓練完成 |
| `results.ai.training.failed` | 結果 | AI → Core | 訓練失敗 |
| `events.ai.experience.created` | 事件 | AI → All | 經驗記錄創建 |
| `events.ai.trace.completed` | 事件 | AI → All | 追蹤完成 |
| `events.ai.model.updated` | 事件 | AI → All | 模型更新 |
| `events.ai.scenario.loaded` | 事件 | AI → All | 場景載入 |
| `commands.ai.model.deploy` | 命令 | Core → AI | 部署模型 |

**使用範例**:
```python
# 啟動 AI 訓練
await broker.publish(
    topic=Topic.TASK_AI_TRAINING_START,
    body=json.dumps({
        "scenario_id": "sqli_training_001",
        "episodes": 100,
        "learning_rate": 0.001
    }).encode()
)
```

---

### RAG 知識庫 Topics (RAG Knowledge)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.rag.knowledge.update` | 任務 | Core → RAG | 更新知識庫 |
| `tasks.rag.query` | 任務 | Any → RAG | 查詢知識 |
| `results.rag.response` | 結果 | RAG → Any | 查詢結果 |

**使用範例**:
```python
# 查詢 RAG 知識庫
await broker.publish(
    topic=Topic.TASK_RAG_QUERY,
    body=json.dumps({
        "query": "如何檢測 SQL 注入漏洞?",
        "context": "PostgreSQL database",
        "top_k": 5
    }).encode()
)
```

---

### 整合分析 Topics (Integration)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.integration.analysis.start` | 任務 | Core → Integration | 開始分析 |
| `results.integration.analysis.progress` | 結果 | Integration → Core | 分析進度 |
| `results.integration.analysis.completed` | 結果 | Integration → Core | 分析完成 |
| `commands.integration.report.generate` | 命令 | Core → Integration | 生成報告 |
| `events.integration.report.generated` | 事件 | Integration → All | 報告已生成 |

**使用範例**:
```python
# 生成綜合報告
await broker.publish(
    topic=Topic.COMMAND_INTEGRATION_REPORT_GENERATE,
    body=json.dumps({
        "scan_session_id": "scan_12345",
        "format": "pdf",
        "include_charts": True
    }).encode()
)
```

---

### 威脅情報 Topics (Threat Intel)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.threat_intel.lookup` | 任務 | Any → ThreatIntel | 查詢威脅情報 |
| `tasks.threat_intel.ioc_enrichment` | 任務 | Any → ThreatIntel | IOC 豐富化 |
| `tasks.threat_intel.mitre_mapping` | 任務 | Any → ThreatIntel | MITRE ATT&CK 映射 |
| `results.threat_intel` | 結果 | ThreatIntel → Any | 情報結果 |

---

### 授權檢測 Topics (AuthZ)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.authz.check` | 任務 | Core → AuthZ | 授權檢查 |
| `tasks.authz.analyze` | 任務 | Core → AuthZ | 授權分析 |
| `results.authz` | 結果 | AuthZ → Core | 授權結果 |

---

### 後滲透 Topics (PostEx - 僅測試環境)

⚠️ **警告**: 僅用於授權測試環境

| Topic | 說明 |
|-------|------|
| `tasks.postex.test` | 後滲透測試 |
| `tasks.postex.privilege_escalation` | 權限提升 |
| `tasks.postex.lateral_movement` | 橫向移動 |
| `tasks.postex.data_exfiltration` | 資料外洩 |
| `tasks.postex.persistence` | 持久化 |
| `results.postex` | 後滲透結果 |

---

### 修復建議 Topics (Remediation)

| Topic | 類型 | 方向 | 說明 |
|-------|------|------|------|
| `tasks.remediation.generate` | 任務 | Integration → Remediation | 生成修復建議 |
| `results.remediation` | 結果 | Remediation → Integration | 修復建議結果 |

---

### 通用控制 Topics (General)

| Topic | 類型 | 說明 |
|-------|------|------|
| `findings.detected` | 事件 | 發現檢測結果 |
| `log.results.all` | 日誌 | 全域日誌收集 |
| `status.task.update` | 狀態 | 任務狀態更新 |
| `feedback.core.strategy` | 反饋 | 策略優化反饋 |
| `module.heartbeat` | 狀態 | 模組心跳 |
| `command.task.cancel` | 命令 | 取消任務 |
| `config.global.update` | 配置 | 全域配置更新 |

---

## 🔌 各模組通訊接口

### Core Module (核心模組)

#### 發送的 Topics
```python
# 任務分發
Topic.TASK_SCAN_START             # → Scan
Topic.TASK_FUNCTION_SQLI          # → Function
Topic.TASK_FUNCTION_XSS           # → Function
Topic.TASK_FUNCTION_SSRF          # → Function
Topic.FUNCTION_IDOR_TASK          # → Function
Topic.TASK_AI_TRAINING_START      # → AI
Topic.TASK_INTEGRATION_ANALYSIS_START  # → Integration

# 指令控制
Topic.COMMAND_TASK_CANCEL         # → All
Topic.COMMAND_AI_MODEL_DEPLOY     # → AI
Topic.COMMAND_INTEGRATION_REPORT_GENERATE  # → Integration
Topic.CONFIG_GLOBAL_UPDATE        # → All
```

#### 接收的 Topics
```python
# 結果回傳
Topic.RESULTS_SCAN_COMPLETED      # ← Scan
Topic.RESULTS_FUNCTION_COMPLETED  # ← Function
Topic.RESULTS_AI_TRAINING_COMPLETED  # ← AI
Topic.RESULTS_INTEGRATION_ANALYSIS_COMPLETED  # ← Integration

# 狀態更新
Topic.STATUS_TASK_UPDATE          # ← All
Topic.MODULE_HEARTBEAT            # ← All

# 反饋學習
Topic.FEEDBACK_CORE_STRATEGY      # ← Integration
```

#### CLI 命令範例
```bash
# 啟動 Core 模組
python -m services.core.aiva_core.app

# 環境變數配置
export RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
export EXCHANGE_NAME="aiva.tasks"
```

---

### Scan Module (掃描模組)

#### 發送的 Topics
```python
# 結果回傳
Topic.RESULTS_SCAN_COMPLETED      # → Core
Topic.RESULTS_SCAN_PROGRESS       # → Core
Topic.RESULTS_SCAN_FAILED         # → Core

# 事件通知
Topic.EVENT_SCAN_ASSET_DISCOVERED # → All

# 狀態同步
Topic.MODULE_HEARTBEAT            # → Core
```

#### 接收的 Topics
```python
# 任務接收
Topic.TASK_SCAN_START             # ← Core

# 指令控制
Topic.COMMAND_TASK_CANCEL         # ← Core
```

#### CLI 命令範例
```bash
# Python Scanner
python -m services.scan.aiva_scan.worker

# TypeScript Scanner (Node.js)
cd services/scan/aiva_scan_node
npm start

# Rust Info Gatherer
cd services/scan/info_gatherer_rust
cargo run --release
```

---

### Function Module (檢測功能模組)

#### Python Functions

**SQLi Detection**
```bash
# 啟動 SQLi Worker
python -m services.function.function_sqli.aiva_func_sqli.worker

# 接收 Topic
Topic.TASK_FUNCTION_SQLI

# 發送 Topics
Topic.RESULTS_FUNCTION_COMPLETED
Topic.RESULTS_FUNCTION_PROGRESS
Topic.EVENT_FUNCTION_VULN_FOUND
```

**XSS Detection**
```bash
# 啟動 XSS Worker
python -m services.function.function_xss.aiva_func_xss.worker

# 接收 Topic
Topic.TASK_FUNCTION_XSS
```

**SSRF Detection**
```bash
# 啟動 SSRF Worker
python -m services.function.function_ssrf.aiva_func_ssrf.worker

# 接收 Topic
Topic.TASK_FUNCTION_SSRF
```

**IDOR Detection**
```bash
# 啟動 IDOR Worker
python -m services.function.function_idor.aiva_func_idor.worker

# 接收 Topic (需修正)
Topic.FUNCTION_IDOR_TASK  # 應改為 TASK_FUNCTION_IDOR
```

#### Go Functions

**AuthN Detection**
```bash
cd services/function/function_authn_go
go run cmd/worker/main.go
```

**CSPM Scanner**
```bash
cd services/function/function_cspm_go
go run cmd/worker/main.go
```

**SCA Scanner**
```bash
cd services/function/function_sca_go
go run cmd/worker/main.go
```

#### Rust Functions

**SAST Analyzer**
```bash
cd services/function/function_sast_rust
cargo run --release
```

---

### Integration Module (整合模組)

#### 發送的 Topics
```python
# 結果回傳
Topic.RESULTS_INTEGRATION_ANALYSIS_COMPLETED  # → Core

# 事件通知
Topic.EVENT_INTEGRATION_REPORT_GENERATED     # → All

# 反饋優化
Topic.FEEDBACK_CORE_STRATEGY                 # → Core

# 日誌
Topic.LOG_RESULTS_ALL                        # → Logger
```

#### 接收的 Topics
```python
# 任務接收
Topic.TASK_INTEGRATION_ANALYSIS_START        # ← Core

# 指令控制
Topic.COMMAND_INTEGRATION_REPORT_GENERATE    # ← Core

# 結果聚合
Topic.RESULTS_FUNCTION_COMPLETED             # ← Function
Topic.RESULTS_SCAN_COMPLETED                 # ← Scan
```

#### CLI 命令範例
```bash
# 啟動 Integration Service
python -m services.integration.aiva_integration.app

# API Gateway
python -m services.integration.api_gateway.api_gateway.app
```

---

## 🛠️ CLI 指令參考

### 完整啟動流程

#### 1. 啟動基礎設施
```bash
# 啟動 RabbitMQ
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management

# 啟動 PostgreSQL
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=aiva123 \
  -p 5432:5432 \
  postgres:15

# 啟動 Redis
docker run -d --name redis \
  -p 6379:6379 \
  redis:7
```

#### 2. 啟動核心模組
```bash
# Core Module
python -m services.core.aiva_core.app &

# 等待 Core 初始化
sleep 5
```

#### 3. 啟動掃描模組
```bash
# Python Scanner
python -m services.scan.aiva_scan.worker &

# TypeScript Scanner
cd services/scan/aiva_scan_node && npm start &

# Rust Gatherer
cd services/scan/info_gatherer_rust && cargo run --release &
```

#### 4. 啟動檢測模組
```bash
# Python Functions
python -m services.function.function_sqli.aiva_func_sqli.worker &
python -m services.function.function_xss.aiva_func_xss.worker &
python -m services.function.function_ssrf.aiva_func_ssrf.worker &
python -m services.function.function_idor.aiva_func_idor.worker &

# Go Functions
cd services/function/function_authn_go && go run cmd/worker/main.go &
cd services/function/function_cspm_go && go run cmd/worker/main.go &
cd services/function/function_sca_go && go run cmd/worker/main.go &

# Rust Functions
cd services/function/function_sast_rust && cargo run --release &
```

#### 5. 啟動整合模組
```bash
# Integration Service
python -m services.integration.aiva_integration.app &

# API Gateway
python -m services.integration.api_gateway.api_gateway.app &
```

### Docker Compose 快速啟動

```bash
# 使用 Docker Compose 啟動所有服務
docker-compose -f docker/docker-compose.yml up -d

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f core
docker-compose logs -f function-sqli
```

---

## 📚 整合範例

### 範例 1: 完整掃描流程

```python
"""完整掃描流程示例"""
import asyncio
import json
from services.aiva_common import Topic, RabbitBroker
from services.aiva_common.schemas import AivaMessage, FunctionTaskPayload

async def complete_scan_example():
    """執行完整掃描流程"""
    broker = RabbitBroker()
    await broker.connect()
    
    # 1. 啟動掃描
    scan_task = AivaMessage(
        source_module="CoreModule",
        target_module="ScanModule",
        topic=Topic.TASK_SCAN_START,
        payload={
            "target_url": "https://testphp.vulnweb.com",
            "scan_depth": 3,
            "scan_type": "full"
        }
    )
    await broker.publish(
        topic=Topic.TASK_SCAN_START,
        body=scan_task.model_dump_json().encode()
    )
    
    # 2. 訂閱掃描結果
    async for msg in broker.subscribe(Topic.RESULTS_SCAN_COMPLETED):
        scan_result = json.loads(msg.body)
        print(f"掃描完成: {scan_result}")
        
        # 3. 針對發現的端點進行 SQLi 檢測
        for endpoint in scan_result.get("endpoints", []):
            sqli_task = FunctionTaskPayload(
                task_id=f"sqli_{endpoint['id']}",
                target=endpoint["url"],
                parameters=endpoint["parameters"],
                detection_type="sqli"
            )
            
            await broker.publish(
                topic=Topic.TASK_FUNCTION_SQLI,
                body=sqli_task.model_dump_json().encode()
            )
        
        break  # 只處理第一個結果
    
    # 4. 訂閱檢測結果
    async for msg in broker.subscribe(Topic.RESULTS_FUNCTION_COMPLETED):
        result = json.loads(msg.body)
        print(f"檢測完成: {result}")
        # 可繼續處理...
    
    await broker.close()

# 執行
asyncio.run(complete_scan_example())
```

### 範例 2: AI 訓練流程

```python
"""AI 訓練流程示例"""
async def ai_training_example():
    """執行 AI 訓練"""
    broker = RabbitBroker()
    await broker.connect()
    
    # 1. 啟動訓練
    training_task = {
        "scenario_id": "sqli_detection_001",
        "episodes": 100,
        "learning_rate": 0.001,
        "batch_size": 32
    }
    
    await broker.publish(
        topic=Topic.TASK_AI_TRAINING_START,
        body=json.dumps(training_task).encode()
    )
    
    # 2. 監控訓練進度
    async for msg in broker.subscribe(Topic.RESULTS_AI_TRAINING_PROGRESS):
        progress = json.loads(msg.body)
        print(f"訓練進度: {progress['episode']}/{progress['total_episodes']}")
        print(f"當前獎勵: {progress['reward']}")
        
        if progress['episode'] >= progress['total_episodes']:
            break
    
    # 3. 部署模型
    await broker.publish(
        topic=Topic.COMMAND_AI_MODEL_DEPLOY,
        body=json.dumps({"model_id": "sqli_v2"}).encode()
    )
    
    await broker.close()
```

### 範例 3: 監控所有事件

```python
"""監控系統所有事件"""
async def monitor_all_events():
    """訂閱所有事件進行監控"""
    broker = RabbitBroker()
    await broker.connect()
    
    # 訂閱多個事件 Topic
    events_to_monitor = [
        Topic.EVENT_SCAN_ASSET_DISCOVERED,
        Topic.EVENT_FUNCTION_VULN_FOUND,
        Topic.EVENT_AI_MODEL_UPDATED,
        Topic.EVENT_INTEGRATION_REPORT_GENERATED,
    ]
    
    # 同時訂閱多個 Topic
    tasks = [
        broker.subscribe(topic) 
        for topic in events_to_monitor
    ]
    
    async def process_events(topic, subscription):
        """處理事件"""
        async for msg in subscription:
            event = json.loads(msg.body)
            print(f"[{topic}] {event}")
    
    # 並行處理所有事件
    await asyncio.gather(*[
        process_events(topic, sub)
        for topic, sub in zip(events_to_monitor, tasks)
    ])
    
    await broker.close()
```

### 範例 4: 錯誤處理與重試

```python
"""帶錯誤處理的任務提交"""
import backoff

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=30
)
async def submit_task_with_retry(broker, topic, payload):
    """帶重試機制的任務提交"""
    try:
        await broker.publish(
            topic=topic,
            body=json.dumps(payload).encode()
        )
        print(f"任務提交成功: {topic}")
    except Exception as e:
        print(f"任務提交失敗: {e}")
        raise

# 使用
async def main():
    broker = RabbitBroker()
    await broker.connect()
    
    await submit_task_with_retry(
        broker,
        Topic.TASK_FUNCTION_SQLI,
        {"target": "https://example.com"}
    )
    
    await broker.close()
```

---

## 🔧 故障排除

### 常見問題

#### 1. RabbitMQ 連接失敗
```bash
# 檢查 RabbitMQ 是否運行
docker ps | grep rabbitmq

# 查看 RabbitMQ 日誌
docker logs rabbitmq

# 檢查端口
netstat -an | grep 5672
```

#### 2. 訊息未送達
```python
# 啟用 Publisher Confirms
broker = RabbitBroker()
await broker.connect()

# 檢查 Exchange 是否存在
# 在 RabbitMQ Management UI (http://localhost:15672)
```

#### 3. Worker 未收到訊息
```bash
# 檢查 Worker 是否正確訂閱
# 查看 Queue 綁定
rabbitmqctl list_bindings
```

---

## 📈 效能優化建議

### 1. 批次處理
```python
# 批次發送多個任務
async def batch_publish(broker, tasks):
    """批次發送任務"""
    for task in tasks:
        await broker.publish(task.topic, task.body)
    # 建議: 使用 asyncio.gather 並行發送
```

### 2. 預取控制
```python
# 調整 prefetch_count
await channel.set_qos(prefetch_count=50)  # 提高並發
```

### 3. 連接池
```python
# 使用連接池管理多個連接
class BrokerPool:
    def __init__(self, size=10):
        self.brokers = [RabbitBroker() for _ in range(size)]
```

---

## 📝 更新日誌

| 日期 | 版本 | 更新內容 |
|------|------|---------|
| 2025-10-16 | 1.0 | 初始版本，完整分析 14 個架構圖 |

---

## 🔗 相關文檔

- [AIVA 架構設計文檔](../docs/ARCHITECTURE/)
- [RabbitMQ 官方文檔](https://www.rabbitmq.com/documentation.html)
- [Pydantic 文檔](https://docs.pydantic.dev/)

---

**© 2025 AIVA Security Testing Platform**
