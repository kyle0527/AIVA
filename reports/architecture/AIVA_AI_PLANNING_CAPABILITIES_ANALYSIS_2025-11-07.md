# AIVA AI 規劃功能分析報告

**分析日期**: 2025年11月7日  
**分析範圍**: services/core 模組中的 AI 規劃相關功能  
**分析重點**: AI 如何進行攻擊規劃、決策制定、任務編排

---

## 📊 執行摘要

### AIVA 的 AI 規劃架構

AIVA 擁有**多層次的 AI 規劃系統**，從高層戰略規劃到底層任務執行，形成完整的決策鏈：

```
┌─────────────────────────────────────────────────┐
│  Layer 1: AI Commander (戰略指揮層)              │
│  - 任務分析與分配                                │
│  - AI 組件協調                                   │
│  - 9 種任務類型決策                              │
└────────────────┬────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────┐
│  Layer 2: Decision & Planning (決策規劃層)       │
│  - EnhancedDecisionAgent (智能決策)             │
│  - SkillGraph (技能圖譜規劃)                    │
│  - ExecutionPlanner (執行計劃器)                │
└────────────────┬────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────┐
│  Layer 3: Plan Execution (計劃執行層)            │
│  - PlanExecutor (攻擊計劃執行)                  │
│  - TaskGenerator (任務生成)                     │
│  - AttackPlanMapper (計劃映射)                  │
└────────────────┬────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────┐
│  Layer 4: RAG & Learning (知識增強層)            │
│  - RAG Engine (知識檢索)                        │
│  - ExperienceManager (經驗學習)                 │
│  - TrainingOrchestrator (持續訓練)              │
└─────────────────────────────────────────────────┘
```

### 核心發現 ⭐

1. **完整的四層規劃架構** ✅
   - 從戰略決策到戰術執行全覆蓋
   - 每層職責清晰，無重疊

2. **多種規劃能力並存** ✅
   - 攻擊計劃生成
   - 策略決策
   - 風險評估
   - 任務編排
   - 工具選擇

3. **知識驅動的智能規劃** ✅
   - RAG 增強決策
   - 經驗學習反饋
   - 技能圖譜指導

4. **跨語言協調規劃** ✅
   - Python/Go/Rust/TypeScript 統一調度
   - Multi-Language AI Coordinator

---

## 🏗️ 四層規劃架構詳解

### Layer 1: AI Commander - 戰略指揮層 🎖️

**文件位置**: `services/core/aiva_core/ai_commander.py`  
**代碼行數**: 1,104 行  
**核心類別**: `AICommander`

#### 職責定位

AI Commander 是 AIVA 的**最高指揮官**，統一管理和協調所有 AI 組件。

#### 支持的 9 種任務類型

```python
class AITaskType(str, Enum):
    # === 決策類 ===
    ATTACK_PLANNING = "attack_planning"        # 攻擊計畫生成
    STRATEGY_DECISION = "strategy_decision"    # 策略決策
    RISK_ASSESSMENT = "risk_assessment"        # 風險評估
    
    # === 執行類 ===
    VULNERABILITY_DETECTION = "vulnerability_detection"  # 漏洞檢測
    EXPLOIT_EXECUTION = "exploit_execution"              # 漏洞利用
    CODE_ANALYSIS = "code_analysis"                      # 代碼分析
    
    # === 學習類 ===
    EXPERIENCE_LEARNING = "experience_learning"  # 經驗學習
    MODEL_TRAINING = "model_training"            # 模型訓練
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # 知識檢索
```

#### 協調的 7 個 AI 組件

```python
class AIComponent(str, Enum):
    BIO_NEURON_AGENT = "bio_neuron_agent"          # Python 主控 AI
    RAG_ENGINE = "rag_engine"                      # RAG 引擎
    TRAINING_SYSTEM = "training_system"            # 訓練系統
    MULTILANG_COORDINATOR = "multilang_coordinator" # 多語言協調器
    
    # 語言專屬 AI
    GO_AI_MODULE = "go_ai_module"          # Go AI 模組
    RUST_AI_MODULE = "rust_ai_module"      # Rust AI 模組
    TS_AI_MODULE = "ts_ai_module"          # TypeScript AI 模組
```

#### 核心功能

1. **任務分析與分配**
   - 分析用戶意圖
   - 確定任務類型
   - 選擇最佳 AI 組件

2. **AI 組件協調**
   - 協調 BioNeuronRAGAgent
   - 調用 RAG Engine 增強決策
   - 管理多語言 AI 模組

3. **決策整合**
   - 整合各組件的輸出
   - 風險評估與控制
   - 生成最終決策

4. **經驗積累**
   - 記錄決策過程
   - 存儲執行結果
   - 持續學習優化

**統計數據**:
- 文件大小: 1,104 行
- 主要類別: 3 個 (AICommander, AITaskType, AIComponent)
- 支持任務類型: 9 種
- 協調組件數: 7 個

---

### Layer 2: Decision & Planning - 決策規劃層 🧠

這一層包含三個核心規劃組件：

#### 2.1 EnhancedDecisionAgent - 智能決策代理

**文件位置**: `services/core/aiva_core/decision/enhanced_decision_agent.py`  
**代碼行數**: 568 行  
**核心類別**: `EnhancedDecisionAgent`, `DecisionContext`, `Decision`

**設計理念**:
```python
"""AIVA 決策代理增強模組
用途: 整合風險評估和經驗驅動決策，提升 AI 決策的智能化水平
基於: BioNeuron_模型_AI核心大腦.md 中的決策代理分析
"""
```

**決策流程**:

```
用戶請求
    ↓
DecisionContext (決策上下文)
├── risk_level              # 風險等級
├── discovered_vulns        # 已發現漏洞
├── attempts_without_success # 失敗嘗試次數
├── target_info             # 目標資訊
├── previous_results        # 歷史結果
├── time_constraints        # 時間限制
└── available_tools         # 可用工具
    ↓
決策規則引擎 (Decision Rules Engine)
├── high_risk_confirmation   # 高風險確認規則
├── sql_injection_found      # SQL 注入發現規則
├── multiple_failures        # 多次失敗規則
├── web_service_detected     # Web 服務檢測規則
└── ssh_service_available    # SSH 服務可用規則
    ↓
Decision (決策結果)
├── action                  # 具體動作
├── params                  # 動作參數
├── confidence              # 信心度 (0-1)
├── reasoning               # 決策理由
├── alternatives            # 替代方案
└── risk_assessment         # 風險評估
```

**工具選擇策略**:
```python
self.tool_preferences = {
    "sql_injection": ["sqlmap", "havij", "manual_test"],
    "xss": ["xsser", "xsstrike", "manual_test"],
    "directory_traversal": ["dirb", "gobuster", "manual_enum"],
    "port_scan": ["nmap", "masscan", "unicornscan"],
    "web_scan": ["nikto", "dirb", "wpscan"],
    "brute_force": ["hydra", "medusa", "john"],
}
```

**決策規則示例**:
```python
{
    "name": "high_risk_confirmation",
    "condition": lambda ctx: ctx.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
    "action": "REQUIRE_CONFIRMATION",
    "priority": 100,
    "description": "高風險操作需要用戶確認",
}
```

**統計數據**:
- 文件大小: 568 行
- 決策規則數: 5 個（可擴展）
- 工具選擇策略: 6 種攻擊類型
- 支持的操作模式: 與 BioNeuronMaster 集成

**核心能力**:
- ✅ 基於上下文的智能決策
- ✅ 風險感知決策
- ✅ 經驗驅動的策略調整
- ✅ 多次失敗後自動改變策略
- ✅ 工具選擇優化

---

#### 2.2 SkillGraph - 技能圖譜規劃

**文件位置**: `services/core/aiva_core/decision/skill_graph.py`  
**代碼行數**: 618 行  
**核心類別**: `SkillGraphBuilder`, `SkillGraphAnalyzer`, `SkillNode`, `SkillEdge`, `SkillPath`

**設計理念**:
```python
"""AIVA 技能圖 (Skill Graph) 模組
實現能力關係映射和決策支援
"""
```

**技能圖結構**:

```
SkillNode (技能節點)
├── id                    # 技能 ID
├── name                  # 技能名稱
├── language              # 實現語言 (Python/Go/Rust/TS)
├── topic                 # 所屬主題
├── tags                  # 標籤列表
├── prerequisites         # 前置技能
├── dependencies          # 依賴技能
├── success_rate          # 成功率
├── avg_latency           # 平均延遲
├── last_used             # 最後使用時間
├── usage_count           # 使用次數
└── metadata              # 元數據

SkillEdge (技能邊)
├── source                # 起始節點
├── target                # 目標節點
├── relationship_type     # 關係類型
│   ├── "prerequisite"    # 前置條件
│   ├── "alternative"     # 替代方案
│   ├── "complement"      # 互補關係
│   └── "sequence"        # 順序關係
├── weight                # 權重
├── confidence            # 信心度
└── metadata              # 元數據
```

**技能路徑規劃**:

```python
@dataclass
class SkillPath:
    """技能執行路徑"""
    nodes: list[str]              # 技能節點序列
    edges: list[SkillEdge]        # 連接邊
    total_weight: float           # 總權重
    estimated_time: float         # 預估時間
    success_probability: float    # 成功概率
    description: str = ""         # 路徑描述
```

**關係分析類型**:

1. **前置條件關係** (`_analyze_prerequisite_relationships`)
   - 分析技能的前置依賴
   - 確保執行順序正確

2. **標籤相似性關係** (`_analyze_tag_similarity_relationships`)
   - 基於標籤找出相關技能
   - 發現替代方案

3. **語言生態關係** (`_analyze_language_ecosystem_relationships`)
   - 同語言技能優先組合
   - 跨語言協調

4. **主題關聯關係** (`_analyze_topic_relationships`)
   - 同主題技能協同
   - 構建攻擊鏈

5. **輸入輸出關係** (`_analyze_io_relationships`)
   - 數據流分析
   - 確保參數匹配

**規劃能力**:

```python
class SkillGraphAnalyzer:
    """技能圖分析器"""
    
    async def find_optimal_path(
        self, 
        start_capability: str, 
        goal_capability: str
    ) -> SkillPath:
        """找出最優技能執行路徑"""
        
    async def suggest_next_skills(
        self, 
        current_skills: list[str], 
        context: dict
    ) -> list[str]:
        """建議下一步技能"""
        
    async def evaluate_attack_chain(
        self, 
        skill_sequence: list[str]
    ) -> dict:
        """評估攻擊鏈的可行性"""
```

**統計數據**:
- 文件大小: 618 行
- 主要類別: 5 個
- 關係分析類型: 5 種
- 使用圖論算法: NetworkX
- 支持的語言: Python, Go, Rust, TypeScript

**核心能力**:
- ✅ 技能關係建模
- ✅ 最優路徑規劃
- ✅ 攻擊鏈構建
- ✅ 成功率預測
- ✅ 跨語言技能協調
- ✅ 動態技能推薦

---

#### 2.3 ExecutionPlanner - 執行計劃器

**文件位置**: `services/core/aiva_core/execution_planner.py`  
**代碼行數**: 558 行  
**核心類別**: `ExecutionPlanner`

**設計理念**:
```python
"""AIVA Execution Planner - 執行計劃器
從 aiva_core_v2 遷移到核心模組

異步執行計劃和步驟編排系統
"""
```

**執行計劃結構**:

```python
plan = {
    "plan_id": "plan_<timestamp>_<id>",
    "context": CommandContext,
    "route_info": dict,
    "steps": [
        {
            "type": "validate_input",
            "handler": "input_validator",
            "critical": True,
        },
        {
            "type": "execute_command",
            "handler": "simple_executor",
            "critical": True,
        },
        # ... 更多步驟
    ],
    "estimated_time": 1.0,           # 預估時間（秒）
    "resources_required": [],        # 需要的資源
    "dependencies": [],              # 依賴項
    "created_at": timestamp,
    "status": "created",
}
```

**支持的命令類型與計劃**:

| 命令類型 | 執行步驟 | 預估時間 | 資源需求 |
|---------|---------|---------|---------|
| **SIMPLE** | 1. 驗證輸入<br>2. 執行命令<br>3. 格式化輸出 | 1 秒 | 無 |
| **SCAN** | 1. 驗證目標<br>2. 準備掃描<br>3. 執行掃描<br>4. 處理結果<br>5. 生成報告 | 30 秒 | rust_adapter, scan_engine |
| **ANALYSIS** | 1. 驗證輸入<br>2. 收集上下文<br>3. 分析數據<br>4. 生成洞察<br>5. 格式化響應 | 5 秒 | analysis_engine |
| **AI_TASK** | 1. 分析意圖<br>2. 收集上下文<br>3. AI 推理<br>4. 執行動作<br>5. 格式化響應 | 10 秒 | ai_engine, rag_system |
| **REPORT** | 1. 驗證輸入<br>2. 收集數據<br>3. 分析趨勢<br>4. 生成報告<br>5. 格式化輸出 | 15 秒 | report_generator |

**執行管理**:

```python
class ExecutionPlanner:
    def __init__(self):
        self._execution_queue: list[dict]      # 執行隊列
        self._running_tasks: dict[str, Task]   # 運行中任務
        self._plan_history: dict[str, dict]    # 計劃歷史
        self._execution_lock: asyncio.Lock     # 執行鎖
```

**統計數據**:
- 文件大小: 558 行
- 支持命令類型: 5 種
- 執行步驟範圍: 3-5 步
- 預估時間範圍: 1-30 秒
- 異步執行: ✅

**核心能力**:
- ✅ 多命令類型計劃生成
- ✅ 異步任務編排
- ✅ 資源依賴管理
- ✅ 執行時間預估
- ✅ 計劃歷史追蹤

---

### Layer 3: Plan Execution - 計劃執行層 ⚙️

#### 3.1 PlanExecutor - 攻擊計劃執行器

**文件位置**: `services/core/aiva_core/execution/plan_executor.py`  
**代碼行數**: 771 行  
**核心類別**: `PlanExecutor`

**設計理念**:
```python
"""Plan Executor - 攻擊計畫執行器

負責執行攻擊計畫，管理會話狀態，協調任務分發和結果收集

符合標準：
- 支持順序執行多步驟攻擊鏈
- 透過 RabbitMQ 發送任務到各功能模組
- 使用 TraceLogger 記錄執行過程
- 管理會話生命週期
"""
```

**執行流程**:

```
AttackPlan (攻擊計劃)
├── plan_id              # 計劃 ID
├── scan_id              # 掃描 ID
├── steps                # 攻擊步驟列表
│   └── AttackStep
│       ├── step_id      # 步驟 ID
│       ├── action       # 動作類型
│       ├── target       # 目標
│       ├── parameters   # 參數
│       └── dependencies # 依賴
└── metadata             # 元數據

    ↓

PlanExecutor.execute_plan()
    ↓
創建會話 (SessionState)
├── session_id
├── plan_id
├── status
├── start_time
└── steps_status
    ↓
順序執行步驟
    ├── 檢查依賴
    ├── 生成任務 (FunctionTaskPayload)
    ├── 發送到 RabbitMQ
    ├── 等待結果
    └── 記錄追蹤 (TraceRecord)
    ↓
收集結果
├── findings (漏洞發現)
├── trace_records (追蹤記錄)
├── anomalies (異常)
└── metrics (指標)
    ↓
PlanExecutionResult
```

**會話管理**:

```python
class SessionState:
    session_id: str
    plan_id: str
    scan_id: str
    status: str              # "active", "completed", "failed"
    start_time: datetime
    end_time: datetime | None
    steps_completed: list[str]
    steps_failed: list[str]
    timeout_minutes: int
```

**任務協調**:

```python
# 任務管理
self.running_tasks: dict[str, dict[str, Any]] = {}
self.completed_tasks: dict[str, dict[str, Any]] = {}

# 透過 RabbitMQ 發送任務
await self.mq_client.publish_task(
    task_payload=FunctionTaskPayload(
        task_id=task_id,
        target=FunctionTaskTarget(...),
        function_type=step.action,
        parameters=step.parameters,
    ),
    routing_key=f"function.{step.action}",
)
```

**執行結果**:

```python
@dataclass
class PlanExecutionResult:
    plan_id: str
    session_id: str
    success: bool
    findings: list[FindingPayload]      # 漏洞發現
    trace_records: list[TraceRecord]    # 執行追蹤
    anomalies: list[str]                # 異常記錄
    metrics: PlanExecutionMetrics       # 執行指標
    start_time: datetime
    end_time: datetime
    error_message: str | None
```

**統計數據**:
- 文件大小: 771 行
- 主要功能: 攻擊計劃執行
- 支持: RabbitMQ 任務分發
- 會話管理: ✅
- 追蹤記錄: ✅ (TraceLogger)
- 異步執行: ✅

**核心能力**:
- ✅ 多步驟攻擊鏈執行
- ✅ 會話生命週期管理
- ✅ RabbitMQ 任務分發
- ✅ 實時追蹤記錄
- ✅ 結果聚合與報告
- ✅ 超時控制
- ✅ 沙箱模式支持

---

#### 3.2 執行層其他組件

**execution/ 目錄結構**:
```
services/core/aiva_core/execution/
├── plan_executor.py              # 主執行器 (771 行)
├── task_generator.py             # 任務生成器
├── task_queue_manager.py         # 任務隊列管理
├── attack_plan_mapper.py         # 計劃映射器
├── execution_status_monitor.py   # 狀態監控
└── trace_logger.py               # 追蹤記錄器
```

**組件職責**:

| 組件 | 職責 | 關鍵功能 |
|------|------|----------|
| **PlanExecutor** | 主執行器 | 執行攻擊計劃，協調所有組件 |
| **TaskGenerator** | 任務生成 | 將攻擊步驟轉換為具體任務 |
| **TaskQueueManager** | 隊列管理 | 管理任務隊列，優先級排序 |
| **AttackPlanMapper** | 計劃映射 | 映射高層計劃到底層任務 |
| **ExecutionStatusMonitor** | 狀態監控 | 實時監控執行狀態 |
| **TraceLogger** | 追蹤記錄 | 記錄執行過程，支持回溯 |

---

### Layer 4: RAG & Learning - 知識增強層 📚

#### 4.1 RAG Engine - 知識檢索增強

**相關文件**:
- `services/core/aiva_core/rag/rag_engine.py`
- `services/core/aiva_core/rag/knowledge_base.py`
- `services/core/aiva_core/rag/demo_rag_integration.py`

**增強規劃的方式**:

```python
# 示例：使用 RAG 增強攻擊計劃
async def generate_attack_plan(
    self, 
    target: dict, 
    context: dict
) -> AttackPlan:
    """使用 RAG 生成增強的攻擊計劃"""
    
    # 1. 檢索相關知識
    knowledge = await self.rag_engine.retrieve(
        query=f"攻擊計劃生成：{target['type']}",
        top_k=5
    )
    
    # 2. 整合知識生成計劃
    plan = await self._generate_plan_with_knowledge(
        target=target,
        context=context,
        knowledge=knowledge
    )
    
    return plan
```

**支持的知識類型**:
1. 歷史攻擊模式
2. 漏洞利用技術
3. 工具使用經驗
4. 成功/失敗案例
5. 目標特徵識別
6. 防禦繞過技巧
7. 最佳實踐指南

#### 4.2 Experience Manager - 經驗管理

**功能**:
- 存儲執行經驗
- 檢索相似場景
- 學習成功模式
- 避免失敗路徑

#### 4.3 Training Orchestrator - 訓練編排

**文件位置**: `services/core/aiva_core/training/training_orchestrator.py`

**與規劃的關係**:
```python
async def _generate_ai_attack_plan(
    self, 
    scenario, 
    rag_context: dict[str, Any]
):
    """基於場景和 RAG 上下文生成 AI 攻擊計劃"""
    
    # 使用 RAG 增強計劃生成
    attack_plan = await self.rag_engine.enhance_attack_plan(
        scenario=scenario,
        context=rag_context
    )
    
    return attack_plan
```

**持續學習循環**:
```
執行攻擊計劃
    ↓
收集執行結果
    ↓
存儲為經驗
    ↓
訓練模型
    ↓
改進決策
    ↓
生成更好的計劃
```

---

## 🔄 完整的規劃決策流程

### 端到端流程示例：SQL 注入攻擊規劃

```
用戶輸入: "測試目標網站的 SQL 注入漏洞"
    ↓
【Layer 1: AI Commander】
├─ 任務分析: AITaskType.ATTACK_PLANNING
├─ 選擇組件: BIO_NEURON_AGENT + RAG_ENGINE
└─ 初始化上下文
    ↓
【Layer 2: Decision & Planning】
├─ EnhancedDecisionAgent
│   ├─ 分析上下文 (DecisionContext)
│   │   ├─ risk_level: MEDIUM
│   │   ├─ available_tools: ["sqlmap", "havij", "manual_test"]
│   │   └─ target_info: {url: "http://example.com"}
│   ├─ 應用決策規則
│   │   └─ web_service_detected → WEB_ATTACK
│   └─ 生成決策 (Decision)
│       ├─ action: "sql_injection_test"
│       ├─ params: {tool: "sqlmap", level: 2}
│       └─ confidence: 0.85
│
├─ SkillGraph
│   ├─ 查找技能路徑
│   │   ├─ start: "web_recon"
│   │   ├─ goal: "sql_injection_exploit"
│   │   └─ path: [
│   │       "web_recon",           # Web 偵察
│   │       "parameter_discovery",  # 參數發現
│   │       "sql_injection_test",  # SQL 注入測試
│   │       "data_extraction"      # 數據提取
│   │   ]
│   ├─ 評估路徑
│   │   ├─ success_probability: 0.78
│   │   └─ estimated_time: 120s
│   └─ 推薦工具序列
│       └─ ["nikto", "sqlmap", "sqlmap"]
│
└─ ExecutionPlanner
    └─ 創建執行計劃
        ├─ plan_id: "plan_1730956800_12345"
        ├─ steps: [
        │     {type: "validate_target", handler: "target_validator"},
        │     {type: "web_recon", handler: "nikto"},
        │     {type: "sql_test", handler: "sqlmap"},
        │     {type: "data_extract", handler: "sqlmap"},
        │     {type: "generate_report", handler: "report_generator"}
        │ ]
        ├─ estimated_time: 120s
        └─ resources_required: ["sqlmap", "nikto"]
    ↓
【Layer 4: RAG Enhancement】
├─ RAG Engine 檢索
│   ├─ 查詢: "SQL injection attack planning for login forms"
│   └─ 檢索結果 (top 5):
│       ├─ "SQL injection bypass techniques"
│       ├─ "Common SQL injection patterns in PHP"
│       ├─ "WAF bypass strategies"
│       ├─ "Error-based SQL injection examples"
│       └─ "Time-based blind SQL injection"
│
└─ 增強計劃
    └─ 添加知識驅動的參數
        ├─ sqlmap_options: ["--risk=2", "--level=3", "--tamper=space2comment"]
        └─ bypass_techniques: ["union-based", "error-based", "time-based"]
    ↓
【Layer 3: Plan Execution】
└─ PlanExecutor
    ├─ 創建會話: session_abc123
    ├─ 執行步驟 1: validate_target ✅
    ├─ 執行步驟 2: web_recon (nikto) ✅
    │   └─ 發現: 登錄表單在 /login.php
    ├─ 執行步驟 3: sql_test (sqlmap) ✅
    │   └─ 發現: SQL 注入漏洞 (Boolean-based)
    ├─ 執行步驟 4: data_extract (sqlmap) ✅
    │   └─ 提取: 數據庫名稱、表名稱
    ├─ 執行步驟 5: generate_report ✅
    │   └─ 生成: 漏洞報告 PDF
    └─ 返回結果
        └─ PlanExecutionResult
            ├─ success: true
            ├─ findings: [SQLiFinding(...)]
            └─ metrics: {duration: 118s, steps_completed: 5}
    ↓
【Layer 4: Learning Feedback】
└─ 經驗存儲
    ├─ 成功案例記錄
    ├─ 更新技能成功率
    │   └─ sql_injection_test: 0.78 → 0.79
    └─ 訓練模型
        └─ 改進未來的 SQL 注入攻擊規劃
```

---

## 📊 規劃能力統計總表

### 代碼規模

| 層次/組件 | 文件 | 代碼行數 | 主要類別 | 核心功能 |
|----------|------|----------|----------|----------|
| **Layer 1: AI Commander** | ai_commander.py | 1,104 | 3 | 戰略指揮 |
| **Layer 2: Decision** | enhanced_decision_agent.py | 568 | 3 | 智能決策 |
| **Layer 2: Skill Graph** | skill_graph.py | 618 | 5 | 技能規劃 |
| **Layer 2: Execution Planner** | execution_planner.py | 558 | 1 | 執行編排 |
| **Layer 3: Plan Executor** | plan_executor.py | 771 | 1 | 計劃執行 |
| **Layer 3: 其他執行組件** | task_generator.py 等 | ~800 | 5 | 任務管理 |
| **Layer 4: RAG Engine** | rag_engine.py 等 | ~1,500 | 8 | 知識增強 |
| **總計** | - | **5,919** | **26** | - |

### 規劃能力矩陣

| 能力類型 | 支持程度 | 實現組件 | 代碼行數 |
|---------|---------|---------|----------|
| **戰略規劃** | ✅✅✅ | AICommander | 1,104 |
| **決策制定** | ✅✅✅ | EnhancedDecisionAgent | 568 |
| **技能路徑規劃** | ✅✅✅ | SkillGraph | 618 |
| **任務編排** | ✅✅✅ | ExecutionPlanner | 558 |
| **計劃執行** | ✅✅✅ | PlanExecutor | 771 |
| **知識增強** | ✅✅✅ | RAG Engine | ~1,500 |
| **經驗學習** | ✅✅ | ExperienceManager | ~300 |
| **風險評估** | ✅✅✅ | RiskAssessmentEngine | 380 |
| **跨語言協調** | ✅✅ | MultiLanguageAICoordinator | ~500 |

**圖例**: ✅✅✅ 完整支持 | ✅✅ 大部分支持 | ✅ 基本支持

---

## 💡 規劃能力評估

### 優勢 ⭐

1. **完整的四層架構** ⭐⭐⭐
   - 從戰略到戰術全覆蓋
   - 每層職責清晰
   - 無重疊或缺失

2. **知識驅動的智能規劃** ⭐⭐⭐
   - RAG 增強決策質量
   - 經驗學習持續改進
   - 技能圖譜指導路徑選擇

3. **多維度決策能力** ⭐⭐⭐
   - 風險感知決策
   - 工具選擇優化
   - 失敗自動調整策略

4. **跨語言統一調度** ⭐⭐⭐
   - Python/Go/Rust/TypeScript 協調
   - 統一的任務編排
   - 語言無關的規劃邏輯

5. **完善的追蹤與監控** ⭐⭐
   - TraceLogger 記錄執行過程
   - ExecutionStatusMonitor 實時監控
   - 支持回溯分析

### 可改進之處 🔹

1. **計劃生成的自動化程度** 🔹
   - **現狀**: 部分依賴預定義規則
   - **建議**: 增加基於深度學習的計劃生成
   - **優點**: 更靈活適應新場景

2. **技能圖的動態更新** 🔹
   - **現狀**: 需要手動構建技能圖
   - **建議**: 自動從執行結果更新技能關係
   - **優點**: 自動發現新的攻擊模式

3. **多目標優化** 🔹
   - **現狀**: 主要優化成功率和時間
   - **建議**: 增加隱蔽性、資源消耗等多目標
   - **優點**: 更符合實際滲透測試需求

4. **分布式執行規劃** 🔹
   - **現狀**: 單機執行為主
   - **建議**: 支持分布式並行攻擊規劃
   - **優點**: 提高大規模測試效率

---

## 🎯 使用場景示例

### 場景 1: 自動滲透測試

```python
# 用戶輸入
target = "http://example.com"
goal = "找出並利用所有高危漏洞"

# AI Commander 分析並規劃
result = await ai_commander.execute_task(
    task_type=AITaskType.ATTACK_PLANNING,
    target=target,
    goal=goal
)

# 自動生成的攻擊計劃
attack_plan = result.plan
# steps:
# 1. Port Scan (nmap) - 發現開放端口
# 2. Service Detection - 識別服務版本
# 3. Vulnerability Scan (nikto) - 掃描 Web 漏洞
# 4. SQL Injection Test (sqlmap) - 測試 SQL 注入
# 5. XSS Test (xsser) - 測試 XSS
# 6. SSRF Test (custom) - 測試 SSRF
# 7. Exploit Execution - 利用高危漏洞
# 8. Report Generation - 生成報告
```

### 場景 2: 適應性攻擊

```python
# 初始計劃失敗
context = DecisionContext()
context.attempts_without_success = 3
context.discovered_vulns = []
context.target_info = {"waf": "Cloudflare"}

# 智能決策調整策略
decision = await enhanced_decision_agent.make_decision(context)
# decision.action = "CHANGE_STRATEGY"
# decision.params = {
#     "new_approach": "stealth_mode",
#     "tools": ["manual_test"],
#     "techniques": ["waf_bypass", "slow_scan"]
# }
```

### 場景 3: 技能路徑優化

```python
# 找出最優攻擊路徑
path = await skill_graph_analyzer.find_optimal_path(
    start_capability="web_recon",
    goal_capability="privilege_escalation"
)

# 返回:
# SkillPath(
#     nodes=["web_recon", "sqli_test", "database_enum", 
#            "credential_harvest", "ssh_access", "privilege_escalation"],
#     total_weight=6.5,
#     estimated_time=300s,
#     success_probability=0.72
# )
```

---

## 📈 未來發展建議

### 短期優化（1-3 個月）

1. **增強 RAG 知識庫** 📚
   - 添加更多攻擊模式和技術
   - 整合公開漏洞數據庫
   - 支持自定義知識注入

2. **優化決策規則引擎** 🧠
   - 增加更多決策規則
   - 支持規則優先級動態調整
   - 添加規則衝突解決機制

3. **改進技能圖構建** 🔗
   - 自動化技能關係發現
   - 基於執行結果更新技能成功率
   - 支持技能版本管理

### 中期發展（3-6 個月）

1. **深度學習規劃器** 🤖
   - 使用強化學習訓練規劃模型
   - 端到端的攻擊序列生成
   - 自適應參數優化

2. **分布式執行支持** 🌐
   - 支持多節點並行執行
   - 分布式任務調度
   - 跨節點狀態同步

3. **高級對抗規劃** 🛡️
   - 防禦感知的攻擊規劃
   - WAF/IDS 繞過策略
   - 隱蔽性優化

### 長期願景（6-12 個月）

1. **自主學習系統** 🎓
   - 從成功/失敗案例自動學習
   - 無監督發現新攻擊模式
   - 持續演進的攻擊策略

2. **多目標優化框架** 🎯
   - 平衡成功率、時間、隱蔽性、資源
   - 帕累托最優解搜索
   - 用戶偏好學習

3. **協作式 AI 規劃** 👥
   - 多 AI Agent 協同規劃
   - 專家系統與神經網絡融合
   - 人機協作規劃界面

---

## 📝 結論

### 總體評價 ⭐⭐⭐⭐⭐ (5/5)

AIVA 的 AI 規劃能力**非常完整和成熟**，具有：

1. ✅ **完整的四層規劃架構**
   - 戰略、決策、執行、學習全覆蓋
   - 總計近 6,000 行核心規劃代碼

2. ✅ **多樣化的規劃能力**
   - 攻擊計劃生成
   - 智能決策制定
   - 技能路徑規劃
   - 任務異步編排
   - 知識增強決策

3. ✅ **知識驅動的智能化**
   - RAG 增強規劃質量
   - 經驗學習持續改進
   - 技能圖譜指導決策

4. ✅ **工程實現完善**
   - 異步執行支持
   - 完整的追蹤監控
   - 會話生命週期管理
   - RabbitMQ 任務分發

### 與原始聲稱的對比

**原始聲稱**: "許多本應屬於『AI 大腦』核心模組的功能，被錯誤地分散到了其他模組"

**實際情況**: ❌ **完全不符**

- ✅ 所有規劃功能都正確地在 `services/core` 模組
- ✅ 分層架構清晰，職責劃分合理
- ✅ 沒有功能錯誤放置或重複
- ✅ 符合軟件工程最佳實踐

### 建議

1. **繼續保持現有架構** ✅
   - 不需要大規模重構
   - 架構設計優秀

2. **漸進式優化** ✅
   - 增強 RAG 知識庫
   - 添加深度學習規劃器
   - 支持分布式執行

3. **文檔完善** 📝
   - 添加規劃流程圖
   - 編寫使用示例
   - 說明各層職責

---

**報告完成時間**: 2025年11月7日  
**下一步**: 可以基於此報告進行具體的優化工作

**報告作者**: AIVA 架構分析系統
