# AIVA Core 完整架構深度分析

**分析日期**: 2026-01-28  
**分析範圍**: 整個 `services/core/aiva_core` 模組及所有子模組  
**分析方法**: 靜態代碼分析 + 文檔研究 + 數據流追蹤  
**分析目標**: 理解整體設計理念、模組職責、13 步驟執行流程

---

## 📋 目錄

- [執行摘要](#執行摘要)
- [AIVA Core 五大核心模組](#aiva-core-五大核心模組)
- [Task Planning 深度分析](#task-planning-深度分析)
- [Commander 子模組完整解析](#commander-子模組完整解析)
- [AttackCoordinator 詳細分析](#attackcoordinator-詳細分析)
- [13 步驟流程中的模組協作](#13-步驟流程中的模組協作)
- [設計理念與架構原則](#設計理念與架構原則)
- [發現的問題與建議](#發現的問題與建議)

---

## 📊 執行摘要

### AIVA Core 整體定位

**AIVA Core** 是整個 AIVA 系統的「**大腦**」，負責：
- 🧠 **AI 認知決策** - 使用 5M 神經網路進行智能決策
- 📋 **任務規劃編排** - 將高層次目標分解為可執行步驟
- 🎯 **能力管理調度** - 管理 840+ 個攻擊/掃描能力
- 🏗️ **基礎設施服務** - 提供 API、消息、存儲等支援
- 🔍 **自我分析探索** - 動態發現和分類自身能力

### 架構特點

```
用戶輸入
   ↓
Service Backbone (API/消息)
   ↓
Task Planning (任務規劃)
   ├── Commander (AI 指揮)
   │   └── AttackCoordinator (攻擊協調) ← 本次重點
   ├── Planner (計劃生成)
   └── Executor (計劃執行)
   ↓
Cognitive Core (AI 決策)
   ├── Neural (5M 模型)
   ├── Decision (決策引擎)
   └── RAG (知識檢索)
   ↓
Core Capabilities (能力執行)
   ├── Orchestration (兩階段掃描)
   ├── Attack (攻擊執行)
   └── Analysis (代碼分析)
   ↓
Internal Exploration (內部探索)
   └── FlowExecutor (執行 313-318 flows)
```

---

## 🏛️ AIVA Core 五大核心模組

### 1. 🧠 Cognitive Core - 認知核心 (48 文件, 7 子模組)

**設計理念**: **AI 的大腦，所有智能決策的來源**

#### 子模組結構
```
cognitive_core/
├── neural/                    # 神經網路推理層
│   ├── real_ai_core.py       # 5M 參數 PyTorch 模型
│   └── inference_engine.py   # 推理引擎
│
├── decision/                  # 決策支援層 ⭐
│   ├── enhanced_decision_agent.py  # Bug Bounty 四大決策方法
│   ├── capability_orchestrator.py  # 能力編排器
│   └── strategy_selector.py        # 策略選擇器
│
├── rag/                       # 知識檢索層
│   ├── vector_store.py       # 384 維語意向量
│   ├── experience_db.py      # 經驗數據庫
│   └── sync_experiences.py   # 經驗同步工具
│
├── learning_system/          # 學習系統
│   ├── experience_manager.py
│   ├── model_trainer.py
│   └── knowledge/            # 嵌入式知識庫
│       ├── sqli_knowledge.json
│       ├── xss_knowledge.json
│       └── cve_database.json
│
├── anti_hallucination/       # 反幻覺機制
│   └── v2_reflection_engine.py  # v2.1 去語意化引擎
│
└── internal_loop_connector.py  # 與 Internal Exploration 連接器
```

#### 核心職責
1. **AI 決策** - 四大 Bug Bounty 決策方法：
   - `decide_scan_strategy()` - 智能掃描工具選擇（步驟 6）
   - `decide_phase1_strategy()` - Phase1 深度掃描決策（步驟 6）
   - `decide_phase2_targets()` - 攻擊目標優先級排序（步驟 9）
   - `evaluate_phase2_results()` - 結果評估和後續行動（步驟 11）

2. **神經網路推理** - 5M 參數模型，384 維語意向量

3. **知識檢索** - RAG 增強，經驗學習，嵌入式安全知識庫

4. **反幻覺保護** - v2.1 去語意化引擎，12/12 驗證測試通過

#### 關鍵發現
- ✅ **已完成 Bug Bounty 四大決策方法**（2026-01-21）
- ✅ 整合到 `AttackCoordinator.process_scan_command()` (L561-674)
- ✅ 整合到兩階段掃描編排器

---

### 2. 📋 Task Planning - 任務規劃 (28 文件, 4 子模組)

**設計理念**: **將高層次目標轉換為可執行的攻擊計劃**

#### 子模組結構
```
task_planning/
├── commander/                 # AI 指揮協調 ⭐⭐⭐
│   ├── __init__.py           # CommanderCoordinator (統一入口)
│   ├── attack_coordinator.py # AttackCoordinator (本次重點)
│   ├── plan_builder.py       # 攻擊計劃建構器
│   ├── strategy_engine.py    # 策略決策引擎
│   ├── capability_manager.py # 能力選單管理
│   ├── learning_adapter.py   # 學習系統適配
│   └── types.py              # AITaskType, AIComponent 枚舉
│
├── planner/                   # 計劃生成器
│   ├── execution_planner.py  # 執行計劃生成
│   ├── task_generator.py     # 任務生成器
│   └── tool_selector.py      # 工具選擇器
│
├── executor/                  # 計劃執行器
│   ├── plan_executor.py      # 主執行器
│   ├── task_executor.py      # 任務執行器
│   └── status_tracker.py     # 狀態追蹤器
│
├── persistence/               # 任務持久化
│   ├── session_store.py
│   └── checkpoint_manager.py
│
├── unified_executor.py        # 統一攻擊執行器 (841 行)
├── command_builder.py         # AI 決策 → CLI 命令轉換
├── command_router.py          # 智能命令路由
├── dispatcher.py              # 任務派發器（整合 internal_exploration）
└── mode_manager.py            # 執行模式管理（sandbox/production）
```

#### 核心職責
1. **任務接收與分解** (步驟 0-2)
   - 接收用戶輸入 → 解析意圖 → 生成攻擊計劃
   
2. **AI 指揮協調** (步驟 6, 9, 11)
   - CommanderCoordinator 統一入口
   - AttackCoordinator 協調攻擊執行
   - 調用 Cognitive Core 決策引擎

3. **計劃執行** (步驟 3-4, 7, 10)
   - PlanExecutor 順序執行 AttackPlan.steps
   - 支持 Phase 0/1/2 三階段掃描
   - 狀態追蹤和錯誤處理

4. **CLI 命令執行**
   - 使用 `subprocess` 執行外部工具
   - 異步執行管理（asyncio）
   - 輸出解析和結果收集

#### 關鍵發現
- ✅ Commander 子模組是核心協調層
- ⚠️ **AttackCoordinator 初始化參數錯誤**（已發現但未修復）
- ✅ 與 Cognitive Core 整合完整
- ✅ 與 Internal Exploration 整合完整

---

### 3. 🎯 Core Capabilities - 核心能力 (21 文件, 8 子模組)

**設計理念**: **管理和執行所有攻擊/掃描能力**

#### 子模組結構
```
core_capabilities/
├── orchestration/            # 編排層 ⭐
│   ├── two_phase_scan_orchestrator.py  # 兩階段掃描 (引用 AttackCoordinator)
│   └── capability_orchestrator.py      # 能力編排器
│
├── attack/                   # 攻擊層
│   ├── exploit_orchestrator.py
│   └── payload_generator.py
│
├── analysis/                 # 分析層
│   ├── bizlogic_scanner.py  # 業務邏輯掃描
│   └── code_analyzer.py
│
├── cli/                      # CLI 接口
│   └── aiva_cli.py          # 整合 FlowExecutor
│
├── dialog/                   # 對話層
│   └── ai_menu.py           # 智能選單 (696 行)
│
├── ingestion/               # 數據攝取
├── output/                  # 輸出處理
└── processing/              # 數據處理
```

#### 核心職責
1. **能力註冊管理** - 840+ 個真實註冊的能力
2. **兩階段掃描編排** - Phase 0/1/2 執行流程
3. **攻擊編排** - 協調多步驟攻擊執行
4. **CLI 整合** - 與 Internal Exploration FlowExecutor 整合

#### 關鍵發現
- ✅ `TwoPhaseScanOrchestrator` 直接調用 `AttackCoordinator`
- ✅ 整合 Bug Bounty 決策方法
- ✅ CLI 命令執行架構完整

---

### 4. 🧭 Internal Exploration - 內部探索 (16 文件, 2 子模組)

**設計理念**: **自我分析和能力發現，動態執行系統**

#### 子模組結構
```
internal_exploration/
├── python_tools/             # Python AST 分析工具
│   ├── aiva_flow_analyzer.py           # 數據流分析
│   ├── aiva_internal_classifier.py     # 內部模組分類
│   ├── aiva_external_classifier.py     # 外部模組分類
│   └── aiva_cli_implementation.py      # FlowExecutor 實現 ⭐
│
├── self_healing/            # 自我修復
│   └── diagnostic_tools.py
│
├── go_tools/                # Go AST 分析
│   └── go2mermaid.go
│
├── rust_tools/              # Rust AST 分析
│   └── src/main.rs
│
├── typescript_tools/        # TypeScript AST 分析
│   └── ts2mermaid.ts
│
└── classification_data/     # 分類結果存儲
    └── latest_classification.json  # 系統指針
```

#### 核心職責
1. **能力發現** - 多語言 AST 分析，自動分類模組
2. **數據流分析** - 追蹤 313-318 個 flows
3. **FlowExecutor** - 動態執行 Python/Go/Rust/TypeScript 模組
4. **自我診斷** - 檢測系統健康狀態

#### 關鍵發現
- ✅ FlowExecutor 是實際執行引擎（Line 99-650）
- ✅ 被 `task_planning/dispatcher.py` 調用（21 個整合點）
- ✅ 被 `core_capabilities/cli/aiva_cli.py` 調用

---

### 5. 🏗️ Service Backbone - 服務骨幹 (37 文件, 5 子模組)

**設計理念**: **基礎設施服務層**

#### 子模組結構
```
service_backbone/
├── api/                     # API 服務
│   ├── gateway.py          # CommandCenter 入口
│   ├── unified_function_caller.py
│   └── ai_service.py
│
├── messaging/              # 消息系統
│   ├── task_dispatcher.py  # 任務派發
│   └── rabbitmq_client.py
│
├── coordination/           # 組件協調
│   └── ai_manager.py
│
├── performance/            # 效能監控
│   ├── health_check.py
│   └── diagnose.py
│
└── storage/               # 存儲服務
    └── vector_store_client.py
```

#### 核心職責
1. **API 網關** - CommandCenter 接收用戶輸入
2. **消息代理** - RabbitMQ 異步任務派發
3. **健康檢查** - 系統診斷和監控
4. **存儲服務** - 向量數據庫、會話存儲

---

## 🎯 Task Planning 深度分析

### 整體架構設計理念

**Task Planning** 是 AIVA 的「**執行大腦**」，負責：
1. **接收任務** - 從 Service Backbone 接收任務
2. **AI 決策** - 調用 Cognitive Core 進行智能決策
3. **分解計劃** - 將目標轉換為可執行步驟
4. **協調執行** - 編排多步驟攻擊流程
5. **結果收集** - 整合執行結果並返回

### 根目錄核心組件 (6 個文件)

#### 1. unified_executor.py (841 行) ⭐⭐⭐

**職責**: 統一攻擊執行器，靶場與實戰統一

```python
class UnifiedAttackExecutor:
    """統一攻擊執行器
    
    整合:
    - Bug Bounty 決策引擎
    - 持續學習系統
    - 靶場/實戰雙模式
    """
    
    async def execute(
        self,
        target: str,
        objective: str,
        scenario: Optional[str] = None,
        constraints: Optional[Dict] = None
    ) -> ExecutionResult:
        """執行攻擊任務
        
        流程:
        1. 解析目標和意圖
        2. 調用決策引擎選擇策略
        3. 執行攻擊計劃
        4. 收集結果和學習數據
        5. 返回標準化結果
        """
```

**關鍵特性**:
- ✅ 整合 EnhancedDecisionAgent（四大決策方法）
- ✅ 支持持續學習（learning_info）
- ✅ 雙模式執行（sandbox/production）

#### 2. command_builder.py

**職責**: AI 決策 → CLI 命令轉換器

```python
class CommandBuilder:
    """將 AI 決策轉換為可執行的 CLI 命令"""
    
    def build_scan_command(
        self,
        tool: str,  # "nmap", "masscan", etc.
        target: str,
        strategy: str,  # "fast", "balanced", etc.
        params: dict
    ) -> str:
        """生成掃描命令
        
        範例:
        tool="nmap", strategy="fast"
        → "nmap -sS -T4 -p- example.com"
        """
```

#### 3. command_router.py

**職責**: 智能命令路由器

```python
class CommandRouter:
    """根據任務類型路由到正確的執行器"""
    
    def route(self, task_type: str, context: dict):
        """路由規則:
        - "scan" → MultiEngineCoordinator
        - "attack" → AttackCoordinator
        - "exploit" → AttackExecutor
        """
```

#### 4. dispatcher.py ⭐

**職責**: 任務派發器，整合 Internal Exploration

```python
class PlanningDispatcher:
    """任務規劃統一派發器
    
    整合點:
    - internal_exploration (21 個調用點)
    - RabbitMQ 異步消息
    - 跨模組通信
    """
```

#### 5. mode_manager.py

**職責**: 執行模式管理器

```python
class ModeManager:
    """管理執行模式
    
    模式:
    - sandbox: 靶場測試
    - production: 實戰執行
    - aggressive: 激進攻擊
    """
```

---

## 🎖️ Commander 子模組完整解析

### 設計理念

**Commander** 是 Task Planning 的「**AI 指揮官**」，負責：
1. **統一協調** - CommanderCoordinator 統一入口
2. **AI 決策整合** - 調用 Cognitive Core 決策引擎
3. **攻擊協調** - AttackCoordinator 編排攻擊執行
4. **策略選擇** - StrategyEngine 決定執行策略
5. **學習適配** - LearningAdapter 記錄經驗數據

### 子模組架構 (8 個文件)

#### 1. __init__.py - CommanderCoordinator (統一入口) ⭐⭐⭐

**設計模式**: Facade Pattern + Lazy Loading

```python
class CommanderCoordinator:
    """AI 指揮官協調器
    
    職責:
    1. 提供統一的 execute_command() 接口
    2. 延遲加載子模組（Lazy Loading）
    3. 路由任務到正確的子模組
    """
    
    def __init__(
        self,
        data_directory: Optional[Path] = None,
        learning_enabled: bool = True,
    ):
        # 延遲初始化所有子模組
        self._capability_manager: Optional[CapabilityManager] = None
        self._plan_builder: Optional[PlanBuilder] = None
        self._strategy_engine: Optional[StrategyEngine] = None
        self._attack_coordinator: Optional[AttackCoordinator] = None  # ⚠️ 問題在這
        self._learning_adapter: Optional[LearningAdapter] = None
    
    async def execute_command(
        self,
        task_type: AITaskType,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """統一命令執行接口
        
        路由規則:
        - ATTACK_PLANNING → plan_builder
        - STRATEGY_DECISION → strategy_engine
        - VULNERABILITY_DETECTION → attack_coordinator ⭐
        - ATTACK_EXECUTION → attack_coordinator ⭐
        - TWO_PHASE_SCAN → attack_coordinator ⭐
        - EXPERIENCE_LEARNING → learning_adapter
        """
```

**關鍵發現**:
- ✅ 設計模式優秀（Facade + Lazy Loading）
- ⚠️ **初始化參數錯誤** - `AttackCoordinator` 參數不匹配
- ✅ 路由邏輯清晰

#### 2. types.py - 類型定義

```python
class AITaskType(str, Enum):
    """AI 任務類型"""
    # 決策類
    ATTACK_PLANNING = "attack_planning"
    STRATEGY_DECISION = "strategy_decision"
    RISK_ASSESSMENT = "risk_assessment"
    
    # 執行類
    VULNERABILITY_DETECTION = "vulnerability_detection"  # → AttackCoordinator
    EXPLOIT_EXECUTION = "exploit_execution"
    ATTACK_EXECUTION = "attack_execution"                # → AttackCoordinator
    TWO_PHASE_SCAN = "two_phase_scan"                   # → AttackCoordinator
    
    # 學習類
    EXPERIENCE_LEARNING = "experience_learning"
    CAPABILITY_QUERY = "capability_query"
```

#### 3. plan_builder.py - 攻擊計劃建構器

**設計問題**: ⚠️ 初始化參數與調用不匹配

```python
# 實際定義
class PlanBuilder:
    def __init__(
        self,
        rag_engine: Any,          # 需要 RAG 引擎
        decision_engine: Any,      # 需要決策引擎
        experience_manager: Any,   # 需要經驗管理器
        ...
    ):
        pass

# 實際調用（__init__.py:73）
self._plan_builder = PlanBuilder(
    data_directory=self.data_directory / "plans"  # ❌ 參數錯誤！
)
```

#### 4. strategy_engine.py - 策略決策引擎

**職責**: 決定執行策略（fast/balanced/comprehensive）

#### 5. capability_manager.py - 能力選單管理

**職責**: 管理和查詢可用能力清單

#### 6. learning_adapter.py - 學習系統適配

**職責**: 記錄執行經驗到學習系統

---

## ⚔️ AttackCoordinator 詳細分析

### 設計定位

**AttackCoordinator** 是 Commander 子模組中的「**攻擊執行協調器**」，職責：

1. **漏洞檢測協調** (步驟 7, 10)
   - 調用 XSS/SQLi/SSRF 檢測器
   - 使用 `httpx.AsyncClient` 抓取網頁
   - 返回標準化 `FindingPayload` 格式

2. **多引擎掃描協調** (步驟 3-4)
   - 協調 Python/TypeScript/Rust/Go 掃描引擎
   - 策略選擇（fast/balanced/comprehensive）
   - 結果整合

3. **攻擊計劃執行** (步驟 10)
   - 調用 `AttackExecutor` 執行攻擊
   - 安全模式管理（safe/testing/aggressive）
   - 步驟追蹤和錯誤處理

4. **兩階段掃描** (步驟 3-10)
   - 整合 `TwoPhaseScanOrchestrator`
   - Phase 0/1 執行協調

5. **能力查詢** (v11.0 新增)
   - 調用 `internal_loop.query_capabilities_async()`
   - RAG 語意搜尋

6. **統一攻擊接口** (v4.4.0)
   - 調用 `unified_executor.execute()`
   - 整合 Bug Bounty 決策

7. **用戶命令處理** (完整流程)
   - 解析自然語言輸入
   - AI 決策（四大決策方法）
   - 執行掃描/攻擊
   - Phase2 目標排序
   - Phase2 結果評估

### 代碼結構 (674 行)

```python
class AttackCoordinator:
    """攻擊執行協調器
    
    設計問題: ⚠️ 初始化參數錯誤
    """
    
    def __init__(
        self,
        unified_executor: Any,      # ← 需要這個
        multilang_coordinator: Any, # ← 需要這個
        internal_loop: Any,         # ← 需要這個
    ):
        """但實際調用時只傳了 data_directory ❌"""
        pass
    
    # ========== 7 個核心方法 ==========
    
    async def detect_vulnerabilities(
        self, 
        context: dict
    ) -> dict:
        """檢測漏洞（XSS/SQLi）
        
        流程:
        1. 創建 httpx.AsyncClient
        2. 執行 XSS 檢測（TraditionalXssDetector）
        3. 執行 SQLi 檢測（SqliDetector）
        4. 轉換為標準 FindingPayload 格式
        
        使用場景: 步驟 7, 10 (Phase 1/2 漏洞檢測)
        """
    
    async def coordinate_multilang(
        self,
        context: dict
    ) -> dict:
        """協調多引擎掃描
        
        流程:
        1. 初始化 MultiEngineCoordinator
        2. 選擇策略（fast/balanced/comprehensive）
        3. 執行掃描
        4. 返回結果
        
        使用場景: 步驟 3-4 (Phase 0 快速偵察)
        """
    
    async def execute_attack(
        self,
        context: dict
    ) -> dict:
        """執行攻擊計劃
        
        流程:
        1. 創建 AttackExecutor
        2. 設置執行模式（safe/testing/aggressive）
        3. 執行計劃
        4. 返回結果
        
        使用場景: 步驟 10 (Phase 2 攻擊測試)
        """
    
    async def execute_two_phase_scan(
        self,
        context: dict
    ) -> dict:
        """執行兩階段掃描
        
        流程:
        1. 創建 TwoPhaseScanOrchestrator
        2. 執行 Phase 0 + Phase 1
        3. 返回整合結果
        
        使用場景: 步驟 3-7 (完整兩階段流程)
        """
    
    async def query_capabilities(
        self,
        query: str,
        filters: dict,
        top_k: int = 5
    ) -> dict:
        """查詢自身能力
        
        流程:
        1. 調用 internal_loop.query_capabilities_async()
        2. RAG 語意搜尋
        3. 返回匹配能力
        
        使用場景: 內閉環自我探索
        """
    
    async def unified_attack(
        self,
        target: str,
        objective: str
    ) -> dict:
        """統一攻擊執行
        
        流程:
        1. 調用 unified_executor.execute()
        2. 整合 Bug Bounty 決策
        3. 持續學習
        
        使用場景: 步驟 6-11 (完整攻擊流程)
        """
    
    async def process_scan_command(
        self,
        user_input: str
    ) -> dict:
        """處理用戶掃描命令 ⭐⭐⭐
        
        流程:
        1. 解析自然語言輸入（parse_user_input_to_context）
        2. AI 決策 1: 掃描策略（decide_scan_strategy）
        3. 執行掃描（unified_executor.execute）
        4. AI 決策 2: Phase2 目標排序（decide_phase2_targets）
        5. AI 決策 3: 結果評估（evaluate_phase2_results）
        6. 返回完整結果
        
        使用場景: 步驟 0-13 (完整外閉環流程)
        整合: 四大 Bug Bounty 決策方法
        """
```

### 理論職責 vs 實際問題

#### ✅ 理論上應該負責的功能

1. **漏洞檢測協調** ✅
   - 代碼完整實現
   - XSS/SQLi 檢測器調用
   - 標準化輸出格式

2. **多引擎掃描協調** ✅
   - MultiEngineCoordinator 整合
   - 策略選擇邏輯
   - 結果聚合

3. **攻擊執行協調** ✅
   - AttackExecutor 調用
   - 模式管理
   - 步驟追蹤

4. **兩階段掃描** ✅
   - TwoPhaseScanOrchestrator 整合
   - Phase 0/1 協調

5. **Bug Bounty 決策整合** ✅
   - 四大決策方法完整調用
   - `process_scan_command()` 實現完整

6. **能力查詢** ✅
   - RAG 搜尋整合
   - internal_loop 連接

#### ❌ 實際發現的問題

**問題 1: 初始化參數不匹配（Critical）**

```python
# attack_coordinator.py:51-56 (定義)
def __init__(
    self,
    unified_executor: Any,      # 需要統一執行器
    multilang_coordinator: Any, # 需要多語言協調器
    internal_loop: Any,         # 需要內部循環連接器
):
    self.unified_executor = unified_executor
    self.multilang_coordinator = multilang_coordinator
    self.internal_loop = internal_loop

# __init__.py:87 (實際調用)
self._attack_coordinator = AttackCoordinator(
    data_directory=self.data_directory / "attacks"  # ❌ 完全錯誤！
)
```

**結果**: `TypeError: missing 3 required positional arguments`

**影響範圍**:
- ✅ CommanderCoordinator 初始化失敗
- ✅ 所有 AI 任務類型無法執行：
  - `VULNERABILITY_DETECTION`
  - `ATTACK_EXECUTION`
  - `TWO_PHASE_SCAN`
  - `CAPABILITY_QUERY`

**問題 2: PlanBuilder 也有相同問題**

```python
# plan_builder.py:17 (定義)
def __init__(
    self,
    rag_engine: Any,
    decision_engine: Any,
    experience_manager: Any,
    ...
):

# __init__.py:73 (實際調用)
self._plan_builder = PlanBuilder(
    data_directory=self.data_directory / "plans"  # ❌ 也錯誤！
)
```

**問題 3: 整個 Commander 子模組可能無法運行**

---

## 🔄 13 步驟流程中的模組協作

### 完整數據流追蹤

```
步驟 0: 用戶輸入
  └─> Service Backbone (API Gateway)
      └─> CommandCenter.route_command()

步驟 1: Core 接收分析
  └─> Task Planning (CommanderCoordinator)
      └─> execute_command(AITaskType.ATTACK_PLANNING)
          └─> PlanBuilder.build_attack_plan()  ⚠️ 初始化失敗

步驟 2: Coordinator 分解任務
  └─> PlanExecutor.execute_plan(AttackPlan)
      └─> 順序執行 plan.steps

步驟 3-4: Phase 0 快速偵察
  └─> AttackCoordinator.coordinate_multilang()  ⚠️ 無法調用
      └─> MultiEngineCoordinator
          ├─> Python Engine (masscan)
          ├─> TypeScript Engine (subfinder)
          ├─> Rust Engine (rustscan)
          └─> Go Engine (gobuster)

步驟 6: AI 決策 1（是否需要深掃）
  └─> Cognitive Core (EnhancedDecisionAgent)
      └─> decide_phase1_strategy(phase0_result)  ✅
          └─> 返回 deep_scan_required: bool

步驟 7: Phase 1 深度掃描
  └─> AttackCoordinator.detect_vulnerabilities()  ⚠️ 無法調用
      ├─> TraditionalXssDetector.execute()
      └─> SqliDetector.detect_sqli()

步驟 9: AI 決策 2（攻擊目標選擇）
  └─> Cognitive Core (EnhancedDecisionAgent)
      └─> decide_phase2_targets(phase1_result)  ✅
          └─> 返回 targets: List[Target] (Tier 1-3 排序)

步驟 10: Phase 2 攻擊測試
  └─> AttackCoordinator.execute_attack()  ⚠️ 無法調用
      └─> AttackExecutor.execute_plan_with_ai_analysis()

步驟 11: AI 決策 3（結果評估）
  └─> Cognitive Core (EnhancedDecisionAgent)
      └─> evaluate_phase2_results(phase2_results)  ✅
          └─> 返回 next_action: str (SUBMIT_REPORT/CONTINUE_DEEP_DIVE)

步驟 12-13: 結果返回
  └─> Service Backbone (API Gateway)
      └─> 返回給用戶
```

### 關鍵協作點

1. **Task Planning ↔ Cognitive Core**
   - CommanderCoordinator 調用 EnhancedDecisionAgent
   - 四大決策方法整合
   - ✅ 整合完整

2. **Task Planning ↔ Core Capabilities**
   - AttackCoordinator 調用 TwoPhaseScanOrchestrator
   - AttackCoordinator 調用 AttackExecutor
   - ⚠️ 初始化問題導致無法調用

3. **Task Planning ↔ Internal Exploration**
   - Dispatcher 整合 FlowExecutor（21 個調用點）
   - AttackCoordinator 調用 internal_loop
   - ✅ 整合完整

4. **Core Capabilities ↔ Cognitive Core**
   - TwoPhaseScanOrchestrator 調用 EnhancedDecisionAgent
   - CapabilityOrchestrator 編排執行
   - ✅ 整合完整

---

## 🏗️ 設計理念與架構原則

### 五大核心設計原則

#### 1. 單一數據源（Single Source of Truth）

**原則**: 所有共享數據結構定義在 `aiva_common`

```
aiva_common/
├── schemas/         # 標準化數據結構
│   ├── tasks.py    # AttackPlan, AttackStep, FunctionTaskPayload
│   ├── findings.py # FindingPayload, Vulnerability, FindingEvidence
│   └── results.py  # ExecutionResult, ScanResult
│
├── enums/          # 標準枚舉
│   ├── task_type.py    # VulnerabilityType, Severity
│   └── status.py       # TaskStatus, ScanStatus
│
└── utils/          # 通用工具
```

**優點**:
- ✅ 避免數據重複定義
- ✅ 保證介面一致性
- ✅ 便於維護和升級

#### 2. 有錯就報錯（Fail Fast）

**原則**: 不隱藏錯誤，不使用降級邏輯

```python
# ✅ 好的實踐
try:
    from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
except ImportError as e:
    raise ImportError(
        "❌ 缺少必要依賴 MultiEngineCoordinator\n"
        "請確認模組已實現\n"
        f"原始錯誤: {e}"
    ) from e

# ❌ 壞的實踐（不要使用）
try:
    from xxx import XXX
except ImportError:
    XXX = None  # 靜默降級，隱藏問題
```

**優點**:
- ✅ 快速暴露問題
- ✅ 防止錯誤傳播
- ✅ 便於調試

#### 3. 事件驅動（Event-Driven）

**原則**: 使用 `asyncio.Future` 取代輪詢等待

```python
# ✅ 事件驅動
async def execute_task(task):
    future = asyncio.Future()
    
    async def on_complete(result):
        future.set_result(result)
    
    task.add_callback(on_complete)
    return await future

# ❌ 輪詢等待（不要使用）
while not task.is_complete():
    await asyncio.sleep(0.1)
```

**優點**:
- ✅ 減少 CPU 佔用
- ✅ 提高響應速度
- ✅ 便於並發控制

#### 4. 模組化設計（Modular Architecture）

**原則**: 五大模組獨立但協同工作

```
各模組職責清晰:
- Cognitive Core: AI 決策
- Task Planning: 任務編排
- Core Capabilities: 能力執行
- Internal Exploration: 自我探索
- Service Backbone: 基礎服務

介面標準化:
- 統一使用 aiva_common 數據結構
- 相對路徑 import
- 明確的依賴關係
```

#### 5. 真實執行（Real Execution）

**原則**: 所有能力真實註冊，無模擬數據

```python
# ✅ 真實註冊 840+ 個能力
from services.integration.capability import register_all_capabilities
register_all_capabilities()

# ❌ 不使用 mock/stub（測試除外）
```

**優點**:
- ✅ 確保功能可用
- ✅ 減少環境差異
- ✅ 提高可靠性

---

## 🚨 發現的問題與建議

### Critical 級別問題

#### 問題 1: Commander 子模組初始化全面失敗

**影響**:
- ❌ `AttackCoordinator` 無法實例化
- ❌ `PlanBuilder` 無法實例化
- ❌ 所有 AI 任務類型無法執行
- ❌ 整個 13 步驟流程無法運行

**根本原因**:
```python
# CommanderCoordinator.__init__.py 中的錯誤模式
@property
def attack_coordinator(self) -> AttackCoordinator:
    if self._attack_coordinator is None:
        # ❌ 傳入錯誤的參數
        self._attack_coordinator = AttackCoordinator(
            data_directory=self.data_directory / "attacks"
        )
    return self._attack_coordinator

# 應該是（需要確認依賴來源）:
self._attack_coordinator = AttackCoordinator(
    unified_executor=self._get_unified_executor(),
    multilang_coordinator=self._get_multilang_coordinator(),
    internal_loop=self._get_internal_loop()
)
```

**修復建議**:

**方案 A: 修正依賴注入（推薦）**
1. 在 `CommanderCoordinator.__init__()` 中初始化依賴
2. 傳遞正確的參數給各子模組
3. 確保依賴鏈完整

**方案 B: 修改子模組接受 data_directory**
1. 修改 `AttackCoordinator.__init__()` 簽名
2. 內部創建預設依賴
3. 向後兼容

**方案 C: 使用工廠模式**
1. 創建 `CommanderFactory` 
2. 統一管理依賴創建
3. 確保參數正確

---

### High 級別問題

#### 問題 2: 缺少依賴驗證

**現象**:
```bash
python -c "from services.core.aiva_core.task_planning.commander.attack_coordinator import AttackCoordinator"
# ❌ Import 失敗: No module named 'aiva_common.error_handling'
```

**建議**: 添加依賴檢查腳本

---

### Medium 級別問題

#### 問題 3: 文檔與實際代碼不一致

**現象**:
- README 中描述的初始化方式與實際代碼不符
- 部分方法簽名已變更但文檔未更新

**建議**: 同步更新文檔

---

## 📝 總結

### AIVA Core 整體評價

**優點** ⭐⭐⭐⭐:
1. ✅ **架構設計優秀** - 五大模組職責清晰
2. ✅ **設計原則完善** - Fail Fast, SOT, Event-Driven
3. ✅ **AI 決策完整** - 四大 Bug Bounty 決策方法已實現
4. ✅ **整合度高** - 模組間介面清晰，數據流完整
5. ✅ **代碼質量高** - 類型標註、錯誤處理、日誌記錄

**問題** ⚠️:
1. ❌ **Critical**: Commander 子模組初始化失敗
2. ⚠️ **High**: 依賴鏈不完整
3. ⚠️ **Medium**: 文檔與代碼不一致

### AttackCoordinator 評價

**理論設計** ⭐⭐⭐⭐⭐:
- 職責定義清晰（7 個核心方法）
- 整合完整（Bug Bounty 決策、多引擎掃描、兩階段流程）
- 代碼結構優秀（674 行，邏輯清晰）

**實際狀態** ⭐:
- ❌ 無法實例化（初始化參數錯誤）
- ❌ 從未被實際執行過
- ⚠️ 可能只是設計草稿或範例代碼

### 建議行動

#### 立即修復（Critical）
1. 修正 `CommanderCoordinator` 中所有子模組的初始化
2. 確保依賴鏈完整
3. 添加單元測試驗證

#### 短期改進（High）
1. 補充依賴檢查腳本
2. 添加整合測試
3. 更新所有文檔

#### 長期優化（Medium）
1. 考慮使用依賴注入框架
2. 統一初始化模式
3. 添加性能監控

---

**報告生成**: 2026-01-28  
**分析者**: AI Assistant  
**文檔版本**: v1.0
