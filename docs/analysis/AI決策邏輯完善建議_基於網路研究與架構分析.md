# 程式決策邏輯完善建議 - 基於架構研究與 AIVA 系統分析

**文檔日期**: 2025-12-06  
**研究基礎**: 
- Reflexion (語言代理自我反思學習, arXiv:2303.11366)
- ReAct (推理與行動協同, arXiv:2210.03629)  
- Constitutional AI (自我批判與改進, Anthropic)
- AutoGPT (自主代理架構)
- LangGraph (狀態機編排框架)
- AIVA 現有架構 (六大模組系統)

---

## 📑 目錄

1. [核心問題回顧](#核心問題回顧)
2. [業界最佳實踐分析](#業界最佳實踐分析)
3. [完整程式決策架構設計](#完整-程式決策架構設計)
4. [具體實現方案](#具體實現方案)
5. [代碼實現範例](#代碼實現範例)
6. [整合路線圖](#整合路線圖)

---

## 🚨 核心問題回顧

基於 `AI核心關鍵缺陷報告.md` 的分析，AIVA 當前存在以下關鍵缺陷：

### 問題 1: 程式決策核心功能缺失
```python
# ❌ 現況: BioNeuronDecisionController 只做 NLU (自然語言理解)
async def _parse_ui_command(self, text: str):
    if "掃描" in text:  # 簡單關鍵字匹配
        return "start_scan", {}
    # 沒有實際 AI 決策邏輯
```

### 問題 2: 內閉環數據無法使用
```python
# ✅ 方法存在
async def query_capabilities(self, query: str) -> RAGQueryResult:
    pass  # 可以查詢 RAG 知識庫

# ❌ 問題: 從未被調用!
# grep -r "query_capabilities" → 只有定義，無調用
```

### 問題 3: 13 步驟自動化流程斷裂
- Step 2: "程式核心生成命令" - **不存在**
- Step 4: "決定引擎組合" - **不存在**  
- Step 7: "動態選擇攻擊模組" - **不存在**

---

## 🌐 業界最佳實踐分析

### 1. **ReAct 框架** (Reasoning + Acting)

**核心理念**: 推理軌跡與行動交織，相互增強

```
思考(Thought) → 行動(Action) → 觀察(Observation) → 思考 → ...
```

**應用於 AIVA**:
```python
# Thought: 分析目標和上下文
"需要掃描 example.com，首先判斷技術棧"

# Action: 執行探測
execute_tech_detection(target="example.com")

# Observation: 獲取結果
"發現 PHP + MySQL，可能存在 SQL 注入風險"

# Thought: 基於觀察調整策略
"優先使用 SQL 注入掃描器，次要 XSS 檢測"

# Action: 生成具體命令
AICommand(type=SCAN_SQL_INJECTION, params={...})
```

**優勢**:
- ✅ 動態調整策略
- ✅ 可解釋的決策過程
- ✅ 錯誤自我修正

---

### 2. **Reflexion 框架** (自我反思與學習)

**核心理念**: 通過語言反饋強化決策，無需權重更新

```
執行任務 → 評估結果 → 語言化反思 → 儲存記憶 → 改進下次決策
```

**應用於 AIVA**:
```python
class ReflexionMemory:
    """自我反思記憶系統"""
    
    def __init__(self):
        self.episodic_memory = []  # 任務執行記錄
        self.reflections = []       # 反思結論
    
    async def reflect_on_failure(self, task, result):
        """失敗反思"""
        reflection = await self.llm.generate(
            prompt=f"""
            任務: {task}
            結果: {result}
            
            請分析失敗原因並提供改進建議:
            1. 哪個步驟出錯?
            2. 為什麼會失敗?
            3. 下次如何改進?
            """
        )
        
        self.reflections.append({
            "task": task,
            "failure_reason": reflection,
            "timestamp": datetime.now()
        })
        
        # 注入到 RAG 知識庫
        await self.rag.add_reflection(reflection)
    
    async def retrieve_similar_experience(self, current_task):
        """檢索相似經驗"""
        return await self.rag.query(
            f"與 {current_task} 類似的失敗經驗和改進建議"
        )
```

**優勢**:
- ✅ 從失敗中學習
- ✅ 知識累積不需重訓練
- ✅ 透明的改進過程

---

### 3. **Constitutional AI** (憲法式 AI - 自我批判)

**核心理念**: AI 自己評估輸出品質並自我改進

```
生成初始輸出 → 自我批判 → 修正輸出 → 驗證 → 最終輸出
```

**應用於 AIVA**:
```python
class ConstitutionalDecisionMaker:
    """憲法式決策制定器"""
    
    def __init__(self, principles: List[str]):
        """
        principles: 決策原則清單
        例如:
        - "優先使用被動掃描，避免破壞目標"
        - "SQL 注入測試前必須確認授權"
        - "發現高危漏洞時必須評估利用風險"
        """
        self.principles = principles
    
    async def make_decision_with_critique(self, context):
        """帶自我批判的決策"""
        
        # 1. 初始決策
        initial_command = await self.generate_command(context)
        
        # 2. 自我批判
        critique = await self.llm.generate(
            prompt=f"""
            決策原則:
            {self.principles}
            
            初始決策:
            {initial_command}
            
            請評估這個決策是否符合原則:
            1. 是否違反任何原則?
            2. 存在什麼風險?
            3. 如何改進?
            """
        )
        
        # 3. 如果有問題，修正
        if critique["has_issues"]:
            revised_command = await self.revise_command(
                initial_command, 
                critique["suggestions"]
            )
            return revised_command
        
        return initial_command
```

**優勢**:
- ✅ 內建安全檢查
- ✅ 符合倫理和法規
- ✅ 降低誤操作風險

---

### 4. **LangGraph 狀態機編排**

**核心理念**: 複雜決策流程建模為狀態圖

```python
from langgraph.graph import StateGraph

class AIVADecisionGraph:
    """AIVA 決策狀態圖"""
    
    def build_graph(self):
        workflow = StateGraph(DecisionState)
        
        # 定義節點
        workflow.add_node("analyze_target", self.analyze_target)
        workflow.add_node("query_capabilities", self.query_capabilities)
        workflow.add_node("select_strategy", self.select_strategy)
        workflow.add_node("generate_command", self.generate_command)
        workflow.add_node("validate_command", self.validate_command)
        workflow.add_node("execute", self.execute)
        workflow.add_node("reflect", self.reflect)
        
        # 定義邊 (流程)
        workflow.set_entry_point("analyze_target")
        workflow.add_edge("analyze_target", "query_capabilities")
        workflow.add_edge("query_capabilities", "select_strategy")
        workflow.add_edge("select_strategy", "generate_command")
        workflow.add_edge("generate_command", "validate_command")
        
        # 條件分支
        workflow.add_conditional_edges(
            "validate_command",
            self.should_execute,
            {
                "execute": "execute",
                "revise": "select_strategy",  # 回到策略選擇
                "abort": END
            }
        )
        
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", END)
        
        return workflow.compile()
```

**優勢**:
- ✅ 複雜流程可視化
- ✅ 支援條件分支和循環
- ✅ 狀態持久化 (斷點恢復)

---

## 🏗️ 完整 AI 決策架構設計

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                   用戶輸入 / 事件觸發                           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 理解與分析 (ReAct: Thought)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ NLU: 自然語言理解 (已存在)                              │   │
│  │  - 解析用戶意圖                                         │   │
│  │  - 提取關鍵參數 (目標、約束等)                           │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Context Analysis: 上下文分析 (NEW!)                    │   │
│  │  - 目標技術棧分析                                       │   │
│  │  - 歷史數據檢索 (Reflexion Memory)                     │   │
│  │  - 當前系統狀態評估                                     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: 能力查詢與匹配 (RAG Integration)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Internal Loop Query: 查詢內閉環 (NEW!)                │   │
│  │  - 查詢 RAG 知識庫獲取可用能力                          │   │
│  │  - 根據任務需求過濾能力                                 │   │
│  │  - 獲取能力詳細資訊 (參數、限制)                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Capability Ranking: 能力評分 (NEW!)                    │   │
│  │  - 使用神經網路評估匹配度                               │   │
│  │  - 考慮歷史成功率                                       │   │
│  │  - 整合風險評估                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: 策略決策 (ReAct: Reasoning)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Strategy Selection: 策略選擇 (NEW!)                    │   │
│  │  - 單一模組 vs 組合模組                                │   │
│  │  - Phase 0/1/2 決策                                    │   │
│  │  - 引擎選擇 (Python/TS/Rust/Go)                        │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Task Decomposition: 任務分解 (NEW!)                    │   │
│  │  - 複雜任務拆分為子任務                                 │   │
│  │  - 依賴關係分析                                         │   │
│  │  - 執行順序規劃                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: 命令生成與驗證 (ReAct: Action)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Command Generation: 命令生成 (NEW!)                    │   │
│  │  - 生成 AICommand 對象                                 │   │
│  │  - 填充參數和配置                                       │   │
│  │  - 設置回調和錯誤處理                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Constitutional Validation: 憲法式驗證 (NEW!)           │   │
│  │  - 檢查是否符合安全原則                                 │   │
│  │  - 風險評估 (RiskLevel)                                │   │
│  │  - 權限驗證 (RBAC)                                     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: 執行與監控 (ReAct: Observation)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Execution: 執行 (已存在 - AICommandCenter)             │   │
│  │  - 調用 AICommandCenter                               │   │
│  │  - 實時狀態監控                                         │   │
│  │  - 錯誤捕獲與處理                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 6: 反思與學習 (Reflexion)                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Result Analysis: 結果分析 (NEW!)                       │   │
│  │  - 評估執行成功/失敗                                    │   │
│  │  - 提取關鍵指標                                         │   │
│  │  - 識別異常模式                                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Reflexion: 自我反思 (NEW!)                             │   │
│  │  - 語言化總結經驗                                       │   │
│  │  - 失敗原因分析                                         │   │
│  │  - 改進建議生成                                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Knowledge Update: 知識更新 (NEW!)                      │   │
│  │  - 反思注入 RAG 知識庫                                 │   │
│  │  - 更新能力成功率統計                                   │   │
│  │  - 更新技能圖譜                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 具體實現方案

### 核心類設計

#### 1. **AIDecisionEngine** (主決策引擎)

```python
"""
services/core/aiva_core/cognitive_core/decision/ai_decision_engine.py
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from enum import Enum

from aiva_common.schemas import (
    AICommand, 
    HighLevelIntent, 
    RiskLevel,
    TargetInfo
)
from aiva_common.utils.logging import get_logger

from ..rag import RAGEngine
from ..internal_loop_connector import InternalLoopConnector
from ..neural import RealNeuralCore
from .enhanced_decision_agent import EnhancedDecisionAgent
from .reflexion_memory import ReflexionMemory
from .constitutional_validator import ConstitutionalValidator

logger = get_logger(__name__)


class DecisionPhase(Enum):
    """決策階段"""
    UNDERSTANDING = "understanding"          # Phase 1: 理解
    CAPABILITY_QUERY = "capability_query"    # Phase 2: 能力查詢
    STRATEGY_PLANNING = "strategy_planning"  # Phase 3: 策略規劃
    COMMAND_GENERATION = "command_generation" # Phase 4: 命令生成
    VALIDATION = "validation"                # Phase 5: 驗證
    EXECUTION = "execution"                  # Phase 6: 執行
    REFLECTION = "reflection"                # Phase 7: 反思


class AIDecisionEngine:
    """AI 決策引擎 - AIVA 的決策大腦
    
    整合:
    - ReAct: 推理與行動協同
    - Reflexion: 自我反思學習
    - Constitutional AI: 自我批判驗證
    - RAG: 檢索增強生成
    - Neural Network: 深度評估
    
    職責:
    1. 理解用戶意圖和上下文
    2. 查詢內閉環獲取可用能力
    3. 決定執行策略和引擎組合
    4. 生成具體的 AICommand
    5. 驗證命令安全性和合規性
    6. 執行後反思與學習
    """
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        internal_loop_connector: InternalLoopConnector,
        neural_core: RealNeuralCore,
        decision_agent: EnhancedDecisionAgent,
        reflexion_memory: ReflexionMemory,
        constitutional_validator: ConstitutionalValidator
    ):
        self.rag = rag_engine
        self.internal_loop = internal_loop_connector
        self.neural_core = neural_core
        self.decision_agent = decision_agent
        self.reflexion = reflexion_memory
        self.validator = constitutional_validator
        
        # 決策歷史
        self.decision_history: List[Dict[str, Any]] = []
        
        # 當前決策狀態
        self.current_phase = DecisionPhase.UNDERSTANDING
        self.reasoning_trace: List[str] = []  # ReAct 推理軌跡
        
        logger.info("AIDecisionEngine initialized with full ReAct + Reflexion pipeline")
    
    async def decide(
        self, 
        user_intent: HighLevelIntent,
        context: Optional[Dict[str, Any]] = None
    ) -> AICommand:
        """主決策方法 - 完整的 ReAct + Reflexion 流程
        
        Args:
            user_intent: 用戶高層意圖
            context: 額外上下文資訊
            
        Returns:
            AICommand: 可執行的 AI 命令
        """
        self.reasoning_trace = []  # 重置推理軌跡
        context = context or {}
        
        try:
            # === Phase 1: 理解與分析 ===
            self._log_thought("開始理解用戶意圖和分析上下文")
            analysis = await self._analyze_context(user_intent, context)
            
            # === Phase 2: 能力查詢與匹配 ===
            self._log_thought(f"查詢適合 {user_intent.intent_type} 任務的能力")
            capabilities = await self._query_and_rank_capabilities(
                user_intent, analysis
            )
            
            if not capabilities:
                raise ValueError(f"未找到適合 {user_intent.intent_type} 的能力")
            
            # === Phase 3: 策略決策 ===
            self._log_thought("基於可用能力和歷史經驗制定策略")
            strategy = await self._select_strategy(
                user_intent, capabilities, analysis
            )
            
            # === Phase 4: 命令生成 ===
            self._log_thought(f"生成執行計畫: {strategy['approach']}")
            command = await self._generate_command(
                user_intent, strategy, capabilities
            )
            
            # === Phase 5: 憲法式驗證 ===
            self._log_thought("驗證命令的安全性和合規性")
            validation_result = await self.validator.validate(command, context)
            
            if not validation_result["is_valid"]:
                self._log_thought(f"驗證失敗: {validation_result['reason']}")
                # 嘗試修正
                command = await self._revise_command(
                    command, validation_result["suggestions"]
                )
            
            # 記錄決策
            self._record_decision(user_intent, command, analysis, strategy)
            
            self._log_thought("決策完成，命令已生成")
            logger.info(
                f"Decision completed: {command.action_type}",
                extra={"reasoning_steps": len(self.reasoning_trace)}
            )
            
            return command
            
        except Exception as e:
            logger.error(f"Decision failed: {e}", exc_info=True)
            # 記錄失敗以供反思
            await self.reflexion.record_failure(user_intent, str(e))
            raise
    
    async def _analyze_context(
        self, 
        intent: HighLevelIntent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 1: 上下文分析
        
        包含:
        1. 目標技術棧分析 (如果是 Web 目標)
        2. 歷史經驗檢索 (Reflexion)
        3. 當前系統狀態
        """
        analysis = {
            "intent_type": intent.intent_type,
            "target": intent.target,
            "constraints": intent.constraints,
            "timestamp": datetime.now(UTC)
        }
        
        # 1. 目標分析
        if intent.target:
            target_info = await self._analyze_target(intent.target)
            analysis["target_analysis"] = target_info
            self._log_thought(
                f"目標分析: {target_info.get('tech_stack', 'Unknown')}"
            )
        
        # 2. 檢索相似經驗 (Reflexion)
        similar_cases = await self.reflexion.retrieve_similar_experience(
            task_description=f"{intent.intent_type} on {intent.target}",
            top_k=3
        )
        
        if similar_cases:
            analysis["past_experiences"] = similar_cases
            self._log_thought(
                f"發現 {len(similar_cases)} 個相似案例，"
                f"成功率: {self._calculate_success_rate(similar_cases):.1%}"
            )
        
        # 3. 系統狀態
        system_state = await self._get_system_state()
        analysis["system_state"] = system_state
        
        return analysis
    
    async def _query_and_rank_capabilities(
        self,
        intent: HighLevelIntent,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Phase 2: 查詢並排序可用能力
        
        核心: 使用 InternalLoopConnector 查詢 RAG 知識庫
        """
        # 構建查詢語句
        query = self._build_capability_query(intent, analysis)
        
        # 🔥 關鍵: 調用內閉環查詢能力 (之前缺失!)
        rag_result = await self.internal_loop.query_capabilities(
            query=query,
            top_k=10
        )
        
        if not rag_result.results:
            logger.warning(f"No capabilities found for query: {query}")
            return []
        
        self._log_thought(
            f"找到 {len(rag_result.results)} 個相關能力"
        )
        
        # 使用神經網路評分
        capabilities_with_scores = []
        for cap in rag_result.results:
            score = await self._score_capability(cap, intent, analysis)
            capabilities_with_scores.append({
                "capability": cap,
                "score": score,
                "relevance": cap.get("relevance_score", 0)
            })
        
        # 排序
        ranked = sorted(
            capabilities_with_scores,
            key=lambda x: (x["score"], x["relevance"]),
            reverse=True
        )
        
        self._log_thought(
            f"最佳能力: {ranked[0]['capability']['name']} "
            f"(評分: {ranked[0]['score']:.2f})"
        )
        
        return ranked
    
    async def _select_strategy(
        self,
        intent: HighLevelIntent,
        capabilities: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 3: 策略選擇
        
        決定:
        1. 單一模組 vs 多模組組合
        2. Phase 0/1/2 策略
        3. 引擎選擇 (Python/TS/Rust/Go)
        4. 執行順序
        """
        # 使用 EnhancedDecisionAgent
        decision = await self.decision_agent.decide(
            intent=intent,
            available_capabilities=capabilities,
            context=analysis
        )
        
        strategy = {
            "approach": decision.approach,  # "single" or "combined"
            "phase": decision.phase,        # 0, 1, or 2
            "engines": decision.engines,    # ["python", "typescript", ...]
            "sequence": decision.sequence,  # 執行順序
            "risk_level": decision.risk_level,
            "reasoning": decision.reasoning
        }
        
        self._log_thought(
            f"策略: {strategy['approach']} | "
            f"Phase {strategy['phase']} | "
            f"引擎: {', '.join(strategy['engines'])}"
        )
        
        return strategy
    
    async def _generate_command(
        self,
        intent: HighLevelIntent,
        strategy: Dict[str, Any],
        capabilities: List[Dict[str, Any]]
    ) -> AICommand:
        """Phase 4: 生成 AICommand
        
        🔥 關鍵: 之前完全缺失的功能!
        """
        # 選擇最佳能力
        best_capability = capabilities[0]["capability"]
        
        # 構建命令參數
        params = self._build_command_params(
            intent, strategy, best_capability
        )
        
        # 創建 AICommand
        command = AICommand(
            command_id=str(uuid4()),
            action_type=self._map_to_action_type(
                intent.intent_type, best_capability
            ),
            parameters=params,
            source_module="ai_decision_engine",
            target_module=best_capability["module_name"],
            priority=self._calculate_priority(intent, strategy),
            risk_level=strategy["risk_level"],
            metadata={
                "strategy": strategy,
                "capability": best_capability["name"],
                "reasoning_trace": self.reasoning_trace.copy(),
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        
        self._log_thought(f"生成命令: {command.action_type}")
        
        return command
    
    async def reflect_on_execution(
        self,
        command: AICommand,
        result: Any,
        success: bool
    ):
        """Phase 6: 執行後反思 (Reflexion)
        
        將執行結果轉化為語言化經驗，注入 RAG
        """
        self._log_thought("開始反思執行結果")
        
        # 生成反思
        reflection = await self.reflexion.reflect(
            task={
                "intent": command.metadata.get("original_intent"),
                "command": command.dict(),
                "strategy": command.metadata.get("strategy")
            },
            result=result,
            success=success
        )
        
        # 注入 RAG 知識庫
        await self.rag.add_document(
            content=reflection["reflection_text"],
            metadata={
                "type": "execution_reflection",
                "success": success,
                "command_type": command.action_type,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        
        # 更新能力統計
        capability_name = command.metadata.get("capability")
        await self.internal_loop.update_capability_stats(
            capability_name=capability_name,
            success=success,
            execution_time=result.get("execution_time", 0)
        )
        
        self._log_thought(
            f"反思完成: {'成功經驗' if success else '失敗教訓'}已記錄"
        )
        
        logger.info(
            f"Reflection completed for {command.action_type}",
            extra={"success": success, "reflection_id": reflection["id"]}
        )
    
    # ========== 輔助方法 ==========
    
    def _log_thought(self, thought: str):
        """記錄推理軌跡 (ReAct Thought)"""
        self.reasoning_trace.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "thought": thought
        })
        logger.debug(f"[THOUGHT] {thought}")
    
    def _build_capability_query(
        self, 
        intent: HighLevelIntent, 
        analysis: Dict[str, Any]
    ) -> str:
        """構建能力查詢語句"""
        query_parts = [f"能處理 {intent.intent_type} 任務的能力"]
        
        if intent.target:
            target_info = analysis.get("target_analysis", {})
            tech_stack = target_info.get("tech_stack")
            if tech_stack:
                query_parts.append(f"針對 {tech_stack} 技術棧")
        
        if intent.constraints:
            constraints_str = ", ".join(
                f"{k}={v}" for k, v in intent.constraints.items()
            )
            query_parts.append(f"約束條件: {constraints_str}")
        
        return " ".join(query_parts)
    
    async def _score_capability(
        self,
        capability: Dict[str, Any],
        intent: HighLevelIntent,
        analysis: Dict[str, Any]
    ) -> float:
        """使用神經網路評分能力匹配度"""
        # 構建特徵向量
        features = self._extract_capability_features(
            capability, intent, analysis
        )
        
        # 神經網路推理
        score_tensor = await self.neural_core.forward(features)
        score = float(score_tensor.item())
        
        # 調整: 整合歷史成功率
        historical_success_rate = capability.get("success_rate", 0.5)
        adjusted_score = 0.7 * score + 0.3 * historical_success_rate
        
        return adjusted_score
    
    def _calculate_success_rate(self, cases: List[Dict]) -> float:
        """計算歷史案例成功率"""
        if not cases:
            return 0.0
        successful = sum(1 for c in cases if c.get("success", False))
        return successful / len(cases)
    
    def _record_decision(
        self,
        intent: HighLevelIntent,
        command: AICommand,
        analysis: Dict[str, Any],
        strategy: Dict[str, Any]
    ):
        """記錄決策歷史"""
        self.decision_history.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "intent": intent.dict(),
            "command_id": command.command_id,
            "strategy": strategy,
            "reasoning_trace": self.reasoning_trace.copy(),
            "analysis": analysis
        })
    
    # ... 更多輔助方法
```

---

#### 2. **ReflexionMemory** (反思記憶系統)

```python
"""
services/core/aiva_core/cognitive_core/decision/reflexion_memory.py
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
from uuid import uuid4

from aiva_common.utils.logging import get_logger
from ..rag import RAGEngine

logger = get_logger(__name__)


class ReflexionMemory:
    """Reflexion 記憶系統
    
    基於論文: Reflexion: Language Agents with Verbal Reinforcement Learning
    (arXiv:2303.11366)
    
    核心思想:
    - 不通過權重更新學習，而是通過語言反饋
    - 將執行經驗轉化為反思文本
    - 儲存在情節記憶中供未來檢索
    """
    
    def __init__(self, rag_engine: RAGEngine, llm_client: Any):
        self.rag = rag_engine
        self.llm = llm_client
        self.episodic_memory: List[Dict[str, Any]] = []
        
        logger.info("ReflexionMemory initialized")
    
    async def reflect(
        self,
        task: Dict[str, Any],
        result: Any,
        success: bool
    ) -> Dict[str, Any]:
        """執行反思並生成語言化經驗
        
        Args:
            task: 任務描述 (包含意圖、命令、策略)
            result: 執行結果
            success: 是否成功
            
        Returns:
            反思記錄
        """
        # 構建反思提示
        prompt = self._build_reflection_prompt(task, result, success)
        
        # 使用 LLM 生成反思
        reflection_text = await self.llm.generate(prompt)
        
        # 創建反思記錄
        reflection = {
            "id": str(uuid4()),
            "task": task,
            "result": result,
            "success": success,
            "reflection_text": reflection_text,
            "timestamp": datetime.now(UTC).isoformat(),
            "lessons_learned": self._extract_lessons(reflection_text)
        }
        
        # 添加到情節記憶
        self.episodic_memory.append(reflection)
        
        # 持久化到 RAG
        await self._persist_to_rag(reflection)
        
        logger.info(
            f"Reflection generated for {'successful' if success else 'failed'} task",
            extra={"reflection_id": reflection["id"]}
        )
        
        return reflection
    
    async def retrieve_similar_experience(
        self,
        task_description: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """檢索相似經驗
        
        Args:
            task_description: 當前任務描述
            top_k: 返回前 k 個最相關的經驗
            
        Returns:
            相似經驗列表
        """
        # 查詢 RAG
        rag_result = await self.rag.query(
            query=f"與以下任務相似的執行經驗和反思: {task_description}",
            top_k=top_k,
            filter_metadata={"type": "execution_reflection"}
        )
        
        # 解析結果
        experiences = []
        for doc in rag_result.results:
            experiences.append({
                "content": doc["content"],
                "success": doc["metadata"].get("success", False),
                "relevance_score": doc.get("score", 0),
                "timestamp": doc["metadata"].get("timestamp")
            })
        
        return experiences
    
    async def record_failure(
        self,
        intent: Any,
        error_message: str
    ):
        """記錄失敗以供未來分析"""
        failure_record = {
            "id": str(uuid4()),
            "intent": intent.dict() if hasattr(intent, "dict") else str(intent),
            "error": error_message,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        self.episodic_memory.append(failure_record)
        
        # 生成失敗反思
        prompt = f"""
        任務失敗分析:
        
        原始意圖: {intent}
        錯誤訊息: {error_message}
        
        請分析:
        1. 可能的失敗原因
        2. 如何避免類似問題
        3. 建議的改進措施
        """
        
        failure_analysis = await self.llm.generate(prompt)
        
        await self.rag.add_document(
            content=failure_analysis,
            metadata={
                "type": "failure_analysis",
                "intent_type": getattr(intent, "intent_type", "unknown"),
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        
        logger.warning(f"Failure recorded: {error_message[:100]}")
    
    def _build_reflection_prompt(
        self,
        task: Dict[str, Any],
        result: Any,
        success: bool
    ) -> str:
        """構建反思提示詞"""
        status = "成功" if success else "失敗"
        
        prompt = f"""
        執行任務反思:
        
        任務類型: {task.get('intent', {}).get('intent_type', 'Unknown')}
        執行策略: {task.get('strategy', {})}
        執行結果: {status}
        
        結果詳情:
        {result}
        
        請進行深入反思，回答以下問題:
        
        1. **執行過程分析**:
           - 哪些步驟執行順利？
           - 遇到了什麼問題？
           
        2. **{'成功因素' if success else '失敗原因'}**:
           - 關鍵因素是什麼？
           - 是否可預測？
           
        3. **經驗教訓**:
           - 學到了什麼？
           - 下次如何改進？
           
        4. **策略建議**:
           - 當前策略是否最優？
           - 有更好的替代方案嗎？
        
        請用清晰、結構化的方式總結反思結果。
        """
        
        return prompt
    
    def _extract_lessons(self, reflection_text: str) -> List[str]:
        """從反思文本中提取關鍵經驗教訓"""
        # 簡單實現: 提取包含關鍵詞的句子
        keywords = ["教訓", "學到", "改進", "建議", "注意"]
        lessons = []
        
        for line in reflection_text.split("\n"):
            if any(kw in line for kw in keywords):
                lessons.append(line.strip())
        
        return lessons[:5]  # 最多 5 條
    
    async def _persist_to_rag(self, reflection: Dict[str, Any]):
        """持久化反思到 RAG 知識庫"""
        await self.rag.add_document(
            content=reflection["reflection_text"],
            metadata={
                "type": "execution_reflection",
                "reflection_id": reflection["id"],
                "success": reflection["success"],
                "task_type": reflection["task"].get("intent", {}).get("intent_type"),
                "timestamp": reflection["timestamp"],
                "lessons": reflection["lessons_learned"]
            }
        )
```

---

#### 3. **ConstitutionalValidator** (憲法式驗證器)

```python
"""
services/core/aiva_core/cognitive_core/decision/constitutional_validator.py
"""

from typing import Any, Dict, List
from enum import Enum

from aiva_common.schemas import AICommand, RiskLevel
from aiva_common.utils.logging import get_logger

logger = get_logger(__name__)


class SafetyPrinciple(Enum):
    """安全原則 (憲法)"""
    NO_UNAUTHORIZED_ACCESS = "禁止未經授權的訪問"
    MINIMIZE_DISRUPTION = "最小化對目標系統的破壞"
    RESPECT_RATE_LIMITS = "遵守速率限制"
    VERIFY_TARGET_AUTHORIZATION = "驗證目標授權"
    AVOID_DATA_EXFILTRATION = "避免數據外洩"
    LOG_ALL_ACTIONS = "記錄所有行動"
    REQUIRE_HUMAN_APPROVAL_HIGH_RISK = "高風險操作需人工批准"


class ConstitutionalValidator:
    """憲法式驗證器
    
    基於 Anthropic 的 Constitutional AI 概念
    
    作用:
    - 在執行前驗證命令是否符合安全原則
    - 提供自我批判和改進建議
    - 降低誤操作和合規風險
    """
    
    def __init__(
        self,
        principles: List[SafetyPrinciple],
        llm_client: Any,
        authz_checker: Any
    ):
        self.principles = principles
        self.llm = llm_client
        self.authz = authz_checker
        
        logger.info(
            f"ConstitutionalValidator initialized with {len(principles)} principles"
        )
    
    async def validate(
        self,
        command: AICommand,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """驗證命令
        
        Args:
            command: 待驗證的命令
            context: 上下文資訊
            
        Returns:
            驗證結果: {
                "is_valid": bool,
                "violations": List[str],
                "suggestions": List[str],
                "risk_assessment": RiskLevel
            }
        """
        violations = []
        suggestions = []
        
        # 1. 檢查授權
        if not await self._check_authorization(command, context):
            violations.append("目標未經授權")
            suggestions.append("請先獲取目標系統的測試授權")
        
        # 2. 風險評估
        risk = await self._assess_risk(command)
        if risk == RiskLevel.HIGH:
            if SafetyPrinciple.REQUIRE_HUMAN_APPROVAL_HIGH_RISK in self.principles:
                violations.append("高風險操作需要人工批准")
                suggestions.append("建議降低操作風險或請求人工審批")
        
        # 3. 使用 LLM 進行憲法式批判
        critique = await self._llm_critique(command, context)
        
        if critique.get("has_concerns"):
            violations.extend(critique.get("concerns", []))
            suggestions.extend(critique.get("recommendations", []))
        
        # 4. 檢查速率限制
        if not await self._check_rate_limits(command):
            violations.append("超出速率限制")
            suggestions.append("降低請求頻率或增加延遲")
        
        is_valid = len(violations) == 0
        
        result = {
            "is_valid": is_valid,
            "violations": violations,
            "suggestions": suggestions,
            "risk_assessment": risk,
            "critique": critique
        }
        
        if not is_valid:
            logger.warning(
                f"Command validation failed: {len(violations)} violations",
                extra={"command_id": command.command_id, "violations": violations}
            )
        
        return result
    
    async def _llm_critique(
        self,
        command: AICommand,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 LLM 進行憲法式批判"""
        principles_text = "\n".join(f"- {p.value}" for p in self.principles)
        
        prompt = f"""
        作為 AI 安全審查員，請評估以下命令是否符合安全原則:
        
        **安全原則**:
        {principles_text}
        
        **待執行命令**:
        - 類型: {command.action_type}
        - 目標: {command.parameters.get('target', 'N/A')}
        - 風險等級: {command.risk_level}
        - 參數: {command.parameters}
        
        **上下文**:
        {context}
        
        請回答:
        1. 這個命令是否違反任何原則？
        2. 存在什麼潛在風險？
        3. 如何改進以符合原則？
        
        以 JSON 格式回答:
        {{
            "has_concerns": bool,
            "concerns": [列表],
            "recommendations": [列表],
            "revised_parameters": {{修正建議}}
        }}
        """
        
        response = await self.llm.generate(prompt, response_format="json")
        return response
    
    async def _check_authorization(
        self,
        command: AICommand,
        context: Dict[str, Any]
    ) -> bool:
        """檢查是否有執行授權"""
        # 檢查 RBAC 權限
        has_permission = await self.authz.check_permission(
            user=context.get("user"),
            action=command.action_type,
            resource=command.parameters.get("target")
        )
        
        return has_permission
    
    async def _assess_risk(self, command: AICommand) -> RiskLevel:
        """評估命令風險等級"""
        # 基於命令類型和參數評估
        high_risk_actions = ["EXPLOIT", "ATTACK", "MODIFY"]
        
        if any(action in command.action_type for action in high_risk_actions):
            return RiskLevel.HIGH
        elif "SCAN" in command.action_type:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _check_rate_limits(self, command: AICommand) -> bool:
        """檢查速率限制"""
        # 實現速率限制檢查邏輯
        # 這裡簡化為始終通過
        return True
```

---

## 📝 代碼實現範例

### 完整使用示例

```python
"""
使用範例: 完整的 AI 決策流程
"""

from aiva_core.cognitive_core.decision import (
    AIDecisionEngine,
    ReflexionMemory,
    ConstitutionalValidator,
    SafetyPrinciple
)
from aiva_core.cognitive_core.rag import RAGEngine
from aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
from aiva_core.cognitive_core.neural import RealNeuralCore
from aiva_core.cognitive_core.decision import EnhancedDecisionAgent

from aiva_common.schemas import HighLevelIntent, IntentType, TargetInfo


async def main():
    """主流程示例"""
    
    # 1. 初始化所有組件
    rag_engine = RAGEngine()
    internal_loop = InternalLoopConnector(rag_knowledge_base=rag_engine.kb)
    neural_core = RealNeuralCore(use_5m_model=True)
    decision_agent = EnhancedDecisionAgent()
    reflexion = ReflexionMemory(rag_engine, llm_client=...)
    validator = ConstitutionalValidator(
        principles=[
            SafetyPrinciple.NO_UNAUTHORIZED_ACCESS,
            SafetyPrinciple.MINIMIZE_DISRUPTION,
            SafetyPrinciple.REQUIRE_HUMAN_APPROVAL_HIGH_RISK
        ],
        llm_client=...,
        authz_checker=...
    )
    
    # 2. 創建 AI 決策引擎
    decision_engine = AIDecisionEngine(
        rag_engine=rag_engine,
        internal_loop_connector=internal_loop,
        neural_core=neural_core,
        decision_agent=decision_agent,
        reflexion_memory=reflexion,
        constitutional_validator=validator
    )
    
    # 3. 用戶輸入 (通過 UI 或 API)
    user_input = "掃描 example.com 尋找 SQL 注入漏洞"
    
    # 4. 構建高層意圖
    intent = HighLevelIntent(
        intent_type=IntentType.SCAN,
        target=TargetInfo(url="example.com"),
        constraints={
            "scan_types": ["sql_injection"],
            "max_duration": 300,
            "stealth_mode": True
        }
    )
    
    # 5. AI 決策 (完整的 ReAct + Reflexion 流程)
    try:
        command = await decision_engine.decide(
            user_intent=intent,
            context={"user": "admin", "session_id": "sess_123"}
        )
        
        print(f"✅ 決策完成!")
        print(f"   命令 ID: {command.command_id}")
        print(f"   類型: {command.action_type}")
        print(f"   風險: {command.risk_level}")
        print(f"   推理步驟: {len(decision_engine.reasoning_trace)}")
        
        # 6. 執行命令 (通過 AICommandCenter)
        from aiva_core.task_planning import AICommandCenter
        
        command_center = AICommandCenter()
        result = await command_center.execute(command)
        
        print(f"✅ 執行完成!")
        print(f"   結果: {result.status}")
        
        # 7. 執行後反思 (Reflexion)
        await decision_engine.reflect_on_execution(
            command=command,
            result=result,
            success=(result.status == "success")
        )
        
        print(f"✅ 反思完成，經驗已記錄到 RAG 知識庫")
        
    except Exception as e:
        print(f"❌ 決策失敗: {e}")
        # 失敗也會被記錄供未來學習


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 🗺️ 整合路線圖

### Phase 1: 核心決策引擎 (1-2 週)

**目標**: 實現基本的 AI 決策能力

```
□ 創建 AIDecisionEngine 類
□ 實現 _query_and_rank_capabilities (關鍵!)
□ 實現 _generate_command (關鍵!)
□ 整合 InternalLoopConnector.query_capabilities
□ 基本單元測試
```

### Phase 2: ReAct 推理軌跡 (1 週)

**目標**: 增加可解釋性

```
□ 實現推理軌跡記錄
□ 實現上下文分析
□ 實現策略選擇邏輯
□ 可視化推理過程 (UI)
```

### Phase 3: Reflexion 學習系統 (1-2 週)

**目標**: 自我學習能力

```
□ 創建 ReflexionMemory 類
□ 實現執行後反思
□ 實現經驗檢索
□ RAG 知識庫整合
```

### Phase 4: Constitutional 驗證 (1 週)

**目標**: 安全性保障

```
□ 創建 ConstitutionalValidator 類
□ 定義安全原則
□ 實現 LLM 批判
□ 風險評估系統
```

### Phase 5: 完整集成測試 (1 週)

**目標**: 端到端驗證

```
□ 集成測試套件
□ 靶場實戰測試
□ 性能優化
□ 文檔完善
```

---

## 📊 預期效果

### 解決的問題

| 問題 | 現況 | 改進後 |
|------|------|--------|
| AI 決策能力 | ❌ 只有關鍵字匹配 | ✅ 完整 ReAct 決策流程 |
| 內閉環使用 | ❌ 從未調用 | ✅ 核心查詢機制 |
| 命令生成 | ❌ 不存在 | ✅ 完整 AICommand 生成 |
| 策略選擇 | ❌ 無策略 | ✅ 智能策略規劃 |
| 學習能力 | ❌ 無學習 | ✅ Reflexion 自我學習 |
| 安全驗證 | ❌ 無驗證 | ✅ 憲法式自我批判 |
| 可解釋性 | ❌ 黑盒 | ✅ 完整推理軌跡 |

### 性能指標

```python
# 預期指標
{
    "決策準確率": "> 85%",
    "命令生成成功率": "> 95%",
    "RAG 查詢響應時間": "< 500ms",
    "完整決策時間": "< 3s",
    "反思生成時間": "< 2s",
    "推理步驟數": "5-10 步"
}
```

---

## 🎓 參考文獻

1. **ReAct: Synergizing Reasoning and Acting in Language Models**  
   Yao et al., ICLR 2023  
   arXiv:2210.03629

2. **Reflexion: Language Agents with Verbal Reinforcement Learning**  
   Shinn et al., NeurIPS 2023  
   arXiv:2303.11366

3. **Constitutional AI: Harmlessness from AI Feedback**  
   Bai et al., Anthropic 2022  
   arXiv:2212.08073

4. **AutoGPT: Building Autonomous AI Agents**  
   GitHub: Significant-Gravitas/AutoGPT

5. **LangGraph: Building stateful, multi-actor applications with LLMs**  
   LangChain AI

---

## 📝 總結

本方案基於業界最佳實踐，為 AIVA 設計了完整的 AI 決策邏輯系統，核心特點:

1. **ReAct 推理與行動協同** - 動態調整策略
2. **Reflexion 自我學習** - 從經驗中持續改進
3. **Constitutional AI 自我批判** - 確保安全合規
4. **完整 RAG 整合** - 使用內閉環知識
5. **可解釋決策過程** - 透明的推理軌跡

這將徹底解決 `AI核心關鍵缺陷報告.md` 中指出的三大核心問題，使 AIVA 真正實現 AI 自主決策與指揮能力。
