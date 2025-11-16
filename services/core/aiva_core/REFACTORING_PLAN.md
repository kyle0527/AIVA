# AIVA Core 重構計劃 - 六大模組架構

**計劃日期**: 2025年11月15日  
**重構版本**: v2.2.0 → v3.0.0  
**核心理念**: AI 自我優化雙重閉環設計  
**預計完成時間**: 4-6 週

---

## 🎯 重構目標

### **核心目標**
1. ✅ 實現 AI 自我優化雙重閉環架構
2. ✅ 提升代碼組織清晰度和可維護性
3. ✅ 建立明確的模組邊界和接口
4. ✅ 完善內部閉環和外部閉環的連接

### **設計理念對應**
- **內部閉環 (Know Thyself)**: 探索(對內) + 分析(靜態) + RAG → 了解自身能力
- **外部閉環 (Learn from Battle)**: 掃描(對外) + 攻擊(實戰) → 收集優化方向
- **視覺化優先**: 用圖表展示優化方案,減少 NLP 負擔

---

## 📊 六大模組架構設計

### **新架構總覽**

```
services/core/aiva_core/
├── 🧠 cognitive_core/          # 模組 1: AI 認知核心
│   ├── neural/                 # 神經網路核心
│   ├── rag/                    # RAG 增強系統
│   ├── decision/               # 決策支援
│   ├── anti_hallucination/     # 反幻覺模組
│   ├── internal_loop_connector.py  # 內部閉環連接器
│   ├── external_loop_connector.py  # 外部閉環連接器
│   ├── __init__.py
│   └── README.md
│
├── 🧭 internal_exploration/    # 模組 2: 對內探索 (系統自我認知)
│   ├── module_explorer.py      # 模組探索器
│   ├── capability_analyzer.py  # 能力分析器
│   ├── ast_code_analyzer.py    # AST 代碼解析器 (新建)
│   ├── knowledge_graph.py      # 知識圖譜組合器 (新建)
│   ├── self_diagnostics.py     # 自我診斷工具 (新建)
│   ├── __init__.py
│   └── README.md
│
├── ⚙️ task_planning/            # 模組 3: 任務規劃與執行
│   ├── planner/                # 規劃子系統
│   │   ├── ast_parser.py       # AST 解析器
│   │   ├── orchestrator.py     # 編排器
│   │   ├── task_converter.py   # 任務轉換器
│   │   └── tool_selector.py    # 工具選擇器
│   ├── executor/               # 執行子系統
│   │   ├── plan_executor.py    # 計劃執行器
│   │   ├── task_executor.py    # 任務執行器
│   │   └── attack_plan_mapper.py # 攻擊計劃映射器
│   ├── __init__.py
│   └── README.md
│
├── 📈 external_learning/        # 模組 4: 對外探索與學習 (實戰反饋)
│   ├── analysis/               # 分析子系統
│   │   ├── ast_trace_comparator.py    # AST vs Trace 對比
│   │   └── dynamic_strategy_adjustment.py # 動態策略調整
│   ├── tracing/                # 追蹤子系統
│   │   └── unified_tracer.py   # 統一追蹤器
│   ├── learning/               # 學習子系統
│   │   ├── model_trainer.py    # 模型訓練器
│   │   ├── experience_manager.py # 經驗管理器 (新建/確認)
│   │   └── learning_engine.py  # 學習引擎
│   ├── __init__.py
│   └── README.md
│
├── 🛠️ core_capabilities/        # 模組 5: 核心能力 (具體工具)
│   ├── attack/                 # 攻擊能力
│   │   ├── payload_generator.py
│   │   ├── exploit_manager.py
│   │   ├── attack_chain.py
│   │   ├── attack_executor.py
│   │   └── attack_validator.py
│   ├── analysis/               # 分析能力
│   │   └── analysis_engine.py
│   ├── plugins/                # 插件能力
│   │   └── ai_summary_plugin.py
│   ├── capability_registry.py  # 能力註冊中心
│   ├── __init__.py
│   └── README.md
│
├── 🏗️ service_backbone/         # 模組 6: 服務骨幹 (基礎設施)
│   ├── api/                    # API 層
│   │   └── app.py              # FastAPI 入口
│   ├── coordination/           # 協調層
│   │   └── core_service_coordinator.py
│   ├── messaging/              # 消息系統
│   ├── performance/            # 性能優化
│   ├── authz/                  # 授權控制
│   ├── monitoring/             # 監控系統
│   ├── __init__.py
│   └── README.md
│
├── ai_commander.py             # AI 指揮官 (整合兩個連接器)
├── __init__.py                 # 主入口文件
└── README.md                   # 主文檔
```

---

## 🔄 模組職責定義

### **模組 1: 🧠 cognitive_core (AI 認知核心)**

**定位**: AIVA 的「大腦」,負責思考和決策

**核心職責**:
- 執行 5M 參數神經網路推理
- 管理 RAG 知識庫 (包含對內和對外知識)
- 整合反幻覺模組確保決策可靠性
- **連接內部閉環**: 通過 `internal_loop_connector.py` 將探索結果灌入 RAG
- **連接外部閉環**: 通過 `external_loop_connector.py` 將偏差報告灌入學習系統

**關鍵組件**:
```python
from cognitive_core.neural import RealNeuralCore
from cognitive_core.rag import RAGEngine
from cognitive_core.decision import EnhancedDecisionAgent
from cognitive_core.anti_hallucination import AntiHallucinationModule
from cognitive_core import InternalLoopConnector, ExternalLoopConnector
```

**對應設計理念**: **AI 決策中心** (整合雙重閉環數據)

---

### **模組 2: 🧭 internal_exploration (對內探索)**

**定位**: AI 的「自我認知」能力

**核心職責**:
- 掃描 AIVA 五大模組 (ai_core, attack_engine, scan_engine, integration, features)
- 解析 Python 代碼為 AST 圖
- 構建全專案組合知識圖譜
- 識別 `@register_capability` 標記的能力
- 進行系統自我診斷 (找出「路不通」的地方)

**關鍵組件**:
```python
from internal_exploration import ModuleExplorer
from internal_exploration import CapabilityAnalyzer
from internal_exploration import ASTCodeAnalyzer  # 新建
from internal_exploration import KnowledgeGraph   # 新建
from internal_exploration import SelfDiagnostics  # 新建
```

**對應設計理念**: **內部閉環步驟 1+2** (探索 + 分析)

---

### **模組 3: ⚙️ task_planning (任務規劃與執行)**

**定位**: 將「決策」轉化為「行動」

**核心職責**:
- 將 AI 決策轉換為 AST 攻擊計劃
- 編排任務序列 (支持拓撲排序和並行)
- 選擇合適的工具/功能服務
- 執行具體任務並管理狀態

**關鍵組件**:
```python
from task_planning.planner import ASTParser, AttackOrchestrator
from task_planning.planner import TaskConverter, ToolSelector
from task_planning.executor import PlanExecutor, TaskExecutor
```

**對應設計理念**: **執行循環** (決策 → 行動)

---

### **模組 4: 📈 external_learning (對外探索與學習)**

**定位**: 從「行動結果」中學習

**核心職責**:
- 記錄實際執行的 Trace (軌跡)
- 對比 AST (計劃) vs Trace (現實)
- 生成偏差報告作為訓練數據
- 動態調整攻擊策略
- 優化 AI 模型權重

**關鍵組件**:
```python
from external_learning.analysis import ASTTraceComparator
from external_learning.analysis import DynamicStrategyAdjustment
from external_learning.tracing import UnifiedTracer
from external_learning.learning import ModelTrainer, ExperienceManager
```

**對應設計理念**: **外部閉環步驟 4+5+6** (掃描 + 攻擊 + 數據收集)

---

### **模組 5: 🛠️ core_capabilities (核心能力)**

**定位**: AI 實際擁有的「手」和「腳」

**核心職責**:
- 提供所有可執行的攻擊能力
- 提供代碼分析和模式識別能力
- 提供插件化的擴展能力
- 被 `@register_capability` 標記以便被探索

**關鍵組件**:
```python
from core_capabilities.attack import PayloadGenerator, ExploitManager
from core_capabilities.analysis import AnalysisEngine
from core_capabilities.plugins import AISummaryPlugin
from core_capabilities import CapabilityRegistry
```

**對應設計理念**: **內部閉環發現對象** (被探索的能力)

---

### **模組 6: 🏗️ service_backbone (服務骨幹)**

**定位**: 支撐整個系統運行的基礎設施

**核心職責**:
- 提供 FastAPI 服務入口
- 協調各模組間的調用
- 提供事件驅動的消息系統
- 提供性能優化和監控
- 提供安全和權限控制

**關鍵組件**:
```python
from service_backbone.api import app  # FastAPI
from service_backbone.coordination import CoreServiceCoordinator
from service_backbone.messaging import EnhancedMessageBroker
from service_backbone.performance import OptimizedCore
from service_backbone.authz import RiskGuard
```

**對應設計理念**: **基礎設施層** (支撐雙重閉環運行)

---

## 📋 模組遷移映射表

| 當前位置 | 新位置 | 模組 | 備註 |
|---------|--------|------|------|
| `ai_engine/real_neural_core.py` | `cognitive_core/neural/` | 1 | 神經網路核心 |
| `ai_engine/real_bio_net_adapter.py` | `cognitive_core/neural/` | 1 | 生物網路適配器 |
| `rag/rag_engine.py` | `cognitive_core/rag/` | 1 | RAG 引擎 |
| `rag/knowledge_base.py` | `cognitive_core/rag/` | 1 | 知識庫 |
| `decision/enhanced_decision_agent.py` | `cognitive_core/decision/` | 1 | 決策代理 |
| `ai_engine/anti_hallucination_module.py` | `cognitive_core/anti_hallucination/` | 1 | 反幻覺 |
| `ai_engine/module_explorer.py` | `internal_exploration/` | 2 | 模組探索器 |
| `ai_engine/capability_analyzer.py` | `internal_exploration/` | 2 | 能力分析器 |
| **新建** | `internal_exploration/ast_code_analyzer.py` | 2 | AST 解析器 |
| **新建** | `internal_exploration/knowledge_graph.py` | 2 | 知識圖譜 |
| **新建** | `internal_exploration/self_diagnostics.py` | 2 | 自我診斷 |
| `planner/*` | `task_planning/planner/` | 3 | 整個規劃器 |
| `execution/*` | `task_planning/executor/` | 3 | 整個執行器 |
| `analysis/ast_trace_comparator.py` | `external_learning/analysis/` | 4 | AST 對比器 |
| `analysis/dynamic_strategy_adjustment.py` | `external_learning/analysis/` | 4 | 策略調整 |
| `execution/unified_tracer.py` | `external_learning/tracing/` | 4 | 統一追蹤器 |
| `learning/*` | `external_learning/learning/` | 4 | 整個學習系統 |
| **確認/新建** | `external_learning/learning/experience_manager.py` | 4 | 經驗管理器 |
| `attack/*` | `core_capabilities/attack/` | 5 | 整個攻擊模組 |
| `ai_analysis/analysis_engine.py` | `core_capabilities/analysis/` | 5 | 分析引擎 |
| `plugins/*` | `core_capabilities/plugins/` | 5 | 插件系統 |
| `app.py` | `service_backbone/api/` | 6 | FastAPI 入口 |
| `core_service_coordinator.py` | `service_backbone/coordination/` | 6 | 核心協調器 |
| `messaging/*` | `service_backbone/messaging/` | 6 | 消息系統 |
| `performance/*` | `service_backbone/performance/` | 6 | 性能優化 |
| `authz/*` | `service_backbone/authz/` | 6 | 授權控制 |
| `monitoring/*` | `service_backbone/monitoring/` | 6 | 監控系統 |

---

## 🔗 關鍵連接器設計

### **InternalLoopConnector (內部閉環連接器)**

**位置**: `cognitive_core/internal_loop_connector.py`

**功能**: 將對內探索結果 → RAG 知識庫

```python
from typing import Dict, Any
from internal_exploration import ModuleExplorer, CapabilityAnalyzer, KnowledgeGraph
from cognitive_core.rag import RAGEngine

class InternalLoopConnector:
    """連接內部閉環: 探索 → 分析 → RAG → AI 決策"""
    
    def __init__(self):
        self.explorer = ModuleExplorer()
        self.analyzer = CapabilityAnalyzer()
        self.knowledge_graph = KnowledgeGraph()
        self.rag_engine = RAGEngine()
    
    async def sync_capabilities_to_rag(self) -> Dict[str, Any]:
        """將系統能力灌入 RAG 知識庫"""
        # 1. 探索五大模組
        modules_info = await self.explorer.explore_all_modules()
        
        # 2. 分析能力
        capabilities = await self.analyzer.analyze_capabilities(modules_info)
        
        # 3. 構建知識圖譜
        capability_graph = self.knowledge_graph.build_graph(capabilities)
        
        # 4. 灌入 RAG
        await self.rag_engine.ingest_knowledge_graph(capability_graph)
        
        return {
            "status": "success",
            "capabilities_count": len(capabilities),
            "graph_nodes": len(capability_graph.nodes),
            "graph_edges": len(capability_graph.edges)
        }
    
    async def diagnose_system_health(self) -> Dict[str, Any]:
        """診斷系統健康狀況"""
        graph = await self.knowledge_graph.get_current_graph()
        issues = self.knowledge_graph.find_broken_paths(graph)
        return {
            "health_status": "healthy" if not issues else "degraded",
            "issues": issues
        }
```

---

### **ExternalLoopConnector (外部閉環連接器)**

**位置**: `cognitive_core/external_loop_connector.py`

**功能**: 將偏差報告 → 學習/經驗管理器

```python
from typing import Dict, Any
from external_learning.analysis import ASTTraceComparator, DynamicStrategyAdjustment
from external_learning.learning import ExperienceManager, ModelTrainer

class ExternalLoopConnector:
    """連接外部閉環: 掃描 → 攻擊 → 偏差分析 → 學習優化"""
    
    def __init__(self):
        self.comparator = ASTTraceComparator()
        self.experience_mgr = ExperienceManager()
        self.trainer = ModelTrainer()
        self.strategy_adjuster = DynamicStrategyAdjustment()
    
    async def process_execution_result(
        self,
        ast_plan: Dict[str, Any],
        actual_trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        """處理執行結果並觸發學習"""
        # 1. 對比 AST vs Trace
        deviation_report = self.comparator.compare(ast_plan, actual_trace)
        
        # 2. 記錄到經驗管理器
        await self.experience_mgr.record_deviation(deviation_report)
        
        # 3. 判斷是否需要訓練
        if await self.experience_mgr.should_trigger_training():
            valuable_deviations = await self.experience_mgr.get_valuable_deviations()
            training_result = await self.trainer.train_from_deviations(valuable_deviations)
        else:
            training_result = None
        
        # 4. 動態調整策略
        await self.strategy_adjuster.adjust_from_report(deviation_report)
        
        return {
            "status": "success",
            "deviation_severity": deviation_report.get("severity"),
            "training_triggered": training_result is not None,
            "strategy_adjusted": True
        }
```

---

## 📊 數據流圖

### **完整雙重閉環數據流**

```
┌──────────────────────────────────────────────────────────────┐
│                    AIVA Core v3.0 數據流                       │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  外部請求 (FastAPI) → service_backbone.api.app              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  AI 指揮官 (ai_commander.py)                                 │
│  • 分析任務類型                                              │
│  • 協調各模組                                                │
└─────────────────────────────────────────────────────────────┘
          ↓                                    ↓
    [對內探索路徑]                       [對外執行路徑]
          ↓                                    ↓
┌──────────────────────┐         ┌──────────────────────────┐
│ internal_exploration │         │ cognitive_core           │
│ • 探索五大模組       │         │ • RAG 檢索知識           │
│ • 分析能力           │         │ • 神經網路推理           │
│ • 構建知識圖譜       │         │ • 生成決策               │
└──────────────────────┘         └──────────────────────────┘
          ↓                                    ↓
┌──────────────────────┐         ┌──────────────────────────┐
│ InternalLoopConnector│         │ task_planning            │
│ • 同步到 RAG         │         │ • AST 規劃               │
└──────────────────────┘         │ • 任務編排               │
                                  │ • 工具選擇               │
                                  └──────────────────────────┘
                                             ↓
                                  ┌──────────────────────────┐
                                  │ core_capabilities        │
                                  │ • 執行攻擊               │
                                  │ • 記錄 Trace             │
                                  └──────────────────────────┘
                                             ↓
                                  ┌──────────────────────────┐
                                  │ external_learning        │
                                  │ • AST vs Trace 對比      │
                                  │ • 生成偏差報告           │
                                  └──────────────────────────┘
                                             ↓
                                  ┌──────────────────────────┐
                                  │ ExternalLoopConnector    │
                                  │ • 記錄經驗               │
                                  │ • 觸發訓練               │
                                  │ • 調整策略               │
                                  └──────────────────────────┘
                                             ↓
                                    (循環回到 cognitive_core)
```

---

## ⚠️ 風險與挑戰

### **技術風險**

1. **導入路徑大量變更** 🔴 高風險
   - **影響**: 所有文件的 import 語句需要批量更新
   - **緩解**: 使用腳本批量替換,保留向後兼容別名

2. **模組間循環依賴** 🟡 中風險
   - **影響**: 可能出現循環導入錯誤
   - **緩解**: 使用依賴注入,定義清晰的接口邊界

3. **測試覆蓋率下降** 🟡 中風險
   - **影響**: 重構後測試可能失效
   - **緩解**: 先執行測試備份,逐步修復

### **業務風險**

1. **功能暫時不可用** 🟡 中風險
   - **影響**: 重構期間部分功能可能無法使用
   - **緩解**: 採用分支開發,主分支保持穩定

2. **遷移成本高** 🟢 低風險
   - **影響**: 需要更新文檔和指南
   - **緩解**: 創建詳細的遷移指南

---

## 📅 實施時間表

### **Week 1-2: 基礎架構搭建**
- ✅ 創建六大模組目錄結構
- ✅ 創建所有 README.md
- ✅ 創建兩個連接器骨架

### **Week 3-4: 核心模組遷移**
- ✅ 遷移 cognitive_core (模組 1)
- ✅ 遷移 internal_exploration (模組 2)
- ✅ 遷移 task_planning (模組 3)

### **Week 5: 學習與能力模組**
- ✅ 遷移 external_learning (模組 4)
- ✅ 遷移 core_capabilities (模組 5)

### **Week 6: 基礎設施與測試**
- ✅ 遷移 service_backbone (模組 6)
- ✅ 批量更新導入路徑
- ✅ 執行完整測試驗證

---

## ✅ 成功標準

1. ✅ 所有測試通過 (單元測試 + 集成測試)
2. ✅ 性能無退化 (對比基準測試)
3. ✅ 雙重閉環數據流正常運作
4. ✅ 文檔完整更新
5. ✅ 代碼覆蓋率維持或提升
6. ✅ 無循環依賴錯誤
7. ✅ 遷移指南清晰可用

---

## 📚 參考文檔

- [`AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)
- [`TERMINOLOGY_GLOSSARY.md`](../../TERMINOLOGY_GLOSSARY.md)
- [`services/core/aiva_core/README.md`](README.md)
- [`guides/modules/AI_ENGINE_GUIDE.md`](../../guides/modules/AI_ENGINE_GUIDE.md)

---

**📝 文檔資訊**
- **創建日期**: 2025-11-15
- **維護者**: AIVA 核心團隊
- **版本**: v1.0
- **狀態**: 📋 規劃中
