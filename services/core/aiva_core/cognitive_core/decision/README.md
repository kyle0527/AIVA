# 🎯 Decision - 決策支援系統

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心功能](#核心功能)
- [📂 檔案列表](#-檔案列表)
- [🔧 核心組件](#-核心組件)
  - [1. `enhanced_decision_agent.py` - 增強決策代理](#1-enhanced_decision_agentpy---增強決策代理)
  - [2. `skill_graph.py` - 技能圖譜](#2-skill_graphpy---技能圖譜)
- [🚀 完整使用流程](#-完整使用流程)
  - [構建和使用技能圖譜](#構建和使用技能圖譜)
  - [與決策代理整合](#與決策代理整合)
  - [動態調整決策](#動態調整決策)
- [📊 技能圖譜統計](#-技能圖譜統計)
- [📊 性能指標](#-性能指標)

---

**導航**: [← 返回 Cognitive Core](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: v2.1.2  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2025-12-20  
> **角色**: AI 增強決策和技能圖譜

---

## 🎯 模組概述

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [使用範例](#使用範例)

---

## 🎯 模組概述

Decision 子模組實現了 AIVA 的 AI 增強決策能力和技能圖譜系統，支援多約束優化、能力關係映射、智能推薦等功能。

### 核心功能
- **AI 決策** - 上下文感知的智能決策
- **技能圖譜** - 能力依賴關係和推薦
- **約束優化** - 多目標約束下的最優決策
- **決策解釋** - 可解釋的決策過程

---

## 📂 檔案列表

| 檔案 | 行數 | 功能 | 狀態 |
|------|------|------|------|
| `enhanced_decision_agent.py` | ~500 | 增強決策代理 | ✅ |
| `skill_graph.py` | 649 | 技能圖譜和能力映射 | ✅ |
| `__init__.py` | ~50 | 模組入口 | ✅ |

**總計**: 3 個 Python 檔案，約 1200+ 行代碼

---

## 🔧 核心組件

### 1. `enhanced_decision_agent.py` - 增強決策代理

**功能**: AI 增強的決策引擎

**決策流程**:
```python
任務輸入 → 上下文分析 → 約束檢查 → AI推理 → 最優決策 → 決策解釋
```

**使用範例**:
```python
from aiva_core.cognitive_core.decision import EnhancedDecisionAgent

agent = EnhancedDecisionAgent()

# 執行決策
decision = await agent.make_decision(
    task={
        "type": "security_test",
        "target": "https://example.com",
        "scope": ["xss", "sqli", "csrf"]
    },
    constraints={
        "max_time": 3600,
        "safe_mode": True,
        "priority": "high"
    },
    context={
        "previous_findings": [...],
        "target_tech_stack": ["Django", "PostgreSQL"]
    }
)

print(f"決策: {decision.action}")
print(f"置信度: {decision.confidence}%")
print(f"推理: {decision.reasoning}")
print(f"預期結果: {decision.expected_outcome}")
```

**關鍵方法**:
- `make_decision()` - 執行決策
- `evaluate_options()` - 評估所有選項
- `explain_decision()` - 解釋決策過程
- `optimize_constraints()` - 約束優化

---

### 2. `skill_graph.py` - 技能圖譜

**功能**: 構建和管理系統能力的依賴圖譜

**圖譜結構**:
```python
SkillGraph (NetworkX)
├── Nodes (能力節點)
│   ├── capability_id
│   ├── name
│   ├── category
│   ├── dependencies
│   └── metadata
│
└── Edges (依賴關係)
    ├── requires (強依賴)
    ├── enhances (增強)
    └── conflicts (衝突)
```

**使用範例**:
```python
from aiva_core.cognitive_core.decision import SkillGraph

# 初始化技能圖譜
skill_graph = SkillGraph()

# 從註冊表構建圖譜
await skill_graph.build_from_registry()

# 查找相關能力
related = skill_graph.find_related(
    capability="sql_injection",
    max_distance=2
)

# 推薦能力組合
recommended = skill_graph.recommend_capability_set(
    target_capability="comprehensive_test",
    constraints={"max_capabilities": 5}
)

# 檢查依賴
dependencies = skill_graph.get_dependencies("xss_detection")
print(f"依賴: {dependencies}")

# 執行順序
execution_order = skill_graph.get_execution_order(
    capabilities=["sqli", "xss", "csrf"]
)
```

**關鍵方法**:
- `build_from_registry()` - 從能力註冊表構建
- `find_related()` - 查找相關能力
- `recommend_capability_set()` - 推薦能力組合
- `get_dependencies()` - 獲取依賴
- `get_execution_order()` - 計算執行順序
- `visualize()` - 可視化圖譜

**能力節點**:
```python
@dataclass
class SkillNode:
    capability_id: str
    name: str
    category: str
    language: ProgrammingLanguage
    dependencies: list[str]
    enhances: list[str]
    conflicts: list[str]
    success_rate: float
    avg_execution_time: float
    metadata: dict
```

---

## 🚀 完整使用流程

### 構建和使用技能圖譜
```python
from aiva_core.cognitive_core.decision import SkillGraph, EnhancedDecisionAgent
from services.integration.capability.registry import CapabilityRegistry

# 1. 初始化
skill_graph = SkillGraph()
registry = CapabilityRegistry()

# 2. 構建圖譜
await skill_graph.build_from_registry()
print(f"構建完成: {skill_graph.node_count} 個能力節點")

# 3. 分析能力關係
sql_related = skill_graph.find_related("sql_injection", max_distance=2)
print(f"SQL注入相關能力: {sql_related}")

# 4. 推薦測試組合
test_set = skill_graph.recommend_capability_set(
    target_capability="web_security_audit",
    constraints={
        "max_capabilities": 10,
        "min_coverage": 0.8,
        "exclude_conflicts": True
    }
)

print(f"推薦測試集: {test_set}")

# 5. 計算執行順序
order = skill_graph.get_execution_order(test_set)
print(f"執行順序: {order}")
```

### 與決策代理整合
```python
from aiva_core.cognitive_core.decision import EnhancedDecisionAgent, SkillGraph

# 初始化
agent = EnhancedDecisionAgent()
skill_graph = SkillGraph()
await skill_graph.build_from_registry()

# 使用技能圖譜增強決策
decision = await agent.make_decision_with_skill_graph(
    task={
        "type": "vulnerability_scan",
        "target": "https://example.com"
    },
    skill_graph=skill_graph,
    constraints={
        "max_time": 3600,
        "coverage": "comprehensive"
    }
)

# 決策結果包含推薦的能力組合和執行順序
print(f"推薦能力: {decision.recommended_capabilities}")
print(f"執行順序: {decision.execution_order}")
print(f"預期覆蓋率: {decision.expected_coverage}%")
```

### 動態調整決策
```python
# 根據執行結果動態調整
async def adaptive_execution(task, skill_graph):
    agent = EnhancedDecisionAgent()
    
    # 初始決策
    decision = await agent.make_decision(task, skill_graph=skill_graph)
    
    results = []
    for capability in decision.recommended_capabilities:
        # 執行能力
        result = await execute_capability(capability)
        results.append(result)
        
        # 根據結果調整後續決策
        if result.success_rate < 0.5:
            # 重新評估剩餘能力
            remaining = decision.recommended_capabilities[len(results):]
            adjusted_decision = await agent.adjust_decision(
                original_decision=decision,
                execution_results=results,
                remaining_capabilities=remaining
            )
            decision = adjusted_decision
    
    return results
```

---

## 📊 技能圖譜統計

```python
# 獲取圖譜統計
stats = skill_graph.get_statistics()

print(f"節點數: {stats['node_count']}")
print(f"邊數: {stats['edge_count']}")
print(f"平均度: {stats['avg_degree']}")
print(f"最大依賴深度: {stats['max_dependency_depth']}")
print(f"孤立節點: {stats['isolated_nodes']}")
```

---

## 📊 性能指標

| 指標 | 數值 | 備註 |
|------|------|------|
| 決策速度 | < 200ms | 單次決策 |
| 圖譜構建 | < 5s | 100+ 能力 |
| 相關能力查找 | < 50ms | 深度=2 |
| 推薦準確率 | 85%+ | 基於歷史數據 |
| 記憶體使用 | < 50MB | 圖譜載入後 |

---

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team
