# 📝 Planner - 任務規劃器

**導航**: [← 返回 Task Planning](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **狀態**: 生產就緒  
> **角色**: 策略轉換和任務生成

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [使用範例](#使用範例)

---

## 🎯 模組概述

Planner 子模組負責將高層策略轉換為可執行任務，包含 AST 解析、任務生成、編排、工具選擇等核心功能。

### 核心功能
- **AST 解析** - 解析攻擊流程抽象語法樹
- **任務生成** - 從策略生成具體任務
- **執行編排** - 協調多個任務的執行順序
- **工具選擇** - 為任務選擇合適的執行工具
- **任務轉換** - 將任務轉換為可執行格式

---

## 📂 檔案列表

| 檔案 | 行數 | 功能 | 狀態 |
|------|------|------|------|
| `task_generator.py` | ~400 | 任務生成器 | ✅ |
| `orchestrator.py` | ~500 | 攻擊編排器 | ✅ |
| `execution_planner.py` | ~450 | 執行計劃器 | ✅ |
| `ast_parser.py` | 281 | AST 攻擊流程圖解析 | ✅ |
| `task_converter.py` | ~300 | 任務轉換器 | ✅ |
| `tool_selector.py` | 219 | 工具選擇器 | ✅ |
| `strategy_generator.py` | ~350 | 策略生成器（舊版） | 🔧 |
| `plan_comparator.py` | ~200 | 計畫比較器 | ✅ |
| `__init__.py` | ~50 | 模組入口 | ✅ |

**總計**: 9 個 Python 檔案，約 2750+ 行代碼

---

## 🔧 核心組件

### 1. `task_generator.py` - 任務生成器

**功能**: 從策略生成具體的可執行任務

**生成流程**:
```python
策略 → 解析目標 → 選擇測試項目 → 生成任務參數 → 分配佇列主題
```

**使用範例**:
```python
from task_planning.planner import TaskGenerator

generator = TaskGenerator()

# 從攻擊策略生成任務
tasks = generator.from_strategy(
    attack_plan={
        "target": "https://example.com",
        "modules": ["sqli", "xss", "csrf"],
        "depth": "medium"
    },
    scan_payload={
        "user_id": "user_001",
        "scan_id": "scan_001"
    }
)

# 返回格式: [(topic, task_payload), ...]
for topic, task in tasks:
    print(f"佇列主題: {topic}")
    print(f"任務: {task}")
```

**生成的任務類型**:
- `初始掃描任務` - 信息收集和指紋識別
- `漏洞檢測任務` - 各種漏洞類型的測試
- `業務邏輯任務` - 業務流程測試
- `驗證任務` - 結果確認和驗證

---

### 2. `orchestrator.py` - 攻擊編排器

**功能**: 創建和編排多步驟攻擊執行計劃

**編排策略**:
```python
AttackOrchestrator
├── 依賴分析 - 識別任務依賴關係
├── 順序規劃 - 確定執行順序
├── 並行優化 - 識別可並行任務
└── 資源分配 - 分配執行資源
```

**使用範例**:
```python
from task_planning.planner import AttackOrchestrator

orchestrator = AttackOrchestrator()

# 創建執行計劃
execution_plan = orchestrator.create_execution_plan(
    ast_input={
        "nodes": [
            {"type": "scan", "target": "example.com"},
            {"type": "analyze", "depends_on": ["scan"]},
            {"type": "exploit", "depends_on": ["analyze"]}
        ],
        "edges": [...]
    },
    context={
        "max_parallel": 5,
        "timeout": 3600
    }
)

# 執行計劃包含
print(f"階段數: {len(execution_plan.stages)}")
print(f"總任務數: {execution_plan.total_tasks}")
print(f"並行度: {execution_plan.max_parallel}")
```

**編排結果**:
```python
@dataclass
class ExecutionPlan:
    plan_id: str
    stages: list[Stage]  # 執行階段
    total_tasks: int
    estimated_time: int
    max_parallel: int
    dependencies: dict[str, list[str]]
```

---

### 3. `execution_planner.py` - 執行計劃器

**功能**: 高層執行計劃的創建和優化

**計劃維度**:
- **時間維度** - 任務執行時間估算
- **資源維度** - CPU/內存/網絡資源分配
- **優先級維度** - 任務優先級排序
- **風險維度** - 執行風險評估

**使用範例**:
```python
from task_planning.planner import ExecutionPlanner

planner = ExecutionPlanner()

# 創建詳細執行計劃
plan = planner.create_plan(
    tasks=task_list,
    constraints={
        "max_time": 3600,
        "max_resource": {"cpu": 4, "memory": "8GB"},
        "priority": "high"
    },
    optimization_goal="speed"  # 或 "resource", "balance"
)

# 優化計劃
optimized_plan = planner.optimize(
    plan=plan,
    feedback=execution_feedback
)
```

---

### 4. `ast_parser.py` - AST 解析器

**功能**: 解析 AI 生成的攻擊流程抽象語法樹

**節點類型**:
```python
class NodeType(Enum):
    START = "start"        # 開始節點
    SCAN = "scan"          # 掃描/探測
    ANALYZE = "analyze"    # 分析
    EXPLOIT = "exploit"    # 漏洞利用
    VALIDATE = "validate"  # 驗證
    BRANCH = "branch"      # 條件分支
    LOOP = "loop"          # 循環
    END = "end"            # 結束節點
```

**使用範例**:
```python
from task_planning.planner import ASTParser, NodeType

parser = ASTParser()

# 解析 AST
ast_graph = parser.parse(
    ast_input={
        "nodes": [
            {"id": "n1", "type": "START"},
            {"id": "n2", "type": "SCAN", "params": {"target": "example.com"}},
            {"id": "n3", "type": "ANALYZE", "params": {"focus": "sqli"}},
            {"id": "n4", "type": "EXPLOIT", "params": {"payload": "..."}},
            {"id": "n5", "type": "END"}
        ],
        "edges": [
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"},
            {"from": "n3", "to": "n4"},
            {"from": "n4", "to": "n5"}
        ]
    }
)

# 遍歷圖
execution_order = parser.topological_sort(ast_graph)
print(f"執行順序: {execution_order}")

# 檢測循環
has_cycle = parser.detect_cycle(ast_graph)
```

**AST 結構**:
```python
@dataclass
class ASTNode:
    node_id: str
    node_type: NodeType
    params: dict[str, Any]
    children: list[str]
    metadata: dict[str, Any]

@dataclass
class ASTGraph:
    nodes: dict[str, ASTNode]
    edges: list[tuple[str, str]]
    entry_point: str
    exit_points: list[str]
```

---

### 5. `task_converter.py` - 任務轉換器

**功能**: 將任務轉換為可執行格式

**轉換類型**:
```python
TaskConverter
├── 策略 → 任務
├── AST → 任務
├── 自然語言 → 任務
└── 舊格式 → 新格式
```

**使用範例**:
```python
from task_planning.planner import TaskConverter, ExecutableTask

converter = TaskConverter()

# 轉換任務
executable = converter.convert(
    source_task={
        "type": "vulnerability_scan",
        "target": "https://example.com",
        "modules": ["sqli", "xss"]
    },
    format="strategy"
)

# 執行任務結構
@dataclass
class ExecutableTask:
    task_id: str
    task_type: str
    params: dict[str, Any]
    dependencies: list[str]
    priority: int
    timeout: int
```

---

### 6. `tool_selector.py` - 工具選擇器

**功能**: 為任務選擇合適的執行工具和服務

**選擇策略**:
```python
ToolSelector
├── 任務類型匹配
├── 能力評估
├── 資源可用性
└── 性能考量
```

**服務類型**:
```python
class ServiceType(Enum):
    SCAN_SERVICE = "scan_service"
    FUNCTION_SQLI = "function_sqli"
    FUNCTION_XSS = "function_xss"
    FUNCTION_SSRF = "function_ssrf"
    FUNCTION_IDOR = "function_idor"
    INTEGRATION_SERVICE = "integration_service"
    CORE_ANALYZER = "core_analyzer"
```

**使用範例**:
```python
from task_planning.planner import ToolSelector, ToolDecision

selector = ToolSelector()

# 選擇工具
decision = selector.select_tool(
    task={
        "type": "sql_injection_test",
        "target": "https://example.com/api",
        "complexity": "high"
    }
)

print(f"選擇的服務: {decision.service_type}")
print(f"工具參數: {decision.tool_params}")
print(f"預期能力: {decision.expected_capabilities}")
```

---

### 7. `plan_comparator.py` - 計畫比較器

**功能**: 比較和評估不同執行計劃

**比較維度**:
- **效率** - 執行時間和資源使用
- **完整性** - 測試覆蓋度
- **風險** - 執行風險評估
- **成本** - 資源成本

**使用範例**:
```python
from task_planning.planner import PlanComparator

comparator = PlanComparator()

# 比較兩個計劃
comparison = comparator.compare(
    plan_a=plan_a,
    plan_b=plan_b,
    criteria=["efficiency", "coverage", "risk"]
)

print(f"Plan A 評分: {comparison.plan_a_score}")
print(f"Plan B 評分: {comparison.plan_b_score}")
print(f"推薦: {comparison.recommendation}")
```

---

## 🚀 完整使用流程

### 從策略到執行
```python
from task_planning.planner import (
    TaskGenerator,
    ASTParser,
    AttackOrchestrator,
    ToolSelector
)

# 1. 解析 AI 生成的 AST
parser = ASTParser()
ast_graph = parser.parse(ai_generated_ast)

# 2. 編排執行計劃
orchestrator = AttackOrchestrator()
execution_plan = orchestrator.create_execution_plan(ast_graph)

# 3. 生成具體任務
generator = TaskGenerator()
tasks = generator.from_execution_plan(execution_plan)

# 4. 為每個任務選擇工具
selector = ToolSelector()
for task in tasks:
    tool_decision = selector.select_tool(task)
    task.tool = tool_decision
    task.service = tool_decision.service_type

# 5. 返回完整的執行計劃
return {
    "plan": execution_plan,
    "tasks": tasks,
    "total_tasks": len(tasks),
    "estimated_time": execution_plan.estimated_time
}
```

---

## 📊 性能指標

| 指標 | 數值 | 備註 |
|------|------|------|
| AST 解析 | < 100ms | 中等複雜度 |
| 任務生成 | < 200ms | 10-20 任務 |
| 編排計劃 | < 500ms | 50+ 任務 |
| 工具選擇 | < 50ms | 單次選擇 |

---

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team
