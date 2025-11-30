# 🎯 Task Planning - 任務規劃與執行

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心職責](#核心職責)
  - [設計理念](#設計理念)
  - [🎯 核心職責](#-核心職責)
- [🏗️ 架構設計](#-架構設計)
  - [執行流程](#執行流程)
  - [2. 🔀 Command Router (命令路由器)](#2--command-router-命令路由器)
  - [3. 📝 Planner (規劃器)](#3--planner-規劃器)
  - [4. ⚙️ Executor (執行器)](#4--executor-執行器)
- [📖 使用範例](#-使用範例)
  - [完整的任務規劃與執行流程](#完整的任務規劃與執行流程)
- [🛠️ 開發指南](#-開發指南)
  - [🔨 aiva_common 修復規範](#-aiva_common-修復規範)
  - [添加新的任務類型](#添加新的任務類型)
  - [實現自定義命令處理器](#實現自定義命令處理器)
- [📊 性能指標](#-性能指標)
- [🔗 相關模組](#-相關模組)

---

**導航**: [← 返回 AIVA Core](../README.md)

> **版本**: v2.1.2 (生產就緒)  
> **狀態**: ✅ 生產就緒，100% 類型安全  
> **🧪 測試狀態**: 階段 5 測試 100% 通過 (3/3 組件)  
> **代碼品質**: Phase 3 完成 - 0 個真實錯誤  
> **角色**: AIVA 的「執行大腦」- 將策略轉化為任務並協調執行  
> **最後更新**: 2025年12月20日

---

## 🎯 模組概述

**Task Planning** 是 AIVA 六大模組架構中的執行規劃層，負責將抽象的攻擊策略轉化為具體的可執行任務。整合了 AI 指揮系統、命令路由、任務生成、執行編排、狀態監控等核心能力，確保攻擊流程高效有序地執行。

### 核心職責
1. **AI 指揮** - 統一指揮所有 AI 組件的協調中心
2. **任務生成** - 將策略轉換為具體的功能測試任務
3. **執行編排** - AST 解析、工具選擇、任務序列化
4. **任務調度** - 優先級佇列、負載平衡、並行執行
5. **狀態監控** - Worker 心跳、SLA 追蹤、健康檢查
6. **命令路由** - 智能路由、複雜度分析、執行模式選擇

### 設計理念
- **策略驅動** - 從高層策略到底層任務的自動轉換
- **AI 增強** - 整合多個 AI 組件智能決策
- **異步執行** - 支援並行和流式執行模式
- **可觀測性** - 全程追蹤任務執行狀態

### 🎯 核心職責

- ✅ **AST 規劃**: 將 AI 決策轉換為 AST 攻擊計劃
---

## 🏗️ 架構設計

```
task_planning/
├── 📁 planner/                   # 規劃器 (9 檔案) - [📖 README](./planner/README.md)
│   ├── task_generator.py         # ✅ 任務生成器
│   ├── orchestrator.py           # ✅ 攻擊編排器
│   ├── execution_planner.py      # ✅ 執行計劃器
│   ├── ast_parser.py             # ✅ AST 攻擊流程圖解析 (281行)
│   ├── task_converter.py         # ✅ 任務轉換器
│   ├── tool_selector.py          # ✅ 工具選擇器 (219行)
│   ├── plan_comparator.py        # ✅ 計畫比較器
│   ├── strategy_generator.py     # 🔧 策略生成器（舊版）
│   └── __init__.py
│
├── 📁 executor/                  # 執行器 (6 檔案) - [📖 README](./executor/README.md)
│   ├── task_executor.py          # ✅ 任務執行器 (279行)
│   ├── task_queue_manager.py    # ✅ 任務佇列管理器
│   ├── execution_status_monitor.py  # ✅ 執行狀態監控器
│   ├── plan_executor.py          # ✅ 計畫執行器
│   ├── attack_plan_mapper.py     # 🔧 攻擊計畫映射器（舊版）
│   └── __init__.py
│
├── ai_commander.py               # ✅ AI 指揮系統
├── command_router.py             # ✅ 命令路由器
└── __init__.py

總計: 18 個 Python 檔案
```

### 執行流程
```
┌─────────────────────────────────────────────────────────┐
│           Task Planning (任務規劃與執行)                  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │           AI Commander (AI 指揮中心)              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │  │
│  │  │BioNeuronRAG │  │  RAG Engine │  │ Training │ │  │
│  │  │   Agent     │  │   (知識)    │  │  (學習)  │ │  │
│  │  └─────────────┘  └─────────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
│                           ▼                            │
│  ┌──────────────────────────────────────────────────┐  │
│  │       Command Router (命令路由器)                 │  │
│  │    智能路由 • 複雜度分析 • 執行模式選擇           │  │
│  └──────────────────────────────────────────────────┘  │
│                           ▼                            │
│         ┌─────────────────┴─────────────────┐          │
│         │                                   │          │
│    ┌────▼──────┐                      ┌────▼──────┐   │
│    │  Planner  │                      │ Executor  │   │
│    │  (規劃器) │                      │ (執行器)  │   │
│    └───────────┘                      └───────────┘   │
│         │                                   │          │
│    ┌────▼──────────────┐         ┌─────────▼────────┐ │
│    │ Task Generator    │         │  Task Queue Mgr  │ │
│    │ Orchestrator      │────────▶│  Status Monitor  │ │
│    │ Execution Planner │         │  Task Executor   │ │
│    └───────────────────┘         └──────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼────┐     ┌─────▼─────┐    ┌───▼────┐
     │  Core   │     │  Service  │    │External│
     │Capability│     │ Backbone  │    │Learning│
     └─────────┘     └───────────┘    └────────┘
```

---

### 2. 🔀 Command Router (命令路由器)

#### `command_router.py` - 智能命令路由系統
**功能**: AI vs 非 AI 自動判斷和複雜度分析
```python
from task_planning import CommandRouter, CommandContext

router = CommandRouter()

context = CommandContext(
    command="analyze_vulnerability",
    args={"target": "https://example.com"}
)

# 自動路由
route_info = router.route_command(context)
print(f"需要 AI: {route_info['requires_ai']}")
```

### 3. 📝 Planner (規劃器)

#### `task_generator.py` - 任務生成器
```python
from task_planning.planner import TaskGenerator

generator = TaskGenerator()
tasks = generator.from_strategy(attack_plan, scan_payload)
```

#### `orchestrator.py` - 攻擊編排器
```python
from task_planning.planner import AttackOrchestrator

orchestrator = AttackOrchestrator()
execution_plan = orchestrator.create_execution_plan(ast_input)
```

### 4. ⚙️ Executor (執行器)

#### `task_queue_manager.py` - 任務佇列管理器
```python
from task_planning.executor import TaskQueueManager

queue_manager = TaskQueueManager()
queue_manager.enqueue_task(topic, task_payload)
```

#### `execution_status_monitor.py` - 執行狀態監控器
```python
from task_planning.executor import ExecutionStatusMonitor

monitor = ExecutionStatusMonitor()
monitor.record_task_start(task_id, worker_id)
```

---

## 📖 使用範例

### 完整的任務規劃與執行流程
```python
from task_planning import (
    AICommander,
    TaskGenerator,
    TaskQueueManager,
    ExecutionStatusMonitor
)

# 初始化組件
ai_commander = AICommander()
generator = TaskGenerator()
queue_manager = TaskQueueManager()
monitor = ExecutionStatusMonitor()

await ai_commander.initialize()

# AI 生成攻擊策略
strategy = await ai_commander.execute_ai_task(
    task_type=AITaskType.ATTACK_PLANNING,
    context={"target": "https://example.com"}
)

# 生成任務
tasks = generator.from_strategy(strategy["attack_plan"], scan_payload)

# 添加到佇列
for topic, task in tasks:
    queue_manager.enqueue_task(topic, task)

# 監控執行
stats = queue_manager.get_scan_statistics("scan_001")
print(f"進度: {stats['completed']}/{stats['total']}")
```

---

## 🛠️ 開發指南

### 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../aiva_common/README.md#-開發指南) 的修復規範。

**完整規範**: [aiva_common 開發指南](../../../aiva_common/README.md#-開發指南)

#### 關鍵原則

```python
# ✅ 正確：使用 aiva_common 標準
from aiva_common import TaskStatus, Severity, Confidence

# ❌ 禁止：重複定義
class TaskStatus(str, Enum): pass  # 錯誤！

# ✅ 合理的模組專屬枚舉（task_converter.py 範例）
class TaskPriority(str, Enum):
    """任務優先級 (AI 規劃器專用)"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
```

**判斷標準**:
- ✅ 模組內部專用 → 可自定義
- ❌ 通用概念（狀態、嚴重度） → 必須用 aiva_common

📖 **詳細說明**: [修復規範完整版](../../../aiva_common/README.md#-開發規範與最佳實踐)

---

### 添加新的任務類型
```python
# 在 TaskGenerator 添加生成邏輯
class TaskGenerator:
    def from_strategy(self, plan, payload):
        tasks = []
        for index, x in enumerate(plan.get("custom", [])):
            tasks.append((
                Topic.TASK_FUNCTION_CUSTOM,
                FunctionTaskPayload(...)
            ))
        return tasks
```

### 實現自定義命令處理器
```python
class CustomHandler(CommandHandler):
    async def handle(self, context):
        result = await self._process(context)
        return ExecutionResult(success=True, result=result)

router.register_handler("custom", CustomHandler())
```

---

## 📊 性能指標

- **任務生成速度**: 1000+ 任務/秒
- **佇列容量**: 100,000+ 任務
- **調度延遲**: < 1ms
- **並發 Worker**: 100+
- **AI 響應時間**: 1-5 秒

---

## 🔗 相關模組

- **cognitive_core** - 提供 BioNeuronRAG 和 RAG Engine
- **external_learning** - 提供 Training Orchestrator
- **core_capabilities** - 接收並執行生成的任務
- **service_backbone** - 提供消息代理和狀態管理

---

**最後更新**: 2025-11-15  
**維護者**: AIVA Development Team  
**授權**: MIT License
