# 📋 Executor - 任務執行器

> **版本**: v2.5.0  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2026-01-07  
> **父模組**: [Task Planning](../README.md)  
> **符合規範**: [aiva_common](../../../../aiva_common/README.md)  
> **檔案數**: 6 個 Python 模組  
> **代碼行數**: 2,134 行

---

## 📋 目錄

- [模組概述](#-模組概述)
- [核心組件](#-核心組件)
- [使用範例](#-使用範例)
- [API 參考](#-api-參考)

---

## 🎯 模組概述

Executor 是 Task Planning 的任務執行子模組，負責實際執行攻擊計劃中的各個步驟。

**核心職責**：
- ⚡ **計劃執行** - 執行 AttackPlan 中定義的步驟序列
- 📊 **狀態追蹤** - 監控任務執行狀態和進度
- 🔄 **任務映射** - 將 AI 決策映射為具體的 FunctionTaskPayload
- 📝 **執行記錄** - 記錄執行軌跡和結果

---

## 🏗️ 核心組件

### 1. PlanExecutor (`plan_executor.py`)

攻擊計劃執行器，負責執行整個攻擊計劃。

**主要功能**：
```python
from services.core.aiva_core.task_planning.executor import PlanExecutor

# 初始化執行器
executor = PlanExecutor(
    message_broker=message_broker,
    unified_tracer=unified_tracer,
)

# 執行攻擊計劃
result = await executor.execute_plan(
    plan=attack_plan,
    sandbox_mode=True,
    timeout_minutes=30,
)
```

### 2. TaskExecutor (`task_executor.py`)

單一任務執行器，與各種服務整合。

**主要功能**：
```python
from services.core.aiva_core.task_planning.executor import TaskExecutor

executor = TaskExecutor(execution_monitor=monitor)

result = await executor.execute_task(
    task=executable_task,
    tool_decision=tool_decision,
    trace_session_id=session_id,
)
```

### 3. AttackPlanMapper (`attack_plan_mapper.py`)

AI 攻擊計劃映射器，將 AI 決策轉換為可執行任務。

**主要功能**：
```python
from services.core.aiva_core.task_planning.executor import AttackPlanMapper

mapper = AttackPlanMapper(default_scan_id="scan_001")

# 映射 AI 決策
tasks = await mapper.map_ai_decision(
    ai_decision=decision_message,
    scan_context=context,
    scan_id="scan_001",
)
```

### 4. ExecutionStatusMonitor (`execution_status_monitor.py`)

執行狀態監控器，追蹤任務執行和系統健康狀態。

**主要類別**：
- `ExecutionContext` - 執行上下文，追蹤任務環境信息
- `ExecutionMonitor` - 執行監控器，記錄決策和工具調用
- `ExecutionStatusMonitor` - 狀態監控器，SLA 追蹤和警報

### 5. TaskQueueManager (`task_queue_manager.py`)

任務隊列管理器，管理任務的排程和優先級。

---

## 📊 數據流

```
AttackPlan → PlanExecutor → TaskExecutor → 服務層
     ↓              ↓             ↓
  SessionState  TraceRecord  ExecutionResult
```

---

## 🔗 依賴關係

**內部依賴**：
- `services.aiva_common.schemas` - FunctionTaskPayload, TraceRecord, SessionState
- `services.aiva_common.enums` - ModuleName, Topic
- `cognitive_core.learning_system.tracing` - UnifiedTracer

**外部依賴**：
- `service_backbone.messaging` - MessageBroker
- `core_capabilities` - CapabilityRegistry

---

## 📝 符合規範

本模組符合 aiva_common 規範：
- ✅ 使用標準化 Schema（FunctionTaskPayload, TraceRecord 等）
- ✅ 使用標準枚舉（ModuleName, Topic）
- ✅ 無重複定義
- ✅ 正確的模組間通信格式
