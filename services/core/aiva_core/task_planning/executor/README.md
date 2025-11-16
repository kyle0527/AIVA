# ⚙️ Executor - 任務執行器

**導航**: [← 返回 Task Planning](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **狀態**: 生產就緒  
> **角色**: 任務執行和狀態監控

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [使用範例](#使用範例)

---

## 🎯 模組概述

Executor 子模組負責實際執行任務、管理任務佇列、監控執行狀態，確保任務按計劃順利執行。

### 核心功能
- **任務執行** - 實際執行各類測試任務
- **佇列管理** - 管理任務優先級佇列
- **狀態監控** - 追蹤任務執行狀態和健康度
- **結果收集** - 收集和聚合執行結果
- **錯誤處理** - 處理執行過程中的異常

---

## 📂 檔案列表

| 檔案 | 行數 | 功能 | 狀態 |
|------|------|------|------|
| `task_executor.py` | 279 | 任務執行器 | ✅ |
| `task_queue_manager.py` | ~400 | 任務佇列管理器 | ✅ |
| `execution_status_monitor.py` | ~500 | 執行狀態監控器 | ✅ |
| `plan_executor.py` | ~350 | 計畫執行器 | ✅ |
| `attack_plan_mapper.py` | ~300 | 攻擊計畫映射器（舊版） | 🔧 |
| `__init__.py` | ~50 | 模組入口 | ✅ |

**總計**: 6 個 Python 檔案，約 1880+ 行代碼

---

## 🔧 核心組件

### 1. `task_executor.py` - 任務執行器

**功能**: 實際執行任務並與各種服務整合

**執行流程**:
```python
接收任務 → 驗證參數 → 選擇服務 → 執行任務 → 收集結果 → 錯誤處理
```

**使用範例**:
```python
from task_planning.executor import TaskExecutor, ExecutionResult

executor = TaskExecutor()

# 執行任務
result = await executor.execute(
    task={
        "task_id": "task_001",
        "type": "sql_injection_test",
        "target": "https://example.com/api",
        "params": {
            "payload": "' OR '1'='1",
            "method": "POST"
        }
    }
)

# 處理結果
if result.success:
    print(f"任務完成: {result.output}")
else:
    print(f"執行失敗: {result.error}")

# 執行結果結構
@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    output: dict[str, Any]
    error: str | None = None
    trace_session_id: str | None = None
    execution_time: float = 0.0
    resource_usage: dict = field(default_factory=dict)
```

**支援的任務類型**:
- `vulnerability_scan` - 漏洞掃描
- `sql_injection_test` - SQL 注入測試
- `xss_test` - XSS 測試
- `business_logic_test` - 業務邏輯測試
- `custom_function` - 自定義函數執行

**服務整合**:
```python
# 與不同服務整合
executor = TaskExecutor(
    scan_service=scan_service,
    function_registry=function_registry,
    integration_service=integration_service
)

# 自動路由到正確的服務
result = await executor.execute(task)
```

---

### 2. `task_queue_manager.py` - 任務佇列管理器

**功能**: 管理任務優先級佇列和調度

**佇列架構**:
```python
TaskQueueManager
├── High Priority Queue (高優先級)
├── Normal Priority Queue (正常優先級)
├── Low Priority Queue (低優先級)
└── Dead Letter Queue (失敗任務)
```

**使用範例**:
```python
from task_planning.executor import TaskQueueManager

queue_manager = TaskQueueManager()

# 添加任務到佇列
queue_manager.enqueue_task(
    topic="vulnerability_scan",
    task_payload={
        "task_id": "task_001",
        "target": "https://example.com",
        "priority": "high"
    }
)

# 從佇列獲取任務
task = await queue_manager.dequeue_task(
    topic="vulnerability_scan",
    worker_id="worker_001"
)

# 確認任務完成
queue_manager.acknowledge_task(
    task_id="task_001",
    success=True
)

# 重試失敗任務
queue_manager.retry_task(
    task_id="task_001",
    delay_seconds=60
)

# 獲取佇列統計
stats = queue_manager.get_queue_stats(topic="vulnerability_scan")
print(f"待處理: {stats.pending}")
print(f"執行中: {stats.processing}")
print(f"已完成: {stats.completed}")
print(f"失敗: {stats.failed}")
```

**佇列特性**:
- ✅ 優先級調度
- ✅ 任務去重
- ✅ 自動重試
- ✅ 死信佇列
- ✅ 負載平衡
- ✅ 持久化（可選）

**佇列配置**:
```python
queue_manager = TaskQueueManager(
    config={
        "max_retries": 3,
        "retry_delay": 60,
        "enable_persistence": True,
        "max_queue_size": 10000,
        "worker_timeout": 300
    }
)
```

---

### 3. `execution_status_monitor.py` - 執行狀態監控器

**功能**: 追蹤和監控任務執行狀態

**監控維度**:
```python
ExecutionStatusMonitor
├── Task Status (任務狀態)
│   ├── Pending
│   ├── Running
│   ├── Completed
│   ├── Failed
│   └── Cancelled
│
├── Worker Health (Worker 健康度)
│   ├── Heartbeat
│   ├── Resource Usage
│   └── Performance Metrics
│
└── System Metrics (系統指標)
    ├── Throughput
    ├── Latency
    └── Error Rate
```

**使用範例**:
```python
from task_planning.executor import ExecutionStatusMonitor, ExecutionContext

monitor = ExecutionStatusMonitor()

# 記錄任務開始
monitor.record_task_start(
    task_id="task_001",
    worker_id="worker_001",
    context=ExecutionContext(
        task_type="sql_injection_test",
        target="https://example.com"
    )
)

# 更新任務進度
monitor.update_progress(
    task_id="task_001",
    progress=50,  # 0-100
    message="正在測試 SQL 注入點..."
)

# 記錄任務完成
monitor.record_task_completion(
    task_id="task_001",
    success=True,
    result={"vulnerabilities_found": 3}
)

# 獲取任務狀態
status = monitor.get_task_status("task_001")
print(f"狀態: {status.state}")
print(f"進度: {status.progress}%")
print(f"執行時間: {status.execution_time}s")

# Worker 心跳
monitor.record_worker_heartbeat(
    worker_id="worker_001",
    metrics={
        "cpu_usage": 45.2,
        "memory_usage": 1024,
        "active_tasks": 3
    }
)

# 獲取系統監控數據
system_metrics = monitor.get_system_metrics()
print(f"總任務數: {system_metrics.total_tasks}")
print(f"完成率: {system_metrics.completion_rate}%")
print(f"平均執行時間: {system_metrics.avg_execution_time}s")
print(f"錯誤率: {system_metrics.error_rate}%")
```

**監控告警**:
```python
# 設定告警規則
monitor.set_alert_rule(
    name="high_error_rate",
    condition="error_rate > 10",
    action=lambda: send_notification("錯誤率過高！")
)

monitor.set_alert_rule(
    name="worker_timeout",
    condition="worker_heartbeat_missing > 300",
    action=lambda worker_id: restart_worker(worker_id)
)
```

**執行上下文**:
```python
@dataclass
class ExecutionContext:
    task_id: str
    task_type: str
    worker_id: str
    start_time: datetime
    target: str
    params: dict[str, Any]
    parent_task_id: str | None = None
    retry_count: int = 0
```

---

### 4. `plan_executor.py` - 計畫執行器

**功能**: 執行完整的多任務執行計劃

**執行模式**:
- **順序執行** - 依序執行所有任務
- **並行執行** - 同時執行無依賴任務
- **流式執行** - 邊執行邊處理結果
- **自適應執行** - 根據結果動態調整

**使用範例**:
```python
from task_planning.executor import PlanExecutor

plan_executor = PlanExecutor()

# 執行計劃
result = await plan_executor.execute_plan(
    plan=execution_plan,
    mode="parallel",  # sequential, parallel, streaming
    config={
        "max_parallel": 5,
        "timeout": 3600,
        "stop_on_error": False
    }
)

# 獲取執行摘要
print(f"總任務: {result.total_tasks}")
print(f"成功: {result.successful_tasks}")
print(f"失敗: {result.failed_tasks}")
print(f"執行時間: {result.execution_time}s")

# 獲取詳細結果
for task_result in result.task_results:
    print(f"任務 {task_result.task_id}: {task_result.status}")
```

**自適應執行**:
```python
# 根據執行結果動態調整
async def adaptive_execution(plan):
    executor = PlanExecutor()
    
    for stage in plan.stages:
        # 執行當前階段
        stage_result = await executor.execute_stage(stage)
        
        # 根據結果調整後續計劃
        if stage_result.success_rate < 0.5:
            # 降低並行度
            plan.max_parallel = max(1, plan.max_parallel // 2)
        elif stage_result.success_rate > 0.9:
            # 提高並行度
            plan.max_parallel = min(10, plan.max_parallel * 2)
        
        # 如果發現高危漏洞，調整優先級
        if stage_result.critical_findings:
            plan.reorder_by_priority()
    
    return executor.get_summary()
```

---

## 🚀 完整使用流程

### 任務執行流程
```python
from task_planning.executor import (
    TaskQueueManager,
    TaskExecutor,
    ExecutionStatusMonitor
)

# 1. 初始化組件
queue_manager = TaskQueueManager()
executor = TaskExecutor()
monitor = ExecutionStatusMonitor()

# 2. 添加任務到佇列
queue_manager.enqueue_task(
    topic="vulnerability_scan",
    task_payload={
        "task_id": "task_001",
        "type": "sql_injection_test",
        "target": "https://example.com"
    }
)

# 3. Worker 從佇列獲取任務
task = await queue_manager.dequeue_task(
    topic="vulnerability_scan",
    worker_id="worker_001"
)

# 4. 記錄開始執行
monitor.record_task_start(
    task_id=task["task_id"],
    worker_id="worker_001"
)

# 5. 執行任務
try:
    result = await executor.execute(task)
    
    # 6. 記錄完成
    monitor.record_task_completion(
        task_id=task["task_id"],
        success=result.success,
        result=result.output
    )
    
    # 7. 確認任務
    queue_manager.acknowledge_task(
        task_id=task["task_id"],
        success=result.success
    )
    
except Exception as e:
    # 記錄失敗
    monitor.record_task_failure(
        task_id=task["task_id"],
        error=str(e)
    )
    
    # 重試任務
    queue_manager.retry_task(
        task_id=task["task_id"],
        delay_seconds=60
    )
```

### 計畫執行流程
```python
from task_planning.executor import PlanExecutor
from task_planning.planner import AttackOrchestrator

# 1. 創建執行計劃
orchestrator = AttackOrchestrator()
plan = orchestrator.create_execution_plan(ast_input)

# 2. 執行計劃
plan_executor = PlanExecutor(
    queue_manager=queue_manager,
    task_executor=executor,
    status_monitor=monitor
)

result = await plan_executor.execute_plan(
    plan=plan,
    mode="parallel",
    config={"max_parallel": 5}
)

# 3. 分析結果
print(f"執行摘要:")
print(f"  總任務: {result.total_tasks}")
print(f"  成功: {result.successful_tasks}")
print(f"  失敗: {result.failed_tasks}")
print(f"  成功率: {result.success_rate}%")
print(f"  執行時間: {result.execution_time}s")
```

---

## 📊 性能指標

| 指標 | 數值 | 備註 |
|------|------|------|
| 任務執行 | 10-100 ms | 依任務複雜度 |
| 佇列吞吐 | 1000+ tasks/s | 單實例 |
| 監控開銷 | < 5% CPU | 運行時 |
| 並行度 | 100+ tasks | 同時執行 |
| 重試延遲 | 可配置 | 預設 60s |

---

**最後更新**: 2025-11-16  
**維護者**: AIVA Development Team
