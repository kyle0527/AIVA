# Messaging - 消息中間件

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [message_broker.py](#message_brokerpy-700-行-)
  - [task_dispatcher.py](#task_dispatcherpy-422-行-)
  - [result_collector.py](#result_collectorpy-312-行)
- [📨 消息模式](#-消息模式)
- [🔄 消息流程](#-消息流程)
- [⚡ 性能優化](#-性能優化)
- [📚 相關模組](#-相關模組)
- [🔧 配置最佳實踐](#-配置最佳實踐)

---

## 📋 概述

**定位**: 異步消息傳遞和事件驅動架構  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (1,434 行)

## 📂 文件結構

```
messaging/
├── message_broker.py (700 行) ⭐⭐ - 消息代理
├── task_dispatcher.py (422 行) ⭐ - 任務調度器
├── result_collector.py (312 行) - 結果收集器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### message_broker.py (700 行) ⭐⭐

**職責**: 核心消息代理,支援多種消息模式

**主要類/函數**:
- `MessageBroker` - 消息代理主類
- `publish(topic, message)` - 發布消息
- `subscribe(topic, callback)` - 訂閱主題
- `request_response(queue, message)` - 請求-響應模式

**支援的消息模式**:
- ✅ **發布/訂閱** (Pub/Sub): 一對多廣播
- ✅ **點對點** (P2P): 隊列消費
- ✅ **請求/響應** (Req/Rep): 同步調用
- ✅ **推送/拉取** (Push/Pull): 任務分發

**使用範例**:
```python
from aiva_core.service_backbone.messaging import MessageBroker

broker = MessageBroker()

# 發布/訂閱模式
broker.subscribe("scan_completed", on_scan_completed)
broker.publish("scan_completed", {
    "scan_id": "123",
    "status": "success",
    "findings": [...]
})

# 請求/響應模式
response = await broker.request_response(
    queue="analysis_queue",
    message={"type": "analyze", "data": scan_results}
)
```

**支援的後端**:
- RabbitMQ
- Redis Pub/Sub
- 內存隊列 (開發用)

---

### task_dispatcher.py (422 行) ⭐

**職責**: 任務調度和分發器

**主要功能**:
- 任務優先級管理
- 負載均衡分發
- 死信隊列處理
- 任務重試機制

**調度策略**:
| 策略 | 描述 | 使用場景 |
|------|------|---------|
| **輪詢** (Round Robin) | 平均分發到所有工作者 | 任務執行時間相近 |
| **最小連接** | 分發到最空閒的工作者 | 任務執行時間差異大 |
| **優先級** | 高優先級任務優先執行 | 緊急任務處理 |
| **親和性** | 相同類型任務分發到同一工作者 | 利用快取和上下文 |

**使用範例**:
```python
from aiva_core.service_backbone.messaging import TaskDispatcher

dispatcher = TaskDispatcher()

# 分發高優先級任務
await dispatcher.dispatch(
    task={
        "type": "sql_injection_scan",
        "target": "critical_system.com"
    },
    priority="high",
    retry_policy={"max_attempts": 5}
)

# 獲取任務狀態
status = dispatcher.get_task_status(task_id)
```

---

### result_collector.py (312 行)

**職責**: 分布式任務結果收集和聚合

**主要功能**:
- 異步結果收集
- 結果聚合和合併
- 超時處理
- 部分結果返回

**使用範例**:
```python
from aiva_core.service_backbone.messaging import ResultCollector

collector = ResultCollector()

# 等待多個任務完成
results = await collector.collect_all(
    task_ids=["task1", "task2", "task3"],
    timeout=300
)

# 或使用回調方式
collector.on_result("task1", lambda result: process_result(result))
```

## 📨 消息流架構

### 典型工作流

```
1. 任務創建
   ↓
2. TaskDispatcher 分發到隊列
   ↓
3. MessageBroker 傳遞消息
   ↓
4. Worker 消費任務
   ↓
5. Worker 發布結果
   ↓
6. ResultCollector 收集結果
   ↓
7. 返回給調用者
```

### 事件驅動流程

```
掃描服務
  ↓ publish("scan.started")
MessageBroker
  ↓ notify subscribers
  ├→ Logging Service (記錄事件)
  ├→ Monitoring Service (更新指標)
  └→ UI Service (更新界面)
```

## 🔔 事件類型

### 系統事件

```python
# 任務生命週期事件
"task.created"      # 任務創建
"task.dispatched"   # 任務已分發
"task.started"      # 任務開始執行
"task.completed"    # 任務完成
"task.failed"       # 任務失敗

# 掃描事件
"scan.initiated"    # 掃描啟動
"scan.progress"     # 掃描進度
"scan.completed"    # 掃描完成
"scan.error"        # 掃描錯誤

# 系統事件
"system.health_check"  # 健康檢查
"system.alert"         # 系統告警
```

## 💡 最佳實踐

### 1. 消息設計

```python
# ✅ 良好的消息結構
{
    "message_id": "uuid",
    "timestamp": "2025-11-16T10:00:00Z",
    "type": "scan.completed",
    "payload": {
        "scan_id": "123",
        "status": "success",
        "findings": [...]
    },
    "metadata": {
        "source": "scanner_service",
        "correlation_id": "request_456"
    }
}
```

### 2. 錯誤處理

```python
# 配置重試策略
dispatcher.dispatch(
    task=task,
    retry_policy={
        "max_attempts": 3,
        "backoff_factor": 2,  # 2s, 4s, 8s
        "exceptions": [ConnectionError, TimeoutError]
    }
)
```

### 3. 消息持久化

```python
# 啟用消息持久化
broker = MessageBroker(
    backend="rabbitmq",
    persistent=True,
    durable_queues=True
)
```

## 📚 相關模組

- [coordination](../coordination/README.md) - 服務協調
- [task_planning](../../task_planning/README.md) - 任務規劃
- [external_learning](../../external_learning/README.md) - 事件學習

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md) 的修復規範。

```python
# ✅ 正確：使用標準消息類型和主題
from aiva_common import AivaMessage, MessageHeader, ModuleName, Topic

# 創建消息
message = AivaMessage(
    header=MessageHeader(
        source=ModuleName.MESSAGING,
        target=ModuleName.SCANNING
    ),
    payload={"action": "start_scan"}
)

# 發布到標準主題
broker.publish(Topic.TASK_EVENTS, message)

# ❌ 禁止：自定義主題名稱
broker.publish("my_custom_topic", message)  # 使用 Topic 枚舉

# ❌ 禁止：自定義消息格式
class CustomMessage:
    def __init__(self, content):
        self.content = content  # 使用 AivaMessage
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md)

---

## 🔧 配置示例

```python
# MessageBroker 配置
broker_config = {
    "backend": "rabbitmq",
    "host": "localhost",
    "port": 5672,
    "username": "aiva",
    "password": "***",
    "exchange": "aiva_events",
    "prefetch_count": 10
}

broker = MessageBroker(config=broker_config)
```

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
