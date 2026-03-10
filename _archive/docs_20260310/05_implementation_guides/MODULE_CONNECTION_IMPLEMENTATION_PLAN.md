# 模組連接打通實施計劃

## 📑 目錄

- [目標](#目標)
- [通信機制](#通信機制)
  - [1. MessageBroker (RabbitMQ)](#1-messagebroker-rabbitmq)
  - [2. CLI 命令 (subprocess)](#2-cli-命令-subprocess)
- [已實施文件](#已實施文件)
- [Dispatcher 使用方式](#dispatcher-使用方式)
  - [cognitive_core/dispatcher.py](#cognitive_coredispatcherpy)
  - [internal_exploration/dispatcher.py](#internal_explorationdispatcherpy)
  - [task_planning/dispatcher.py](#task_planningdispatcherpy)
- [預期效果](#預期效果)
- [通信架構說明](#通信架構說明)
- [新增連接](#新增連接)
- [通信方式選擇指南](#通信方式選擇指南)

---


**日期**: 2026-01-01
**狀態**: ✅ 已實施

## 目標

讓 cognitive_core、internal_exploration、task_planning 三個單向模組能夠**主動發送訊息**到其他模組。

## 通信機制

### 1. MessageBroker (RabbitMQ)

- **用途**: 異步消息、事件驅動、跨語言
- **交換機**: `aiva.tasks`, `aiva.results`, `aiva.events`, `aiva.feedback`
- **適合場景**: 事件通知、長時間任務、廣播消息

### 2. CLI 命令 (subprocess)

- **用途**: 同步執行、跨語言調用
- **支援**: Python, Rust, Go, TypeScript, Docker
- **適合場景**: 需要返回值、緊急決策、命令行工具

## 已實施文件

```
services/core/aiva_core/
├── cognitive_core/
│   └── dispatcher.py          ✅ 已創建
├── internal_exploration/
│   └── dispatcher.py          ✅ 已創建
├── task_planning/
│   └── dispatcher.py          ✅ 已創建
└── service_backbone/
    └── dispatcher_base.py     ✅ 已創建 (基礎類)
```

## Dispatcher 使用方式

### cognitive_core/dispatcher.py

```python
from services.core.aiva_core.cognitive_core.dispatcher import CognitiveDispatcher

dispatcher = CognitiveDispatcher()

# 異步消息方式
await dispatcher.request_plan("執行安全掃描", context)
await dispatcher.execute_capability("scan_ports", {"target": "192.168.1.1"})
await dispatcher.trigger_learning({"data": training_data})

# CLI 同步方式
result = dispatcher.call_task_planning_sync("generate", objective="...")
result = dispatcher.call_core_capabilities_sync("scan", target="...")
```

### internal_exploration/dispatcher.py

```python
from services.core.aiva_core.internal_exploration.dispatcher import ExplorationDispatcher

dispatcher = ExplorationDispatcher()

# 異步消息方式
await dispatcher.notify_analysis_complete(analysis_result)
await dispatcher.request_decision(issue)
await dispatcher.broadcast_discovery(discovery, "vulnerability")

# CLI 同步方式（支援跨語言）
result = dispatcher.call_rust_tool("analyzer", param="...")
result = dispatcher.call_go_tool("scanner", target="...")
result = dispatcher.trigger_training_sync(training_data)
```

### task_planning/dispatcher.py

```python
from services.core.aiva_core.task_planning.dispatcher import PlanningDispatcher

dispatcher = PlanningDispatcher()

# 異步消息方式
await dispatcher.execute_plan_step(step, capability_id)
await dispatcher.confirm_decision(plan, "是否繼續?")
await dispatcher.notify_plan_status(plan_id, "running")

# CLI 同步方式
result = dispatcher.execute_attack_sync("sql_injection", "target.com")
result = dispatcher.execute_scan_sync("port", "192.168.1.1")
```

## 預期效果

| 指標 | 打通前 | 打通後 |
|------|--------|--------|
| 單向模組 | 3 | 0 |
| cognitive_core 出站 | 0 | 4+ |
| internal_exploration 出站 | 0 | 3+ |
| task_planning 出站 | 0 | 4+ |
| 連接密度 | 30% | 50%+ |

## 通信架構說明

AIVA v2.0 採用**雙軌通信**設計，靈活應對不同場景：

1. **MessageBroker (RabbitMQ)** - 異步事件驅動
   - 事件通知、廣播消息
   - 跨模組狀態同步
   - 長時間任務協調

2. **CLI Subprocess** - 同步命令執行
   - 跨語言工具調用 (Python/Rust/Go/TypeScript)
   - 需要返回值的操作
   - 緊急決策場景

兩者互補，共同構成完整的模組間通信機制。

## 新增連接

| 來源模組 | 目標模組 | 方法 | 用途 |
|----------|----------|------|------|
| cognitive_core | task_planning | request_plan() | 決策後生成計劃 |
| cognitive_core | core_capabilities | execute_capability() | 執行能力 |
| cognitive_core | external_learning | trigger_learning() | 觸發訓練 |
| internal_exploration | external_learning | notify_analysis_complete() | 分析結果訓練 |
| internal_exploration | cognitive_core | request_decision() | 請求決策 |
| task_planning | core_capabilities | execute_plan_step() | 執行計劃步驟 |
| task_planning | cognitive_core | confirm_decision() | 確認決策 |

## 通信方式選擇指南

| 場景 | 推薦方式 | 原因 |
|------|----------|------|
| 事件通知 | MessageBroker | 異步、不阻塞、可廣播 |
| 需要返回值 | CLI subprocess | 同步等待結果 |
| 跨語言調用 | CLI subprocess | Rust/Go/TS 都支持 |
| 長時間任務 | MessageBroker | 異步處理、可監控 |
| 緊急決策 | CLI subprocess | 立即執行返回 |
