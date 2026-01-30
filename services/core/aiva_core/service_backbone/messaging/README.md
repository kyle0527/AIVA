# Messaging 消息通訊模組

> **路徑**: `services/core/aiva_core/service_backbone/messaging`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-07

## 概述

統一的消息代理系統，管理 RabbitMQ 連接和消息路由，支援任務分發和結果收集。

## 核心組件

### message_broker.py
- `MessageBroker` - 消息代理
  - RabbitMQ 連接管理
  - 消息發布/訂閱
  - Exchange 和 Queue 管理
  - Consumer 任務管理

- `RPCClient` - RPC 客戶端，支援同步調用模式
- `EventPriority` - 事件優先級枚舉
- `AIVAEvent` - AIVA 事件結構
- `EventSubscription` - 事件訂閱結構
- `EnhancedMessageBroker` - 增強版消息代理（繼承 MessageBroker）
  - 支援事件優先級
  - 支援批量發送
  - 支援消息追蹤

### task_dispatcher.py
- `TaskDispatcher` - 任務分發器
  - 將任務分發到對應的處理隊列
  - 支援負載均衡
  - 任務重試機制

### result_collector.py
- `ResultCollector` - 結果收集器
  - 收集任務執行結果
  - 結果聚合和統計
  - 超時處理

### __init__.py
- 模組初始化和導出

## 消息流程

```
TaskDispatcher（分發任務）
        ↓
MessageBroker（發送到 RabbitMQ）
        ↓
Features/Integration 模組（處理）
        ↓
MessageBroker（接收結果）
        ↓
ResultCollector（收集結果）
```

## 依賴關係

- `aio_pika` - RabbitMQ 異步客戶端
- `aiva_common.config` - get_settings（RabbitMQ 配置）
- `aiva_common.schemas` - AivaMessage
- `aiva_common.enums.modules` - ModuleName
- `aiva_common.error_handling` - 統一錯誤處理
