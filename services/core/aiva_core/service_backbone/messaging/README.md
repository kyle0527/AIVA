# Messaging 消息通訊模組

> **路徑**: `services/core/aiva_core/service_backbone/messaging`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-07

## 概述

統一的消息代理系統，管理 RabbitMQ 連接和消息路由，支援任務分發和結果收集。

## 📄 檔案詳細資訊 (Files Details)

### `message_broker.py`
**說明**: Message Broker - 消息代理

**類別 (Classes)**:
- `MessageBroker` - 消息代理
- `RPCClient` - RPC 客戶端
- `EventPriority` - 事件優先級 (整合自 AI 模組)
- `AIVAEvent` - AIVA 統一事件格式 (整合自 AI 模組)
- `EventSubscription` - 事件訂閱 (整合自 AI 模組)
- `EnhancedMessageBroker` - 增強的消息代理 (整合事件驅動系統)
**函式 (Functions)**:
- `get_enhanced_message_broker()` - 獲取全域增強消息代理實例

### `result_collector.py`
**說明**: Result Collector - 結果收集器

**類別 (Classes)**:
- `ResultCollector` - 結果收集器

### `task_dispatcher.py`
**說明**: Task Dispatcher - 任務派發器

**類別 (Classes)**:
- `TaskDispatcher` - 任務派發器

