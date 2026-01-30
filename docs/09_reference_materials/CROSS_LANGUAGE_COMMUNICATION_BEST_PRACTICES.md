# 跨語言模組通信最佳實踐建議

## 📑 目錄

- [來源參考](#來源參考)
- [1. 通信類型分類](#1-通信類型分類)
  - [軸一：同步 vs 異步](#軸一同步-vs-異步)
  - [軸二：單一接收者 vs 多接收者](#軸二單一接收者-vs-多接收者)
- [2. 跨語言通信方案比較](#2-跨語言通信方案比較)
  - [方案一：消息隊列 (RabbitMQ/AMQP) ✅ 推薦](#方案一消息隊列-rabbitmqamqp-推薦)
  - [方案二：CLI/subprocess ✅ 推薦](#方案二clisubprocess-推薦)
  - [方案三：gRPC + Protocol Buffers](#方案三grpc-protocol-buffers)
  - [方案四：REST API (HTTP)](#方案四rest-api-http)
- [3. 關鍵最佳實踐](#3-關鍵最佳實踐)
  - [3.1 避免同步鏈式調用 (反模式)](#31-避免同步鏈式調用-反模式)
  - [3.2 RPC 注意事項 (來自 RabbitMQ 官方)](#32-rpc-注意事項-來自-rabbitmq-官方)
  - [3.3 使用 Correlation ID](#33-使用-correlation-id)
  - [3.4 異步請求-回覆模式](#34-異步請求-回覆模式)
- [4. AIVA 系統建議](#4-aiva-系統建議)
  - [4.1 推薦架構](#41-推薦架構)
  - [4.2 通信方式選擇矩陣](#42-通信方式選擇矩陣)
  - [4.3 JSON 作為統一數據格式](#43-json-作為統一數據格式)
- [5. 總結](#5-總結)
  - [✅ 對 AIVA 的建議](#-對-aiva-的建議)
  - [📊 與現有實現的對比](#-與現有實現的對比)
  - [🔧 可選改進](#-可選改進)

---


**基於網路資源整理 | 日期**: 2026-01-21 (Updated)

## 來源參考

- Microsoft: [Communication in Microservice Architecture](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/architect-microservice-container-applications/communication-in-microservice-architecture)
- Microsoft: [Async Request-Reply Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/async-request-reply)
- Microservices.io: [Messaging Pattern](https://microservices.io/patterns/communication-style/messaging.html)
- RabbitMQ: [RPC Tutorial](https://www.rabbitmq.com/tutorials/tutorial-six-python.html)
- gRPC: [Introduction](https://grpc.io/docs/what-is-grpc/introduction/)
- AWS: [Microservices](https://aws.amazon.com/microservices/)

---

## 1. 通信類型分類

根據 Microsoft 的微服務架構指南，通信可分為兩個軸：

### 軸一：同步 vs 異步

| 類型 | 協議 | 特性 | 適用場景 |
|------|------|------|----------|
| **同步** | HTTP/HTTPS | 客戶端等待響應 | 即時查詢、緊急操作 |
| **異步** | AMQP/RabbitMQ | 不等待響應 | 事件通知、長任務 |

### 軸二：單一接收者 vs 多接收者

| 類型 | 模式 | 範例 |
|------|------|------|
| **單一接收者** | Command Pattern | RPC 調用 |
| **多接收者** | Publish/Subscribe | 事件廣播 |

---

## 2. 跨語言通信方案比較

### 方案一：消息隊列 (RabbitMQ/AMQP) ✅ 推薦

```
┌─────────────────────────────────────────────────────────────┐
│                    MessageBroker (RabbitMQ)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Python ──┐                           ┌── Rust              │
│            │                         │                       │
│  Go ───────┼────► Exchange ────►─────┼── TypeScript         │
│            │    (aiva.tasks)         │                       │
│  C# ──────┘                           └── Java               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**優點**:
- ✅ 語言無關 - 任何語言都可使用 AMQP 協議
- ✅ 鬆散耦合 - 發送者不需要知道接收者
- ✅ 彈性擴展 - 可動態增減消費者
- ✅ 可靠性 - 消息持久化、重試機制
- ✅ 異步處理 - 不阻塞發送方

**缺點**:
- ❌ 需要額外的基礎設施 (RabbitMQ server)
- ❌ 增加系統複雜度
- ❌ 不適合需要即時返回值的場景

**適用於 AIVA**:
- cognitive_core → task_planning (決策通知)
- internal_exploration → cognitive_core/learning_system (分析結果)
- 任何事件通知場景

### 方案二：CLI/subprocess ✅ 推薦

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Python 調用:                                                │
│  subprocess.run(["rust_binary", "--params", json_args])     │
│  subprocess.run(["go", "run", "tool.go", json_args])        │
│  subprocess.run(["npx", "ts-node", "script.ts", args])      │
│                                                              │
│  所有語言都支持 stdin/stdout/stderr 標準輸入輸出             │
│  使用 JSON 作為統一數據交換格式                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**優點**:
- ✅ 無需額外依賴 - 所有語言都支持進程調用
- ✅ 同步執行 - 可獲得返回值
- ✅ 隔離性 - 每個工具獨立進程
- ✅ 跨語言 - 任何可執行程序都可調用

**缺點**:
- ❌ 進程啟動開銷
- ❌ 不適合高頻調用
- ❌ 需要處理進程管理

**適用於 AIVA**:
- task_planning → core_capabilities (執行攻擊/掃描)
- internal_exploration → rust_tools (調用 Rust 分析工具)
- 任何需要返回值的場景

### 方案三：gRPC + Protocol Buffers

```protobuf
// 定義服務接口
service TaskPlanning {
    rpc GeneratePlan (PlanRequest) returns (PlanResponse);
    rpc ExecuteStep (StepRequest) returns (StepResponse);
}

message PlanRequest {
    string objective = 1;
    map<string, string> context = 2;
}
```

**優點**:
- ✅ 高效二進制協議
- ✅ 強類型接口定義
- ✅ 支援多語言 (Python, Go, Rust, C++, Java, C#, Node.js)
- ✅ 雙向流支持

**缺點**:
- ❌ 學習曲線較陡
- ❌ 需要定義 .proto 文件
- ❌ 需要代碼生成步驟

**適用場景**:
- 高性能服務間調用
- 需要強類型的大型系統

### 方案四：REST API (HTTP)

**優點**:
- ✅ 標準協議
- ✅ 易於調試
- ✅ 廣泛支持

**缺點**:
- ❌ 同步阻塞
- ❌ 相較 gRPC 效率較低
- ❌ 文本格式 (JSON) 開銷

---

## 3. 關鍵最佳實踐

### 3.1 避免同步鏈式調用 (反模式)

**❌ 不推薦**:
```
Client → Service A → Service B → Service C
         (等待)      (等待)      (等待)
```

**✅ 推薦**:
```
Client → Service A (立即返回)
              ↓
         Message Queue
              ↓
         Service B, C (異步處理)
```

> "The more you add synchronous dependencies between microservices, the worse the overall response time gets for the client apps." - Microsoft

### 3.2 RPC 注意事項 (來自 RabbitMQ 官方)

```
📌 重要建議：
1. 清楚區分本地調用和遠程調用
2. 文檔化系統依賴關係
3. 處理錯誤情況（服務不可用時的重試策略）
4. 如有疑問，優先選擇異步管道而非 RPC

"When in doubt avoid RPC. If you can, you should use an asynchronous 
 pipeline - instead of RPC-like blocking, results are asynchronously 
 pushed to a next computation stage."
```

### 3.3 使用 Correlation ID

```python
message = {
    "action": "execute_plan",
    "source_module": "cognitive_core",
    "timestamp": "2026-01-01T12:00:00",
    "correlation_id": "uuid-for-tracking",  # 用於追蹤請求
    "reply_to": "callback_queue_name",      # 回調隊列
    "payload": {...}
}
```

### 3.4 異步請求-回覆模式

```
┌─────────────────────────────────────────────────────────────┐
│              Async Request-Reply Pattern                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Client → Server: POST /tasks                             │
│     Response: 202 Accepted + Location: /status/{id}         │
│                                                              │
│  2. Client → Server: GET /status/{id}                       │
│     Response: 200 { status: "processing" }                  │
│                                                              │
│  3. Client → Server: GET /status/{id}                       │
│     Response: 302 Redirect → /results/{id}                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. AIVA 系統建議

### 4.1 推薦架構

```
┌─────────────────────────────────────────────────────────────┐
│                      AIVA 通信架構                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │ cognitive   │     │    task     │     │    core     │   │
│  │   _core     │     │  _planning  │     │ _capabilities│   │
│  │  (Python)   │     │  (Python)   │     │ (Python/CLI)│   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘   │
│         │                   │                    │          │
│         │    MessageBroker (RabbitMQ)            │          │
│         ├───────────────────┴────────────────────┤          │
│         │         異步消息 (事件驅動)             │          │
│         │                                        │          │
│         │    CLI/subprocess (同步調用)           │          │
│         ├────────────────────────────────────────┤          │
│         │                                        │          │
│  ┌──────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐   │
│  │  internal   │     │  cognitive  │     │  service    │   │
│  │ _exploration│     │  _core/     │     │  _backbone  │   │
│  │(Python/Rust/│     │  learning_  │     │  (Python)   │   │
│  │  Go/TS)     │     │  system     │     │             │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 通信方式選擇矩陣

| 場景 | 推薦方式 | 理由 |
|------|----------|------|
| 事件通知 | MessageBroker | 異步、可廣播 |
| 需要返回值 | CLI subprocess | 同步、可獲結果 |
| 高頻內部調用 | gRPC (可選) | 高效二進制 |
| 跨語言調用 | CLI + JSON | 通用、簡單 |
| 長時間任務 | MessageBroker | 異步、可追蹤 |
| 緊急決策 | CLI subprocess | 立即執行返回 |

### 4.3 JSON 作為統一數據格式

```python
# 所有跨模組通信使用 JSON
message = {
    "action": "execute",
    "source_module": "task_planning",
    "target_module": "core_capabilities",
    "timestamp": "2026-01-01T12:00:00Z",
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "payload": {
        "capability": "port_scan",
        "target": "192.168.1.1",
        "options": {"ports": "1-1000"}
    }
}
```

---

## 5. 總結

### ✅ 對 AIVA 的建議

1. **維持現有 MessageBroker + CLI 雙軌制** - 已經是業界推薦的模式
2. **異步優先** - 除非必須同步，否則使用消息隊列
3. **使用 JSON** - 作為跨語言數據交換格式
4. **添加 Correlation ID** - 用於請求追蹤和錯誤排查
5. **處理超時和重試** - CLI 調用設置 timeout，消息隊列使用 Dead Letter Queue

### 📊 與現有實現的對比

| 功能 | 業界最佳實踐 | AIVA 現有實現 | 狀態 |
|------|-------------|---------------|------|
| 異步消息 | RabbitMQ/Kafka | MessageBroker (RabbitMQ) | ✅ 符合 |
| 同步調用 | gRPC/REST/CLI | subprocess + CLI | ✅ 符合 |
| 跨語言支持 | Protocol Buffers/JSON | JSON | ✅ 符合 |
| 請求追蹤 | Correlation ID | 已實現 | ✅ 符合 |
| 錯誤處理 | Retry + DLQ | 部分實現 | ⚠️ 可加強 |

### 🔧 可選改進

1. **考慮 gRPC** - 如果需要更高效的內部通信
2. **添加 Circuit Breaker** - 防止級聯故障
3. **實現 Saga Pattern** - 用於分布式事務
4. **增加監控** - 使用 OpenTelemetry 追蹤跨模組調用
