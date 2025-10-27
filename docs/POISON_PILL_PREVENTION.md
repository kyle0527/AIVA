# AIVA 平台重試策略配置

## 概述

為了解決 poison pill 消息問題，AIVA 平台實施了統一的重試策略和死信隊列機制。本文檔描述了配置細節和最佳實踐。

## 重試策略

### 統一配置

- **最大重試次數**: 3 次
- **退避策略**: 指數退避
- **基礎延遲**: 1 秒
- **最大延遲**: 30 秒
- **死信隊列**: `aiva.dead_letter.failed`

### 實施語言

#### Python
- 文件: `services/aiva_common/messaging/retry_handler.py`
- 類: `RetryHandler`
- 策略: `STANDARD_RETRY_POLICY`

#### Go
- 文件: `services/features/common/go/aiva_common_go/mq/client.go`
- 方法: `shouldRetryMessage()`
- 集成: 所有 Go workers 自動使用

#### Rust
- 文件: `services/scan/info_gatherer_rust/src/main.rs`
- 文件: `services/features/function_sast_rust/src/worker.rs`
- 函數: `should_retry_message()`

## 死信隊列配置

### RabbitMQ 策略

使用 RabbitMQ policies 而非 x-arguments 來配置死信交換機：

- **任務隊列策略**: `aiva-tasks-dlx-policy`
  - 適用隊列: `tasks.*`
  - 死信交換機: `aiva.dead_letter`
  - 路由鍵: `failed`

- **發現隊列策略**: `aiva-findings-dlx-policy`
  - 適用隊列: `findings.*`
  - 死信交換機: `aiva.dead_letter`
  - 路由鍵: `failed`

- **結果隊列策略**: `aiva-results-dlx-policy`
  - 適用隊列: `*results`
  - 死信交換機: `aiva.dead_letter`
  - 路由鍵: `failed`

### 配置腳本

- **Linux/macOS**: `scripts/setup_dead_letter_queues.sh`
- **Windows**: `scripts/setup_dead_letter_queues.ps1`

## 消息頭部追蹤

### 自定義頭部

- `x-aiva-retry-count`: 當前重試次數
- `x-aiva-first-attempt`: 首次處理時間
- `x-aiva-last-attempt`: 最後重試時間
- `x-aiva-error-history`: 錯誤歷史 (JSON 格式)
- `x-aiva-original-routing-key`: 原始路由鍵

### RabbitMQ 標準頭部

- `x-death`: RabbitMQ 死信歷史 (已清除，使用自定義追蹤)
- `x-first-death-*`: 首次死信信息
- `x-last-death-*`: 最後死信信息

## 監控和告警

### 死信隊列監控

```bash
# 查看死信隊列消息數量
rabbitmqctl list_queues name messages | grep dead_letter

# 查看死信隊列詳細信息
rabbitmqctl list_queues name messages consumers memory --vhost / | grep dead_letter
```

### 日誌關鍵字

- `"達到最大重試次數"`: 消息進入死信隊列
- `"消息重試"`: 消息正在重試
- `"poison pill"`: 可疑的問題消息

## 故障排除

### 常見問題

1. **消息無限循環**
   - 檢查 `requeue: true` 配置
   - 確認死信隊列策略已應用
   - 驗證重試邏輯實施

2. **死信隊列堆積**
   - 檢查消息處理邏輯
   - 分析錯誤模式
   - 考慮增加處理資源

3. **重試次數過多/過少**
   - 調整 `MAX_RETRY_ATTEMPTS` 常數
   - 根據業務需求自訂策略

### 診斷工具

```bash
# 檢查隊列策略
rabbitmqctl list_policies --vhost /

# 查看交換機
rabbitmqctl list_exchanges --vhost / | grep dead_letter

# 監控隊列狀態
watch -n 5 'rabbitmqctl list_queues name messages consumers'
```

## 最佳實踐

1. **使用策略而非 x-arguments**: 允許動態更新而不需要重新部署
2. **記錄錯誤歷史**: 幫助診斷根本原因
3. **設置合理的 TTL**: 防止隊列無限增長
4. **監控死信隊列**: 及時發現問題消息
5. **定期清理**: 移除已解決的死信消息

## 相關文檔

- [RabbitMQ Dead Letter Exchanges](https://www.rabbitmq.com/docs/dlx)
- [AIVA 架構文檔](../docs/ARCHITECTURE.md)
- [12-Factor App 原則](https://12factor.net/)
- [消息重試模式](https://www.enterpriseintegrationpatterns.com/patterns/messaging/DeadLetterChannel.html)