# 05｜POSTEX 介面與資料契約對照
**日期**：2025-11-06  

## 1. AMQP 主題
- 任務：`Topic.TASK_FUNCTION_POSTEX`
- 結果：`Topic.FINDING_DETECTED`
- 狀態：`Topic.STATUS_TASK_UPDATE`

## 2. 請求（PostExTaskPayload 範例）
```json
{
  "task_id":"task-321",
  "scan_id":"scan-555",
  "test_type":"privilege_escalation",
  "target":"10.0.0.0/24",
  "safe_mode":true,
  "authorization_token": null
}
```

## 3. 回應（FindingPayload）
- `vulnerability.name`：PRIVILEGE_ESCALATION / ACCESS_CONTROL / WEAK_AUTH  
- `evidence.proof`：模擬發現訊息。  
- `impact`：營運風險陳述。

## 4. 安全護欄
- 未提供 token → 一律 safe-mode。
- 所有操作寫入審計。

## 5. 失敗訊息
- 發佈到 `Topic.STATUS_TASK_UPDATE`：`status=FAILED` + `error`。
