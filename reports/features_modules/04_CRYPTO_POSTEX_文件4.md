# 04｜CRYPTO 介面與資料契約對照
**日期**：2025-11-06  

## 1. AMQP 主題
- 任務：`Topic.TASK_FUNCTION_CRYPTO`
- 結果：`Topic.FINDING_DETECTED`
- 狀態：`Topic.STATUS_TASK_UPDATE`

## 2. 請求（FunctionTaskPayload 範例）
```json
{
  "task_id":"task-123",
  "scan_id":"scan-999",
  "target":{"url":"./sample_code.py"},
  "options":{"mode":"static"}
}
```

## 3. 回應（FindingPayload 主要欄位）
- `vulnerability.name`：INFO_LEAK / WEAK_AUTH / …  
- `evidence.proof`：匹配證據（片段/註解）。  
- `recommendation.fix`：修復建議。

## 4. SARIF 對應
- result.ruleId = CWE / 規則代碼  
- result.level = severity（CRITICAL/HIGH/…）  
- result.message = evidence.proof / description

## 5. 失敗訊息
- 發佈到 `Topic.STATUS_TASK_UPDATE`：`status=FAILED` + `error`。
