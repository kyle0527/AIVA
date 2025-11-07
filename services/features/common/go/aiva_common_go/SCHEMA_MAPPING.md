# AIVA Schema Mapping - Python ↔ Go

**最後更新:** 2025-10-14  
**版本:** 1.0.0

---

## Schema 對照表

### 核心訊息結構

| Python Schema | Go Schema | 狀態 | 說明 |
|--------------|-----------|------|------|
| `MessageHeader` | `MessageHeader` | ✅ 完成 | 訊息標頭 |
| `AivaMessage` | `AivaMessage` | ✅ 完成 | 統一訊息格式 |

### 掃描相關

| Python Schema | Go Schema | 狀態 | 說明 |
|--------------|-----------|------|------|
| `ScanStartPayload` | `ScanStartPayload` | 🔄 進行中 | 掃描啟動 |
| `ScanCompletedPayload` | `ScanCompletedPayload` | 🔄 進行中 | 掃描完成 |
| `ScanScope` | `ScanScope` | 🔄 進行中 | 掃描範圍 |
| `Authentication` | `Authentication` | 🔄 進行中 | 認證資訊 |
| `RateLimit` | `RateLimit` | 🔄 進行中 | 速率限制 |

### 功能任務

| Python Schema | Go Schema | 狀態 | 說明 |
|--------------|-----------|------|------|
| `FunctionTaskPayload` | `FunctionTaskPayload` | 🔄 進行中 | 功能任務載荷 |
| `FunctionTaskTarget` | `FunctionTaskTarget` | 🔄 進行中 | 任務目標 |
| `FunctionTaskContext` | `FunctionTaskContext` | 🔄 進行中 | 任務上下文 |
| `FunctionTaskTestConfig` | `FunctionTaskTestConfig` | 🔄 進行中 | 測試配置 |

### 漏洞發現

| Python Schema | Go Schema | 狀態 | 說明 |
|--------------|-----------|------|------|
| `FindingPayload` | `FindingPayload` | ⚠️ 需更新 | 漏洞發現 (結構不同) |
| `Vulnerability` | `Vulnerability` | 🔄 進行中 | 漏洞資訊 |
| `Target` / `FindingTarget` | `FindingTarget` | 🔄 進行中 | 目標資訊 |
| `FindingEvidence` | `FindingEvidence` | ⚠️ 需更新 | 證據 (欄位不同) |
| `FindingImpact` | `FindingImpact` | ⚠️ 需更新 | 影響 (欄位不同) |
| `FindingRecommendation` | `FindingRecommendation` | ⚠️ 需更新 | 修復建議 (欄位不同) |

---

## 欄位名稱對照

### FindingPayload

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `finding_id` | `FindingID` | `finding_id` | string | ✅ |
| `task_id` | `TaskID` | `task_id` | string | ✅ |
| `scan_id` | `ScanID` | `scan_id` | string | ✅ |
| `status` | `Status` | `status` | string | ✅ |
| `vulnerability` | `Vulnerability` | `vulnerability` | Vulnerability | ✅ |
| `target` | `Target` | `target` | Target | ✅ |
| `strategy` | `Strategy` | `strategy,omitempty` | *string | ❌ |
| `evidence` | `Evidence` | `evidence,omitempty` | *FindingEvidence | ❌ |
| `impact` | `Impact` | `impact,omitempty` | *FindingImpact | ❌ |
| `recommendation` | `Recommendation` | `recommendation,omitempty` | *FindingRecommendation | ❌ |
| `metadata` | `Metadata` | `metadata,omitempty` | map[string]interface{} | ❌ |
| `created_at` | `CreatedAt` | `created_at` | time.Time | ✅ |
| `updated_at` | `UpdatedAt` | `updated_at` | time.Time | ✅ |

### Vulnerability

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `name` | `Name` | `name` | string (VulnerabilityType) | ✅ |
| `cwe` | `CWE` | `cwe,omitempty` | *string | ❌ |
| `severity` | `Severity` | `severity` | string (Severity) | ✅ |
| `confidence` | `Confidence` | `confidence` | string (Confidence) | ✅ |
| `description` | `Description` | `description,omitempty` | *string | ❌ |

### Target (FindingTarget)

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `url` | `URL` | `url` | string | ✅ |
| `parameter` | `Parameter` | `parameter,omitempty` | *string | ❌ |
| `method` | `Method` | `method,omitempty` | *string | ❌ |
| `headers` | `Headers` | `headers,omitempty` | map[string]string | ❌ |
| `params` | `Params` | `params,omitempty` | map[string]interface{} | ❌ |
| `body` | `Body` | `body,omitempty` | *string | ❌ |

### FindingEvidence

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `payload` | `Payload` | `payload,omitempty` | *string | ❌ |
| `response_time_delta` | `ResponseTimeDelta` | `response_time_delta,omitempty` | *float64 | ❌ |
| `db_version` | `DBVersion` | `db_version,omitempty` | *string | ❌ |
| `request` | `Request` | `request,omitempty` | *string | ❌ |
| `response` | `Response` | `response,omitempty` | *string | ❌ |
| `proof` | `Proof` | `proof,omitempty` | *string | ❌ |

### FindingImpact

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `description` | `Description` | `description,omitempty` | *string | ❌ |
| `business_impact` | `BusinessImpact` | `business_impact,omitempty` | *string | ❌ |
| `technical_impact` | `TechnicalImpact` | `technical_impact,omitempty` | *string | ❌ |
| `affected_users` | `AffectedUsers` | `affected_users,omitempty` | *int | ❌ |
| `estimated_cost` | `EstimatedCost` | `estimated_cost,omitempty` | *float64 | ❌ |

### FindingRecommendation

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `fix` | `Fix` | `fix,omitempty` | *string | ❌ |
| `priority` | `Priority` | `priority,omitempty` | *string | ❌ |
| `remediation_steps` | `RemediationSteps` | `remediation_steps,omitempty` | []string | ❌ |
| `references` | `References` | `references,omitempty` | []string | ❌ |

### FunctionTaskPayload

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `task_id` | `TaskID` | `task_id` | string | ✅ |
| `scan_id` | `ScanID` | `scan_id` | string | ✅ |
| `priority` | `Priority` | `priority` | int | ✅ |
| `target` | `Target` | `target` | FunctionTaskTarget | ✅ |
| `context` | `Context` | `context` | FunctionTaskContext | ✅ |
| `strategy` | `Strategy` | `strategy` | string | ✅ |
| `custom_payloads` | `CustomPayloads` | `custom_payloads,omitempty` | []string | ❌ |
| `test_config` | `TestConfig` | `test_config` | FunctionTaskTestConfig | ✅ |

### FunctionTaskTarget

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `url` | `URL` | `url` | string | ✅ |
| `parameter` | `Parameter` | `parameter,omitempty` | *string | ❌ |
| `method` | `Method` | `method` | string | ✅ |
| `parameter_location` | `ParameterLocation` | `parameter_location` | string | ✅ |
| `headers` | `Headers` | `headers` | map[string]string | ✅ |
| `cookies` | `Cookies` | `cookies` | map[string]string | ✅ |
| `form_data` | `FormData` | `form_data` | map[string]interface{} | ✅ |
| `json_data` | `JSONData` | `json_data,omitempty` | map[string]interface{} | ❌ |
| `body` | `Body` | `body,omitempty` | *string | ❌ |

### FunctionTaskContext

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `db_type_hint` | `DBTypeHint` | `db_type_hint,omitempty` | *string | ❌ |
| `waf_detected` | `WAFDetected` | `waf_detected` | bool | ✅ |
| `related_findings` | `RelatedFindings` | `related_findings,omitempty` | []string | ❌ |

### FunctionTaskTestConfig

| Python 欄位 | Go 欄位 | JSON Tag | 類型 | 必填 |
|------------|---------|----------|------|------|
| `payloads` | `Payloads` | `payloads` | []string | ✅ |
| `custom_payloads` | `CustomPayloads` | `custom_payloads` | []string | ✅ |
| `blind_xss` | `BlindXSS` | `blind_xss` | bool | ✅ |
| `dom_testing` | `DOMTesting` | `dom_testing` | bool | ✅ |
| `timeout` | `Timeout` | `timeout,omitempty` | *float64 | ❌ |

---

## 命名規範

### Go 命名

- **類型名稱:** PascalCase (例: `MessageHeader`, `FindingPayload`)
- **欄位名稱:** PascalCase (例: `FindingID`, `TaskID`)
- **JSON Tag:** snake_case (例: `finding_id`, `task_id`)
- **Package:** lowercase (例: `schemas`, `config`)

### Python 命名

- **類型名稱:** PascalCase (例: `MessageHeader`, `FindingPayload`)
- **欄位名稱:** snake_case (例: `finding_id`, `task_id`)
- **Module:** lowercase (例: `schemas`, `config`)

---

## 差異分析

### 現有問題

1. **FindingPayload 結構不同**
   - Python: 使用 `status` 欄位,包含 `created_at`, `updated_at`
   - Go: 缺少這些欄位

2. **FindingEvidence 欄位不同**
   - Python: 包含 `response_time_delta`, `db_version`, `proof`
   - Go (舊): 只有 `request`, `response`, `payload`, `proof_of_concept`

3. **FindingImpact 欄位不同**
   - Python: 包含 `affected_users`, `estimated_cost`
   - Go (舊): 只有 CIA triad 欄位

4. **FunctionTaskPayload 缺失**
   - Go schemas 中完全缺少此結構

5. **命名不一致**
   - 部分 Go struct 使用舊的命名 (例: `CVEID`, `GHSAID`)
   - 應統一為 `CWE`, `CVE` 等

---

## 遷移策略

### Phase 1: 擴充 Go schemas ✅
1. 添加完整的 `FunctionTaskPayload` 家族
2. 更新 `FindingPayload` 匹配 Python 版本
3. 更新 `Vulnerability`, `FindingEvidence`, `FindingImpact`, `FindingRecommendation`

### Phase 2: 統一命名 ✅
1. 確保所有 JSON tag 使用 snake_case
2. 修正欄位命名 (CVEID → CVE)
3. 統一可選欄位使用指標

### Phase 3: 移除本地 models 🔄
1. 更新 `function_sca_go` 使用共用 schemas
2. 更新 `function_cspm_go` 使用共用 schemas
3. 更新 `function_authn_go` 使用共用 schemas
4. 更新 `function_ssrf_go` 使用共用 schemas

### Phase 4: 添加測試 ✅
1. 序列化/反序列化測試
2. JSON 相容性測試
3. 與 Python 的互操作測試

---

## 測試策略

### 單元測試

```go
func TestFindingPayloadSerialization(t *testing.T) {
    finding := schemas.FindingPayload{
        FindingID: "finding_123",
        TaskID:    "task_456",
        ScanID:    "scan_789",
        // ...
    }
    
    data, err := json.Marshal(finding)
    assert.NoError(t, err)
    
    var decoded schemas.FindingPayload
    err = json.Unmarshal(data, &decoded)
    assert.NoError(t, err)
    assert.Equal(t, finding.FindingID, decoded.FindingID)
}
```

### 互操作測試

1. Python → JSON → Go
2. Go → JSON → Python
3. 驗證所有欄位正確映射

---

## 維護指南

### 添加新 Schema

1. 在 Python `aiva_common/schemas.py` 定義
2. 在 Go `aiva_common_go/schemas/` 建立對應文件
3. 更新本文件的對照表
4. 添加測試

### 修改現有 Schema

1. 同步修改 Python 和 Go
2. 更新文件
3. 運行互操作測試
4. 更新版本號

---

**維護者:** AIVA Architecture Team  
**聯絡:** [待填寫]
