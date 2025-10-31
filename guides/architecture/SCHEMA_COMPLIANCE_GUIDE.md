# AIVA Schema 標準化開發規範

## 📑 目錄

- [📋 概述](#-概述)
- [🎯 核心原則](#-核心原則)
- [🔧 各語言實作規範](#-各語言實作規範)
- [📝 代碼規範](#-代碼規範)
- [🔄 同步流程](#-同步流程)
- [✅ 驗證檢查](#-驗證檢查)
- [🐛 常見問題](#-常見問題)
- [🔗 相關資源](#-相關資源)

## 概述

為了確保 AIVA 專案的跨語言一致性和可維護性，本規範強制執行單一事實來源原則，所有模組必須使用標準化的 schema 定義。

## 核心原則

### 1. 單一事實來源 (Single Source of Truth)
- `aiva_common` 為所有 schema 定義的權威來源
- 禁止在各模組中定義自訂的 `FindingPayload`, `Vulnerability`, `Target`, `Evidence` 等結構
- 所有語言的 schema 實現必須與 `aiva_common` 保持同步

### 2. 標準化欄位命名
- 使用 `finding_id` 而非 `FindingID` 或其他變體
- 使用 `created_at`, `updated_at` 表示時間戳
- 使用 `evidence` 而非 `evidences`

### 3. 跨語言一致性
- Go、Rust、TypeScript 三種語言的 schema 必須在結構上保持一致
- 支援相同的資料類型和驗證規則
- 維持相同的序列化/反序列化行為

## 各語言實作規範

### Go 語言模組

#### 必須使用的導入
```go
import schemas "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"
```

#### 禁止的定義
```go
// ❌ 禁止：自訂 FindingPayload
type FindingPayload struct {
    FindingID   string `json:"finding_id"`
    // ...
}

// ❌ 禁止：自訂 Vulnerability
type Vulnerability struct {
    Severity string `json:"severity"`
    // ...
}
```

#### 正確的使用方式
```go
// ✅ 正確：使用標準 schema
func createFinding(vulnType string, severity string) *schemas.FindingPayload {
    return &schemas.FindingPayload{
        FindingId: generateID(),
        Vulnerability: &schemas.Vulnerability{
            Type:     vulnType,
            Severity: severity,
        },
        // ...
    }
}
```

### Rust 語言模組

#### 必須實現的模組結構
```rust
// src/schemas/generated/mod.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FindingPayload {
    pub finding_id: String,
    pub vulnerability: Vulnerability,
    pub target: Target,
    pub evidence: Evidence,
    // ...
}
```

#### 禁止的定義
```rust
// ❌ 禁止：在 main.rs 或其他檔案中定義
struct FindingPayload {
    finding_id: String,
    // ...
}
```

#### 正確的使用方式
```rust
// ✅ 正確：使用標準 schema 模組
use crate::schemas::generated::{FindingPayload, Vulnerability};

fn create_finding() -> FindingPayload {
    FindingPayload {
        finding_id: Uuid::new_v4().to_string(),
        vulnerability: Vulnerability {
            // ...
        },
        // ...
    }
}
```

### TypeScript 語言模組

#### 必須使用的導入
```typescript
import { FindingPayload, Vulnerability, Target, Evidence } from '../../../schemas/aiva_schemas';
```

#### 禁止的定義
```typescript
// ❌ 禁止：自訂介面
interface FindingPayload {
    finding_id: string;
    // ...
}

// ❌ 禁止：自訂型別
type CustomVulnerability = {
    severity: string;
    // ...
}
```

#### 正確的使用方式
```typescript
// ✅ 正確：使用標準 schema
function createFinding(vulnType: string): FindingPayload {
    return {
        finding_id: generateId(),
        vulnerability: {
            type: vulnType,
            severity: 'high'
        } as Vulnerability,
        // ...
    };
}
```

## 欄位標準化規範

### 核心欄位定義

#### FindingPayload
```json
{
    "finding_id": "string (UUID)",
    "scan_id": "string (UUID)",
    "vulnerability": "Vulnerability",
    "target": "Target", 
    "evidence": "Evidence",
    "metadata": "object",
    "created_at": "string (ISO 8601)",
    "updated_at": "string (ISO 8601)"
}
```

#### Vulnerability
```json
{
    "type": "string",
    "severity": "string (low|medium|high|critical)",
    "cwe_id": "string",
    "cvss_score": "number",
    "description": "string",
    "remediation": "string"
}
```

#### Target
```json
{
    "host": "string",
    "port": "number",
    "protocol": "string",
    "path": "string",
    "method": "string"
}
```

#### Evidence
```json
{
    "description": "string",
    "request": "string",
    "response": "string",
    "additional_info": "object"
}
```

### 禁止的欄位變體

| 禁止使用 | 標準欄位 | 說明 |
|---------|---------|------|
| `FindingID` | `finding_id` | 使用 snake_case |
| `ScanID` | `scan_id` | 使用 snake_case |
| `CreatedAt` | `created_at` | 使用 snake_case |
| `UpdatedAt` | `updated_at` | 使用 snake_case |
| `evidences` | `evidence` | 使用單數形式 |
| `addtional_info` | `additional_info` | 正確拼寫 |

## 合規性檢查

### 自動化檢查工具

1. **Schema 合規性驗證器**
   ```bash
   python tools/schema_compliance_validator.py --check-all
   ```

2. **CI/CD 集成檢查**
   ```bash
   python tools/ci_schema_check.py --strict
   ```

3. **Pre-commit Hook**
   ```bash
   pre-commit install
   ```

### 檢查頻率

- **Pre-commit**: 每次提交前自動檢查
- **CI/CD**: 每次 Push 和 Pull Request 時檢查
- **定期檢查**: 每週運行完整合規性報告

### 合規性閾值

- **通過標準**: 平均分數 ≥ 90 分且無不合規模組
- **警告標準**: 平均分數 ≥ 80 分但有部分問題
- **失敗標準**: 平均分數 < 80 分或有不合規模組

## 開發工作流程

### 新模組開發

1. **確定模組類型和語言**
2. **使用對應的標準 schema 導入**
3. **禁止定義自訂結構**
4. **執行合規性檢查**
5. **提交前運行 pre-commit hook**

### 現有模組修改

1. **檢查是否影響 schema 使用**
2. **確保使用標準定義**
3. **更新相關測試**
4. **運行合規性檢查**
5. **通過 CI/CD 檢查**

### 發現不合規問題

1. **運行詳細檢查**: `python tools/schema_compliance_validator.py`
2. **查看問題報告和建議**
3. **修復不合規程式碼**
4. **再次檢查直到通過**
5. **提交修復**

## 例外情況處理

### 臨時例外

如果因為特殊情況需要臨時例外，必須：

1. **在 `tools/schema_compliance.toml` 中註冊例外**
2. **說明例外原因和期限**
3. **建立對應的 GitHub Issue**
4. **設定自動提醒**

例如：
```toml
temporary_exceptions = [
    "services/legacy/old_module.go:遺留系統，計劃重構:2025-12-31:ISSUE-123"
]
```

### 長期例外

長期例外需要：

1. **架構委員會批准**
2. **記錄在設計文件中**
3. **定期審查必要性**

## 違規處理

### 警告級別
- 部分合規但有改進空間
- 提供改進建議
- 不阻止合併但需要後續修復

### 錯誤級別
- 存在不合規模組
- 阻止合併直到修復
- 需要強制修復才能繼續

### 嚴重級別
- 嚴重的 schema 不一致
- 可能影響系統穩定性
- 需要立即修復

## 最佳實踐

### 開發建議

1. **早期檢查**: 開發過程中頻繁運行合規性檢查
2. **測試覆蓋**: 確保 schema 相關程式碼有足夠測試
3. **文檔更新**: 更改 schema 時同步更新文檔
4. **版本管理**: 使用語義化版本管理 schema 變更

### 團隊協作

1. **Code Review**: 重點檢查 schema 使用是否正確
2. **知識分享**: 定期分享 schema 最佳實踐
3. **問題追蹤**: 建立 schema 相關問題的追蹤機制

## 工具使用指南

### 開發時檢查
```bash
# 檢查特定語言
python tools/schema_compliance_validator.py --language go

# 生成詳細報告
python tools/schema_compliance_validator.py --format markdown --output report.md

# CI 模式檢查
python tools/ci_schema_check.py --strict --threshold 90
```

### 問題修復
```bash
# 快速檢查變更的檔案
python tools/git-hooks/pre-commit-schema-check.py

# 檢視合規性統計
python tools/schema_compliance_validator.py --format json | jq '.summary'
```

## 更新與維護

### Schema 更新流程

1. **更新 `aiva_common` 定義**
2. **同步各語言實現**
3. **更新驗證工具**
4. **運行全面測試**
5. **發布版本**

### 工具維護

1. **定期更新檢查規則**
2. **改進檢查效能**
3. **增加新的檢查項目**
4. **修復誤報問題**

---

**重要提醒**: 此規範為強制性規範，所有開發人員必須遵循。違反此規範的程式碼將無法通過 CI/CD 檢查。如有疑問，請諮詢架構團隊。