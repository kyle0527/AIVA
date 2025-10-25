# AIVA Schema 架構分析報告

## 📋 執行摘要

經過檢查各模組的 README 和實際代碼結構,AIVA 採用了**雙軌 Schema 管理架構**:

1. **services/aiva_common** - Python 專用共用模組
2. **services/features/common/go/aiva_common_go** - Go 語言共用模組  
3. **統一生成機制** - 從 `core_schema_sot.yaml` 生成跨語言 schemas

## 🏗️ 架構詳解

### 1. Schema 定義來源 (Single Source of Truth)

```yaml
位置: services/aiva_common/core_schema_sot.yaml
用途: 所有語言 Schema 的唯一事實來源
生成工具: services/aiva_common/tools/schema_codegen_tool.py
```

### 2. 生成目標路徑

| 語言 | 生成路徑 | 用途 |
|------|---------|------|
| **Python** | `services/aiva_common/schemas/generated/` | Python 模組共用 |
| **Go** | `services/features/common/go/aiva_common_go/schemas/generated/` | Go 服務共用 |
| **Rust** | `services/scan/info_gatherer_rust/src/schemas/generated/` | Rust 模組共用 |

### 3. 各語言模組的引用方式

#### Python 模組

```python
# ✅ 正確做法 - 引用 services/aiva_common
from aiva_common.enums import Severity, Confidence
from aiva_common.schemas import FindingPayload, SARIFResult

# ❌ 錯誤做法 - Fallback 重複定義 (已發現需修復)
try:
    from aiva_common.enums import Severity
except ImportError:
    class Severity(str, Enum):  # 重複定義!
        CRITICAL = "critical"
```

**應用模組**:
- `services/features/` - 所有 Python 功能模組
- `services/scan/` - 掃描引擎
- `services/core/` - 核心服務
- `services/integration/` - 整合服務

#### Go 模組

```go
// ✅ 正確做法 - 引用 aiva_common_go/schemas/generated
import "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"

// 使用生成的 schemas
func processTask(payload schemas.FunctionTaskPayload) {
    // ...
}

// ❌ 錯誤做法 - 在各模組內重複定義 (已發現)
type FunctionTaskPayload struct {  // 與生成的重複!
    TaskID string `json:"task_id"`
    // ...
}
```

**應用模組**:
- `services/features/function_sca_go/` - ⚠️ 有重複定義 (9個類型)
- `services/features/function_ssrf_go/` - ✅ 無重複定義
- `services/features/function_cspm_go/` - ✅ 無重複定義
- `services/features/function_authn_go/` - ✅ 無重複定義

#### Rust 模組

```rust
// ✅ 正確做法 - 引用生成的 schemas
use crate::schemas::generated::{FunctionTaskPayload, FindingPayload};

// ❌ 錯誤做法 - 在模組內重複定義 (已發現)
pub struct FunctionTaskPayload {  // 與生成的重複!
    pub task_id: String,
    // ...
}
```

**應用模組**:
- `services/features/function_sast_rust/` - ⚠️ 有重複定義 (5個類型)
- `services/scan/info_gatherer_rust/` - 使用生成的 schemas

## ⚠️ 發現的問題

### 問題 1: Go 模組 function_sca_go 重複定義

**文件**: `services/features/function_sca_go/pkg/models/models.go`

**重複的類型** (9個):
1. `FunctionTaskPayload`
2. `FunctionTaskTarget`
3. `FunctionTaskContext`
4. `FunctionTaskTestConfig`
5. `FindingPayload`
6. `Vulnerability`
7. `FindingTarget`
8. `FindingEvidence`
9. `FindingImpact`
10. `FindingRecommendation`

**衝突**: 這些類型已在 `services/features/common/go/aiva_common_go/schemas/generated/schemas.go` 中生成

### 問題 2: Rust 模組 function_sast_rust 重複定義

**文件**: `services/features/function_sast_rust/src/models.rs`

**重複的類型** (5個):
1. `FunctionTaskPayload`
2. `FindingPayload`
3. `Vulnerability`
4. `FindingEvidence`
5. `FindingImpact`

**衝突**: 這些類型應該從生成的 Rust schemas 中引用 (但 Rust 生成目標在 `services/scan/info_gatherer_rust/src/schemas/generated/`)

### 問題 3: Python Fallback 代碼

**文件**: `services/features/client_side_auth_bypass/worker.py`

```python
# ❌ 不應該存在的 Fallback
try:
    from aiva_common.enums import Severity, Confidence
except ImportError:
    class Severity(str, Enum):
        # 重複定義...
```

**影響**: 如果 aiva_common 導入失敗,會使用不一致的本地定義

## 🔄 架構規範

### 正確的架構關係

```
core_schema_sot.yaml (唯一來源)
        ↓
schema_codegen_tool.py (生成工具)
        ↓
    ┌───────┴───────┬───────────────┐
    ↓               ↓               ↓
Python schemas   Go schemas    Rust schemas
(aiva_common)   (aiva_common_go) (info_gatherer)
    ↓               ↓               ↓
Python 模組      Go 服務        Rust 模組
(直接引用)      (直接引用)      (直接引用)
```

### 不應該存在的模式

```
❌ 各模組自行定義相同的類型
❌ Fallback 重複定義 (try/except ImportError)
❌ 手動維護與生成 schemas 相同的類型
```

## 📊 檢查結果統計

| 模組 | 語言 | 狀態 | 重複數 | 備註 |
|------|------|------|--------|------|
| function_sca_go | Go | ❌ 有問題 | 9個類型 | 需移除 models.go 重複定義 |
| function_ssrf_go | Go | ✅ 正常 | 0 | 正確使用生成的 schemas |
| function_cspm_go | Go | ✅ 正常 | 0 | 正確使用生成的 schemas |
| function_authn_go | Go | ✅ 正常 | 0 | 正確使用生成的 schemas |
| function_sast_rust | Rust | ⚠️ 有問題 | 5個類型 | 需確認 Rust schemas 位置 |
| client_side_auth_bypass | Python | ⚠️ 有問題 | Fallback 代碼 | 需移除 try/except 重複定義 |

## ✅ 修復建議

### 1. 修復 function_sca_go

```go
// 在 pkg/models/models.go 中
package models

// ⚠️  注意：FunctionTaskPayload, FindingPayload, Vulnerability 等類型已移至統一生成的 schemas
// 請使用: import "aiva_common_go/schemas/generated"
// 
// 此文件保留 SCA 模組專用的業務邏輯類型

// SCA 專用的業務類型 (不與生成的 schemas 重複)
type DependencyInfo struct {
    Name      string `json:"name"`
    Version   string `json:"version"`
    Ecosystem string `json:"ecosystem"`
    // ...
}
```

**修改文件**:
- `cmd/worker/main.go` - 改用 `schemas.FunctionTaskPayload`
- `internal/scanner/sca_scanner.go` - 改用 `schemas.FindingPayload`

### 2. 修復 function_sast_rust

**選項 A**: 在 function_sast_rust 中也生成 schemas
```yaml
# 在 core_schema_sot.yaml 中添加
rust:
  targets:
    - "services/scan/info_gatherer_rust/src/schemas/generated"
    - "services/features/function_sast_rust/src/schemas/generated"  # 新增
```

**選項 B**: 從 info_gatherer_rust 引用 schemas
```rust
// 在 Cargo.toml 中添加依賴
[dependencies]
info_gatherer_schemas = { path = "../../../scan/info_gatherer_rust" }
```

### 3. 移除 Python Fallback 代碼

```python
# 在所有 Python 模組中移除 try/except ImportError
# 確保 aiva_common 正確安裝: pip install -e services/aiva_common

# ✅ 簡化為直接導入
from aiva_common.enums import Severity, Confidence
from aiva_common.schemas import FindingPayload
```

## 🎯 長期規範

### 新增模組時的檢查清單

- [ ] **Python 模組**: 確認從 `aiva_common` 導入標準類型
- [ ] **Go 模組**: 確認從 `aiva_common_go/schemas/generated` 導入
- [ ] **Rust 模組**: 確認 schemas 生成或引用策略
- [ ] **禁止**: 在業務模組中重複定義標準 Schema 類型
- [ ] **禁止**: 使用 Fallback 代碼重複定義枚舉

### 代碼審查要點

```bash
# 檢查是否有重複定義
grep -r "type FunctionTaskPayload" services/features --include="*.go"
grep -r "pub struct FunctionTaskPayload" services/features --include="*.rs"
grep -r "class.*Severity.*Enum" services/features --include="*.py"

# 檢查是否有 Fallback 代碼
grep -r "except ImportError" services/features --include="*.py" -A 5
```

## 📝 結論

AIVA 的 Schema 架構**設計正確**,使用 YAML SOT + 代碼生成確保跨語言一致性。

**發現的問題**是**實現偏差**,部分模組沒有遵循架構規範:
- ❌ Go 模組 function_sca_go 手動重複定義
- ❌ Rust 模組 function_sast_rust 手動重複定義  
- ❌ Python 模組存在 Fallback 重複定義

**修復後**,所有模組將統一使用生成的 schemas,消除重複定義和潛在的不一致性。

---

**報告時間**: 2025-10-25  
**分析者**: AIVA Architecture Analysis Tool  
**相關文件**: 
- `services/features/README.md`
- `services/features/common/go/aiva_common_go/README.md`
- `services/features/function_sca_go/README.md`
