# AIVA Unified Schema - Single Source of Truth

## Overview

The AIVA V2 架構使用單一 YAML 檔案作為所有資料合約的唯一定義來源，消除重複定義並提供跨語言一致性。

**檔案位置**: `services/aiva_common/core_schema_sot.yaml`  
**行數**: 2305 lines  
**支援語言**: Python, Go, Rust, TypeScript

## Why Single Source of Truth (SOT)?

### V1 架構的問題

在 V1 架構中，我們有 7 個分散的 `schemas.py` 檔案：

1. `services/core/aiva_core/schemas.py` (290 lines)
2. `services/scan/aiva_scan/schemas.py` (190 lines)
3. `services/features/function_idor/schemas.py` (226 lines)
4. `services/features/function_postex/schemas.py` (305 lines)
5. `services/features/function_sqli/schemas.py` (198 lines)
6. `services/features/function_ssrf/schemas.py` (140 lines)
7. `services/features/function_xss/schemas.py` (178 lines)

**總計**: 1527 lines of duplicate/scattered definitions

**問題**:
- ❌ 重複定義導致不一致
- ❌ 跨語言同步困難
- ❌ 難以維護版本控制
- ❌ 無法強制統一標準

### V2 架構的解決方案

**單一 SOT**: `core_schema_sot.yaml` (2305 lines)

**優勢**:
- ✅ 單一定義，零重複
- ✅ 自動生成多語言程式碼
- ✅ 集中化驗證規則
- ✅ 版本控制更容易
- ✅ 跨語言一致性保證

## Schema 結構

### 檔案組織

```yaml
version: 1.1.0

metadata:
  description: AIVA跨語言Schema統一定義
  last_updated: '2025-10-30T00:00:00.000000'
  total_schemas: 72

base_types:
  # 基礎訊息類型
  MessageHeader: {...}
  Target: {...}
  Vulnerability: {...}
  
  # 分析結果
  AssetAnalysis: {...}
  VulnerabilityCandidate: {...}
  
  # 發現與報告
  Finding: {...}
  FindingEvidence: {...}
  CVSSMetrics: {...}
  
  # ... 更多 schema 定義
```

### Schema 定義範例

```yaml
base_types:
  Finding:
    description: '漏洞發現記錄 - 符合 OWASP、CWE、CVE 標準'
    fields:
      finding_id:
        type: str
        required: true
        description: '唯一識別碼（UUID格式）'
      
      title:
        type: str
        required: true
        description: '漏洞標題'
      
      severity:
        type: str
        required: true
        description: '嚴重程度'
        enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
      
      cvss_score:
        type: float
        required: false
        description: 'CVSS 評分 (0.0-10.0)'
        ge: 0.0
        le: 10.0
      
      # ... 更多欄位
```

## 使用方式

### 1. Python (Pydantic)

目前 V1 schema 檔案仍可使用，但會顯示棄用警告：

```python
# 會顯示 DeprecationWarning
from services.core.aiva_core.schemas import AssetAnalysis

# V2 (未來，codegen 完成後)
from services.aiva_common.schemas.generated import AssetAnalysis
```

### 2. Go (Struct)

未來生成：

```go
// Auto-generated from core_schema_sot.yaml
package schemas

type Finding struct {
    FindingID string `json:"finding_id" validate:"required"`
    Title     string `json:"title" validate:"required"`
    Severity  string `json:"severity" validate:"required,oneof=CRITICAL HIGH MEDIUM LOW INFO"`
    CVSSScore *float64 `json:"cvss_score,omitempty" validate:"omitempty,gte=0,lte=10"`
}
```

### 3. Rust (Struct)

未來生成：

```rust
// Auto-generated from core_schema_sot.yaml
#[derive(Debug, Serialize, Deserialize)]
pub struct Finding {
    pub finding_id: String,
    pub title: String,
    pub severity: Severity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cvss_score: Option<f64>,
}
```

### 4. TypeScript (Interface)

未來生成：

```typescript
// Auto-generated from core_schema_sot.yaml
export interface Finding {
  finding_id: string;
  title: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  cvss_score?: number;
}
```

## Code Generation (計劃中)

### 生成工具

```bash
# 生成所有語言的 schema
python tools/schema_codegen_tool.py --input services/aiva_common/core_schema_sot.yaml

# 只生成 Python
python tools/schema_codegen_tool.py --lang python

# 只生成 Go
python tools/schema_codegen_tool.py --lang go

# 驗證 schema
python tools/schema_codegen_tool.py --validate
```

### 輸出位置

```
services/aiva_common/schemas/
├── generated/
│   ├── python/
│   │   ├── __init__.py
│   │   ├── base_types.py
│   │   ├── findings.py
│   │   └── ...
│   ├── go/
│   │   ├── base_types.go
│   │   ├── findings.go
│   │   └── ...
│   ├── rust/
│   │   ├── mod.rs
│   │   ├── base_types.rs
│   │   ├── findings.rs
│   │   └── ...
│   └── typescript/
│       ├── index.ts
│       ├── base_types.ts
│       ├── findings.ts
│       └── ...
```

## Schema 規範

### 欄位類型

支援的類型：

- **基本類型**: `str`, `int`, `float`, `bool`
- **容器類型**: `List[T]`, `Dict[str, T]`, `Optional[T]`
- **特殊類型**: `datetime`, `UUID`, `Any`
- **枚舉類型**: `enum: [value1, value2, ...]`

### 驗證規則

支援的驗證：

- `required`: true/false
- `default`: 預設值
- `ge`: 大於等於 (>=)
- `le`: 小於等於 (<=)
- `gt`: 大於 (>)
- `lt`: 小於 (<)
- `min_length`: 最小長度
- `max_length`: 最大長度
- `pattern`: 正則表達式
- `enum`: 枚舉值列表

### 範例

```yaml
base_types:
  UserInput:
    description: '使用者輸入驗證'
    fields:
      username:
        type: str
        required: true
        min_length: 3
        max_length: 50
        pattern: '^[a-zA-Z0-9_]+$'
        description: '使用者名稱'
      
      age:
        type: int
        required: false
        ge: 0
        le: 150
        description: '年齡'
      
      role:
        type: str
        required: true
        enum: ['admin', 'user', 'guest']
        description: '角色'
```

## 如何新增 Schema

### Step 1: 編輯 YAML 檔案

打開 `services/aiva_common/core_schema_sot.yaml`，新增你的 schema：

```yaml
base_types:
  YourNewSchema:
    description: '你的 schema 描述'
    fields:
      field1:
        type: str
        required: true
        description: '欄位1描述'
      field2:
        type: int
        required: false
        default: 0
        description: '欄位2描述'
```

### Step 2: 驗證 Schema

```bash
python tools/schema_codegen_tool.py --validate
```

### Step 3: 生成程式碼 (未來)

```bash
python tools/schema_codegen_tool.py
```

### Step 4: 使用生成的 Schema

```python
from services.aiva_common.schemas.generated import YourNewSchema

data = YourNewSchema(field1="value", field2=42)
```

## 版本控制

### Schema 版本規範

遵循 Semantic Versioning:

- **Major** (1.x.x): 破壞性變更
- **Minor** (x.1.x): 新增欄位（向後相容）
- **Patch** (x.x.1): 修復、文檔更新

### 版本歷史

- **v1.1.0** (2025-10-30): 新增 async_utils、plugins、cli schemas
- **v1.0.0** (2025-10-01): 初始統一 schema

### 相容性政策

- 新增欄位：向後相容（Minor 版本）
- 修改欄位類型：破壞性變更（Major 版本）
- 刪除欄位：破壞性變更（Major 版本）
- 修改驗證規則：視影響程度（Minor 或 Major）

## 遷移指南

詳見：`docs/V2_ARCHITECTURE_MIGRATION_GUIDE.md`

### 快速遷移檢查清單

- [ ] 檢查現有 schema 檔案的棄用警告
- [ ] 確認你的 schema 是否已在 `core_schema_sot.yaml` 中定義
- [ ] 如未定義，新增到 `core_schema_sot.yaml`
- [ ] 等待 codegen 工具完成後，更新 import 路徑
- [ ] 測試新的 schema
- [ ] 移除舊的 schema 檔案

## 標準符合性

AIVA schema 符合以下業界標準：

- **CWE** (Common Weakness Enumeration): 漏洞分類
- **CVE** (Common Vulnerabilities and Exposures): 漏洞識別
- **CVSS v3.1/v4.0**: 漏洞評分系統
- **OWASP**: 應用安全標準
- **NIST**: 安全框架

## 常見問題 (FAQ)

### Q: 為什麼選擇 YAML 而不是 JSON Schema 或 Protocol Buffers?

**A**: 
- YAML 更易讀、易寫
- 支援註解，便於文檔化
- 可輕鬆轉換為任何格式（JSON Schema, Proto, etc.）
- 更適合手動維護

### Q: codegen 工具何時完成？

**A**: 預計 2025 Q4 完成，目前專注於架構統一。

### Q: V1 schema 何時移除？

**A**: 在所有使用處遷移完成後（預計 2026 Q2）。

### Q: 如何貢獻新的 schema？

**A**: 
1. Fork repository
2. 編輯 `core_schema_sot.yaml`
3. 提交 Pull Request
4. 通過 review 後合併

## 相關文件

- [V2 架構遷移指南](../docs/V2_ARCHITECTURE_MIGRATION_GUIDE.md)
- [跨語言通訊框架](./cross_language/README.md)
- [Protocol Buffers 定義](./protocols/README.md)

---

**維護者**: AIVA Development Team  
**最後更新**: 2025-11-05  
**版本**: 1.1.0
