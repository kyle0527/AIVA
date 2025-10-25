# 跨語言 Schema 重複定義分析報告

**分析日期**: 2025-10-25  
**範圍**: Features 模組 (Python, Go, Rust)  
**狀態**: 🔴 發現嚴重問題

## 執行摘要

在檢查 Features 模組的跨語言代碼時,發現了嚴重的 Schema 重複定義和不一致問題:

### 關鍵發現

1. **代碼生成工具存在缺陷**: `schema_codegen_tool.py` 生成的 Go 代碼包含 Python 語法
2. **重複定義**: 9 個類型同時存在於手動編寫和自動生成的代碼中
3. **未統一使用**: Go 模組使用手動編寫的 models,而非生成的 schemas
4. **架構不一致**: YAML SOT 設計未被完全實施

## 詳細分析

### 1. Schema 定義來源

AIVA 設計了一個 **Single Source of Truth (SOT)** 架構:

```
core_schema_sot.yaml (YAML)
         ↓
schema_codegen_tool.py (代碼生成器)
         ↓
├── Python (Pydantic v2) → services/aiva_common/schemas/generated/
├── Go (structs) → services/features/common/go/aiva_common_go/schemas/generated/
└── Rust (Serde) → services/scan/info_gatherer_rust/src/schemas/generated/
```

**實際配置**:
```yaml
generation_config:
  python:
    target_dir: "services/aiva_common/schemas/generated"
  go:
    target_dir: "services/features/common/go/aiva_common_go/schemas/generated"
  rust:
    target_dir: "services/scan/info_gatherer_rust/src/schemas/generated"
```

### 2. 發現的問題

#### 問題 1: 生成的 Go 代碼包含 Python 語法錯誤

**位置**: `services/features/common/go/aiva_common_go/schemas/generated/schemas.go`

**錯誤示例**:
```go
// ❌ 錯誤: 使用了 Python 語法
type AIVAResponse struct {
    Payload  Optional[Dict[str, Any]]  `json:"payload,omitempty"`  // 響應載荷
}

// ❌ 錯誤: 使用了 Python 語法  
type FunctionTaskTarget struct {
    JsonData  Optional[Dict[str, Any]]  `json:"json_data,omitempty"`  // JSON資料
}

// ❌ 錯誤: 使用了 Python 語法
type FunctionTaskTestConfig struct {
    Timeout  Optional[float]  `json:"timeout,omitempty"`  // 請求逾時(秒)
}
```

**應該是**:
```go
// ✅ 正確: Go 語法
type AIVAResponse struct {
    Payload  map[string]interface{}  `json:"payload,omitempty"`  // 響應載荷
}

// ✅ 正確: Go 語法
type FunctionTaskTarget struct {
    JsonData  map[string]interface{}  `json:"json_data,omitempty"`  // JSON資料
}

// ✅ 正確: Go 語法
type FunctionTaskTestConfig struct {
    Timeout  *float64  `json:"timeout,omitempty"`  // 請求逾時(秒)
}
```

**統計**: 發現 **11 處** Python 語法錯誤

#### 問題 2: 重複定義 (Go 代碼)

**手動編寫的 Models** (`services/features/function_sca_go/pkg/models/models.go`):
- FunctionTaskPayload
- FunctionTaskTarget
- FunctionTaskContext
- FunctionTaskTestConfig
- FindingPayload
- Vulnerability
- FindingTarget
- FindingEvidence
- FindingImpact
- FindingRecommendation

**自動生成的 Schemas** (`services/features/common/go/aiva_common_go/schemas/generated/schemas.go`):
- MessageHeader
- Target
- Vulnerability ⚠️ 重複
- AivaMessage
- AIVARequest
- AIVAResponse
- FunctionTaskPayload ⚠️ 重複
- FunctionTaskTarget ⚠️ 重複
- FunctionTaskContext ⚠️ 重複
- FunctionTaskTestConfig ⚠️ 重複
- FindingPayload ⚠️ 重複
- FindingEvidence ⚠️ 重複
- FindingImpact ⚠️ 重複
- FindingRecommendation ⚠️ 重複

**重複的類型** (9 個):
1. `Vulnerability`
2. `FunctionTaskPayload`
3. `FunctionTaskTarget`
4. `FunctionTaskContext`
5. `FunctionTaskTestConfig`
6. `FindingPayload`
7. `FindingEvidence`
8. `FindingImpact`
9. `FindingRecommendation`

#### 問題 3: Rust 代碼也存在相同問題

**位置**: `services/features/function_sast_rust/src/models.rs`

手動定義的類型與 YAML SOT 重複:
- FunctionTaskPayload
- TaskTarget (應對應 Target)
- FindingPayload
- Vulnerability
- FindingTarget
- FindingEvidence
- FindingImpact

### 3. 影響範圍

#### 受影響的 Go 模組
1. **function_sca_go** - 軟體組成分析
   - 使用手動 models,未使用生成的 schemas
   
2. **function_ssrf_go** - SSRF 檢測
   - 位置: `services/features/function_ssrf_go/`
   - 需要檢查是否有重複定義

3. **function_cspm_go** - 雲安全態勢管理
   - 位置: `services/features/function_cspm_go/`
   - 需要檢查是否有重複定義

4. **function_authn_go** - 認證測試
   - 位置: `services/features/function_authn_go/`
   - 需要檢查是否有重複定義

#### 受影響的 Rust 模組
1. **function_sast_rust** - 靜態應用安全測試
   - 使用手動 models,未使用生成的 schemas

### 4. 根本原因分析

#### 原因 1: 代碼生成器的類型映射錯誤

`schema_codegen_tool.py` 中的類型映射可能未正確處理複雜類型:

**YAML 配置**:
```yaml
generation_config:
  go:
    field_mapping:
      "Dict[str, Any]": "map[string]interface{}"
      "Dict[str, str]": "map[string]string"
      "List[str]": "[]string"
      "Optional[str]": "*string"
```

**問題**: 映射規則未處理嵌套類型,如 `Optional[Dict[str, Any]]`

#### 原因 2: 生成的代碼未被使用

各 Go 模組開發者選擇手動編寫 models 而非使用生成的 schemas,可能因為:
1. 生成的代碼存在錯誤(如上述 Python 語法問題)
2. 生成的代碼不符合實際需求
3. 開發者不知道存在生成的 schemas
4. 生成工具未被整合到開發流程中

#### 原因 3: SOT 架構未完全實施

雖然設計了 YAML SOT,但實際執行中:
1. 代碼生成工具存在 bug
2. 缺乏自動化流程(CI/CD 整合)
3. 沒有強制檢查機制
4. 文檔不足,開發者不了解 SOT 架構

### 5. 字段差異分析

#### FunctionTaskPayload 比較

**YAML SOT 定義**:
```yaml
FunctionTaskPayload:
  fields:
    task_id: str (required)
    scan_id: str (required)
    priority: int (required, 0-10)
    target: FunctionTaskTarget (required)
    context: FunctionTaskContext (required)
    strategy: str (required, enum)
    custom_payloads: List[str] (optional)
    test_config: FunctionTaskTestConfig (required)
```

**手動 Go models** (`function_sca_go/pkg/models/models.go`):
```go
type FunctionTaskPayload struct {
    TaskID       string                 `json:"task_id"`
    FunctionType string                 `json:"function_type"`  // ⚠️ 額外字段
    Target       FunctionTaskTarget     `json:"target"`
    Context      FunctionTaskContext    `json:"context,omitempty"`  // ⚠️ 標記為可選
    TestConfig   FunctionTaskTestConfig `json:"test_config,omitempty"`  // ⚠️ 標記為可選
}
```

**差異**:
- ❌ 缺少: `scan_id`, `priority`, `strategy`, `custom_payloads`
- ✅ 額外: `function_type`
- ⚠️ `context` 和 `test_config` 被標記為可選,但 YAML 定義為必需

**手動 Rust models** (`function_sast_rust/src/models.rs`):
```rust
pub struct FunctionTaskPayload {
    pub task_id: String,
    pub function_type: String,  // ⚠️ 額外字段
    pub target: TaskTarget,
    pub options: Option<TaskOptions>,  // ⚠️ 不同的字段名
}
```

**差異**:
- ❌ 缺少大部分 YAML 定義的字段
- ✅ 額外: `function_type`, `options`
- ⚠️ 使用不同的字段名(`options` vs `test_config`)

### 6. 一致性問題總結

| 類型 | YAML SOT | Python生成 | Go生成 | Go手動 | Rust手動 | 一致性 |
|------|----------|-----------|--------|--------|----------|--------|
| FunctionTaskPayload | ✅ | ✅ | ❌語法錯誤 | ⚠️部分 | ⚠️部分 | 🔴 不一致 |
| FindingPayload | ✅ | ✅ | ❌語法錯誤 | ⚠️部分 | ⚠️部分 | 🔴 不一致 |
| Vulnerability | ✅ | ✅ | ✅ | ✅ | ⚠️簡化 | 🟡 基本一致 |
| FunctionTaskTarget | ✅ | ✅ | ❌語法錯誤 | ⚠️簡化 | ⚠️簡化 | 🔴 不一致 |

## 建議的修復方案

### 方案 A: 修復代碼生成器並統一使用 (推薦)

**步驟**:
1. ✅ **修復 `schema_codegen_tool.py`**
   - 修正類型映射邏輯,正確處理嵌套類型
   - 添加單元測試驗證生成的代碼
   - 添加語法檢查(Go: `go fmt`, `go vet`; Rust: `cargo fmt`, `cargo check`)

2. ✅ **重新生成所有 schemas**
   ```bash
   python services/aiva_common/tools/schema_codegen_tool.py --generate-all --validate
   ```

3. ✅ **更新各模組使用生成的 schemas**
   - 移除手動編寫的重複定義
   - 導入並使用生成的 schemas
   - 保留模組專屬的擴展類型

4. ✅ **整合到 CI/CD 流程**
   - Pre-commit hook: 檢查是否有手動修改生成的文件
   - CI 驗證: 確保生成的代碼可編譯
   - 定期重新生成: 保持與 YAML SOT 同步

**優點**:
- 真正實現 Single Source of Truth
- 自動保持跨語言一致性
- 減少維護成本
- 降低人為錯誤

**缺點**:
- 需要重構現有代碼
- 可能影響正在進行的開發

### 方案 B: 保持手動定義並移除生成器 (不推薦)

**步驟**:
1. 移除或廢棄 `schema_codegen_tool.py`
2. 移除生成的 `generated/` 目錄
3. 更新 `core_schema_sot.yaml` 僅作為文檔

**優點**:
- 不影響現有代碼
- 開發者保持靈活性

**缺點**:
- 無法保證跨語言一致性
- 增加維護成本
- 容易出現不一致問題

### 方案 C: 混合方案 - 生成基礎類型,允許擴展 (平衡)

**步驟**:
1. 修復代碼生成器
2. 只生成基礎共享類型(Base Types, Messaging, Tasks, Findings)
3. 各模組可以擴展基礎類型,添加專屬字段
4. 使用組合而非重複定義

**示例**:
```go
// 使用生成的基礎類型
import "aiva_common_go/schemas/generated"

// 模組專屬擴展
type EnhancedFunctionTaskPayload struct {
    generated.FunctionTaskPayload  // 嵌入基礎類型
    ModuleSpecificField string     // 添加專屬字段
}
```

## 行動計劃

### 立即行動 (P0)

1. **修復代碼生成器的類型映射** ⏰ 1-2 天
   - [ ] 修正 `Optional[Dict[str, Any]]` → `map[string]interface{}`
   - [ ] 修正 `Optional[float]` → `*float64`
   - [ ] 添加嵌套類型處理邏輯
   - [ ] 添加生成後的語法驗證

2. **驗證生成的代碼** ⏰ 0.5 天
   - [ ] 重新生成 Python schemas
   - [ ] 重新生成 Go schemas  
   - [ ] 重新生成 Rust schemas
   - [ ] 確保所有生成的代碼可編譯

### 短期行動 (P1)

3. **統一 Go 模組使用生成的 schemas** ⏰ 2-3 天
   - [ ] function_sca_go: 使用生成的 schemas,移除重複定義
   - [ ] function_ssrf_go: 檢查並統一
   - [ ] function_cspm_go: 檢查並統一
   - [ ] function_authn_go: 檢查並統一
   - [ ] 測試所有模組功能正常

4. **統一 Rust 模組使用生成的 schemas** ⏰ 1 天
   - [ ] function_sast_rust: 使用生成的 schemas

### 中期行動 (P2)

5. **整合到開發流程** ⏰ 1-2 天
   - [ ] 添加 pre-commit hook
   - [ ] 添加 CI 檢查
   - [ ] 更新開發文檔
   - [ ] 團隊培訓

6. **文檔更新** ⏰ 1 天
   - [ ] 更新 DEVELOPMENT_STANDARDS.md
   - [ ] 創建 SCHEMA_SOT_GUIDE.md
   - [ ] 添加代碼生成器使用說明

## 技術細節

### schema_codegen_tool.py 需要修復的部分

```python
# 當前的類型映射(有問題)
def _map_type_to_go(self, type_str: str) -> str:
    mapping = self.sot_data['generation_config']['go']['field_mapping']
    return mapping.get(type_str, type_str)  # ❌ 無法處理嵌套類型

# 應該修復為
def _map_type_to_go(self, type_str: str) -> str:
    # 處理 Optional[...]
    if type_str.startswith('Optional['):
        inner = type_str[9:-1]  # 提取內部類型
        mapped = self._map_type_to_go(inner)  # 遞歸映射
        return f'*{mapped}' if not mapped.startswith('*') else mapped
    
    # 處理 Dict[K, V]
    if type_str.startswith('Dict['):
        # 提取鍵值類型
        match = re.match(r'Dict\[(\w+),\s*(\w+)\]', type_str)
        if match:
            key_type = self._map_type_to_go(match.group(1))
            val_type = self._map_type_to_go(match.group(2))
            return f'map[{key_type}]{val_type}'
    
    # 處理 List[T]
    if type_str.startswith('List['):
        inner = type_str[5:-1]
        mapped = self._map_type_to_go(inner)
        return f'[]{mapped}'
    
    # 基本類型映射
    mapping = self.sot_data['generation_config']['go']['field_mapping']
    return mapping.get(type_str, type_str)
```

## 結論

Features 模組存在嚴重的跨語言 Schema 重複定義和不一致問題:

1. **代碼生成器有 bug**: 生成的 Go 代碼包含 Python 語法
2. **重複定義**: 9 個類型在手動和自動生成代碼中重複
3. **架構未實施**: YAML SOT 設計良好,但執行不足

**建議**: 採用**方案 A(修復並統一)**,真正實現 Single Source of Truth 架構,確保跨語言一致性。

**預計工作量**: 5-7 天完成完整修復和統一

**優先級**: **P0 - 緊急** (影響代碼質量和可維護性)
