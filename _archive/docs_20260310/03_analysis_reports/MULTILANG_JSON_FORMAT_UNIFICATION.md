# 多語言分析工具 JSON 格式統一方案

**版本**: v1.0  
**日期**: 2026-01-13  
**目的**: 統一各語言分析工具的 JSON 輸出格式，使 FlowExecutor 能讀取並執行所有語言的分析結果

---

## 📊 當前情況分析

### Python 工具 (標準格式 - 參照基準)

**位置**: `services/core/aiva_core/internal_exploration/python_tools/`

**輸出檔案**: `classification_data.json`

**格式** (v15 - aiva_core 內部模組):
```json
{
  "metadata": {
    "generated_at": "2026-01-13T00:09:27.102504",
    "total_flows": 286,
    "module_distribution": {
      "task_planning": 36,
      "service_backbone": 53,
      "cognitive_core": 75
    },
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["backends", "unified_executor"],
      "full_path": [
        "C:\\D\\fold7\\AIVA-git\\services\\core\\aiva_core\\service_backbone\\storage\\backends.py",
        "C:\\D\\fold7\\AIVA-git\\services\\core\\aiva_core\\task_planning\\unified_executor.py"
      ],
      "func_names": [
        "JSONLBackend.get_experience_samples",
        "ExperienceSample"
      ],
      "length": 2,
      "start": "backends",
      "end": "unified_executor",
      "classifications": [
        {
          "script": "backends",
          "module": "service_backbone",
          "component_type": "程式組件",
          "description": "backends - 功能組件"
        },
        {
          "script": "unified_executor",
          "module": "task_planning",
          "component_type": "程式組件",
          "description": "unified_executor - 功能組件"
        }
      ],
      "modules": ["service_backbone", "task_planning"],
      "primary_module": "task_planning",
      "is_ai_capability": false,
      "cli_command": "python -m services.core.aiva_core.task_planning.unified_executor execute",
      "parameters": [],
      "return_type": "unknown",
      "structured_tags": ["module:task_planning", "type:程式", "length:short"]
    }
  ]
}
```

**格式** (v13 - features_ready 外部模組):
```json
{
  "metadata": {
    "type": "external_modules",
    "total_flows": 235,
    "total_modules": 5,
    "total_languages": 1,
    "generated_at": "2026-01-13T00:07:52.018585"
  },
  "modules": {
    "function_sqli": 36,
    "function_xss": 110,
    "function_idor": 48
  },
  "languages": {
    "Python": 235
  },
  "flows": [
    {
      "id": 1,
      "path": ["BountyHunterManager.add_high_value_target", "HighValueTarget"],
      "file_path": "C:\\D\\fold7\\AIVA-git\\services\\features\\features_ready\\function_sqli\\integration_tools\\bounty_hunter.py",
      "length": 2,
      "func_names": ["BountyHunterManager.add_high_value_target", "HighValueTarget"],
      "function_module": "function_sqli",
      "function_module_desc": "SQL 注入檢測",
      "language": "Python",
      "entry_points": [],
      "has_entry_point": false
    }
  ]
}
```

---

### TypeScript 工具 (當前格式 - 需要修改)

**位置**: `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts`

**輸出檔案**: `analysis_results.json`

**當前格式** (第 748-759 行):
```typescript
const report = {
    summary: {
        total_files: files.length,
        total_funcs: allMeta.length,
        real_connections: stitcher.realConnections.length,
    },
    classification: classResult,
    branch_analysis: branchStats,
    flow_chains: stitcher.realConnections,  // ← 問題：不是 "flows"
    functions: allMeta                      // ← 問題：單個函數列表
};
```

**realConnections 的內容結構**:
```typescript
interface Connection {
    fromScript: string;    // 來源檔案
    fromFunc: string;      // 來源函數
    toScript: string;      // 目標檔案
    toFunc: string;        // 目標函數
    callExpr: string;      // 調用表達式
}
```

**FlowExecutor 需要但當前缺少的欄位**:
- ❌ 沒有 `flows` 鍵（使用 `flow_chains` 和 `functions`）
- ❌ 沒有 `id` 欄位
- ❌ 沒有 `path` 欄位（使用 `fromFunc`, `toFunc`）
- ❌ 沒有 `full_path` 欄位（使用 `fromScript`, `toScript`）
- ❌ 沒有 `classifications` 欄位
- ❌ `metadata` 結構不完整

---

### Go 工具 (當前格式 - 需要修改)

**位置**: `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go`

**輸出檔案**: `analysis_results.json`

**當前格式** (第 765-777 行):
```go
report := map[string]interface{}{
    "summary": map[string]interface{}{
        "total_files": len(stitcher.ScriptNodes),
        "total_funcs": len(allMeta),
        "real_connections": len(stitcher.RealConnections),
        "categories": classResult.Summary,
    },
    "branch_analysis": branchStats,
    "flow_chains": stitcher.RealConnections,  // ← 問題：不是 "flows"
    "functions": allMeta,                      // ← 問題：單個函數列表
}
```

**RealConnections 的內容結構**:
```go
type Connection struct {
    FromScript   string `json:"from_script"`
    FromFunc     string `json:"from_func"`
    ToScript     string `json:"to_script"`
    ToFunc       string `json:"to_func"`
    CallExpr     string `json:"call_expr"`
}
```

**FlowExecutor 需要但當前缺少的欄位**:
- ❌ 沒有 `flows` 鍵
- ❌ 沒有 `id` 欄位
- ❌ 沒有 `path` 欄位
- ❌ 沒有 `full_path` 欄位
- ❌ 沒有 `classifications` 欄位
- ❌ `metadata` 結構不完整

---

### Rust 工具 (當前格式 - 需要修改)

**位置**: `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs`

**輸出檔案**: `analysis_results.json`

**當前格式** (第 722-734 行):
```rust
let final_report = serde_json::json!({
    "summary": {
        "total_files": files.len(),
        "total_funcs": all_meta.len(),
        "real_connections": stitcher.real_connections.len(),
    },
    "classification": class_result,
    "branch_analysis": branch_stats,
    "flow_chains": stitcher.real_connections,  // ← 問題：不是 "flows"
    "functions": all_meta                       // ← 問題：單個函數列表
});
```

**real_connections 的內容結構**:
```rust
struct Connection {
    from_file: String,
    from_func: String,
    to_file: String,
    to_func: String,
    call_expr: String,
}
```

**FlowExecutor 需要但當前缺少的欄位**:
- ❌ 沒有 `flows` 鍵
- ❌ 沒有 `id` 欄位
- ❌ 沒有 `path` 欄位
- ❌ 沒有 `full_path` 欄位
- ❌ 沒有 `classifications` 欄位
- ❌ `metadata` 結構不完整

---

## 二、各語言工具現況分析

### 2.1 TypeScript 工具 (ts2mermaid.ts)

**檔案位置**: `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts`

**核心數據結構** (line 141-146):
```typescript
interface Connection {
  fromScript: string;   // 來源檔案路徑
  fromFunc: string;     // 來源函數名稱
  toScript: string;     // 目標檔案路徑
  toFunc: string;       // 目標函數名稱
  callExpr: string;     // 調用表達式
}
```

**當前輸出格式** (line 748-759):
```typescript
const report = {
    summary: {
        total_files: files.length,
        total_funcs: allMeta.length,
        real_connections: stitcher.realConnections.length,
    },
    classification: classResult,
    branch_analysis: branchStats,
    flow_chains: stitcher.realConnections,  // ← Connection[] 數組
    functions: allMeta
};
```

**輸出檔案**: `analysis_results.json`

**問題**:
- ❌ 沒有 `flows` 欄位 (使用 `flow_chains`)
- ❌ `flow_chains` 中的 Connection 對象沒有 `id`, `path`, `full_path`, `classifications` 等欄位
- ❌ 缺少 `metadata` 統一元數據結構

---

### 2.2 Go 工具 (go2mermaid.go)

**檔案位置**: `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go`

**核心數據結構** (line 88-94):
```go
type Connection struct {
	FromScript   string `json:"from_script"`
	FromFunc     string `json:"from_func"`
	ToScript     string `json:"to_script"`
	ToFunc       string `json:"to_func"`
	CallExpr     string `json:"call_expr"`
}
```

**當前輸出格式** (line 766-776):
```go
report := map[string]interface{}{
    "summary": map[string]interface{}{
        "total_files": len(stitcher.ScriptNodes),
        "total_funcs": len(allMeta),
        "real_connections": len(stitcher.RealConnections),
        "categories": classResult.Summary,
    },
    "branch_analysis": branchStats,
    "flow_chains": stitcher.RealConnections,  // ← []Connection
    "functions": allMeta,
}
```

**輸出檔案**: `analysis_results.json`

**問題**:
- ❌ 沒有 `flows` 欄位 (使用 `flow_chains`)
- ❌ Connection 結構與 FlowExecutor 需求不匹配
- ❌ 缺少統一的 `metadata`

---

### 2.3 Rust 工具 (main.rs)

**檔案位置**: `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs`

**核心數據結構** (line 187-192):
```rust
struct Connection {
    from_script: String,
    from_func: String,
    to_script: String,
    to_func: String,
    call_expr: String,
}
```

**當前輸出格式** (line 726-735):
```rust
let final_report = serde_json::json!({
    "summary": {
        "total_files": files.len(),
        "total_funcs": all_meta.len(),
        "real_connections": stitcher.real_connections.len(),
    },
    "classification": class_result,
    "branch_analysis": branch_stats,
    "flow_chains": stitcher.real_connections,  // ← Vec<Connection>
    "functions": all_meta
});
```

**輸出檔案**: `analysis_results.json`

**問題**:
- ❌ 沒有 `flows` 欄位
- ❌ Connection 結構缺少 FlowExecutor 需要的欄位
- ❌ 缺少 `metadata`

---

## 二之二、FlowExecutor 的要求分析

**初始化時讀取** (第 155-180 行):
```python
def _load_data(self) -> Dict[str, Any]:
    with open(self.json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 使用
self.data = self._load_data()
```

**獲取 Flow** (第 193-198 行):
```python
def get_flow_by_id(self, flow_id: int) -> Optional[Dict[str, Any]]:
    for flow in self.data.get("flows", []):  # ← 必須有 "flows" 鍵
        if flow["id"] == flow_id:             # ← 必須有 "id" 欄位
            return flow
    return None
```

**執行 Flow** (第 540-560 行):
```python
def execute_flow(self, flow_id: int, ...):
    flow = self.get_flow_by_id(flow_id)
    
    # 讀取路徑
    path = flow.get('path', flow.get('original_path', []))  # ← 必須有 "path"
    
    # 遍歷每個步驟
    for idx, step_info in enumerate(flow.get("classifications", [])):  # ← 必須有 "classifications"
        script_name = step_info["script"]
        full_path = flow["full_path"][idx]  # ← 必須有 "full_path"
        
        # 解析模組路徑
        module_path = self._full_path_to_module(full_path)
        # ... 動態導入並執行
```

### FlowExecutor 必須的 JSON 結構

```json
{
  "metadata": {
    "generated_at": "ISO 8601 時間戳",
    "total_flows": 數量,
    "language": "typescript | go | rust | python"
  },
  "flows": [
    {
      "id": 唯一ID (整數),
      "path": ["函數A", "函數B", "函數C"],
      "full_path": ["絕對路徑A", "絕對路徑B", "絕對路徑C"],
      "func_names": ["完整函數名A", "完整函數名B"],
      "length": 流程長度,
      "start": "起點函數",
      "end": "終點函數",
      "classifications": [
        {
          "script": "腳本名稱",
          "module": "所屬模組",
          "component_type": "程式組件",
          "description": "功能描述"
        }
      ],
      "language": "typescript | go | rust"
    }
  ]
}
```

---

## 二之三、修改總結

### 需要做的修改（按優先級排序）

#### 1. **各語言工具輸出格式增強** [高優先級]

**目標**: 在現有輸出基礎上，添加 `flows` 欄位

**三個工具都需要做**:
- ✅ 新增轉換函數 `convertToFlows()` - 將 Connection 轉為 Flow 格式
- ✅ 在輸出 JSON 時添加 `flows` 欄位
- ✅ 添加統一的 `metadata` 欄位
- ✅ **保留**所有現有欄位 (`flow_chains`, `functions`, `summary` 等)

**修改量估算**:
- TypeScript: ~35 行 (新增函數 25 行 + 修改輸出 10 行)
- Go: ~40 行 (新增函數 30 行 + 修改輸出 10 行)
- Rust: ~35 行 (新增函數 25 行 + 修改輸出 10 行)
- **總計**: ~110 行代碼

---

#### 2. **FlowExecutor 多語言支持** [中優先級] - 暫緩

**注意**: 此部分**暫不實施**，因為：
- 當前 aiva_cli_implementation.py 是針對 aiva_core 內部模組
- 未來需要創建**對外版本** CLI (針對 features/scan 模組)
- 等各語言工具修改完成後再處理

**未來需要做的**:
- 創建新的 CLI 執行器用於外部模組
- 支持讀取多份 JSON (Python/TS/Go/Rust)
- 添加語言前綴標識 (flow1, tsflow1, goflow1, rsflow1)

---

#### 3. **測試與驗證** [高優先級]

**需要測試的內容**:
1. 各語言工具能正常輸出新格式 JSON
2. 新增的 `flows` 欄位結構正確
3. 原有的 `flow_chains`, `functions` 等欄位保持不變
4. FlowExecutor 能讀取並解析 `flows` 數據（待多語言支持實施後測試）

---

#### 4. **文檔與規範** [中優先級]

**需要創建的文檔**:
- [ ] `JSON_FORMAT_SPECIFICATION.md` - 統一 JSON 格式規範
- [ ] 各工具的 `DATAFLOW_SPEC.md` - 語言特定的數據流規格
- [ ] 修改日誌與版本記錄

---

## 三、詳細修改步驟

### 修改策略重申

**核心原則 - 增量式添加，不破壞原有格式**：
1. ✅ **保留所有現有字段** - `flow_chains`, `functions`, `summary` 等完全不變
2. ✅ **只添加 FlowExecutor 需要的字段** - 新增 `flows` 數組
3. ✅ **保留各語言工具的分析邏輯** - Stitcher 等完全不動
4. ✅ **只修改最後輸出 JSON 的部分** - 在輸出時添加轉換後的 `flows`

**關鍵理解**：
- ❌ **不是** 統一所有格式 → 刪除 `flow_chains`、`functions`
- ✅ **而是** 添加 FlowExecutor 能讀的 `flows` → 保留 `flow_chains`、`functions`
- 📊 輸出 JSON 會同時包含兩套數據：
  - `flows` - 給 FlowExecutor 用（統一格式）
  - `flow_chains`, `functions` - 給語言工具自己用（原始格式）

**修改範圍**：
- 每個語言工具只需修改 **輸出部分** (~30 行程式碼)
- 新增 1 個轉換函數 (~20 行) - 從 `realConnections` 生成 `flows`
- 修改 JSON 輸出結構 (~10 行) - 添加 `flows` 字段

---

### TypeScript 工具修改方案

**檔案**: `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts`

**修改位置**: 第 735-765 行（main 函數的輸出部分）

**修改內容**:

#### 步驟1: 新增轉換函數 (插入在第 735 行之前)

```typescript
/**
 * 將 realConnections 轉換成 Python FlowExecutor 需要的 flows 格式
 */
function convertConnectionsToFlows(connections: Connection[]): any[] {
    return connections.map((conn, idx) => ({
        id: idx + 1,
        path: [conn.fromFunc, conn.toFunc],
        full_path: [conn.fromScript, conn.toScript],
        func_names: [conn.fromFunc, conn.toFunc],
        length: 2,
        start: conn.fromFunc,
        end: conn.toFunc,
        classifications: [
            {
                script: conn.fromFunc,
                module: "typescript_module",
                component_type: "程式組件",
                description: `${conn.fromFunc} - TypeScript 功能`
            },
            {
                script: conn.toFunc,
                module: "typescript_module",
                component_type: "程式組件",
                description: `${conn.toFunc} - TypeScript 功能`
            }
        ],
        language: "typescript",
        cli_command: `ts-node ${conn.toScript}`,
        structured_tags: ["language:typescript", "type:程式"]
    }));
}
```

#### 步驟2: 修改輸出 JSON 結構 (修改第 748-759 行)

**修改前**:
```typescript
    // Full JSON Report
    const report = {
        summary: {
            total_files: files.length,
            total_funcs: allMeta.length,
            real_connections: stitcher.realConnections.length,
        },
        classification: classResult,
        branch_analysis: branchStats,
        flow_chains: stitcher.realConnections,
        functions: allMeta
    };
```

**修改後**:
```typescript
    // Full JSON Report - 統一格式，相容 Python FlowExecutor
    const flows = convertConnectionsToFlows(stitcher.realConnections);
    
    const report = {
        metadata: {
            tool: "ts2mermaid",
            version: "2.0",
            language: "typescript",
            generated_at: new Date().toISOString(),
            total_flows: flows.length,
            total_files: files.length,
            schema_version: "3.3",
            ai_compatible: true
        },
        // ✅ 新增：FlowExecutor 需要的統一格式
        flows: flows,
        
        // ✅ 保留：TypeScript 工具原有的所有字段
        summary: {
            total_files: files.length,
            total_funcs: allMeta.length,
            real_connections: stitcher.realConnections.length,
        },
        classification: classResult,
        branch_analysis: branchStats,
        flow_chains: stitcher.realConnections,
        functions: allMeta
    };
```

**重要說明**：
- ✅ `flows` - 新增，給 FlowExecutor 用
- ✅ `summary`, `classification`, `branch_analysis`, `flow_chains`, `functions` - **保留**，TypeScript 工具自己用
- 📊 不刪除任何現有字段，只是**添加** `flows`

---

### Go 工具修改方案

**檔案**: `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go`

**修改位置**: 第 745-785 行（main 函數的輸出部分）

**修改內容**:

#### 步驟1: 新增轉換函數 (插入在第 745 行之前)

```go
// convertConnectionsToFlows 將 RealConnections 轉換成 Python FlowExecutor 需要的 flows 格式
func convertConnectionsToFlows(connections []Connection) []map[string]interface{} {
	flows := make([]map[string]interface{}, len(connections))
	
	for i, conn := range connections {
		flows[i] = map[string]interface{}{
			"id":         i + 1,
			"path":       []string{conn.FromFunc, conn.ToFunc},
			"full_path":  []string{conn.FromScript, conn.ToScript},
			"func_names": []string{conn.FromFunc, conn.ToFunc},
			"length":     2,
			"start":      conn.FromFunc,
			"end":        conn.ToFunc,
			"classifications": []map[string]interface{}{
				{
					"script":         conn.FromFunc,
					"module":         "go_module",
					"component_type": "程式組件",
					"description":    fmt.Sprintf("%s - Go 功能", conn.FromFunc),
				},
				{
					"script":         conn.ToFunc,
					"module":         "go_module",
					"component_type": "程式組件",
					"description":    fmt.Sprintf("%s - Go 功能", conn.ToFunc),
				},
			},
			"language":        "go",
			"cli_command":     fmt.Sprintf("go run %s", conn.ToScript),
			"structured_tags": []string{"language:go", "type:程式"},
		}
	}
	
	return flows
}
```

#### 步驟2: 修改輸出 JSON 結構 (修改第 765-777 行)

**修改前**:
```go
	// C. 完整 JSON 報告
	report := map[string]interface{}{
		"summary": map[string]interface{}{
			"total_files": len(stitcher.ScriptNodes),
			"total_funcs": len(allMeta),
			"real_connections": len(stitcher.RealConnections),
			"categories": classResult.Summary,
		},
		"branch_analysis": branchStats,
		"flow_chains": stitcher.RealConnections,
		"functions": allMeta,
	}
```

**修改後**:
```go
	// C. 完整 JSON 報告 - 統一格式，相容 Python FlowExecutor
	flows := convertConnectionsToFlows(stitcher.RealConnections)
	
	report := map[string]interface{}{
		"metadata": map[string]interface{}{
			"tool":           "go2mermaid",
			"version":        "2.0",
			"language":       "go",
			"generated_at":   time.Now().Format(time.RFC3339),
			"total_flows":    len(flows),
			"total_files":    len(stitcher.ScriptNodes),
			"schema_version": "3.3",
			"ai_compatible":  true,
		},
		// ✅ 新增：FlowExecutor 需要的統一格式
		"flows": flows,
		
		// ✅ 保留：Go 工具原有的所有字段
		"summary": map[string]interface{}{
			"total_files":      len(stitcher.ScriptNodes),
			"total_funcs":      len(allMeta),
			"real_connections": len(stitcher.RealConnections),
			"categories":       classResult.Summary,
		},
		"branch_analysis": branchStats,
		"flow_chains":     stitcher.RealConnections,
		"functions":       allMeta,
	}
```

**重要說明**：
- ✅ `flows` - 新增，給 FlowExecutor 用
- ✅ `summary`, `branch_analysis`, `flow_chains`, `functions` - **保留**，Go 工具自己用

---

### Rust 工具修改方案

**檔案**: `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs`

**修改位置**: 第 690-750 行（main 函數的輸出部分）

**修改內容**:

#### 步驟1: 新增轉換函數 (插入在第 690 行之前)

```rust
/// 將 real_connections 轉換成 Python FlowExecutor 需要的 flows 格式
fn convert_connections_to_flows(connections: &[Connection]) -> Vec<serde_json::Value> {
    connections
        .iter()
        .enumerate()
        .map(|(i, conn)| {
            serde_json::json!({
                "id": i + 1,
                "path": [&conn.from_func, &conn.to_func],
                "full_path": [&conn.from_file, &conn.to_file],
                "func_names": [&conn.from_func, &conn.to_func],
                "length": 2,
                "start": &conn.from_func,
                "end": &conn.to_func,
                "classifications": [
                    {
                        "script": &conn.from_func,
                        "module": "rust_module",
                        "component_type": "程式組件",
                        "description": format!("{} - Rust 功能", &conn.from_func)
                    },
                    {
                        "script": &conn.to_func,
                        "module": "rust_module",
                        "component_type": "程式組件",
                        "description": format!("{} - Rust 功能", &conn.to_func)
                    }
                ],
                "language": "rust",
                "cli_command": format!("cargo run --bin {}", &conn.to_func),
                "structured_tags": ["language:rust", "type:程式"]
            })
        })
        .collect()
}
```

#### 步驟2: 修改輸出 JSON 結構 (修改第 722-734 行)

**修改前**:
```rust
    // C. 完整 JSON 報告
    let final_report = serde_json::json!({
        "summary": {
            "total_files": files.len(),
            "total_funcs": all_meta.len(),
            "real_connections": stitcher.real_connections.len(),
        },
        "classification": class_result,
        "branch_analysis": branch_stats,
        "flow_chains": stitcher.real_connections,
        "functions": all_meta
    });
```

**修改後**:
```rust
    // C. 完整 JSON 報告 - 統一格式，相容 Python FlowExecutor
    let flows = convert_connections_to_flows(&stitcher.real_connections);
    
    let final_report = serde_json::json!({
        "metadata": {
            "tool": "rs2mermaid",
            "version": "2.0",
            "language": "rust",
            "generated_at": chrono::Utc::now().to_rfc3339(),
            "total_flows": flows.len(),
            "total_files": files.len(),
            "schema_version": "3.3",
            "ai_compatible": true
        },
        "flows": flows,
        // 保留原始資料供參考和除錯
        "_original": {
            "summary": {
                "total_files": files.len(),
                "total_funcs": all_meta.len(),
                "real_connections": stitcher.real_connections.len(),
            },
            "classification": class_result,
            "branch_analysis": branch_stats,
            "flow_chains": stitcher.real_connections,
            "functions": all_meta
        }
    });
```

**需要新增的依賴** (Cargo.toml):
```toml
[dependencies]
chrono = "0.4"  # 新增這行
# ... 其他依賴
```

---

## 📝 修改後的統一格式範例

### 所有語言工具統一輸出格式

```json
{
  "metadata": {
    "tool": "ts2mermaid | go2mermaid | rs2mermaid",
    "version": "2.0",
    "language": "typescript | go | rust",
    "generated_at": "2026-01-13T10:30:00Z",
    "total_flows": 150,
    "total_files": 45,
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["detect_xss", "scan_url"],
      "full_path": [
        "C:/path/to/detector.ts",
        "C:/path/to/scanner.ts"
      ],
      "func_names": ["detect_xss", "scan_url"],
      "length": 2,
      "start": "detect_xss",
      "end": "scan_url",
      "classifications": [
        {
          "script": "detect_xss",
          "module": "typescript_module",
          "component_type": "程式組件",
          "description": "detect_xss - TypeScript 功能"
        },
        {
          "script": "scan_url",
          "module": "typescript_module",
          "component_type": "程式組件",
          "description": "scan_url - TypeScript 功能"
        }
      ],
      "language": "typescript",
      "cli_command": "ts-node C:/path/to/scanner.ts",
      "structured_tags": ["language:typescript", "type:程式"]
    }
  ],
  // ✅ 保留：所有原有字段完整保留
  "summary": {
    "total_files": 45,
    "total_funcs": 320,
    "real_connections": 150
  },
  "classification": { ... },
  "branch_analysis": { ... },
  "flow_chains": [ ... ],      // ← 保留原始 Connection 數組
  "functions": [ ... ]          // ← 保留原始函數元數據
}
```

**重要**: 
- ✅ `flows` 是新增的，專門給 FlowExecutor 用
- ✅ `summary`, `classification`, `flow_chains`, `functions` 等全部保留
- ✅ 不刪除任何現有字段
- ✅ 各語言工具可繼續使用自己的原始數據結構

---

## 五、實施檢查清單

### Phase 1: TypeScript 工具修改

- [ ] **步驟 1**: 在 line 735 前添加 `convertConnectionsToFlows()` 函數
  - [ ] 實現 Connection → Flow 的轉換邏輯
  - [ ] 確保所有必要欄位都包含 (id, path, full_path, classifications)
  
- [ ] **步驟 2**: 修改 line 748-759 的輸出結構
  - [ ] 添加 `metadata` 欄位
  - [ ] 添加 `flows` 欄位 (調用轉換函數)
  - [ ] **保留**所有現有欄位 (summary, classification, branch_analysis, flow_chains, functions)
  
- [ ] **步驟 3**: 測試驗證
  - [ ] 運行工具生成 analysis_results.json
  - [ ] 檢查 JSON 包含 `flows` 欄位
  - [ ] 檢查原有欄位仍然存在
  - [ ] 驗證 flows 數組結構正確

---

### Phase 2: Go 工具修改

- [ ] **步驟 1**: 在 line 745 前添加 `convertConnectionsToFlows()` 函數
  - [ ] 實現轉換邏輯
  - [ ] 處理 Go 特定的類型轉換
  
- [ ] **步驟 2**: 修改 line 766-776 的輸出結構
  - [ ] 添加 `metadata`
  - [ ] 添加 `flows`
  - [ ] 保留所有現有欄位
  - [ ] 確認需要 `import "time"` (用於 timestamp)
  
- [ ] **步驟 3**: 測試驗證
  - [ ] 運行 Go 工具
  - [ ] 檢查輸出 JSON 格式
  - [ ] 驗證數據完整性

---

### Phase 3: Rust 工具修改

- [ ] **步驟 1**: 在 line 690 前添加 `convert_connections_to_flows()` 函數
  - [ ] 實現轉換邏輯
  - [ ] 使用 `serde_json::json!` 宏構建
  
- [ ] **步驟 2**: 修改 line 726-735 的輸出結構
  - [ ] 添加 `metadata`
  - [ ] 添加 `flows`
  - [ ] 保留所有現有欄位
  - [ ] 在 Cargo.toml 添加 `chrono = "0.4"` 依賴
  
- [ ] **步驟 3**: 測試驗證
  - [ ] 運行 Rust 工具
  - [ ] 檢查輸出
  - [ ] 驗證編譯無錯誤

---

### Phase 4: 整合測試（待 FlowExecutor 多語言支持後進行）

- [ ] FlowExecutor 能讀取 TypeScript JSON
- [ ] FlowExecutor 能讀取 Go JSON
- [ ] FlowExecutor 能讀取 Rust JSON
- [ ] FlowExecutor 能讀取 Python JSON (現有)
- [ ] 測試 dry-run 模式
- [ ] 測試實際執行

---

## 六、注意事項與風險

### 兼容性考慮

1. **向後兼容**: 保留所有原有字段，確保現有依賴這些字段的代碼不受影響
2. **漸進式升級**: 可以先只修改一個工具測試，確認無誤後再修改其他
3. **數據備份**: 修改前建議備份現有的分析結果

### 潛在問題

1. **路徑格式**: Windows 絕對路徑 (C:\...) vs Unix 路徑，需要確保一致性
2. **編碼問題**: 確保所有工具都使用 UTF-8 編碼
3. **JSON 大小**: 添加 flows 後，JSON 文件會變大，需注意性能

### 建議的實施順序

1. ✅ **先修改 TypeScript 工具** (最常用，問題最容易發現)
2. ✅ **再修改 Go 工具** (類似 TypeScript)
3. ✅ **最後修改 Rust 工具** (相對獨立)
4. ⏳ **創建對外版本 CLI** (新項目，針對 features/scan 模組)

---

## 七、成功標準

### 修改成功的判斷標準

1. ✅ 各工具能正常運行並生成 JSON
2. ✅ JSON 包含 `metadata` 和 `flows` 欄位
3. ✅ `flows` 數組中每個元素包含所有必要欄位 (id, path, full_path, classifications)
4. ✅ 所有原有欄位 (summary, classification, flow_chains, functions) 完整保留
5. ✅ JSON 格式有效，能被正確解析
6. ✅ 不影響工具的其他功能 (Mermaid 圖生成、CLI 文檔生成等)

### 預期結果

完成修改後，每個語言工具會輸出包含兩套數據的 JSON：
- **統一格式** (`flows`) - 供 FlowExecutor 執行使用
- **原始格式** (`flow_chains`, `functions` 等) - 供工具自己分析使用

這樣既滿足了統一執行的需求，又保持了各工具的特性和靈活性。

---

## 附錄：快速參考

### 文件位置
- TypeScript 工具: `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts`
- Go 工具: `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go`  
- Rust 工具: `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs`
- FlowExecutor: `services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py`

### 修改行數參考
- TypeScript: line 735 (添加函數), line 748-759 (修改輸出)
- Go: line 745 (添加函數), line 766-776 (修改輸出)
- Rust: line 690 (添加函數), line 726-735 (修改輸出)

### 關鍵欄位
必須包含的 Flow 欄位:
- `id` (整數)
- `path` (字符串數組)
- `full_path` (字符串數組)
- `classifications` (對象數組)
- `length` (整數)
- `language` (字符串)

---

**最後更新**: 2026-01-13  
**狀態**: 待實施
  }
}
```

---

## ✅ 驗證檢查清單

### 修改完成後的驗證步驟

#### 1. TypeScript 工具驗證
```bash
cd services/core/aiva_core/internal_exploration/typescript_tools
node ts2mermaid.ts --input ./test_code --output ./test_output
```

**檢查**:
- [ ] 輸出檔案包含 `metadata` 欄位
- [ ] 輸出檔案包含 `flows` 欄位（不是 `flow_chains`）
- [ ] 每個 flow 包含 `id`, `path`, `full_path`, `classifications`
- [ ] `metadata.language` 為 "typescript"

#### 2. Go 工具驗證
```bash
cd services/core/aiva_core/internal_exploration/go_tools
go run go2mermaid.go --input ./test_code --output ./test_output
```

**檢查**:
- [ ] 輸出檔案包含 `metadata` 欄位
- [ ] 輸出檔案包含 `flows` 欄位
- [ ] 每個 flow 包含必要欄位
- [ ] `metadata.language` 為 "go"

#### 3. Rust 工具驗證
```bash
cd services/core/aiva_core/internal_exploration/rust_tools
cargo run -- --input=./test_code --output=./test_output
```

**檢查**:
- [ ] 輸出檔案包含 `metadata` 欄位
- [ ] 輸出檔案包含 `flows` 欄位
- [ ] 每個 flow 包含必要欄位
- [ ] `metadata.language` 為 "rust"

#### 4. FlowExecutor 整合驗證
```python
cd services/core/aiva_core/internal_exploration/python_tools
python aiva_cli_implementation.py --data ../typescript_tools/test_output/analysis_results.json --list
```

**檢查**:
- [ ] FlowExecutor 能成功載入 TypeScript 的 JSON
- [ ] FlowExecutor 能成功載入 Go 的 JSON
- [ ] FlowExecutor 能成功載入 Rust 的 JSON
- [ ] 能列出所有 flows
- [ ] 能執行 dry-run 測試

---

## 📚 各語言工具數據流說明文檔

為了讓 AI 更好地操作各語言工具，需要為每個工具建立獨立的說明文檔。

### 建立的文檔清單

1. **TypeScript 工具數據流說明**
   - 檔案: `services/core/aiva_core/internal_exploration/typescript_tools/DATAFLOW_SPEC.md`
   - 內容: TypeScript 工具的數據流分析方法、輸出格式、使用範例

2. **Go 工具數據流說明**
   - 檔案: `services/core/aiva_core/internal_exploration/go_tools/DATAFLOW_SPEC.md`
   - 內容: Go 工具的數據流分析方法、輸出格式、使用範例

3. **Rust 工具數據流說明**
   - 檔案: `services/core/aiva_core/internal_exploration/rust_tools/DATAFLOW_SPEC.md`
   - 內容: Rust 工具的數據流分析方法、輸出格式、使用範例

4. **統一格式規範**
   - 檔案: `docs/03_analysis_reports/JSON_FORMAT_SPECIFICATION.md`
   - 內容: 所有語言工具必須遵守的 JSON 格式規範

---

## 🎯 實施順序

### 階段1: 修改各語言工具輸出格式 (預計 2-3 小時)

1. ✅ 修改 TypeScript 工具 (ts2mermaid.ts)
   - 新增 convertConnectionsToFlows 函數
   - 修改 JSON 輸出結構
   - 測試驗證

2. ✅ 修改 Go 工具 (go2mermaid.go)
   - 新增 convertConnectionsToFlows 函數
   - 修改 JSON 輸出結構
   - 新增 time import
   - 測試驗證

3. ✅ 修改 Rust 工具 (main.rs)
   - 新增 convert_connections_to_flows 函數
   - 修改 JSON 輸出結構
   - 新增 chrono 依賴
   - 測試驗證

### 階段2: 建立各語言工具說明文檔 (預計 1-2 小時)

1. ✅ 建立 TypeScript 工具數據流說明
2. ✅ 建立 Go 工具數據流說明
3. ✅ 建立 Rust 工具數據流說明
4. ✅ 建立統一格式規範文檔

### 階段3: 整合測試 (預計 1 小時)

1. ✅ 使用各語言工具分析測試代碼
2. ✅ 驗證 JSON 格式正確性
3. ✅ 測試 FlowExecutor 讀取各語言 JSON
4. ✅ 執行跨語言整合測試

---

## 🔄 後續擴展計劃

完成格式統一後，下一步可以：

1. **修改 FlowExecutor 支持多語言執行**
   - 根據 `flow.language` 選擇執行策略
   - 實現 TypeScript/Go/Rust 的執行邏輯

2. **實現歷史版本追蹤**
   - 各語言工具支持差異比對
   - 生成版本變更報告

3. **實現能力查詢介面**
   - 各語言工具支持本地查詢
   - 統一查詢 API

---

## 📞 聯絡與確認

**確認點**:

1. ✅ 是否同意只修改輸出部分，保留分析邏輯不動？
2. ✅ 是否同意使用 `_original` 欄位保留原始資料？
3. ✅ 是否同意統一使用 `flows` 而不是 `flow_chains`？
4. ✅ 是否同意新增 `metadata` 欄位？
5. ✅ 是否需要為每個語言工具建立獨立的說明文檔？

**如果以上都確認無誤，我將開始執行修改。**

---

## 🔄 更新記錄

| 日期 | 版本 | 更新內容 | 作者 |
|------|------|---------|------|
| 2026-01-13 | v1.0 | 建立統一方案文檔 | GitHub Copilot |
