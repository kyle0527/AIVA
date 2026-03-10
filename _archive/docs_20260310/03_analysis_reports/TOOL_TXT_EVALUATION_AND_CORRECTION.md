# 工具.txt 文檔評估與修正建議

**版本**: v1.0  
**日期**: 2026-01-13  
**目的**: 評估外部工具建議文檔的正確性，並提出修正方案

---

## 🎯 架構理解修正

### ✅ 正確的架構

**用戶確認的架構**：
```
各語言工具獨立分析 → 輸出統一格式JSON → Python FlowExecutor 統一讀取並執行

Rust 工具分析 function_crypto  → crypto_analysis.json     ┐
Python 工具分析 function_xss    → xss_analysis.json       ├→ aiva_cli_implementation.py
Go 工具分析某個Go模組           → go_module_analysis.json  ┘   (FlowExecutor 統一讀取執行)
TS 工具分析某個TS模組           → ts_module_analysis.json  ┘
```

**關鍵點**：
- ✅ 各語言工具只負責**代碼分析**
- ✅ 各語言工具輸出**統一格式的 JSON**
- ✅ **只有 Python** 的 `aiva_cli_implementation.py` 負責執行
- ❌ **不需要**每個語言都實現執行器

---

## ❌ 工具.txt 文檔的錯誤建議

### 錯誤1: 建議各語言實現執行器

**原文建議**：
```
TypeScript: 新增 ts_cli_runner.ts (執行與參數傳遞) ★重點
Go:        新增 runner/main.go (手動維護 switch-case) ★重點  
Rust:      新增 bin/runner.rs (手動維護 match) ★重點
```

**為什麼錯誤**：
1. **架構不符**：用戶的架構是 Python 統一執行，不是各語言獨立執行
2. **重複開發**：Python 已有成熟的 FlowExecutor，無需重複實現
3. **維護成本高**：每種語言都維護執行邏輯會導致不一致
4. **動態語言優勢**：Python 的動態導入能力最適合做執行層

**正確做法**：
```
❌ 各語言實現執行器
✅ 只需要 Python 的 aiva_cli_implementation.py 整合各語言 JSON
```

---

## ✅ 正確的優先級排序

### 修正後的優先級

| 優先級 | 任務 | 說明 | 工具.txt原排序 |
|--------|------|------|--------------|
| **★★★ 最高** | **統一JSON輸出格式** | 確保所有語言工具輸出符合 FlowExecutor 要求 | ❌ 未提及 |
| **★★ 高** | 歷史與差異比對 | 各工具支持版本追蹤和 diff 報告 | ✅ 優先級2 |
| **★ 中** | 能力查詢介面 | 各工具支持本地快速查詢 | ✅ 優先級3 |
| **❌ 不需要** | 各語言執行器 | ~~各語言實現 cli_runner~~ | ❌ 原優先級1 |

---

## 📊 FlowExecutor 需要的 JSON 格式

### 當前 Python 工具輸出格式 (v15 - aiva_core)

```json
{
  "metadata": {
    "generated_at": "2026-01-13T00:09:27.102504",
    "total_flows": 286,
    "module_distribution": { "task_planning": 36, "service_backbone": 53, ... },
    "component_type_distribution": { "程式組件": 236, "AI組件": 19, ... },
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["backends", "unified_executor"],
      "full_path": [
        "C:\\...\\backends.py",
        "C:\\...\\unified_executor.py"
      ],
      "func_names": ["JSONLBackend.get_experience_samples", "ExperienceSample"],
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
      "component_types": ["程式組件", "程式組件"],
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

### 當前 Python 工具輸出格式 (v13 - features_ready 外部模組)

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
    "function_idor": 48,
    "function_bizlogic": 4,
    "function_ssrf": 37
  },
  "languages": { "Python": 235 },
  "flows": [
    {
      "id": 1,
      "path": ["BountyHunterManager.add_high_value_target", "HighValueTarget"],
      "file_path": "C:\\...\\bounty_hunter.py",
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

### 當前其他語言工具輸出格式 (TS/Go/Rust)

```json
{
  "summary": {
    "total_files": 數量,
    "total_funcs": 數量,
    "real_connections": 數量
  },
  "classification": { /* 分類統計 */ },
  "branch_analysis": { /* 分支分析 */ },
  "flow_chains": [ /* 跨檔案連接 */ ],
  "functions": [
    {
      "function_name": "名稱",
      "module": "模組",
      "category": "分類",
      "description": "描述",
      "inputs": [],
      "outputs": []
    }
  ]
}
```

### ❌ **格式不兼容問題**

| 欄位 | FlowExecutor需要 | 當前TS/Go/Rust輸出 | 狀態 |
|------|-----------------|-------------------|------|
| `flows[]` | ✅ 必須 | ❌ 沒有，使用 `functions[]` | **不兼容** |
| `flows[].id` | ✅ 必須 | ❌ 沒有 | **不兼容** |
| `flows[].path` | ✅ 必須 | ❌ 沒有 | **不兼容** |
| `flows[].full_path` | ✅ 需要 | ❌ 沒有 | **不兼容** |
| `flows[].func_names` | ✅ 需要 | ❌ 沒有 | **不兼容** |
| `flows[].classifications` | ✅ 需要 | ❌ 沒有 | **不兼容** |
| `flows[].cli_command` | ✅ 需要 | ❌ 沒有 | **不兼容** |
| `metadata.generated_at` | ✅ 需要 | ❌ 沒有 | **不兼容** |

**結論**：當前其他語言工具的輸出格式**完全不能被 FlowExecutor 使用**！

---

## 🔧 正確的實施計劃

### 階段1: 統一 JSON 輸出格式 (★★★ 最高優先級)

**目標**：讓 TS/Go/Rust 工具輸出與 Python 相同的格式

#### 1.1 定義統一的 Flow 結構

**必須包含的欄位**：
```json
{
  "metadata": {
    "tool": "rs2mermaid | go2mermaid | ts2mermaid",
    "language": "rust | go | typescript",
    "generated_at": "ISO 8601 時間",
    "total_flows": 數量,
    "schema_version": "3.3"
  },
  "flows": [
    {
      "id": 唯一ID,
      "path": ["函數A", "函數B", "函數C"],
      "full_path": ["絕對路徑A", "絕對路徑B", "絕對路徑C"],
      "func_names": ["完整函數名A", "完整函數名B", "完整函數名C"],
      "length": 流程長度,
      "start": "起點函數",
      "end": "終點函數",
      "classifications": [
        {
          "script": "腳本名稱",
          "module": "所屬模組",
          "component_type": "程式組件 | AI組件",
          "description": "功能描述"
        }
      ],
      "modules": ["模組列表"],
      "primary_module": "主要模組",
      "is_ai_capability": true/false,
      "cli_command": "執行指令",
      "parameters": ["參數列表"],
      "return_type": "返回類型",
      "structured_tags": ["標籤列表"]
    }
  ]
}
```

#### 1.2 修改 TypeScript 工具

**文件**: `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts`

**修改位置**: 740-770行的輸出邏輯

**修改前**：
```typescript
const report = {
    summary: { total_files, total_funcs, real_connections },
    classification: classResult,
    branch_analysis: branchStats,
    flow_chains: stitcher.realConnections,
    functions: allMeta
};
```

**修改後**：
```typescript
const report = {
    metadata: {
        tool: "ts2mermaid",
        language: "typescript",
        generated_at: new Date().toISOString(),
        total_flows: stitcher.realConnections.length,
        schema_version: "3.3",
        ai_compatible: true
    },
    flows: convertToFlowFormat(stitcher.realConnections, allMeta),
    // 保留原有數據作為補充
    _legacy: {
        summary: { total_files, total_funcs, real_connections },
        classification: classResult,
        branch_analysis: branchStats
    }
};

function convertToFlowFormat(connections, metadata): any[] {
    return connections.map((conn, idx) => ({
        id: idx + 1,
        path: [conn.from_function, conn.to_function],
        full_path: [conn.from_file, conn.to_file],
        func_names: [conn.from_function, conn.to_function],
        length: 2,
        start: conn.from_function,
        end: conn.to_function,
        classifications: [
            { script: conn.from_function, module: "typescript_module", component_type: "程式組件" },
            { script: conn.to_function, module: "typescript_module", component_type: "程式組件" }
        ],
        cli_command: `ts-node ${conn.to_file}`,
        structured_tags: ["language:typescript", "type:程式"]
    }));
}
```

#### 1.3 修改 Go 工具

**文件**: `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go`

**修改位置**: 706-785行的輸出邏輯

**修改前**：
```go
report := map[string]interface{}{
    "summary": map[string]interface{}{
        "total_files": len(stitcher.ScriptNodes),
        "total_funcs": len(allMeta),
        "real_connections": len(stitcher.RealConnections),
    },
    "branch_analysis": branchStats,
    "flow_chains": stitcher.RealConnections,
    "functions": allMeta,
}
```

**修改後**：
```go
flows := convertToFlowFormat(stitcher.RealConnections, allMeta)

report := map[string]interface{}{
    "metadata": map[string]interface{}{
        "tool": "go2mermaid",
        "language": "go",
        "generated_at": time.Now().Format(time.RFC3339),
        "total_flows": len(flows),
        "schema_version": "3.3",
        "ai_compatible": true,
    },
    "flows": flows,
    "_legacy": map[string]interface{}{
        "summary": map[string]interface{}{
            "total_files": len(stitcher.ScriptNodes),
            "total_funcs": len(allMeta),
        },
        "branch_analysis": branchStats,
    },
}

func convertToFlowFormat(connections []Connection, metadata []FlowMetadata) []map[string]interface{} {
    flows := make([]map[string]interface{}, len(connections))
    for i, conn := range connections {
        flows[i] = map[string]interface{}{
            "id": i + 1,
            "path": []string{conn.FromFunc, conn.ToFunc},
            "full_path": []string{conn.FromScript, conn.ToScript},
            "func_names": []string{conn.FromFunc, conn.ToFunc},
            "length": 2,
            "start": conn.FromFunc,
            "end": conn.ToFunc,
            "classifications": []map[string]interface{}{
                {"script": conn.FromFunc, "module": "go_module", "component_type": "程式組件"},
                {"script": conn.ToFunc, "module": "go_module", "component_type": "程式組件"},
            },
            "cli_command": fmt.Sprintf("go run %s", conn.ToScript),
            "structured_tags": []string{"language:go", "type:程式"},
        }
    }
    return flows
}
```

#### 1.4 修改 Rust 工具

**文件**: `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs`

**修改位置**: 630-750行的輸出邏輯

**修改前**：
```rust
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

**修改後**：
```rust
let flows = convert_to_flow_format(&stitcher.real_connections, &all_meta);

let final_report = serde_json::json!({
    "metadata": {
        "tool": "rs2mermaid",
        "language": "rust",
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "total_flows": flows.len(),
        "schema_version": "3.3",
        "ai_compatible": true
    },
    "flows": flows,
    "_legacy": {
        "summary": {
            "total_files": files.len(),
            "total_funcs": all_meta.len(),
        },
        "classification": class_result,
        "branch_analysis": branch_stats
    }
});

fn convert_to_flow_format(connections: &[Connection], metadata: &[FlowMetadata]) -> Vec<serde_json::Value> {
    connections.iter().enumerate().map(|(i, conn)| {
        serde_json::json!({
            "id": i + 1,
            "path": [&conn.from_func, &conn.to_func],
            "full_path": [&conn.from_file, &conn.to_file],
            "func_names": [&conn.from_func, &conn.to_func],
            "length": 2,
            "start": &conn.from_func,
            "end": &conn.to_func,
            "classifications": [
                {"script": &conn.from_func, "module": "rust_module", "component_type": "程式組件"},
                {"script": &conn.to_func, "module": "rust_module", "component_type": "程式組件"}
            ],
            "cli_command": format!("cargo run --bin {}", &conn.to_func),
            "structured_tags": ["language:rust", "type:程式"]
        })
    }).collect()
}
```

### 階段2: 修改 Python FlowExecutor 整合多語言 JSON (★★★ 最高優先級)

**目標**：讓 `aiva_cli_implementation.py` 能夠讀取並執行所有語言的 JSON

#### 2.1 修改 FlowExecutor 初始化

**文件**: `services/core/aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py`

**修改位置**: 106-180行

**新增功能**：
```python
class FlowExecutor:
    def __init__(self, json_paths: Optional[List[str]] = None, auto_discover: bool = True):
        """
        初始化 FlowExecutor，支持多語言 JSON 整合
        
        Args:
            json_paths: JSON 檔案路徑列表，若為 None 則自動發現
            auto_discover: 是否自動掃描並載入所有語言的分析結果
        """
        self.flows_by_language = {}  # 按語言分類的 flows
        self.all_flows = []          # 合併後的所有 flows
        
        if auto_discover:
            json_paths = self._discover_analysis_results()
        
        if not json_paths:
            json_paths = [self._get_default_python_json()]
        
        for json_path in json_paths:
            self._load_language_json(json_path)
        
        self._merge_all_flows()
    
    def _discover_analysis_results(self) -> List[str]:
        """
        自動掃描並發現所有語言的分析結果
        
        掃描位置:
        - services/integration/data/internal_exploration/analysis_history/v*/classification_data.json (Python內部)
        - services/integration/data/internal_exploration/analysis_history/v*/external_modules.json (Python外部)
        - services/integration/data/rust_analysis/latest/analysis_results.json (Rust)
        - services/integration/data/go_analysis/latest/analysis_results.json (Go)
        - services/integration/data/typescript_analysis/latest/analysis_results.json (TypeScript)
        """
        json_files = []
        
        # Python 分析結果
        analysis_history_dir = SERVICES_ROOT / "integration" / "data" / "internal_exploration" / "analysis_history"
        if analysis_history_dir.exists():
            for version_dir in sorted(analysis_history_dir.iterdir(), reverse=True):
                if version_dir.is_dir():
                    classification_json = version_dir / "classification_data.json"
                    if classification_json.exists():
                        json_files.append(str(classification_json))
                        break  # 只取最新版本
        
        # 其他語言分析結果
        for lang in ["rust_analysis", "go_analysis", "typescript_analysis"]:
            lang_dir = SERVICES_ROOT / "integration" / "data" / lang / "latest"
            if lang_dir.exists():
                analysis_json = lang_dir / "analysis_results.json"
                if analysis_json.exists():
                    json_files.append(str(analysis_json))
        
        return json_files
    
    def _load_language_json(self, json_path: str):
        """載入單一語言的 JSON 並分類儲存"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            language = metadata.get("language", "python")  # 預設 Python
            tool = metadata.get("tool", "unknown")
            
            flows = data.get("flows", [])
            
            if language not in self.flows_by_language:
                self.flows_by_language[language] = {
                    "tool": tool,
                    "flows": [],
                    "metadata": metadata
                }
            
            self.flows_by_language[language]["flows"].extend(flows)
            
            print(f"[Info] 載入 {language} 分析結果: {len(flows)} flows (來源: {Path(json_path).name})")
            
        except Exception as e:
            print(f"[Warning] 無法載入 {json_path}: {e}")
    
    def _merge_all_flows(self):
        """合併所有語言的 flows，並重新分配 ID"""
        flow_id = 1
        for language, lang_data in self.flows_by_language.items():
            for flow in lang_data["flows"]:
                flow["id"] = flow_id
                flow["_original_language"] = language
                flow["_original_tool"] = lang_data["tool"]
                self.all_flows.append(flow)
                flow_id += 1
        
        print(f"\n[Info] 總計載入 {len(self.all_flows)} flows，來自 {len(self.flows_by_language)} 種語言")
        for lang, lang_data in self.flows_by_language.items():
            print(f"  - {lang}: {len(lang_data['flows'])} flows ({lang_data['tool']})")
```

#### 2.2 修改執行邏輯支持多語言

**修改位置**: 485-550行的 `execute_flow` 方法

**新增語言判斷**：
```python
def execute_flow(self, flow_id: int, context_data: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> None:
    flow = self.get_flow_by_id(flow_id)
    if not flow:
        print(f"[Error] Flow ID {flow_id} 不存在")
        return
    
    # 檢查語言類型
    language = flow.get("_original_language", "python")
    tool = flow.get("_original_tool", "unknown")
    
    print(f"\n{'='*60}")
    print(f"🚀 準備執行 Flow {flow_id} ({language.upper()} 代碼)")
    print(f"🔧 分析工具: {tool}")
    print(f"{'='*60}\n")
    
    # 根據語言選擇執行策略
    if language == "python":
        self._execute_python_flow(flow, context_data, dry_run)
    elif language == "rust":
        self._execute_rust_flow(flow, context_data, dry_run)
    elif language == "go":
        self._execute_go_flow(flow, context_data, dry_run)
    elif language == "typescript":
        self._execute_typescript_flow(flow, context_data, dry_run)
    else:
        print(f"[Error] 不支持的語言: {language}")

def _execute_python_flow(self, flow, context_data, dry_run):
    """執行 Python 代碼流程 (現有邏輯)"""
    # 保留原有的動態導入和執行邏輯
    pass

def _execute_rust_flow(self, flow, context_data, dry_run):
    """執行 Rust 代碼流程"""
    print("[Info] Rust 流程執行:")
    cli_command = flow.get("cli_command", "")
    
    if dry_run:
        print(f"[Dry Run] 將執行: {cli_command}")
        return
    
    # 提取 Cargo 項目路徑
    full_path = flow.get("full_path", [])[0] if flow.get("full_path") else None
    if not full_path:
        print("[Error] 找不到 Rust 檔案路徑")
        return
    
    cargo_dir = Path(full_path).parent
    while cargo_dir != cargo_dir.parent:
        if (cargo_dir / "Cargo.toml").exists():
            break
        cargo_dir = cargo_dir.parent
    
    if not (cargo_dir / "Cargo.toml").exists():
        print("[Error] 找不到 Cargo.toml")
        return
    
    print(f"[Info] Cargo 項目: {cargo_dir}")
    print(f"[Info] 執行指令: {cli_command}")
    
    import subprocess
    result = subprocess.run(
        cli_command.split(),
        cwd=cargo_dir,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"[Error] 執行失敗:\n{result.stderr}")

def _execute_go_flow(self, flow, context_data, dry_run):
    """執行 Go 代碼流程"""
    print("[Info] Go 流程執行:")
    cli_command = flow.get("cli_command", "")
    
    if dry_run:
        print(f"[Dry Run] 將執行: {cli_command}")
        return
    
    full_path = flow.get("full_path", [])[0] if flow.get("full_path") else None
    if not full_path:
        print("[Error] 找不到 Go 檔案路徑")
        return
    
    go_dir = Path(full_path).parent
    
    print(f"[Info] Go 項目: {go_dir}")
    print(f"[Info] 執行指令: {cli_command}")
    
    import subprocess
    result = subprocess.run(
        cli_command.split(),
        cwd=go_dir,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"[Error] 執行失敗:\n{result.stderr}")

def _execute_typescript_flow(self, flow, context_data, dry_run):
    """執行 TypeScript 代碼流程"""
    print("[Info] TypeScript 流程執行:")
    cli_command = flow.get("cli_command", "")
    
    if dry_run:
        print(f"[Dry Run] 將執行: {cli_command}")
        return
    
    full_path = flow.get("full_path", [])[0] if flow.get("full_path") else None
    if not full_path:
        print("[Error] 找不到 TypeScript 檔案路徑")
        return
    
    ts_dir = Path(full_path).parent
    
    print(f"[Info] TypeScript 項目: {ts_dir}")
    print(f"[Info] 執行指令: {cli_command}")
    
    import subprocess
    result = subprocess.run(
        cli_command.split(),
        cwd=ts_dir,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"[Error] 執行失敗:\n{result.stderr}")
```

### 階段3: 歷史與差異比對 (★★ 高優先級)

**保留工具.txt 的建議**，這部分是正確的。

### 階段4: 能力查詢介面 (★ 中優先級)

**保留工具.txt 的建議**，這部分是正確的。

---

## 📝 總結

### ❌ 工具.txt 文檔的問題

1. **最大錯誤**：建議各語言實現執行器 (優先級1)
   - 與用戶架構不符
   - 造成重複開發
   - 忽略了 Python 動態語言優勢

2. **遺漏關鍵需求**：沒有提到統一 JSON 格式
   - 這是真正的優先級1
   - 當前格式完全不兼容 FlowExecutor

### ✅ 正確的實施順序

1. **優先級1 (★★★)**：統一JSON輸出格式
   - 修改 TS/Go/Rust 工具輸出結構
   - 確保符合 FlowExecutor 要求

2. **優先級1 (★★★)**：整合多語言JSON
   - 修改 Python FlowExecutor
   - 支持自動發現和載入各語言分析結果
   - 實現多語言執行邏輯

3. **優先級2 (★★)**：歷史與差異比對
   - 工具.txt 建議正確

4. **優先級3 (★)**：能力查詢介面
   - 工具.txt 建議正確

---

## 🎯 下一步行動

建議按以下順序執行：

1. **立即開始**：修改 TS/Go/Rust 工具的 JSON 輸出格式
2. **同步進行**：修改 Python FlowExecutor 整合多語言支持
3. **測試驗證**：執行跨語言流程測試
4. **後續優化**：實現歷史比對和查詢功能

---

## 🔄 更新記錄

| 日期 | 版本 | 更新內容 | 作者 |
|------|------|---------|------|
| 2026-01-13 | v1.0 | 評估工具.txt並提出修正方案 | GitHub Copilot |
