# CLI 參數語義映射規範

**版本**: v1.0  
**日期**: 2026-01-13  
**目的**: 定義各語言工具CLI參數的語義對標關係，不強制統一命名

---

## 🎯 核心原則

1. **各語言保持自己的命名習慣** - 不強制統一參數名稱
2. **建立語義映射** - 定義參數的"意義"對應關係
3. **統一JSON輸出鍵名** - 只在輸出結果時使用統一的鍵名
4. **最小化轉換成本** - 參數總數不多，映射簡單

---

## 📋 語義映射表

### 1. 輸入目錄/文件 (INPUT_PATH)

**語義**: 指定要分析的代碼路徑

| 語言 | 參數名稱 | 短參數 | 預設值 | 範例 |
|------|---------|--------|--------|------|
| Python | `--target` | `-t` | `"aiva_core"` | `--target services/features` |
| TypeScript | `--input` | - | `"."` | `--input ./src` |
| Go | `--input` | - | `"."` | `--input ./app` |
| Rust | `--input=` | - | `"."` | `--input=./rust_core` |

**JSON輸出鍵名**: `"input_path"`

---

### 2. 輸出目錄 (OUTPUT_DIR)

**語義**: 指定分析結果的儲存目錄

| 語言 | 參數名稱 | 短參數 | 預設值 | 範例 |
|------|---------|--------|--------|------|
| Python | `--output-dir` | - | 動態生成 | `--output-dir ./analysis` |
| TypeScript | `--output` | - | `"./analysis_output"` | `--output ./ts_analysis` |
| Go | `--output` | - | `"./go_analysis"` | `--output ./results` |
| Rust | `--output=` | - | `"./analysis_output"` | `--output=./rust_results` |

**JSON輸出鍵名**: `"output_dir"`

---

### 3. 模組名稱 (MODULE_NAME)

**語義**: 指定目標模組分類 (用於 Python 的內外部分類)

| 語言 | 參數名稱 | 短參數 | 預設值 | 範例 |
|------|---------|--------|--------|------|
| Python | `--module` | `-m` | `"core"` | `--module features` |
| TypeScript | - | - | - | - |
| Go | - | - | - | - |
| Rust | - | - | - | - |

**說明**: 僅 Python 需要，因為 AIVA 的內外部模組分類邏輯在 Python 層實現

**JSON輸出鍵名**: `"module_name"` (僅 Python 輸出)

---

### 4. 分析深度 (ANALYSIS_DEPTH)

**語義**: AST 分析的遞迴深度限制

| 語言 | 參數名稱 | 短參數 | 預設值 | 範例 |
|------|---------|--------|--------|------|
| Python | `--depth` | `-d` | `10` | `--depth 15` |
| TypeScript | - | - | - | - |
| Go | - | - | - | - |
| Rust | - | - | - | - |

**說明**: 僅 Python 實現，其他語言暫無深度限制需求

**JSON輸出鍵名**: `"analysis_depth"` (僅 Python 輸出)

---

### 5. 輸出格式 (OUTPUT_FORMAT) - 未來擴展

**語義**: 指定輸出格式類型

| 語言 | 參數名稱 | 短參數 | 可選值 | 範例 |
|------|---------|--------|--------|------|
| Python | `--format` | `-f` | `json, mermaid, both` | `--format both` |
| TypeScript | `--format` | - | `json, mermaid, both` | `--format json` |
| Go | `--format` | - | `json, mermaid, both` | `--format both` |
| Rust | `--format=` | - | `json, mermaid, both` | `--format=json` |

**當前狀態**: 未實現，所有工具預設輸出 JSON + Mermaid

**JSON輸出鍵名**: `"output_format"`

---

### 6. 詳細輸出 (VERBOSE_MODE) - 未來擴展

**語義**: 啟用詳細日誌輸出

| 語言 | 參數名稱 | 短參數 | 類型 | 範例 |
|------|---------|--------|------|------|
| Python | `--verbose` | `-v` | flag | `--verbose` |
| TypeScript | `--verbose` | - | flag | `--verbose` |
| Go | `--verbose` | - | bool | `--verbose=true` |
| Rust | `--verbose` | - | flag | `--verbose` |

**當前狀態**: 未實現

**JSON輸出鍵名**: `"verbose"` (boolean)

---

## 📝 統一JSON輸出結構

### 所有工具必須包含的元數據欄位

```json
{
  "metadata": {
    "tool": "python_analyzer | ts2mermaid | go2mermaid | rs2mermaid",
    "version": "工具版本",
    "timestamp": "ISO 8601 格式時間",
    "input_path": "實際輸入路徑",
    "output_dir": "實際輸出目錄",
    "language": "python | typescript | go | rust",
    "analysis_duration_seconds": 執行時間
  },
  "summary": {
    "total_files": 檔案數量,
    "total_functions": 函數數量,
    "total_flows": 流程數量,
    "real_connections": 跨檔案連接數量
  },
  "functions": [
    {
      "function_name": "函數名稱",
      "module": "所屬模組",
      "category": "功能分類",
      "description": "能力描述",
      "cli_command": "執行指令",
      "inputs": ["參數列表"],
      "outputs": ["返回值"],
      "flow_steps": 流程步驟數
    }
  ],
  "classification": {
    "categories": {
      "XSS": 數量,
      "SQLi": 數量,
      "IDOR": 數量,
      "SSRF": 數量,
      "BizLogic": 數量,
      "InfoLeak": 數量,
      "Crypto": 數量
    }
  },
  "flow_chains": [
    {
      "from_file": "來源檔案",
      "from_function": "來源函數",
      "to_file": "目標檔案",
      "to_function": "目標函數",
      "call_expression": "調用表達式"
    }
  ]
}
```

---

## 🔄 參數轉換邏輯 (僅在需要時)

### 場景1: Python調用其他語言工具

當 Python 主控程序需要調用其他語言工具時：

```python
def call_typescript_analyzer(target_path: str, output_dir: str) -> dict:
    """將 Python 參數轉換為 TypeScript 參數"""
    cmd = [
        "node", "ts2mermaid.ts",
        "--input", target_path,      # Python --target → TS --input
        "--output", output_dir        # Python --output-dir → TS --output
    ]
    result = subprocess.run(cmd, capture_output=True)
    return json.loads(result.stdout)

def call_go_analyzer(target_path: str, output_dir: str) -> dict:
    """將 Python 參數轉換為 Go 參數"""
    cmd = [
        "./go2mermaid",
        "-input", target_path,        # Python --target → Go --input
        "-output", output_dir         # Python --output-dir → Go --output
    ]
    result = subprocess.run(cmd, capture_output=True)
    return json.loads(result.stdout)

def call_rust_analyzer(target_path: str, output_dir: str) -> dict:
    """將 Python 參數轉換為 Rust 參數"""
    cmd = [
        "./rs2mermaid",
        f"--input={target_path}",     # Python --target → Rust --input=
        f"--output={output_dir}"      # Python --output-dir → Rust --output=
    ]
    result = subprocess.run(cmd, capture_output=True)
    return json.loads(result.stdout)
```

### 場景2: 獨立使用各工具

用戶直接使用各語言工具時，**不需要轉換**，按各語言習慣使用：

```bash
# Python
python aiva_exploration_pipeline.py --target services/features --module features

# TypeScript
node ts2mermaid.ts --input ./src --output ./analysis

# Go
./go2mermaid --input ./app --output ./results

# Rust
./rs2mermaid --input=./rust_core --output=./analysis
```

---

## 📌 實現檢查清單

### 當前狀態

- [✅] Python 工具參數已定義
- [✅] TypeScript 工具參數已定義
- [✅] Go 工具參數已定義
- [✅] Rust 工具參數已定義
- [❌] 統一 JSON 輸出結構 (需要添加 metadata 欄位)
- [❌] Python 轉換函數 (如果需要跨語言調用)

### 待實現

#### 階段1: 統一 JSON 輸出 metadata (1天)

**修改位置**：
- `services/core/aiva_core/internal_exploration/python_tools/aiva_exploration_pipeline.py` - 添加 metadata 欄位
- `services/core/aiva_core/internal_exploration/typescript_tools/ts2mermaid.ts` - 添加 metadata 欄位
- `services/core/aiva_core/internal_exploration/go_tools/go2mermaid.go` - 添加 metadata 欄位
- `services/core/aiva_core/internal_exploration/rust_tools/src/main.rs` - 添加 metadata 欄位

**統一添加**：
```json
{
  "metadata": {
    "tool": "工具名稱",
    "version": "版本號",
    "timestamp": "2026-01-13T10:30:00Z",
    "input_path": "實際路徑",
    "output_dir": "輸出目錄",
    "language": "語言名稱",
    "analysis_duration_seconds": 12.5
  },
  "summary": { /* 現有內容 */ },
  "functions": [ /* 現有內容 */ ]
}
```

#### 階段2: 統一能力描述格式 (2天)

**目標**: 確保所有工具的 `functions[].description` 和 `functions[].cli_command` 格式一致

**範例格式**：
```json
{
  "description": "檢測反射型XSS漏洞通過在URL參數注入測試腳本 | Detect reflected XSS by injecting scripts in URL params",
  "cli_command": "aiva xss detect --type reflected --target <url>",
  "category": "XSS"
}
```

**修改位置**：
- Python: `aiva_flow_classifier.py` 的 `classify()` 方法
- TypeScript: `ts2mermaid.ts` 的 `Classifier.classify()` 方法
- Go: `go2mermaid.go` 的 `Classifier.Classify()` 方法
- Rust: `rust_tools/src/main.rs` 的 `Classifier::classify()` 方法

---

## 🎯 總結

### 設計哲學

1. **尊重語言慣例** - 不強求統一命名
2. **語義對標清晰** - 明確各參數的"意義"
3. **輸出格式統一** - JSON 結構和鍵名一致
4. **轉換邏輯簡單** - 參數不多，映射容易維護

### 優勢

- ✅ 各語言開發者使用習慣的參數名稱
- ✅ 降低學習成本和記憶負擔
- ✅ 工具可獨立使用，不依賴統一入口
- ✅ 跨語言整合時僅需簡單映射

### 維護成本

- 參數總數: **2個核心參數** (input_path, output_dir)
- 映射複雜度: **極低** (1:1 對應)
- 未來擴展: 新增參數時更新此映射表即可

---

## 📚 相關文檔

- [MULTILANG_TOOL_UNIFICATION_PLAN.md](./MULTILANG_TOOL_UNIFICATION_PLAN.md) - 多語言工具整合計劃
- [CAPABILITY_DESCRIPTION_SPEC.md](./CAPABILITY_DESCRIPTION_SPEC.md) - 能力描述規範 (待創建)

---

## 🔄 更新記錄

| 日期 | 版本 | 更新內容 | 作者 |
|------|------|---------|------|
| 2026-01-13 | v1.0 | 初始版本，定義參數語義映射 | GitHub Copilot |
