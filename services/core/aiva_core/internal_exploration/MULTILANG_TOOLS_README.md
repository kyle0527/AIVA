# 多語言 AST 分析工具套件

**版本**: v3.0  
**更新日期**: 2026-01-18  
**狀態**: ✅ 生產就緒
**設計定位**: 語言層 AST 解析器（只輸出 JSON，不含分類/執行邏輯）

---

## 🎯 設計理念

本目錄的語言工具只負責 **AST 分析**，輸出統一的 JSON Schema v3.3 格式，不包含：
- ❌ 分類邏輯（由 aiva_internal_classifier.py / aiva_external_classifier.py 處理）
- ❌ 執行邏輯（由 aiva_internal_executor.py / aiva_external_executor.py 處理）
- ❌ CLI 生成（由執行器處理）

### 架構分層

```
語言層 (Language Layer)
  │
  ├── python_tools/aiva_flow_analyzer.py    → analysis_results.json
  ├── go_tools/go2mermaid.go               → analysis_results.json
  ├── rust_tools/src/main.rs               → analysis_results.json
  └── typescript_tools/ts2mermaid.ts       → analysis_results.json
                │
                ↓ JSON Schema v3.3
                │
業務邏輯層 (Business Logic Layer)
  │
  ├── aiva_internal_classifier.py  (分類 AI Core 數據流)
  ├── aiva_internal_executor.py    (執行 AI Core flows)
  ├── aiva_external_classifier.py  (分類 Features/Scan 數據流)
  └── aiva_external_executor.py    (執行 Features/Scan flows)
```

---

## 📋 工具概覽

本目錄包含四種語言版本的 AST 分析工具，支援：
- 🐍 **Python** (`python_tools/aiva_flow_analyzer.py`) - Python 代碼 AST 分析
- 🔷 **Go** (`go_tools/go2mermaid.go`) - Go 代碼 AST 分析
- 📘 **TypeScript** (`typescript_tools/ts2mermaid.ts`) - TypeScript/JavaScript AST 分析
- 🦀 **Rust** (`rust_tools/src/main.rs`) - Rust 代碼 AST 分析

**核心功能**：
1. ✅ **AST 解析** - 分析代碼結構，提取函數調用關係
2. ✅ **JSON 輸出** - 統一 Schema v3.3 格式，供上層業務邏輯使用
3. ✅ **Mermaid 圖表** - 可選生成 .mmd 流程圖檔案

**不包含的功能** (由上層腳本處理):
- ❌ 分類數據流到模組
- ❌ 生成 CLI 命令
- ❌ 執行代碼流程

---

## 🐍 Python 版本

### 安裝與運行
```bash
# 直接運行 (無需額外依賴)
python aiva_flow_analyzer.py --input=../../ --output=./py_analysis

# 指定方向
python aiva_flow_analyzer.py --input=. --output=./analysis --direction=LR

# 限制文件數量
python aiva_flow_analyzer.py --input=. --max-files=50
```

### 輸出文件
- `*.mmd` - Mermaid 流程圖文件
- `classification_data.json` - 分類數據
- `cli_commands.sh` - 執行命令腳本

---

## 🔷 Go 版本

### 安裝
```bash
# 初始化 Go 模組
go mod init github.com/aiva/go2mermaid
go mod tidy
```

### 運行
```bash
# 編譯並運行
go run go2mermaid.go --input=. --output=./go_analysis

# 編譯為可執行文件
go build -o go2mermaid go2mermaid.go
./go2mermaid --input=../../../tools --output=./go_analysis

# 指定參數
go run go2mermaid.go \
  --input=../../ \
  --output=./go_analysis \
  --max-files=100 \
  --direction=TB
```

### 特點
- ⚡ 高性能並發處理
- 📊 自動分析 Go 標準庫調用
- 🔍 支援 goroutine 和 channel 流程

---

## 📘 TypeScript 版本

### 安裝
```bash
# 安裝依賴
npm install

# 或使用 yarn
yarn install
```

### 運行
```bash
# 使用 ts-node 直接運行
npx ts-node ts2mermaid.ts --input=. --output=./ts_analysis

# 使用 npm script
npm run analyze

# 編譯後運行
npm run build
node ts2mermaid.js --input=../../../web --output=./ts_analysis

# 指定參數
npx ts-node ts2mermaid.ts \
  --input=../../ \
  --output=./ts_analysis \
  --max-files=100 \
  --direction=LR
```

### 特點
- 🎯 支援 TypeScript 和 JavaScript (.ts, .tsx, .js, .jsx)
- 🔄 處理 async/await 異步流程
- 📦 分析 ES6 模組導入導出

---

## 🦀 Rust 版本

### 安裝
```bash
# 添加 Cargo.toml 依賴後構建
cargo build --release
```

### 運行
```bash
# 使用 cargo run
cargo run --bin rs2mermaid -- --input=. --output=./rs_analysis

# 使用編譯後的二進制
./target/release/rs2mermaid --input=../../ --output=./rs_analysis

# 指定參數
cargo run --bin rs2mermaid -- \
  --input=../../../tools \
  --output=./rs_analysis \
  --max-files=100 \
  --direction=TB
```

### 特點
- 🚀 極致性能，適合大型專案
- 🔒 記憶體安全分析
- ⚙️ 支援 Rust 特有的所有權和生命週期

---

## 📊 統一輸出格式 (JSON Schema v3.3)

所有語言工具輸出相同的 `analysis_results.json`：

```json
{
  "metadata": {
    "tool": "python_analyzer | go2mermaid | rs2mermaid | ts2mermaid",
    "version": "3.0",
    "language": "python | go | rust | typescript",
    "generated_at": "2026-01-18T10:30:00+08:00",
    "total_flows": 157,
    "total_files": 119,
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["from_function", "to_function"],
      "full_path": ["module.submodule.from_func", "module.submodule.to_func"],
      "func_names": ["from_func", "to_func"],
      "length": 2,
      "start": "from_function",
      "end": "to_function",
      "from_script": "path/to/file1.py",
      "to_script": "path/to/file2.py"
    }
  ],
  "functions": {
    "path/to/file.py": [
      {
        "name": "function_name",
        "type": "function",
        "line": 42,
        "calls": ["other_function"]
      }
    ]
  }
}
```

### 可選輸出

部分工具可選生成 Mermaid 流程圖（`.mmd`），但主要輸出為 JSON。

---

## 🔗 與上層腳本的集成

語言工具輸出的 JSON 由上層腳本處理：

### Internal CLI 流程（AI Core 模組）
```bash
# 1. AST 分析
python python_tools/aiva_flow_analyzer.py --target services/core/aiva_core

# 2. 分類（由分類器讀取 analysis_results.json）
python aiva_internal_classifier.py

# 3. 執行（由執行器讀取 classification_data.json）
python aiva_internal_executor.py --flow 11
```

### External CLI 流程（Features/Scan 模組）
```bash
# 1. 多語言 AST 分析（各語言工具分別執行）
python python_tools/aiva_flow_analyzer.py --target services/features/function_xss
go run go_tools/go2mermaid.go --input services/features/function_authn_go
cargo run --bin rs2mermaid -- --input services/features/function_crypto
npx ts-node typescript_tools/ts2mermaid.ts --input services/scan/typescript_engine

# 2. 整合分類（讀取所有 analysis_results.json）
python aiva_external_classifier.py

# 3. 執行（讀取整合後的 classification_data.json）
python aiva_external_executor.py --lang python --flow 1
```
---

## 🔧 參數說明

所有工具支援的共同參數：

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--input` | 輸入目錄路徑 | `.` | `--input=../../services` |
| `--output` | 輸出目錄路徑 | `./analysis_output` | `--output=./my_analysis` |
| `--max-files` | 最大處理文件數 | `100` | `--max-files=500` |
| `--direction` | 流程圖方向（可選） | `TB` | `--direction=LR` |

### 流程圖方向選項（僅用於 .mmd 生成）
- `TB` / `TD` - 從上到下 (Top to Bottom)
- `BT` - 從下到上 (Bottom to Top)
- `LR` - 從左到右 (Left to Right)
- `RL` - 從右到左 (Right to Left)

---

## 📈 語言工具對比

| 語言 | 核心檔案 | 輸出格式 | 執行方式 | 適用場景 |
|------|---------|---------|---------|---------|
| **Python** | `aiva_flow_analyzer.py` | JSON v3.3 | `python` 直接執行 | AI Core 模組分析 |
| **Go** | `go2mermaid.go` | JSON v3.3 | 編譯或 `go run` | 認證模組 (function_authn_go) |
| **Rust** | `src/main.rs` | JSON v3.3 | `cargo run` | 加密模組 (function_crypto) |
| **TypeScript** | `ts2mermaid.ts` | JSON v3.3 | `npx ts-node` | 掃描引擎 (typescript_engine) |

**所有工具**：
- ✅ 輸出統一 JSON Schema v3.3
- ✅ 支援大型專案分析
- ✅ AI 相容格式
- ⚠️ 不包含分類/執行邏輯（由上層腳本處理）

---

## 📝 使用注意事項

### ✅ 應該做的事
- 使用語言工具只做 AST 分析
- 輸出 JSON 給上層分類器/執行器使用
- 保持各語言工具輸出格式一致
- 針對不同語言的模組選用對應工具

### ❌ 不應該做的事
- 不要在語言工具中加入分類邏輯（屬於 classifier 職責）
- 不要在語言工具中加入執行邏輯（屬於 executor 職責）
- 不要直接執行語言工具輸出的流程（需經過 classifier）
- 不要混用不同語言的輸出格式

---

## 🔄 工作流程總覽

```
Step 1: 語言層 AST 分析
  Python → analysis_results.json (JSON v3.3)
  Go     → analysis_results.json (JSON v3.3)
  Rust   → analysis_results.json (JSON v3.3)
  TS     → analysis_results.json (JSON v3.3)
           ↓
Step 2: 業務邏輯層分類
  Internal → aiva_internal_classifier.py  → classification_data.json
  External → aiva_external_classifier.py  → classification_data.json
           ↓
Step 3: 執行層
  Internal → aiva_internal_executor.py    → Execute flows
  External → aiva_external_executor.py    → Execute flows
```
| **Python** | ~60s | 200MB | 通用分析，原型開發 |
| **Go** | ~15s | 100MB | 大型專案，並發處理 |
| **TypeScript** | ~45s | 250MB | 前端專案，Node.js 應用 |
| **Rust** | ~10s | 50MB | 超大專案，極致性能 |

---

## 🛠️ 進階用法

### 1. 自定義分類規則

修改各語言文件中的 `Classifier.classify()` 方法：

**Python:**
```python
def classify(self, metadata: FlowMetadata):
    func_name = metadata.function_name.lower()
    
    # 添加自定義規則
    if "custom" in func_name:
        metadata.category = "custom_category"
    # ... 其他規則
```

**Go:**
```go
func (c *Classifier) Classify(metadata *FlowMetadata) {
    funcName := strings.ToLower(metadata.FunctionName)
    
    // 添加自定義規則
    if strings.Contains(funcName, "custom") {
        metadata.Category = "custom_category"
    }
    // ... 其他規則
}
```

### 2. 整合到 CI/CD

**GitHub Actions 範例:**
```yaml
name: Code Flow Analysis

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Python Analysis
        run: |
          python aiva_flow_analyzer.py --input=. --output=./analysis
          
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: flow-analysis
          path: ./analysis
```

### 3. 與 AIVA CLI 整合

生成的 `cli_commands.sh` 可直接用於 AIVA CLI：

```bash
# 執行特定分類的流程
./cli_commands.sh reconnaissance

# 或直接調用
python aiva_cli_implementation.py --category=exploitation
```

---

## 📚 相關文檔

- [AIVA Flow Analyzer 完整指南](./AIVA_FLOW_ANALYZER_GUIDE.md)
- [Internal Exploration README](./README.md)
- [Self-Healing 模組](./self_healing/README.md)
- [操作手冊](./OPERATION_MANUAL.md)

---

## 🤝 貢獻指南

歡迎貢獻新的語言支援！請確保：

1. ✅ 實現相同的三大功能（產圖、分類、CLI）
2. ✅ 輸出格式與現有工具一致
3. ✅ 添加完整的使用文檔
4. ✅ 通過基礎測試驗證

---

## 📝 更新日誌

### v10.0.0 (2025-12-10)
- ✅ 新增 Go 版本支援
- ✅ 新增 TypeScript 版本支援
- ✅ 新增 Rust 版本支援
- ✅ 統一輸出格式和分類邏輯
- ✅ 完整文檔和範例

---

## 📞 技術支援

如有問題，請查閱：
- [GitHub Issues](https://github.com/kyle0527/AIVA/issues)
- [AIVA 文檔](../../../docs/)
- [Self-Healing README](./self_healing/README.md)

---

**維護者**: AIVA Team  
**最後更新**: 2025-12-10
