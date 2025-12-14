# 多語言 AST 分析工具套件

**版本**: v10.0.0  
**更新日期**: 2025-12-10  
**狀態**: ✅ 生產就緒

---

## 📋 工具概覽

本目錄包含四種語言版本的 AST 分析與 Mermaid 流程圖生成工具，支援：
- 🐍 **Python** (`aiva_flow_analyzer.py`) - 原生 AIVA 版本
- 🔷 **Go** (`go2mermaid.go`) - 高性能並發處理
- 📘 **TypeScript** (`ts2mermaid.ts`) - 前端專案分析
- 🦀 **Rust** (`rs2mermaid.rs`) - 系統級代碼分析

所有工具都支援三大核心功能：
1. **產圖** - 生成 Mermaid 流程圖
2. **分類** - 按功能自動分類代碼流
3. **CLI 產生** - 自動生成執行命令腳本

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

## 📊 輸出格式

所有工具生成相同格式的輸出：

### 1. Mermaid 流程圖 (`.mmd`)
```mermaid
flowchart TB
    n1([開始])
    n2[函數: example]
    n3{if condition}
    n4[處理邏輯]
    n5([結束])
    
    n1 --> n2
    n2 --> n3
    n3 -->|Yes| n4
    n3 -->|No| n5
    n4 --> n5
```

### 2. 分類數據 (`classification_data.json`)
```json
{
  "total_flows": 125,
  "categories": {
    "reconnaissance": [...],
    "exploitation": [...],
    "analysis": [...],
    "reporting": [...],
    "persistence": [...],
    "other": [...]
  },
  "summary": {
    "reconnaissance": 25,
    "exploitation": 30,
    "analysis": 20,
    "reporting": 15,
    "persistence": 10,
    "other": 25
  }
}
```

### 3. CLI 命令 (`cli_commands.sh`)
```bash
# AIVA Flow Analysis CLI Commands

## Execute by Category

# Reconnaissance
python aiva_flow_analyzer.py --category=reconnaissance --execute

# Exploitation
python aiva_flow_analyzer.py --category=exploitation --execute

# Analysis
python aiva_flow_analyzer.py --category=analysis --execute
```

---

## 🎯 分類邏輯

所有工具使用相同的分類規則（基於函數名）：

| 分類 | 關鍵字 | 說明 |
|------|--------|------|
| **reconnaissance** | scan, detect, discover | 偵察和掃描功能 |
| **exploitation** | exploit, attack, inject | 漏洞利用和攻擊 |
| **analysis** | analyze, parse, inspect | 代碼分析和解析 |
| **reporting** | report, generate, export | 報告生成 |
| **persistence** | store, save, persist | 數據持久化 |
| **other** | (其他) | 未分類功能 |

---

## 🔧 參數說明

所有工具支援的共同參數：

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--input` | 輸入目錄路徑 | `.` | `--input=../../services` |
| `--output` | 輸出目錄路徑 | `./analysis_output` | `--output=./my_analysis` |
| `--max-files` | 最大處理文件數 | `100` | `--max-files=500` |
| `--direction` | 流程圖方向 | `TB` | `--direction=LR` |

### 流程圖方向選項
- `TB` / `TD` - 從上到下 (Top to Bottom)
- `BT` - 從下到上 (Bottom to Top)
- `LR` - 從左到右 (Left to Right)
- `RL` - 從右到左 (Right to Left)

---

## 🚀 快速開始

### 一鍵分析所有語言
```bash
# Python
python aiva_flow_analyzer.py --input=../../ --output=./py_analysis

# Go
go run go2mermaid.go --input=../../ --output=./go_analysis

# TypeScript
npx ts-node ts2mermaid.ts --input=../../ --output=./ts_analysis

# Rust
cargo run --bin rs2mermaid -- --input=../../ --output=./rs_analysis
```

### 批量分析腳本 (Windows PowerShell)
```powershell
# 分析所有語言
Write-Host "開始多語言分析..."

# Python
python aiva_flow_analyzer.py --input=..\..\..\ --output=.\analysis\python

# Go  
go run go2mermaid.go --input=..\..\..\tools --output=.\analysis\go

# TypeScript
npx ts-node ts2mermaid.ts --input=..\..\..\web --output=.\analysis\typescript

# Rust
cargo run --bin rs2mermaid -- --input=..\..\..\tools --output=.\analysis\rust

Write-Host "分析完成！"
```

### 批量分析腳本 (Linux/Mac)
```bash
#!/bin/bash
echo "開始多語言分析..."

# Python
python3 aiva_flow_analyzer.py --input=../../ --output=./analysis/python

# Go
go run go2mermaid.go --input=../../ --output=./analysis/go

# TypeScript
npx ts-node ts2mermaid.ts --input=../../ --output=./analysis/typescript

# Rust
cargo run --bin rs2mermaid -- --input=../../ --output=./analysis/rust

echo "分析完成！"
```

---

## 📈 性能對比

基於 1000 個文件的測試：

| 語言 | 處理時間 | 記憶體使用 | 適用場景 |
|------|----------|-----------|---------|
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
