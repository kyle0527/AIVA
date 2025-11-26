# AST 分析工具集 - Mermaid 流程圖生成器

這個工具集提供四種程式語言的 AST 分析與 Mermaid 流程圖自動生成功能。

## 📋 工具概覽

| 工具 | 語言 | 目標引擎 | 狀態 |
|-----|------|---------|------|
| `py2mermaid.py` | Python | Python Engine | ✅ 完成 |
| `go2mermaid.go` | Go | Go Engine | ✅ 完成 |
| `ts2mermaid.ts` | TypeScript | TypeScript Engine | ✅ 完成 |
| `rs2mermaid.rs` | Rust | Rust Engine | ✅ 完成 |

## 🎯 功能特性

所有工具都提供以下核心功能：

- ✨ **AST 解析**：深度解析程式碼結構
- 📊 **流程圖生成**：為每個函數生成獨立的 Mermaid 流程圖
- 🔄 **控制流分析**：識別 if/else、循環、switch 等控制結構
- 🎨 **可視化**：生成標準 Mermaid 語法，可直接在文檔中使用
- 📁 **批量處理**：支援目錄掃描與批量生成

## 🚀 使用方法

### 1. Python 版本 (py2mermaid.py)

```bash
# 基本使用
python tools/common/development/py2mermaid.py -i ./services -o ./docs/diagrams/python

# 自訂參數
python py2mermaid.py \
  --input ./src \
  --output ./output \
  --direction LR \
  --max-files 100

# 參數說明
# -i, --input      輸入目錄或檔案路徑
# -o, --output     輸出目錄
# -d, --direction  流程圖方向 (TB/LR/RL/BT)
# -m, --max-files  最大處理檔案數
```

### 2. Go 版本 (go2mermaid.go)

```bash
# 基本使用
go run tools/common/development/go2mermaid.go \
  -i ./services/scan/engines/go_engine \
  -o ./docs/diagrams/go

# 單一檔案
go run go2mermaid.go -i ./main.go -o ./output

# 參數說明
# -i  輸入文件或目錄路徑
# -o  輸出目錄
# -d  流程圖方向 (TB/LR/RL/BT)
# -m  最大處理文件數
```

### 3. TypeScript 版本 (ts2mermaid.ts)

```bash
# 安裝依賴
cd tools/common/development
npm install typescript @types/node ts-node

# 基本使用
npm run analyze:typescript

# 或直接運行
ts-node ts2mermaid.ts \
  -i ./services/scan/engines/typescript_engine \
  -o ./docs/diagrams/typescript

# 自訂參數
ts-node ts2mermaid.ts \
  --input ./src \
  --output ./output \
  --direction LR \
  --max-files 100

# 參數說明
# -i, --input      輸入目錄或檔案路徑
# -o, --output     輸出目錄
# -d, --direction  流程圖方向 (TB/LR/RL/BT)
# -m, --max-files  最大處理檔案數
```

### 4. Rust 版本 (rs2mermaid.rs)

```bash
# 第一次使用需要編譯
cd tools/common/development
cargo build --release --bin rs2mermaid

# 基本使用
cargo run --bin rs2mermaid -- \
  -i ./services/scan/engines/rust_engine \
  -o ./docs/diagrams/rust

# 使用編譯後的二進制
./target/release/rs2mermaid \
  --input ./src \
  --output ./output \
  --direction LR

# 參數說明
# -i, --input      輸入目錄或檔案路徑
# -o, --output     輸出目錄
# -d, --direction  流程圖方向 (TB/LR/RL/BT)
# -m, --max-files  最大處理檔案數
```

## 📂 輸出格式

所有工具生成的輸出格式一致：

```
docs/diagrams/
├── python/
│   ├── services_scan_scanner_Module.mmd
│   ├── services_scan_scanner_Function_scan.mmd
│   └── ...
├── go/
│   ├── internal_ssrf_detector_Function_DetectSSRF.mmd
│   └── ...
├── typescript/
│   ├── src_index_Function_initialize.mmd
│   ├── src_services_scan_service_Function_scan.mmd
│   └── ...
└── rust/
    ├── src_main_Function_main.mmd
    ├── src_scanner_Function_scan.mmd
    └── ...
```

每個 `.mmd` 檔案包含標準的 Mermaid 流程圖語法，可以：
- 直接在 Markdown 文件中引用
- 使用 Mermaid Live Editor 查看
- 集成到文檔生成工具中

## 🎨 流程圖示例

生成的 Mermaid 流程圖包含以下元素：

```mermaid
flowchart TB
    n1(["開始"])
    n2["初始化變量"]
    n3{"if condition"}
    n4["執行 A"]
    n5["執行 B"]
    n6[""]
    n7(["結束"])
    
    n1 --> n2
    n2 --> n3
    n3 -->|Yes| n4
    n3 -->|No| n5
    n4 --> n6
    n5 --> n6
    n6 --> n7
```

## 🔧 技術細節

### Python (py2mermaid.py)
- **解析器**: `ast` (標準庫)
- **支援結構**: 函數、類別、if/else、for/while、try/except
- **特殊處理**: with 語句、async/await

### Go (go2mermaid.go)
- **解析器**: `go/ast` (標準庫)
- **支援結構**: 函數、if/else、for/range、switch/select、defer/go
- **特殊處理**: goroutine、channel 操作

### TypeScript (ts2mermaid.ts)
- **解析器**: `typescript` (官方編譯器 API)
- **支援結構**: 函數、箭頭函數、if/else、for/while、switch、try/catch
- **特殊處理**: async/await、Promise、泛型

### Rust (rs2mermaid.rs)
- **解析器**: `syn` crate
- **支援結構**: 函數、if/else、for/while/loop、match、Result/Option
- **特殊處理**: 所有權、借用、生命週期標註

## 📊 支援的控制結構

| 結構 | Python | Go | TypeScript | Rust |
|-----|--------|-----|-----------|------|
| if/else | ✅ | ✅ | ✅ | ✅ |
| for 循環 | ✅ | ✅ | ✅ | ✅ |
| while 循環 | ✅ | ✅ | ✅ | ✅ |
| switch/match | ✅ | ✅ | ✅ | ✅ |
| try/catch | ✅ | ❌ | ✅ | ❌ |
| defer | ❌ | ✅ | ❌ | ❌ |
| async/await | ✅ | ❌ | ✅ | ✅ |

## 🎯 目標引擎分析

### TypeScript Engine (`services/scan/engines/typescript_engine`)
```bash
ts-node ts2mermaid.ts \
  -i ../../../services/scan/engines/typescript_engine \
  -o ../../../docs/diagrams/typescript
```

主要分析目標：
- `src/index.ts` - 主程序入口
- `src/services/scan-service.ts` - 掃描服務
- `src/services/enhanced-dynamic-scan.service.ts` - 增強掃描
- `src/services/network-interceptor.service.ts` - 網路攔截

### Rust Engine (`services/scan/engines/rust_engine`)
```bash
cargo run --bin rs2mermaid -- \
  -i ../../../services/scan/engines/rust_engine \
  -o ../../../docs/diagrams/rust
```

主要分析目標：
- `src/main.rs` - CLI 入口
- `src/scanner.rs` - 核心掃描器
- `src/endpoint_discovery.rs` - 端點發現
- `src/js_analyzer.rs` - JS 分析
- `src/attack_surface.rs` - 攻擊面評估

## 🔍 最佳實踐

1. **選擇合適的方向**
   - `TB` (Top to Bottom): 適合深度較深的函數
   - `LR` (Left to Right): 適合寬度較大的流程

2. **批量處理建議**
   - 大型專案建議設定 `--max-files` 限制
   - 使用 `--output` 分離不同模組的輸出

3. **整合到 CI/CD**
   ```yaml
   # .github/workflows/generate-diagrams.yml
   - name: Generate Flow Diagrams
     run: |
       python tools/common/development/py2mermaid.py -i ./services -o ./docs/diagrams
       cargo run --bin rs2mermaid -- -i ./services -o ./docs/diagrams
   ```

## 📝 常見問題

### Q: 如何預覽生成的流程圖？
A: 可以使用以下方式：
1. VS Code + Mermaid 擴充功能
2. [Mermaid Live Editor](https://mermaid.live/)
3. GitHub/GitLab 的 Markdown 渲染

### Q: 生成的圖表過於複雜怎麼辦？
A: 考慮：
1. 重構複雜函數
2. 使用 `LR` 方向增加可讀性
3. 調整 Mermaid 顯示設定

### Q: 如何處理非標準語法？
A: 工具會跳過無法解析的檔案，並在終端顯示警告訊息。

## 🤝 貢獻

歡迎為這些工具貢獻改進：
1. 支援更多控制結構
2. 改善流程圖佈局
3. 增加自訂樣式選項
4. 支援更多輸出格式

## 📄 授權

MIT License

---

**維護團隊**: AIVA Development Team  
**最後更新**: 2025-11-22
