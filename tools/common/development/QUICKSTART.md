# 快速開始指南 - AST 分析工具

## 🎯 目標

這些工具能夠自動分析您的程式碼並生成視覺化的流程圖，幫助理解：
- TypeScript Engine (`services/scan/engines/typescript_engine`)
- Rust Engine (`services/scan/engines/rust_engine`)
- 以及任何其他 Python/Go/TypeScript/Rust 程式碼

## 🚀 快速開始

### 方法 1: 自動測試腳本（推薦）

最簡單的方式是使用自動測試腳本：

```powershell
# 在專案根目錄執行
cd tools/common/development
.\test_ast_tools.ps1
```

這個腳本會：
1. ✅ 自動安裝所需依賴
2. ✅ 測試所有四種工具
3. ✅ 生成流程圖到 `docs/diagrams/`
4. ✅ 顯示統計結果

### 方法 2: 單獨使用各工具

#### TypeScript Engine 分析

```bash
# 1. 安裝依賴（首次使用）
cd tools/common/development
cp ts2mermaid-package.json package.json
npm install

# 2. 分析 TypeScript Engine
npx ts-node ts2mermaid.ts \
  -i ../../../services/scan/engines/typescript_engine \
  -o ../../../docs/diagrams/typescript

# 輸出示例:
# docs/diagrams/typescript/
#   ├── src_index_Function_initialize.mmd
#   ├── src_index_Function_consumeTasks.mmd
#   ├── src_services_scan_service_Function_scan.mmd
#   └── ...
```

#### Rust Engine 分析

```bash
# 1. 設置 Cargo 項目（首次使用）
cd tools/common/development
cp rs2mermaid-Cargo.toml Cargo.toml

# 2. 分析 Rust Engine
cargo run --bin rs2mermaid -- \
  -i ../../../services/scan/engines/rust_engine \
  -o ../../../docs/diagrams/rust

# 輸出示例:
# docs/diagrams/rust/
#   ├── src_main_Function_main.mmd
#   ├── src_scanner_Function_scan.mmd
#   ├── src_endpoint_discovery_Function_discover.mmd
#   └── ...
```

## 📊 查看生成的流程圖

### 在 VS Code 中查看

1. 安裝 Mermaid 擴充功能：
   - 打開 VS Code
   - 搜尋並安裝 "Mermaid Preview" 或 "Markdown Preview Mermaid Support"

2. 打開 `.mmd` 檔案：
   ```
   docs/diagrams/typescript/src_index_Function_initialize.mmd
   ```

3. 使用預覽功能查看流程圖

### 在線查看

複製 `.mmd` 檔案內容到 [Mermaid Live Editor](https://mermaid.live/)

### 在 Markdown 中引用

```markdown
## 函數流程圖

```mermaid
<!-- 將 .mmd 檔案內容貼在這裡 -->
flowchart TB
    n1(["開始"])
    n2["初始化"]
    ...
\```
```

## 🎨 自定義選項

### 調整流程圖方向

```bash
# 從上到下（預設）
--direction TB

# 從左到右（適合寬的流程）
--direction LR

# 從右到左
--direction RL

# 從下到上
--direction BT
```

### 限制處理檔案數

```bash
# 只處理前 20 個檔案（快速測試）
--max-files 20

# 處理所有檔案
--max-files 10000
```

### 指定輸入/輸出

```bash
# 分析特定目錄
--input ./my-code

# 輸出到特定位置
--output ./my-diagrams

# 分析單一檔案
--input ./src/main.ts
```

## 📝 實際應用案例

### 案例 1: 分析 TypeScript Scan Service

```bash
cd tools/common/development

npx ts-node ts2mermaid.ts \
  -i ../../../services/scan/engines/typescript_engine/src/services \
  -o ../../../docs/diagrams/typescript/services \
  -d LR
```

生成結果：
- `scan_service_Function_scan.mmd` - 主掃描函數流程
- `enhanced_dynamic_scan_service_Function_performScan.mmd` - 增強掃描流程
- `network_interceptor_service_Function_intercept.mmd` - 網路攔截流程

### 案例 2: 分析 Rust Scanner 核心

```bash
cd tools/common/development

cargo run --bin rs2mermaid -- \
  -i ../../../services/scan/engines/rust_engine/src \
  -o ../../../docs/diagrams/rust/core \
  -d TB
```

生成結果：
- `scanner_Function_scan.mmd` - 核心掃描邏輯
- `endpoint_discovery_Function_discover_endpoints.mmd` - 端點發現
- `js_analyzer_Function_analyze.mmd` - JavaScript 分析
- `attack_surface_Function_assess.mmd` - 攻擊面評估

### 案例 3: 比較不同版本的流程

```bash
# 1. 生成當前版本流程圖
npx ts-node ts2mermaid.ts -i ./src -o ./diagrams/v1

# 2. 切換到其他分支
git checkout feature-branch

# 3. 生成新版本流程圖
npx ts-node ts2mermaid.ts -i ./src -o ./diagrams/v2

# 4. 比較差異
diff ./diagrams/v1 ./diagrams/v2
```

## 🔧 疑難排解

### 問題 1: TypeScript 工具找不到模組

**錯誤訊息**: `Cannot find module 'typescript'`

**解決方法**:
```bash
cd tools/common/development
npm install typescript @types/node ts-node
```

### 問題 2: Rust 工具編譯錯誤

**錯誤訊息**: `could not find Cargo.toml`

**解決方法**:
```bash
cd tools/common/development
cp rs2mermaid-Cargo.toml Cargo.toml
cargo build
```

### 問題 3: 生成的圖表過於複雜

**解決方法**:
1. 使用 `LR` 方向增加可讀性
2. 考慮重構複雜函數
3. 分析特定子目錄而非整個專案

### 問題 4: 無法解析某些檔案

**原因**: 檔案包含語法錯誤或不標準的語法

**解決方法**:
- 檢查終端輸出的警告訊息
- 確保程式碼能正常編譯
- 跳過問題檔案繼續處理其他檔案（工具會自動處理）

## 📚 更多資訊

- 完整文檔: `AST_ANALYSIS_TOOLS_README.md`
- Mermaid 語法: https://mermaid.js.org/
- 工具原始碼:
  - `py2mermaid.py` - Python 版本
  - `go2mermaid.go` - Go 版本
  - `ts2mermaid.ts` - TypeScript 版本
  - `rs2mermaid.rs` - Rust 版本

## 💡 提示

1. **首次使用**: 建議先用測試腳本 `test_ast_tools.ps1` 驗證所有工具
2. **大型專案**: 使用 `--max-files 50` 限制處理範圍
3. **文檔整合**: 將生成的 `.mmd` 檔案內容嵌入到 Markdown 文檔中
4. **持續監控**: 將工具整合到 CI/CD 流程，自動更新流程圖

## 🎯 下一步

1. 執行測試腳本驗證工具正常運作
2. 分析您的目標引擎代碼
3. 在文檔中整合生成的流程圖
4. 根據流程圖優化代碼結構

---

**需要幫助？** 請查看完整的 `AST_ANALYSIS_TOOLS_README.md` 文檔
