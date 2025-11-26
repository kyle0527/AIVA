# go2mermaid.go - Go 程式碼流程圖生成工具

## 功能特點

🎯 **與 py2mermaid.py 相同的功能**，但用於 Go 語言：

- ✅ 解析 Go 語言的 AST（抽象語法樹）
- ✅ 自動生成 Mermaid 流程圖
- ✅ 支持所有 Go 控制流語句
- ✅ 為每個函數生成獨立的流程圖文件
- ✅ 可配置流程圖方向（TB/LR/RL/BT）
- ✅ 自動處理複雜的嵌套結構

## 支持的 Go 語法

### 控制流語句
- `if/else` - 條件判斷（菱形節點）
- `for` - 循環（菱形節點 + 循環邊）
- `range` - 範圍循環
- `switch/case` - 多路分支
- `select/case` - 通道選擇
- `defer` - 延遲執行
- `go` - 協程啟動
- `return` - 函數返回

### 表達式和語句
- 賦值語句（`=`, `:=`, `+=` 等）
- 函數調用
- 通道發送（`<-`）
- 自增/自減（`++`, `--`）

## 使用方法

### 1. 分析單個文件

```bash
go run go2mermaid.go -i path/to/file.go -o ./output
```

### 2. 分析整個目錄

```bash
go run go2mermaid.go -i path/to/directory -o ./output -m 1000
```

### 3. 指定流程圖方向

```bash
# 從左到右
go run go2mermaid.go -i file.go -o ./output -d LR

# 從上到下（默認）
go run go2mermaid.go -i file.go -o ./output -d TB

# 從右到左
go run go2mermaid.go -i file.go -o ./output -d RL

# 從下到上
go run go2mermaid.go -i file.go -o ./output -d BT
```

## 命令行參數

| 參數 | 簡寫 | 默認值 | 說明 |
|------|------|--------|------|
| `--input` | `-i` | - | 輸入文件或目錄路徑（必需） |
| `--output` | `-o` | `./diagrams` | 輸出目錄 |
| `--direction` | `-d` | `TB` | 流程圖方向 |
| `--max-files` | `-m` | `1000` | 最大處理文件數 |

## 輸出格式

每個函數會生成一個 `.mmd` 文件，文件名格式：

```
<source_file>_<function_name>.mmd
```

例如：
- `ssrf_testSSRF.mmd` - testSSRF 函數的流程圖
- `ssrf_Scan.mmd` - Scan 函數的流程圖
- `ssrf_scanSingleTarget.mmd` - scanSingleTarget 函數的流程圖

## 實際案例

### 分析 SSRF 檢測器

```bash
cd C:\D\fold7\AIVA-git\tools\common\development

go run go2mermaid.go \
  -i C:\D\fold7\AIVA-git\services\scan\engines\go_engine\internal\ssrf\detector\ssrf.go \
  -o ./go_diagrams \
  -d LR
```

**結果：**
```
找到 1 個 Go 文件
完成！共生成 21 個流程圖文件
輸出目錄: ./go_diagrams
```

**生成的文件：**
- `ssrf_NewSSRFDetector.mmd` - 初始化函數
- `ssrf_Scan.mmd` - 主掃描邏輯
- `ssrf_scanSingleTarget.mmd` - 單個目標掃描
- `ssrf_testSSRF.mmd` - SSRF 測試邏輯（關鍵函數）
- `ssrf_buildTestURL.mmd` - URL 構建
- `ssrf_isSSRFVulnerable.mmd` - 漏洞判斷
- ... 等 21 個文件

## testSSRF 函數流程圖解讀

從生成的 `ssrf_testSSRF.mmd` 可以清楚看到：

```mermaid
flowchart LR
    n1([開始])
    n10[L232: startTime := time.Now()]
    n11[L233: resp, err := d.client.Do(req)]  ← HTTP 請求發送點
    n12[L234: duration := time.Since(startTime)]
    n14{if err != nil}  ← 關鍵判斷點
    n15[L246: d.logger.Error(...)]
    n19[return nil]  ← 錯誤時返回 nil
    n20[]
    n22[L270: body, err := io.ReadAll(...)]  ← HTTP 響應接收點
    
    n10 --> n11
    n11 --> n12
    n12 --> n13
    n13 --> n14
    n14 -->|Yes| n15  ← 如果有錯誤
    n14 -->|No| n20   ← 如果沒有錯誤
    n15 --> n16
    ...
    n19 --> n20       ← 錯誤分支合併到主流程
    n20 --> n21
    n21 --> n22       ← 繼續讀取響應
```

**關鍵發現：**
1. ✅ HTTP 請求確實發送了（Line 233）
2. ⚠️  如果 `err != nil`（Line 245），會進入錯誤處理
3. ⚠️  錯誤處理後返回 `nil`（Line 254）
4. ✅ 但流程圖顯示有合併點（n20），說明即使有錯誤也會繼續
5. ❌ 實際代碼中 Line 254 是 `return nil`，會直接返回，不會到達 n20

## 與 py2mermaid.py 的對比

| 特性 | py2mermaid.py | go2mermaid.go |
|------|---------------|---------------|
| 語言 | Python | Go |
| AST 解析 | `ast` 模組 | `go/ast`, `go/parser` |
| 流程圖生成 | ✅ | ✅ |
| 控制流分析 | ✅ | ✅ |
| 函數級流程圖 | ✅ | ✅ |
| 模組級流程圖 | ✅ | ❌（Go 沒有模組級代碼） |
| 並發語法支持 | ❌ | ✅（`go`, `select`） |
| 泛型支持 | ✅ | ✅ |
| 錯誤處理 | `try/except` | `if err != nil` |

## 節點類型

| 類型 | 形狀 | Mermaid 語法 | 用途 |
|------|------|--------------|------|
| `start` | 圓角矩形 | `([開始])` | 函數開始 |
| `end` | 圓角矩形 | `([結束])` | 函數結束 |
| `op` | 矩形 | `[操作]` | 普通操作 |
| `cond` | 菱形 | `{條件}` | 條件判斷 |
| `subgraph` | 梯形 | `[/子圖/]` | 子圖（保留） |

## 邊的標籤

| 標籤 | 用途 |
|------|------|
| `Yes` | 條件為真 |
| `No` | 條件為假 |
| `loop` | 循環回邊 |
| `exit` | 退出循環 |

## 自動化工作流

可以將此工具整合到 CI/CD 流程中：

```yaml
# .github/workflows/generate-diagrams.yml
name: Generate Flow Diagrams

on:
  push:
    paths:
      - '**/*.go'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      
      - name: Generate diagrams
        run: |
          cd tools/common/development
          go run go2mermaid.go -i ../../../services -o ../../../docs/diagrams
      
      - name: Commit diagrams
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add docs/diagrams
          git commit -m "Update flow diagrams" || exit 0
          git push
```

## 實用技巧

### 1. 只分析特定包

```bash
go run go2mermaid.go -i ./internal/ssrf -o ./diagrams/ssrf
```

### 2. 批量轉換

```bash
# 遍歷所有子目錄
for dir in internal/*/; do
  name=$(basename "$dir")
  go run go2mermaid.go -i "$dir" -o "./diagrams/$name"
done
```

### 3. 查找複雜函數

生成流程圖後，複雜度高的函數會有更多節點：

```bash
# 統計每個流程圖的節點數
for file in diagrams/*.mmd; do
  count=$(grep -c "^\s*n[0-9]" "$file")
  echo "$count: $file"
done | sort -rn
```

## 故障排除

### 問題：編譯錯誤

確保文件語法正確：

```bash
go fmt your_file.go
go build your_file.go
```

### 問題：生成的圖為空

檢查文件是否包含函數定義：

```bash
grep -n "^func " your_file.go
```

### 問題：某些語法沒有正確顯示

支持的語法有限，可以提交 Issue 或擴展 `exprToString` 函數。

## 未來擴展

- [ ] 支持跨文件的調用鏈分析
- [ ] 生成包級別的依賴圖
- [ ] 支持接口實現關係圖
- [ ] 集成代碼複雜度分析
- [ ] 支持自定義節點樣式
- [ ] 生成互動式 HTML 流程圖

## 貢獻指南

歡迎貢獻！請確保：

1. 代碼通過 `go fmt` 格式化
2. 添加測試用例
3. 更新文檔
4. 提交 PR 前測試所有功能

## 許可證

與項目主許可證相同。

---

**作者**: AIVA 開發團隊  
**最後更新**: 2025-11-21  
**版本**: 1.0.0
