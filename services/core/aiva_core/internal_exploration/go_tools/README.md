# AIVA Go AST 分析工具

> **版本**: v3.1  
> **最後更新**: 2026-01-20  
> **狀態**: ✅ 生產就緒（已修復 struct 參數提取）  
> **核心文件**: go2mermaid.go  
> **代碼行數**: 891 行  
> **執行檔**: go2mermaid.exe

## ⚡ 最新更新 (2026-01-20)

### 修復：Struct 參數提取

**問題**：`go_engine` 分析結果原本是 0 flows，因為缺少 struct 參數提取

**原因**：
- Go 微服務使用 stdin JSON 接收參數
- 參數定義在 struct 欄位（如 `ScanRequest`）
- 舊版 go2mermaid 只分析函數參數，忽略 struct 欄位

**解決方案**：新增 `_convert_struct_to_flows()` 方法
```go
// 將 struct 定義轉換為虛擬流程
// ScanRequest struct → 8 個參數欄位
func _convert_struct_to_flows(structDef StructDefinition) {
    // Target, Options, PayloadType, CustomPayload...
}
```

**結果**：
- ✅ go_engine: 0 flows → 1 flow
- ✅ 成功提取 8 個參數（Target, Options, PayloadType, etc.）
- ✅ 分類器可正確識別為 "AI Core - 啟動"

**技術細節**：
- 使用 Go 標準庫 `go/ast` 解析 struct tags
- 支援 `json:"field_name"` tag 提取
- 轉換為統一的 function_details 格式

## 📑 目錄

- [📋 概述](#-概述)
- [🎯 設計定位](#-設計定位)
- [🚀 快速開始](#-快速開始)
- [📊 輸出檔案說明](#-輸出檔案說明)
- [🔧 與其他語言工具對比](#-與其他語言工具對比)
- [📝 使用注意事項](#-使用注意事項)
- [⚙️ 重新編譯](#️-重新編譯)
- [🤝 與 AIVA 核心整合](#-與-aiva-核心整合)
- [🐛 疑難排解](#-疑難排解)
- [📚 延伸閱讀](#-延伸閱讀)
- [📄 授權與維護](#-授權與維護)

---

## 📋 概述

**go_tools/** 是 AIVA 多語言 AST 分析工具套件的 Go 語言實現，專注於 **Go 代碼的 AST 解析與數據流分析**。

---

## 🎯 設計定位

根據 AIVA **雙 CLI 架構設計**，本工具專注於 **語言層** 的 AST 解析：

```
┌─────────────────────────────────────┐
│  語言工具層（AST 解析）            │
│  ├─ python_tools/                  │
│  ├─ go_tools/      ← 本工具        │
│  ├─ rust_tools/                    │
│  └─ typescript_tools/              │
└─────────────────────────────────────┘
              ↓ 輸出 JSON
┌─────────────────────────────────────┐
│  業務邏輯層（分類與執行）          │
│  ├─ aiva_internal_classifier.py   │
│  ├─ aiva_internal_executor.py     │
│  ├─ aiva_external_classifier.py   │
│  └─ aiva_external_executor.py     │
└─────────────────────────────────────┘
```

**職責範圍**：
- ✅ Go AST 解析
- ✅ 函數調用關係提取
- ✅ 數據流串接（Stitching）
- ✅ 輸出統一 JSON 格式（Schema v3.3）
- ❌ 不包含分類邏輯（由 aiva_external_classifier.py 負責）
- ❌ 不包含執行邏輯（由 aiva_external_executor.py 負責）

---

## 🚀 快速開始

### 基本用法

```powershell
# 分析當前目錄
.\go2mermaid.exe

# 指定輸入和輸出目錄
.\go2mermaid.exe --input "目標路徑" --output "./output"
```

### 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--input` | 要分析的 Go 程式碼目錄 | `.` (當前目錄) |
| `--output` | 分析結果輸出目錄 | `./analysis_output` |

---

## 📊 輸出檔案說明

執行後會在輸出目錄生成：

### analysis_results.json

**JSON 結構**（Schema v3.3）:
```json
{
  "metadata": {
    "tool": "go2mermaid",
    "version": "3.0",
    "language": "go",
    "generated_at": "2026-01-18T10:30:00",
    "total_flows": 4,
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "flow_id": 1,
      "start": {
        "module": "main",
        "function": "StartBroker"
      },
      "end": {
        "module": "messaging",
        "function": "DialBroker"
      },
      "steps": [...],
      "call_chain": ["StartBroker", "DialBroker"]
    }
  ],
  "functions": [
    {
      "name": "StartBroker",
      "module": "main",
      "file_path": "services/features/function_authn_go/main.go",
      "parameters": ["ctx context.Context", "config *BrokerConfig"],
      "calls": ["messaging.DialBroker"],
      "is_async": false
    }
  ]
}
```

**重要說明**：
- ✅ 本工具只輸出 **analysis_results.json**
- ❌ 不包含分類信息（由 aiva_external_classifier.py 處理）
- ❌ 不包含 CLI 命令（由 aiva_external_executor.py 生成）
- ❌ 不包含 Mermaid 圖表（可選功能，非核心輸出）

---

## 🔧 與其他語言工具對比

| 特性 | Python | Go | Rust | TypeScript |
|------|--------|----|----|-----------|
| **核心文件** | aiva_flow_analyzer.py | go2mermaid.go | main.rs | ts2mermaid.ts |
| **代碼行數** | 701 | 891 | 864 | 865 |
| **輸出格式** | JSON | JSON | JSON | JSON |
| **AST 解析** | ✅ ast 模組 | ✅ go/parser | ✅ syn crate | ✅ typescript API |
| **數據流串接** | ✅ | ✅ | ✅ | ✅ |
| **執行方式** | python | .exe | .exe | npx ts-node |
| **職責** | AST 解析 | AST 解析 | AST 解析 | AST 解析 |

**統一特點**：
- 所有語言工具只負責 AST 解析
- 輸出統一 JSON Schema v3.3
- 不包含分類和執行邏輯
- 保持架構對稱性

---

## 📝 使用注意事項

### ✅ 應該做的

1. **只用於 AST 解析**
   - 分析 Go 代碼結構
   - 提取函數調用關係
   - 生成標準 JSON 輸出

2. **作為語言層工具**
   - 專注於 Go 語法分析
   - 不涉及業務邏輯
   - 輸出給上層使用

### ❌ 不應該做的

1. **不要用於分類**
   - 攻擊類型分類 → 使用 `aiva_external_classifier.py`

2. **不要用於執行**
   - Go 模組執行 → 使用 `aiva_external_executor.py --lang go`

3. **不要修改輸出格式**
   - 必須保持 JSON Schema v3.3 兼容性
   - 確保與其他語言工具輸出一致

---

## 🔍 分析結果解讀

### 功能分類 (Categories)

工具會根據函數名稱和檔案路徑自動分類：

| 分類 | 關鍵字 | 說明 |
|------|--------|------|
| **reconnaissance** | scan, detect | 偵察、掃描功能 |
| **exploitation** | exploit, attack | 攻擊、利用功能 |
| **analysis** | analyze, parse | 分析、解析功能 |
| **reporting** | report, generate | 報告生成功能 |
| **persistence** | store, save, db | 資料持久化 |
| **other** | - | 其他未分類功能 |

### 瓶頸識別 (Bottleneck Analysis)

**Fan-out (扇出)**: 一個模組調用多個其他模組
- **高扇出 (> 2)**: 表示該模組責任過重，可能需要拆分

**Fan-in (扇入)**: 一個模組被多個其他模組調用
- **高扇入 (> 2)**: 表示該模組是核心依賴，變更需謹慎

**範例解讀**:
```json
"branch_analysis": {
  "fan_out_nodes": {"controller.go": 5},
  "fan_in_nodes": {"utils.go": 4}
}
```
- `controller.go` 扇出 5：過度依賴外部模組
- `utils.go` 扇入 4：核心工具模組，變更影響廣

---

## 🛠️ 進階技巧

### 1. 批次分析多個目錄

```powershell
$targets = @(
    "c:\D\fold7\AIVA-git\services\core",
    "c:\D\fold7\AIVA-git\services\api",
    "c:\D\fold7\AIVA-git\tools"
)

foreach ($target in $targets) {
    $outputName = Split-Path $target -Leaf
    .\go2mermaid.exe --input $target --output "./analysis_$outputName"
}
```

### 2. 篩選特定類別的函數

分析後可用 `jq` 或 PowerShell 篩選：

```powershell
# 讀取 JSON 並篩選 reconnaissance 類別
$json = Get-Content "./analysis_output/analysis_results.json" | ConvertFrom-Json
$reconFuncs = $json.functions | Where-Object { $_.category -eq "reconnaissance" }
$reconFuncs | Format-Table function_name, source_file
```

### 3. 比較版本差異

```powershell
# 分析舊版本
git checkout v1.0
.\go2mermaid.exe --input "." --output "./analysis_v1"

# 分析新版本
git checkout v2.0
.\go2mermaid.exe --input "." --output "./analysis_v2"

# 比較 JSON 報告
Compare-Object `
  (Get-Content "./analysis_v1/analysis_results.json") `
  (Get-Content "./analysis_v2/analysis_results.json")
```

---

## ⚙️ 重新編譯

如需修改工具源碼後重新編譯：

```powershell
cd c:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\go_tools

# 編譯
go build -o go2mermaid.exe go2mermaid.go

# 測試
.\go2mermaid.exe --input "." --output "./test_output"
```

---

## 📝 注意事項

### 1. 僅分析 Go 檔案
- 工具僅處理 `.go` 檔案
- 自動跳過 `_test.go` 測試檔案

### 2. 跨 Package 調用解析
- 目前採用「模糊匹配」策略
- 標準庫和第三方庫調用會被忽略
- 僅追蹤專案內部的跨檔案調用

### 3. 大型專案建議
- 建議分批分析子目錄，避免輸出過多檔案
- 大型專案的 `system_flow.mmd` 可能非常複雜

### 4. Mermaid 圖表預覽
- 推薦安裝 VS Code 擴充套件: `Markdown Preview Mermaid Support`
- 或使用線上工具: https://mermaid.live/

---

## 🆚 與 Python 工具對比

| 功能 | Python 工具 | Go 工具 (本工具) |
|------|-------------|------------------|
| **流程圖生成** | `aiva_flow_analyzer.py` | ✅ 整合 |
| **數據流串接** | `aiva_flow_analyzer.py` | ✅ 整合 |
| **功能分類** | `aiva_flow_classifier.py` | ✅ 整合 |
| **CLI 生成** | `aiva_cli_implementation.py` | ✅ 整合 |
| **系統分析** | `aiva_exploration_pipeline.py` | ✅ 整合 |
| **語言** | Python (解釋執行) | Go (編譯執行) |
| **執行效率** | 較慢 | 快 10-100 倍 |
| **單檔整合** | 4 個獨立腳本 | 1 個可執行檔 |

---

## 🐛 疑難排解

### 問題 1: 編譯失敗

**錯誤訊息**: `cannot find package ...`

**解決方法**:
```powershell
go mod init go2mermaid
go mod tidy
go build -o go2mermaid.exe go2mermaid.go
```

### 問題 2: 中文亂碼

**原因**: PowerShell 編碼問題

**解決方法**:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
.\go2mermaid.exe --input "." --output "./output"
```

### 問題 3: 無法找到跨檔案連接

**原因**: 
- 可能是標準庫調用 (會被忽略)
- Package 名稱解析失敗

**檢查方法**:
```powershell
# 查看 analysis_results.json 中的 flow_chains
$json = Get-Content "./analysis_output/analysis_results.json" | ConvertFrom-Json
$json.flow_chains
```

---

## 📚 延伸閱讀

- [Mermaid 語法文檔](https://mermaid.js.org/)
- [Go AST 官方文檔](https://pkg.go.dev/go/ast)
- [AIVA 專案架構說明](../../_PROJECT_STRUCTURE_OPTIMIZATION_RECOMMENDATIONS.md)

---

## 📄 授權與維護

- **版本**: 1.0 (整合版)
- **最後更新**: 2025-12-10
- **維護者**: AIVA Team
- **對應 Python 工具**: `python_tools/` 目錄下的 4 個腳本

如有問題或建議，請參考專案根目錄的貢獻指南。
