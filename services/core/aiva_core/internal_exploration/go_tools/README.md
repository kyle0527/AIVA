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

## 📄 檔案詳細資訊 (Files Details)

### `go2mermaid.go`
**說明**: go2mermaid.go - All-in-One Go AST Analysis Tool  功能整合： 1. AST 解析與流程圖生成 (對標 aiva_flow_analyzer.py) 2.


### `paths_config.go`
**說明**: paths_config.go - Go 路徑配置 自動從 Python paths.py 生成


