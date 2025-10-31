# AIVA 語言伺服器優化設定指南

> **📋 適用對象**: 開發者、DevOps工程師、VS Code用戶  
> **🎯 使用場景**: IDE性能優化、語言伺服器調優、多語言開發環境配置  
> **⏱️ 預計閱讀時間**: 20 分鐘  
> **🔧 技術需求**: VS Code、Python、Rust、Go開發環境

---

## 📑 目錄

1. [🎯 語言伺服器優化原理](#-語言伺服器優化原理)
2. [🔧 完整設定指南](#-完整設定指南)
3. [⚡ 快速配置](#-快速配置)
4. [📊 效果驗證](#-效果驗證)
5. [🛠️ 故障排除](#-故障排除)
6. [💡 最佳實踐](#-最佳實踐)

---

## 🎯 語言伺服器優化原理

### 核心優化策略
開發時可以透過設定**語言伺服器 (LSP)**或編輯器選項，來限制掃描僅針對打開的檔案，並調整診斷觸發時機和延遲。本指南針對 Python 的 **Pylance**、Rust 的 **Rust Analyzer**、Go 的 **gopls** 等主流 LSP 進行優化。

### 統一延遲檢查標準
所有程式語言都遵循相同的延遲檢查原則：
- **只檢查開啟的檔案** - 避免掃描整個專案
- **程式碼變動後30秒才開始檢查** - 減少即時分析頻率
- **減少不必要的即時檢查和警告** - 降低CPU和記憶體使用

## 🔧 完整設定指南

### 🐍 Python (Pylance)

#### 核心優化設定
- **僅分析已開啟檔案**：Pylance 提供設定 `python.analysis.diagnosticMode`，可設為 `"openFilesOnly"`，使其**只分析打開中的檔案**，不掃描整個工作區。此模式能顯著改善效能，避免對未開啟檔案進行語意分析。
- **避免專案全域索引**：為進一步降低背景掃描，可停用 Pylance 的索引功能。將 `python.analysis.indexing` 設為 `false`，或使用**輕量模式 (languageServerMode = "light")**以自動關閉索引。
- **診斷觸發延遲**：透過 `python.analysis.diagnosticRefreshDelay` 設定30秒延遲。

#### 完整 JSON 配置
```json
{
    // 🐍 Python (Pylance) 完整優化設定
    "python.analysis.diagnosticMode": "openFilesOnly",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.indexing": false,
    "python.analysis.userFileIndexingLimit": -1,
    "python.analysis.persistAllIndices": false,
    "python.analysis.diagnosticRefreshDelay": 30000,
    "python.analysis.autoSearchPaths": false,
    "python.analysis.useLibraryCodeForTypes": false,
    "python.analysis.memory.keepLibraryAst": false,
    "python.analysis.inlayHints.variableTypes": false,
    "python.analysis.inlayHints.functionReturnTypes": false,
    "python.analysis.inlayHints.callArgumentNames": "off"
}
```

### 🦀 Rust (Rust Analyzer)

#### 核心優化設定
- **預設行為**：Rust Analyzer 本身偏向延遲深入分析。**預設只在儲存檔案時**執行完整的編譯檢查（類似 `cargo check`）來提供大部分診斷。
- **僅針對已開檔案**：由於 Rust 語言特性，Rust Analyzer **無法**完全只分析單一檔案而忽略其模組/crate 其他部分。當你開啟一個 Rust 檔，RA 會解析該 crate 的 Cargo.toml，以及該 crate 中與此檔案相關的模組。
- **調整診斷觸發**：確認 `rust-analyzer.checkOnSave.enable` 為 `true`，表示**儲存時**執行 `cargo check` 獲取完整編譯診斷。

#### 完整 JSON 配置
```json
{
    // 🦀 Rust (rust-analyzer) 完整優化設定
    "rust-analyzer.checkOnSave.enable": true,
    "rust-analyzer.checkOnSave.command": "check",
    "rust-analyzer.diagnostics.enable": true,
    "rust-analyzer.diagnostics.enableExperimental": false,
    "rust-analyzer.cargo.runBuildScripts": false,
    "rust-analyzer.procMacro.enable": false
}
```

### 🟢 Go (gopls)

#### 核心優化設定
- **僅分析開啟檔案**：Go 的 gopls 預設會加載並類型檢查**整個模組**來提供完整的編輯體驗。我們可以調整診斷觸發條件，讓它**只在特定時機**執行分析。
- **診斷觸發時機**：`gopls` 提供 `diagnosticsTrigger` 設定，可在**"編輯"**或**"儲存"**時觸發診斷。預設值是 `"Edit"`，可以改為 `"Save"` 來表示**僅在檔案儲存時**才重新計算診斷。
- **延遲去抖動**：`diagnosticsDelay` 設定用於控制在編輯後等待多久才執行較昂貴的完整診斷分析。可以將其調大到 `"30s"`，達到**類似 30 秒防抖**的效果。

#### 完整 JSON 配置
```json
{
    // 🟢 Go (gopls) 完整優化設定
    "go.toolsManagement.autoUpdate": false,
    "gopls": {
        "diagnosticsDelay": "30s",
        "diagnosticsTrigger": "Edit"
    },
    "go.lintOnSave": "off",
    "go.vetOnSave": "off",
    "go.formatOnSave": false
}
```

### 🔧 VS Code 編輯器層級設定

#### 延遲設定
```json
{
    // ⏲️ 編輯器檢查延遲設定
    "editor.hover.delay": 3000,
    "editor.quickSuggestionsDelay": 3000,
    "editor.parameterHints.delay": 3000,
    "editor.suggest.delay": 1000,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 30000
}
```

#### 關閉即時功能
```json
{
    // 📝 關閉不必要的即時功能
    "editor.codeLens": false,
    "editor.inlineSuggest.enabled": false,
    "editor.lightbulb.enabled": "off",
    "editor.suggest.preview": false,
    "editor.wordBasedSuggestions": "off",
    "editor.semanticHighlighting.enabled": true
}
```

#### 檔案監控優化
```json
{
    // 🎛️ 工作區檔案監控優化
    "files.watcherExclude": {
        "**/.git/objects/**": true,
        "**/.git/subtree-cache/**": true,
        "**/node_modules/**": true,
        "**/.venv/**": true,
        "**/__pycache__/**": true,
        "_archive/**": true,
        "_out/**": true,
        "logs/**": true,
        "models/**": true,
        "backup/**": true
    }
}
```

## ⚡ 快速配置

### 完整設定檔範例
以下是一個完整的 VSCode `settings.json` 檔案範例，整合了所有語言的優化設定：

```json
{
    // 🐍 Python (Pylance) 優化設定
    "python.analysis.diagnosticMode": "openFilesOnly",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.indexing": false,
    "python.analysis.diagnosticRefreshDelay": 30000,
    
    // 🦀 Rust (rust-analyzer) 優化設定
    "rust-analyzer.checkOnSave.enable": true,
    "rust-analyzer.checkOnSave.command": "check",
    "rust-analyzer.diagnostics.enable": true,
    
    // 🟢 Go (gopls) 優化設定
    "go.toolsManagement.autoUpdate": false,
    "gopls": {
        "diagnosticsDelay": "30s",
        "diagnosticsTrigger": "Edit"
    },
    
    // 🔧 編輯器通用設定
    "editor.hover.delay": 3000,
    "editor.quickSuggestionsDelay": 3000,
    "files.autoSaveDelay": 30000,
    "editor.codeLens": false,
    "editor.lightbulb.enabled": "off"
}
```

### 工作區設定
在 `AIVA.code-workspace` 中的核心設定：

```json
{
    "settings": {
        // 🐍 Python 核心優化設定
        "python.analysis.diagnosticMode": "openFilesOnly",
        "python.analysis.typeCheckingMode": "standard", 
        "python.analysis.indexing": false,
        
        // 🦀 Rust 優化設定
        "rust-analyzer.checkOnSave.enable": true,
        "rust-analyzer.checkOnSave.command": "check",
        
        // 🟢 Go 優化設定  
        "gopls": {
            "diagnosticsDelay": "30s",
            "diagnosticsTrigger": "Edit"
        }
    }
}
```

## 📊 效果驗證

### 驗證設定生效

#### Python (Pylance)
```bash
# 使用 Pylance MCP 工具檢查設定
# 確認 diagnosticMode 為 "openFilesOnly"
# 確認 indexing 為 false
```

#### Go (gopls)
```bash
# 檢查 Go 模組是否正常
go mod tidy -C services/features/function_ssrf_go
go version
```

#### Rust (rust-analyzer)
```bash
# 檢查 Rust 項目
cargo check --manifest-path services/scan/info_gatherer_rust/Cargo.toml
rustc --version
```

### 預期效果
配置完成後，您應該體驗到：
- ✅ 修改程式碼時不會立即觸發檢查
- ✅ 只有開啟的檔案才會被分析
- ✅ 30秒靜默時間後才開始語法檢查
- ✅ 大幅減少編輯器卡頓和CPU使用
- ✅ 記憶體佔用顯著降低
- ✅ 背景進程數量減少

## 🛠️ 故障排除

### 常見問題

#### 設定未生效
1. **重新載入 VS Code**：修改設定後需要重新載入
   - 快捷鍵：`Ctrl+Shift+P`
   - 命令：`Developer: Reload Window`

2. **檢查設定層級**：確認設定是在正確的層級（使用者/工作區）

3. **語言伺服器重啟**：
   - Python：`Python: Restart Language Server`
   - Rust：`Rust Analyzer: Restart Server`
   - Go：`Go: Restart Language Server`

#### 效能問題持續
1. **檢查擴充功能**：停用不必要的語言擴充
2. **清理快取**：刪除 `.vscode` 快取檔案
3. **記憶體監控**：使用工作管理員監控記憶體使用

### 設定驗證腳本
使用專案提供的驗證腳本：
```powershell
# Windows PowerShell
.\verify-language-configs.ps1
```

## 💡 最佳實踐

### 開發建議
1. **漸進式優化**：先應用基本設定，再根據需要調整
2. **團隊一致性**：將設定加入 `.vscode/settings.json` 供團隊共享
3. **定期檢查**：定期驗證設定是否仍然適用

### 效能監控
如需監控效能改善效果，可觀察：
- CPU使用率降低
- 記憶體佔用減少  
- 編輯響應性提升
- 背景進程數量減少

### 個人化調整
根據個人偏好和專案需求，可以調整：
- 延遲時間（建議保持在15-45秒之間）
- 診斷觸發模式（編輯 vs 儲存）
- 特定語言的額外優化

---

## 🔄 設定總結

以下設定能夠實現以下行為：

1. **首次打開檔案時才啟動語意掃描**：語言伺服器僅在你以編輯器開啟檔案時，才對其進行分析初始化。未打開的檔案不會主動掃描。

2. **後續只有變更才重新分析，並延遲執行**：對已開啟檔案，只有當你修改並**儲存**或停止編輯後過一段閒置時間，語言伺服器才重新執行分析。透過延遲設定，將延後約30秒才更新診斷結果。

3. **互不影響的檔案分析**：每個打開的檔案各自觸發自己的分析流程。打開或編輯A檔案不會連帶主動重新分析B、C檔案（除非B、C自身也發生變動）。

這些設定在 VSCode 等主流編輯器中均有相應配置方式，可根據需要選擇最佳方案，有效減少本機開發時語言服務器不必要的資源佔用。

---

*最後更新: 2025-10-31*  
*基於: 調整語言伺服器預設掃描行為指南*  
*相關指南: [性能優化配置指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)*