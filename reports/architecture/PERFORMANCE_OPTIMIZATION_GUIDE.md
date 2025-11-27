# AIVA 性能優化配置指南

## 📑 目錄

- [🎯 統一延遲檢查標準](#統一延遲檢查標準)
- [📋 各語言配置檔案清單](#各語言配置檔案清單)
  - [🐍 Python (Pylance)](#python-pylance)
  - [🟨 TypeScript/JavaScript (ESLint + TypeScript)](#typescriptjavascript-eslint-typescript)
  - [🟢 Go (gopls)](#go-gopls)
  - [🦀 Rust (rust-analyzer)](#rust-rustanalyzer)
- [🔧 VS Code 編輯器層級設定](#vs-code-編輯器層級設定)
  - [延遲設定](#延遲設定)
  - [關閉即時功能](#關閉即時功能)
- [� 語言伺服器詳細設定](#語言伺服器詳細設定)
  - [Python (Pylance) 完整配置](#python-pylance-完整配置)
  - [Go (gopls) 完整配置](#go-gopls-完整配置)
  - [Rust (rust-analyzer) 完整配置](#rust-rustanalyzer-完整配置)
- [�📁 檔案監控優化](#檔案監控優化)
  - [排除目錄](#排除目錄)
- [⚡ 效能優化原則](#效能優化原則)
- [🔄 重新載入設定](#重新載入設定)
- [📖 詳細設定指南](#詳細設定指南)
- [✅ 驗證設定生效](#驗證設定生效)
  - [Python](#python)
  - [TypeScript](#typescript)
  - [Go](#go)
  - [Rust](#rust)
- [🎉 預期效果](#預期效果)
- [🔗 相關資源](#相關資源)
  - [故障排除指南](#故障排除指南)
  - [修復指南](#修復指南)
  - [架構指南](#架構指南)
  - [開發指南](#開發指南)

---

## 🎯 統一延遲檢查標準
所有程式語言都遵循相同的延遲檢查原則：
- 只檢查開啟的檔案
- 程式碼變動後30秒才開始檢查
- 減少不必要的即時檢查和警告

## 📋 各語言配置檔案清單

### 🐍 Python (Pylance)
- 配置檔案: `.vscode/settings.json` ✅ (pyrightconfig.json 不存在，使用 VS Code 設定)
- 主要設定:
  - `diagnosticMode: "openFilesOnly"`
  - `diagnosticRefreshDelay: 30000`
  - `userFileIndexingLimit: -1` (無限制)

### 🟨 TypeScript/JavaScript (ESLint + TypeScript)
- 配置檔案: `services/scan/aiva_scan_node/.eslintrc.json` ✅
- 主要設定:
  - `eslint.run: "onSave"`
  - `typescript.disableAutomaticTypeAcquisition: true`
  - 關閉即時lint，只在儲存時檢查

### 🟢 Go (gopls)
- 配置檔案: `.vscode/settings.json`
- 主要設定:
  - `gopls.diagnosticsDelay: "30s"` (編輯後延遲30秒診斷)
  - `gopls.diagnosticsTrigger: "Edit"` (編輯觸發模式)
  - `go.toolsManagement.autoUpdate: false` (防止自動更新干擾)
  - `go.lintOnSave: "off"`
  - `go.vetOnSave: "off"`
  - `go.formatOnSave: false`

### 🦀 Rust (rust-analyzer)
- 配置檔案: `.vscode/settings.json`, `Cargo.toml`
- 主要設定:
  - `rust-analyzer.checkOnSave.enable: true` (只在儲存時完整檢查)
  - `rust-analyzer.checkOnSave.command: "check"` (使用cargo check)
  - `rust-analyzer.diagnostics.enable: true` (保持基本診斷)
  - `rust-analyzer.cargo.runBuildScripts: false`

## 🔧 VS Code 編輯器層級設定

### 延遲設定
- `editor.hover.delay: 3000`
- `editor.quickSuggestionsDelay: 3000`
- `editor.parameterHints.delay: 3000`
- `files.autoSaveDelay: 30000`

### 關閉即時功能
- `editor.codeLens: false`
- `editor.lightbulb.enabled: "off"`
- `editor.wordBasedSuggestions: "off"`
- `editor.inlineSuggest.enabled: false`

## � 語言伺服器詳細設定

### Python (Pylance) 完整配置
```json
{
    "python.analysis.diagnosticMode": "openFilesOnly",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.indexing": false,
    "python.analysis.userFileIndexingLimit": -1,
    "python.analysis.diagnosticRefreshDelay": 30000,
    "python.analysis.autoSearchPaths": false,
    "python.analysis.useLibraryCodeForTypes": false
}
```

### Go (gopls) 完整配置
```json
{
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

### Rust (rust-analyzer) 完整配置
```json
{
    "rust-analyzer.checkOnSave.enable": true,
    "rust-analyzer.checkOnSave.command": "check",
    "rust-analyzer.diagnostics.enable": true,
    "rust-analyzer.cargo.runBuildScripts": false,
    "rust-analyzer.procMacro.enable": false
}
```

## �📁 檔案監控優化

### 排除目錄
- Python: `__pycache__`
- TypeScript: `node_modules`, `dist`
- Go: `vendor`, `bin`
- Rust: `target`, `Cargo.lock`
- 共用: `.git`, `logs`, `models`, `backup`

## ⚡ 效能優化原則

1. **只檢查開啟檔案**: 避免掃描整個專案
2. **30秒延遲檢查**: 程式碼修改後等待30秒
3. **關閉背景索引**: 減少CPU和記憶體使用
4. **減少即時提示**: 降低編輯器卡頓
5. **優化檔案監控**: 排除不必要的目錄

## 🔄 重新載入設定

修改配置後需要重新載入VS Code視窗：
- 快捷鍵: `Ctrl+Shift+P`
- 命令: `Developer: Reload Window`

## 📖 詳細設定指南

如需了解語言伺服器的詳細優化原理和完整設定步驟，請參考：

📋 **完整指南**: [語言伺服器優化設定指南](../development/LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md)

該指南包含：
- 🔧 各語言伺服器的詳細設定說明
- ⚡ 完整的JSON配置範例  
- 📊 效果驗證和故障排除
- 💡 個人化調整建議

## ✅ 驗證設定生效

### Python
```bash
# 檢查Pylance設定
# 確認 diagnosticMode 為 "openFilesOnly"
# 確認 indexing 為 false
```

### TypeScript
```bash
# 檢查ESLint配置
npm run lint --prefix services/scan/aiva_scan_node
```

### Go
```bash
# 檢查Go模組
go mod tidy -C services/features/function_ssrf_go
```

### Rust
```bash
# 檢查Rust項目
cargo check --manifest-path services/scan/info_gatherer_rust/Cargo.toml
```

## 🎉 預期效果

配置完成後，您應該體驗到：
- ✅ 修改程式碼時不會立即觸發檢查
- ✅ 只有開啟的檔案才會被分析
- ✅ 30秒靜默時間後才開始語法檢查
- ✅ 大幅減少編輯器卡頓和CPU使用
- ✅ 所有語言都遵循相同的延遲標準

---

## 🔗 相關資源

### 故障排除指南
- 📖 [導入問題解決](./IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [前向引用修復](./FORWARD_REFERENCE_REPAIR_GUIDE.md)
- 📖 [性能優化指南](./PERFORMANCE_OPTIMIZATION_GUIDE.md)
- 📖 [測試重現指南](./TESTING_REPRODUCTION_GUIDE.md)

### 修復指南
- �� [AI 修復指南](../repairs/AIVA_AI_REPAIR_GUIDE.md)

### 架構指南
- 📖 [兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 開發指南
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

