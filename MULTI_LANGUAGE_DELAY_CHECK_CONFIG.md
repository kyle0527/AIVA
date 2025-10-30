# AIVA 多語言延遲檢查配置指南
# =====================================

## 🎯 統一延遲檢查標準
所有程式語言都遵循相同的延遲檢查原則：
- 只檢查開啟的檔案
- 程式碼變動後30秒才開始檢查
- 減少不必要的即時檢查和警告

## 📋 各語言配置檔案清單

### 🐍 Python (Pylance)
- 配置檔案: `.vscode/settings.json`, `pyrightconfig.json`
- 主要設定:
  - `diagnosticMode: "openFilesOnly"`
  - `diagnosticRefreshDelay: 30000`
  - `userFileIndexingLimit: -1` (無限制)

### 🟨 TypeScript/JavaScript (ESLint + TypeScript)
- 配置檔案: `services/scan/aiva_scan_node/.eslintrc.json`
- 主要設定:
  - `eslint.run: "onSave"`
  - `typescript.disableAutomaticTypeAcquisition: true`
  - 關閉即時lint，只在儲存時檢查

### 🟢 Go (gopls)
- 配置檔案: `.vscode/settings.json`
- 主要設定:
  - `go.lintOnSave: "off"`
  - `go.vetOnSave: "off"`
  - `go.formatOnSave: false`
  - 關閉unused參數/變數警告

### 🦀 Rust (rust-analyzer)
- 配置檔案: `.vscode/settings.json`, `Cargo.toml`
- 主要設定:
  - `rust-analyzer.checkOnSave.enable: false`
  - `rust-analyzer.diagnostics.refresh.delay: 30000`
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

## 📁 檔案監控優化

### 排除目錄
- Python: `__pycache__`, `.venv`
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

## ✅ 驗證設定生效

### Python
```bash
# 檢查Pylance設定
mcp_pylance_mcp_s_pylanceSettings
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