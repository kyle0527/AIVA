# VS Code 多語言開發配置最佳化指南

## 📑 目錄

- [📋 文件資訊](#文件資訊)
- [🎯 配置策略](#配置策略)
- [🔧 核心配置解析](#核心配置解析)
  - [1. Python/Pylance 最佳化策略](#1-pythonpylance-最佳化策略)
  - [2. TypeScript 平衡配置](#2-typescript-平衡配置)
  - [3. Go/gopls 精簡配置](#3-gogopls-精簡配置)
  - [4. Rust/rust-analyzer 輕量化](#4-rustrustanalyzer-輕量化)
  - [5. 編輯器通用最佳化](#5-編輯器通用最佳化)
  - [6. 檔案系統最佳化](#6-檔案系統最佳化)
- [🛠️ 語言特定設定](#語言特定設定)
  - [每種語言的獨立配置](#每種語言的獨立配置)
- [🔍 配置驗證方法](#配置驗證方法)
  - [1. 檢查配置衝突](#1-檢查配置衝突)
  - [2. 性能監測](#2-性能監測)
  - [3. 語言服務狀態](#3-語言服務狀態)
- [🚨 常見問題與解決](#常見問題與解決)
  - [1. ESLint 配置錯誤](#1-eslint-配置錯誤)
  - [2. Python 解譯器路徑問題](#2-python-解譯器路徑問題)
  - [3. TypeScript 編譯錯誤](#3-typescript-編譯錯誤)
- [📋 配置檢查清單](#配置檢查清單)
- [🎯 效能預期](#效能預期)

---
---
---
## 📋 文件資訊
- **創建日期**: 2025-10-31
- **最後更新**: 2025-10-31
- **適用版本**: VS Code 1.85+
- **狀態**: ✅ 已驗證 (10/31實測驗證)

## 🎯 配置策略

本指南確立 AIVA 專案在 VS Code 中的標準配置，重點在於：
- 🚀 **性能最佳化**: 減少背景分析和即時檢查
- 🎯 **多語言支援**: Python, TypeScript, Go, Rust 統一配置
- 🔧 **開發體驗**: 保持核心功能，去除干擾項目
- 📁 **檔案管理**: 智能排除和搜索最佳化

## 🔧 核心配置解析

### 1. Python/Pylance 最佳化策略

```jsonc
{
  // 🎯 基本設定 - 保持核心功能
  "python.testing.pytestEnabled": true,
  "python.defaultInterpreterPath": "python",
  
  // 🚀 性能最佳化 - 減少背景分析
  "python.analysis.diagnosticMode": "openFilesOnly",    // 只分析開啟的檔案
  "python.analysis.backgroundAnalysis": "off",          // 關閉背景分析
  "python.analysis.indexing": false,                    // 關閉索引建立
  "python.analysis.watchForSourceChanges": false,       // 不監視原始碼變更
  "python.analysis.diagnosticRefreshDelay": 10000,      // 延遲診斷刷新
  
  // 🎯 功能調整 - 保留重要，移除干擾
  "python.analysis.autoImportCompletions": false,       // 關閉自動匯入建議
  "python.analysis.inlayHints.variableTypes": false,    // 關閉類型提示
  "python.analysis.memory.keepLibraryAst": false        // 不保留函式庫 AST
}
```

**設計理念**: 保持類型檢查和測試功能，移除干擾性的即時分析

### 2. TypeScript 平衡配置

```jsonc
{
  // ✅ 保持核心 TypeScript 功能
  "typescript.validate.enable": true,
  "typescript.suggest.autoImports": true,
  "typescript.preferences.includePackageJsonAutoImports": "on",
  
  // 🎯 適度最佳化
  "typescript.updateImportsOnFileMove.enabled": "prompt",
  "typescript.disableAutomaticTypeAcquisition": false
}
```

**設計理念**: TypeScript 需要較多語言服務支援，保持平衡

### 3. Go/gopls 精簡配置

```jsonc
{
  // 🎯 基本功能保留
  "go.useLanguageServer": true,
  
  // 🚀 性能最佳化
  "go.lintOnSave": "off",
  "go.vetOnSave": "off",
  "go.formatOnSave": false,
  "go.toolsManagement.autoUpdate": false,
  
  // ⚠️ 重要: 移除已廢棄設定
  "gopls": {
    "diagnosticsTrigger": "Save",
    // ❌ 已移除: "experimentalWorkspaceModule": false
    "ui.diagnostic.analyses": {
      "unusedparams": false,
      "unusedvariable": false,
      "shadow": false
    }
  }
}
```

**關鍵修復**: 移除 `experimentalWorkspaceModule` 避免 ESLint 錯誤

### 4. Rust/rust-analyzer 輕量化

```jsonc
{
  // 🚀 最大化性能 - Rust 編譯較慢，減少即時檢查
  "rust-analyzer.checkOnSave.enable": false,
  "rust-analyzer.diagnostics.enable": false,
  "rust-analyzer.procMacro.enable": false,
  "rust-analyzer.cargo.runBuildScripts": false,
  "rust-analyzer.completion.autoimport.enable": false,
  "rust-analyzer.inlayHints.enable": false
}
```

**設計理念**: Rust 編譯耗時，手動觸發檢查較合適

### 5. 編輯器通用最佳化

```jsonc
{
  // 🚀 即時反應最佳化
  "editor.formatOnSave": false,
  "editor.formatOnType": false,
  "editor.codeActionsOnSave": {},
  
  // 🎯 建議系統調整
  "editor.quickSuggestions": {
    "other": false,
    "comments": false,
    "strings": false
  },
  "editor.quickSuggestionsDelay": 5000,
  "editor.hover.delay": 5000,
  
  // 📱 UI 簡化
  "editor.codeLens": false,
  "editor.inlineSuggest.enabled": false,
  "editor.lightbulb.enabled": "off"
}
```

### 6. 檔案系統最佳化

```jsonc
{
  // 📁 監視排除 - 減少檔案系統負擔
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/node_modules/**": true,
    "**/__pycache__/**": true,
    "**/target/**": true,        // Rust 建置目錄
    "**/dist/**": true,          // TypeScript 輸出
    "**/logs/**": true,          // 日誌檔案
    "**/models/**": true,        // AI 模型檔案
    "**/_out/**": true,          // 輸出目錄
    "**/_archive/**": true,      // 封存目錄
    "**/*.egg-info/**": true     // Python 套件資訊
  },
  
  // 🔍 搜尋排除 - 提升搜尋效率
  "search.exclude": {
    "**/.git": true,
    "**/node_modules": true,
    "**/__pycache__": true,
    "**/target": true,
    "**/dist": true,
    "**/logs": true,
    "**/models": true,
    "**/_out": true,
    "**/_archive": true,
    "**/backup": true
  }
}
```

## 🛠️ 語言特定設定

### 每種語言的獨立配置
```jsonc
{
  "[python]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  },
  "[typescript]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  },
  "[rust]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  },
  "[go]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  }
}
```

**原因**: 統一關閉自動格式化，手動或透過工具控制

## 🔍 配置驗證方法

### 1. 檢查配置衝突
```powershell
# 檢查 VS Code 是否有錯誤訊息
# 開啟 VS Code > View > Problems
```

### 2. 性能監測
```powershell
# 檢查 VS Code 記憶體使用
# 開啟 VS Code > Help > Toggle Developer Tools > Performance
```

### 3. 語言服務狀態
```powershell
# Python: Ctrl+Shift+P > "Python: Show Output"
# TypeScript: Ctrl+Shift+P > "TypeScript: Open TS Server Log"
# Go: Ctrl+Shift+P > "Go: Toggle Language Server Trace"
# Rust: Ctrl+Shift+P > "rust-analyzer: Status"
```

## 🚨 常見問題與解決

### 1. ESLint 配置錯誤
**錯誤**: `Invalid settings: setting option "experimentalWorkspaceModule"`

**解決**: 
```jsonc
// ❌ 錯誤配置
"gopls": {
  "experimentalWorkspaceModule": false
}

// ✅ 正確配置
"gopls": {
  "diagnosticsTrigger": "Save"
}
```

### 2. Python 解譯器路徑問題
**錯誤**: `Python interpreter not found`

**解決**:
```jsonc
// 確認使用全域 Python
"python.defaultInterpreterPath": "python"
```

### 3. TypeScript 編譯錯誤
**錯誤**: `Cannot find tsconfig.json`

**解決**:
```javascript
// .eslintrc.js 中確保路徑正確
parserOptions: {
  project: './tsconfig.json',  // 確認此檔案存在
}
```

## 📋 配置檢查清單

- [ ] VS Code settings.json 無語法錯誤
- [ ] Python 解譯器路徑正確
- [ ] TypeScript tsconfig.json 存在且有效
- [ ] Go mod 初始化完成
- [ ] Rust Cargo.toml 配置正確
- [ ] 所有語言服務啟動無錯誤
- [ ] 檔案監視排除設定生效
- [ ] 編輯器性能表現良好

## 🎯 效能預期

正確配置後應該達到：
- 🚀 **啟動速度**: VS Code 開啟時間 < 5秒
- 🎯 **回應時間**: 文件切換延遲 < 1秒  
- 💾 **記憶體使用**: 基礎使用量 < 500MB
- 🔄 **背景活動**: 最小化不必要的檔案掃描

---

**✅ 驗證狀態**: 此配置已於 2025-10-31 完整測試，多語言環境運行穩定