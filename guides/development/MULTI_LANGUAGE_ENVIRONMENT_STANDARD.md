# AIVA 多語言開發環境標準配置指南

## 📑 目錄

- [📋 文件資訊](#文件資訊)
- [🎯 配置目標](#配置目標)
- [🔧 VS Code 統一設定標準](#vs-code-統一設定標準)
  - [1. 工作區設定檔 (.vscode/settings.json)](#1-工作區設定檔-vscodesettingsjson)
- [🐍 Python 環境標準](#python-環境標準)
  - [1. 全域 Python 環境](#1-全域-python-環境)
  - [2. 核心依賴管理](#2-核心依賴管理)
  - [3. Python 配置檔案](#3-python-配置檔案)
    - [mypy.ini](#mypyini)
    - [pyrightconfig.json](#pyrightconfigjson)
    - [ruff.toml](#rufftoml)
- [🔧 TypeScript 環境標準](#typescript-環境標準)
  - [1. 專案結構](#1-專案結構)
  - [2. package.json 標準配置](#2-packagejson-標準配置)
  - [3. ESLint 標準配置 (.eslintrc.js)](#3-eslint-標準配置-eslintrcjs)
  - [4. tsconfig.json 標準配置](#4-tsconfigjson-標準配置)
- [🐹 Go 環境標準](#go-環境標準)
  - [1. 模組初始化](#1-模組初始化)
  - [2. go.mod 標準配置](#2-gomod-標準配置)
  - [3. 建置命令](#3-建置命令)
- [🦀 Rust 環境標準](#rust-環境標準)
  - [1. Cargo.toml 專案配置](#1-cargotoml-專案配置)
  - [2. 建置和檢查命令](#2-建置和檢查命令)
- [🔍 驗證標準流程](#驗證標準流程)
  - [1. 多語言建置測試](#1-多語言建置測試)
  - [2. 代碼品質檢查](#2-代碼品質檢查)
- [📋 故障排除](#故障排除)
  - [常見問題解決](#常見問題解決)
- [🎯 最佳實踐](#最佳實踐)

---
---
---
## 📋 文件資訊
- **創建日期**: 2025-10-31
- **最後更新**: 2025-10-31
- **適用版本**: AIVA v2.0
- **狀態**: ✅ 已驗證 (10/31實測驗證)

## 🎯 配置目標

確立 AIVA 多語言開發環境的標準配置，包含：
- Python 3.13+ 環境
- TypeScript/Node.js 環境  
- Go 1.25+ 環境
- Rust 1.90+ 環境
- VS Code 最佳化設定

## 🔧 VS Code 統一設定標準

### 1. 工作區設定檔 (.vscode/settings.json)

```jsonc
{
  // 🎯 基本 Python 環境設定
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.defaultInterpreterPath": "python",
  "python.envFile": "${workspaceFolder}/.env",

  // 🐍 Python/Pylance 最佳化設定
  "python.analysis.diagnosticMode": "openFilesOnly",
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.indexing": false,
  "python.analysis.backgroundAnalysis": "off",
  "python.analysis.watchForSourceChanges": false,
  "python.analysis.watchForLibraryChanges": false,
  "python.analysis.diagnosticRefreshDelay": 10000,
  "python.analysis.autoImportCompletions": false,
  "python.analysis.completeFunctionParens": false,
  "python.analysis.enablePytestSupport": false,
  "python.analysis.supportRestructuredText": false,
  "python.analysis.memory.keepLibraryAst": false,
  "python.analysis.inlayHints.variableTypes": false,
  "python.analysis.inlayHints.functionReturnTypes": false,
  "python.analysis.inlayHints.callArgumentNames": "off",

  // 🔧 TypeScript 最佳化設定
  "typescript.validate.enable": true,
  "typescript.check.npmIsInstalled": true,
  "typescript.updateImportsOnFileMove.enabled": "prompt",
  "typescript.suggest.autoImports": true,
  "typescript.preferences.includePackageJsonAutoImports": "on",
  "typescript.disableAutomaticTypeAcquisition": false,

  // 🦀 Rust/rust-analyzer 最佳化設定
  "rust-analyzer.checkOnSave.enable": false,
  "rust-analyzer.diagnostics.enable": false,
  "rust-analyzer.diagnostics.enableExperimental": false,
  "rust-analyzer.procMacro.enable": false,
  "rust-analyzer.cargo.runBuildScripts": false,
  "rust-analyzer.imports.granularity.enforce": false,
  "rust-analyzer.completion.autoimport.enable": false,
  "rust-analyzer.inlayHints.enable": false,

  // 🐹 Go/gopls 最佳化設定
  "go.lintOnSave": "off",
  "go.vetOnSave": "off",
  "go.formatOnSave": false,
  "go.useLanguageServer": true,
  "go.toolsManagement.autoUpdate": false,
  "gopls": {
    "diagnosticsTrigger": "Save",
    "ui.diagnostic.analyses": {
      "unusedparams": false,
      "unusedvariable": false,
      "shadow": false
    },
    "ui.completion.usePlaceholders": false
  },

  // ⚡ 編輯器通用最佳化
  "editor.formatOnSave": false,
  "editor.formatOnType": false,
  "editor.codeActionsOnSave": {},
  "editor.semanticHighlighting.enabled": false,
  "editor.codeLens": false,
  "editor.inlineSuggest.enabled": false,
  "editor.lightbulb.enabled": "off",
  "editor.suggest.preview": false,
  "editor.suggest.showKeywords": false,
  "editor.suggest.showSnippets": false,
  "editor.wordBasedSuggestions": "off",
  "editor.quickSuggestions": {
    "other": false,
    "comments": false,
    "strings": false
  },
  "editor.quickSuggestionsDelay": 5000,
  "editor.parameterHints.delay": 5000,
  "editor.hover.delay": 5000,
  "editor.suggest.delay": 2000,

  // 📁 檔案系統最佳化
  "files.autoSave": "off",
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/node_modules/**": true,
    "**/__pycache__/**": true,
    "**/target/**": true,
    "**/dist/**": true,
    "**/build/**": true,
    "**/logs/**": true,
    "**/models/**": true,
    "**/_out/**": true,
    "**/_archive/**": true,
    "**/backup/**": true,
    "**/*.egg-info/**": true
  },
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
  },

  // 🔧 語言特定設定
  "[python]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  },
  "[typescript]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  },
  "[javascript]": {
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
  },
  "[json]": {
    "editor.formatOnSave": false,
    "editor.codeActionsOnSave": {}
  }
}
```

## 🐍 Python 環境標準

### 1. 全域 Python 環境
```powershell
# 升級 pip
python -m pip install --upgrade pip

# 安裝專案依賴到全域
pip install -r requirements.txt

# 安裝專案本身 (可編輯模式)
pip install -e .
```

### 2. 核心依賴管理
```powershell
# 安裝核心依賴
pip install -r requirements.txt

# 開發依賴
pip install pytest mypy ruff black isort
```

### 3. Python 配置檔案

#### mypy.ini
```ini
[mypy]
python_version = 3.13
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

#### pyrightconfig.json
```json
{
  "include": [
    "services/**",
    "api/**",
    "utilities/**"
  ],
  "exclude": [
    "**/__pycache__",
    "**/node_modules",
    "**/target"
  ],
  "reportMissingImports": true,
  "reportMissingTypeStubs": false,
  "pythonVersion": "3.13"
}
```

#### ruff.toml
```toml
target-version = "py313"
line-length = 88
select = ["E", "F", "W", "C", "N"]
ignore = ["E501", "W503"]

[per-file-ignores]
"__init__.py" = ["F401"]
"test_*.py" = ["S101"]
```

## 🔧 TypeScript 環境標準

### 1. 專案結構
```
services/features/common/typescript/aiva_common_ts/
├── src/
├── dist/
├── package.json
├── tsconfig.json
├── .eslintrc.js
└── README.md
```

### 2. package.json 標準配置
```json
{
  "name": "@aiva/common-ts",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "lint": "eslint src/**/*.ts",
    "test": "jest"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.0.0",
    "typescript": "^5.0.0"
  }
}
```

### 3. ESLint 標準配置 (.eslintrc.js)
```javascript
module.exports = {
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    project: './tsconfig.json',
  },
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    '@typescript-eslint/recommended-requiring-type-checking',
  ],
  root: true,
  env: {
    node: true,
    jest: true,
  },
  ignorePatterns: ['.eslintrc.js', 'dist/', 'node_modules/'],
  rules: {
    '@typescript-eslint/interface-name-prefix': 'off',
    '@typescript-eslint/explicit-function-return-type': 'warn',
    '@typescript-eslint/explicit-module-boundary-types': 'warn',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/prefer-const': 'error',
    '@typescript-eslint/no-inferrable-types': 'off',
  },
};
```

### 4. tsconfig.json 標準配置
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

## 🐹 Go 環境標準

### 1. 模組初始化
```bash
go mod init github.com/kyle0527/aiva
go mod tidy
```

### 2. go.mod 標準配置
```go
module github.com/kyle0527/aiva

go 1.25

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/golang-jwt/jwt/v5 v5.0.0
)
```

### 3. 建置命令
```bash
# 建置
go build ./...

# 測試
go test ./...

# 格式化
go fmt ./...
```

## 🦀 Rust 環境標準

### 1. Cargo.toml 專案配置
```toml
[package]
name = "aiva"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
clap = { version = "4.0", features = ["derive"] }

[workspace]
members = [
    "services/features/function_sast_rust"
]
```

### 2. 建置和檢查命令
```bash
# 檢查語法
cargo check

# 建置
cargo build

# 測試
cargo test

# 格式化
cargo fmt
```

## 🔍 驗證標準流程

### 1. 多語言建置測試
```powershell
# Python 環境驗證
python -c "import sys; print(f'Python {sys.version}')"

# TypeScript 建置驗證
cd services/features/common/typescript/aiva_common_ts
npm run build

# Go 建置驗證
cd services/features/function_authn_go
go build .

# Rust 檢查驗證
cd services/features/function_sast_rust
cargo check
```

### 2. 代碼品質檢查
```powershell
# Python 代碼檢查
ruff check .
mypy services/

# TypeScript 代碼檢查
npm run lint

# Go 代碼檢查
go vet ./...

# Rust 代碼檢查
cargo clippy
```

## 📋 故障排除

### 常見問題解決

1. **ESLint 配置錯誤**
   - 移除已廢棄的設定選項 (如 `experimentalWorkspaceModule`)
   - 確保 tsconfig.json 路徑正確

2. **Python 套件問題**
   - 確認 Python 在系統 PATH 中
   - 重新安裝套件: `pip install -r requirements.txt --force-reinstall`

3. **Go 模組問題**
   - 運行 `go mod tidy` 清理依賴
   - 確認 Go 版本符合要求

4. **Rust 編譯問題**
   - 運行 `cargo clean` 清理建置緩存
   - 檢查 Cargo.toml 依賴版本

## 🎯 最佳實踐

1. **統一代碼風格**: 使用各語言標準格式化工具
2. **依賴管理**: 固定主要版本，定期更新
3. **測試覆蓋**: 每個模組都需要單元測試
4. **文檔維護**: 保持 README 和配置文件同步
5. **性能最佳化**: VS Code 設定減少背景分析

---

**✅ 驗證狀態**: 此配置已於 2025-10-31 完整驗證，所有語言環境建置成功