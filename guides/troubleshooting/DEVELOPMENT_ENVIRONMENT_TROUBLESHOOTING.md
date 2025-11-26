# 開發環境配置故障排除快速參考

## 📋 目錄

- [📋 文件資訊](#文件資訊)
- [🚨 緊急故障排除](#緊急故障排除)
  - [⚡ 30秒快速診斷](#30秒快速診斷)
- [🔧 VS Code 配置問題](#vs-code-配置問題)
  - [問題 1: ESLint 設定錯誤](#問題-1-eslint-設定錯誤)
  - [問題 2: Python 套件缺失](#問題-2-python-套件缺失)
  - [問題 3: TypeScript 編譯失敗](#問題-3-typescript-編譯失敗)
- [🐍 Python 環境問題](#python-環境問題)
  - [套件依賴問題](#套件依賴問題)
  - [套件版本衝突解決](#套件版本衝突解決)
- [🔧 TypeScript 環境問題](#typescript-環境問題)
  - [Node.js 依賴問題](#nodejs-依賴問題)
  - [編譯錯誤排除](#編譯錯誤排除)
- [🐹 Go 環境問題](#go-環境問題)
  - [模組依賴問題](#模組依賴問題)
  - [Go 工具鏈問題](#go-工具鏈問題)
- [🦀 Rust 環境問題](#rust-環境問題)
  - [Cargo 建置問題](#cargo-建置問題)
  - [Rust 工具鏈問題](#rust-工具鏈問題)
- [🔍 診斷工具和命令](#診斷工具和命令)
  - [全面健康檢查腳本](#全面健康檢查腳本)
  - [VS Code 問題日誌收集](#vs-code-問題日誌收集)
- [📋 配置驗證檢查表](#配置驗證檢查表)
  - [基礎環境 ✅](#基礎環境)
  - [專案配置 ✅](#專案配置)
  - [VS Code 設定 ✅](#vs-code-設定)
  - [建置測試 ✅](#建置測試)
- [🆘 求助資源](#求助資源)
  - [官方文檔](#官方文檔)
  - [VS Code 擴展設定](#vs-code-擴展設定)

## 📋 文件資訊
- **創建日期**: 2025-10-31
- **最後更新**: 2025-11-25
- **適用場景**: 開發環境配置問題快速診斷
- **Python 環境**: 全域環境 (341 個套件)
- **狀態**: ✅ 已更新為全域環境配置

## 🚨 緊急故障排除

### ⚡ 30秒快速診斷

```powershell
# 1. 檢查基礎環境
python --version
node --version  
go version
rustc --version

# 2. 檢查 Python 套件
python -m pip list | Select-String "aiva-platform-integrated"

# 3. 檢查專案建置
cd services/features/common/typescript/aiva_common_ts && npm run build
cd ../../function_authn_go && go build .
cd ../function_sast_rust && cargo check
```

## 🔧 VS Code 配置問題

### 問題 1: ESLint 設定錯誤
```
❌ Invalid settings: setting option "experimentalWorkspaceModule"
```

**快速修復**:
```jsonc
// 在 .vscode/settings.json 中找到並修改
"gopls": {
  "diagnosticsTrigger": "Save",
  // ❌ 移除這行: "experimentalWorkspaceModule": false,
  "ui.diagnostic.analyses": {
    "unusedparams": false
  }
}
```

### 問題 2: Python 套件缺失
```
❌ ModuleNotFoundError: No module named 'xxx'
```

**診斷命令**:
```powershell
# 檢查 Python 路徑
python -c "import sys; print(sys.executable)"

# 檢查套件是否安裝
python -m pip list | Select-String "package-name"

# 重新安裝缺少的套件
pip install -r requirements.txt
pip install -e .
```

### 問題 3: TypeScript 編譯失敗
```
❌ Cannot find tsconfig.json
```

**檢查步驟**:
```powershell
# 1. 確認檔案存在
Test-Path "services/features/common/typescript/aiva_common_ts/tsconfig.json"

# 2. 檢查 .eslintrc.js 路徑
# 確保 project: './tsconfig.json' 指向正確位置
```

## 🐍 Python 環境問題

### 套件依賴問題
```powershell
# 檢查依賴衝突
pip check

# 重新安裝所有依賴
pip install -r requirements.txt --force-reinstall
pip install -e . --force-reinstall
```

### 套件版本衝突解決
```powershell
# 檢查特定套件版本
pip show package-name

# 升級特定套件
pip install --upgrade package-name

# 降級到指定版本
pip install package-name==1.2.3
```

## 🔧 TypeScript 環境問題

### Node.js 依賴問題
```powershell
# 清理並重新安裝
cd services/features/common/typescript/aiva_common_ts
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item package-lock.json -ErrorAction SilentlyContinue
npm install
```

### 編譯錯誤排除
```powershell
# 檢查 TypeScript 編譯器
npm run build 2>&1 | Tee-Object -FilePath "typescript_errors.log"

# 檢查 ESLint 規則
npm run lint -- --debug
```

## 🐹 Go 環境問題

### 模組依賴問題
```powershell
cd services/features/function_authn_go

# 清理模組緩存
go clean -modcache
go mod tidy
go mod download

# 重新建置
go build .
```

### Go 工具鏈問題
```powershell
# 檢查 Go 環境
go env GOPATH
go env GOROOT
go env GOPROXY

# 更新 Go 工具
go install -a std
```

## 🦀 Rust 環境問題

### Cargo 建置問題
```powershell
cd services/features/function_sast_rust

# 清理建置緩存
cargo clean

# 更新依賴
cargo update

# 重新檢查
cargo check --verbose
```

### Rust 工具鏈問題
```powershell
# 檢查 Rust 版本
rustc --version
cargo --version

# 更新 Rust 工具鏈
rustup update stable
rustup default stable
```

## 🔍 診斷工具和命令

### 全面健康檢查腳本
```powershell
# 創建診斷腳本
@"
Write-Host "=== AIVA 開發環境診斷 ===" -ForegroundColor Cyan

Write-Host "`n🐍 Python 環境:" -ForegroundColor Yellow
python --version
python -c "import sys; print(f'路徑: {sys.executable}')"
python -m pip list | Select-String "aiva-platform-integrated"

Write-Host "`n🔧 TypeScript 環境:" -ForegroundColor Yellow
node --version
npm --version

Write-Host "`n🐹 Go 環境:" -ForegroundColor Yellow
go version

Write-Host "`n🦀 Rust 環境:" -ForegroundColor Yellow
rustc --version
cargo --version

Write-Host "`n📁 關鍵檔案檢查:" -ForegroundColor Yellow
Test-Path ".vscode\settings.json"
Test-Path "services\features\common\typescript\aiva_common_ts\tsconfig.json"
Test-Path "services\features\function_authn_go\go.mod"
Test-Path "services\features\function_sast_rust\Cargo.toml"

Write-Host "`n🔍 建置測試:" -ForegroundColor Yellow
Write-Host "TypeScript: " -NoNewline
cd services\features\common\typescript\aiva_common_ts
if (npm run build 2>$null) { Write-Host "✅" -ForegroundColor Green } else { Write-Host "❌" -ForegroundColor Red }
cd ..\..\..\..\

Write-Host "Go: " -NoNewline
cd services\features\function_authn_go
if (go build . 2>$null) { Write-Host "✅" -ForegroundColor Green } else { Write-Host "❌" -ForegroundColor Red }
cd ..\..\..\

Write-Host "Rust: " -NoNewline
cd services\features\function_sast_rust
if (cargo check --quiet 2>$null) { Write-Host "✅" -ForegroundColor Green } else { Write-Host "❌" -ForegroundColor Red }
cd ..\..\..\

Write-Host "`n=== 診斷完成 ===" -ForegroundColor Cyan
"@ | Out-File -FilePath "diagnose_environment.ps1" -Encoding UTF8

# 執行診斷
.\diagnose_environment.ps1
```

### VS Code 問題日誌收集
```powershell
# 收集 VS Code 問題資訊
@"
{
  "timestamp": "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "vscode_version": "$(code --version)",
  "workspace_folders": [
    "$(Get-Location)"
  ],
  "settings_file_exists": $(Test-Path ".vscode\settings.json"),
  "python_interpreter": "$((Get-Content .vscode\settings.json | ConvertFrom-Json).'python.defaultInterpreterPath')",
  "active_extensions": "Run 'code --list-extensions' manually"
}
"@ | Out-File -FilePath "vscode_diagnostic.json" -Encoding UTF8
```

## 📋 配置驗證檢查表

### 基礎環境 ✅
- [ ] Python 3.13+ 安裝並可執行
- [ ] Node.js 18+ 安裝並可執行  
- [ ] Go 1.25+ 安裝並可執行
- [ ] Rust 1.90+ 安裝並可執行
- [ ] VS Code 最新版本安裝

### 專案配置 ✅
- [ ] `requirements.txt` 依賴安裝完成
- [ ] `pip install -e .` 專案安裝成功
- [ ] TypeScript 專案建置成功
- [ ] Go 模組初始化完成
- [ ] Rust 專案檢查通過

### VS Code 設定 ✅
- [ ] `.vscode/settings.json` 語法正確
- [ ] 無 ESLint 配置錯誤訊息
- [ ] Python 解譯器路徑正確
- [ ] 各語言服務啟動正常

### 建置測試 ✅
- [ ] `npm run build` 成功
- [ ] `go build .` 成功  
- [ ] `cargo check` 成功
- [ ] Python 測試執行正常

## 🆘 求助資源

### 官方文檔
- [Python 套件管理](https://docs.python.org/3/installing/index.html)
- [TypeScript 配置](https://www.typescriptlang.org/tsconfig)
- [Go 模組管理](https://golang.org/ref/mod)
- [Rust Cargo 指南](https://doc.rust-lang.org/cargo/)

### VS Code 擴展設定
- [Python 擴展](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [TypeScript 內建支援](https://code.visualstudio.com/docs/languages/typescript)
- [Go 擴展](https://marketplace.visualstudio.com/items?itemName=golang.Go)
- [Rust 擴展](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

---

**🚨 緊急情況**: 如果以上方法都無法解決，請備份專案後重新克隆倉庫並重新配置環境