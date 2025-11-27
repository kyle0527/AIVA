# AIVA 多語言工具整合設置

## 📑 目錄

- [📊 專案語言分布](#-專案語言分布)
- [🔧 已安裝的 VS Code 擴充功能](#-已安裝的-vs-code-擴充功能)
  - [語言支援](#語言支援)
  - [跨語言工具](#跨語言工具)
- [🛠️ AIVA 自訂工具](#-aiva-自訂工具)
  - [Python 工具](#python-工具)
  - [使用方式](#使用方式)
- [📋 質量檢查工作流程](#-質量檢查工作流程)
  - [Python 代碼](#python-代碼)
  - [Go 代碼](#go-代碼)
  - [Rust 代碼](#rust-代碼)
- [🔄 CI/CD 整合建議](#-cicd-整合建議)
  - [GitHub Actions 範例](#github-actions-範例)
- [📦 依賴管理](#-依賴管理)
  - [Dependi 使用](#dependi-使用)
  - [手動檢查](#手動檢查)
- [🎯 下一步](#-下一步)
  - [建議安裝的額外工具](#建議安裝的額外工具)
  - [待開發工具](#待開發工具)
- [📊 當前狀態](#-當前狀態)

---

## 📊 專案語言分布

根據專案掃描結果：

| 語言 | 檔案數 | 狀態 | 工具支援 |
|------|--------|------|----------|
| Python | 273 | ✅ 主要語言 | aiva-schemas-plugin, aiva-enums-plugin, aiva-contracts-tooling |
| Go | 18 | ✅ 支援 | golang.go, dependi |
| Rust | 10 | ✅ 支援 | rust-analyzer, even-better-toml, dependi |
| TypeScript | 8 | ✅ 支援 | aiva-contracts-tooling |
| Markdown | 17 | ✅ 文檔 | VS Code 內建 |

## 🔧 已安裝的 VS Code 擴充功能

### 語言支援
1. **golang.go** - Go 語言完整支援
   - 語法高亮
   - IntelliSense
   - 代碼導航
   - 格式化 (gofmt)
   - 測試執行

2. **rust-lang.rust-analyzer** - Rust 語言伺服器
   - 智能補全
   - 型別檢查
   - 內聯提示
   - 代碼導航
   - 重構工具

3. **tamasfe.even-better-toml** - TOML 檔案支援
   - 語法驗證
   - 自動補全
   - Cargo.toml 支援

4. **formulahendry.code-runner** - 多語言執行器
   - 支援 30+ 種語言
   - 快速執行代碼
   - 自訂執行命令

### 跨語言工具
5. **fill-labs.dependi** - 依賴管理器 🆕
   - Python: `requirements.txt`, `pyproject.toml`
   - Go: `go.mod`
   - Rust: `Cargo.toml`
   - JavaScript/TypeScript: `package.json`
   - PHP: `composer.json`
   - 自動檢測過時依賴
   - 安全漏洞掃描

6. **sonarsource.sonarlint-vscode** - 代碼質量分析 🆕
   - 支援：Python, Go, C/C++, Java, JavaScript/TypeScript, PHP
   - 即時靜態分析
   - 代碼異味檢測
   - 安全漏洞識別
   - 與 SonarQube 整合

## 🛠️ AIVA 自訂工具

### Python 工具
位置：`tools/`

1. **aiva-schemas-plugin** (v0.1.0)
   - 從 `aiva_common.schemas` 動態導出 Python schemas
   - 自動 re-export 所有 Pydantic 模型
   - 支援模組化結構

2. **aiva-enums-plugin** (v0.1.0)
   - 從 `aiva_common.enums` 導出枚舉
   - TypeScript 枚舉生成
   - 跨語言枚舉同步

3. **aiva-contracts-tooling** (v0.1.0)
   - JSON Schema 匯出
   - TypeScript `.d.ts` 型別定義生成
   - 跨語言契約保證

### 使用方式

#### 生成 TypeScript 型別定義
```powershell
# 使用整合腳本
.\tools\generate-contracts.ps1

# 或手動執行
cd tools/aiva-contracts-tooling/aiva-contracts-tooling
pip install -e .
aiva-contracts export-jsonschema --out ../../schemas/aiva_schemas.json
aiva-contracts gen-ts --json ../../schemas/aiva_schemas.json --out ../../schemas/aiva_schemas.d.ts
```

#### 生成 TypeScript 枚舉
```powershell
cd tools/aiva-enums-plugin/aiva-enums-plugin
pip install -e .
aiva-enums export-ts --out ../../schemas/enums.ts
```

## 📋 質量檢查工作流程

### Python 代碼
```powershell
# 格式化
python -m ruff format services/aiva_common/
python -m isort services/aiva_common/ --profile black

# 檢查
python -m ruff check services/aiva_common/ --fix
python -m flake8 services/aiva_common/ --max-line-length=120

# 型別檢查
python -m mypy services/aiva_common/

# 安全掃描
# 使用 SonarLint 自動掃描
```

### Go 代碼
```powershell
# 格式化
go fmt ./services/function/function_cspm_go/...

# 檢查
go vet ./services/function/function_cspm_go/...

# 測試
go test ./services/function/function_cspm_go/...

# 依賴更新
go get -u ./services/function/function_cspm_go/...
go mod tidy

# 安全掃描
# 使用 SonarLint 自動掃描
```

### Rust 代碼
```powershell
# 格式化
cargo fmt --manifest-path services/function/function_rust/Cargo.toml

# 檢查
cargo clippy --manifest-path services/function/function_rust/Cargo.toml

# 測試
cargo test --manifest-path services/function/function_rust/Cargo.toml

# 依賴更新
cargo update --manifest-path services/function/function_rust/Cargo.toml

# 安全審計
cargo audit --manifest-path services/function/function_rust/Cargo.toml
```

## 🔄 CI/CD 整合建議

### GitHub Actions 範例
```yaml
name: Multi-Language Quality Check

on: [push, pull_request]

jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - run: pip install ruff flake8 mypy
      - run: ruff check services/aiva_common/
      - run: flake8 services/aiva_common/
      
  go:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      - run: go fmt ./...
      - run: go vet ./...
      - run: go test ./...
      
  rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo fmt --check
      - run: cargo clippy
      - run: cargo test
```

## 📦 依賴管理

### Dependi 使用
擴充功能會自動：
- 在 `Cargo.toml`, `go.mod`, `pyproject.toml` 中顯示版本信息
- 標記過時的依賴
- 提供一鍵更新功能
- 顯示安全漏洞警告

### 手動檢查
```powershell
# Python
pip list --outdated

# Go
go list -u -m all

# Rust
cargo outdated
```

## 🎯 下一步

### 建議安裝的額外工具
- **Error Lens** - 行內錯誤顯示
- **GitLens** - Git 增強功能
- **Better Comments** - 註解增強
- **TODO Highlight** - TODO 標記高亮

### 待開發工具
1. **aiva-go-plugin** - Go 結構體生成器
   - 從 Python schemas 生成 Go structs
   - JSON tag 自動添加
   - 驗證邏輯生成

2. **aiva-rust-plugin** - Rust 型別生成器
   - 從 Python schemas 生成 Rust structs
   - Serde 支援
   - 型別安全保證

## 📊 當前狀態

✅ **完成**
- Python 工具鏈完整
- Go/Rust 語言支援
- 跨語言依賴管理
- 代碼質量檢查
- TypeScript 型別定義

⏳ **進行中**
- 自動化 CI/CD 整合
- Go/Rust 型別生成器
- 跨語言測試框架

---

**最後更新**: 2025-10-16
**維護者**: AIVA Team
