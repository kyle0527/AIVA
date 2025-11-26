# AIVA 擴充功能安裝指南

## 📑 目錄

- [� 目錄](#目錄)
- [�📦 插件類型說明](#插件類型說明)
  - [1️⃣ 本地 Python 插件 (會隨專案走)](#1-本地-python-插件-會隨專案走)
  - [2️⃣ VS Code 擴充功能 (不會自動帶走)](#2-vs-code-擴充功能-不會自動帶走)
- [🔄 換電腦時的操作](#換電腦時的操作)
  - [方法 1: 手動重新安裝 (推薦)](#方法-1-手動重新安裝-推薦)
  - [方法 2: 使用命令列批次安裝](#方法-2-使用命令列批次安裝)
  - [方法 3: 使用 VS Code Settings Sync](#方法-3-使用-vs-code-settings-sync)
- [📋 快速檢查清單](#快速檢查清單)
  - [確認 Python 插件已安裝](#確認-python-插件已安裝)
  - [確認 VS Code 擴充功能已安裝](#確認-vs-code-擴充功能已安裝)
- [🎯 總結](#總結)
- [🔗 相關資源](#相關資源)
  - [開發指南](#開發指南)
  - [架構指南](#架構指南)
  - [故障排除](#故障排除)
  - [使用者手冊](#使用者手冊)

---
## � 目錄

- [📦 插件類型說明](#-插件類型說明)
- [🐍 本地 Python 插件安裝](#-本地-python-插件安裝)
- [🔌 VS Code 擴充功能](#-vs-code-擴充功能)
- [⚙️ 配置與設定](#-配置與設定)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

## �📦 插件類型說明

### 1️⃣ 本地 Python 插件 (會隨專案走)

這些插件位於 `tools/` 目錄，**會被 Git 追蹤**，換電腦時會自動帶走：

```
tools/
├── aiva-schemas-plugin/     ✅ 會帶走
└── aiva-enums-plugin/       ✅ 會帶走
```

**安裝方式**：
```powershell
# 進入專案根目錄
cd C:\F\AIVA

# 安裝 schemas 插件
pip install -e tools/aiva-schemas-plugin/aiva-schemas-plugin

# 安裝 enums 插件
pip install -e tools/aiva-enums-plugin/aiva-enums-plugin
```

### 2️⃣ VS Code 擴充功能 (不會自動帶走)

這些擴充功能安裝在 `C:\Users\{你的用戶名}\.vscode\extensions\`，**不在 Git 版控內**。

**位置**: `~/.vscode/extensions/` (用戶目錄)

**已安裝的擴充功能**：
- ✅ `golang.go` - Go 語言支援
- ✅ `rust-lang.rust-analyzer` - Rust 語言伺服器
- ✅ `tamasfe.even-better-toml` - TOML 支援
- ✅ `formulahendry.code-runner` - 多語言執行器
- ✅ `fill-labs.dependi` - 跨語言依賴管理
- ✅ `sonarsource.sonarlint-vscode` - 代碼質量分析
- ✅ `ms-python.python` - Python 支援
- ✅ `ms-python.vscode-pylance` - Python 語言伺服器
- ✅ `ms-python.black-formatter` - Black 格式化
- ✅ `ms-python.isort` - Import 排序
- ✅ `charliermarsh.ruff` - Ruff 檢查器

## 🔄 換電腦時的操作

### 方法 1: 手動重新安裝 (推薦)

當你打開 AIVA 專案時，VS Code 會自動提示安裝建議的擴充功能：

1. 打開專案
2. VS Code 右下角會顯示 "此工作區建議安裝一些擴充功能"
3. 點擊 "顯示建議" 或 "全部安裝"

### 方法 2: 使用命令列批次安裝

```powershell
# 語言支援
code --install-extension golang.go
code --install-extension rust-lang.rust-analyzer
code --install-extension tamasfe.even-better-toml

# Python 工具
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension charliermarsh.ruff

# 開發工具
code --install-extension formulahendry.code-runner
code --install-extension fill-labs.dependi
code --install-extension sonarsource.sonarlint-vscode

# Git 與協作 (選用)
code --install-extension eamodio.gitlens
code --install-extension github.vscode-pull-request-github

# 開發體驗增強 (選用)
code --install-extension usernamehw.errorlens
code --install-extension aaron-bond.better-comments
code --install-extension gruntfuggly.todo-tree
```

### 方法 3: 使用 VS Code Settings Sync

1. 在當前電腦啟用 Settings Sync:
   - 按 `Ctrl+Shift+P`
   - 輸入 "Settings Sync: Turn On"
   - 選擇同步項目 (勾選 Extensions)
   - 使用 GitHub/Microsoft 帳號登入

2. 在新電腦:
   - 安裝 VS Code
   - 登入相同帳號
   - 啟用 Settings Sync
   - 自動同步所有擴充功能

## 📋 快速檢查清單

### 確認 Python 插件已安裝
```powershell
# 應該看到三個 aiva- 開頭的包
pip list | Select-String "aiva-"
```

預期輸出：
```
aiva-enums-plugin       0.1.0  ...
aiva-schemas-plugin     0.1.0  ...
```

### 確認 VS Code 擴充功能已安裝
```powershell
# 列出所有已安裝的擴充功能
code --list-extensions | Select-String "golang|rust|dependi|sonar|python"
```

預期輸出：
```
charliermarsh.ruff
fill-labs.dependi
formulahendry.code-runner
golang.go
ms-python.black-formatter
ms-python.isort
ms-python.python
ms-python.vscode-pylance
rust-lang.rust-analyzer
sonarsource.sonarlint-vscode
tamasfe.even-better-toml
```

## 🎯 總結

| 項目 | 位置 | Git 追蹤 | 換電腦 |
|------|------|----------|--------|
| **Python 插件** | `tools/` | ✅ 是 | ✅ 自動帶走 |
| **VS Code 擴充** | `~/.vscode/extensions/` | ❌ 否 | ❌ 需重裝 |
| **推薦配置** | `.vscode/extensions.json` | ✅ 是 | ✅ 自動提示安裝 |

**結論**：
- ✅ **Python 插件**會隨 Git 專案一起走
- ❌ **VS Code 擴充功能**不會，但有 3 種方式可以快速重新安裝
- 💡 建議使用 **Settings Sync** 讓所有電腦的 VS Code 設置保持一致

---

**最後更新**: 2025-10-16


---

## 🔗 相關資源

### 開發指南
- 📖 [開發快速指南](./DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [開發任務指南](./DEVELOPMENT_TASKS_GUIDE.md)
- 📖 [開發者指南](./DEVELOPER_GUIDE.md)
- 📖 [依賴管理指南](./DEPENDENCY_MANAGEMENT_GUIDE.md)
- 📖 [API 驗證指南](./API_VERIFICATION_GUIDE.md)
- 📖 [Schema 導入指南](./SCHEMA_IMPORT_GUIDE.md)

### 架構指南
- 📖 [Schema 統一指南](../architecture/SCHEMA_GUIDE.md)
- 📖 [架構兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 故障排除
- 📖 [導入問題解決](../troubleshooting/IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)

### 使用者手冊
- 📚 [Core 模組手冊](../../docs/user_guides/01_core/AIVA_CORE_使用者手冊.md)

