# AIVA 統一 CLI 安裝與使用指南

## 📦 安裝

### 1. 安裝主專案（開發模式）

```powershell
cd C:\F\AIVA
python -m pip install -U pip setuptools wheel
pip install -e .
```

### 2. 安裝 aiva-contracts 工具（可選，用於型別導出）

```powershell
pip install -e tools/aiva-contracts-tooling/aiva-contracts-tooling
```

## ✅ 驗證安裝

```powershell
# 檢查 aiva 指令是否可用
aiva --help

# 應該看到：
# AIVA - AI-powered Vulnerability Analysis Platform
# 
# positional arguments:
#   {scan,detect,ai,report,system}
#     scan                掃描管理
#     detect              漏洞檢測
#     ai                  AI 訓練和管理
#     report              報告生成
#     system              系統管理
```

## 🚀 快速開始

### 掃描網站

```powershell
# 啟動掃描
aiva scan start https://example.com --max-depth 3

# 掃描並等待結果
aiva scan start https://example.com --wait
```

### 漏洞檢測

```powershell
# SQL 注入檢測
aiva detect sqli https://example.com/login --param username --wait

# XSS 檢測
aiva detect xss https://example.com/search --param q --type reflected
```

### AI 訓練

```powershell
# 實時訓練模式
aiva ai train --mode realtime --epochs 10

# 模擬訓練
aiva ai train --mode simulation --scenarios 100 --epochs 5

# 查看 AI 狀態
aiva ai status
```

### 報告生成

```powershell
# 生成 HTML 報告
aiva report generate scan_xxx --format html --output report.html

# 生成 PDF 報告
aiva report generate scan_xxx --format pdf --output report.pdf

# 生成 JSON 報告（機器可讀）
aiva report generate scan_xxx --format json --output report.json
```

### 系統管理

```powershell
# 查看系統狀態
aiva system status
```

## 🔧 進階功能

### 使用環境變數（計劃中）

```powershell
# 設定環境變數（優先級低於命令列參數）
$env:AIVA_MAX_DEPTH = "5"
$env:AIVA_TIMEOUT = "60"

aiva scan start https://example.com
```

### 使用設定檔（計劃中）

建立 `config.json`：

```json
{
  "max_depth": 3,
  "max_pages": 100,
  "timeout": 30,
  "concurrency": 8
}
```

使用：

```powershell
aiva scan start https://example.com --config config.json
```

### JSON 輸出模式（計劃中）

```powershell
# 所有指令都支援 JSON 輸出
aiva scan start https://example.com --format json

# 可用於自動化和 CI/CD 整合
aiva ai status --format json | jq '.model_params'
```

## 🔌 與 aiva-contracts 整合

### 導出 JSON Schema

```powershell
# 使用內建的 aiva-contracts 工具
aiva-contracts export-jsonschema --out _out/aiva.schemas.json

# 列出所有模型
aiva-contracts list-models
```

### 導出 TypeScript 型別定義

```powershell
# 用於前端或其他語言整合
aiva-contracts export-dts --out _out/aiva.d.ts
```

## 📊 退出碼

AIVA CLI 遵循標準退出碼規範：

- `0`: 成功
- `1`: 使用錯誤（參數錯誤）
- `2`: 系統錯誤（執行時錯誤）
- `10+`: 業務邏輯錯誤

範例：

```powershell
aiva scan start https://example.com
if ($LASTEXITCODE -eq 0) {
    Write-Host "掃描成功"
} else {
    Write-Host "掃描失敗，退出碼：$LASTEXITCODE"
}
```

## 🏗️ 架構說明

```
services/cli/
├── __init__.py           # 模組入口
├── aiva_cli.py           # 主 CLI 邏輯（現有）
├── _utils.py             # 工具函式（參數合併、輸出格式）
└── tools.py              # aiva-contracts 包裝器
```

入口點配置（在 `pyproject.toml`）：

```toml
[project.scripts]
aiva = "services.cli.aiva_cli:main"
```

## 🔄 後續增強計劃

### 階段 1：參數合併（已完成）
- ✅ 創建 `_utils.py`
- ✅ 添加 `[project.scripts]` 入口點
- ⏳ 在 `aiva_cli.py` 中整合 `merge_params()`

### 階段 2：輸出格式標準化
- ⏳ 為所有指令添加 `--format` 參數
- ⏳ 統一 JSON 輸出結構
- ⏳ 改善 human-readable 輸出（考慮使用 `rich` 庫）

### 階段 3：設定檔支援
- ⏳ 添加 `--config` 參數到所有指令
- ⏳ 支援 JSON、YAML、TOML 格式
- ⏳ 實現優先級：旗標 > 環境變數 > 設定檔

### 階段 4：多語言協定
- ⏳ 使用 JSON Schema 定義輸入/輸出
- ⏳ 建立 STDIN/STDOUT JSON 協定模式
- ⏳ 為 Go/Rust 實現準備基礎

## 🐛 故障排除

### 問題：`aiva` 指令找不到

```powershell
# 重新安裝
pip install -e . --force-reinstall

# 檢查安裝路徑
pip show aiva-platform-integrated

# 確認 Scripts 目錄在 PATH 中
$env:PATH
```

### 問題：import 錯誤

```powershell
# 確認專案結構
tree /F services\cli

# 應該看到：
# services\cli\
# ├── __init__.py
# ├── aiva_cli.py
# ├── _utils.py
# └── tools.py
```

### 問題：aiva-contracts 工具找不到

```powershell
# 確認子專案已安裝
pip list | findstr aiva

# 重新安裝
cd tools\aiva-contracts-tooling\aiva-contracts-tooling
pip install -e .
cd ..\..\..
```

## 📚 相關文件

- [快速開始指南](./QUICK_START.md)
- [AI 訓練指南](./CLI_AND_AI_TRAINING_GUIDE.md)
- [架構文件](./AI_ARCHITECTURE_ANALYSIS.md)
- [專案組織](./PROJECT_ORGANIZATION_COMPLETE.md)

## 💡 使用技巧

### Tip 1: 組合使用管道

```powershell
# 掃描 → 檢測 → 報告（自動化流程）
$scan_id = (aiva scan start https://example.com --format json | ConvertFrom-Json).scan_id
aiva detect sqli https://example.com/login --param username --wait
aiva report generate $scan_id --format pdf --output final_report.pdf
```

### Tip 2: 批量目標掃描

```powershell
# 從檔案讀取目標列表
Get-Content targets.txt | ForEach-Object {
    aiva scan start $_ --max-depth 2
}
```

### Tip 3: 監控訓練進度

```powershell
# 在背景執行訓練，定期檢查狀態
Start-Job -ScriptBlock { aiva ai train --mode realtime --epochs 100 }
while ($true) {
    aiva ai status --format json
    Start-Sleep -Seconds 30
}
```

---

**更新日期**: 2025-10-17  
**版本**: 1.0.0
