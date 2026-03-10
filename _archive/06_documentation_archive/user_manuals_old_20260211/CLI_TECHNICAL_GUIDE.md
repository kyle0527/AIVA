# AIVA CLI 統一使用指南

> **版本**: v1.1  
> **更新日期**: 2026-01-28  
> **狀態**: ✅ 已驗證並更新  
> **適用對象**: 所有 AIVA 使用者

---

## 📑 目錄

1. [系統概述](#系統概述)
2. [CLI 系統說明](#cli-系統說明)
3. [快速啟動方式](#快速啟動方式)
4. [統一 CLI 系統使用](#統一-cli-系統使用)
5. [常見問題](#常見問題)
6. [故障排除](#故障排除)

---

## 系統概述

AIVA 提供**統一的 CLI 系統**，基於動態 Flow 執行架構：

| 系統 | 用途 | 入口文件 | 主要功能 |
|------|------|---------|---------|
| **統一 CLI 系統** | Flow 執行、能力調用、批次處理 | `services/core/aiva_core/core_capabilities/cli/aiva_cli.py` | 動態 Flow 執行、參數化命令 |

---

## CLI 系統說明

### 🎯 統一 CLI 系統（基於動態 Flow）

**文件位置**: `C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\cli\aiva_cli.py`

**核心功能**:
- 🔀 執行動態 Flow 工作流（從 latest_classification.json 讀取）
- 📦 自動在模組間傳遞數據
- 🎯 支持多種參數格式（target, data, query, param）
- 🧪 Dry-run 模式預覽執行計畫
- 💡 AI 強度控制（0.0-1.0）
- 📊 統一的錯誤處理和日誌

**架構特點**:
```
latest_classification.json  ← Flow 定義來源
         ↓
  FlowExecutor 引擎
         ↓
  動態模組導入 + 執行
         ↓
      結果輸出
```

**適用場景**:
- 執行特定的攻擊流程
- 串接多個模組進行複雜分析
- 測試和驗證新開發的能力
- 批次處理和自動化任務

---

## 快速啟動方式

### ⚡ 方式一：使用批次檔（推薦新手）

**位置**: 專案根目錄 `C:\D\fold7\AIVA-git\`

| 文件 | 功能 | 說明 |
|------|------|------|
| `啟動能力選單.bat` | 啟動互動式能力選單 | 舊版，可能需要更新 |
| `執行Flow.bat [ID]` | 執行指定 Flow | 需要更新以使用新 CLI |
| `預覽Flow.bat [ID]` | 預覽 Flow（不執行） | 需要更新以使用新 CLI |

> ⚠️ **注意**: 批次檔可能需要更新以指向新的 CLI 入口點

---

### 🖥️ 方式二：直接使用 CLI 命令（推薦）

**基本格式**:
```powershell
cd C:\D\fold7\AIVA-git
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli [COMMAND] [OPTIONS]
```

#### 常用命令範例

```powershell
# 列出所有可用的 Flows
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli list

# 執行指定 Flow（帶目標參數）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 --target https://example.com --intensity 0.8

# 執行 Flow（帶數據路徑）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow1 --data /path/to/data.json

# 執行 Flow（帶查詢字串）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow2 --query "SQL injection test"

# 執行 Flow（帶多個自定義參數）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow3 --param key1=value1 --param key2=value2

# Dry Run 模式（預覽但不執行）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 --target https://example.com --dry-run

# 調整 AI 強度（0.0-1.0）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 --target https://example.com --intensity 0.3
```

---

## 統一 CLI 系統使用

### 命令格式說明

```bash
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli <flow_command> [OPTIONS]
```

### 可用選項

| 選項 | 簡寫 | 類型 | 說明 | 範例 |
|------|------|------|------|------|
| `--target` | `-t` | string | 目標 URL/路徑/對象 | `--target https://example.com` |
| `--data` | `-d` | string | 數據文件路徑 | `--data /path/to/data.json` |
| `--query` | `-q` | string | 查詢字串 | `--query "SQL injection"` |
| `--param` | `-p` | multiple | 額外參數 (key=value) | `--param timeout=30` |
| `--intensity` | `-i` | float | AI 強度 (0.0-1.0) | `--intensity 0.8` |
| `--dry-run` | - | flag | 預覽模式，不實際執行 | `--dry-run` |

### Flow 命令列表

執行 `list` 命令查看所有可用的 Flow：
```powershell
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli list
```

輸出示例：
```
📋 已載入 210 個 flows (來源: latest_classification.json)

可用 Flow 命令:
  flow0    - [模組名] 執行第 0 個 Flow (長度: 3)
  flow1    - [模組名] 執行第 1 個 Flow (長度: 4)
  flow2    - [模組名] 執行第 2 個 Flow (長度: 2)
  ...
```

### 實際使用範例

#### 範例 1: 基本 Web 掃描

```powershell
# 執行 Web 掃描 Flow
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --target https://example.com \
  --intensity 0.5

# 輸出:
# 🚀 執行 Flow 0
#    模組: scan
#    路徑: initial_scan -> ...
#    AI 強度: 0.50
# 
# ✅ Flow 0 執行完成
# 📊 結果: {...}
```

#### 範例 2: SQL 注入測試（帶參數）

```powershell
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow15 \
  --target https://example.com/login \
  --param method=POST \
  --param param=username \
  --intensity 0.7
```

#### 範例 3: 數據分析 Flow

```powershell
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow5 \
  --data C:\data\scan_results.json \
  --param format=json \
  --intensity 0.6
```

#### 範例 4: Dry Run 預覽

```powershell
# 預覽 Flow 執行計畫但不實際運行
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --target https://example.com \
  --dry-run

# 輸出:
# 🔍 Dry Run 模式
# 📋 Flow 0 執行計畫:
#    步驟 1: initial_scan.InitialScan.execute()
#    步驟 2: analyzer.VulnAnalyzer.analyze()
#    步驟 3: reporter.ReportGenerator.generate()
# 
# ℹ️  預覽完成，未實際執行
```

#### 範例 5: 調整 AI 強度

```powershell
# 低強度（快速掃描）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --target https://example.com \
  --intensity 0.2

# 高強度（深度分析）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --target https://example.com \
  --intensity 0.9
```

### 批次執行範例

```powershell
# PowerShell 腳本批次執行多個目標
$targets = @(
    "https://example1.com",
    "https://example2.com",
    "https://example3.com"
)

foreach ($target in $targets) {
    Write-Host "掃描: $target"
    python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 `
        --target $target `
        --intensity 0.5
    Start-Sleep -Seconds 5
}
```

---

## 常見問題

### Q1: 如何查看所有可用的 Flow？

```powershell
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli list
```

這會列出從 `latest_classification.json` 讀取的所有 Flow 定義。

### Q2: Flow 數量為什麼和之前不同？

Flow 數量會隨系統更新而變化。目前的 Flow 數量由 `latest_classification.json` 決定，通常在 200-300 個之間。

### Q3: 如何知道某個 Flow 做什麼？

使用 Dry Run 模式查看 Flow 的執行計畫：

```powershell
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --target https://example.com \
  --dry-run
```

### Q4: --param 參數如何使用？

`--param` 接受 `key=value` 格式，可以多次使用：

```powershell
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --param timeout=30 \
  --param method=POST \
  --param retries=3
```

### Q5: AI 強度（intensity）是什麼？

AI 強度控制執行的深度和資源使用：
- `0.0-0.3`: 快速掃描，低資源消耗
- `0.4-0.6`: 標準掃描（預設 0.5）
- `0.7-1.0`: 深度分析，高資源消耗

### Q6: 如何更新 Flow 定義？

Flow 定義來自 `latest_classification.json`，通常位於：
- `C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json`

重新生成此文件後，CLI 會自動讀取最新定義。

### Q7: 批次檔 (.bat) 還能用嗎？

批次檔需要更新以使用新的 CLI 入口點。建議直接使用 Python 命令：

```powershell
# 舊版（可能不工作）
.\執行Flow.bat 11

# 新版（推薦）
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow11 --target URL
```

---

## 故障排除

### 問題 1: 找不到模組

**錯誤訊息**: `ModuleNotFoundError: No module named 'services'`

**解決方案**:
```powershell
# 確保在專案根目錄執行
cd C:\D\fold7\AIVA-git

# 或設定 PYTHONPATH
$env:PYTHONPATH = "C:\D\fold7\AIVA-git"
```

### 問題 2: 找不到 Flow 定義文件

**錯誤訊息**: `⚠️ 未找到 flow 定義文件`

**解決方案**:
檢查 `latest_classification.json` 是否存在：
```powershell
Test-Path "C:\D\fold7\AIVA-git\services\integration\data\internal_exploration\latest_classification.json"
```

如不存在，需要重新生成分類數據。

### 問題 3: 編碼錯誤

**錯誤訊息**: `UnicodeEncodeError: 'cp950' codec can't encode`

**解決方案**:
```powershell
# PowerShell 設定 UTF-8
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 執行命令
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli list
```

### 問題 4: Flow 執行失敗

**症狀**: Flow 開始執行但中途報錯

**解決方案**:
```powershell
# 1. 使用 Dry Run 檢查執行計畫
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 --dry-run

# 2. 檢查日誌文件
cat logs/aiva_cli.log  # Linux/macOS
type logs\aiva_cli.log  # Windows

# 3. 降低 AI 強度重試
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --target URL \
  --intensity 0.2
```

### 問題 5: 參數傳遞不正確

**症狀**: 傳遞的參數沒有被正確識別

**解決方案**:
```powershell
# 確保參數格式正確
--param key=value    # ✅ 正確
--param key value    # ❌ 錯誤
--param "key=value"  # ✅ 正確（有空格時需引號）

# 範例
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 \
  --param "user agent=Mozilla/5.0" \
  --param timeout=30
```

### 問題 6: 命令行太長

**症狀**: Windows 命令行字符限制

**解決方案**:
使用 PowerShell 變數簡化命令：
```powershell
$cli = "python -m services.core.aiva_core.core_capabilities.cli.aiva_cli"
$target = "https://example.com"

& $cli flow0 --target $target --intensity 0.5
```

或創建批次腳本：
```powershell
# run_flow.ps1
param(
    [string]$FlowId,
    [string]$Target,
    [double]$Intensity = 0.5
)

$cli = "python -m services.core.aiva_core.core_capabilities.cli.aiva_cli"
& $cli "flow$FlowId" --target $Target --intensity $Intensity
```

使用：
```powershell
.\run_flow.ps1 -FlowId 0 -Target "https://example.com" -Intensity 0.7
```

---

## 進階技巧

### 技巧 1: 批次執行多個 Flows

```powershell
# PowerShell 腳本
$flows = 0..10
$target = "https://example.com"

foreach ($id in $flows) {
    Write-Host "執行 Flow $id ..."
    python -m services.core.aiva_core.core_capabilities.cli.aiva_cli "flow$id" `
        --target $target `
        --intensity 0.5
    Start-Sleep -Seconds 2
}
```

### 技巧 2: 結果記錄到文件

```powershell
# 將輸出記錄到文件
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 `
    --target https://example.com `
    --intensity 0.5 `
    | Tee-Object -FilePath "flow0_result.txt"
```

### 技巧 3: 條件執行

```powershell
# 只有當前一個 Flow 成功時才執行下一個
python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow0 --target URL
if ($LASTEXITCODE -eq 0) {
    python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow1 --target URL
}
```

### 技巧 4: 並行執行（謹慎使用）

```powershell
# PowerShell 並行作業
$jobs = @()
$flows = 0..5

foreach ($id in $flows) {
    $jobs += Start-Job -ScriptBlock {
        param($id)
        python -m services.core.aiva_core.core_capabilities.cli.aiva_cli "flow$id" `
            --target "https://example.com" `
            --intensity 0.3
    } -ArgumentList $id
}

# 等待所有作業完成
$jobs | Wait-Job | Receive-Job
```

---

## 參考資源

- **CLI 源碼**: `services/core/aiva_core/core_capabilities/cli/aiva_cli.py`
- **Flow 定義**: `services/integration/data/internal_exploration/latest_classification.json`
- **FlowExecutor 引擎**: `services/core/aiva_core/internal_exploration/flow_executor.py`

---

## 版本歷史

| 版本 | 日期 | 變更說明 |
|------|------|---------|
| v1.1 | 2026-01-28 | 重大更新：統一為單一 CLI 系統，基於動態 Flow 架構 |
| v1.0 | 2026-01-10 | 初始版本，兩套 CLI 系統說明 |

---

## 重要變更通知

> ⚠️ **2026-01-28 架構變更**
> 
> AIVA CLI 已統一為單一系統，基於動態 Flow 執行架構。
> 
> **主要變更**:
> - 新 CLI 入口點: `services/core/aiva_core/core_capabilities/cli/aiva_cli.py`
> - 統一的命令格式: `python -m services.core.aiva_core.core_capabilities.cli.aiva_cli`
> - 動態 Flow 執行: 從 `latest_classification.json` 讀取定義
> - 支持參數化命令: `--target`, `--data`, `--query`, `--param`
> - AI 強度控制: `--intensity` (0.0-1.0)
> - Dry Run 預覽: `--dry-run`
> 
> **舊版 CLI 參考**:
> - `scripts/common/aiva_cli.py` - 可能已過時
> - `internal_exploration/python_tools/aiva_cli_implementation.py` - 可能已整合

---

**維護者**: AIVA 開發團隊  
**更新頻率**: 隨系統更新  
**反饋**: 發現問題請提交 Issue
