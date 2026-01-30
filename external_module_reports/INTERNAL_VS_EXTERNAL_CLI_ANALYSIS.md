# AIVA CLI 系統架構分析報告

**生成日期**: 2026-01-13  
**分析範圍**: 內部 CLI vs 外部 CLI  
**核心發現**: 兩者架構完全相同，只是數據來源不同

---

## 📊 核心架構對比

| 特性 | 內部 CLI | 外部 CLI | 說明 |
|------|---------|---------|------|
| **核心實現** | `aiva_cli_implementation.py` | `aiva_external_module_cli.py` | 相同設計模式 |
| **數據來源** | `latest_classification.json` | `analysis_results.json` | 不同的分析結果 |
| **支援語言** | Python (主要) | Python + Rust + Go + TypeScript | 外部支援更多語言 |
| **Flow 數量** | 286 條 | 222 條 | 內部包含 Core 系統 |
| **執行方式** | FlowExecutor 動態執行 | 同樣的 FlowExecutor | 完全一致 |
| **啟動腳本** | `.bat` 檔案 | `.bat` 檔案 | Windows 快速啟動 |

---

## 🎯 .bat 檔案功用分析

### 內部模組 CLI 檔案

#### 1. `執行Flow.bat` ⭐
**功用**: 執行指定 Flow ID 的數據流

**實際命令**:
```bat
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_cli_implementation.py --flow [ID]
```

**使用範例**:
```bat
執行Flow.bat 11
# 執行 Flow ID=11: unified_executor -> capability_orchestrator
```

**核心功能**:
- 讀取 `latest_classification.json` 中的 Flow 定義
- 動態導入 Python 模組
- 自動推斷類別名稱 (snake_case → CamelCase)
- 啟發式偵測入口方法 (train, execute, run, process, analyze)
- Pipeline 執行：自動傳遞步驟間數據

---

#### 2. `預覽Flow.bat` ⭐
**功用**: Dry Run 模式，只顯示執行計畫不實際運行

**實際命令**:
```bat
python aiva_cli_implementation.py --flow [ID] --dry-run
```

**使用範例**:
```bat
預覽Flow.bat 11
# 顯示：
#   Step 1/2: 導入 aiva_core.core_capabilities.unified_executor
#   Step 2/2: 實例化 CapabilityOrchestrator
#   入口方法: train/execute/run (按優先級嘗試)
```

**核心功能**:
- 顯示完整執行計畫
- 驗證模組路徑是否正確
- 檢查類別是否存在
- 預覽方法調用順序
- 不實際執行任何代碼

---

#### 3. `啟動能力選單.bat` ⭐
**功用**: 互動式選單，列出所有可用能力

**實際命令**:
```bat
python aiva_cli_implementation.py --menu
```

**顯示內容**:
```
========================================
  AIVA 能力執行選單
========================================

可用 Flow 列表:
ID: 1  | Path: backends -> unified_executor
ID: 2  | Path: task_executor -> unified_function_caller
...
ID: 286 | Path: xxx -> yyy

請選擇要執行的 Flow ID (或輸入 q 退出): 
```

**核心功能**:
- 列出所有 286 條 Flow
- 互動式選擇執行
- 即時反饋執行結果
- 錯誤處理和重試

---

### 外部模組 CLI 檔案

#### 4. `外部模組選單.bat`
**功用**: 外部功能模組選單（XSS、SQLi、SSRF 等）

**實際命令**:
```bat
python aiva_external_module_cli.py --lang python --flow 1 --target [URL]
```

**選單內容**:
```
1. SQL Injection (function_sqli)
2. XSS Detection (function_xss)
3. SSRF Detection (function_ssrf)
4. IDOR Detection (function_idor)
5. Business Logic (function_bizlogic)
6. List All Capabilities
0. Exit
```

**核心功能**:
- 外部攻擊功能模組選單
- 需要輸入目標 URL
- 執行實際的安全測試
- 支援 Python 功能模組

---

#### 5. `執行外部模組.bat`
**功用**: 直接命令行執行外部模組（不使用 Flow）

**實際命令**:
```bat
python scripts\run_module.py [module] [url]
```

**使用範例**:
```bat
執行外部模組.bat xss http://localhost:3000
執行外部模組.bat sqli http://testphp.vulnweb.com/artists.php
```

**核心功能**:
- **Direct Import 模式** - 直接調用 Detector 類
- 不經過 Flow 系統
- 最快速的執行方式
- 適合快速測試單一模組

---

#### 6. `分類外部模組.bat`
**功用**: 批次分析外部模組，生成分類報告

**實際命令**:
```bat
python aiva_external_module_batch_classifier.py -w [input] -o [output]
```

**使用範例**:
```bat
分類外部模組.bat module_analysis external_module_reports
# 掃描 module_analysis/ 下的所有模組
# 生成報告到 external_module_reports/
```

**核心功能**:
- 掃描所有外部模組的 analysis_results.json
- 統計流程數、語言分布
- 生成整合報告 (Markdown + JSON)
- 攻擊類型分類

---

#### 7. `啟動AIVA系統.bat`
**功用**: 啟動完整的 AIVA 系統（包含所有服務）

**核心功能**:
- 啟動 Core 服務
- 啟動 RabbitMQ/Redis
- 啟動 Web UI
- 初始化所有模組

---

## 🔍 內部 CLI vs 外部 CLI 詳細對比

### 架構設計

#### 內部 CLI (`aiva_cli_implementation.py`)

**數據來源**:
```python
# 優先使用 latest_classification.json (由 pipeline 自動更新)
LATEST_DATA_PATH = "latest_classification.json"
# 後備: classification_data.json
```

**核心類別**: `FlowExecutor`

**執行流程**:
```
1. 讀取 latest_classification.json (286 flows)
2. 根據 Flow ID 找到對應的 flow 定義
3. 解析 flow.path: ["file1", "file2", "file3"]
4. 動態導入: from aiva_core.xxx import Yyy
5. 實例化類別: instance = Yyy()
6. 啟發式偵測入口方法: train() / execute() / run()
7. Pipeline 執行: output1 -> input2 -> output2
```

**支援的模組**:
- cognitive_core (認知核心)
- internal_exploration (內探)
- task_planning (任務規劃)
- core_capabilities (核心能力)
- service_backbone (服務骨幹)

---

#### 外部 CLI (`aiva_external_module_cli.py`)

**數據來源**:
```python
ANALYSIS_PATHS = {
    "python": "features_classification/classification_data.json",
    "rust": "function_crypto/rust_core/analysis_results.json",
    "go": "function_authn_go/analysis_output/analysis_results.json",
    "typescript": "typescript_engine/analysis_output/analysis_results.json",
}
```

**核心類別**: `MultiLangExecutor`

**執行流程**:
```
1. 讀取各語言的 analysis_results.json
2. 根據語言類型選擇執行方式：
   - Python: 同樣使用 FlowExecutor 動態執行
   - Rust: subprocess.run(["cargo", "run", "--", args])
   - Go: subprocess.run(["go", "run", "main.go", args])
   - TypeScript: subprocess.run(["npx", "ts-node", "index.ts", args])
3. 傳遞參數和目標 URL
4. 收集執行結果
```

**支援的模組**:
- function_xss (XSS 檢測)
- function_sqli (SQL 注入)
- function_ssrf (SSRF 檢測)
- function_idor (IDOR 檢測)
- function_bizlogic (業務邏輯)
- function_authn_go (Go 認證模組)
- function_crypto (Rust 加密模組)
- typescript_engine (TypeScript 引擎)

---

## 💡 關鍵發現

### 1. 架構完全相同 ✅

兩者都使用：
- `FlowExecutor` 類別進行動態執行
- JSON 配置文件定義 Flow
- 相同的啟發式方法偵測邏輯
- 相同的 Pipeline 數據傳遞機制

**唯一差異**:
- **數據來源不同**: 內部讀 `latest_classification.json`，外部讀 `analysis_results.json`
- **模組範圍不同**: 內部是 Core 系統，外部是攻擊功能模組

---

### 2. Flow 格式統一 ✅

內部和外部的 Flow 格式完全一致：

```json
{
  "id": 11,
  "path": ["unified_executor", "capability_orchestrator"],
  "full_path": [
    "aiva_core/core_capabilities/unified_executor.py",
    "aiva_core/core_capabilities/capability_orchestrator.py"
  ],
  "length": 2,
  "start": "unified_executor",
  "end": "capability_orchestrator",
  "classifications": ["core_capabilities"],
  "language": "Python"
}
```

---

### 3. 執行方式統一 ✅

#### Python 模組 (內部 + 外部)
```python
# 1. 動態導入
module = importlib.import_module("aiva_core.xxx.yyy")

# 2. 類別名稱推斷
class_name = "Yyy"  # snake_case -> CamelCase

# 3. 實例化
instance = getattr(module, class_name)()

# 4. 啟發式方法偵測
for method in ["train", "execute", "run", "process", "analyze"]:
    if hasattr(instance, method):
        result = getattr(instance, method)()
        break
```

#### Rust/Go/TypeScript 模組 (外部)
```python
# 直接調用編譯後的可執行檔或腳本
subprocess.run([
    "cargo", "run", "--",  # Rust
    # "go", "run", "main.go",  # Go
    # "npx", "ts-node", "index.ts",  # TypeScript
    "--cookies-json", json.dumps(cookies),
    "--url", target_url
])
```

---

## 🎯 如何執行

### 內部 CLI 執行方式

#### 方式 1: 使用 .bat 檔案 (推薦 Windows 用戶)

```bat
# 列出所有 Flow
啟動能力選單.bat

# 執行特定 Flow
執行Flow.bat 11

# 預覽執行計畫
預覽Flow.bat 11
```

#### 方式 2: 直接使用 Python

```bash
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 列出所有 Flow
python aiva_cli_implementation.py --list

# 執行 Flow
python aiva_cli_implementation.py --flow 11

# Dry Run
python aiva_cli_implementation.py --flow 11 --dry-run

# 生成文檔
python aiva_cli_implementation.py --generate-doc md
```

---

### 外部 CLI 執行方式

#### 方式 1: 使用 .bat 檔案

```bat
# 互動式選單
外部模組選單.bat

# 直接執行
執行外部模組.bat xss http://localhost:3000

# 批次分類
分類外部模組.bat module_analysis reports
```

#### 方式 2: 直接使用 Python

```bash
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 列出所有外部模組
python aiva_external_module_cli.py --list

# 列出特定語言
python aiva_external_module_cli.py --list --lang python

# 執行 Python Flow
python aiva_external_module_cli.py --lang python --flow 1 --target http://example.com

# 執行 Rust 功能
python aiva_external_module_cli.py --lang rust --func analyze_cookies \
    --cookies-json '[...]' --url https://example.com

# Dry Run
python aiva_external_module_cli.py --lang rust --func analyze_cookies --dry-run
```

---

## 📊 執行狀態驗證

### 內部 CLI - 已驗證 ✅

```
測試命令: python aiva_cli_implementation.py --list
結果: 成功列出 286 條 Flow

可用 Flow 範例:
ID: 1  | Len: 2 | Path: backends -> unified_executor
ID: 11 | Len: 2 | Path: unified_executor -> capability_orchestrator
...
ID: 286 | (還有更多)
```

### 外部 CLI - 待測試 ⚠️

建議測試：
```bash
# 1. 列出外部模組
python aiva_external_module_cli.py --list

# 2. 測試 Python 模組
python aiva_external_module_cli.py --lang python --flow 1 --target http://testphp.vulnweb.com

# 3. 測試 Rust 模組
python aiva_external_module_cli.py --lang rust --func analyze_cookies --dry-run
```

---

## 🚀 建議的統一執行腳本

基於分析，可以創建統一的執行腳本：

### `統一CLI執行器.bat`

```bat
@echo off
chcp 65001 >nul 2>&1

:MENU
cls
echo ========================================
echo   AIVA 統一 CLI 執行器
echo ========================================
echo.
echo [內部模組 - Core 系統]
echo 1. 列出所有內部 Flow (286 條)
echo 2. 執行內部 Flow
echo 3. 預覽內部 Flow (Dry Run)
echo.
echo [外部模組 - 攻擊功能]
echo 4. 列出所有外部模組 (222 flows)
echo 5. 執行外部 Python 模組
echo 6. 執行外部 Rust 模組
echo 7. 執行外部 Go 模組
echo.
echo 0. 退出
echo.
set /p choice=請選擇 (0-7): 

if "%choice%"=="0" goto END
if "%choice%"=="1" goto INTERNAL_LIST
if "%choice%"=="2" goto INTERNAL_EXEC
if "%choice%"=="3" goto INTERNAL_PREVIEW
if "%choice%"=="4" goto EXTERNAL_LIST
if "%choice%"=="5" goto EXTERNAL_PYTHON
if "%choice%"=="6" goto EXTERNAL_RUST
if "%choice%"=="7" goto EXTERNAL_GO

:INTERNAL_LIST
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_cli_implementation.py --list
pause
goto MENU

:INTERNAL_EXEC
set /p flow_id=請輸入 Flow ID: 
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_cli_implementation.py --flow %flow_id%
pause
goto MENU

:INTERNAL_PREVIEW
set /p flow_id=請輸入 Flow ID: 
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_cli_implementation.py --flow %flow_id% --dry-run
pause
goto MENU

:EXTERNAL_LIST
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_external_module_cli.py --list
pause
goto MENU

:EXTERNAL_PYTHON
set /p target=請輸入目標 URL: 
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_external_module_cli.py --lang python --flow 1 --target %target%
pause
goto MENU

:EXTERNAL_RUST
set /p func=請輸入功能名稱 (analyze_cookies/analyze_headers/scan_javascript/analyze_tls): 
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_external_module_cli.py --lang rust --func %func% --dry-run
pause
goto MENU

:EXTERNAL_GO
cd /d C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_external_module_cli.py --lang go --func DialBroker --broker-url amqp://localhost
pause
goto MENU

:END
echo 再見！
```

---

## 📝 總結

### 關鍵結論

1. ✅ **架構完全相同** - 內部和外部 CLI 使用相同的 FlowExecutor 設計
2. ✅ **只是數據來源不同** - 內部讀 `latest_classification.json`，外部讀 `analysis_results.json`
3. ✅ **執行方式統一** - 都使用動態導入、啟發式方法偵測、Pipeline 傳遞
4. ✅ **內部 CLI 已驗證可執行** - 成功列出 286 條 Flow
5. ⚠️ **外部 CLI 需要測試** - 建議測試 Python/Rust/Go/TypeScript 各語言模組

### .bat 檔案功用總結

| 檔案 | 用途 | 目標 |
|------|------|------|
| `執行Flow.bat` | 執行內部 Flow | Core 系統 (286 flows) |
| `預覽Flow.bat` | Dry Run 內部 Flow | 驗證執行計畫 |
| `啟動能力選單.bat` | 互動式選單 | 內部模組 |
| `外部模組選單.bat` | 互動式選單 | 外部攻擊模組 |
| `執行外部模組.bat` | Direct Import | 快速執行單一模組 |
| `分類外部模組.bat` | 批次分析 | 生成整合報告 |
| `啟動AIVA系統.bat` | 啟動完整系統 | 所有服務 |

### 建議

1. **統一執行入口** - 創建 `統一CLI執行器.bat` 整合內外部 CLI
2. **測試外部 CLI** - 驗證各語言模組是否正常執行
3. **文檔同步** - 更新 README 說明內外部 CLI 的關係
4. **移除冗餘** - 如果 CommandHandler 只是包裝層，考慮統一使用 Direct Import

---

**報告結束**
