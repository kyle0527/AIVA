# AIVA CLI 完整命令參考

## 📋 目錄

- [概覽](#概覽)
- [掃描命令 (scan)](#掃描命令-scan)
- [漏洞檢測命令 (detect)](#漏洞檢測命令-detect)
- [AI 命令 (ai)](#ai-命令-ai)
- [報告命令 (report)](#報告命令-report)
- [系統命令 (system)](#系統命令-system)
- [工具命令 (tools)](#工具命令-tools)
- [使用範例](#使用範例)

---

## 概覽

AIVA 提供統一的命令列介面，支援以下主要功能模組：

```bash
aiva --help
```

**頂層指令：**
- `scan` - 掃描管理
- `detect` - 漏洞檢測
- `ai` - AI 訓練和管理
- `report` - 報告生成
- `system` - 系統管理
- `tools` - 開發者工具（跨模組整合）

---

## 掃描命令 (scan)

### scan start

啟動網站掃描任務

```bash
aiva scan start <url> [選項]
```

**必需參數：**
- `url` - 目標 URL

**選項：**
- `--max-depth <數字>` - 最大爬取深度（預設: 3）
- `--max-pages <數字>` - 最大頁面數（預設: 100）
- `--wait` - 等待掃描完成（阻塞式）

**範例：**

```bash
# 基本掃描
aiva scan start https://example.com

# 深度掃描
aiva scan start https://example.com --max-depth 5 --max-pages 200

# 掃描並等待結果
aiva scan start https://example.com --wait
```

---

## 漏洞檢測命令 (detect)

### detect sqli

SQL 注入漏洞檢測

```bash
aiva detect sqli <url> --param <參數名> [選項]
```

**必需參數：**
- `url` - 目標 URL
- `--param <名稱>` - 要測試的參數名

**選項：**
- `--method <GET|POST>` - HTTP 方法
- `--engines <引擎列表>` - 檢測引擎（逗號分隔）
- `--wait` - 等待檢測完成

**範例：**

```bash
# GET 參數測試
aiva detect sqli https://example.com/login --param username

# POST 參數測試
aiva detect sqli https://example.com/api --param id --method POST

# 使用特定引擎並等待結果
aiva detect sqli https://example.com/search --param q --engines "union,boolean,time" --wait
```

### detect xss

XSS（跨站腳本）漏洞檢測

```bash
aiva detect xss <url> --param <參數名> [選項]
```

**必需參數：**
- `url` - 目標 URL
- `--param <名稱>` - 要測試的參數名

**選項：**
- `--type <reflected|stored|dom>` - XSS 類型
- `--wait` - 等待檢測完成

**範例：**

```bash
# 反射型 XSS 檢測
aiva detect xss https://example.com/search --param q --type reflected

# 存儲型 XSS 檢測
aiva detect xss https://example.com/comment --param content --type stored --wait

# DOM 型 XSS 檢測
aiva detect xss https://example.com/page --param hash --type dom
```

---

## AI 命令 (ai)

### ai train

訓練 AI 模型

```bash
aiva ai train [選項]
```

**選項：**
- `--mode <realtime|replay|simulation>` - 訓練模式（預設: realtime）
  - `realtime` - 實時訓練：監聽實際任務執行
  - `replay` - 回放訓練：從歷史經驗學習
  - `simulation` - 模擬訓練：使用模擬場景
- `--epochs <數字>` - 訓練輪數（預設: 10）
- `--scenarios <數字>` - 模擬場景數量（僅 simulation 模式，預設: 100）
- `--storage-path <路徑>` - 存儲路徑（預設: ./data/ai）

**範例：**

```bash
# 實時訓練模式
aiva ai train --mode realtime --epochs 10

# 從歷史記錄學習
aiva ai train --mode replay --epochs 20

# 模擬訓練
aiva ai train --mode simulation --scenarios 500 --epochs 15

# 自訂存儲路徑
aiva ai train --mode realtime --storage-path /data/ai-models
```

### ai status

查看 AI 系統狀態

```bash
aiva ai status [選項]
```

**選項：**
- `--storage-path <路徑>` - 存儲路徑（預設: ./data/ai）

**範例：**

```bash
# 查看 AI 狀態
aiva ai status

# 指定存儲路徑
aiva ai status --storage-path /data/ai-models
```

---

## 報告命令 (report)

### report generate

生成掃描報告

```bash
aiva report generate <scan_id> [選項]
```

**必需參數：**
- `scan_id` - 掃描任務 ID

**選項：**
- `--format <pdf|html|json>` - 報告格式（預設: html）
- `--output <檔案路徑>` - 輸出檔案（預設: report.html）
- `--no-findings` - 不包含漏洞詳情

**範例：**

```bash
# 生成 HTML 報告
aiva report generate scan_20231017_001 --format html --output report.html

# 生成 PDF 報告
aiva report generate scan_20231017_001 --format pdf --output final_report.pdf

# 生成 JSON 報告（機器可讀）
aiva report generate scan_20231017_001 --format json --output data.json

# 僅生成統計資訊，不包含漏洞詳情
aiva report generate scan_20231017_001 --no-findings
```

---

## 系統命令 (system)

### system status

查看系統狀態

```bash
aiva system status
```

顯示所有模組的運行狀態，包括：
- Core 核心模組
- Scan 掃描模組
- Function 功能模組（SQLi, XSS, SSRF, IDOR）
- Integration 整合模組

**範例：**

```bash
aiva system status
```

**輸出範例：**

```
🔧 AIVA 系統狀態

📡 模組狀態:
   core: 🟢 運行中
   scan: 🟢 運行中
   function.sqli: 🟢 運行中
   function.xss: 🟢 運行中
   integration: 🟢 運行中
```

---

## 工具命令 (tools)

### tools schemas

導出 JSON Schema（用於跨語言協定）

```bash
aiva tools schemas [選項]
```

**選項：**
- `--out <檔案路徑>` - 輸出檔案路徑（預設: ./_out/aiva.schemas.json）
- `--format <human|json>` - 輸出格式（預設: human）

**範例：**

```bash
# 導出 JSON Schema
aiva tools schemas

# 指定輸出路徑
aiva tools schemas --out contracts/aiva.schemas.json

# JSON 格式輸出（用於管道處理）
aiva tools schemas --out schemas.json --format json
```

### tools typescript

導出 TypeScript 型別定義

```bash
aiva tools typescript [選項]
```

**選項：**
- `--out <檔案路徑>` - 輸出檔案路徑（預設: ./_out/aiva.d.ts）
- `--format <human|json>` - 輸出格式（預設: human）

**範例：**

```bash
# 導出 TypeScript 定義
aiva tools typescript

# 指定輸出路徑
aiva tools typescript --out types/aiva.d.ts

# JSON 格式輸出
aiva tools typescript --out aiva.d.ts --format json
```

### tools models

列出所有 Pydantic 模型

```bash
aiva tools models [選項]
```

**選項：**
- `--format <human|json>` - 輸出格式（預設: human）

**範例：**

```bash
# 列出所有模型
aiva tools models

# JSON 格式輸出
aiva tools models --format json
```

### tools export-all

一鍵導出所有型別定義

```bash
aiva tools export-all [選項]
```

**選項：**
- `--out-dir <目錄路徑>` - 輸出目錄（預設: ./_out）
- `--format <human|json>` - 輸出格式（預設: human）

**範例：**

```bash
# 導出所有型別定義
aiva tools export-all

# 指定輸出目錄
aiva tools export-all --out-dir contracts

# JSON 格式輸出
aiva tools export-all --out-dir exports --format json
```

**輸出內容：**
- `aiva.schemas.json` - JSON Schema 定義
- `aiva.d.ts` - TypeScript 型別定義

---

## 使用範例

### 完整掃描流程

```bash
# 1. 啟動掃描
aiva scan start https://example.com --max-depth 3 --wait

# 2. 執行 SQL 注入檢測
aiva detect sqli https://example.com/login --param username --wait

# 3. 執行 XSS 檢測
aiva detect xss https://example.com/search --param q --wait

# 4. 生成報告
aiva report generate scan_xxx --format pdf --output final_report.pdf
```

### CI/CD 整合範例

```bash
#!/bin/bash
# CI/CD 自動化掃描腳本

# 掃描目標
TARGET="https://staging.example.com"

# 執行掃描（JSON 輸出便於解析）
SCAN_ID=$(aiva scan start "$TARGET" --format json | jq -r '.scan_id')

# 等待掃描完成
aiva scan wait "$SCAN_ID"

# 執行漏洞檢測
aiva detect sqli "$TARGET/api" --param id --format json
aiva detect xss "$TARGET/search" --param q --format json

# 生成 JSON 報告
aiva report generate "$SCAN_ID" --format json --output report.json

# 檢查是否有高危漏洞
HIGH_RISK=$(jq '.findings[] | select(.severity == "high") | length' report.json)
if [ "$HIGH_RISK" -gt 0 ]; then
    echo "❌ 發現高危漏洞！"
    exit 1
fi

echo "✅ 掃描通過"
exit 0
```

### AI 訓練工作流程

```bash
# 1. 查看當前 AI 狀態
aiva ai status

# 2. 使用實時模式訓練（從實際任務學習）
aiva ai train --mode realtime --epochs 10

# 3. 使用模擬模式補充訓練
aiva ai train --mode simulation --scenarios 1000 --epochs 20

# 4. 再次查看狀態
aiva ai status
```

### 跨語言協定導出

```bash
# 導出所有型別定義（用於前端/其他語言整合）
aiva tools export-all --out-dir contracts

# 目錄結構：
# contracts/
#   ├── aiva.schemas.json  (JSON Schema - 適用於所有語言)
#   └── aiva.d.ts          (TypeScript 定義)

# 可用於：
# - 前端 TypeScript 專案
# - Go/Rust 透過 JSON Schema 生成型別
# - API 文件生成
# - 測試資料驗證
```

### 批量掃描範例

```powershell
# PowerShell 批量掃描腳本

# 讀取目標列表
$targets = Get-Content "targets.txt"

# 批量掃描
foreach ($target in $targets) {
    Write-Host "🔍 掃描: $target"
    aiva scan start $target --max-depth 2 --format json | Out-File "results/$($target -replace '[:/]', '_').json"
}

# 生成彙總報告
Write-Host "📊 生成彙總報告..."
# （可以使用 Python 腳本處理 JSON 檔案）
```

---

## 🔧 進階配置

### 環境變數（計劃支援）

```bash
# 設定預設參數
export AIVA_MAX_DEPTH=5
export AIVA_TIMEOUT=60
export AIVA_CONCURRENCY=16

# 使用環境變數（會被命令列參數覆蓋）
aiva scan start https://example.com
```

### 配置檔案（計劃支援）

**config.json**
```json
{
  "scan": {
    "max_depth": 5,
    "max_pages": 200,
    "timeout": 60
  },
  "detect": {
    "concurrency": 16,
    "timeout": 30
  },
  "ai": {
    "storage_path": "./data/ai",
    "auto_train": true
  }
}
```

**使用配置檔案：**
```bash
aiva scan start https://example.com --config config.json
```

---

## 📊 退出碼

所有 AIVA 命令遵循標準退出碼：

- `0` - 成功
- `1` - 使用錯誤（參數錯誤）
- `2` - 系統錯誤（執行時錯誤）
- `10+` - 業務邏輯錯誤

**範例：**

```bash
aiva scan start https://example.com
if [ $? -eq 0 ]; then
    echo "✅ 掃描成功"
else
    echo "❌ 掃描失敗"
fi
```

---

## 🆘 取得幫助

任何命令都可以使用 `--help` 查看詳細說明：

```bash
# 主幫助
aiva --help

# 子命令幫助
aiva scan --help
aiva detect --help
aiva tools --help

# 具體操作幫助
aiva scan start --help
aiva detect sqli --help
aiva tools schemas --help
```

---

## 📚 相關文件

- [快速開始](./QUICK_START.md)
- [CLI 安裝指南](./CLI_UNIFIED_SETUP_GUIDE.md)
- [AI 訓練指南](./CLI_AND_AI_TRAINING_GUIDE.md)
- [架構文件](./AI_ARCHITECTURE_ANALYSIS.md)

---

**版本**: 1.0.0  
**更新日期**: 2025-10-17  
**維護者**: AIVA Team
