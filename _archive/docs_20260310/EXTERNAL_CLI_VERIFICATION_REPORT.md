# AIVA 外部模組 CLI 執行驗證報告

## 📑 目錄

- [✅ CLI 執行能力驗證](#-cli-執行能力驗證)
  - [統一執行器狀態](#統一執行器狀態)
  - [執行命令格式](#執行命令格式)
- [已驗證的模組和流程](#已驗證的模組和流程)
  - [1. function_xss (XSS 漏洞檢測)](#1-function_xss-xss-漏洞檢測)
  - [2. function_ssrf (SSRF 漏洞檢測)](#2-function_ssrf-ssrf-漏洞檢測)
  - [3. function_sqli (SQL 注入檢測)](#3-function_sqli-sql-注入檢測)
  - [4. function_idor (IDOR 漏洞檢測)](#4-function_idor-idor-漏洞檢測)
  - [5. function_bizlogic (業務邏輯漏洞)](#5-function_bizlogic-業務邏輯漏洞)
  - [6. function_authn_go (身份驗證測試 - Go)](#6-function_authn_go-身份驗證測試---go)
  - [7. function_crypto (加密分析 - Rust)](#7-function_crypto-加密分析---rust)
  - [8. typescript_engine (TypeScript 引擎)](#8-typescript_engine-typescript-引擎)
- [CLI 功能驗證](#cli-功能驗證)
  - [✅ 已驗證功能](#-已驗證功能)
  - [⏳ 待完整實現功能](#-待完整實現功能)
- [驗證總結](#驗證總結)
  - [成功項目 ✅](#成功項目-)
  - [待改進項目 ⏳](#待改進項目-)
  - [技術限制說明](#技術限制說明)
- [建議的執行工作流程](#建議的執行工作流程)
  - [方案1: 使用統一執行器（推薦用於學習和理解）](#方案1-使用統一執行器推薦用於學習和理解)
  - [方案2: 使用模組原生 API（推薦用於實際測試）](#方案2-使用模組原生-api推薦用於實際測試)
  - [方案3: 使用 Worker 服務（推薦用於生產環境）](#方案3-使用-worker-服務推薦用於生產環境)
- [下一步行動](#下一步行動)

---


**驗證日期**: 2026-01-14  
**驗證工具**: `aiva_external_executor.py`  
**靶場環境**: Juice Shop (localhost:3000), WebGoat (localhost:8080)

## ✅ CLI 執行能力驗證

### 統一執行器狀態
- **執行器**: `aiva_external_executor.py`
- **位置**: `services/core/aiva_core/internal_exploration/`
- **支援語言**: Python (203), Go (4), TypeScript (3)
- **總流程數**: 210 flows
- **總模組數**: 8 modules

### 執行命令格式
```bash
# 基本格式
python aiva_external_executor.py --lang <語言> --flow <ID> --target <URL> [--dry-run]

# 範例
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run
```

## 已驗證的模組和流程

### 1. function_xss (XSS 漏洞檢測)
- **流程總數**: 109
- **能力數**: 90 種
- **驗證流程**: Flow #101

**執行命令**:
```bash
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run
```

**驗證結果**: ✅ 成功
- 模組識別正確: function_xss
- 類型識別: injection / web_security
- 流程路徑: bruteforcer → getUrl
- 用途說明顯示正確

**實際用途**: 從目標提取所有可測試的 URL 端點，用於批量掃描

### 2. function_ssrf (SSRF 漏洞檢測)
- **流程總數**: 35
- **能力數**: 32 種
- **驗證流程**: Flow #64

**執行命令**:
```bash
python aiva_external_executor.py --lang python --flow 64 \
  --target http://localhost:3000/rest/user/whoami --dry-run
```

**驗證結果**: ✅ 成功
- 模組識別正確: function_ssrf
- 類型識別: ssrf / network_security
- 流程路徑: SsrfWorkerService.process_task → SsrfResultPublisher
- 參數傳遞正確

**實際用途**: SSRF 結果發布器，用於記錄和報告 SSRF 測試結果

### 3. function_sqli (SQL 注入檢測)
- **流程總數**: 32
- **能力數**: 32 種
- **驗證流程**: Flow #35

**執行命令**:
```bash
python aiva_external_executor.py --lang python --flow 35 \
  --target "http://localhost:3000/rest/products/search?q=test" --dry-run
```

**驗證結果**: ✅ 成功
- 模組識別正確: function_sqli
- 類型識別: injection / database_security
- 流程路徑: SQLInjectionManager.__init__ → NoSQLInjectionScanner
- URL 參數正確處理

**實際用途**: NoSQL 注入檢測（MongoDB, Redis 等）

### 4. function_idor (IDOR 漏洞檢測)
- **流程總數**: 19
- **能力數**: 19 種
- **狀態**: 可用（未詳細測試）

### 5. function_bizlogic (業務邏輯漏洞)
- **流程總數**: 8
- **能力數**: 8 種
- **狀態**: 可用（未詳細測試）

### 6. function_authn_go (身份驗證測試 - Go)
- **流程總數**: 4
- **能力數**: 4 種
- **語言**: Go
- **狀態**: 可用（未詳細測試）

### 7. function_crypto (加密分析 - Rust)
- **流程總數**: 0
- **狀態**: 已分析但無流程

### 8. typescript_engine (TypeScript 引擎)
- **流程總數**: 3
- **能力數**: 3 種
- **語言**: TypeScript
- **狀態**: 可用（未詳細測試）

## CLI 功能驗證

### ✅ 已驗證功能

1. **流程列表** (`--list`)
   ```bash
   python aiva_external_executor.py --list --lang python
   ```
   - 按模組分組顯示
   - 顯示流程數量和能力數
   - 顯示前 5 個流程

2. **Dry-Run 模式** (`--dry-run`)
   - 顯示將要執行的步驟
   - 不實際執行模組代碼
   - 驗證參數傳遞

3. **參數傳遞** (`--target`, `--flow`)
   - 目標 URL 正確傳遞
   - Flow ID 正確識別
   - 顯示完整流程資訊

4. **互動式選單** (`--menu`)
   - 三層選單結構
   - 按語言 → 模組 → 能力分組
   - 顯示流程變體

5. **文檔生成** (`--generate-doc`)
   - Markdown 格式: `EXTERNAL_CLI_COMMANDS_REFERENCE.md`
   - JSON 格式: `external_cli_commands_db.json`

### ⏳ 待完整實現功能

1. **實際執行邏輯** (不帶 `--dry-run`)
   - 需要動態導入模組
   - 需要實例化類別
   - 需要調用正確的入口方法
   - **原因**: 流程定義中的路徑（如 bruteforcer, getUrl）是分析工具提取的調用鏈，不是直接可導入的類別名

2. **建議執行方式**:
   - 方式1: 使用各模組的 CommandHandler
     ```python
     from services.features.function_xss import XSSCommandHandler
     handler = XSSCommandHandler()
     await handler.handle_command(command)
     ```
   
   - 方式2: 使用各模組的 Worker 服務
     ```bash
     python -m services.features.function_xss.worker
     ```

## 驗證總結

### 成功項目 ✅
1. CLI 工具可以正確加載所有 210 個流程
2. 流程資訊顯示完整（模組、類型、描述、用途）
3. 參數傳遞機制正常
4. Dry-run 模式運作正常
5. 互動式選單功能完整
6. 文檔生成功能正常

### 待改進項目 ⏳
1. **實際執行邏輯**: 需要實現從流程路徑到實際類別/函數的映射
2. **錯誤處理**: 需要更完善的錯誤提示
3. **執行日誌**: 需要詳細的執行過程記錄

### 技術限制說明
**為什麼不能直接執行**:
- 流程定義中的 `start` 和 `end` 是 AST 分析工具提取的**函數調用鏈**
- 例如: `bruteforcer → getUrl` 表示 bruteforcer 函數內部調用了 getUrl
- 這些不是可以直接導入的模組或類別名稱
- 實際的入口點是 `XSSCommandHandler`, `TraditionalXssDetector` 等

**解決方案**:
1. 使用模組的公開 API (CommandHandler, Worker)
2. 或者：建立流程名稱到實際類別的映射表
3. 或者：重新定義流程，使用實際的類別名稱

## 建議的執行工作流程

### 方案1: 使用統一執行器（推薦用於學習和理解）
```bash
# 查看可用能力
python aiva_external_executor.py --list --lang python

# Dry-run 模式了解流程
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run

# 互動式選單瀏覽
python aiva_external_executor.py --menu
```

### 方案2: 使用模組原生 API（推薦用於實際測試）
```python
# XSS 測試
from services.features.function_xss import XSSCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

handler = XSSCommandHandler()
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"target_url": "http://localhost:3000"}
)
result = await handler.handle_command(command)
```

### 方案3: 使用 Worker 服務（推薦用於生產環境）
```bash
# 啟動 Worker
python -m services.features.function_xss.worker

# 通過消息隊列發送任務
# (需要 RabbitMQ)
```

## 下一步行動

1. ✅ **已完成**: CLI 工具驗證
2. ✅ **已完成**: 流程資訊展示
3. ✅ **已完成**: Dry-run 模式
4. ⏳ **進行中**: 實際執行邏輯實現
5. ⏳ **計劃中**: 與靶場的完整集成測試
6. ⏳ **計劃中**: 執行結果分析和報告

---

**報告生成**: AIVA 外部模組整合系統  
**驗證人**: AI Assistant  
**最後更新**: 2026-01-14 01:30
