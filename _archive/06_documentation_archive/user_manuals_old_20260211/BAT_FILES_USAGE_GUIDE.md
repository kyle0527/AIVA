# AIVA 模組執行檔案使用指南

## 📑 目錄

- [📋 目錄](#-目錄)
- [內部模組執行](#內部模組執行)
  - [可用檔案](#可用檔案)
    - [1. `執行Flow.bat` - 執行特定 Flow](#1-執行flowbat---執行特定-flow)
    - [2. `啟動能力選單.bat` - 互動式選單](#2-啟動能力選單bat---互動式選單)
    - [3. `預覽Flow.bat` - Dry Run 模式](#3-預覽flowbat---dry-run-模式)
- [外部模組執行](#外部模組執行)
  - [可用檔案](#可用檔案)
    - [1. `執行外部模組.bat` - 執行特定模組](#1-執行外部模組bat---執行特定模組)
    - [2. `外部模組選單.bat` - 互動式選單](#2-外部模組選單bat---互動式選單)
    - [3. `分類外部模組.bat` - 批次分析分類](#3-分類外部模組bat---批次分析分類)
- [快速對照表](#快速對照表)
  - [內部模組 vs 外部模組](#內部模組-vs-外部模組)
  - [檔案命名規範](#檔案命名規範)
- [典型使用場景](#典型使用場景)
  - [場景 1: 執行內部 AI 能力](#場景-1-執行內部-ai-能力)
  - [場景 2: 執行漏洞檢測](#場景-2-執行漏洞檢測)
  - [場景 3: 預覽執行計畫](#場景-3-預覽執行計畫)
  - [場景 4: 批次分析](#場景-4-批次分析)
- [注意事項](#注意事項)
  - [環境要求](#環境要求)
  - [編碼設置](#編碼設置)
  - [錯誤處理](#錯誤處理)
- [常見問題](#常見問題)
- [技術架構](#技術架構)
  - [內部模組架構](#內部模組架構)
  - [外部模組架構](#外部模組架構)
- [更新日誌](#更新日誌)
  - [v2.0 (2026-01-13)](#v20-2026-01-13)
  - [v1.0 (2026-01-03)](#v10-2026-01-03)

---


## 📋 目錄

- [內部模組執行](#內部模組執行)
- [外部模組執行](#外部模組執行)
- [快速對照表](#快速對照表)

---

## 內部模組執行

內部模組指的是 AIVA AI Core 的 5 大模組：
- `cognitive_core` - 認知核心模組
- `internal_exploration` - 內探模組
- `task_planning` - 任務規劃模組
- `core_capabilities` - 核心能力模組
- `service_backbone` - 服務骨幹模組

### 可用檔案

#### 1. `執行Flow.bat` - 執行特定 Flow

```batch
用法: 執行Flow.bat [Flow ID]
範例: 執行Flow.bat 11

功能:
  - 執行指定 ID 的 Flow
  - 自動追蹤執行路徑
  - 顯示執行結果
```

**使用範例**:
```cmd
REM 執行 Flow 11
執行Flow.bat 11

REM 列出所有可用 Flow
執行Flow.bat
```

#### 2. `啟動能力選單.bat` - 互動式選單

```batch
用法: 直接雙擊執行

功能:
  - 顯示互動式選單
  - 列出所有可用 Flow
  - 提供友好的執行界面
```

**使用範例**:
```cmd
REM 啟動選單
啟動能力選單.bat

REM 然後在選單中選擇要執行的 Flow
```

#### 3. `預覽Flow.bat` - Dry Run 模式

```batch
用法: 預覽Flow.bat [Flow ID]
範例: 預覽Flow.bat 11

功能:
  - 只顯示執行計畫
  - 不實際運行代碼
  - 用於檢查執行路徑
```

**使用範例**:
```cmd
REM 預覽 Flow 11 的執行計畫
預覽Flow.bat 11
```

---

## 外部模組執行

外部模組指的是功能檢測模組和掃描引擎：
- **功能模組 (Features)**: SQL 注入、XSS、SSRF、IDOR 等
- **掃描引擎 (Scan)**: scan_engine、typescript_engine、rust_engine

### 可用檔案

#### 1. `執行外部模組.bat` - 執行特定模組

```batch
用法: 執行外部模組.bat [模組名稱] [目標URL]
範例: 執行外部模組.bat function_sqli http://example.com

可用模組:
  - function_sqli      : SQL 注入檢測
  - function_xss       : XSS 漏洞檢測
  - function_ssrf      : SSRF 漏洞檢測
  - function_idor      : IDOR 漏洞檢測
  - function_info_leak : 信息洩露檢測
  - function_bizlogic  : 業務邏輯漏洞
  - function_crypto    : 加密相關漏洞
  - function_authn_go  : 身份驗證檢測
  - scan_engine        : 掃描引擎
  - typescript_engine  : TypeScript 掃描
  - rust_engine        : Rust 掃描
```

**使用範例**:
```cmd
REM 執行 SQL 注入檢測
執行外部模組.bat function_sqli http://testsite.com

REM 執行 XSS 檢測
執行外部模組.bat function_xss http://target.com

REM 列出所有可用模組
執行外部模組.bat
```

#### 2. `外部模組選單.bat` - 互動式選單

```batch
用法: 直接雙擊執行

功能:
  - 顯示所有外部模組選單
  - 互動式選擇模組和輸入目標
  - 支援批次分類功能
```

**使用範例**:
```cmd
REM 啟動選單
外部模組選單.bat

REM 選單選項:
REM 1-8: 執行各種檢測模組
REM 9: 批次分類所有外部模組
REM 0: 退出
```

#### 3. `分類外部模組.bat` - 批次分析分類

```batch
用法: 分類外部模組.bat [輸入目錄] [輸出目錄]
範例: 分類外部模組.bat . ./reports

功能:
  - 自動掃描所有外部模組
  - 分析模組結構和語言
  - 生成詳細分類報告
```

**使用範例**:
```cmd
REM 分析當前專案的外部模組
分類外部模組.bat . ./external_reports

REM 使用預設輸出目錄
分類外部模組.bat .

REM 查看使用說明
分類外部模組.bat
```

---

## 快速對照表

### 內部模組 vs 外部模組

| 特性 | 內部模組 | 外部模組 |
|------|---------|---------|
| **目標** | AI Core 5 大模組 | 功能檢測 + 掃描引擎 |
| **執行方式** | Flow ID | 模組名稱 + 目標 URL |
| **主要檔案** | 執行Flow.bat<br>啟動能力選單.bat<br>預覽Flow.bat | 執行外部模組.bat<br>外部模組選單.bat<br>分類外部模組.bat |
| **分類器** | aiva_flow_classifier.py | aiva_external_module_classifier.py |
| **執行器** | aiva_cli_implementation.py | aiva_external_module_cli.py |
| **數據源** | features_classification/ | module_analysis/ |

### 檔案命名規範

| 用途 | 內部模組 | 外部模組 |
|------|---------|---------|
| **執行** | 執行Flow.bat | 執行外部模組.bat |
| **選單** | 啟動能力選單.bat | 外部模組選單.bat |
| **預覽** | 預覽Flow.bat | _(無對應)_ |
| **分類** | _(集成在 classifier 中)_ | 分類外部模組.bat |

---

## 典型使用場景

### 場景 1: 執行內部 AI 能力

```cmd
REM 方式 1: 直接執行
執行Flow.bat 15

REM 方式 2: 使用選單
啟動能力選單.bat
```

### 場景 2: 執行漏洞檢測

```cmd
REM 方式 1: 直接執行
執行外部模組.bat function_sqli http://target.com

REM 方式 2: 使用選單
外部模組選單.bat
```

### 場景 3: 預覽執行計畫

```cmd
REM 內部模組預覽
預覽Flow.bat 11

REM 外部模組無預覽功能（直接執行）
```

### 場景 4: 批次分析

```cmd
REM 分析所有外部模組
分類外部模組.bat . ./reports

REM 或使用選單中的選項 9
外部模組選單.bat
```

---

## 注意事項

### 環境要求

所有 .bat 檔案都會自動設置：
```batch
set PYTHONPATH=C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services
set PYTHONIOENCODING=utf-8
```

### 編碼設置

自動使用 UTF-8 編碼：
```batch
chcp 65001 >nul 2>&1
```

### 錯誤處理

- 內部模組：自動檢測 Flow ID 有效性
- 外部模組：驗證目標 URL 格式
- 所有檔案：提供友好的錯誤提示

---

## 常見問題

**Q: 如何查看所有可用的 Flow？**  
A: 執行 `執行Flow.bat` 不帶參數，會列出所有可用 Flow。

**Q: 如何查看所有外部模組？**  
A: 執行 `執行外部模組.bat` 不帶參數，會列出所有可用模組。

**Q: 預覽模式和正常執行有什麼區別？**  
A: 預覽模式 (--dry-run) 只顯示執行計畫，不實際運行代碼。

**Q: 外部模組為什麼需要目標 URL？**  
A: 外部模組是漏洞檢測工具，需要指定目標網站進行測試。

**Q: 如何生成外部模組的分類報告？**  
A: 使用 `分類外部模組.bat . ./reports` 或在選單中選擇選項 9。

---

## 技術架構

### 內部模組架構

```
執行Flow.bat
    ↓
aiva_cli_implementation.py
    ↓
aiva_flow_classifier.py
    ↓
features_classification/*.json
```

### 外部模組架構

```
執行外部模組.bat
    ↓
aiva_external_module_cli.py
    ↓
aiva_external_module_classifier.py
    ↓
module_analysis/function_*/
```

---

## 更新日誌

### v2.0 (2026-01-13)
- ✅ 新增外部模組執行支援
- ✅ 創建 3 個外部模組 .bat 檔案
- ✅ 統一內外模組命名規範
- ✅ 修復所有類型錯誤和警告
- ✅ 按照 aiva_common 規範進行修復

### v1.0 (2026-01-03)
- ✅ 初始版本
- ✅ 內部模組 3 個 .bat 檔案

---

**相關文檔**:
- [ARCHITECTURE_REFACTOR.md](services/core/aiva_core/internal_exploration/ARCHITECTURE_REFACTOR.md) - 架構重構報告
- [CLEANUP_REPORT.md](services/core/aiva_core/internal_exploration/CLEANUP_REPORT.md) - 清理報告
