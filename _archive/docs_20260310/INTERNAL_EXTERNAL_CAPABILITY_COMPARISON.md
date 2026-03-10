# AIVA 內部/外部模組能力展示對比

## 📑 目錄

- [問題分析](#問題分析)
  - [原本的問題](#原本的問題)
- [解決方案：複製內部模組的成功模式](#解決方案複製內部模組的成功模式)
  - [核心概念：起點→終點聚合](#核心概念起點終點聚合)
  - [實現內容](#實現內容)
    - [1. 互動式選單 (InteractiveMenu)](#1-互動式選單-interactivemenu)
    - [2. 啟動方式](#2-啟動方式)
- [對比結果](#對比結果)
- [實際展示](#實際展示)
  - [統計數據](#統計數據)
  - [能力聚合範例](#能力聚合範例)
- [AI 可理解性](#ai-可理解性)
  - [改進前](#改進前)
  - [改進後](#改進後)
  - [改進後的展示](#改進後的展示)
- [使用建議](#使用建議)
  - [給人類用戶](#給人類用戶)
  - [給 AI 代理](#給-ai-代理)
- [文件清單](#文件清單)
  - [新增文件](#新增文件)
  - [修改文件](#修改文件)
  - [生成文件](#生成文件)
- [下一步](#下一步)

---


## 問題分析

### 原本的問題
1. **外部模組描述不清晰**
   - 所有 XSS 流程的 `use_case` 都一樣："用於檢測 XSS 漏洞..."
   - 無法區分 `bruteforcer → getUrl` 和 `bruteforcer → getParams` 的不同
   - AI 看不懂什麼時候該用哪個能力

2. **缺少能力聚合**
   - 流程列表是扁平的，沒有按「相同能力的不同實現」分組
   - 無法理解哪些流程是做同樣的事情（起點→終點相同）
   - 無法理解中間路徑的意義（不同實現方式）

## 解決方案：複製內部模組的成功模式

### 核心概念：起點→終點聚合
```
起點 (Start) → 終點 (End) = 能力的目的
中間路徑 (Middle Path) = 實現方式的變體
```

### 實現內容

#### 1. 互動式選單 (InteractiveMenu)
```
層級 1: 語言分類
  ├─ Python (203 flows)
  ├─ Go (4 flows)
  └─ TypeScript (3 flows)

層級 2: 模組列表
  function_xss (90 種能力, 109 flows)
  ├─ 類型: injection
  └─ 描述: XSS 漏洞檢測

層級 3: 能力列表（按起點→終點聚合）
  [1] bruteforcer → getUrl (1 變體)
  [2] bruteforcer → getParams (1 變體)
  [3] bruteforcer → converter (1 變體)
  [4] StoredXSSDetector → XSSVulnerability (1 變體)
  ...

層級 4: 路徑變體
  Flow 101 (長度: 2): 直連
  Flow 103 (長度: 3): ... → encoder → ...
```

#### 2. 啟動方式

**內部模組**:
```bash
# 方式1: 雙擊
啟動能力選單.bat

# 方式2: 命令行
python aiva_internal_executor.py --menu
```

**外部模組**:
```bash
# 方式1: 雙擊
啟動外部能力選單.bat

# 方式2: 命令行
python aiva_external_executor.py --menu
```

## 對比結果

| 項目 | 內部模組 | 外部模組 | 狀態 |
|------|----------|----------|------|
| 互動式選單 | ✅ | ✅ | 已對齊 |
| 起點→終點聚合 | ✅ | ✅ | 已對齊 |
| 路徑變體展示 | ✅ | ✅ | 已對齊 |
| 能力統計 | ✅ | ✅ | 已對齊 |
| Dry-run 預覽 | ✅ | ✅ | 已對齊 |
| 文檔生成 | ✅ | ✅ | 已對齊 |

## 實際展示

### 統計數據

**內部模組** (Core系統):
- 總流程: 286
- 語言: Python only
- 模組: 6 (cognitive_core, internal_exploration, task_planning, core_capabilities, service_backbone, learning_system)

**外部模組** (Features功能):
- 總流程: 210
- 語言: Python (203), Go (4), TypeScript (3)
- 模組: 8 (function_xss, function_sqli, function_ssrf, function_idor, function_bizlogic, function_authn_go, function_crypto, typescript_engine)

### 能力聚合範例

**function_xss** (XSS 漏洞檢測):
- 90 種能力（起點→終點組合）
- 109 個流程變體

主要能力分類：
1. **bruteforcer** 系列（暴力測試）
   - `bruteforcer → getUrl`: 提取可測試 URL
   - `bruteforcer → getParams`: 提取 URL 參數
   - `bruteforcer → converter`: Payload 編碼轉換

2. **Detector** 系列（檢測器）
   - `StoredXSSDetector → XSSVulnerability`: 存儲型 XSS
   - `BlindXSSDetector → XSSVulnerability`: 盲注 XSS
   - `DOMXSSDetector → XSSVulnerability`: DOM 型 XSS

3. **Engine** 系列（引擎）
   - `CrossLanguageXSSEngine → LanguageEnvironment`: 語言環境檢測
   - `XSSEngine → XSSScanner`: XSS 掃描引擎

## AI 可理解性

### 改進前
```json
{
  "id": 101,
  "path": ["bruteforcer", "getUrl"],
  "use_case": "用於檢測跨站腳本(XSS)漏洞，適合掃描 Web 應用的輸入點和輸出點"
}
```
❌ AI 無法理解 `bruteforcer → getUrl` 具體做什麼

### 改進後
```
能力: bruteforcer → getUrl
分類: XSS 暴力測試
說明: 從目標提取所有可測試的 URL 端點，用於批量掃描
適用場景: 大規模端點發現，自動化測試前置步驟
```
✅ AI 清楚知道：這是提取 URL 的能力，用於暴力測試的第一步

### 改進後的展示
```
[function_xss] XSS 漏洞檢測
  90 種能力，109 個流程變體

  能力聚合（相同起點→終點 = 相同功能）:
  ├─ bruteforcer → getUrl (1 變體): 提取 URL
  ├─ bruteforcer → getParams (1 變體): 提取參數
  ├─ bruteforcer → converter (1 變體): 編碼轉換
  ├─ StoredXSSDetector → XSSVulnerability (1 變體): 存儲型
  └─ BlindXSSDetector → XSSVulnerability (1 變體): 盲注
```

## 使用建議

### 給人類用戶
1. 使用 `.bat` 文件雙擊啟動選單
2. 按模組瀏覽能力
3. 選擇起點→終點看具體實現
4. 使用 `v` 命令預覽再執行

### 給 AI 代理
1. 讀取 `external_cli_commands_db.json` 了解所有能力
2. 根據 `start` 和 `end` 理解能力目的
3. 根據 `use_case` 了解使用場景
4. 根據 `length` 和 `path` 選擇合適的實現

## 文件清單

### 新增文件
- `啟動外部能力選單.bat`: 外部模組選單啟動器
- `test_external_menu.py`: 能力聚合統計測試

### 修改文件
- `aiva_external_executor.py`: 
  - 新增 `InteractiveMenu` 類
  - 新增 `--menu` 參數
  - 改進能力展示邏輯

### 生成文件
- `EXTERNAL_CLI_COMMANDS_REFERENCE.md`: 外部模組參考手冊
- `external_cli_commands_db.json`: 外部模組 JSON 資料庫

## 下一步

1. ✅ 複製內部模組的成功模式
2. ✅ 實現起點→終點聚合
3. ✅ 創建互動式選單
4. ⏳ 改進 use_case 生成（更具體的描述）
5. ⏳ 實際執行測試
6. ⏳ 文檔增強（添加範例和最佳實踐）

---

*生成時間: 2026-01-14*
*作者: AIVA 外部模組整合系統*
