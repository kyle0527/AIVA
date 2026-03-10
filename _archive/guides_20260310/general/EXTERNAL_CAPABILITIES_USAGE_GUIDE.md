# ⚠️ [已废弃] AIVA 外部能力使用指南

> **⚠️ 废弃声明**: 本文档基于错误的架构理解创建，已于 2026-01-23 标注废弃  
> **废弃原因**: 文档讨论"外部接口"、"统一调用方式"等概念是架构误解  
> **正确理解**: 
> - "外部"指的是模组扫描**外部目标网站**，不是提供"外部API"
> - AI 通过 Executor 直接调用任何模组的类方法，无需简化接口
> - 模组不需要为 AI 提供特殊的 wrapper 函数
>
> **版本**: v3.3 | **废弃日期**: 2026-01-23

---

## 📋 架構概覽

AIVA 的外部能力通過**統一執行器**管理，確保一致的調用方式：

```
┌─────────────────────────────────────────────────────────────┐
│  外部能力工作流程                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. [代碼分析] → 4 個語言分析器 (Python/Rust/Go/TypeScript)    │
│     ├── aiva_flow_analyzer.py (Python)                      │
│     ├── rs2mermaid (Rust)                                   │
│     ├── go_analyzer (Go)                                     │
│     └── ts2mermaid.ts (TypeScript)                          │
│                                                             │
│  2. [統一分類] → aiva_external_classifier.py                 │
│     └── 輸出: external_classification.json (628 flows)       │
│                                                             │
│  3. [統一執行] → aiva_external_executor.py ⭐                 │
│     └── 讀取 external_classification.json                   │
│     └── 提供 CLI 接口執行所有能力                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**重要原則**:
- ❌ **不要**直接 Python 導入模組（如 `from services.features.function_xss import ...`）
- ✅ **應該**通過 `aiva_external_executor.py` 統一調用
- ✅ 這樣可確保：參數一致、錯誤處理、日誌記錄、多語言支援

---

## 🎯 正確使用方式

### 1. 列出所有可用能力

```bash
# 列出所有語言的能力
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration
python aiva_external_executor.py --list

# 列出特定語言
python aiva_external_executor.py --list --lang python
python aiva_external_executor.py --list --lang rust
python aiva_external_executor.py --list --lang go
python aiva_external_executor.py --list --lang typescript
```

**輸出示例**:
```
============================================================
  PYTHON 能力 (607 個)
============================================================

[function_xss]
  類型: injection / web_security
  說明: XSS 漏洞檢測
  用途: XSS testing from OastHttpCallbackStore.register_probe to isinstance
  流程數: 174

    1. register_probe → isinstance (長度: 11)
    2. fetch_events → append (長度: 9)
    3. __init__ → OastHttpCallbackStore (長度: 3)
    ... 還有 171 個流程

[function_sqli]
  類型: injection / web_security
  說明: SQL 注入檢測
  流程數: 146
  ...
```

---

### 2. 互動式選單（推薦新手使用）

```bash
# 啟動互動式選單
python aiva_external_executor.py --menu
```

選單功能：
- 按模組瀏覽能力
- 按語言篩選
- 查看詳細參數
- 直接執行測試

---

### 3. 執行特定能力

#### 方式 A: 使用 Flow ID

```bash
# 執行 flow ID = 101（XSS 相關）
python aiva_external_executor.py --lang python --flow 101

# 帶參數執行
python aiva_external_executor.py --lang python --flow 101 --target https://example.com
```

#### 方式 B: 使用函數名稱

```bash
# 執行 XSS 綜合掃描
python aiva_external_executor.py --lang python \
    --func XSSManager.comprehensive_scan \
    --target https://example.com/search?q=test

# 執行 DOM XSS 檢測
python aiva_external_executor.py --lang python \
    --func DOMXSSDetector.scan_dom_xss \
    --target https://example.com

# 執行 SQL 注入檢測
python aiva_external_executor.py --lang python \
    --func detect_sqli \
    --target https://example.com/login \
    --param username

# 執行 SSRF 檢測
python aiva_external_executor.py --lang python \
    --func SSRFDetector.scan_ssrf \
    --target https://example.com/fetch
```

---

### 4. Dry Run 模式（測試命令）

```bash
# 只顯示命令，不實際執行
python aiva_external_executor.py --lang python \
    --func run_reflected_test \
    --target https://example.com \
    --dry-run
```

**輸出示例**:
```
===========================================================
執行 Python 流程 #58
===========================================================

[模組] function_xss
[類型] injection / web_security
[描述] XSS 漏洞檢測
[用途] XSS testing from run_reflected_test to detector.execute

[流程] run_reflected_test → detector.execute
[路徑] run_reflected_test → FunctionTaskPayload → XssPayloadGenerator → 
       generator.generate_basic_payloads → TraditionalXssDetector → detector.execute
[長度] 6 步驟

[目標] https://example.com

===========================================================
Dry Run 模式 - 不實際執行
===========================================================

[說明] 這個流程將會：
  1. 導入模組: services/features/function_xss
  2. 執行入口: run_reflected_test
  3. 測試目標: https://example.com

[建議] 何時使用此流程：
  XSS testing from run_reflected_test to detector.execute
```

---

## 📊 可用模組總覽

### 高完成度模組（10 個）

| 模組 | 完成度 | 流程數 | 主要功能 |
|------|--------|--------|----------|
| **function_xss** | 高 | 174 | XSS 漏洞檢測（DOM/反射/存儲/盲測） |
| **function_sqli** | 高 | 146 | SQL 注入檢測（時間盲注/布爾盲注/錯誤注入） |
| **function_ssrf** | 高 | 76 | SSRF 漏洞檢測 |
| **function_idor** | 高 | 33 | IDOR 漏洞檢測 |
| **function_bizlogic** | 高 | 28 | 業務邏輯漏洞檢測 |
| **function_crypto** | 高 | 4 (Rust) | 加密漏洞檢測 |
| **function_info_leak** | 高 | - | 信息洩露檢測 |
| **function_authn_go** | 中 | 13 (Go) | 認證繞過檢測 |
| **function_postex** | 中 | - | 後滲透工具 |
| **function_web_scanner** | 中 | 74 | Web 掃描器 |

### 語言分布

| 語言 | 流程數 | 模組 |
|------|--------|------|
| Python | 607 | 9 個功能模組 |
| Rust | 4 | function_crypto |
| Go | 13 | function_authn_go |
| TypeScript | 4 | typescript_engine |

---

## 💡 使用範例

### 範例 1: XSS 檢測完整流程

```bash
# 1. 查看 XSS 模組的所有能力
python aiva_external_executor.py --list --lang python | findstr "function_xss"

# 2. 使用反射型 XSS 測試（已驗證可用）
python aiva_external_executor.py --lang python \
    --func run_reflected_test \
    --target https://example.com/search \
    --param q \
    --method GET \
    --timeout 30

# 3. DOM XSS 測試
python aiva_external_executor.py --lang python \
    --func run_dom_test \
    --target https://example.com

# 4. 存儲型 XSS 測試
python aiva_external_executor.py --lang python \
    --func run_stored_test \
    --target https://example.com/comment \
    --param message
```

**實際測試結果示例** (Juice Shop靶場):
```bash
# 發現真實漏洞
python aiva_external_executor.py --lang python \
    --func run_reflected_test \
    --target http://localhost:3000/rest/track-order/ \
    --param test

# 輸出：
[INFO] 啟動反射型 XSS 測試: http://localhost:3000/rest/track-order/ (Param: test)
[INFO] 已生成 3 個測試 Payloads
[INFO] HTTP Request: GET http://localhost:3000/rest/track-order/?test=%3Cscript%3Ealert%281%29%3C%2Fscript%3E "HTTP/1.1 500 Internal Server Error"
[INFO] HTTP Request: GET http://localhost:3000/rest/track-order/?test=%22%27%3E%3Csvg%2Fonload%3Dalert%281%29%3E "HTTP/1.1 500 Internal Server Error"
[INFO] HTTP Request: GET http://localhost:3000/rest/track-order/?test=%3Cimg+src%3Dx+onerror%3Dalert%281%29%3E "HTTP/1.1 500 Internal Server Error"

[結果] 發現 3 個 XSS 漏洞（payload 反射在錯誤訊息中）
```

### 範例 2: SQL 注入檢測

```bash
# 時間盲注檢測
python aiva_external_executor.py --lang python \
    --func detect_time_based_sqli \
    --target https://example.com/login \
    --param username

# 布爾盲注檢測
python aiva_external_executor.py --lang python \
    --func detect_boolean_sqli \
    --target https://example.com/product?id=1
```

### 範例 3: SSRF 檢測

```bash
python aiva_external_executor.py --lang python \
    --func SSRFDetector.scan_ssrf \
    --target https://example.com/proxy \
    --param url
```

### 範例 4: Rust 加密檢測

```bash
python aiva_external_executor.py --lang rust \
    --func analyze_cookies \
    --cookies-json '["sessionid=abc123"]' \
    --url 'https://example.com'
```

---

## 📚 相關文檔

### 功能參考文檔

| 模組 | 文檔位置 |
|------|----------|
| XSS | [`function_xss_operable_classification.md`](function_xss_operable_classification.md) |
| SQLi | 待生成 |
| SSRF | 待生成 |
| IDOR | 待生成 |
| Others | 待生成 |

### 技術文檔

- **執行器 README**: [`services/core/aiva_core/internal_exploration/README.md`](services/core/aiva_core/internal_exploration/README.md)
- **分類器說明**: [`services/core/aiva_core/internal_exploration/python_tools/README.md`](services/core/aiva_core/internal_exploration/python_tools/README.md)
- **路徑配置**: [`services/integration/data/internal_exploration/paths_config.py`](services/integration/data/internal_exploration/paths_config.py)

---

## ⚙️ 進階用法

### 生成文檔

```bash
# 生成 Markdown 參考文件
python aiva_external_executor.py --generate-doc md

# 生成 JSON 數據庫
python aiva_external_executor.py --generate-doc json
```

### 動態參數

```bash
# 執行器支援動態參數（未知參數會被傳遞給能力）
python aiva_external_executor.py --lang python \
    --func custom_function \
    --custom-param1 value1 \
    --custom-param2 value2
```

### 分類篩選

```bash
# 按分類列出能力
python aiva_external_executor.py --list --category injection
python aiva_external_executor.py --list --category reconnaissance
```

---

## 🔧 故障排除

### 問題 1: 找不到分類數據

**錯誤**:
```
[ERROR] 找不到分類數據: external_classification.json
```

**解決**:
```bash
# 先執行分類器生成數據
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration
python aiva_external_classifier.py
```

### 問題 2: 能力執行失敗

**檢查步驟**:
1. 確認 flow ID 或函數名正確
2. 檢查必要參數是否提供
3. 使用 `--dry-run` 查看實際命令
4. 查看執行器日誌

### 問題 3: 參數不匹配

**解決**:
```bash
# 使用 --help 查看可用參數
python aiva_external_executor.py --help

# 使用 --list 查看能力詳情
python aiva_external_executor.py --list --lang python
```

---

## 🎯 重要提醒

### ❌ 錯誤做法

```python
# ❌ 不要直接導入
from services.features.function_xss.xss_manager import XSSManager
manager = XSSManager()
result = manager.scan(...)

# ❌ 不要直接執行腳本
python services/features/function_xss/main.py --target ...
```

### ✅ 正確做法

```bash
# ✅ 通過執行器調用
python aiva_external_executor.py --lang python \
    --func XSSManager.comprehensive_scan \
    --target https://example.com
```

---

## 📝 適用範圍

本指南適用於：

- ✅ **所有 Features 模組**（function_xss, function_sqli, function_ssrf, ...）
- ✅ **所有 Scan 模組**（typescript_engine, ...）
- ✅ **多語言能力**（Python, Rust, Go, TypeScript）
- ✅ **其他中高完成度模組**（參考 [`services/features/README.md`](services/features/README.md)）

---

## 📞 支援

- **技術問題**: 查看 [internal_exploration/README.md](services/core/aiva_core/internal_exploration/README.md)
- **功能請求**: 提交至 AIVA 項目倉庫
- **Bug 報告**: 提供 `--dry-run` 輸出和錯誤日誌

---

**更新日期**: 2026-01-21  
**維護者**: AIVA Development Team
