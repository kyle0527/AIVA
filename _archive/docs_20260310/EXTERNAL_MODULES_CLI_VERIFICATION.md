# AIVA 外部模組 CLI 執行驗證報告

## 📑 目錄

- [執行結果總結](#執行結果總結)
  - [✅ 可用的 CLI 執行方式](#-可用的-cli-執行方式)
    - [1. 統一 CLI 執行器 (推薦)](#1-統一-cli-執行器-推薦)
    - [2. 模組直接 CLI (function_xss)](#2-模組直接-cli-function_xss)
  - [❌ 不可用的 CLI 方式](#-不可用的-cli-方式)
- [模組導入能力分析](#模組導入能力分析)
  - [Python 外部模組 (8個)](#python-外部模組-8個)
  - [Go 模組 (1個)](#go-模組-1個)
  - [TypeScript 模組 (1個)](#typescript-模組-1個)
- [當前狀況](#當前狀況)
  - [✅ 已完成](#-已完成)
  - [⏳ 待完成](#-待完成)
- [推薦執行方案](#推薦執行方案)
  - [方案 A: 使用模組自帶 CLI（僅 XSS）](#方案-a-使用模組自帶-cli僅-xss)
  - [方案 B: 使用 CommandHandler（需異步）](#方案-b-使用-commandhandler需異步)
  - [方案 C: 完善統一執行器（推薦）](#方案-c-完善統一執行器推薦)
- [下一步行動](#下一步行動)
  - [立即可做](#立即可做)
  - [中期目標](#中期目標)
- [驗證命令快查](#驗證命令快查)
- [結論](#結論)

---


生成時間: 2026-01-14
測試環境: Windows, Python 3.13
靶場: juice-shop (localhost:3000), webgoat (localhost:8080)

## 執行結果總結

### ✅ 可用的 CLI 執行方式

#### 1. 統一 CLI 執行器 (推薦)
```bash
# 查看所有可用流程
python aiva_external_executor.py --list

# 按語言篩選
python aiva_external_executor.py --list --lang python

# Dry-run 預覽
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run

# 啟動互動式選單
python aiva_external_executor.py --menu
```

**狀態**: ✅ 基礎功能完成
- 流程列表顯示: ✅
- 互動式選單: ✅  
- Dry-run 模式: ✅
- 實際執行: ⏳ 待完善（需要動態導入邏輯）

#### 2. 模組直接 CLI (function_xss)
```bash
# XSS 反射型測試
python -m services.features.function_xss \
  --url "http://localhost:3000" \
  --type reflected \
  --param "q" \
  --method GET \
  --location query \
  --timeout 10
```

**測試結果**:
```json
{
  "target": "http://localhost:3000",
  "type": "reflected",
  "findings_count": 0,
  "vulnerable": false,
  "findings": []
}
```

✅ **已驗證**: XSS 模組可以實際執行並返回結果

**可用類型**:
- `reflected`: 反射型 XSS
- `dom`: DOM 型 XSS
- `stored`: 存儲型 XSS

### ❌ 不可用的 CLI 方式

以下模組**沒有** `__main__.py`，無法直接用 `-m` 執行：
- `services.features.function_ssrf`
- `services.features.function_sqli`
- `services.features.function_idor`
- `services.features.function_bizlogic`
- `services.features.function_authn_go`

## 模組導入能力分析

### Python 外部模組 (8個)

| 模組 | 直接CLI | CommandHandler | Detector/Worker | 流程數 |
|------|---------|----------------|-----------------|--------|
| function_xss | ✅ | ✅ | ✅ | 109 |
| function_ssrf | ❌ | ✅ | ✅ | 35 |
| function_sqli | ❌ | ✅ | ⏳ | 32 |
| function_idor | ❌ | ✅ | ✅ | 19 |
| function_bizlogic | ❌ | ✅ | ✅ | 8 |

### Go 模組 (1個)

| 模組 | 可執行 | 流程數 |
|------|--------|--------|
| function_authn_go | ⏳ | 4 |

### TypeScript 模組 (1個)

| 模組 | 可執行 | 流程數 |
|------|--------|--------|
| typescript_engine | ⏳ | 3 |

## 當前狀況

### ✅ 已完成
1. **分類數據生成**: 210 個流程，8 個模組，3 種語言
2. **文檔生成**: Markdown + JSON 資料庫
3. **互動式選單**: 三層展示（語言→模組→能力→變體）
4. **Dry-run 模式**: 可預覽執行計畫
5. **XSS 模組驗證**: 實際執行成功

### ⏳ 待完成
1. **統一執行器的實際執行邏輯**: 
   - 當前只有 dry-run
   - 需要實現動態導入和實例化
   - 需要根據模組類型選擇執行方式

2. **其他模組的 CLI 入口**:
   - SSRF, SQLi, IDOR, BizLogic 沒有 __main__.py
   - 可以添加或統一通過 CommandHandler

3. **Go 和 TypeScript 執行**:
   - 需要調用外部命令 (go run, npx ts-node)
   - 需要處理跨語言參數傳遞

## 推薦執行方案

### 方案 A: 使用模組自帶 CLI（僅 XSS）
```bash
# 直接使用 XSS 模組
python -m services.features.function_xss --url <TARGET> --type reflected --param q
```
✅ 優點: 直接可用，參數明確
❌ 缺點: 只有 XSS 可用

### 方案 B: 使用 CommandHandler（需異步）
```python
from services.features.function_xss import XSSCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

handler = XSSCommandHandler()
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"target_url": "http://localhost:3000"}
)
result = await handler.handle_command(command)
```
✅ 優點: 標準化接口，功能完整
❌ 缺點: 需要異步環境，不是純 CLI

### 方案 C: 完善統一執行器（推薦）
```bash
# 目標: 讓這個命令真正執行
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000
```
✅ 優點: 統一接口，支持所有模組
⏳ 狀態: 需要實現動態導入邏輯

## 下一步行動

### 立即可做
1. ✅ **為其他模組添加 __main__.py**
   - 複製 function_xss 的模式
   - SSRF, SQLi, IDOR 都可以快速添加

2. ✅ **完善統一執行器**
   - 實現 execute_python() 的動態導入
   - 根據流程定義找到正確的類別和方法
   - 處理參數傳遞

### 中期目標
3. **標準化所有模組**
   - 確保都有 CommandHandler
   - 確保都有 CLI 入口
   - 統一參數格式

4. **文檔完善**
   - 每個模組的使用範例
   - 參數說明
   - 常見問題

## 驗證命令快查

```bash
# 1. 列出所有能力
python aiva_external_executor.py --list

# 2. 互動式選單
python aiva_external_executor.py --menu

# 3. XSS 測試 (實際執行)
python -m services.features.function_xss --url http://localhost:3000 --type reflected --param q

# 4. Dry-run 預覽
python aiva_external_executor.py --lang python --flow 101 --target http://localhost:3000 --dry-run

# 5. 生成文檔
python aiva_external_executor.py --generate-doc md
python aiva_external_executor.py --generate-doc json
```

## 結論

**當前狀態**: ✅ 框架完成，XSS 可執行
**主要問題**: 統一執行器需要實現實際執行邏輯
**優先級**: 完善 execute_python() 方法，使其能夠動態導入和執行

---

*報告生成時間: 2026-01-14 01:00*
*測試執行者: AIVA 外部模組整合系統*
