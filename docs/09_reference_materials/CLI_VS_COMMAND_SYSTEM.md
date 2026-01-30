# Internal Exploration CLI 與 AI 命令系統的關係

> **文檔目的**: 說明 `internal_exploration` 的 CLI 設計與 `aiva_common` 命令系統的關係，澄清兩者不衝突  
> **更新日期**: 2026-01-07

---

## 🎯 核心結論：**兩者不衝突，互補使用**

### 簡單對比

| 特性 | Internal Exploration CLI | aiva_common 命令系統 |
|------|--------------------------|---------------------|
| **目的** | **代碼分析與自我診斷** | **AI 執行與模組調用** |
| **使用者** | 開發者/分析師 | AI 決策引擎 |
| **操作方式** | CLI 腳本 (argparse) | Python API (async) |
| **輸入** | 命令行參數 | AICommand 對象 |
| **輸出** | 分析報告/JSON 文件 | AICommandResult 對象 |
| **典型場景** | 查看能力、分析架構 | 執行 XSS 測試、掃描 |

---

## 📚 兩個系統的詳細說明

### 1️⃣ Internal Exploration CLI (internal_exploration/)

**核心定位**: **AIVA 的自我認知系統 - 代碼分析工具**

#### 主要功能

```bash
# ========== 能力查詢 ==========
# 搜尋所有 XSS 相關能力
python aiva_capability_cli.py --search xss

# 查看 Flow 313 的詳細資訊
python aiva_capability_cli.py --info 313

# 列出所有可用能力
python aiva_capability_cli.py --list

# ========== 代碼分析 ==========
# 分析核心模組的數據流
python aiva_flow_analyzer.py --target core --depth 5

# 分類所有能力
python aiva_flow_classifier.py --input flows.json

# ========== 自我修復 ==========
# 檢測數據流斷點
python core_analyzer.py --breakpoints

# 分析缺失連接
python core_analyzer.py --missing-connections
```

#### 典型使用場景

1. **開發者想知道系統有哪些能力**
   ```bash
   python aiva_capability_cli.py --list
   # 輸出：201 個 Flow 列表
   ```

2. **查看某個能力的詳細資訊**
   ```bash
   python aiva_capability_cli.py --info 313
   # 輸出：Flow 313 的函數調用鏈、參數、模組等
   ```

3. **分析系統架構**
   ```bash
   python aiva_flow_analyzer.py --target core
   # 輸出：Mermaid 流程圖、數據流分析
   ```

4. **診斷代碼問題**
   ```bash
   python core_analyzer.py --breakpoints
   # 輸出：數據流斷點、缺失連接報告
   ```

#### 關鍵特徵

- ✅ **靜態分析**: 不執行代碼，只分析結構
- ✅ **開發工具**: 給開發者用於理解和維護系統
- ✅ **文檔生成**: 自動生成流程圖、能力清單
- ✅ **自我診斷**: 檢測架構問題

---

### 2️⃣ aiva_common 命令系統 (services/aiva_common/)

**核心定位**: **AI 執行與模組調用的統一介面**

#### 主要功能

```python
# ========== AI 執行功能模組 ==========
from services.aiva_common.command_center import AICommandCenter
from services.aiva_common.schemas.commands import AICommand, CommandType

command_center = AICommandCenter()

# 註冊處理器
command_center.register_module("features.xss", XSSCommandHandler())

# 執行 XSS 測試
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features.xss",
    payload={"target_url": "https://example.com"}
)
result = await command_center.execute(command)

# ========== AI 決策驅動 ==========
from cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent

decision = await decision_agent.decide(context)
result = await decision_agent.execute_decision(decision, context)
```

#### 典型使用場景

1. **AI 執行 XSS 測試**
   ```python
   # AI 決策：需要測試 XSS
   command = AICommand(
       command_type=CommandType.FEATURE_XSS_TEST,
       payload={"target_url": "https://example.com"}
   )
   result = await command_center.execute(command)
   # 返回：發現的漏洞列表
   ```

2. **AI 自動選擇測試工具**
   ```python
   # AI 根據技術棧決定使用哪些工具
   context = DecisionContext(tech_stack=["PHP", "MySQL"])
   decision = await decision_agent.decide(context)
   # AI 決定：使用 SQLi 和 XSS 測試
   ```

3. **批次執行安全掃描**
   ```python
   # 並行執行多個測試
   commands = [xss_command, sqli_command, ssrf_command]
   results = await asyncio.gather(*[
       command_center.execute(cmd) for cmd in commands
   ])
   ```

#### 關鍵特徵

- ✅ **動態執行**: 實際執行測試和掃描
- ✅ **AI 驅動**: 由 AI 決策引擎調用
- ✅ **異步執行**: 支援並行和批次操作
- ✅ **標準化介面**: 統一的命令和結果格式

---

## 🔗 兩者的協作關係

### 完整工作流程

```
┌──────────────────────────────────────────────────────────────┐
│                        開發階段                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  開發者使用 Internal Exploration CLI                          │
│                                                               │
│  1️⃣ 查看系統有哪些能力                                        │
│     python aiva_capability_cli.py --list                     │
│     → 輸出：201 個 Flow，包含 XSS、SQLi、SSRF 等             │
│                                                               │
│  2️⃣ 查看 Flow 313 (XSS 測試) 的詳細資訊                       │
│     python aiva_capability_cli.py --info 313                 │
│     → 輸出：入口函數、參數、依賴模組                          │
│                                                               │
│  3️⃣ 分析代碼架構                                              │
│     python aiva_flow_analyzer.py --target features           │
│     → 輸出：Mermaid 流程圖、數據流分析                        │
│                                                               │
│  4️⃣ 診斷潛在問題                                              │
│     python core_analyzer.py --breakpoints                    │
│     → 輸出：數據流斷點、缺失連接                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
                   開發者理解了系統架構
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                        運行階段                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  AI 使用 aiva_common 命令系統                                 │
│                                                               │
│  1️⃣ AI 決策：需要執行 XSS 測試                                │
│     context = DecisionContext(...)                           │
│     decision = await decision_agent.decide(context)          │
│     → AI 決定：執行 XSS 測試                                  │
│                                                               │
│  2️⃣ AI 調用 XSS 命令處理器                                    │
│     command = AICommand(                                     │
│         command_type=CommandType.FEATURE_XSS_TEST,           │
│         payload={"target_url": "https://example.com"}        │
│     )                                                         │
│     result = await command_center.execute(command)           │
│                                                               │
│  3️⃣ XSS 測試執行                                              │
│     XSSCommandHandler.handle_command()                       │
│     → XSSManager.comprehensive_scan()                        │
│     → 返回：AICommandResult (發現的漏洞)                      │
│                                                               │
│  4️⃣ AI 評估結果並決定下一步                                   │
│     if result.success:                                       │
│         # 發現漏洞，更新上下文                                │
│         context.discovered_vulns.extend(...)                 │
│         # AI 決定下一步行動                                   │
│         next_decision = await decision_agent.decide(context) │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 具體範例：XSS 測試

### 場景：開發者想了解 XSS 測試能力

#### 使用 Internal Exploration CLI

```bash
# 1. 搜尋 XSS 相關能力
$ python aiva_capability_cli.py --search xss

找到 5 個相關能力：
┌─────┬────────────────────────┬─────────────┬──────┐
│ ID  │ 能力名稱                │ 模組        │ 標籤 │
├─────┼────────────────────────┼─────────────┼──────┤
│ 313 │ XSS 綜合檢測           │ features    │ 攻擊 │
│ 314 │ DOM XSS 掃描           │ features    │ 攻擊 │
│ 315 │ 存儲型 XSS 測試        │ features    │ 攻擊 │
│ 316 │ 反射型 XSS 測試        │ features    │ 攻擊 │
│ 317 │ 盲 XSS 測試            │ features    │ 攻擊 │
└─────┴────────────────────────┴─────────────┴──────┘

# 2. 查看 Flow 313 的詳細資訊
$ python aiva_capability_cli.py --info 313

╔═════════════════════════════════════════════════════════════╗
║                    Flow 313 詳細資訊                        ║
╚═════════════════════════════════════════════════════════════╝

📌 基本資訊:
   能力名稱: XSS 綜合檢測
   所屬模組: features (功能模組)
   分類標籤: 攻擊 (Attack)
   
📍 入口函數:
   function_xss/integration_tools/xss_tools.py::comprehensive_scan
   
🔗 函數調用鏈:
   1. comprehensive_scan()
   2.   → detect_traditional_xss()
   3.   → detect_dom_xss()
   4.   → detect_stored_xss()
   5.   → detect_blind_xss()
   
📥 輸入參數:
   - target_url: str (必填)
   - scan_options: dict (可選)
   
📤 返回結果:
   - vulnerabilities: list[XSSVulnerability]
   - summary: ScanSummary
   
🏷️  依賴模組:
   - services.features.function_xss
   - services.aiva_common.schemas
```

#### 使用 aiva_common 命令系統 (AI 執行)

```python
# AI 實際執行 XSS 測試

from services.features.function_xss.command_handler import XSSCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

# 1. 創建命令處理器
xss_handler = XSSCommandHandler()

# 2. 構建 AI 命令
command = AICommand(
    command_id="xss_test_001",
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features.xss",
    payload={
        "target_url": "https://example.com/search",
        "scan_type": "comprehensive",
        "options": {
            "use_dalfox": True,
            "scan_dom": True,
            "scan_stored": True
        }
    }
)

# 3. 執行測試
result = await xss_handler.handle_command(command)

# 4. 處理結果
if result.success:
    print(f"✅ 發現 {len(result.result['vulnerabilities'])} 個漏洞")
    for vuln in result.result['vulnerabilities']:
        print(f"  - {vuln['type']}: {vuln['severity']}")
        print(f"    URL: {vuln['url']}")
        print(f"    Payload: {vuln['payload']}")
else:
    print(f"❌ 測試失敗: {result.error}")
```

---

## 🎨 使用建議

### 何時使用 Internal Exploration CLI？

**✅ 適合場景**:
- 開發者想了解系統有哪些能力
- 查看某個功能的實現細節
- 分析代碼架構和數據流
- 診斷系統問題（斷點、缺失連接）
- 生成文檔和流程圖
- 重構前的架構分析

**範例命令**:
```bash
# 查看所有能力
python aiva_capability_cli.py --list

# 搜尋特定功能
python aiva_capability_cli.py --search sqli

# 分析代碼架構
python aiva_flow_analyzer.py --target core

# 診斷問題
python core_analyzer.py --breakpoints
```

---

### 何時使用 aiva_common 命令系統？

**✅ 適合場景**:
- AI 需要執行實際的測試或掃描
- 運行時調用功能模組
- 批次執行多個任務
- AI 決策驅動的自動化操作
- 需要返回結構化結果供 AI 分析

**範例代碼**:
```python
# AI 執行測試
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"target_url": "https://example.com"}
)
result = await command_center.execute(command)

# AI 決策驅動
decision = await decision_agent.decide(context)
result = await decision_agent.execute_decision(decision, context)
```

---

## 📊 對比總結表

| 特性 | Internal Exploration CLI | aiva_common 命令系統 |
|------|--------------------------|---------------------|
| **主要用途** | 代碼分析、能力查詢、自我診斷 | AI 執行、模組調用、測試運行 |
| **使用者** | 開發者、分析師 | AI 決策引擎、自動化系統 |
| **執行方式** | CLI 腳本 (bash/python) | Python API (async) |
| **是否執行代碼** | ❌ 否 (靜態分析) | ✅ 是 (動態執行) |
| **輸入格式** | 命令行參數 | AICommand 對象 |
| **輸出格式** | 文本/JSON 文件/Mermaid 圖 | AICommandResult 對象 |
| **典型場景** | 查看能力清單、分析架構 | 執行 XSS 測試、AI 決策 |
| **依賴關係** | 獨立運行 | 需要模組處理器 |
| **適用階段** | 開發階段 | 運行階段 |
| **文檔生成** | ✅ 是 | ❌ 否 |
| **AI 整合** | ❌ 否 | ✅ 是 |
| **並行執行** | ❌ 否 | ✅ 是 |

---

## 🎯 結論

### ✅ 兩者**不衝突**，而是**互補**

1. **Internal Exploration CLI** = 開發者工具
   - 幫助開發者**理解**系統
   - 分析代碼結構
   - 診斷問題
   - 生成文檔

2. **aiva_common 命令系統** = AI 執行引擎
   - 幫助 AI **操作**系統
   - 執行測試
   - 調用功能
   - 返回結果

### 📝 使用建議

**您說「打算用 CLI 操作」是完全正確的！**

- **開發階段**: 使用 `aiva_capability_cli.py` 查看能力
- **運行階段**: AI 使用 `aiva_common` 命令系統執行

兩者可以同時存在，互不干擾：
- CLI 工具幫助您**了解**系統
- 命令系統幫助 AI **使用**系統

---

**文檔完成**: 2026-01-07  
**結論**: Internal Exploration CLI 與 aiva_common 命令系統**不衝突**，請放心使用 CLI 進行能力查詢和代碼分析！
