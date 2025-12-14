# ⚡ XSS 攻擊檢測模組

**版本**: v2.0 | **狀態**: ✅ 生產就緒 | **更新**: 2025-12-12

**什麼是 XSS 檢測？**  
跨站腳本攻擊（XSS）允許攻擊者在受害者瀏覽器中執行惡意腳本。本模組支援三種主要 XSS 類型檢測：反射型（Reflected）、儲存型（Stored）和 DOM 型（DOM-based），並整合 blind XSS 檢測能力。

## 📚 快速導航

- [🚀 CLI 使用方式](#-cli-使用方式) - **推薦：無需 MQ，直接測試**
- [⚙️ 運作流程](#️-運作流程)
- [🔧 核心能力](#-核心能力)
- [📋 參數說明](#-參數說明)

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                    三合一 XSS 檢測架構                         │
├─────────────────────────────────────────────────────────────┤
│ AI Command      │command_handler │  XSS Detectors  │ 外部工具 │
│ Interface       │               │                 │ 整合     │
│       ↓         │       ↓       │        ↓        │    ↓     │
│ FEATURE_XSS_    │ FunctionTask  │traditional_     │ dalfox   │
│ TEST            │ Payload       │ detector        │ xsstrike │
│       │         │               │stored_detector  │          │
│       └─────────┼───────────────┼─dom_xss_       │    ↓     │
│                 │               │ detector        │ blind_xss│
│                 ↓               │        ↓        │ listener │
│         XssDetectionResult      │ blind_xss_      │          │
│         (integration_tools)     │ validator       │          │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **輸入點分析** - 掃描表單字段、URL 參數、Header 和 Cookie
2. **檢測器選擇** - 根據上下文選擇適當的檢測策略
3. **多類型檢測** - 並行執行三種檢測模式：
   - **Reflected XSS**: 即時反射，檢測 payload 是否直接返回頁面
   - **Stored XSS**: 持久化存儲，檢測 payload 是否儲存後執行
   - **DOM XSS**: 客戶端檢測，分析 JavaScript 執行環境
4. **Blind XSS 驗證** - 使用外部監聽器確認隱藏執行

## 🚀 CLI 使用方式

### ⭐ 推薦：直接 CLI 測試（無需 MQ）

**版本**: v2.0 新增 | **狀態**: ✅ 完整支援

```powershell
# 在專案根目錄執行

# 1. 反射型 XSS (Juice Shop)
python -m services.features.function_xss `
    --url "http://localhost:3000/rest/products/search" `
    --param "q" `
    --type reflected `
    --timeout 30

# 2. DOM XSS
python -m services.features.function_xss `
    --url "http://localhost:3000/#/search" `
    --type dom

# 3. 儲存型 XSS
python -m services.features.function_xss `
    --url "http://localhost:3000/api/comments" `
    --param "comment" `
    --type stored `
    --method POST `
    --location body `
    --view-url "http://localhost:3000/comments"
```

**輸出格式** (JSON):
```json
{
  "target": "http://localhost:3000/rest/products/search",
  "type": "reflected",
  "findings_count": 3,
  "vulnerable": true,
  "findings": [
    {
      "payload": "<script>alert(1)</script>",
      "status": 200,
      "vulnerable": true,
## 📋 參數說明

### CLI 參數

| 參數 | 必填 | 預設值 | 說明 |
|------|------|--------|------|
| `--url` | ✅ | - | 目標 URL |
| `--type` | ❌ | reflected | 檢測類型 (reflected/dom/stored) |
| `--param` | ❌ | q | 測試參數名稱 |
| `--method` | ❌ | GET | HTTP 方法 (GET/POST) |
| `--location` | ❌ | query | 參數位置 (query/body/header) |
| `--timeout` | ❌ | 30 | 超時秒數 |
| `--view-url` | ❌ | - | 查看頁面 URL (僅 stored 類型) |

### 程式化參數

```python
# GET 參數測試
target=FunctionTaskTarget(
    url="http://example.com/search",
    parameter="q",
    method="GET",
    parameter_location="query"
)

# POST Body 測試
target=FunctionTaskTarget(
    url="http://example.com/comment",
    parameter="content",
    method="POST",
    parameter_location="body",
    form_data={"username": "test"}
)

# Header 測試
target=FunctionTaskTarget(
    url="http://example.com/api",
    parameter="X-Custom-Header",
    method="GET",
    parameter_location="header"
)
```

---

### 方式三：透過 Message Queue（已棄用）

**狀態**: ⚠️ 已棄用，請使用 CLI 方式

<details>
<summary>點擊查看舊版 MQ 方式（不推薦）</summary>
**參數變化範例**:
```python
# GET 參數測試
target=FunctionTaskTarget(
    url="http://example.com/search",
    parameter="q",
    method="GET",
    parameter_location="query"
)

# POST Body 測試
target=FunctionTaskTarget(
    url="http://example.com/comment",
    parameter="content",
    method="POST",
    parameter_location="body",
    form_data={"username": "test"}  # 其他表單字段
)

# Header 測試
target=FunctionTaskTarget(
    url="http://example.com/api",
    parameter="X-Custom-Header",
    method="GET",
    parameter_location="header"
)

# Cookie 測試
target=FunctionTaskTarget(
    url="http://example.com/profile",
    parameter="session_id",
    method="GET",
    parameter_location="cookie"
)
```

### 方式二：透過 Message Queue（生產環境用）

**適用場景**: 分散式架構、非同步任務、生產環境
# 舊版 MQ 方式（不推薦使用）
from services.aiva_common.schemas import AICommand, CommandType
command = AICommand(
    command_id="xss_test_001",
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"target_url": "https://example.com"}
)
```

</details>

---

## 🎯 使用場景

### 何時使用？
- ✅ **適用場景**:
  - **表單輸入檢測**: 留言板、評論系統、用戶資料編輯
  - **搜尋功能測試**: 搜尋結果頁面的反射型 XSS
  - **富文本編輯器**: HTML 編輯器的 XSS 過濾測試
  - **API 數據注入**: JSON/XML API 的 XSS 檢測
  
- ⚠️ **使用注意**:
  - 避免在生產環境執行可能影響用戶的 payload
  - 注意 CSP（內容安全策略）可能阻止檢測
  - Stored XSS 檢測後需要清理測試數據

### 如何使用？
```python
# 1. 反射型 XSS 檢測
reflected_test = {
### ✅ 適用場景

- **表單輸入檢測**: 留言板、評論系統、用戶資料編輯
- **搜尋功能測試**: 搜尋結果頁面的反射型 XSS
- **富文本編輯器**: HTML 編輯器的 XSS 過濾測試
- **API 數據注入**: JSON/XML API 的 XSS 檢測

### ⚠️ 使用注意

- 避免在生產環境執行可能影響用戶的 payload
- 注意 CSP（內容安全策略）可能阻止檢測
- Stored XSS 檢測後需要清理測試數據
- 僅在授權的目標上進行測試

---

## 🎯 測試範例

### Juice Shop 完整測試

```powershell
# 1. 搜尋框 (反射型)
python -m services.features.function_xss `
    --url "http://localhost:3000/rest/products/search" `
    --param "q" `
    --type reflected

# 2. 留言板 (儲存型)
python -m services.features.function_xss `
    --url "http://localhost:3000/api/feedbacks" `
    --param "comment" `
    --type stored `
    --method POST `
    --location body `
    --view-url "http://localhost:3000/feedbacks"
- ✅ **三類型全覆蓋**: Reflected/Stored/DOM-based XSS 完整檢測
- ✅ **CLI 直接測試**: 無需 MQ，即開即用
- ✅ **虛假回應過濾**: 執行上下文驗證、WAF 干擾檢測
- ✅ **Blind XSS 監聽**: 外部回調驗證隱蔽執行
- ✅ **Context-aware**: 根據注入上下文選擇合適的 payload
- ✅ **跨語言工具整合**: 支援 Go/Ruby/Python/Rust 工具

## 📝 更新日誌

### v2.0 (2025-12-12)
- ✅ 新增 CLI 入口 (`__main__.py`)
- ✅ 修復 hackingtool_engine.py timeout 參數 bug
- ✅ 強化 traditional_detector.py 虛假回應過濾
- ✅ 強化 stored_detector.py 持久化驗證
- ✅ 移除 MQ 依賴，改用直接調用模式

### v1.0
- 初始版本，支援三種 XSS 檢測類型

## 🎯 後續發展方向

- [ ] **CSP 繞過研究** - 針對嚴格內容安全策略的繞過技術
- [ ] **無文件 XSS** - 基於現代 JavaScript 框架的攻擊
- [ ] **WebAssembly XSS** - 新興技術的 XSS 攻擊向量
- [ ] **AI Payload 生成** - 基於目標特徵自動生成定制 payload

---

## 📚 相關文檔

- [CLI_USAGE.md](./CLI_USAGE.md) - CLI 詳細使用指南
- [SIMPLE_ARCHITECTURE.md](../SIMPLE_ARCHITECTURE.md) - 功能模組架構設計
- [FALSE_POSITIVE_ANALYSIS.md](../FALSE_POSITIVE_ANALYSIS.md) - 虛假回應分析報告