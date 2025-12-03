# ⚡ XSS 攻擊檢測模組

**什麼是 XSS 檢測？**  
跨站腳本攻擊（XSS）允許攻擊者在受害者瀏覽器中執行惡意腳本。本模組支援三種主要 XSS 類型檢測：反射型（Reflected）、儲存型（Stored）和 DOM 型（DOM-based），並整合 blind XSS 檢測能力。

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

## 🚀 支援指令

### 方式一：直接執行檢測器（開發/測試用）

**適用場景**: 快速驗證、單元測試、本地開發

```bash
# 進入專案目錄
cd c:\D\fold7\AIVA-git

# 執行 XSS 檢測測試
python -c "import asyncio; from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget; from services.features.function_xss.traditional_detector import TraditionalXssDetector; exec('''
async def test_xss():
    # 建立任務結構
    task = FunctionTaskPayload(
        task_id=\"task_xss_001\",
        scan_id=\"scan_001\",
        target=FunctionTaskTarget(
            url=\"http://localhost:3000/rest/products/search\",
            parameter=\"q\",
            method=\"GET\",
            parameter_location=\"query\"
        ),
        strategy=\"normal\"
    )
    
    # 建立檢測器
    detector = TraditionalXssDetector(task, timeout=10.0)
    
    # 定義測試 payloads
    payloads = [
        \"<script>alert(1)</script>\",
        \"<img src=x onerror=alert(2)>\"
    ]
    
    # 執行檢測
    results = await detector.execute(payloads)
    print(f\"Reflections found: {len(results)}\")

asyncio.run(test_xss())
''')"
```

**指令參數說明**:
- `task_id`: 任務唯一識別碼，必須以 `task_` 開頭
- `scan_id`: 掃描會話 ID，用於關聯多個任務
- `target.url`: 目標 URL，完整的 HTTP/HTTPS 位址
- `target.parameter`: 要測試的參數名稱（如 `q`, `search`, `input`）
- `target.method`: HTTP 方法 (`GET`, `POST`, `PUT`, `DELETE`)
- `target.parameter_location`: 參數位置 (`query`, `body`, `header`, `cookie`)
- `strategy`: 檢測策略 (`normal`, `aggressive`, `stealth`)
- `timeout`: 超時時間（秒），預設 10.0

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

```python
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common import get_command_center

# 建立命令中心連線
command_center = get_command_center()

# XSS 檢測命令
command = AICommand(
    command_id="xss_test_001",
    command_type=CommandType.FEATURE_XSS_TEST,
    target_module="features.xss",
    payload={
        "target_url": "https://vulnerable-app.com/comment",
        "test_parameters": {
            "comment": "test content",
            "name": "user",
            "email": "user@test.com"
        },
        "xss_types": ["reflected", "stored", "dom"],
        "timeout": 30,
        "blind_callback_url": "https://xss.callback.domain"  # Blind XSS 檢測
    }
)

# 執行檢測
result = await command_center.execute(command)
```

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
    "target_url": "https://app.com/search",
    "test_parameters": {"q": "search_term"},
    "xss_types": ["reflected"],
    "payloads": ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
}

# 2. 儲存型 XSS 檢測
stored_test = {
    "target_url": "https://app.com/api/comments",
    "test_parameters": {"comment": "test", "author": "tester"},
    "xss_types": ["stored"],
    "verification_urls": ["https://app.com/comments/view/123"],  # 檢查儲存的內容
    "cleanup_required": True
}

# 3. DOM-based XSS 檢測
dom_test = {
    "target_url": "https://spa-app.com/#/user/profile",
    "test_parameters": {"name": "test_user"},
    "xss_types": ["dom"],
    "javascript_analysis": True
}

# 4. 綜合檢測（推薦）
comprehensive_test = {
    "target_url": "https://full-app.com/profile/edit",
    "test_parameters": {
        "firstname": "John",
        "lastname": "Doe", 
        "bio": "Test bio"
    },
    "xss_types": ["reflected", "stored", "dom"],
    "blind_callback_url": "https://xss-callback.yourdomain.com",
    "waf_evasion": True,  # 啟用 WAF 繞過技術
    "encoding_tests": ["url", "html", "js", "css"]
}
```

## 🔧 核心能力
- **三類型全覆蓋**: Reflected/Stored/DOM-based XSS 完整檢測
- **Blind XSS 監聽**: 外部回調驗證隱蔽執行
- **WAF 繞過技術**: 多種編碼和混淆方法
- **Context-aware**: 根據注入上下文選擇合適的 payload
- **自動清理**: Stored XSS 測試後自動清理測試數據

## 🎯 後續發展方向
- [ ] **CSP 繞過研究** - 針對嚴格內容安全策略的繞過技術
- [ ] **無文件 XSS** - 基於現代 JavaScript 框架的攻擊
- [ ] **WebAssembly XSS** - 新興技術的 XSS 攻擊向量
- [ ] **AI Payload 生成** - 基於目標特徵自動生成定制 payload