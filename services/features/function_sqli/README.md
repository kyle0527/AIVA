# 🛡️ SQL 注入檢測模組

**什麼是 SQL 注入檢測？**  
SQL 注入是最常見的 Web 應用程式漏洞之一，攻擊者通過在輸入字段中注入惡意 SQL 代碼來操控資料庫。本模組使用多引擎並行檢測技術，能識別布林型、時間型、聯合查詢型等各種注入類型。

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   多引擎 SQL 注入檢測架構                      │
├─────────────────────────────────────────────────────────────┤
│  AI Command     │ command_handler │  SqliDetector   │  外部工具│
│  Interface      │                │                 │ 整合    │
│       ↓         │        ↓       │        ↓        │    ↓    │
│ FEATURE_SQLI_   │ FunctionTask   │ boolean_engine  │ sqlmap  │
│ TEST            │ Payload        │ time_engine     │ hacktool│
│       │         │                │ union_engine    │         │
│       └─────────┼────────────────┼─ error_engine  │    ↓    │
│                 │                │ oob_engine      │ 外部    │
│                 ↓                │        ↓        │ 驗證    │
│           DetectionResult        │ 並行執行 +      │         │
│           (detection_models)     │ 智能篩選        │         │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **目標解析** - 解析 URL 和參數，識別可能的注入點
2. **引擎選擇** - 根據目標特性選擇合適的檢測引擎組合
3. **並行檢測** - 同時執行多個檢測引擎：
   - **Boolean-based**: 基於真/假響應的邏輯判斷
   - **Time-based**: 通過延遲響應檢測盲注
   - **Union-based**: 聯合查詢數據提取檢測
   - **Error-based**: 利用資料庫錯誤訊息檢測
   - **Out-of-band**: 帶外通道檢測（DNS/HTTP）
4. **結果整合** - 綜合多引擎結果，生成統一的檢測報告

## 🚀 支援指令

### 方式一:直接執行檢測器(開發/測試用)

**適用場景**: 快速驗證、單元測試、本地開發

```bash
# 進入專案目錄
cd c:\D\fold7\AIVA-git

# 執行 SQLi Error-Based 檢測測試
python -c "import asyncio; import httpx; from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget; from services.features.function_sqli.engines.error_detection_engine import ErrorDetectionEngine; exec('''
async def test_sqli():
    # 建立任務結構
    task = FunctionTaskPayload(
        task_id=\"task_sqli_001\",
        scan_id=\"scan_001\",
        target=FunctionTaskTarget(
            url=\"http://localhost:3000/rest/products/search\",
            parameter=\"q\",
            method=\"GET\",
            parameter_location=\"query\"
        ),
        strategy=\"normal\"
    )
    
    # 建立 HTTP 客戶端
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 建立檢測引擎
        engine = ErrorDetectionEngine()
        
        # 執行檢測
        results = await engine.detect(task, client)
        print(f\"發現漏洞數: {len(results)}\")
        for result in results:
            print(f\"Payload: {result.payload}\")
            print(f\"Evidence: {result.evidence}\")

asyncio.run(test_sqli())
''')"

# 執行 Boolean-Based 檢測
python -c "import asyncio; import httpx; from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget; from services.features.function_sqli.engines.boolean_detection_engine import BooleanDetectionEngine; exec('''
async def test_boolean():
    task = FunctionTaskPayload(
        task_id=\"task_sqli_boolean_001\",
        scan_id=\"scan_001\",
        target=FunctionTaskTarget(
            url=\"http://localhost:3000/rest/user/login\",
            parameter=\"email\",
            method=\"POST\",
            parameter_location=\"body\",
            form_data={\"password\": \"test123\"}
        )
    )
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        engine = BooleanDetectionEngine()
        results = await engine.detect(task, client)
        print(f\"Boolean 檢測結果: {len(results)} 個漏洞\")

asyncio.run(test_boolean())
''')"

# 執行 Time-Based 檢測
python -c "import asyncio; import httpx; from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget; from services.features.function_sqli.engines.time_detection_engine import TimeDetectionEngine; exec('''
async def test_time():
    task = FunctionTaskPayload(
        task_id=\"task_sqli_time_001\",
        scan_id=\"scan_001\",
        target=FunctionTaskTarget(
            url=\"http://localhost:3000/api/Users\",
            parameter=\"id\",
            method=\"GET\",
            parameter_location=\"query\"
        )
    )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        engine = TimeDetectionEngine()
        results = await engine.detect(task, client)
        print(f\"Time-based 檢測結果: {len(results)} 個漏洞\")

asyncio.run(test_time())
''')"
```

**指令參數說明**:
- `task_id`: 任務唯一識別碼，必須以 `task_` 開頭
- `scan_id`: 掃描會話 ID，用於關聯多個任務
- `target.url`: 目標 URL，完整的 HTTP/HTTPS 位址
- `target.parameter`: 要測試的參數名稱（如 `id`, `email`, `search`）
- `target.method`: HTTP 方法 (`GET`, `POST`, `PUT`, `DELETE`)
- `target.parameter_location`: 參數位置 (`query`, `body`, `header`, `cookie`)
- `target.form_data`: POST 請求的其他表單字段（可選）
- `strategy`: 檢測策略 (`normal`, `aggressive`, `stealth`)

**五種檢測引擎說明**:
1. **ErrorDetectionEngine**: 錯誤檢測，注入 SQL 錯誤 payload 觀察錯誤訊息
   - Payloads: `'`, `"`, `' OR '1'='1'--`, `admin'--` 等
   - 適用於有詳細錯誤訊息的應用

2. **BooleanDetectionEngine**: 布林檢測，通過真/假條件判斷注入點
   - Payloads: `' AND '1'='1`, `' AND '1'='2` 比對響應差異
   - 適用於盲注場景

3. **TimeDetectionEngine**: 時間檢測，通過延遲響應判斷注入
   - Payloads: `'; WAITFOR DELAY '00:00:05'--`, `' AND SLEEP(5)--`
   - 適用於完全盲注場景

4. **UnionDetectionEngine**: 聯合查詢檢測，提取數據庫資料
   - Payloads: `' UNION SELECT NULL--`, `' UNION SELECT 1,2,3--`
   - 適用於數據提取

5. **OOBDetectionEngine**: 帶外檢測，使用 DNS/HTTP 回調驗證
   - Payloads: DNS exfiltration, HTTP callbacks
   - 適用於完全盲注和網路隔離環境

**參數變化範例**:
```python
# GET 參數測試
target=FunctionTaskTarget(
    url="http://example.com/product",
    parameter="id",
    method="GET",
    parameter_location="query"
)

# POST Body JSON 測試
target=FunctionTaskTarget(
    url="http://example.com/api/login",
    parameter="email",
    method="POST",
    parameter_location="body",
    json_data={"password": "test"}
)

# Cookie 測試
target=FunctionTaskTarget(
    url="http://example.com/profile",
    parameter="user_id",
    method="GET",
    parameter_location="cookie",
    cookies={"session": "abc123"}
)
```

### 方式二：透過 Command Handler（推薦生產用法）

**適用場景**: AI 指揮架構、統一命令介面、符合 aiva_common 規範

```python
# 🎯 正確的命令處理器使用方式 (2025-12-03 修正)
import asyncio
from services.features.function_sqli.command_handler import SQLiCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

async def test_sqli_command_handler():
    """標準 SQLi 命令處理器執行示例"""
    
    # 1. 創建命令處理器
    handler = SQLiCommandHandler()
    
    # 2. 構建 AI 命令 (必須包含所有必填參數)
    command = AICommand(
        command_id="sqli_test_001",
        command_type=CommandType.FEATURE_SQLI_TEST,
        target_module="features.sqli",
        payload={
            "target_url": "http://localhost:3000/rest/products/search",
            "method": "GET",
            "parameters": {"q": "apple"},
            "test_engines": ["error_detection", "time_based"],  # 可選引擎
            "headers": {"User-Agent": "AIVA-Scanner"},
            "cookies": {},
            "data": None
        },
        timeout=120,  # 2分鐘超時
        
        # 必填追蹤參數 (符合 aiva_common 規範)
        trace_id="trace_sqli_001",
        session_id="session_001", 
        parent_command_id=None,
        callback_url=None
    )
    
    # 3. 執行命令
    result = await handler.handle_command(command)
    
    # 4. 解析結果
    print(f"✅ 執行狀態: {result.status}")
    print(f"✅ 執行成功: {result.success}")
    print(f"⏱️ 執行時間: {result.execution_time:.2f}秒")
    
    if result.success:
        vulnerability_found = result.result.get("vulnerability_found", False)
        vulnerabilities_count = result.result.get("vulnerabilities_count", 0)
        print(f"🎯 發現漏洞: {vulnerability_found}")
        print(f"🔢 漏洞數量: {vulnerabilities_count}")
        
        if vulnerability_found:
            print("⚠️ 檢測到 SQL 注入漏洞!")
        else:
            print("✅ 未發現 SQL 注入漏洞 (表示目標已適當防護)")
    else:
        print(f"❌ 執行失敗: {result.error}")
        print(f"🔍 錯誤代碼: {result.error_code}")
        
    return result

# 執行測試
asyncio.run(test_sqli_command_handler())
```

**修正重點 (2025-12-03)**:
- ✅ AICommand 必須包含 `trace_id`, `session_id` 等追蹤參數
- ✅ AICommandResult 使用 `started_at`, `completed_at` 而非舊的 `timestamp`
- ✅ 錯誤處理包含 `error_code` 和 `error_details`
- ✅ 符合 aiva_common v2.0 命令系統規範

### 方式三：透過 Message Queue（分散式架構）

**適用場景**: 分散式架構、非同步任務、生產環境

```python
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common import get_command_center

# 建立命令中心連線
command_center = get_command_center()

# SQL 注入檢測命令
command = AICommand(
    command_id="sqli_test_001",
    command_type=CommandType.FEATURE_SQLI_TEST,
    target_module="features.sqli",
    payload={
        "target_url": "https://vulnerable-site.com/search.php",
        "parameters": {
            "id": "1",
            "search": "test"
        },
        "engines": ["boolean", "time", "union"],  # 指定使用的檢測引擎
        "timeout": 30,
        "db_fingerprint": "mysql"  # 可選：指定資料庫類型優化
    }
)

# 執行檢測
result = await command_center.execute(command)
```

### 何時使用？
- ✅ **適用場景**:
  - **Web 表單檢測**: 登入表單、搜索框、用戶輸入
  - **API 端點測試**: REST API 的參數注入檢測
  - **GET/POST 參數**: URL 參數和表單數據檢測
  - **Cookie 和 Header**: 非標準注入點檢測
  
- ⚠️ **使用注意**:
  - 僅在授權的滲透測試環境使用
  - 避免對生產環境造成資料損害
  - 建議在測試前備份重要資料

### 如何使用？
```python
# 1. 基本檢測
basic_payload = {
    "target_url": "http://testsite.com/user.php?id=1",
    "parameters": {"id": "1"}
}

# 2. 深度檢測（使用所有引擎）
comprehensive_payload = {
    "target_url": "http://testsite.com/search",
    "parameters": {"q": "search_term", "category": "all"},
    "engines": ["boolean", "time", "union", "error", "oob"],
    "timeout": 60,
    "retries": 3
}

# 3. 針對特定資料庫的優化檢測
mysql_optimized = {
    "target_url": "http://mysql-app.com/api/users",
    "parameters": {"id": "1", "role": "user"},
    "db_fingerprint": "mysql",
    "engines": ["union", "error"]  # MySQL 適用的引擎
}
```

## 🔧 核心能力
- **五引擎並行**: Boolean/Time/Union/Error/OOB 同步檢測
- **智能指紋**: 自動識別後端資料庫類型（MySQL/PostgreSQL/Oracle/MSSQL）
- **Payload 優化**: 根據資料庫類型選擇最佳 payload
- **WAF 繞過**: 內建編碼和混淆技術
- **誤報過濾**: 多重驗證機制確保檢測準確性

## 🎯 後續發展方向
- [ ] **NoSQL 注入** - MongoDB, CouchDB, Redis 注入檢測
- [ ] **機器學習增強** - 基於歷史數據的智能引擎選擇
- [ ] **高級 WAF 繞過** - 針對 Cloudflare, AWS WAF 的繞過技術
- [ ] **自動化利用** - 檢測到漏洞後自動進行數據提取驗證