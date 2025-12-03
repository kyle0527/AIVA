# 🔍 IDOR 檢測模組

**什麼是 IDOR？**  
Insecure Direct Object References（不安全的直接物件引用）是一種授權缺陷，允許攻擊者通過修改物件標識符來訪問未經授權的資源。本模組支援水平權限提升（訪問同級用戶資源）和垂直權限提升（訪問高權限資源）的自動化檢測。

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   智能 IDOR 檢測架構                          │
├─────────────────────────────────────────────────────────────┤
│ AI Command      │command_handler │ SmartIDORDetector│ 測試器  │
│ Interface       │               │                  │ 集群    │
│       ↓         │       ↓       │        ↓         │    ↓    │
│ FEATURE_IDOR_   │ FunctionTask  │ ResourceId       │ Cross   │
│ TEST            │ Payload       │ Extractor        │ User    │
│       │         │               │        ↓         │ Tester  │
│       └─────────┼───────────────┼─ IDPattern       │    ↓    │
│                 │               │  分析器          │ Vertical│
│                 ↓               │        ↓         │ Escalate│
│         IDORDetectionResult     │ 水平+垂直        │ Tester  │
│         (cross_user/vertical)   │ 權限測試         │         │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **資源ID提取** - 自動識別URL和參數中的物件標識符
2. **模式分析** - 分析ID模式（數字、UUID、雜湊、混合）
3. **權限測試** - 執行雙向權限檢測：
   - **水平測試**: 嘗試訪問其他用戶的同等資源
   - **垂直測試**: 嘗試訪問更高權限的資源
   - **批量驗證**: 生成多個測試ID進行大規模掃描
4. **結果分析** - 比較響應差異確認未授權訪問

## 🚀 支援指令

### 方式一：直接執行檢測引擎（開發/測試用）

**適用場景**: 快速驗證、單元測試、本地開發

```bash
# 進入專案目錄
cd c:\D\fold7\AIVA-git

# 執行 IDOR 檢測測試
python -c "import asyncio; import httpx; from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget; from services.features.function_idor.engine.idor_engine import IDOREngine; exec('''
async def test_idor():
    # 建立任務結構
    task = FunctionTaskPayload(
        task_id=\"task_idor_001\",
        scan_id=\"scan_001\",
        target=FunctionTaskTarget(
            url=\"http://localhost:3000/api/Users/1\",
            parameter=\"id\",
            method=\"GET\",
            parameter_location=\"query\",
            headers={\"Authorization\": \"Bearer test_token\"},
            cookies={\"session\": \"user_session_123\"}
        ),
        strategy=\"normal\"
    )
    
    # 建立 HTTP 客戶端
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 建立 IDOR 引擎
        engine = IDOREngine()
        
        # 執行檢測
        results = await engine.detect(task, client)
        print(f\"\\n發現 IDOR 漏洞數: {len(results)}\")
        
        for result in results:
            print(f\"\\n漏洞詳情:\")
            print(f\"  資源 ID: {result.resource_id}\")
            print(f\"  測試 ID: {result.tested_id}\")
            print(f\"  權限類型: {result.escalation_type}\")
            print(f\"  響應狀態: {result.response_status}\")

asyncio.run(test_idor())
''')"
```

### 方式二：透過 Command Handler（推薦生產用法） 🔧

**適用場景**: AI 指揮架構、統一命令介面、符合 aiva_common 規範

```python
# 🎯 正確的 IDOR 命令處理器使用方式 (2025-12-03 修正)
import asyncio
from services.features.function_idor.command_handler import IDORCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

async def test_idor_command_handler():
    \"\"\"標準 IDOR 命令處理器執行示例\"\"\"
    
    # 1. 創建命令處理器
    handler = IDORCommandHandler()
    
    # 2. 構建 AI 命令 (必須包含所有必填參數)
    command = AICommand(
        command_id=\"idor_test_001\",
        command_type=CommandType.FEATURE_IDOR_TEST,
        target_module=\"features.idor\",
        payload={
            \"target_url\": \"http://localhost:3000/api/users/123\",
            \"resource_patterns\": [\"user_id\", \"post_id\", \"document_id\"],
            \"test_methods\": [\"sequential\"],  # sequential, random, privilege_escalation
            \"credentials\": {\"user1\": \"token123\"}  # 測試憑證
        },
        timeout=60,  # 1分鐘超時
        
        # 必填追蹤參數 (符合 aiva_common 規範)
        trace_id=\"trace_idor_001\",
        session_id=\"session_001\", 
        parent_command_id=None,
        callback_url=None
    )
    
    # 3. 執行命令
    result = await handler.handle_command(command)
    
    # 4. 解析結果
    print(f\"✅ 執行狀態: {result.status}\")
    print(f\"✅ 執行成功: {result.success}\")
    print(f\"⏱️ 執行時間: {result.execution_time:.2f}秒\")
    
    if result.success:
        vulnerability_found = result.result.get(\"vulnerability_found\", False)
        idor_results = result.result.get(\"idor_results\", [])
        extracted_resources = result.result.get(\"extracted_resources\", [])
        
        print(f\"🎯 發現漏洞: {vulnerability_found}\")
        print(f\"🔢 IDOR 結果: {len(idor_results)} 個\")
        print(f\"📊 提取的資源: {extracted_resources}\")
        
        for result_item in idor_results:
            print(f\"  - 漏洞: {result_item.get('vulnerable')}\")
            print(f\"  - 證據: {result_item.get('evidence')}\")
            print(f\"  - 風險等級: {result_item.get('risk_level')}\")
    else:
        print(f\"❌ 執行失敗: {result.error}\")
        print(f\"🔍 錯誤代碼: {result.error_code}\")
        
    return result

# 執行測試
asyncio.run(test_idor_command_handler())
```

**修正重點 (2025-12-03)**:
- ✅ AICommand 必須包含 `trace_id`, `session_id` 等追蹤參數
- ✅ AICommandResult 使用 `started_at`, `completed_at` 而非舊的 `timestamp`
- ✅ 錯誤處理包含 `error_code` 和 `error_details`
- ✅ 資源提取器方法: `extract_from_url()` 而非 `extract_resource_ids()`
- ✅ 檢測器方法: `detect_vulnerabilities()` 統一介面
- ✅ 正確的異步客戶端管理和任務結構創建

asyncio.run(test_idor())
''')"
```

**指令參數說明**:
- `task_id`: 任務唯一識別碼，必須以 `task_` 開頭
- `scan_id`: 掃描會話 ID，用於關聯多個任務
- `target.url`: 目標 URL，包含資源 ID 的完整位址
- `target.parameter`: 要測試的參數名稱（如 `id`, `user_id`, `order_id`）
- `target.method`: HTTP 方法 (`GET`, `POST`, `PUT`, `DELETE`)
- `target.parameter_location`: 參數位置 (`query`, `path`, `body`)
- `target.headers`: 認證 headers（如 Authorization token）
- `target.cookies`: 認證 cookies（如 session ID）
- `strategy`: 檢測策略 (`normal`, `aggressive`, `stealth`)

**IDOR 檢測類型說明**:
1. **水平權限測試 (Horizontal Escalation)**:
   - 測試是否可訪問同級用戶的資源
   - 例如: 用戶A訪問用戶B的個人資料
   - 測試方法: 修改資源ID觀察響應

2. **垂直權限測試 (Vertical Escalation)**:
   - 測試是否可訪問更高權限的資源
   - 例如: 普通用戶訪問管理員功能
   - 測試方法: 嘗試訪問高權限資源ID

**參數變化範例**:
```python
# GET 請求 - 路徑參數測試
target=FunctionTaskTarget(
    url="http://example.com/api/users/123/profile",
    parameter="id",
    method="GET",
    parameter_location="path",
    headers={"Authorization": "Bearer user_token"}
)

# GET 請求 - Query 參數測試
target=FunctionTaskTarget(
    url="http://example.com/api/orders",
    parameter="order_id",
    method="GET",
    parameter_location="query",
    cookies={"session_id": "abc123"}
)

# PUT 請求 - 修改其他用戶資源
target=FunctionTaskTarget(
    url="http://example.com/api/users/456",
    parameter="user_id",
    method="PUT",
    parameter_location="path",
    json_data={"name": "Modified"},
    headers={"Authorization": "Bearer user_token"}
)

# DELETE 請求 - 刪除其他用戶資源
target=FunctionTaskTarget(
    url="http://example.com/api/documents/789",
    parameter="doc_id",
    method="DELETE",
    parameter_location="path",
    headers={"Authorization": "Bearer user_token"}
)
```

**ID 模式支援**:
- 數字型: `1`, `123`, `456789`
- UUID: `550e8400-e29b-41d4-a716-446655440000`
- 雜湊型: `a1b2c3d4e5f6`, MD5/SHA1 雜湊
- 混合型: `user_123`, `order-456`

### 方式二：透過 Message Queue（生產環境用）

**適用場景**: 分散式架構、非同步任務、生產環境

```python
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common import get_command_center

# 建立命令中心連線
command_center = get_command_center()

# IDOR 檢測命令
command = AICommand(
    command_id="idor_test_001",
    command_type=CommandType.FEATURE_IDOR_TEST,
    target_module="features.idor",
    payload={
        "target_url": "https://api.app.com/users/123/profile",
        "authenticated_session": {
            "cookies": {"session": "abc123def456"},
            "headers": {"Authorization": "Bearer your_token"}
        },
        "test_range": 100,  # 測試範圍：當前ID±100
        "id_patterns": ["numeric", "uuid", "hash"],
        "test_types": ["horizontal", "vertical"],
        "response_comparison": True
    }
)

# 執行檢測
result = await command_center.execute(command)
```

### 何時使用？
- ✅ **適用場景**:
  - **用戶資源API**: 個人資料、訂單、文件訪問
  - **管理介面**: 後台管理功能、配置頁面
  - **文件系統**: 檔案下載、圖片預覽
  - **資料庫記錄**: 任何基於ID的資源訪問
  
- ⚠️ **使用注意**:
  - 需要有效的用戶會話進行測試
  - 避免對敏感生產數據進行大範圍掃描
  - 注意可能觸發的安全監控警報
  - 測試後確認未留下異常訪問記錄

### 如何使用？
```python
# 1. 基本數字ID檢測
numeric_idor = {
    "target_url": "https://app.com/api/users/123",
    "authenticated_session": {
        "cookies": {"sessionid": "user123session"}
    },
    "id_patterns": ["numeric"],
    "test_range": 50,  # 測試 ID 73-173
    "test_types": ["horizontal"]
}

# 2. UUID資源檢測
uuid_idor = {
    "target_url": "https://app.com/orders/a1b2c3d4-e5f6-7890-abcd-1234567890ab",
    "authenticated_session": {
        "headers": {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."}
    },
    "id_patterns": ["uuid"],
    "uuid_generation": "random",  # random|sequential|pattern
    "test_count": 20
}

# 3. 垂直權限提升
vertical_escalation = {
    "target_url": "https://app.com/admin/config/456",
    "authenticated_session": {
        "cookies": {"auth": "normal_user_session"}
    },
    "test_types": ["vertical"],
    "privilege_levels": ["admin", "moderator", "premium"],
    "admin_endpoints": [
        "https://app.com/admin/users",
        "https://app.com/admin/settings"
    ]
}

# 4. 混合模式檢測
mixed_pattern = {
    "target_url": "https://app.com/documents/DOC123ABC",
    "authenticated_session": {
        "cookies": {"session": "authenticated_user"}
    },
    "id_patterns": ["mixed", "hash"],
    "pattern_analysis": True,
    "custom_patterns": [
        "DOC{num}ABC",      # DOC456ABC
        "FILE_{hash}",      # FILE_md5hash
        "{prefix}{num}"     # 通用模式
    ]
}

# 5. 大規模掃描（批量檢測）
bulk_scan = {
    "target_url": "https://api.app.com/invoices/{id}",
    "authenticated_session": {
        "headers": {
            "Authorization": "Bearer token123",
            "X-API-Key": "api_key_456"
        }
    },
    "id_patterns": ["numeric", "uuid"],
    "test_range": 1000,  # 大範圍掃描
    "batch_size": 10,    # 批次大小控制請求頻率
    "response_filters": {
        "success_codes": [200, 201],
        "error_codes": [403, 404],
        "content_patterns": ["invoice_id", "amount"]
    },
    "smart_stopping": True  # 智能停止（檢測到模式後優化）
}
```

## 🔧 核心能力
- **智能ID提取**: 自動識別URL路徑和參數中的物件標識符
- **模式識別**: 支援數字、UUID、雜湊、混合等多種ID模式
- **雙向檢測**: 水平（同級用戶）和垂直（權限提升）檢測
- **響應分析**: 智能比較響應內容差異判斷訪問成功
- **批量掃描**: 高效的大規模ID遍歷和測試
- **會話管理**: 支援Cookie、Header、Token等多種認證方式

## 🎯 後續發展方向
- [ ] **GraphQL IDOR** - 現代API的物件引用檢測
- [ ] **機器學習ID預測** - 基於歷史數據的智能ID生成
- [ ] **時間戳ID** - 基於時間戳的資源ID模式攻擊
- [ ] **複合鍵檢測** - 多參數組合的複雜物件引用檢測