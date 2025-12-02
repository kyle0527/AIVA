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

### 實際使用方式
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