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

### 實際使用方式
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