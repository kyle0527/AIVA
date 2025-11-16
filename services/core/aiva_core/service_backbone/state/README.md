# State - 狀態管理

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [session_state_manager.py](#session_state_managerpy-95-行)
- [🔄 會話生命週期](#-會話生命週期)
- [💾 存儲後端](#-存儲後端)
- [🔒 安全考量](#-安全考量)
- [📚 相關模組](#-相關模組)
- [🔧 配置示例](#-配置示例)

---

## 📋 概述

**定位**: 會話和狀態管理  
**狀態**: ✅ 已實現  
**文件數**: 1 個 Python 文件 (95 行)

## 📂 文件結構

```
state/
├── session_state_manager.py (95 行) - 會話狀態管理器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### session_state_manager.py (95 行)

**職責**: 管理用戶會話和對話狀態

**主要類/函數**:
- `SessionStateManager` - 會話管理器
- `create_session(user_id)` - 創建會話
- `get_session(session_id)` - 獲取會話
- `update_session(session_id, data)` - 更新會話
- `delete_session(session_id)` - 刪除會話

**管理的狀態**:
- 用戶會話上下文
- 對話歷史記錄
- 任務執行狀態
- 臨時數據快取

**使用範例**:
```python
from aiva_core.service_backbone.state import SessionStateManager

state_mgr = SessionStateManager()

# 創建新會話
session = state_mgr.create_session(user_id="alice")
print(f"Session ID: {session.id}")

# 保存對話上下文
state_mgr.update_session(session.id, {
    "conversation_history": [
        {"role": "user", "content": "掃描目標網站"},
        {"role": "assistant", "content": "開始掃描..."}
    ],
    "current_task": "sql_injection_scan",
    "target": "https://example.com"
})

# 獲取會話狀態
session_data = state_mgr.get_session(session.id)
print(f"當前任務: {session_data['current_task']}")
```

## 🔄 會話生命週期

```
創建會話
  ↓
初始化狀態
  ↓
用戶互動 (多次)
  ├─ 更新上下文
  ├─ 記錄對話
  └─ 保存進度
  ↓
會話過期/結束
  ↓
清理和歸檔
```

## 💾 狀態存儲

### 存儲選項

| 後端 | 特點 | 適用場景 |
|------|------|---------|
| **內存** | 快速,但不持久 | 開發測試 |
| **Redis** | 快速,持久,支持過期 | 生產環境首選 |
| **數據庫** | 持久,可查詢 | 需要歷史追溯 |

**配置存儲後端**:
```python
# Redis 後端
state_mgr = SessionStateManager(
    backend="redis",
    redis_url="redis://localhost:6379",
    ttl=3600  # 1 小時過期
)

# 數據庫後端
state_mgr = SessionStateManager(
    backend="database",
    db_url="postgresql://localhost/aiva"
)
```

## 🎯 使用場景

### 1. 多輪對話管理

```python
# 維護對話上下文
session = state_mgr.get_session(session_id)
history = session.get("conversation_history", [])

# 添加新對話
history.append({
    "role": "user",
    "content": "繼續上次的掃描",
    "timestamp": "2025-11-16T10:00:00Z"
})

state_mgr.update_session(session_id, {
    "conversation_history": history
})
```

### 2. 任務狀態追蹤

```python
# 更新任務進度
state_mgr.update_session(session_id, {
    "task_status": "running",
    "progress": 45,
    "current_step": "SQL 注入測試"
})

# 任務完成時
state_mgr.update_session(session_id, {
    "task_status": "completed",
    "progress": 100,
    "results": scan_results
})
```

### 3. 用戶偏好設置

```python
# 保存用戶設置
state_mgr.update_session(session_id, {
    "preferences": {
        "language": "zh-TW",
        "scan_depth": "deep",
        "notifications": True
    }
})
```

## 🔒 安全考慮

### 數據隔離

```python
# 確保用戶只能訪問自己的會話
def get_user_session(user_id, session_id):
    session = state_mgr.get_session(session_id)
    
    # 驗證所有權
    if session.get("user_id") != user_id:
        raise PermissionError("無權訪問此會話")
    
    return session
```

### 敏感數據處理

```python
# 不要在會話中存儲敏感信息
# ❌ 錯誤做法
state_mgr.update_session(session_id, {
    "password": "plain_text_password"  # 不要這樣做!
})

# ✅ 正確做法
state_mgr.update_session(session_id, {
    "auth_token_id": "token_ref_123"  # 存儲引用而非實際值
})
```

## 📊 會話統計

```python
# 獲取活躍會話數
active_sessions = state_mgr.get_active_session_count()

# 獲取用戶的所有會話
user_sessions = state_mgr.get_user_sessions(user_id="alice")

# 清理過期會話
expired_count = state_mgr.cleanup_expired_sessions()
```

## 🔧 配置示例

### 生產環境

```python
state_mgr = SessionStateManager(
    backend="redis",
    redis_url="redis://prod-redis:6379",
    ttl=7200,  # 2 小時
    enable_persistence=True,
    max_sessions_per_user=5
)
```

### 開發環境

```python
state_mgr = SessionStateManager(
    backend="memory",
    ttl=3600,  # 1 小時
    debug=True
)
```

## 📚 相關模組

- [api](../api/README.md) - API 會話管理
- [coordination](../coordination/README.md) - 狀態協調
- [storage](../storage/README.md) - 數據持久化

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md) 的修復規範。

```python
# ✅ 正確：使用標準狀態類型
from aiva_common import TaskStatus, ModuleName, UnifiedConfig

# 保存會話狀態
state_mgr.update_session(session_id, {
    "task_status": TaskStatus.RUNNING,
    "module": ModuleName.STATE_MANAGER,
    "config": UnifiedConfig.get_instance()
})

# ❌ 禁止：自定義會話狀態
class SessionStatus(str, Enum):
    ACTIVE = "active"  # 使用 TaskStatus 替代
    IDLE = "idle"

# ❌ 禁止：自定義配置類
class StateConfig:
    def __init__(self):
        self.backend = "redis"  # 使用 UnifiedConfig
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
