# State 狀態管理模組

> **路徑**: `services/core/aiva_core/service_backbone/state`  
> **狀態**: ✅ 正常 | **文件數**: 2 | **最後更新**: 2026-01-07

## 概述

會話狀態管理模組，負責管理用戶會話、執行上下文和臨時狀態數據。

## 核心組件

### session_state_manager.py
- `SessionStateManager` - 會話狀態管理器
  - 會話生命週期管理
  - 狀態持久化和恢復
  - 會話超時處理
  - 狀態查詢和更新

### __init__.py
- 模組初始化和導出

## 會話狀態類型

| 狀態類型 | 描述 |
|----------|------|
| 執行上下文 | 當前任務的執行環境 |
| 掃描會話 | 活動中的掃描任務狀態 |
| 用戶偏好 | 用戶設定和配置 |
| 臨時數據 | 中間計算結果 |

## 使用方式

```python
from service_backbone.state import SessionStateManager

manager = SessionStateManager()

# 創建會話
session_id = manager.create_session(user_id="user_123")

# 更新狀態
manager.update_state(session_id, {"scan_progress": 50})

# 獲取狀態
state = manager.get_state(session_id)

# 結束會話
manager.close_session(session_id)
```

## 整合點

- 被 `ScanResultProcessor` 使用於七階段處理流程
- 被 `TaskExecutor` 使用於任務執行追蹤
- 被 `CoreServiceCoordinator` 使用於服務協調

## 依賴關係

- 無外部套件依賴（純 Python 實現）
- 可選配置持久化後端
