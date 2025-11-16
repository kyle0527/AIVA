# AuthZ - 授權控制子系統

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [permission_matrix.py](#permission_matrixpy-610-行-)
  - [matrix_visualizer.py](#matrix_visualizerpy-541-行-)
  - [authz_mapper.py](#authz_mapperpy-356-行)
- [👥 默認角色](#-默認角色)
- [🔐 權限矩陣示例](#-權限矩陣示例)
- [🎨 可視化功能](#-可視化功能)
- [📚 相關模組](#-相關模組)
- [🔧 配置最佳實踐](#-配置最佳實踐)

---

## 📋 概述

**定位**: 基於角色的訪問控制 (RBAC) 系統  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (1,507 行)

## 📂 文件結構

```
authz/
├── permission_matrix.py (610 行) ⭐ - 權限矩陣
├── matrix_visualizer.py (541 行) ⭐ - 矩陣可視化器
├── authz_mapper.py (356 行) - 授權映射器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### permission_matrix.py (610 行) ⭐

**職責**: 權限矩陣定義和驗證

**主要類/函數**:
- `PermissionMatrix` - 權限矩陣管理器
- `define_role(role, permissions)` - 定義角色權限
- `check_permission(user, action, resource)` - 權限檢查
- `grant_permission(role, permission)` - 授予權限
- `revoke_permission(role, permission)` - 撤銷權限

**權限級別**:
- `ADMIN` - 完全控制
- `OPERATOR` - 操作執行
- `AUDITOR` - 只讀查看
- `GUEST` - 基本訪問

**使用範例**:
```python
from aiva_core.service_backbone.authz import PermissionMatrix

matrix = PermissionMatrix()

# 定義角色
matrix.define_role("security_tester", [
    "scan:read",
    "scan:execute",
    "attack:read",
    "report:create"
])

# 檢查權限
has_perm = matrix.check_permission(
    user="alice",
    action="scan:execute",
    resource="target_system"
)
```

---

### matrix_visualizer.py (541 行) ⭐

**職責**: 權限矩陣可視化和導出

**主要功能**:
- 生成權限矩陣表格 (ASCII/Markdown/HTML)
- 權限關係圖可視化
- 權限衝突檢測
- 導出為 JSON/YAML 配置

**使用範例**:
```python
from aiva_core.service_backbone.authz import MatrixVisualizer

visualizer = MatrixVisualizer(matrix)

# 生成 Markdown 表格
table = visualizer.to_markdown()
print(table)

# 檢測權限衝突
conflicts = visualizer.detect_conflicts()
```

**輸出示例**:
```
| Role            | scan:read | scan:execute | attack:read | attack:execute |
|-----------------|-----------|--------------|-------------|----------------|
| admin           | ✅        | ✅           | ✅          | ✅             |
| security_tester | ✅        | ✅           | ✅          | ❌             |
| auditor         | ✅        | ❌           | ✅          | ❌             |
```

---

### authz_mapper.py (356 行)

**職責**: 授權映射和轉換

**主要功能**:
- 用戶到角色映射
- 角色到權限映射
- LDAP/OAuth 用戶映射
- 臨時權限提升

**使用範例**:
```python
from aiva_core.service_backbone.authz import AuthzMapper

mapper = AuthzMapper()
mapper.map_user_to_role("alice", "security_tester")

# 獲取用戶所有權限
permissions = mapper.get_user_permissions("alice")
```

## 🔒 權限模型

### 資源層級

```
系統
 ├── 掃描 (scan)
 │   ├── scan:read
 │   ├── scan:execute
 │   └── scan:manage
 ├── 攻擊 (attack)
 │   ├── attack:read
 │   ├── attack:execute
 │   └── attack:manage
 └── 報告 (report)
     ├── report:read
     ├── report:create
     └── report:delete
```

### 角色預設權限

| 角色 | 描述 | 典型權限 |
|------|------|---------|
| **admin** | 系統管理員 | 所有權限 |
| **security_tester** | 安全測試人員 | scan:*, attack:read, report:create |
| **operator** | 操作人員 | scan:*, report:read |
| **auditor** | 審計人員 | *:read |
| **guest** | 訪客 | report:read (有限) |

## 📚 相關模組

- [api](../api/README.md) - API 權限控制
- [coordination](../coordination/README.md) - 服務授權協調

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

```python
# ✅ 正確：使用標準枚舉
from aiva_common import ModuleName

# ✅ 合理的權限專屬枚舉
class Permission(str, Enum):
    READ = "read"
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

## 🔐 最佳實踐

1. **最小權限原則**: 僅授予完成任務所需的最小權限
2. **權限審計**: 定期審查權限分配
3. **臨時提權**: 使用臨時提權而非永久授權
4. **權限分離**: 關鍵操作需要多個角色協作

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
