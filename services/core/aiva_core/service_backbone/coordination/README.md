# Coordination - 服務協調中樞

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [ai_controller.py](#ai_controllerpy-816-行-)
  - [core_service_coordinator.py](#core_service_coordinatorpy-518-行-)
  - [optimized_core.py](#optimized_corepy-273-行)
- [🔄 協調流程](#-協調流程)
- [📊 執行模式](#-執行模式)
- [⚡ 性能優化](#-性能優化)
- [📚 相關模組](#-相關模組)

---

## 📋 概述

**定位**: 跨服務協調和編排核心  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (1,607 行)

## 📂 文件結構

```
coordination/
├── ai_controller.py (816 行) ⭐⭐ - AI 控制器
├── core_service_coordinator.py (518 行) ⭐ - 核心服務協調器
├── optimized_core.py (273 行) - 優化核心
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### ai_controller.py (816 行) ⭐⭐

**職責**: AI 系統的中央控制器

**主要類/函數**:
- `AIController` - AI 控制器主類
- `process_request(request)` - 處理請求並協調各模組
- `coordinate_tasks()` - 任務協調
- `manage_resources()` - 資源管理

**關鍵職責**:
1. **請求路由**: 將用戶請求分發到正確的能力模組
2. **任務編排**: 協調多個任務的執行順序
3. **資源分配**: 管理 AI 模型和計算資源
4. **狀態管理**: 維護系統運行狀態

**使用範例**:
```python
from aiva_core.service_backbone.coordination import AIController

controller = AIController()

# 處理用戶請求
result = await controller.process_request({
    "type": "security_scan",
    "target": "https://example.com",
    "depth": "full"
})
```

**架構位置**:
```
用戶請求
  ↓
AIController (協調中心)
  ├→ Task Planning (任務規劃)
  ├→ Core Capabilities (能力執行)
  ├→ Cognitive Core (AI 決策)
  └→ External Learning (經驗學習)
```

---

### core_service_coordinator.py (518 行) ⭐

**職責**: 核心服務間的協調器

**主要功能**:
- 服務註冊和發現
- 服務健康檢查
- 負載均衡
- 故障轉移

**協調的服務**:
- `scan_service` - 掃描服務
- `attack_service` - 攻擊執行服務
- `analysis_service` - 分析服務
- `reporting_service` - 報告生成服務

**使用範例**:
```python
from aiva_core.service_backbone.coordination import CoreServiceCoordinator

coordinator = CoreServiceCoordinator()

# 註冊服務
coordinator.register_service("scan_service", scan_instance)

# 協調服務調用
result = await coordinator.coordinate_call(
    service="scan_service",
    method="execute_scan",
    params={"target": "example.com"}
)
```

---

### optimized_core.py (273 行)

**職責**: 性能優化的核心協調邏輯

**優化特性**:
- ✅ 並行任務執行
- ✅ 請求去重
- ✅ 結果快取
- ✅ 智能重試機制

**使用範例**:
```python
from aiva_core.service_backbone.coordination import OptimizedCore

core = OptimizedCore()

# 批量處理請求 (自動並行化)
results = await core.batch_process([
    {"type": "scan", "target": "site1.com"},
    {"type": "scan", "target": "site2.com"},
    {"type": "scan", "target": "site3.com"}
])
```

## 🔄 協調流程

### 典型請求流程

```
1. 用戶請求 → AIController
                ↓
2. 請求解析和驗證
                ↓
3. CoreServiceCoordinator 選擇服務
                ↓
4. OptimizedCore 優化執行
                ↓
5. 並行/串行執行任務
                ↓
6. 結果聚合和返回
```

### 服務依賴圖

```
AIController (頂層)
  ├─ CoreServiceCoordinator (中層)
  │   ├─ ScanService
  │   ├─ AttackService
  │   └─ AnalysisService
  └─ OptimizedCore (優化層)
      ├─ 並行處理器
      ├─ 快取管理器
      └─ 重試管理器
```

## 📊 協調模式

| 模式 | 描述 | 使用場景 |
|------|------|---------|
| **順序執行** | 任務按順序執行 | 有依賴關係的任務 |
| **並行執行** | 任務同時執行 | 獨立任務批量處理 |
| **管道模式** | 前一個輸出作為下一個輸入 | 數據處理流程 |
| **扇出扇入** | 分發多個任務後聚合結果 | 多目標掃描 |

## 📚 相關模組

- [task_planning](../../task_planning/README.md) - 任務規劃
- [messaging](../messaging/README.md) - 消息傳遞
- [api](../api/README.md) - API 接口

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md) 的修復規範。

```python
# ✅ 正確：使用標準類型
from aiva_common import AivaMessage, MessageHeader, ModuleName, TaskStatus

# 創建協調消息
message = AivaMessage(
    header=MessageHeader(
        source=ModuleName.COORDINATION,
        target=ModuleName.SCANNING
    ),
    payload={"task": "scan", "status": TaskStatus.RUNNING}
)

# ❌ 禁止：自定義協調消息類型
class CoordinationMessage:
    def __init__(self, source, target):
        self.source = source  # 不要自定義消息格式
        self.target = target

# ❌ 禁止：自定義狀態枚舉
class CoordinationStatus(str, Enum):
    COORDINATING = "coordinating"  # 使用 TaskStatus 替代
    WAITING = "waiting"
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md)

---

## 🔧 配置示例

```python
# AIController 配置
controller_config = {
    "max_concurrent_tasks": 10,
    "task_timeout": 300,
    "enable_caching": True,
    "retry_policy": {
        "max_attempts": 3,
        "backoff_factor": 2
    }
}

controller = AIController(config=controller_config)
```

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
