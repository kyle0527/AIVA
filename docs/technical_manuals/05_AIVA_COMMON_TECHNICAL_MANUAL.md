# AIVA Common 模組技術手冊

**版本**: v2.0
**狀態**: Production-Grade Single Source of Truth
**路徑**: `services/aiva_common/`

---

## 1. 模組概述

AIVA Common 是整個 AIVA 系統的共用基礎設施層（Single Source of Truth），提供統一的資料結構、設定管理、安全服務、管線處理等核心功能。**所有其他模組都依賴此模組**，不應繞過 aiva_common 直接定義重複的類型或設定。

**規模**：README 84KB，ARCHITECTURE 23.5KB，15+ 子目錄模組

---

## 2. 子系統一覽

```
aiva_common/
├── config/          設定管理（4 範圍）
├── core/            核心命令中心、錯誤處理
├── security/        安全服務
├── pipeline/        資料管線
├── services/        服務發現與健康監控
├── observability/   監控與日誌
├── messaging/       訊息代理抽象
├── schemas/         統一資料模型
├── enums/           統一列舉定義
├── cli/             CLI 工具
├── protocols/       協定定義
├── ai/              AI 相關功能
├── async_utils/     非同步工具
├── cross_language/  多語言支援
├── detection/       偵測功能
└── plugins/         插件系統
```

---

## 3. 核心子系統詳解

### 3.1 config/ — 設定管理

四層範圍設定，優先級由低到高：

```
Global（全域）
  └── User（使用者）
       └── Session（會話）
            └── Temporary（臨時）
```

**ConfigManager** 自動合併各層設定，臨時設定優先於全域。

```python
from aiva_common.config import ConfigManager

config = ConfigManager()
api_key = config.get("openai.api_key")  # 自動查找最高優先級設定
```

### 3.2 core/ — 錯誤處理

```python
from aiva_common.core import AIVAError, ErrorType, ErrorSeverity

# 統一錯誤類型
class ErrorType(Enum):
    NETWORK_ERROR
    AUTH_FAILURE
    SCAN_FAILURE
    DECISION_ERROR
    STORAGE_ERROR
    ...

# 統一嚴重程度
class ErrorSeverity(Enum):
    CRITICAL   # 系統停止
    HIGH       # 功能降級
    MEDIUM     # 可重試
    LOW        # 記錄即可
```

### 3.3 security/ — 安全服務

| 元件 | 功能 |
|---|---|
| `CryptographyService` | 加解密，Key 管理 |
| `TokenService` | JWT 生成與驗證 |
| `CORSHandler` | 跨域請求控制 |

### 3.4 pipeline/ — 資料管線

```python
# 三種時間窗口類型
TumblingWindow   # 固定大小不重疊視窗
SlidingWindow    # 滑動視窗（可重疊）
SessionWindow    # 基於活動的動態視窗

# 使用範例
pipeline = DataPipeline()
stream = StreamProcessor(window=TumblingWindow(size=60))
```

### 3.5 schemas/ — 統一資料模型

三大資料合約，所有模組必須遵守：

```
dual_loop/     雙閉環相關資料結構
analysis/      分析結果資料結構
reporting/     報告輸出資料結構
```

### 3.6 messaging/ — 訊息代理

支援多 Backend 無縫切換：

```python
# 支援的 Backend
- RabbitMQ（生產環境）
- In-memory（測試環境）

from aiva_common.messaging import MessageBroker
broker = MessageBroker.get_instance()  # 自動根據設定選擇 Backend
```

---

## 4. 統一列舉定義（enums/）

**所有模組必須使用這些列舉，禁止自行定義等效類型。**

### 4.1 Severity（嚴重程度）

```python
class Severity(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"
```

### 4.2 Confidence（信心程度）

```python
class Confidence(Enum):
    HIGH   = "high"    # ≥ 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW    = "low"     # < 0.5
```

### 4.3 VulnerabilityStatus（漏洞狀態）

```python
class VulnerabilityStatus(Enum):
    OPEN       = "open"
    PATCHED    = "patched"
    MITIGATED  = "mitigated"
    ACCEPTED   = "accepted"   # 接受風險
```

### 4.4 TaskStatus（任務狀態）

```python
class TaskStatus(Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
```

### 4.5 ModuleName（模組標識）

```python
class ModuleName(Enum):
    CORE        = "core"
    FEATURES    = "features"
    SCAN        = "scan"
    INTEGRATION = "integration"
    COMMON      = "common"
```

---

## 5. 多語言支援（cross_language/）

為 Go、TypeScript 引擎提供跨語言通信支援：

```
cross_language/
├── typescript/   TypeScript 引擎 binding
└── go/          Go 引擎 binding
```

**通信協定**：JSON over stdin/stdout（CLI 模式）

---

## 6. 服務發現（services/）

```python
from aiva_common.services import ServiceRegistry, ServiceDiscovery, HealthMonitor

# 服務注冊
registry = ServiceRegistry()
registry.register("scan_go", host="localhost", port=8001)

# 健康監控
monitor = HealthMonitor()
status = monitor.check("scan_go")
```

---

## 7. 可觀測性（observability/）

```python
from aiva_common.observability import MetricsCollector

metrics = MetricsCollector()
metrics.record("scan.duration", value=1.23, tags={"engine": "rust"})
metrics.increment("vulnerabilities.found", tags={"severity": "high"})
```

---

## 8. 插件系統（plugins/）

允許第三方擴展 AIVA 功能而無需修改核心程式碼：

```python
from aiva_common.plugins import PluginBase

class MyPlugin(PluginBase):
    def on_scan_complete(self, results: dict) -> None:
        # 自定義後處理邏輯
        pass
```

---

## 9. 正確使用原則

| 必須 | 禁止 |
|---|---|
| 使用 `aiva_common.enums.Severity` | 自行定義 severity 字串 |
| 使用 `aiva_common.schemas` 資料合約 | 各模組自定義重複的結構 |
| 使用 `ConfigManager` 讀取設定 | 直接讀取環境變數 |
| 使用 `AIVAError` 拋出異常 | 使用原生 Exception |

---

## 10. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第1冊_系統入門與架構.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第6冊_進階開發.md`（開發者向）
