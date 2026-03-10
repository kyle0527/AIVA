# AIVA Common 模組技術手冊

**版本**: v2.0 | **狀態**: ✅ Production-Grade Single Source of Truth | **路徑**: `services/aiva_common/`

---

## 目錄

1. [模組概述](#1-模組概述)
2. [子系統一覽](#2-子系統一覽)
3. [核心子系統詳解](#3-核心子系統詳解)
   - 3.1 [config/ — 設定管理](#31-config--設定管理)
   - 3.2 [core/ — 錯誤處理](#32-core--錯誤處理)
   - 3.3 [security/ — 安全服務](#33-security--安全服務)
   - 3.4 [pipeline/ — 資料管線](#34-pipeline--資料管線)
   - 3.5 [messaging/ — 訊息代理](#35-messaging--訊息代理)
   - 3.6 [observability/ — 可觀測性](#36-observability--可觀測性)
   - 3.7 [services/ — 服務發現](#37-services--服務發現)
   - 3.8 [plugins/ — 插件系統](#38-plugins--插件系統)
4. [統一列舉定義（enums/）](#4-統一列舉定義enums)
5. [統一資料模型（schemas/）](#5-統一資料模型schemas)
6. [多語言支援（cross_language/）](#6-多語言支援cross_language)
7. [正確使用原則](#7-正確使用原則)
8. [完成狀態](#8-完成狀態)
   - 8.1 [已完成功能](#81-已完成功能-)
   - 8.2 [待完成 / 目標功能](#82-待完成--目標功能-)
9. [搭配閱讀](#9-搭配閱讀)

---

## 1. 模組概述

AIVA Common 是整個 AIVA 系統的共用基礎設施層（Single Source of Truth），提供統一的資料結構、設定管理、安全服務、管線處理等核心功能。**所有其他模組都依賴此模組**，不應繞過 aiva_common 直接定義重複的類型或設定。

**規模**：README 84KB，ARCHITECTURE 23.5KB，15+ 子目錄模組

**技術優勢**：Pydantic v2 型別安全，減少 80% 型別錯誤；512 維向量規格與 5M 決策引擎一致。

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
├── schemas/         統一資料模型（⭐ 最重要）
├── enums/           統一列舉定義（⭐ 最重要）
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

class ErrorType(Enum):
    NETWORK_ERROR
    AUTH_FAILURE
    SCAN_FAILURE
    DECISION_ERROR
    STORAGE_ERROR

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

pipeline = DataPipeline()
stream = StreamProcessor(window=TumblingWindow(size=60))
```

### 3.5 messaging/ — 訊息代理

支援多 Backend 無縫切換：

```python
# 支援的 Backend
- RabbitMQ（生產環境）
- In-memory（測試環境）

from aiva_common.messaging import MessageBroker
broker = MessageBroker.get_instance()  # 自動根據設定選擇 Backend
```

### 3.6 observability/ — 可觀測性

```python
from aiva_common.observability import MetricsCollector

metrics = MetricsCollector()
metrics.record("scan.duration", value=1.23, tags={"engine": "rust"})
metrics.increment("vulnerabilities.found", tags={"severity": "high"})
```

### 3.7 services/ — 服務發現

```python
from aiva_common.services import ServiceRegistry, HealthMonitor

registry = ServiceRegistry()
registry.register("scan_go", host="localhost", port=8001)

monitor = HealthMonitor()
status = monitor.check("scan_go")
```

### 3.8 plugins/ — 插件系統

```python
from aiva_common.plugins import PluginBase

class MyPlugin(PluginBase):
    def on_scan_complete(self, results: dict) -> None:
        # 自定義後處理邏輯
        pass
```

---

## 4. 統一列舉定義（enums/）

**所有模組必須使用這些列舉，禁止自行定義等效類型。**

```python
class Severity(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"

class Confidence(Enum):
    HIGH   = "high"    # ≥ 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW    = "low"     # < 0.5

class VulnerabilityStatus(Enum):
    OPEN       = "open"
    PATCHED    = "patched"
    MITIGATED  = "mitigated"
    ACCEPTED   = "accepted"   # 接受風險

class TaskStatus(Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"

class ModuleName(Enum):
    CORE        = "core"
    FEATURES    = "features"
    SCAN        = "scan"
    INTEGRATION = "integration"
    COMMON      = "common"
```

---

## 5. 統一資料模型（schemas/）

三大資料合約，所有模組必須遵守：

```
schemas/
├── dual_loop/     雙閉環相關資料結構
├── analysis/      分析結果資料結構
└── reporting/     報告輸出資料結構
```

**Pydantic v2** 驗證，100% 型別安全，確保跨模組資料一致性。

---

## 6. 多語言支援（cross_language/）

為 Go、TypeScript 引擎提供跨語言通信支援：

```
cross_language/
├── typescript/   TypeScript 引擎 binding
└── go/          Go 引擎 binding
```

**通信協定**：JSON over stdin/stdout（CLI 模式）

---

## 7. 正確使用原則

| 必須 | 禁止 |
|---|---|
| 使用 `aiva_common.enums.Severity` | 自行定義 severity 字串 |
| 使用 `aiva_common.schemas` 資料合約 | 各模組自定義重複的結構 |
| 使用 `ConfigManager` 讀取設定 | 直接讀取環境變數 |
| 使用 `AIVAError` 拋出異常 | 使用原生 Exception |
| 使用 `MessageBroker` 跨模組通信 | 直接呼叫其他模組函數 |

---

## 8. 完成狀態

### 8.1 已完成功能 ✅

| 功能 | 說明 |
|---|---|
| 雙軌通信架構 | Async MessageBroker + Sync CLI |
| 資料模型（Schema）| Pydantic v2，100% 型別安全 |
| 設定管理 | 4 層範圍，自動合併 |
| 指令系統 | 完整實作 |
| 可觀測性框架 | MetricsCollector，Prometheus 就緒 |
| 非同步工具 | 完整實作 |
| 插件架構 | 完整實作 |
| 安全功能 | JWT, Crypto, CORS |
| 多語言支援 | Python/Rust/Go/TypeScript |
| 512 維向量規格 | 與 5M 決策引擎對齊 |

### 8.2 待完成 / 目標功能 🎯

| 功能 | 優先級 | 說明 |
|---|---|---|
| Schema 版本管理 | P1 | 資料合約版本控制，支援向後相容 |
| ConfigManager 熱重載 | P2 | 不重啟服務的設定更新 |
| 分散式追蹤整合 | P2 | OpenTelemetry 追蹤跨模組請求鏈路 |
| Plugin Marketplace | P3 | 第三方插件管理與分發機制 |
| 多語言 Schema 生成 | P3 | 從 Python schema 自動生成 Go/TS 型別 |
| 服務網格支援 | P3 | Istio/Envoy 整合，微服務流量管理 |
| 配置加密 | P3 | 敏感配置（API Keys）的加密儲存 |

---

## 9. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第1冊_系統入門與架構.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第6冊_進階開發.md`（開發者向）
