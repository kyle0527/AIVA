# AIVA Common 架構說明文檔

> **版本**: 2.0  
> **最後更新**: 2026年1月27日  
> **狀態**: ✅ 生產級 - 單一事實來源

## 📋 目錄

- [概述](#概述)
- [整體架構](#整體架構)
- [核心模組](#核心模組)
- [資料夾結構](#資料夾結構)
- [使用指南](#使用指南)
- [開發規範](#開發規範)
- [版本歷史](#版本歷史)

---

## 概述

`aiva_common` 是 AIVA 系統的**單一事實來源 (Single Source of Truth)**，提供所有服務共享的核心功能、數據結構、配置管理和工具類。

### 設計原則

1. **零容忍** - 無錯誤、無警告
2. **高內聚** - 相關功能組織在一起
3. **低耦合** - 模組間依賴清晰
4. **可測試** - 所有功能可單元測試
5. **向後兼容** - 維持穩定的 API

### 核心價值

- 🎯 **統一標準** - 所有服務使用相同的數據結構和錯誤處理
- 🔒 **類型安全** - 完整的 type hints 支援
- 📊 **可觀測性** - 內建監控和日誌功能
- 🚀 **高效能** - 異步優先，支援高並發
- 🛡️ **安全可靠** - 內建安全機制和錯誤恢復

---

## 整體架構

```
aiva_common/
├── 📁 config/           # 配置管理系統
├── 📁 core/             # 核心功能
├── 📁 security/         # 安全功能
├── 📁 pipeline/         # 數據流處理
├── 📁 services/         # 服務發現與管理
├── 📁 observability/    # 監控與度量
├── 📁 messaging/        # 消息隊列
├── 📁 schemas/          # 數據結構定義
├── 📁 enums/            # 枚舉類型
├── 📁 utils/            # 工具函數
├── 📁 protocols/        # 協議定義
├── 📁 ai/               # AI 相關功能
├── 📁 async_utils/      # 異步工具
├── 📁 cli/              # 命令行工具
├── 📁 cross_language/   # 跨語言支援
├── 📁 detection/        # 檢測功能
├── 📁 plugins/          # 插件系統
└── 📁 tools/            # 開發工具
```

---

## 核心模組

### 1. 📁 config/ - 配置管理系統

**主文件**: `config_manager.py`

**完整文件列表**:
- `config_manager.py` - 核心配置管理器
- `settings.py` - 應用設置
- `defaults.py` - 默認配置
- `paths.py` - 路徑配置
- `unified_config.py` - 統一配置介面

#### 功能特性

- ✅ **分層配置** - Global > User > Session > Temporary
- ✅ **熱更新** - 支援配置動態重載
- ✅ **加密存儲** - 敏感配置自動加密
- ✅ **驗證機制** - 配置架構驗證
- ✅ **環境變量** - 自動載入環境變量
- ✅ **監聽器** - 配置變更通知
- ✅ **路徑管理** - 統一的路徑配置
- ✅ **默認值** - 智能默認配置

#### 關鍵類

```python
from aiva_common.config import ConfigManager, ConfigScope

# 配置管理器
manager = ConfigManager()

# 設置配置
await manager.set("api.timeout", 30, scope=ConfigScope.USER)

# 獲取配置
timeout = manager.get("api.timeout", default=60)
```

#### 配置作用域

| 作用域 | 說明 | 持久化 | 優先級 |
|--------|------|--------|--------|
| `GLOBAL` | 全局配置 | ✅ | 最低 |
| `USER` | 用戶配置 | ✅ | 中 |
| `SESSION` | 會話配置 | ✅ | 高 |
| `TEMPORARY` | 臨時配置 | ❌ | 最高 |

---

### 2. 📁 core/ - 核心功能

#### 2.1 `command_center.py` - AI 命令中心

**用途**: 統一的命令處理與執行框架

**功能**:
- 命令註冊與驗證
- 異步命令執行
- 命令批次處理
- 執行歷史記錄
- 撤銷/重做機制

**示例**:
```python
from aiva_common.core import AICommandCenter, CommandHandler

# 註冊命令處理器
@command_center.register("scan")
class ScanHandler(CommandHandler):
    async def execute(self, context):
        # 執行掃描邏輯
        return {"status": "completed"}
```

#### 2.2 `error_handling.py` - 錯誤處理系統

**用途**: 統一的錯誤處理和恢復機制

**功能**:
- 結構化錯誤類型
- 錯誤嚴重級別
- 錯誤上下文追蹤
- 自動錯誤恢復
- 錯誤報告生成

**錯誤類型**:
```python
from aiva_common.core import AIVAError, ErrorType, ErrorSeverity

# 拋出結構化錯誤
raise AIVAError(
    error_type=ErrorType.VALIDATION_ERROR,
    severity=ErrorSeverity.HIGH,
    message="Invalid input data",
    context={"field": "username"}
)
```

**錯誤級別**:
- `CRITICAL` - 系統級錯誤，需要立即處理
- `HIGH` - 嚴重錯誤，影響核心功能
- `MEDIUM` - 一般錯誤，功能受限
- `LOW` - 輕微錯誤，不影響使用
- `INFO` - 信息性錯誤

---

### 3. 📁 security/ - 安全功能

**文件結構**:
- `security.py` - 核心安全服務
- `security_config.py` - 安全配置
- `security_middleware.py` - 安全中間件

#### 安全功能

**加密服務** (`CryptographyService`):
```python
from aiva_common.security import CryptographyService

crypto = CryptographyService()

# 數據加密
encrypted = await crypto.encrypt_data(sensitive_data)

# 數據解密
decrypted = await crypto.decrypt_data(encrypted)

# 密碼哈希
hashed = await crypto.hash_password(password)
```

**Token 管理** (`TokenService`):
```python
from aiva_common.security import TokenService

token_service = TokenService()

# 生成 Token
token = token_service.generate_token(user_id, expires_in=3600)

# 驗證 Token
payload = token_service.verify_token(token)
```

**CORS 處理** (`CORSHandler`):
```python
from aiva_common.security import CORSHandler

cors = CORSHandler()
cors.add_allowed_origin("https://example.com")
```

---

### 4. 📁 pipeline/ - 數據流處理

#### 4.1 `data_pipeline.py` - 數據管道

**用途**: ETL 數據處理流水線

**組件**:
- **數據源** (DataSource)
  - `QueueDataSource` - 隊列數據源
  - `FileDataSource` - 文件數據源 (JSON/Text)
  - `MemoryDataSource` - 內存數據源

- **數據接收器** (DataSink)
  - `MemoryDataSink` - 內存接收器
  - `FileDataSink` - 文件接收器
  - `CallbackDataSink` - 回調接收器

- **數據處理器** (DataProcessor)
  - `TransformProcessor` - 數據轉換
  - `FilterProcessor` - 數據過濾
  - `AggregateProcessor` - 數據聚合

**示例**:
```python
from aiva_common.pipeline import (
    DataPipeline, 
    FileDataSource,
    TransformProcessor,
    MemoryDataSink
)

# 創建管道
pipeline = DataPipeline(name="etl_pipeline")

# 設置數據源
pipeline.set_source(FileDataSource("data.json"))

# 添加處理器
pipeline.add_processor(TransformProcessor(lambda x: x.upper()))

# 設置接收器
pipeline.set_sink(MemoryDataSink())

# 執行管道
await pipeline.run()
```

#### 4.2 `stream_processor.py` - 流處理器

**用途**: 實時數據流處理

**功能**:
- 滾動窗口 (Tumbling Window)
- 滑動窗口 (Sliding Window)
- 會話窗口 (Session Window)
- 流聚合 (Count, Sum, Avg, Min, Max)
- 水位線 (Watermark) 處理

**示例**:
```python
from aiva_common.pipeline import StreamProcessor

# 創建流處理器
processor = StreamProcessor(
    name="analytics",
    window_size_ms=60000,  # 1分鐘窗口
    window_type="tumbling"
)

# 啟動處理器
await processor.start()

# 發送事件
await processor.emit_event({"user": "alice", "action": "click"})

# 處理結果
async for result in processor.get_results():
    print(f"Window result: {result}")
```

---

### 5. 📁 services/ - 服務發現與管理

**主文件**: `service_discovery.py`

**完整結構**:
```
services/
├── service_discovery.py          # 服務發現核心
└── features/
    └── common/
        └── typescript/
            └── aiva_common_ts/
                └── schemas/
                    └── generated/
                        ├── index.ts
                        └── schemas.ts
```

**TypeScript 集成**: 提供 TypeScript 類型定義，用於跨語言服務通信

#### 服務發現系統

**核心組件**:
- `ServiceRegistry` - 服務註冊表
- `HealthMonitor` - 健康監控器
- `ServiceDiscoveryManager` - 服務發現管理器

**服務狀態**:
- `HEALTHY` - 健康
- `UNHEALTHY` - 不健康
- `STARTING` - 啟動中
- `STOPPING` - 停止中
- `UNKNOWN` - 未知

**使用示例**:
```python
from aiva_common.services import (
    ServiceRegistry,
    ServiceRegistration,
    ServiceEndpoint,
    HealthCheck
)

# 創建註冊表
registry = ServiceRegistry()

# 註冊服務
registration = ServiceRegistration(
    service_id="api-server-1",
    service_name="api-server",
    endpoints=[
        ServiceEndpoint(host="localhost", port=8000)
    ],
    health_check=HealthCheck(
        endpoint="/health",
        interval_seconds=30
    )
)

await registry.register_service(registration)

# 發現服務
services = registry.discover_services(
    service_name="api-server",
    healthy_only=True
)
```

**健康檢查類型**:
- `HTTP` - HTTP 端點檢查
- `TCP` - TCP 連接檢查
- `CUSTOM` - 自定義檢查

---

### 6. 📁 observability/ - 監控與度量

**文件結構**:
- `metrics.py` - 度量收集
- `monitoring.py` - 監控功能
- `monitoring_log_handler.py` - 日誌處理器

#### 度量系統

**度量類型**:
```python
from aiva_common.observability import MetricType

MetricType.COUNTER    # 計數器 (只增不減)
MetricType.GAUGE      # 儀表盤 (可增可減)
MetricType.HISTOGRAM  # 直方圖
MetricType.SUMMARY    # 摘要
```

**使用示例**:
```python
from aiva_common.observability import MetricsCollector, MetricData

collector = MetricsCollector()

# 記錄度量
collector.record(MetricData(
    name="request_count",
    type=MetricType.COUNTER,
    value=1,
    labels={"endpoint": "/api/users"}
))

# 獲取度量
metrics = collector.get_metrics()
```

#### 監控功能

**日誌級別**:
- `DEBUG` - 調試信息
- `INFO` - 一般信息
- `WARNING` - 警告
- `ERROR` - 錯誤
- `CRITICAL` - 嚴重錯誤

---

### 7. 📁 messaging/ - 消息隊列

**主文件**: `mq.py`

#### 📁 messaging/ - 消息隊列

**主文件**: `mq.py`

**完整組件列表**:
- `mq.py` - 核心消息代理抽象
- `compatibility_layer.py` - 兼容性層
- `retry_handler.py` - 重試處理器
- `unified_topic_manager.py` - 統一主題管理器

#### 消息代理抽象

**支援的代理**:
- RabbitMQ (通過 aio_pika)
- 內存隊列 (測試用)

**功能**:
- 消息發布/訂閱
- 主題管理
- 重試機制
- 消息兼容性轉換
- 統一的消息格式

**使用示例**:
```python
from aiva_common.messaging import get_broker, Topic

# 獲取消息代理
broker = await get_broker()

# 發布消息
await broker.publish(
    topic=Topic.SCAN_RESULT,
    message={"target": "example.com", "status": "completed"}
)

# 訂閱消息
async for message in broker.subscribe(Topic.SCAN_RESULT):
    print(f"Received: {message}")
```

**主題 (Topic)**:
```python
from aiva_common.enums import Topic

Topic.SCAN_REQUEST      # 掃描請求
Topic.SCAN_RESULT       # 掃描結果
Topic.VULNERABILITY     # 漏洞發現
Topic.EXPLOIT           # 漏洞利用
Topic.REPORT            # 報告生成
```

---

## 資料夾結構

### 完整結構說明

#### 📁 schemas/ - 數據結構定義

包含所有服務共享的數據模型，採用分層組織：

**核心基礎** (`_base/`):
- `common.py` - 通用基礎類型
- `messaging.py` - 消息基礎結構

**分析相關** (`analysis/`):
- `ai_models.py` - AI 模型相關結構
- `language_support.py` - 語言支援
- `results.py` - 分析結果

**自動生成** (`generated/`):
- `async_utils.py` - 異步工具類型
- `base_types.py` - 基礎類型
- `cli.py` - CLI 相關類型
- `findings.py` - 發現類型
- `messaging.py` - 消息類型
- `plugins.py` - 插件類型
- `tasks.py` - 任務類型

**基礎設施** (`infrastructure/`):
- `assets.py` - 資產管理
- `plugins.py` - 插件結構
- `system.py` - 系統信息
- `telemetry.py` - 遙測數據

**介面定義** (`interfaces/`):
- `api_standards.py` - API 標準
- `async_utils.py` - 異步介面
- `cli.py` - CLI 介面

**風險管理** (`risk/`):
- `assessment.py` - 風險評估
- `attack_paths.py` - 攻擊路徑
- `references.py` - 參考資料

**安全相關** (`security/`):
- `events.py` - 安全事件
- `findings.py` - 安全發現
- `threat_intel.py` - 威脅情報

**測試相關** (`testing/`):
- `scenarios.py` - 測試場景
- `tasks.py` - 測試任務

**根級別檔案**:
- `ai.py`, `analysis.py`, `api_standards.py`, `assets.py`, `async_utils.py`
- `base.py`, `capability.py`, `cli.py`, `commands.py`, `decision.py`
- `dual_loop.py`, `enhanced.py`, `findings.py`, `languages.py`
- `messaging.py`, `plugins.py`, `references.py`, `risk.py`
- `security_events.py`, `system.py`, `tasks.py`, `telemetry.py`
- `threat_intelligence.py`, `vulnerability_finding.py`

**使用**:
```python
from aiva_common.schemas import CVEReference, FunctionTaskPayload
from aiva_common.schemas.security import SecurityEvent
from aiva_common.schemas.risk import RiskAssessment
```

#### 📁 enums/ - 枚舉類型

所有枚舉定義，涵蓋系統各個方面：

**核心枚舉**:
- `modules.py` - 模組類型、階段類型、主題
- `common.py` - 通用枚舉
- `operations.py` - 操作類型

**業務領域**:
- `ai.py` - AI 相關枚舉 (模型類型、提示類型)
- `capabilities.py` - 能力相關枚舉
- `capability_executor.py` - 能力執行器類型
- `data_models.py` - 數據模型枚舉

**安全與測試**:
- `security.py` - 安全級別、威脅類型
- `pentest.py` - 滲透測試相關

**基礎設施**:
- `infrastructure.py` - 基礎設施類型
- `assets.py` - 資產類型

**學術與商業**:
- `academic.py` - 學術相關枚舉
- `business.py` - 業務相關枚舉

**前端相關**:
- `ui_ux.py` - UI/UX 枚舉
- `web_api_standards.py` - Web API 標準

**使用**:
```python
from aiva_common.enums import ModuleType, PhaseType, Topic
from aiva_common.enums.security import ThreatLevel
from aiva_common.enums.ai import AIModelType
```

#### 📁 utils/ - 工具函數

通用工具類，提供各種輔助功能：

**核心工具**:
- `logging.py` - 日誌工具 (get_logger, 格式化)
- `ids.py` - ID 生成工具
- `retry.py` - 重試機制

**網絡工具** (`network/`):
- `backoff.py` - 退避算法
- `ratelimit.py` - 速率限制

**去重工具** (`dedup/`):
- `dedupe.py` - 數據去重

**使用**:
```python
from aiva_common.utils import get_logger
from aiva_common.utils.network import RateLimiter
from aiva_common.utils.retry import retry_async
```

#### 📁 protocols/ - 協議定義

gRPC 協議緩衝區定義和生成文件：

**生成的協議文件**:
- `aiva_enums_pb2.py` / `aiva_enums_pb2_grpc.py` - 枚舉協議
- `aiva_errors_pb2.py` / `aiva_errors_pb2_grpc.py` - 錯誤協議
- `aiva_services_pb2.py` / `aiva_services_pb2_grpc.py` - 服務協議

**工具**:
- `generate_proto.py` - 協議生成工具

**用途**: 
- 服務間通信協議
- 跨語言數據交換
- gRPC 服務定義

#### 📁 ai/ - AI 相關功能

AI 功能模組，統一管理 AI 能力：

**核心文件**:
- `interfaces.py` - AI 介面定義
- `registry.py` - AI 模型註冊表
- `performance_config.py` - 性能配置

**功能**:
- LLM 模型集成與抽象
- 多模型統一介面
- 提示詞管理
- AI 工具調用框架
- 性能優化配置

**使用**:
```python
from aiva_common.ai import AIRegistry, AIInterface
```

#### 📁 async_utils/ - 異步工具

異步編程工具：
- 異步鎖
- 異步隊列
- 任務管理

#### 📁 cli/ - 命令行工具

CLI 工具和介面：
- 參數解析
- 命令執行
- 輸出格式化

#### 📁 cross_language/ - 跨語言支援

多語言互操作框架，支援 Python 與其他語言的無縫集成：

**核心文件**:
- `core.py` - 跨語言核心功能
- `errors.py` - 跨語言錯誤處理

**適配器** (`adapters/`):
- `go_adapter.py` - Go 語言適配器
- `rust_adapter.py` - Rust 語言適配器

**功能**:
- Python ↔ Go 互操作
- Python ↔ Rust 互操作
- 數據序列化/反序列化
- 類型轉換
- 錯誤映射

**使用**:
```python
from aiva_common.cross_language import GoAdapter, RustAdapter
from aiva_common.cross_language.core import CrossLanguageBridge
```

#### 📁 detection/ - 檢測功能

智能檢測管理模組，提供速率控制和超時管理：

**核心文件**:
- `smart_detection_manager.py` - 智能檢測管理器
- `metrics_collector.py` - 度量收集器
- `rate_limiter.py` - 速率限制器
- `timeout_manager.py` - 超時管理器

**功能**:
- 檢測任務調度
- 智能速率控制
- 超時監控與管理
- 檢測度量收集
- 並發控制

**使用**:
```python
from aiva_common.detection import SmartDetectionManager, RateLimiter
```

#### 📁 plugins/ - 插件系統

可擴展插件框架：
- 插件加載
- 插件管理
- 插件生命週期

#### 📁 tools/ - 開發工具

開發輔助工具集，提供代碼生成、驗證和分析：

**核心工具**:
- `schema_codegen_tool.py` - Schema 代碼生成工具
- `schema_validator.py` - Schema 驗證器
- `module_connectivity_checker.py` - 模組連通性檢查器
- `statistics.py` - 統計分析工具

**功能**:
- 自動生成 TypeScript/Python Schema
- Schema 定義驗證
- 模組依賴關係分析
- 代碼統計與報告

**使用**:
```bash
# 生成 Schema
python -m aiva_common.tools.schema_codegen_tool

# 驗證 Schema
python -m aiva_common.tools.schema_validator

# 檢查模組連通性
python -m aiva_common.tools.module_connectivity_checker
```

---

## 使用指南

### 安裝

```bash
cd services/aiva_common
pip install -e .
```

### 基本導入

```python
# 配置管理
from aiva_common.config import ConfigManager, get_config_manager

# 錯誤處理
from aiva_common.core import AIVAError, ErrorHandler

# 服務發現
from aiva_common.services import ServiceRegistry

# 數據管道
from aiva_common.pipeline import DataPipeline

# 安全功能
from aiva_common.security import CryptographyService

# 監控度量
from aiva_common.observability import MetricsCollector

# 消息隊列
from aiva_common.messaging import get_broker
```

### 常見使用模式

#### 1. 配置管理模式

```python
from aiva_common.config import get_config_manager, ConfigScope

# 獲取全局配置管理器
config = get_config_manager()

# 設置配置
await config.set("database.host", "localhost")
await config.set("database.port", 5432)

# 讀取配置
db_host = config.get("database.host")
db_port = config.get("database.port", default=5432)

# 監聽配置變更
def on_config_change(event):
    print(f"Config changed: {event.key}")

config.add_change_listener(on_config_change)
```

#### 2. 錯誤處理模式

```python
from aiva_common.core import (
    AIVAError,
    ErrorHandler,
    ErrorType,
    ErrorSeverity
)

error_handler = ErrorHandler()

try:
    # 業務邏輯
    result = await process_data(data)
except Exception as e:
    # 創建結構化錯誤
    error = AIVAError(
        error_type=ErrorType.PROCESSING_ERROR,
        severity=ErrorSeverity.MEDIUM,
        message=str(e),
        context={"data_id": data.id}
    )
    
    # 處理錯誤 (記錄、恢復、通知)
    error_handler.handle_error(error)
```

#### 3. 數據管道模式

```python
from aiva_common.pipeline import (
    DataPipeline,
    FileDataSource,
    TransformProcessor,
    FilterProcessor,
    MemoryDataSink
)

# 創建 ETL 管道
pipeline = DataPipeline(name="data_processing")

# 配置數據源
pipeline.set_source(FileDataSource("input.json", format_type="json"))

# 添加處理器
pipeline.add_processor(TransformProcessor(lambda x: x["value"] * 2))
pipeline.add_processor(FilterProcessor(lambda x: x > 0))

# 設置接收器
sink = MemoryDataSink()
pipeline.set_sink(sink)

# 執行管道
await pipeline.run()

# 獲取結果
results = sink.get_data()
```

#### 4. 服務註冊模式

```python
from aiva_common.services import (
    ServiceRegistry,
    ServiceRegistration,
    ServiceEndpoint,
    HealthMonitor
)

# 創建註冊表和監控器
registry = ServiceRegistry()
health_monitor = HealthMonitor(registry)

# 註冊服務
registration = ServiceRegistration(
    service_id="my-service-1",
    service_name="my-service",
    endpoints=[ServiceEndpoint(host="localhost", port=8000)]
)

await registry.register_service(registration)

# 啟動健康監控
await health_monitor.start_monitoring(check_interval=30)

# 發現服務
services = registry.discover_services(
    service_name="my-service",
    healthy_only=True
)
```

---

## 開發規範

### 代碼質量標準

1. **類型提示** - 所有函數必須有完整的 type hints
2. **文檔字符串** - 所有公共 API 必須有 docstring
3. **錯誤處理** - 使用 AIVAError 而非原生異常
4. **異步優先** - 所有 I/O 操作使用 async/await
5. **測試覆蓋** - 所有功能必須有單元測試

### 代碼風格

- 遵循 PEP 8
- 使用 Black 進行格式化
- 使用 Ruff 進行 linting
- 最大行長度: 100 字符

### 導入規範

```python
# 標準庫
import asyncio
import logging
from typing import Any

# 第三方庫
import yaml

# 本地導入 - 使用相對導入
from ..config import ConfigManager
from ..core import AIVAError
from .utils import helper_function
```

### 測試規範

```python
import pytest
from aiva_common.config import ConfigManager

@pytest.mark.asyncio
async def test_config_manager():
    """測試配置管理器基本功能"""
    config = ConfigManager()
    
    await config.set("test.key", "value")
    assert config.get("test.key") == "value"
```

### 版本管理

- 遵循語義化版本 (Semantic Versioning)
- 格式: `MAJOR.MINOR.PATCH`
- 當前版本: 2.0.0

---

## 版本歷史

### v2.0.0 (2026-01-27)

**重大重構**:
- ✅ 重組模組結構 (7個邏輯資料夾)
- ✅ 消除所有錯誤和警告 (52+ → 0)
- ✅ 真正的異步 I/O (asyncio.to_thread)
- ✅ 降低認知複雜度
- ✅ 完善錯誤處理機制

**新增功能**:
- ✅ 配置管理系統 (分層、加密、熱更新)
- ✅ 服務發現與健康監控
- ✅ 數據管道和流處理
- ✅ 統一的監控度量系統

**修復問題**:
- ✅ 循環導入問題
- ✅ 路徑引用錯誤
- ✅ 異步操作不完整
- ✅ 代碼重複和複雜度高

### v1.x (歷史版本)

- 基礎功能實現
- 分散的模組結構
- 部分功能缺失

---

## 附錄

### 依賴關係

**核心依賴**:
```
pydantic >= 2.0
pyyaml >= 6.0
cryptography >= 41.0
```

**可選依賴**:
```
aio-pika >= 9.0  # RabbitMQ 支援
aiofiles >= 23.0  # 異步文件 I/O
```

### 性能考慮

- 所有 I/O 操作使用異步
- 配置文件在線程池中讀寫
- 支援連接池和對象池
- 內建緩存機制

### 安全考慮

- 敏感配置自動加密
- Token 使用 JWT 標準
- 支援 HTTPS/TLS
- CORS 保護

### 故障排除

**常見問題**:

1. **導入錯誤**
   ```python
   # 錯誤: from aiva_common.config_manager import ...
   # 正確: from aiva_common.config import ConfigManager
   ```

2. **異步使用**
   ```python
   # 需要在 async 函數中使用
   async def main():
       config = get_config_manager()
       await config.set("key", "value")
   ```

3. **配置文件位置**
   ```
   默認: ~/.aiva/config/
   可通過環境變量修改: AIVA_CONFIG_DIR
   ```

---

## 聯繫方式

- **倉庫**: https://github.com/kyle0527/AIVA
- **問題追蹤**: GitHub Issues
- **文檔**: 本文件

---

**📝 注意**: 本文檔描述 aiva_common v2.0 的架構。關於系統整體架構，請參考項目根目錄的 README.md。
