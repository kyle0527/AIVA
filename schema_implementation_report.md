# AIVA Schema 統一命名與格式規範實施方案

## 執行摘要

本方案針對 AIVA 四大模組的 schemas 進行全面統一，建立一致的命名規範和格式標準，以提升代碼可維護性和跨模組協作效率。

## 完成的工作

### 1. 架構分析完成 ✅

- **aiva_common/schemas.py**: 公共數據合約，使用 `Payload` 後綴
- **core/schemas.py**: 核心分析邏輯，多數無後綴
- **core/ai_ui_schemas.py**: AI/UI 專用，使用 `Request`/`Response`/`Result`
- **scan/schemas.py**: 掃描引擎專用，使用 `Result`/`Match` 後綴
- **function 模組**: XSS/SQLI/SSRF 使用 `Result`/`Vector`/`Telemetry`

### 2. 缺失內容補充完成 ✅

#### 新增 IDOR 模組 Schemas

創建 `services/function/function_idor/aiva_func_idor/schemas.py`:

- `IdorTestVector`: IDOR 測試向量定義
- `IdorDetectionResult`: 檢測結果和風險評分
- `ResourceAccessPattern`: 資源存取模式分析
- `IdorTelemetry`: 專用遙測數據
- `TaskExecutionResult`: 任務執行結果

**特色功能**:

- 自動風險評分機制
- MITRE ATT&CK 技術映射
- 資源類型和 ID 模式分析

#### 新增 PostEx 模組 Schemas

創建 `services/function/function_postex/schemas.py`:

- `PostExTestVector`: 後滲透測試向量
- `PostExDetectionResult`: 檢測結果和影響分析
- `SystemFingerprint`: 系統指紋信息
- `PostExTelemetry`: 專用遙測數據
- `TaskExecutionResult`: 任務執行結果

**特色功能**:

- MITRE ATT&CK 框架整合
- 隱蔽性評分算法
- 系統指紋收集
- 安全模式強制執行

#### 增強 Scan 模組 TypeScript 兼容性

在 `services/scan/aiva_scan/schemas.py` 新增:

- `DynamicScanTask`: 與 TypeScript 介面對應
- `DynamicScanResult`: 統一動態掃描結果格式

## 統一命名規範方案

### 1. 模組前綴系統

```text
[ModulePrefix][FunctionName][DataType]

示例:
- CoreAssetAnalysis (Core 模組的資產分析)
- ScanNetworkRequest (Scan 模組的網路請求)
- FuncXssDetectionResult (Function XSS 模組的檢測結果)
- CommonFindingPayload (Common 模組的發現數據)
```

### 2. 數據類型後綴標準化

- `Payload`: RabbitMQ 消息隊列傳輸數據
- `Request`: HTTP API 請求數據
- `Response`: HTTP API 響應數據
- `Result`: 處理/分析結果
- `Config`: 配置數據
- `Task`: 任務定義
- `Event`: 事件數據
- `Telemetry`: 遙測統計數據
- `Vector`: 測試向量
- `Match`: 匹配結果

### 3. 字段命名標準

- **時間字段**: 統一使用 `*_at` 後綴 (`created_at`, `updated_at`)
- **ID 字段**: 統一使用 `*_id` 後綴 (`task_id`, `scan_id`)
- **布林字段**: 使用 `is_*` 或 `has_*` 前綴
- **計數字段**: 使用 `*_count` 後綴
- **持續時間**: 使用 `*_duration_*` 或 `*_time_*`

## 格式標準統一

### 1. 文件結構模板

```python
"""
[模組名稱] 專用數據合約
定義 [功能描述] 相關的所有數據結構，基於 Pydantic v2.12.0
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator
from services.aiva_common.schemas import ...

class [ModulePrefix][FunctionName][DataType](BaseModel):
    """[功能描述] - 官方 Pydantic BaseModel"""

    # 必填字段
    id: str

    # 可選字段帶默認值
    optional_field: str | None = None

    # 使用 Field 的複雜字段
    complex_field: list[str] = Field(default_factory=list)

    @field_validator("field_name")
    def validate_field(cls, v: type) -> type:
        """驗證 [字段描述]"""
        # 驗證邏輯
        return v

__all__ = [
    "Class1",
    "Class2",
]
```

### 2. 驗證器標準化

- 所有 HTTP 方法驗證使用相同邏輯
- 狀態碼驗證統一範圍 100-599
- URL 驗證使用 Pydantic 內建驗證器
- 自定義驗證器提供清晰的錯誤訊息

### 3. 遙測數據統一格式

所有 Function 模組的 Telemetry 類繼承 `FunctionTelemetry`，提供:

- 統一的基礎指標收集
- 標準化的 `to_details()` 方法
- 一致的性能統計格式

## 剩餘工作項目

### 1. 重命名現有類別 (待執行)

需要重新命名以符合新規範:

```text
# 現有 -> 新命名
SensitiveMatch -> ScanSensitiveMatch
JavaScriptAnalysisResult -> ScanJavaScriptAnalysisResult
NetworkRequest -> ScanNetworkRequest
AssetAnalysis -> CoreAssetAnalysis
VulnerabilityCandidate -> CoreVulnerabilityCandidate
```

### 2. 導入更新 (待執行)

更新所有模組中對重命名類別的導入引用

### 3. TypeScript 介面同步 (待執行)

確保 `aiva_scan_node` 的 TypeScript 介面與 Python schemas 完全對應

### 4. 文檔更新 (待執行)

更新 API 文檔和開發者指南以反映新的命名規範

## 影響評估

### 優勢

- **一致性**: 跨模組命名統一，降低學習成本
- **可維護性**: 清晰的模組歸屬和類型識別
- **擴展性**: 標準化格式便於新模組開發
- **類型安全**: 統一的 Pydantic 驗證機制

### 風險

- **向後兼容**: 重命名會影響現有代碼
- **測試更新**: 需要大量測試用例修改
- **文檔同步**: 多處文檔需要同步更新

### 緩解策略

1. 採用漸進式重構，保留舊名稱作為別名
2. 提供自動化重構工具和腳本
3. 建立全面的測試覆蓋確保功能不受影響
4. 分階段部署，逐步遷移各模組

## 結論

通過系統性的架構分析和缺失內容補充，AIVA 的 schemas 體系現已具備完整性和一致性基礎。下一步需要執行命名統一和格式標準化，預計可大幅提升代碼品質和開發效率。
